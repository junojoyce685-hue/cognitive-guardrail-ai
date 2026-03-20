"""
pipeline.py
-----------
LangGraph state graph that orchestrates the full pipeline.

Flow:
    User Input
        ↓
    [ANALYST NODE]          - classify distortion
        ↓
    [DEVIL'S ADVOCATE NODE] - challenge or agree (max 2 loops)
        ↓
    [RESPONDER NODE]        - generate therapist-style response
        ↓
    [MEMORY ARCHITECT NODE] - audit memory, extract facts
        ↓
    Output to user

State is passed between nodes as a typed dict.
Each node reads from state and writes back to state.
"""

import sys
import uuid
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Annotated
import operator

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from langgraph.graph import StateGraph, END

from agents.analyst import run_analyst
from agents.devils_advocate import run_consensus_loop
from agents.responder import run_responder
from agents.memory_architect import process_session_end
from memory.fact_vault import search_facts
from memory.inference_layer import search_inferences
from training.csv_loader import load_merged

# ── Load dataset once at startup ──────────────────────────────────────────────
print("[pipeline] Loading dataset...")
DF = load_merged()
print("[pipeline] Dataset ready.")


# ── Pipeline State ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """
    Shared state passed between all nodes in the graph.
    Each node reads what it needs and writes its results back.
    """
    # Input
    user_id          : str
    session_id       : str
    user_input       : str

    # Conversation history (for memory architect at session end)
    conversation     : List[Dict]

    # Memory context (fetched before analyst runs)
    memory_facts     : List[str]
    recent_inferences: List[str]

    # Analyst output
    analyst_result   : Dict

    # Devil's advocate output
    devils_result    : Dict
    consensus_loops  : int

    # Responder output
    responder_result : Dict

    # Memory architect output
    architect_result : Dict

    # Final output to user
    final_response   : str
    final_follow_up  : str
    flagged          : bool
    flag_reason      : str

    # User feedback (set after response is shown)
    user_feedback    : Optional[str]


# ── Node Functions ────────────────────────────────────────────────────────────

def fetch_memory_node(state: PipelineState) -> PipelineState:
    """
    Fetches relevant facts and inferences from memory
    before the analyst runs.
    """
    user_id    = state["user_id"]
    user_input = state["user_input"]

    print(f"\n[pipeline] ── FETCH MEMORY ──")

    # Search ChromaDB fact vault
    memory_facts = search_facts(user_id, user_input, n_results=5)
    print(f"[pipeline] Facts found: {len(memory_facts)}")

    # Search Mem0 inferences
    recent_inferences = search_inferences(user_id, user_input, n_results=3)
    print(f"[pipeline] Inferences found: {len(recent_inferences)}")

    return {
        **state,
        "memory_facts"     : memory_facts,
        "recent_inferences": recent_inferences,
    }


def analyst_node(state: PipelineState) -> PipelineState:
    """
    Classifies the user input for cognitive distortions.
    """
    print(f"\n[pipeline] ── ANALYST ──")

    analyst_result = run_analyst(
        user_input = state["user_input"],
        df         = DF,
        k          = 5,
    )

    # Add turn to conversation history
    conversation = state.get("conversation", [])
    conversation.append({
        "role"   : "user",
        "content": state["user_input"],
    })

    return {
        **state,
        "analyst_result": analyst_result,
        "conversation"  : conversation,
    }


def devils_advocate_node(state: PipelineState) -> PipelineState:
    """
    Challenges or validates the analyst's classification.
    Runs the consensus loop (max 2 iterations).
    """
    print(f"\n[pipeline] ── DEVIL'S ADVOCATE ──")

    final_analyst, final_devils, loops = run_consensus_loop(
        user_input     = state["user_input"],
        analyst_result = state["analyst_result"],
        df             = DF,
        memory_facts   = state["memory_facts"],
        max_loops      = 2,
    )

    return {
        **state,
        "analyst_result" : final_analyst,
        "devils_result"  : final_devils,
        "consensus_loops": loops,
    }


def responder_node(state: PipelineState) -> PipelineState:
    """
    Generates the therapist-style response to the user.
    """
    print(f"\n[pipeline] ── RESPONDER ──")

    responder_result = run_responder(
        user_input        = state["user_input"],
        analyst_result    = state["analyst_result"],
        df                = DF,
        recent_inferences = state["recent_inferences"],
        k                 = 3,
    )

    # Add assistant response to conversation history
    conversation = state.get("conversation", [])
    conversation.append({
        "role"   : "assistant",
        "content": responder_result["response"],
    })

    return {
        **state,
        "responder_result": responder_result,
        "conversation"    : conversation,
        "final_response"  : responder_result["response"],
        "final_follow_up" : responder_result["follow_up"],
        "flagged"         : responder_result["flagged"],
        "flag_reason"     : responder_result["flag_reason"],
    }


def memory_architect_node(state: PipelineState) -> PipelineState:
    """
    Runs the Memory Architect at session end:
    - Audits AI memory for distortions
    - Extracts candidate facts for user confirmation
    - Stores session inference
    """
    print(f"\n[pipeline] ── MEMORY ARCHITECT ──")

    architect_result = process_session_end(
        user_id          = state["user_id"],
        session_id       = state["session_id"],
        conversation     = state["conversation"],
        analyst_result   = state["analyst_result"],
        responder_result = state["responder_result"],
        user_feedback    = state.get("user_feedback"),
    )

    return {
        **state,
        "architect_result": architect_result,
    }


# ── Conditional Edge ──────────────────────────────────────────────────────────

def should_flag(state: PipelineState) -> str:
    """
    After responder runs — if flagged, skip memory and go straight to END.
    Crisis situations should not be stored in memory.
    """
    if state.get("flagged"):
        print(f"[pipeline] Crisis flag detected — skipping memory storage")
        return "flagged"
    return "continue"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Builds and compiles the LangGraph state graph.

    Graph structure:
        fetch_memory → analyst → devils_advocate → responder
            → [conditional: flagged → END | continue → memory_architect → END]
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("fetch_memory",      fetch_memory_node)
    graph.add_node("analyst",           analyst_node)
    graph.add_node("devils_advocate",   devils_advocate_node)
    graph.add_node("responder",         responder_node)
    graph.add_node("memory_architect",  memory_architect_node)

    # Add edges
    graph.set_entry_point("fetch_memory")
    graph.add_edge("fetch_memory",    "analyst")
    graph.add_edge("analyst",         "devils_advocate")
    graph.add_edge("devils_advocate", "responder")

    # Conditional edge after responder
    graph.add_conditional_edges(
        "responder",
        should_flag,
        {
            "flagged" : END,
            "continue": "memory_architect",
        }
    )

    graph.add_edge("memory_architect", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

# Compiled pipeline (singleton)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_pipeline(
    user_input  : str,
    user_id     : str,
    session_id  : str = None,
    user_feedback: str = None,
) -> Dict:
    """
    Runs the full pipeline for a single user turn.

    Args:
        user_input    : raw text from the patient
        user_id       : unique user identifier
        session_id    : optional session ID (auto-generated if not provided)
        user_feedback : optional feedback from previous turn ("helpful" / "not helpful")

    Returns:
        dict with:
            - response        : therapist-style reply
            - follow_up       : follow-up question
            - label           : detected distortion label
            - confidence      : analyst confidence
            - flagged         : True if crisis detected
            - flag_reason     : reason for flag
            - consensus_loops : how many debate loops were needed
    """
    pipeline   = get_pipeline()
    session_id = session_id or str(uuid.uuid4())[:8]

    # Build initial state
    initial_state: PipelineState = {
        "user_id"          : user_id,
        "session_id"       : session_id,
        "user_input"       : user_input,
        "conversation"     : [],
        "memory_facts"     : [],
        "recent_inferences": [],
        "analyst_result"   : {},
        "devils_result"    : {},
        "consensus_loops"  : 0,
        "responder_result" : {},
        "architect_result" : {},
        "final_response"   : "",
        "final_follow_up"  : "",
        "flagged"          : False,
        "flag_reason"      : "",
        "user_feedback"    : user_feedback,
    }

    # Run graph
    final_state = pipeline.invoke(initial_state)

    # Package clean output
    return {
        "response"        : final_state["final_response"],
        "follow_up"       : final_state["final_follow_up"],
        "label"           : final_state["analyst_result"].get("label", ""),
        "confidence"      : final_state["analyst_result"].get("confidence", 0.0),
        "distorted_part"  : final_state["analyst_result"].get("distorted_part", ""),
        "explanation"     : final_state["analyst_result"].get("explanation", ""),
        "technique"       : final_state["responder_result"].get("technique", ""),
        "flagged"         : final_state["flagged"],
        "flag_reason"     : final_state["flag_reason"],
        "consensus_loops" : final_state["consensus_loops"],
        "session_id"      : session_id,
    }



# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_USER = "test_user_pipeline_001"

    test_inputs = [
        "I always ruin everything I touch. Nobody will ever truly love me.",
        "My boss ignored my email today. He definitely hates me and wants me gone.",
        "I had a hard week but I got through it okay.",
    ]

    for text in test_inputs:
        print(f"\n{'═'*60}")
        print(f"USER: {text}")

        result = run_pipeline(
            user_input = text,
            user_id    = TEST_USER,
        )

        print(f"\n── Pipeline Output ──")
        print(f"Label      : {result['label']} ({result['confidence']})")
        print(f"Technique  : {result['technique']}")
        print(f"Response   : {result['response']}")
        print(f"Follow-up  : {result['follow_up']}")
        print(f"Flagged    : {result['flagged']}")
        print(f"DA Loops   : {result['consensus_loops']}")

    # Cleanup
    from memory.inference_layer import clear_user_memory
    from memory.fact_vault import clear_user_vault
    clear_user_memory(TEST_USER)
    clear_user_vault(TEST_USER)
    print("\n── Test cleanup done ──")
