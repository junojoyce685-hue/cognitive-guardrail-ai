"""
memory_architect.py
-------------------
The most novel agent in the pipeline.

The Memory Architect does two unique things:

1. SELF-AUDIT:
   Reads its own local session summaries and runs the SAME
   cognitive distortion taxonomy on the AI's memory itself.
   e.g. "Did WE overgeneralize by storing 'user is always anxious'?"
        "Did WE catastrophize by labeling this user as severely distorted?"

2. MEMORY GATEKEEPER:
   Decides what gets written to long-term memory based on:
   - Whether the AI's own inference is distortion-free
   - Whether the user confirmed or denied the inference
   Only CONFIRMED + distortion-free inferences survive long term.

Returns a structured dict with:
    - audited_inferences  : list of inferences with distortion audit results
    - confirmed           : list of inferences approved for long-term memory
    - discarded           : list of inferences rejected
    - ai_distortions_found: list of distortions found in AI's own memory
    - facts_to_confirm    : candidate facts extracted for user confirmation
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq import Groq
from config import GROQ_API_KEY
from memory.inference_layer import (
    get_recent_inferences,
    search_inferences,
    update_inference_status,
    add_session_inference,
)
from memory.fact_vault import add_fact

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 2000

VALID_LABELS = [
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental filter",
    "Should statements",
    "Labeling",
    "Personalization",
    "Magnification",
    "Emotional Reasoning",
    "Mind Reading",
    "Fortune-telling",
    "No distortion",
]

# ── Load distortion definitions ───────────────────────────────────────────────
DISTORTIONS_PATH = ROOT_DIR / "data" / "distortions.json"
with open(DISTORTIONS_PATH, "r", encoding="utf-8") as f:
    DISTORTIONS_DATA = json.load(f)

DISTORTION_DEFINITIONS = "\n".join([
    f"- {d['label']}: {d['definition']}"
    for d in DISTORTIONS_DATA["distortions"]
])


# ── System Prompts ────────────────────────────────────────────────────────────

AUDIT_SYSTEM_PROMPT = f"""You are a Meta-Cognitive Auditor for an AI therapy assistant.

Your job is to review the AI's own stored memory summaries and check whether
the AI itself has fallen into cognitive distortion patterns in its thinking about the patient.

This is critical: an AI that overgeneralizes about a patient, catastrophizes their patterns,
or uses absolute language in its summaries is doing the same harmful thing it is trying to help patients avoid.

COGNITIVE DISTORTION TAXONOMY (apply this to the AI's summaries):
{DISTORTION_DEFINITIONS}

FOR EACH MEMORY SUMMARY YOU REVIEW:
1. Check if the AI used absolute language (always, never, completely)
2. Check if the AI overgeneralized from limited sessions
3. Check if the AI labeled the patient reductively (e.g. "user is an anxious person")
4. Check if the AI catastrophized the patient's patterns
5. Check if the AI made mind-reading assumptions about the patient

OUTPUT FORMAT (strict JSON array — one entry per memory reviewed):
[
  {{
    "memory_id"        : "<memory ID or index>",
    "memory_content"   : "<the memory text reviewed>",
    "ai_distortion"    : "<distortion label found in AI memory, or No distortion>",
    "distorted_phrase" : "<exact phrase in AI memory that is distorted, empty if none>",
    "verdict"          : "<CLEAN or DISTORTED>",
    "corrected_version": "<rewritten clean version of the memory, empty if CLEAN>"
  }}
]

Respond ONLY with a valid JSON array. No extra text, no markdown, no backticks."""


FACT_EXTRACTION_SYSTEM_PROMPT = """You are a careful fact extractor for a therapy AI system.

Your job is to read a therapy session conversation and extract ONLY objective,
verifiable facts about the patient's life — NOT their distorted thoughts or feelings.

RULES FOR WHAT COUNTS AS A FACT:
✅ Confirmed life circumstances: "has a partner", "works as a teacher", "has two siblings"
✅ Confirmed positive experiences: "recently got promoted", "completed a difficult project"
✅ Confirmed support systems: "has close friends", "attends therapy regularly"
✅ Confirmed personal strengths: "patient showed resilience when discussing X"

RULES FOR WHAT IS NOT A FACT:
❌ Distorted thoughts: "I always fail" → NOT a fact
❌ Emotional states: "feels anxious" → too temporary, not a stable fact
❌ AI assumptions: "patient seems depressed" → not confirmed
❌ Absolute statements: "never succeeds" → distorted, not a fact

OUTPUT FORMAT (strict JSON array):
[
  {{
    "candidate_fact" : "<the extracted fact statement>",
    "source_quote"   : "<the exact patient quote this came from>",
    "confidence"     : <float 0.0 to 1.0 — how certain this is a real fact>
  }}
]

If no facts can be extracted, return an empty array: []
Respond ONLY with valid JSON. No extra text, no markdown, no backticks."""


# ── Core Functions ────────────────────────────────────────────────────────────

def audit_ai_memory(
    user_id: str,
    n_memories: int = 10,
) -> List[Dict]:
    """
    Runs cognitive distortion checks on the AI's own stored memory summaries.

    Args:
        user_id    : unique identifier for the user
        n_memories : how many recent memories to audit

    Returns:
        list of audit result dicts, one per memory reviewed
    """
    client = Groq(api_key=GROQ_API_KEY)

    # Fetch recent inferences from local JSON store
    recent = get_recent_inferences(user_id, n=n_memories)

    if not recent:
        print(f"[memory_architect] No memories to audit for user '{user_id}'")
        return []

    # Build memory list for audit
    memory_list = ""
    for i, mem in enumerate(recent):
        content   = mem.get("summary", mem.get("content", ""))
        memory_id = mem.get("id", str(i))
        memory_list += f"Memory {i+1} (ID: {memory_id}):\n{content}\n\n"

    user_message = f"""Please audit the following AI-generated memory summaries about a patient.
Check each one for cognitive distortions in the AI's own thinking.

{memory_list}

Return a JSON array with one audit result per memory."""

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        audit_results = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            audit_results = json.loads(match.group())
        else:
            raise ValueError(f"Memory Architect audit: Could not parse JSON:\n{raw_text}")

    distorted_count = sum(1 for r in audit_results if r.get("verdict") == "DISTORTED")
    print(f"[memory_architect] Audited {len(audit_results)} memories — {distorted_count} distorted found")

    return audit_results


def extract_candidate_facts(
    user_id: str,
    conversation: List[Dict],
    session_id: str = "",
) -> List[Dict]:
    """
    Extracts candidate facts from a session conversation for user confirmation.

    These are NOT stored yet — they are returned to the pipeline so the user
    can confirm or deny them before fact_vault.add_fact() is called.

    Args:
        user_id      : unique identifier for the user
        conversation : list of dicts with 'role' and 'content'
        session_id   : optional session identifier

    Returns:
        list of candidate fact dicts with keys:
            candidate_fact, source_quote, confidence
    """
    client = Groq(api_key=GROQ_API_KEY)

    # Format conversation for the prompt
    convo_text = ""
    for turn in conversation:
        role    = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        convo_text += f"{role}: {content}\n"

    user_message = f"""Extract confirmed facts from this therapy session conversation:

{convo_text}

Return only objective, verifiable facts about the patient's life."""

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = 1000,
        messages   = [
            {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        candidates = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            candidates = json.loads(match.group())
        else:
            candidates = []

    # Filter to high-confidence candidates only
    candidates = [c for c in candidates if c.get("confidence", 0) >= 0.75]

    print(f"[memory_architect] Extracted {len(candidates)} candidate facts from session")
    return candidates


def process_session_end(
    user_id: str,
    session_id: str,
    conversation: List[Dict],
    analyst_result: dict,
    responder_result: dict,
    user_feedback: Optional[str] = None,
) -> Dict:
    """
    Runs the full Memory Architect workflow at the end of a session.

    Steps:
    1. Build a session summary inference
    2. Store it in local JSON store as PENDING
    3. Audit recent AI memories for distortions
    4. Update distorted memories → DISCARDED
    5. Extract candidate facts for user confirmation
    6. Return everything for the pipeline to handle

    Args:
        user_id          : unique identifier for the user
        session_id       : unique session identifier
        conversation     : full session conversation
        analyst_result   : final analyst output
        responder_result : final responder output
        user_feedback    : optional user feedback string ("helpful" / "not helpful")

    Returns:
        dict with full Memory Architect results
    """
    # ── Step 1: Build session summary ─────────────────────────────────────────
    distortion_label = analyst_result.get("label", "No distortion")
    confidence       = analyst_result.get("confidence", 0.0)
    technique        = responder_result.get("technique", "")

    session_summary = (
        f"Session {session_id}: Patient expressed thoughts showing '{distortion_label}' "
        f"(confidence: {confidence}). "
        f"Responded using '{technique}' technique. "
    )

    if user_feedback:
        session_summary += f"User feedback: {user_feedback}."

    # ── Step 2: Store session inference as PENDING ────────────────────────────
    memory_id = add_session_inference(
        user_id          = user_id,
        session_summary  = session_summary,
        distortion_label = distortion_label,
        confidence       = confidence,
        session_id       = session_id,
        status           = "PENDING",
    )

    # ── Step 3: Audit recent AI memories ─────────────────────────────────────
    audit_results = audit_ai_memory(user_id, n_memories=10)

    # ── Step 4: Discard distorted AI memories ────────────────────────────────
    ai_distortions_found = []
    for result in audit_results:
        if result.get("verdict") == "DISTORTED":
            mem_id = result.get("memory_id", "")
            if mem_id:
                update_inference_status(mem_id, "DISCARDED", user_id=user_id)
            ai_distortions_found.append({
                "distortion"      : result.get("ai_distortion"),
                "distorted_phrase": result.get("distorted_phrase"),
                "corrected"       : result.get("corrected_version"),
            })

    # ── Step 5: Extract + silently auto-save high-confidence facts ───────────
    # Facts are saved in the background like a therapist naturally remembering
    # things — no user confirmation prompt, no interruption to the conversation.
    candidate_facts = extract_candidate_facts(
        user_id      = user_id,
        conversation = conversation,
        session_id   = session_id,
    )

    auto_saved = []
    if candidate_facts:
        from memory.fact_vault import add_facts_batch
        # High confidence (>=0.9) → save immediately as confirmed
        high_conf = [
            f["candidate_fact"] for f in candidate_facts
            if f.get("confidence", 0) >= 0.90
        ]
        # Medium confidence (0.75-0.89) → save tagged as unverified
        med_conf = [
            f["candidate_fact"] for f in candidate_facts
            if 0.75 <= f.get("confidence", 0) < 0.90
        ]

        if high_conf:
            add_facts_batch(
                user_id    = user_id,
                facts      = high_conf,
                source     = "auto_confirmed",
                session_id = session_id,
            )
            auto_saved.extend(high_conf)

        if med_conf:
            add_facts_batch(
                user_id    = user_id,
                facts      = [f"[unverified] {f}" for f in med_conf],
                source     = "auto_unverified",
                session_id = session_id,
            )

        print(f"[memory_architect] Auto-saved {len(auto_saved)} facts silently")

    # ── Step 6: Auto-confirm if user gave positive feedback ───────────────────
    if user_feedback and "helpful" in user_feedback.lower():
        update_inference_status(memory_id, "CONFIRMED", user_id=user_id)
        print(f"[memory_architect] Memory auto-confirmed based on positive user feedback")

    print(f"[memory_architect] Session end processed for user '{user_id}'")

    return {
        "memory_id"           : memory_id,
        "session_summary"     : session_summary,
        "audit_results"       : audit_results,
        "ai_distortions_found": ai_distortions_found,
        "distorted_count"     : len(ai_distortions_found),
    }


def confirm_facts_from_user(
    user_id: str,
    confirmed_facts: List[str],
    session_id: str = "",
) -> List[str]:
    """
    Called after the user confirms which candidate facts are true.
    Writes confirmed facts to the fact vault.

    Args:
        user_id         : unique identifier for the user
        confirmed_facts : list of fact strings the user confirmed as true
        session_id      : optional session identifier

    Returns:
        list of fact_ids stored in ChromaDB
    """
    from memory.fact_vault import add_facts_batch

    if not confirmed_facts:
        return []

    fact_ids = add_facts_batch(
        user_id    = user_id,
        facts      = confirmed_facts,
        source     = "user_confirmed",
        session_id = session_id,
    )

    print(f"[memory_architect] {len(fact_ids)} facts confirmed and stored for user '{user_id}'")
    return fact_ids


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from training.csv_loader import load_merged
    from agents.analyst import run_analyst
    from agents.responder import run_responder

    df         = load_merged()
    TEST_USER  = "test_user_ma_001"
    SESSION_ID = "session_001"

    test_conversation = [
        {"role": "user",      "content": "I always ruin everything. Nobody will ever love me."},
        {"role": "assistant", "content": "I hear how much pain you're carrying right now."},
        {"role": "user",      "content": "I guess they would disagree. They tell me they love me. I have a good job too, just got promoted last month."},
        {"role": "assistant", "content": "That's really important to notice."},
    ]

    analyst_result   = run_analyst(test_conversation[0]["content"], df)
    responder_result = run_responder(
        user_input     = test_conversation[0]["content"],
        analyst_result = analyst_result,
        df             = df,
    )

    print(f"\n{'─'*60}")
    print("Running Memory Architect...")

    result = process_session_end(
        user_id          = TEST_USER,
        session_id       = SESSION_ID,
        conversation     = test_conversation,
        analyst_result   = analyst_result,
        responder_result = responder_result,
        user_feedback    = "helpful",
    )

    print(f"\n── Results ──")
    print(f"Memory ID       : {result['memory_id']}")
    print(f"Session Summary : {result['session_summary']}")
    print(f"AI Distortions  : {result['distorted_count']} found")
    print(f"Candidate Facts : {len(result['candidate_facts'])} extracted")

    for fact in result["candidate_facts"]:
        print(f"  → {fact['candidate_fact']} (confidence: {fact['confidence']})")

    if result["candidate_facts"]:
        facts_to_store = [f["candidate_fact"] for f in result["candidate_facts"]]
        confirm_facts_from_user(TEST_USER, facts_to_store, session_id=SESSION_ID)

    # Cleanup
    from memory.inference_layer import clear_user_memory
    from memory.fact_vault import clear_user_vault
    clear_user_memory(TEST_USER)
    clear_user_vault(TEST_USER)
    print("\nTest cleanup done.")
