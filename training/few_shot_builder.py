"""
few_shot_builder.py
-------------------
Retrieves the most relevant examples from the merged CSV dataset
using semantic similarity (sentence-transformers + cosine similarity).

Exposes two main functions:
    - build_analyst_examples()   → for distortion classification
    - build_responder_examples() → for therapist-style response generation

Uses label-balanced fallback to ensure rare distortion types
are not missed during retrieval.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

# ── Model (lazy loaded on first use) ─────────────────────────────────────────
_EMBEDDER = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        print("[few_shot_builder] Loading sentence-transformer model...")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        print("[few_shot_builder] Model ready.")
    return _EMBEDDER

# ── Cache for dataset embeddings (avoid re-embedding on every call) ───────────
_embedding_cache: Optional[np.ndarray] = None
_cached_df_id: Optional[int] = None  # tracks which df is cached


# ── Core: Embed + Retrieve ────────────────────────────────────────────────────

def _get_dataset_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Embed all Patient Questions in the dataset.
    Caches the result so we only compute once per session.
    """
    global _embedding_cache, _cached_df_id

    df_id = id(df)
    if _embedding_cache is not None and _cached_df_id == df_id:
        return _embedding_cache

    print(f"[few_shot_builder] Embedding {len(df)} patient questions (one-time setup)...")
    questions = df["Patient Question"].tolist()
    _embedding_cache = _get_embedder().encode(questions, show_progress_bar=True, batch_size=64)
    _cached_df_id = df_id
    print("[few_shot_builder] Embedding complete and cached.")
    return _embedding_cache


def _get_top_k_indices(
    input_text: str,
    df: pd.DataFrame,
    k: int = 5
) -> List[int]:
    """
    Embed the input text and return indices of the top-k
    most semantically similar rows in the dataset.
    """
    dataset_embeddings = _get_dataset_embeddings(df)
    input_embedding = _get_embedder().encode([input_text])
    similarities       = cosine_similarity(input_embedding, dataset_embeddings)[0]
    top_k_indices      = np.argsort(similarities)[::-1][:k]
    return top_k_indices.tolist()


# ── Label-Balanced Retrieval ──────────────────────────────────────────────────

def _get_balanced_indices(
    input_text: str,
    df: pd.DataFrame,
    k: int = 5,
    min_per_label: int = 1
) -> List[int]:
    """
    Retrieves top-k examples with a label-balanced fallback.

    Strategy:
    1. Get top 30 candidates by similarity
    2. From those, take the top-k by similarity
    3. Check if any distortion label is completely absent from top-k
    4. For missing labels that appear in top-30, swap in their best match

    This ensures rare distortion types (e.g. Fortune-telling) are
    represented when they appear in the candidate pool.
    """
    dataset_embeddings = _get_dataset_embeddings(df)
    input_embedding = _get_embedder().encode([input_text])
    similarities       = cosine_similarity(input_embedding, dataset_embeddings)[0]

    # Step 1: Get top 30 candidates
    candidate_pool_size = min(30, len(df))
    top_30_indices      = np.argsort(similarities)[::-1][:candidate_pool_size].tolist()

    # Step 2: Start with top-k from pool
    selected_indices = top_30_indices[:k]
    selected_labels  = set(df.iloc[selected_indices]["Dominant Distortion"].tolist())

    # Step 3: Check which labels appear in candidate pool but not in selected
    pool_labels    = df.iloc[top_30_indices]["Dominant Distortion"].unique()
    missing_labels = [l for l in pool_labels if l not in selected_labels]

    # Step 4: Swap in best representative for each missing label
    for label in missing_labels:
        if len(selected_indices) >= k + len(missing_labels):
            break

        label_indices = [
            i for i in top_30_indices
            if df.iloc[i]["Dominant Distortion"] == label
        ]
        if label_indices:
            best_for_label = label_indices[0]
            if best_for_label not in selected_indices:
                selected_indices.append(best_for_label)

    return selected_indices


# ── Public API ────────────────────────────────────────────────────────────────

def build_analyst_examples(
    input_text: str,
    df: pd.DataFrame,
    k: int = 5,
    balanced: bool = True
) -> List[Dict]:
    """
    Returns top-k few-shot examples for the Analyst agent.

    Each example contains:
        - question        : the patient's original text
        - distorted_part  : the annotated distorted snippet
        - label           : dominant distortion label
        - secondary_label : secondary distortion (if any)

    Args:
        input_text : the new user input to find matches for
        df         : merged DataFrame from csv_loader.load_merged()
        k          : number of examples to retrieve
        balanced   : if True, uses label-balanced retrieval

    Returns:
        List of dicts, one per example
    """
    if balanced:
        indices = _get_balanced_indices(input_text, df, k=k)
    else:
        indices = _get_top_k_indices(input_text, df, k=k)

    examples = []
    for idx in indices:
        row = df.iloc[idx]
        examples.append({
            "question"        : row["Patient Question"],
            "distorted_part"  : row["Distorted part"],
            "label"           : row["Dominant Distortion"],
            "secondary_label" : row.get("Secondary distortion (Optional)", ""),
        })

    return examples


def build_responder_examples(
    input_text: str,
    df: pd.DataFrame,
    k: int = 3,
    balanced: bool = True
) -> List[Dict]:
    """
    Returns top-k few-shot examples for the Responder agent.

    Each example contains:
        - question         : the patient's original text
        - label            : dominant distortion label
        - therapist_answer : the ideal therapist response

    Args:
        input_text : the new user input to find matches for
        df         : merged DataFrame from csv_loader.load_merged()
        k          : number of examples to retrieve (default 3)
        balanced   : if True, uses label-balanced retrieval

    Returns:
        List of dicts, one per example
    """
    if balanced:
        indices = _get_balanced_indices(input_text, df, k=k)
    else:
        indices = _get_top_k_indices(input_text, df, k=k)

    examples = []
    for idx in indices:
        row = df.iloc[idx]
        examples.append({
            "question"         : row["Patient Question"],
            "label"            : row["Dominant Distortion"],
            "therapist_answer" : row["Therapist Answer"],
        })

    return examples


# ── Prompt Formatters ─────────────────────────────────────────────────────────

def format_analyst_prompt(examples: List[Dict]) -> str:
    """
    Converts analyst examples into a formatted few-shot prompt block.
    Ready to be injected into the Analyst's system/user prompt.
    """
    lines = ["Here are some annotated examples to guide your classification:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Patient said  : {ex['question']}")
        if ex["distorted_part"]:
            lines.append(f"  Distorted part: {ex['distorted_part']}")
        lines.append(f"  Label         : {ex['label']}")
        if ex["secondary_label"]:
            lines.append(f"  Secondary     : {ex['secondary_label']}")
        lines.append("")
    return "\n".join(lines)


def format_responder_prompt(examples: List[Dict]) -> str:
    """
    Converts responder examples into a formatted few-shot prompt block.
    Ready to be injected into the Responder's system/user prompt.
    """
    lines = ["Here are examples of how a therapist responded to similar situations:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Patient said      : {ex['question']}")
        lines.append(f"  Distortion type   : {ex['label']}")
        lines.append(f"  Therapist response: {ex['therapist_answer']}")
        lines.append("")
    return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from training.csv_loader import load_merged

    df = load_merged()

    test_input = "I always ruin everything I touch, nobody will ever love me."
    print(f"\nTest input: {test_input}\n")

    # Analyst examples
    analyst_examples = build_analyst_examples(test_input, df, k=5, balanced=True)
    print("── Analyst Few-Shot Prompt ──")
    print(format_analyst_prompt(analyst_examples))

    # Responder examples
    responder_examples = build_responder_examples(test_input, df, k=3, balanced=True)
    print("── Responder Few-Shot Prompt ──")
    print(format_responder_prompt(responder_examples))
