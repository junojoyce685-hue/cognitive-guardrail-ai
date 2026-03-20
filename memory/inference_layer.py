"""
inference_layer.py
------------------
Local JSON-backed dynamic memory layer for storing and retrieving
compressed session summaries and AI inferences about the user.

Replaces Mem0 with a simple local JSON file per user — free, offline,
no subscription required.

Unlike fact_vault.py (immutable, user-confirmed facts),
this layer stores AI-generated observations that:
    - Get updated every session
    - Can be overwritten if new evidence contradicts them
    - Are audited by the Memory Architect for distortions
    - Only survive long-term if confirmed by user feedback

Exposes:
    - add_session_inference()     → store a new inference after session
    - get_recent_inferences()     → retrieve last N inferences
    - search_inferences()         → keyword search over inferences
    - update_inference_status()   → mark as CONFIRMED / DISCARDED
    - get_confirmed_inferences()  → only return user-confirmed ones
    - delete_inference()          → remove a specific inference
    - clear_user_memory()         → wipe all inferences for a user
"""

import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ── Storage directory ─────────────────────────────────────────────────────────
MEMORY_DIR = ROOT_DIR / "memory" / "store"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_memory_path(user_id: str) -> Path:
    """Returns the JSON file path for a user's inferences."""
    safe_id = user_id.replace("-", "_").replace("/", "_")
    return MEMORY_DIR / f"{safe_id}_inferences.json"


def _load_memory(user_id: str) -> List[Dict]:
    """Loads all inferences for a user from disk."""
    path = _get_memory_path(user_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_memory(user_id: str, memories: List[Dict]) -> None:
    """Saves all inferences for a user to disk."""
    path = _get_memory_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)


# ── Write ─────────────────────────────────────────────────────────────────────

def add_session_inference(
    user_id         : str,
    session_summary : str,
    distortion_label: str  = "",
    confidence      : float = 0.0,
    session_id      : str  = "",
    status          : str  = "PENDING",
) -> str:
    """
    Stores a compressed inference from a session into local JSON.

    Args:
        user_id          : unique identifier for the user
        session_summary  : compressed observation about the user
        distortion_label : distortion type detected this session
        confidence       : analyst confidence score
        session_id       : optional session identifier
        status           : PENDING / CONFIRMED / DISCARDED

    Returns:
        memory_id : unique ID assigned to this inference
    """
    memories  = _load_memory(user_id)
    memory_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()

    entry = {
        "id"              : memory_id,
        "user_id"         : user_id,
        "summary"         : session_summary,
        "distortion_label": distortion_label or "",
        "confidence"      : confidence,
        "session_id"      : session_id or "unknown",
        "status"          : status,
        "timestamp"       : timestamp,
        "type"            : "session_inference",
    }

    memories.append(entry)
    _save_memory(user_id, memories)

    print(f"[inference_layer] Inference stored for user '{user_id}': {session_summary[:60]}...")
    return memory_id


# ── Read ──────────────────────────────────────────────────────────────────────

def get_recent_inferences(
    user_id: str,
    n: int = 10,
) -> List[Dict]:
    """
    Retrieves the most recent N inferences for a user.

    Args:
        user_id : unique identifier for the user
        n       : number of recent inferences to retrieve

    Returns:
        list of inference dicts ordered by recency (newest first)
    """
    memories = _load_memory(user_id)

    # Filter to session_inference type only
    inferences = [
        m for m in memories
        if m.get("type") == "session_inference"
    ]

    # Sort by timestamp descending
    inferences.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return inferences[:n]


def search_inferences(
    user_id  : str,
    query    : str,
    n_results: int = 5,
) -> List[str]:
    """
    Keyword search over stored inferences for a user.

    Searches the summary field for any word in the query.
    Simple but effective for session history lookup — no embeddings needed.

    Args:
        user_id   : unique identifier for the user
        query     : text to search against
        n_results : number of results to return

    Returns:
        list of matching summary strings
    """
    memories = _load_memory(user_id)

    if not memories:
        return []

    # Tokenize query into keywords
    keywords = [w.lower() for w in query.split() if len(w) > 3]

    scored = []
    for m in memories:
        summary = m.get("summary", "").lower()
        score   = sum(1 for kw in keywords if kw in summary)
        if score > 0:
            scored.append((score, m.get("summary", "")))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [s for _, s in scored[:n_results]]


def get_confirmed_inferences(user_id: str) -> List[Dict]:
    """
    Returns only CONFIRMED inferences for a user.

    Args:
        user_id : unique identifier for the user

    Returns:
        list of confirmed inference dicts
    """
    memories = _load_memory(user_id)
    return [m for m in memories if m.get("status") == "CONFIRMED"]


def get_all_inferences(user_id: str) -> List[Dict]:
    """Returns all inferences for a user regardless of status."""
    return _load_memory(user_id)


# ── Update ────────────────────────────────────────────────────────────────────

def update_inference_status(
    memory_id: str,
    status   : str,
    user_id  : str = "",
) -> bool:
    """
    Updates the status of a stored inference.

    Status options:
        PENDING   → not yet reviewed
        CONFIRMED → user verified this is accurate
        DISCARDED → rejected by user or Memory Architect audit

    Args:
        memory_id : inference ID to update
        status    : new status string
        user_id   : user ID (required to find the right file)

    Returns:
        True if updated successfully
    """
    if status not in ("PENDING", "CONFIRMED", "DISCARDED"):
        raise ValueError(f"Invalid status: {status}. Must be PENDING, CONFIRMED, or DISCARDED.")

    if not user_id:
        print(f"[inference_layer] WARNING: user_id required for update — skipping")
        return False

    memories = _load_memory(user_id)
    updated  = False

    for m in memories:
        if m.get("id") == memory_id:
            m["status"] = status
            updated = True
            break

    if updated:
        _save_memory(user_id, memories)
        print(f"[inference_layer] Memory '{memory_id}' status → {status}")
    else:
        print(f"[inference_layer] Memory '{memory_id}' not found for user '{user_id}'")

    return updated


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_inference(memory_id: str, user_id: str) -> bool:
    """
    Deletes a specific inference by ID.

    Args:
        memory_id : inference ID to delete
        user_id   : user ID (required to find the right file)

    Returns:
        True if deleted successfully
    """
    memories    = _load_memory(user_id)
    original    = len(memories)
    memories    = [m for m in memories if m.get("id") != memory_id]

    if len(memories) < original:
        _save_memory(user_id, memories)
        print(f"[inference_layer] Memory '{memory_id}' deleted.")
        return True

    print(f"[inference_layer] Memory '{memory_id}' not found.")
    return False


def clear_user_memory(user_id: str) -> bool:
    """
    Deletes ALL inferences for a user.
    Use only for testing or if user requests full data deletion.

    Args:
        user_id : unique identifier for the user

    Returns:
        True if cleared successfully
    """
    path = _get_memory_path(user_id)
    try:
        if path.exists():
            path.unlink()
        print(f"[inference_layer] All memories cleared for user '{user_id}'")
        return True
    except Exception as e:
        print(f"[inference_layer] Could not clear memories for '{user_id}': {e}")
        return False


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_USER = "test_user_001"

    print("\n── Adding session inferences ──")
    id1 = add_session_inference(
        user_id          = TEST_USER,
        session_summary  = "User tends to use absolute language like 'always' and 'never' when discussing relationships.",
        distortion_label = "Overgeneralization",
        confidence       = 0.91,
        session_id       = "session_001",
        status           = "PENDING",
    )

    id2 = add_session_inference(
        user_id          = TEST_USER,
        session_summary  = "User responded positively to reality-check questions about past successes.",
        distortion_label = "",
        confidence       = 0.0,
        session_id       = "session_001",
        status           = "PENDING",
    )

    id3 = add_session_inference(
        user_id          = TEST_USER,
        session_summary  = "User shows personalization patterns when discussing team failures at work.",
        distortion_label = "Personalization",
        confidence       = 0.85,
        session_id       = "session_002",
        status           = "PENDING",
    )

    print(f"\n── Recent inferences ──")
    recent = get_recent_inferences(TEST_USER, n=5)
    print(f"Count: {len(recent)}")
    for m in recent:
        print(f"  [{m['id']}] {m['summary'][:60]}... | status: {m['status']}")

    print("\n── Searching inferences ──")
    query   = "user language about relationships"
    results = search_inferences(TEST_USER, query, n_results=3)
    print(f"Query: {query}")
    for r in results:
        print(f"  - {r[:80]}...")

    print("\n── Updating status ──")
    update_inference_status(id1, "CONFIRMED", user_id=TEST_USER)
    update_inference_status(id2, "DISCARDED", user_id=TEST_USER)

    print("\n── Confirmed inferences only ──")
    confirmed = get_confirmed_inferences(TEST_USER)
    print(f"Confirmed count: {len(confirmed)}")
    for m in confirmed:
        print(f"  - {m['summary'][:80]}...")

    print("\n── Cleaning up test data ──")
    clear_user_memory(TEST_USER)
    print("Done.")