"""
fact_vault.py
-------------
Immutable ChromaDB-backed store for confirmed user facts.

Facts are things the user has explicitly confirmed as true about themselves:
    - "I have a supportive partner"
    - "I recently got a promotion"
    - "I have close friends I can rely on"

These facts are:
    - Written ONCE and never overwritten (immutable)
    - Retrieved by the Devil's Advocate to challenge distortion labels
    - Retrieved by the Memory Architect to audit AI memory

Collections:
    - user_facts : confirmed facts per user_id
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH

# ── ChromaDB client (persistent, stored on disk) ──────────────────────────────
_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    """
    Returns a singleton ChromaDB persistent client.
    Creates the DB directory if it doesn't exist.
    """
    global _client
    if _client is None:
        db_path = Path(CHROMA_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(db_path))
        print(f"[fact_vault] ChromaDB connected at: {db_path}")
    return _client


def _get_collection(user_id: str) -> chromadb.Collection:
    """
    Returns (or creates) a ChromaDB collection for a specific user.
    Each user gets their own isolated collection.
    """
    client          = _get_client()
    collection_name = f"facts_{user_id.replace('-', '_')}"

    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"user_id": user_id, "type": "fact_vault"}
    )
    return collection


# ── Write ─────────────────────────────────────────────────────────────────────

def add_fact(
    user_id: str,
    fact: str,
    source: str = "user_confirmed",
    session_id: str = None,
) -> str:
    """
    Adds a confirmed fact to the user's fact vault.

    Facts are immutable — this function only adds, never overwrites.

    Args:
        user_id    : unique identifier for the user
        fact       : the confirmed fact string to store
        source     : where this fact came from (default: user_confirmed)
        session_id : optional session identifier for traceability

    Returns:
        fact_id : the unique ID assigned to this fact
    """
    collection = _get_collection(user_id)

    # Check for near-duplicate before adding
    existing = search_facts(user_id, fact, n_results=1)
    if existing:
        # Simple duplicate check — if first result is very similar, skip
        print(f"[fact_vault] Similar fact already exists — skipping duplicate.")
        return "duplicate"

    fact_id   = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    collection.add(
        ids        = [fact_id],
        documents  = [fact],
        metadatas  = [{
            "user_id"    : user_id,
            "source"     : source,
            "session_id" : session_id or "unknown",
            "timestamp"  : timestamp,
        }]
    )

    print(f"[fact_vault] Fact added for user '{user_id}': {fact[:60]}...")
    return fact_id


def add_facts_batch(
    user_id: str,
    facts: List[str],
    source: str = "user_confirmed",
    session_id: str = None,
) -> List[str]:
    """
    Adds multiple confirmed facts at once.

    Args:
        user_id    : unique identifier for the user
        facts      : list of fact strings
        source     : where facts came from
        session_id : optional session identifier

    Returns:
        list of fact_ids
    """
    fact_ids = []
    for fact in facts:
        fact_id = add_fact(user_id, fact, source=source, session_id=session_id)
        fact_ids.append(fact_id)
    return fact_ids


# ── Read ──────────────────────────────────────────────────────────────────────

def search_facts(
    user_id: str,
    query: str,
    n_results: int = 5,
) -> List[str]:
    """
    Retrieves the most semantically relevant facts for a given query.

    Used by:
        - Devil's Advocate: to find facts that challenge a distortion label
        - Memory Architect: to cross-check AI memory against confirmed facts

    Args:
        user_id   : unique identifier for the user
        query     : text to search against (usually the patient's statement)
        n_results : number of facts to return

    Returns:
        list of fact strings, ordered by relevance
    """
    collection = _get_collection(user_id)

    # Check if collection has any documents
    count = collection.count()
    if count == 0:
        return []

    # Clamp n_results to available count
    n_results = min(n_results, count)

    results = collection.query(
        query_texts = [query],
        n_results   = n_results,
    )

    facts = results["documents"][0] if results["documents"] else []
    return facts


def get_all_facts(user_id: str) -> List[dict]:
    """
    Returns all stored facts for a user with full metadata.

    Args:
        user_id : unique identifier for the user

    Returns:
        list of dicts with keys: fact_id, fact, source, session_id, timestamp
    """
    collection = _get_collection(user_id)

    count = collection.count()
    if count == 0:
        return []

    results = collection.get(include=["documents", "metadatas"])

    facts = []
    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        facts.append({
            "fact_id"    : results["ids"][i],
            "fact"       : doc,
            "source"     : meta.get("source", ""),
            "session_id" : meta.get("session_id", ""),
            "timestamp"  : meta.get("timestamp", ""),
        })

    return facts


def get_fact_count(user_id: str) -> int:
    """Returns the number of facts stored for a user."""
    collection = _get_collection(user_id)
    return collection.count()


# ── Delete (use sparingly — vault is meant to be immutable) ───────────────────

def delete_fact(user_id: str, fact_id: str) -> bool:
    """
    Removes a specific fact by ID.
    Should only be used if a fact was added in error.

    Args:
        user_id : unique identifier for the user
        fact_id : the fact ID returned by add_fact()

    Returns:
        True if deleted, False if not found
    """
    collection = _get_collection(user_id)

    try:
        collection.delete(ids=[fact_id])
        print(f"[fact_vault] Fact '{fact_id}' deleted for user '{user_id}'")
        return True
    except Exception as e:
        print(f"[fact_vault] Could not delete fact '{fact_id}': {e}")
        return False


def clear_user_vault(user_id: str) -> None:
    """
    Deletes ALL facts for a user.
    Use only for testing or if user requests full data deletion.
    """
    client          = _get_client()
    collection_name = f"facts_{user_id.replace('-', '_')}"

    try:
        client.delete_collection(collection_name)
        print(f"[fact_vault] All facts cleared for user '{user_id}'")
    except Exception as e:
        print(f"[fact_vault] Could not clear vault for user '{user_id}': {e}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_USER = "test_user_001"

    print("\n── Adding facts ──")
    id1 = add_fact(TEST_USER, "Patient has a supportive and loving partner.")
    id2 = add_fact(TEST_USER, "Patient recently received a promotion at work.")
    id3 = add_fact(TEST_USER, "Patient has a close group of friends they meet weekly.")
    id4 = add_fact(TEST_USER, "Patient mentioned feeling proud of completing a difficult project.")
    id5 = add_fact(TEST_USER, "Patient has two siblings they are close with.")

    print(f"\n── Total facts stored: {get_fact_count(TEST_USER)} ──")

    print("\n── Searching relevant facts ──")
    query   = "I always fail at everything, nobody loves me"
    results = search_facts(TEST_USER, query, n_results=3)
    print(f"Query  : {query}")
    print(f"Results:")
    for r in results:
        print(f"  - {r}")

    print("\n── All facts ──")
    all_facts = get_all_facts(TEST_USER)
    for f in all_facts:
        print(f"  [{f['fact_id'][:8]}...] {f['fact']}")

    print("\n── Cleaning up test data ──")
    clear_user_vault(TEST_USER)
    print(f"Facts remaining: {get_fact_count(TEST_USER)}")
