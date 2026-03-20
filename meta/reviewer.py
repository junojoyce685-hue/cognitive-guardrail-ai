"""
reviewer.py
-----------
Async Meta-Cognitive Reviewer — the self-evolving loop.

Runs every 10 sessions per user and:
1. Reads Devil's Advocate override logs
2. Detects patterns where the Analyst was wrong repeatedly
3. Detects patterns where the Responder scored low on quality
4. Automatically patches the Analyst and Responder system prompts
   with caution flags so they improve over time

This is what makes the system self-evolving without manual retraining.

Exposes:
    - log_session_result()       → called after every session to log outcomes
    - run_reviewer()             → runs the full review (every 10 sessions)
    - get_analyst_patch()        → returns current patched analyst prompt
    - get_responder_patch()      → returns current patched responder prompt
    - get_session_count()        → how many sessions logged for a user
"""

import json
import re
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from config import GROQ_API_KEY

# ── Storage paths ─────────────────────────────────────────────────────────────
LOGS_DIR    = ROOT_DIR / "meta" / "logs"
PATCHES_DIR = ROOT_DIR / "meta" / "patches"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL               = "llama-3.3-70b-versatile"
MAX_TOKENS          = 1500
REVIEW_EVERY_N      = 10  # run reviewer every N sessions
CHALLENGE_THRESHOLD = 3   # if overridden 3+ times on same label → patch analyst

# ── System Prompts ────────────────────────────────────────────────────────────

ANALYST_PATCH_PROMPT = """You are a system prompt optimizer for a cognitive distortion detection AI.

You will receive a log of sessions where the Analyst agent was CHALLENGED and overridden
by the Devil's Advocate. Your job is to write a SHORT caution addendum to append to the
Analyst's system prompt to help it avoid repeating these mistakes.

RULES:
- Be specific about which distortion types the Analyst is over-detecting or mislabeling
- Keep the patch under 100 words
- Write it as a direct instruction to the Analyst (second person — "You should...", "Be careful...")
- Do NOT rewrite the full system prompt — only write the addendum patch

Respond with ONLY the patch text. No JSON, no markdown, no preamble."""


RESPONDER_PATCH_PROMPT = """You are a system prompt optimizer for a CBT therapy response AI.

You will receive a log of sessions where the Responder's output was rated as unhelpful
or low quality by users. Your job is to write a SHORT caution addendum to append to the
Responder's system prompt to help it improve.

RULES:
- Be specific about what kinds of responses users found unhelpful
- Keep the patch under 100 words
- Write it as a direct instruction to the Responder
- Do NOT rewrite the full system prompt — only write the addendum patch

Respond with ONLY the patch text. No JSON, no markdown, no preamble."""


# ── Session Log Functions ─────────────────────────────────────────────────────

def _get_log_path(user_id: str) -> Path:
    return LOGS_DIR / f"{user_id}_session_log.json"


def _get_patch_path(user_id: str, agent: str) -> Path:
    return PATCHES_DIR / f"{user_id}_{agent}_patch.txt"


def log_session_result(
    user_id         : str,
    session_id      : str,
    analyst_label   : str,
    da_verdict      : str,
    da_suggested    : str,
    consensus_loops : int,
    user_feedback   : str,
    flagged         : bool,
    technique       : str,
) -> None:
    """
    Logs the outcome of a single session for later review.
    Called automatically by main.py after every session.

    Args:
        user_id         : unique user identifier
        session_id      : session identifier
        analyst_label   : what the Analyst classified
        da_verdict      : AGREE or CHALLENGE
        da_suggested    : alternative label suggested by Devil's Advocate
        consensus_loops : how many loops were needed
        user_feedback   : helpful / not helpful / none
        flagged         : whether crisis was detected
        technique       : CBT technique used by Responder
    """
    log_path = _get_log_path(user_id)

    # Load existing log
    if log_path.exists():
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = {"user_id": user_id, "sessions": []}

    # Append new session
    log["sessions"].append({
        "session_id"     : session_id,
        "timestamp"      : datetime.utcnow().isoformat(),
        "analyst_label"  : analyst_label,
        "da_verdict"     : da_verdict,
        "da_suggested"   : da_suggested,
        "consensus_loops": consensus_loops,
        "user_feedback"  : user_feedback or "none",
        "flagged"        : flagged,
        "technique"      : technique,
    })

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    session_count = len(log["sessions"])
    print(f"[reviewer] Session logged ({session_count} total) for user '{user_id}'")

    # Auto-trigger reviewer every N sessions
    if session_count % REVIEW_EVERY_N == 0:
        print(f"[reviewer] {REVIEW_EVERY_N} sessions reached — triggering review...")
        asyncio.run(run_reviewer(user_id))


def get_session_count(user_id: str) -> int:
    """Returns how many sessions have been logged for a user."""
    log_path = _get_log_path(user_id)
    if not log_path.exists():
        return 0
    with open(log_path, "r") as f:
        log = json.load(f)
    return len(log.get("sessions", []))


def get_session_log(user_id: str) -> List[Dict]:
    """Returns the full session log for a user."""
    log_path = _get_log_path(user_id)
    if not log_path.exists():
        return []
    with open(log_path, "r") as f:
        log = json.load(f)
    return log.get("sessions", [])


# ── Analysis Functions ────────────────────────────────────────────────────────

def _analyze_analyst_errors(sessions: List[Dict]) -> Dict:
    """
    Detects patterns in Analyst misclassifications.

    Returns:
        dict with:
            - overridden_labels : Counter of labels that got CHALLENGED
            - needs_patch       : True if any label was overridden 3+ times
            - problem_labels    : list of labels that exceed the threshold
    """
    challenged_sessions = [
        s for s in sessions
        if s.get("da_verdict") == "CHALLENGE"
    ]

    overridden_labels = Counter(
        s["analyst_label"] for s in challenged_sessions
        if s.get("analyst_label")
    )

    problem_labels = [
        label for label, count in overridden_labels.items()
        if count >= CHALLENGE_THRESHOLD
    ]

    return {
        "total_challenges" : len(challenged_sessions),
        "overridden_labels": dict(overridden_labels),
        "problem_labels"   : problem_labels,
        "needs_patch"      : len(problem_labels) > 0,
    }


def _analyze_responder_quality(sessions: List[Dict]) -> Dict:
    """
    Detects patterns in Responder quality issues.

    Returns:
        dict with:
            - unhelpful_count : how many sessions got negative feedback
            - unhelpful_rate  : ratio of unhelpful sessions
            - needs_patch     : True if unhelpful rate > 30%
            - bad_techniques  : techniques associated with negative feedback
    """
    total     = len(sessions)
    unhelpful = [
        s for s in sessions
        if s.get("user_feedback", "").lower() in ("not helpful", "bad", "unhelpful", "no")
    ]

    bad_techniques = Counter(
        s.get("technique", "") for s in unhelpful
        if s.get("technique")
    )

    unhelpful_rate = len(unhelpful) / total if total > 0 else 0.0

    return {
        "total_sessions" : total,
        "unhelpful_count": len(unhelpful),
        "unhelpful_rate" : round(unhelpful_rate, 2),
        "bad_techniques" : dict(bad_techniques),
        "needs_patch"    : unhelpful_rate > 0.30,
    }


# ── Patch Generation ──────────────────────────────────────────────────────────

async def _generate_analyst_patch(
    analysis: Dict,
    sessions: List[Dict],
) -> str:
    """Uses Groq to generate a targeted system prompt patch for the Analyst."""
    client = Groq(api_key=GROQ_API_KEY)

    problem_summary = f"""
The Analyst agent has been overridden by the Devil's Advocate in the following patterns:

Problem labels (overridden {CHALLENGE_THRESHOLD}+ times):
{json.dumps(analysis['problem_labels'], indent=2)}

Full override counts by label:
{json.dumps(analysis['overridden_labels'], indent=2)}

Recent challenge examples:
"""
    challenged = [s for s in sessions if s.get("da_verdict") == "CHALLENGE"][-5:]
    for s in challenged:
        problem_summary += (
            f"  - Analyst said '{s['analyst_label']}' → "
            f"DA suggested '{s.get('da_suggested', 'unknown')}'\n"
        )

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": ANALYST_PATCH_PROMPT},
            {"role": "user",   "content": problem_summary},
        ]
    )

    patch = response.choices[0].message.content.strip()
    print(f"[reviewer] Analyst patch generated ({len(patch)} chars)")
    return patch


async def _generate_responder_patch(
    analysis: Dict,
    sessions: List[Dict],
) -> str:
    """Uses Groq to generate a targeted system prompt patch for the Responder."""
    client = Groq(api_key=GROQ_API_KEY)

    problem_summary = f"""
The Responder agent received negative user feedback in {analysis['unhelpful_count']}
out of {analysis['total_sessions']} sessions ({analysis['unhelpful_rate']*100:.0f}% unhelpful rate).

Techniques associated with negative feedback:
{json.dumps(analysis['bad_techniques'], indent=2)}

Recent unhelpful sessions:
"""
    unhelpful = [
        s for s in sessions
        if s.get("user_feedback", "").lower() in ("not helpful", "bad", "unhelpful", "no")
    ][-5:]

    for s in unhelpful:
        problem_summary += (
            f"  - Technique '{s.get('technique', 'unknown')}' → "
            f"feedback: '{s.get('user_feedback')}'\n"
        )

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": RESPONDER_PATCH_PROMPT},
            {"role": "user",   "content": problem_summary},
        ]
    )

    patch = response.choices[0].message.content.strip()
    print(f"[reviewer] Responder patch generated ({len(patch)} chars)")
    return patch


def _save_patch(user_id: str, agent: str, patch: str) -> None:
    """Saves a patch to disk."""
    patch_path = _get_patch_path(user_id, agent)
    with open(patch_path, "a") as f:
        f.write(f"\n\n# Patch generated: {datetime.utcnow().isoformat()}\n")
        f.write(patch)
    print(f"[reviewer] Patch saved: {patch_path}")


def _load_patch(user_id: str, agent: str) -> str:
    """
    Loads the most recent patch for a user+agent combination.
    Returns empty string if no patch exists.
    """
    patch_path = _get_patch_path(user_id, agent)
    if not patch_path.exists():
        return ""
    with open(patch_path, "r") as f:
        content = f.read()
    blocks = content.strip().split("# Patch generated:")
    if len(blocks) > 1:
        return "# Patch generated:" + blocks[-1].strip()
    return content.strip()


# ── Main Reviewer ─────────────────────────────────────────────────────────────

async def run_reviewer(user_id: str) -> Dict:
    """
    Runs the full Meta-Cognitive Review for a user.

    Analyzes session logs, generates patches if needed,
    and saves them for the pipeline to pick up next session.

    Args:
        user_id : unique user identifier

    Returns:
        dict with review summary
    """
    print(f"\n[reviewer] ══ RUNNING META-COGNITIVE REVIEW for '{user_id}' ══")

    sessions = get_session_log(user_id)
    if not sessions:
        print(f"[reviewer] No sessions to review.")
        return {"status": "no_sessions"}

    # Analyze patterns
    analyst_analysis   = _analyze_analyst_errors(sessions)
    responder_analysis = _analyze_responder_quality(sessions)

    print(f"[reviewer] Analyst overrides  : {analyst_analysis['total_challenges']}")
    print(f"[reviewer] Problem labels     : {analyst_analysis['problem_labels']}")
    print(f"[reviewer] Unhelpful rate     : {responder_analysis['unhelpful_rate']*100:.0f}%")

    analyst_patch_text   = ""
    responder_patch_text = ""

    # Generate and save analyst patch if needed
    if analyst_analysis["needs_patch"]:
        print(f"[reviewer] Generating Analyst patch...")
        analyst_patch_text = await _generate_analyst_patch(analyst_analysis, sessions)
        _save_patch(user_id, "analyst", analyst_patch_text)
    else:
        print(f"[reviewer] Analyst performing well — no patch needed.")

    # Generate and save responder patch if needed
    if responder_analysis["needs_patch"]:
        print(f"[reviewer] Generating Responder patch...")
        responder_patch_text = await _generate_responder_patch(responder_analysis, sessions)
        _save_patch(user_id, "responder", responder_patch_text)
    else:
        print(f"[reviewer] Responder performing well — no patch needed.")

    review_summary = {
        "status"            : "completed",
        "sessions_reviewed" : len(sessions),
        "analyst_patched"   : analyst_analysis["needs_patch"],
        "responder_patched" : responder_analysis["needs_patch"],
        "problem_labels"    : analyst_analysis["problem_labels"],
        "unhelpful_rate"    : responder_analysis["unhelpful_rate"],
        "analyst_patch"     : analyst_patch_text,
        "responder_patch"   : responder_patch_text,
        "timestamp"         : datetime.utcnow().isoformat(),
    }

    print(f"[reviewer] ══ REVIEW COMPLETE ══\n")
    return review_summary


# ── Patch Getters (used by pipeline) ─────────────────────────────────────────

def get_analyst_patch(user_id: str) -> str:
    """
    Returns the current analyst patch for a user.
    Called by pipeline.py to inject into the analyst system prompt.
    Returns empty string if no patch exists yet.
    """
    return _load_patch(user_id, "analyst")


def get_responder_patch(user_id: str) -> str:
    """
    Returns the current responder patch for a user.
    Called by pipeline.py to inject into the responder system prompt.
    Returns empty string if no patch exists yet.
    """
    return _load_patch(user_id, "responder")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_USER = "test_user_reviewer_001"

    print("\n── Simulating 12 sessions ──")

    simulated_sessions = [
        *[{
            "session_id"     : f"s{i:03d}",
            "analyst_label"  : "Overgeneralization",
            "da_verdict"     : "CHALLENGE",
            "da_suggested"   : "Emotional Reasoning",
            "consensus_loops": 1,
            "user_feedback"  : "helpful",
            "flagged"        : False,
            "technique"      : "Socratic questioning",
        } for i in range(1, 5)],
        *[{
            "session_id"     : f"s{i:03d}",
            "analyst_label"  : "Mind Reading",
            "da_verdict"     : "AGREE",
            "da_suggested"   : "",
            "consensus_loops": 0,
            "user_feedback"  : "not helpful",
            "flagged"        : False,
            "technique"      : "Thought reframing",
        } for i in range(5, 9)],
        *[{
            "session_id"     : f"s{i:03d}",
            "analyst_label"  : "Labeling",
            "da_verdict"     : "AGREE",
            "da_suggested"   : "",
            "consensus_loops": 0,
            "user_feedback"  : "helpful",
            "flagged"        : False,
            "technique"      : "Self-compassion nudge",
        } for i in range(9, 13)],
    ]

    log_path = _get_log_path(TEST_USER)
    with open(log_path, "w") as f:
        json.dump({"user_id": TEST_USER, "sessions": simulated_sessions}, f, indent=2)

    print(f"Simulated {len(simulated_sessions)} sessions written.")

    result = asyncio.run(run_reviewer(TEST_USER))

    print(f"\n── Review Summary ──")
    print(f"Status           : {result['status']}")
    print(f"Sessions reviewed: {result['sessions_reviewed']}")
    print(f"Analyst patched  : {result['analyst_patched']}")
    print(f"Responder patched: {result['responder_patched']}")
    print(f"Problem labels   : {result['problem_labels']}")
    print(f"Unhelpful rate   : {result['unhelpful_rate']*100:.0f}%")

    if result["analyst_patch"]:
        print(f"\nAnalyst Patch:\n{result['analyst_patch']}")
    if result["responder_patch"]:
        print(f"\nResponder Patch:\n{result['responder_patch']}")

    print(f"\n── Patch Retrieval Test ──")
    print(f"Analyst patch loaded  : {len(get_analyst_patch(TEST_USER))} chars")
    print(f"Responder patch loaded: {len(get_responder_patch(TEST_USER))} chars")

    # Cleanup
    log_path.unlink(missing_ok=True)
    _get_patch_path(TEST_USER, "analyst").unlink(missing_ok=True)
    _get_patch_path(TEST_USER, "responder").unlink(missing_ok=True)
    print("\nTest cleanup done.")