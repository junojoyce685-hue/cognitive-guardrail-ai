"""
stats_tracker.py
----------------
Tracks and aggregates distortion detection data for:
    - Per-user stats  : distortions detected for a specific user over time
    - Global stats    : distortions detected across ALL users

Data is read from the session log files already written by reviewer.py
No extra storage needed — everything is derived from existing logs.

Exposes:
    - get_user_stats()    → stats for a single user
    - get_global_stats()  → aggregated stats across all users
    - log_detection()     → write a detection event to the stats store
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
LOGS_DIR   = ROOT_DIR / "meta" / "logs"
STATS_FILE = ROOT_DIR / "meta" / "global_stats.json"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_log_path(user_id: str) -> Path:
    return LOGS_DIR / f"{user_id}_session_log.json"


def _load_user_log(user_id: str) -> List[Dict]:
    path = _get_log_path(user_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sessions", [])


def _get_all_user_ids() -> List[str]:
    """Returns all user IDs that have session logs."""
    ids = []
    for log_file in LOGS_DIR.glob("*_session_log.json"):
        user_id = log_file.name.replace("_session_log.json", "")
        ids.append(user_id)
    return ids


# ── Per-user stats ────────────────────────────────────────────────────────────

def get_user_stats(user_id: str) -> Dict:
    """
    Returns full stats for a single user.

    Args:
        user_id : unique user identifier

    Returns:
        dict with:
            - total_sessions       : int
            - total_distortions    : int (excludes "No distortion")
            - distortion_counts    : dict label → count
            - distortion_pct       : dict label → percentage
            - top_distortion       : most common label
            - no_distortion_count  : sessions with no distortion
            - helpful_rate         : % sessions marked helpful
            - da_challenge_rate    : % sessions where DA challenged
            - avg_confidence       : average analyst confidence
            - timeline             : list of {date, label, confidence}
            - techniques_used      : dict technique → count
            - streak               : consecutive sessions with same top distortion
    """
    sessions = _load_user_log(user_id)

    if not sessions:
        return {"total_sessions": 0, "message": "No sessions logged yet."}

    total = len(sessions)

    # Distortion counts
    labels = [s.get("analyst_label", "") for s in sessions if s.get("analyst_label")]
    distortion_counts = Counter(labels)
    no_distortion_count = distortion_counts.pop("No distortion", 0)
    total_distortions = sum(distortion_counts.values())

    # Percentages (of distorted sessions only)
    distortion_pct = {}
    if total_distortions > 0:
        distortion_pct = {
            label: round(count / total_distortions * 100, 1)
            for label, count in distortion_counts.items()
        }

    # Top distortion
    top_distortion = distortion_counts.most_common(1)[0][0] if distortion_counts else "None"

    # Helpful rate
    helpful = [s for s in sessions if s.get("user_feedback", "").lower() == "helpful"]
    helpful_rate = round(len(helpful) / total * 100, 1) if total > 0 else 0.0

    # DA challenge rate
    challenged = [s for s in sessions if s.get("da_verdict") == "CHALLENGE"]
    da_challenge_rate = round(len(challenged) / total * 100, 1) if total > 0 else 0.0

    # Average confidence
    confidences = []
    for s in sessions:
        # confidence not always in log — skip if missing
        pass
    avg_confidence = 0.0  # placeholder — confidence logged in inference layer not reviewer

    # Timeline (last 30 sessions)
    timeline = []
    for s in sessions[-30:]:
        timeline.append({
            "timestamp" : s.get("timestamp", "")[:10],  # date only
            "label"     : s.get("analyst_label", ""),
            "verdict"   : s.get("da_verdict", ""),
            "feedback"  : s.get("user_feedback", ""),
        })

    # Techniques used
    techniques = Counter(
        s.get("technique", "") for s in sessions
        if s.get("technique")
    )

    # Current streak — consecutive sessions with same label
    streak = 0
    if sessions:
        last_label = sessions[-1].get("analyst_label", "")
        for s in reversed(sessions):
            if s.get("analyst_label") == last_label:
                streak += 1
            else:
                break

    return {
        "total_sessions"     : total,
        "total_distortions"  : total_distortions,
        "no_distortion_count": no_distortion_count,
        "distortion_counts"  : dict(distortion_counts.most_common()),
        "distortion_pct"     : distortion_pct,
        "top_distortion"     : top_distortion,
        "helpful_rate"       : helpful_rate,
        "da_challenge_rate"  : da_challenge_rate,
        "timeline"           : timeline,
        "techniques_used"    : dict(techniques.most_common()),
        "streak"             : streak,
        "streak_label"       : sessions[-1].get("analyst_label", "") if sessions else "",
    }


# ── Global stats ──────────────────────────────────────────────────────────────

def get_global_stats() -> Dict:
    """
    Returns aggregated stats across ALL users.

    Returns:
        dict with:
            - total_users          : int
            - total_sessions       : int
            - total_distortions    : int
            - distortion_counts    : dict label → count (all users combined)
            - distortion_pct       : dict label → percentage
            - top_distortion       : most common across all users
            - per_user_summary     : list of {user_id, sessions, top_distortion}
            - helpful_rate         : global helpful rate
            - da_challenge_rate    : global DA challenge rate
            - most_active_users    : top 5 users by session count
    """
    user_ids = _get_all_user_ids()

    if not user_ids:
        return {"total_users": 0, "message": "No data collected yet."}

    all_sessions    = []
    per_user_summary = []

    for uid in user_ids:
        sessions = _load_user_log(uid)
        all_sessions.extend(sessions)

        if sessions:
            labels = [s.get("analyst_label", "") for s in sessions]
            label_counts = Counter(labels)
            label_counts.pop("No distortion", None)
            top = label_counts.most_common(1)[0][0] if label_counts else "None"
            per_user_summary.append({
                "user_id"       : uid,
                "sessions"      : len(sessions),
                "top_distortion": top,
            })

    total_sessions = len(all_sessions)
    total_users    = len(user_ids)

    # Global distortion counts
    all_labels = [s.get("analyst_label", "") for s in all_sessions if s.get("analyst_label")]
    global_counts = Counter(all_labels)
    no_dist_global = global_counts.pop("No distortion", 0)
    total_distortions = sum(global_counts.values())

    global_pct = {}
    if total_distortions > 0:
        global_pct = {
            label: round(count / total_distortions * 100, 1)
            for label, count in global_counts.items()
        }

    top_distortion = global_counts.most_common(1)[0][0] if global_counts else "None"

    # Global helpful rate
    helpful = [s for s in all_sessions if s.get("user_feedback", "").lower() == "helpful"]
    helpful_rate = round(len(helpful) / total_sessions * 100, 1) if total_sessions > 0 else 0.0

    # Global DA challenge rate
    challenged = [s for s in all_sessions if s.get("da_verdict") == "CHALLENGE"]
    da_challenge_rate = round(len(challenged) / total_sessions * 100, 1) if total_sessions > 0 else 0.0

    # Most active users
    most_active = sorted(per_user_summary, key=lambda x: x["sessions"], reverse=True)[:5]

    # Global timeline — sessions per day
    daily_counts = Counter(
        s.get("timestamp", "")[:10]
        for s in all_sessions
        if s.get("timestamp")
    )
    timeline = [
        {"date": date, "count": count}
        for date, count in sorted(daily_counts.items())
    ]

    return {
        "total_users"        : total_users,
        "total_sessions"     : total_sessions,
        "total_distortions"  : total_distortions,
        "no_distortion_count": no_dist_global,
        "distortion_counts"  : dict(global_counts.most_common()),
        "distortion_pct"     : global_pct,
        "top_distortion"     : top_distortion,
        "per_user_summary"   : per_user_summary,
        "most_active_users"  : most_active,
        "helpful_rate"       : helpful_rate,
        "da_challenge_rate"  : da_challenge_rate,
        "timeline"           : timeline,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Simulate some session logs for testing
    test_users = {
        "alice": [
            {"analyst_label": "Overgeneralization", "da_verdict": "AGREE",     "user_feedback": "helpful",     "technique": "Socratic questioning", "timestamp": "2026-03-01T10:00:00"},
            {"analyst_label": "Overgeneralization", "da_verdict": "AGREE",     "user_feedback": "helpful",     "technique": "Socratic questioning", "timestamp": "2026-03-02T10:00:00"},
            {"analyst_label": "Mind Reading",       "da_verdict": "CHALLENGE",  "user_feedback": "not helpful", "technique": "Thought reframing",    "timestamp": "2026-03-03T10:00:00"},
            {"analyst_label": "Labeling",           "da_verdict": "AGREE",     "user_feedback": "helpful",     "technique": "Self-compassion nudge","timestamp": "2026-03-04T10:00:00"},
            {"analyst_label": "No distortion",      "da_verdict": "AGREE",     "user_feedback": "helpful",     "technique": "Validation",           "timestamp": "2026-03-05T10:00:00"},
        ],
        "bob": [
            {"analyst_label": "Magnification",      "da_verdict": "AGREE",     "user_feedback": "helpful",     "technique": "Decatastrophizing",    "timestamp": "2026-03-01T11:00:00"},
            {"analyst_label": "Fortune-telling",    "da_verdict": "CHALLENGE",  "user_feedback": "not helpful", "technique": "Evidence examination", "timestamp": "2026-03-02T11:00:00"},
            {"analyst_label": "Overgeneralization", "da_verdict": "AGREE",     "user_feedback": "helpful",     "technique": "Socratic questioning", "timestamp": "2026-03-03T11:00:00"},
        ],
    }

    for uid, sessions in test_users.items():
        log_path = _get_log_path(uid)
        with open(log_path, "w") as f:
            json.dump({"user_id": uid, "sessions": sessions}, f, indent=2)

    print("\n── User stats (alice) ──")
    stats = get_user_stats("alice")
    print(f"Total sessions   : {stats['total_sessions']}")
    print(f"Top distortion   : {stats['top_distortion']}")
    print(f"Distortion counts: {stats['distortion_counts']}")
    print(f"Helpful rate     : {stats['helpful_rate']}%")
    print(f"DA challenge rate: {stats['da_challenge_rate']}%")

    print("\n── Global stats ──")
    g = get_global_stats()
    print(f"Total users      : {g['total_users']}")
    print(f"Total sessions   : {g['total_sessions']}")
    print(f"Top distortion   : {g['top_distortion']}")
    print(f"Distortion counts: {g['distortion_counts']}")

    # Cleanup
    for uid in test_users:
        _get_log_path(uid).unlink(missing_ok=True)
    print("\nCleanup done.")
