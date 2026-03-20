"""
evaluate.py
-----------
Comprehensive evaluation suite for Cognitive Guardrail AI.

Evaluations:
    1. Classification accuracy + macro F1 + per-label breakdown + confusion matrix
    2. Confidence calibration (when model says 0.9, is it right 90% of the time?)
    3. Memory vs No-Memory (proper pipeline comparison with DA using memory facts)
    4. DA override impact (when DA challenges, does the final label improve?)
    5. Memory audit effectiveness (detection rate + false positive rate)
    6. Cohen's Kappa (agreement between system and human annotators)

Outputs:
    eval_results.json              — all numbers
    eval_confusion_matrix.png      — confusion matrix heatmap
    eval_calibration_curve.png     — confidence calibration curve
    eval_summary_report.txt        — human-readable summary

Run from project root:
    python evaluate.py

Token estimate: ~80,000 tokens total (fits in Groq free tier 100k/day)
To reduce cost: set MAX_SAMPLES = 50, MEMORY_COMPARE_N = 20
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
TEST_SIZE        = 0.20   # 20% held-out test set
RANDOM_SEED      = 42
MAX_SAMPLES      = 100    # cap for eval 1 — increase for full eval
MEMORY_COMPARE_N = 30     # samples for eval 3
DA_EVAL_N        = 40     # samples for DA override eval
SLEEP_BETWEEN    = 2      # seconds between API calls
CALIBRATION_BINS = 5      # number of confidence bins for calibration

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from training.csv_loader import load_merged
from agents.analyst import run_analyst
from agents.devils_advocate import run_consensus_loop
from memory.fact_vault import search_facts, add_fact, clear_user_vault
from memory.inference_layer import (
    add_session_inference,
    clear_user_memory,
)
from agents.memory_architect import audit_ai_memory

LABELS = [
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

LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_call(fn, *args, retries=3, **kwargs):
    """Wraps any function with retry logic on rate limit."""
    for attempt in range(retries):
        try:
            result = fn(*args, **kwargs)
            time.sleep(SLEEP_BETWEEN)
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = (attempt + 1) * 30
                print(f"    [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [error attempt {attempt+1}]: {err[:80]}")
                time.sleep(5)
    return None


def normalize_label(label: str) -> str:
    """Normalizes label strings for consistent comparison."""
    mappings = {
        "all or nothing thinking" : "All-or-nothing thinking",
        "all-or-nothing"          : "All-or-nothing thinking",
        "overgeneralization"      : "Overgeneralization",
        "over generalization"     : "Overgeneralization",
        "mental filter"           : "Mental filter",
        "should statements"       : "Should statements",
        "labeling"                : "Labeling",
        "labelling"               : "Labeling",
        "personalization"         : "Personalization",
        "personalisation"         : "Personalization",
        "magnification"           : "Magnification",
        "catastrophizing"         : "Magnification",
        "emotional reasoning"     : "Emotional Reasoning",
        "mind reading"            : "Mind Reading",
        "fortune telling"         : "Fortune-telling",
        "fortune-telling"         : "Fortune-telling",
        "no distortion"           : "No distortion",
    }
    return mappings.get(label.strip().lower(), label.strip())


def get_test_split(df: pd.DataFrame):
    """Returns stratified train/test split."""
    _, test = train_test_split(
        df,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = df["Dominant Distortion"],
    )
    # Cap and balance
    if len(test) > MAX_SAMPLES:
        test = test.groupby("Dominant Distortion", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(MAX_SAMPLES * len(x) / len(test)))),
                random_state=RANDOM_SEED
            )
        ).reset_index(drop=True)
    return test


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Evaluation 1: Classification Accuracy + Cohen's Kappa ────────────────────

def eval_classification(df: pd.DataFrame) -> dict:
    """
    Measures how well the Analyst classifies distortions
    against human annotations (ground truth from the dataset).

    Metrics:
        - Accuracy
        - Macro F1, Precision, Recall
        - Per-label breakdown
        - Cohen's Kappa (inter-rater agreement)
        - Confusion matrix
    """
    print_section("EVAL 1: Classification Accuracy + Cohen's Kappa")

    test_df = get_test_split(df)
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:\n{test_df['Dominant Distortion'].value_counts().to_string()}\n")

    y_true      = []
    y_pred      = []
    confidences = []
    details     = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        text       = row["Patient Question"]
        true_label = normalize_label(row["Dominant Distortion"])

        print(f"  [{idx+1}/{len(test_df)}] {text[:55]}...")

        result = safe_call(run_analyst, text, df, k=5)
        if result is None:
            print(f"    Skipping — call failed")
            continue

        pred_label = normalize_label(result.get("label", "No distortion"))
        confidence = float(result.get("confidence", 0.5))

        correct = true_label == pred_label
        y_true.append(true_label)
        y_pred.append(pred_label)
        confidences.append(confidence)

        details.append({
            "text"      : text[:100],
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "correct"   : correct,
        })

        print(f"    True: {true_label:<30} Pred: {pred_label:<30} {'✓' if correct else '✗'} ({confidence:.2f})")

    if not y_true:
        return {"error": "No samples evaluated"}

    present_labels = sorted(list(set(y_true + y_pred)))

    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0, labels=present_labels)
    macro_p   = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=present_labels)
    macro_r   = recall_score(y_true, y_pred, average="macro",  zero_division=0, labels=present_labels)
    kappa     = cohen_kappa_score(y_true, y_pred, labels=present_labels)
    cm        = confusion_matrix(y_true, y_pred, labels=present_labels)
    report    = classification_report(y_true, y_pred, labels=present_labels, zero_division=0, output_dict=True)

    print(f"\n── Results ──")
    print(f"  Accuracy     : {accuracy:.3f}  ({accuracy*100:.1f}%)")
    print(f"  Macro F1     : {macro_f1:.3f}")
    print(f"  Macro P      : {macro_p:.3f}")
    print(f"  Macro R      : {macro_r:.3f}")
    print(f"  Cohen's Kappa: {kappa:.3f}  ({_kappa_interpretation(kappa)})")
    print(f"\n  Per-label breakdown:")
    for label in present_labels:
        if label in report:
            m = report[label]
            print(f"    {label:<32} P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}  n={int(m['support'])}")

    return {
        "n_samples"      : len(y_true),
        "accuracy"       : round(accuracy, 4),
        "macro_f1"       : round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall"   : round(macro_r, 4),
        "cohen_kappa"    : round(kappa, 4),
        "kappa_interp"   : _kappa_interpretation(kappa),
        "per_label"      : {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if isinstance(v, dict)},
        "confusion_matrix" : cm.tolist(),
        "confusion_labels" : present_labels,
        "confidences"    : confidences,
        "y_true"         : y_true,
        "y_pred"         : y_pred,
        "details"        : details,
    }


def _kappa_interpretation(k: float) -> str:
    if k < 0:    return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


# ── Evaluation 2: Confidence Calibration ──────────────────────────────────────

def eval_calibration(classification_results: dict) -> dict:
    """
    Measures whether the model's confidence scores are well-calibrated.

    A well-calibrated model that says '0.9 confidence' should be correct
    approximately 90% of the time. Overconfident models say 0.9 but are
    only right 60% of the time.

    Uses results already collected in eval_classification — no extra API calls.
    """
    print_section("EVAL 2: Confidence Calibration")

    details     = classification_results.get("details", [])
    confidences = [d["confidence"] for d in details]
    correct     = [1 if d["correct"] else 0 for d in details]

    if len(confidences) < 10:
        print("  Not enough samples for calibration analysis")
        return {"error": "insufficient samples"}

    # Bin confidences
    bins       = np.linspace(0, 1, CALIBRATION_BINS + 1)
    bin_accs   = []
    bin_confs  = []
    bin_counts = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask   = [lo <= c < hi for c in confidences]
        if sum(mask) == 0:
            continue
        bin_c = [confidences[j] for j in range(len(confidences)) if mask[j]]
        bin_a = [correct[j]     for j in range(len(correct))     if mask[j]]
        bin_accs.append(np.mean(bin_a))
        bin_confs.append(np.mean(bin_c))
        bin_counts.append(len(bin_c))

    # Expected Calibration Error (ECE)
    total  = len(confidences)
    ece    = sum(
        bin_counts[i] / total * abs(bin_accs[i] - bin_confs[i])
        for i in range(len(bin_accs))
    )

    avg_confidence = np.mean(confidences)
    avg_accuracy   = np.mean(correct)
    overconfidence = avg_confidence - avg_accuracy

    print(f"  Avg confidence : {avg_confidence:.3f}")
    print(f"  Avg accuracy   : {avg_accuracy:.3f}")
    print(f"  Overconfidence : {overconfidence:+.3f}  ({'overconfident' if overconfidence > 0.05 else 'well-calibrated' if abs(overconfidence) <= 0.05 else 'underconfident'})")
    print(f"  ECE (↓ better) : {ece:.3f}")
    print(f"\n  Confidence bins:")
    for i in range(len(bin_accs)):
        bar = "█" * int(bin_counts[i] / max(bin_counts) * 20)
        print(f"    conf {bin_confs[i]:.2f} → acc {bin_accs[i]:.2f}  n={bin_counts[i]}  {bar}")

    return {
        "avg_confidence"  : round(float(avg_confidence), 4),
        "avg_accuracy"    : round(float(avg_accuracy), 4),
        "overconfidence"  : round(float(overconfidence), 4),
        "ece"             : round(float(ece), 4),
        "calibration_status": "overconfident" if overconfidence > 0.05 else "well-calibrated" if abs(overconfidence) <= 0.05 else "underconfident",
        "bin_confidences" : [round(float(x), 3) for x in bin_confs],
        "bin_accuracies"  : [round(float(x), 3) for x in bin_accs],
        "bin_counts"      : bin_counts,
    }


# ── Evaluation 3: Memory vs No-Memory (proper pipeline comparison) ────────────

def eval_memory_vs_no_memory(df: pd.DataFrame) -> dict:
    """
    Properly compares the full pipeline WITH vs WITHOUT memory.

    WITH memory:
        - Analyst classifies
        - Devil's Advocate receives real memory facts from ChromaDB
        - DA can challenge based on stored user context

    WITHOUT memory:
        - Same analyst call
        - DA receives empty memory facts list
        - DA has no user context to draw on

    This is the correct comparison — not just calling analyst twice.
    """
    print_section("EVAL 3: Memory vs No-Memory (Full Pipeline)")

    TEST_USER = "eval_memory_test_user_42"

    # Use distorted samples only — memory is most relevant here
    distorted = df[df["Dominant Distortion"] != "No distortion"].sample(
        min(MEMORY_COMPARE_N, len(df[df["Dominant Distortion"] != "No distortion"])),
        random_state=RANDOM_SEED
    )

    print(f"Samples: {len(distorted)}")
    print(f"Setting up test user memory...")

    # Pre-populate ChromaDB with relevant facts for test user
    # These simulate facts the system would have learned over sessions
    test_facts = [
        "User has a supportive partner who cares about them",
        "User recently received a promotion at work",
        "User has a close group of friends they meet regularly",
        "User completed a difficult project successfully last month",
        "User has shown resilience in previous sessions",
        "User has two siblings they are close with",
        "User attends therapy regularly and is committed to growth",
    ]

    clear_user_vault(TEST_USER)
    for fact in test_facts:
        add_fact(TEST_USER, fact, source="eval_setup")

    results = []

    for idx, (_, row) in enumerate(distorted.iterrows()):
        text       = row["Patient Question"]
        true_label = normalize_label(row["Dominant Distortion"])

        print(f"\n  [{idx+1}/{len(distorted)}] {text[:55]}...")

        # ── WITHOUT memory ────────────────────────────────────────────────────
        analyst_no_mem = safe_call(run_analyst, text, df, k=5)
        if analyst_no_mem is None:
            continue

        final_no_mem, da_no_mem, loops_no_mem = safe_call(
            run_consensus_loop,
            user_input     = text,
            analyst_result = analyst_no_mem,
            df             = df,
            memory_facts   = [],       # ← empty memory
            max_loops      = 2,
        ) or (analyst_no_mem, {}, 0)

        label_no_mem   = normalize_label(final_no_mem.get("label", "No distortion"))
        conf_no_mem    = float(final_no_mem.get("confidence", 0.5))
        da_verdict_nm  = da_no_mem.get("verdict", "AGREE") if da_no_mem else "AGREE"

        # ── WITH memory ───────────────────────────────────────────────────────
        memory_facts   = search_facts(TEST_USER, text, n_results=5)

        analyst_mem    = safe_call(run_analyst, text, df, k=5)
        if analyst_mem is None:
            continue

        final_mem, da_mem, loops_mem = safe_call(
            run_consensus_loop,
            user_input     = text,
            analyst_result = analyst_mem,
            df             = df,
            memory_facts   = memory_facts,  # ← real memory facts
            max_loops      = 2,
        ) or (analyst_mem, {}, 0)

        label_mem      = normalize_label(final_mem.get("label", "No distortion"))
        conf_mem       = float(final_mem.get("confidence", 0.5))
        da_verdict_m   = da_mem.get("verdict", "AGREE") if da_mem else "AGREE"

        correct_no_mem = label_no_mem == true_label
        correct_mem    = label_mem    == true_label
        label_changed  = label_no_mem != label_mem
        conf_delta     = conf_mem - conf_no_mem

        print(f"    True          : {true_label}")
        print(f"    No memory     : {label_no_mem:<30} conf={conf_no_mem:.2f}  DA={da_verdict_nm}  {'✓' if correct_no_mem else '✗'}")
        print(f"    With memory   : {label_mem:<30} conf={conf_mem:.2f}  DA={da_verdict_m}  {'✓' if correct_mem else '✗'}")
        print(f"    Conf delta    : {conf_delta:+.2f}  Label changed: {'yes' if label_changed else 'no'}")
        if memory_facts:
            print(f"    Memory used   : {len(memory_facts)} facts")

        results.append({
            "text"            : text[:100],
            "true_label"      : true_label,
            "label_no_mem"    : label_no_mem,
            "label_with_mem"  : label_mem,
            "conf_no_mem"     : round(conf_no_mem, 3),
            "conf_with_mem"   : round(conf_mem, 3),
            "conf_delta"      : round(conf_delta, 3),
            "correct_no_mem"  : correct_no_mem,
            "correct_with_mem": correct_mem,
            "label_changed"   : label_changed,
            "da_no_mem"       : da_verdict_nm,
            "da_with_mem"     : da_verdict_m,
            "memory_facts_n"  : len(memory_facts),
        })

    # Cleanup test user
    clear_user_vault(TEST_USER)

    if not results:
        return {"error": "No samples evaluated"}

    n              = len(results)
    acc_no_mem     = sum(r["correct_no_mem"]   for r in results) / n
    acc_with_mem   = sum(r["correct_with_mem"] for r in results) / n
    acc_delta      = acc_with_mem - acc_no_mem
    avg_conf_delta = sum(r["conf_delta"]       for r in results) / n
    labels_changed = sum(1 for r in results if r["label_changed"])

    # Of cases where label changed, how often did it improve vs worsen?
    changed = [r for r in results if r["label_changed"]]
    improved = sum(1 for r in changed if r["correct_with_mem"] and not r["correct_no_mem"])
    worsened = sum(1 for r in changed if not r["correct_with_mem"] and r["correct_no_mem"])

    # DA challenge rate with vs without memory
    da_challenges_no_mem  = sum(1 for r in results if r["da_no_mem"]   == "CHALLENGE")
    da_challenges_with_mem= sum(1 for r in results if r["da_with_mem"] == "CHALLENGE")

    print(f"\n── Results ──")
    print(f"  Accuracy without memory : {acc_no_mem:.3f}  ({acc_no_mem*100:.1f}%)")
    print(f"  Accuracy with memory    : {acc_with_mem:.3f}  ({acc_with_mem*100:.1f}%)")
    print(f"  Accuracy delta          : {acc_delta:+.3f}  ({acc_delta*100:+.1f}%)")
    print(f"  Avg confidence delta    : {avg_conf_delta:+.3f}")
    print(f"  Labels changed          : {labels_changed}/{n}  ({labels_changed/n*100:.1f}%)")
    if changed:
        print(f"  Of changed labels:")
        print(f"    Improved by memory  : {improved}/{len(changed)}  ({improved/len(changed)*100:.1f}%)")
        print(f"    Worsened by memory  : {worsened}/{len(changed)}  ({worsened/len(changed)*100:.1f}%)")
    print(f"  DA challenges no mem    : {da_challenges_no_mem}/{n}")
    print(f"  DA challenges with mem  : {da_challenges_with_mem}/{n}")

    return {
        "n_samples"              : n,
        "accuracy_no_memory"     : round(acc_no_mem, 4),
        "accuracy_with_memory"   : round(acc_with_mem, 4),
        "accuracy_delta"         : round(acc_delta, 4),
        "avg_conf_delta"         : round(avg_conf_delta, 4),
        "labels_changed"         : labels_changed,
        "labels_changed_pct"     : round(labels_changed / n * 100, 1),
        "improved_by_memory"     : improved,
        "worsened_by_memory"     : worsened,
        "da_challenges_no_memory": da_challenges_no_mem,
        "da_challenges_with_memory": da_challenges_with_mem,
        "details"                : results,
    }


# ── Evaluation 4: DA Override Impact ─────────────────────────────────────────

def eval_da_override_impact(df: pd.DataFrame) -> dict:
    """
    Measures whether the Devil's Advocate actually improves classification.

    For each sample:
        - Record what the Analyst said (before DA)
        - Record what the final label is (after DA consensus loop)
        - Compare both against ground truth

    Metrics:
        - How often did DA challenge?
        - Of challenges, how often did it improve accuracy?
        - Of challenges, how often did it worsen accuracy?
        - Net improvement from DA
    """
    print_section("EVAL 4: Devil's Advocate Override Impact")

    test_samples = df[df["Dominant Distortion"] != "No distortion"].sample(
        min(DA_EVAL_N, len(df[df["Dominant Distortion"] != "No distortion"])),
        random_state=RANDOM_SEED + 1
    )

    print(f"Samples: {len(test_samples)}")

    results = []

    for idx, (_, row) in enumerate(test_samples.iterrows()):
        text       = row["Patient Question"]
        true_label = normalize_label(row["Dominant Distortion"])

        print(f"  [{idx+1}/{len(test_samples)}] {text[:55]}...")

        # Get initial analyst label
        analyst_result = safe_call(run_analyst, text, df, k=5)
        if analyst_result is None:
            continue

        initial_label = normalize_label(analyst_result.get("label", "No distortion"))
        initial_conf  = float(analyst_result.get("confidence", 0.5))

        # Run full consensus loop
        final_result, da_result, loops = safe_call(
            run_consensus_loop,
            user_input     = text,
            analyst_result = analyst_result,
            df             = df,
            memory_facts   = [],
            max_loops      = 2,
        ) or (analyst_result, {"verdict": "AGREE"}, 0)

        final_label   = normalize_label(final_result.get("label", "No distortion"))
        final_conf    = float(final_result.get("confidence", 0.5))
        da_verdict    = da_result.get("verdict", "AGREE") if da_result else "AGREE"
        da_suggested  = da_result.get("suggested_label", "") if da_result else ""
        was_challenged= da_verdict == "CHALLENGE"
        label_changed = initial_label != final_label

        initial_correct = initial_label == true_label
        final_correct   = final_label   == true_label

        if was_challenged:
            outcome = "improved" if final_correct and not initial_correct else \
                      "worsened" if not final_correct and initial_correct else \
                      "no_change"
        else:
            outcome = "not_challenged"

        print(f"    True     : {true_label}")
        print(f"    Initial  : {initial_label:<30} ({initial_conf:.2f}) {'✓' if initial_correct else '✗'}")
        print(f"    DA       : {da_verdict}")
        if was_challenged:
            print(f"    Suggested: {da_suggested}")
        print(f"    Final    : {final_label:<30} ({final_conf:.2f}) {'✓' if final_correct else '✗'}")
        print(f"    Outcome  : {outcome}")

        results.append({
            "text"           : text[:100],
            "true_label"     : true_label,
            "initial_label"  : initial_label,
            "final_label"    : final_label,
            "initial_conf"   : round(initial_conf, 3),
            "final_conf"     : round(final_conf, 3),
            "da_verdict"     : da_verdict,
            "da_suggested"   : da_suggested,
            "loops"          : loops,
            "was_challenged" : was_challenged,
            "label_changed"  : label_changed,
            "initial_correct": initial_correct,
            "final_correct"  : final_correct,
            "outcome"        : outcome,
        })

    if not results:
        return {"error": "No samples evaluated"}

    n              = len(results)
    challenged     = [r for r in results if r["was_challenged"]]
    not_challenged = [r for r in results if not r["was_challenged"]]

    acc_before_da  = sum(r["initial_correct"] for r in results) / n
    acc_after_da   = sum(r["final_correct"]   for r in results) / n
    challenge_rate = len(challenged) / n

    improved = [r for r in challenged if r["outcome"] == "improved"]
    worsened = [r for r in challenged if r["outcome"] == "worsened"]

    print(f"\n── Results ──")
    print(f"  Accuracy before DA     : {acc_before_da:.3f}  ({acc_before_da*100:.1f}%)")
    print(f"  Accuracy after DA      : {acc_after_da:.3f}  ({acc_after_da*100:.1f}%)")
    print(f"  Net improvement from DA: {acc_after_da - acc_before_da:+.3f}  ({(acc_after_da-acc_before_da)*100:+.1f}%)")
    print(f"  Challenge rate         : {challenge_rate:.3f}  ({challenge_rate*100:.1f}% of sessions)")
    if challenged:
        print(f"  Of {len(challenged)} challenges:")
        print(f"    Improved accuracy    : {len(improved)} ({len(improved)/len(challenged)*100:.1f}%)")
        print(f"    Worsened accuracy    : {len(worsened)} ({len(worsened)/len(challenged)*100:.1f}%)")
        print(f"    No change            : {len(challenged)-len(improved)-len(worsened)} ({(len(challenged)-len(improved)-len(worsened))/len(challenged)*100:.1f}%)")

    return {
        "n_samples"              : n,
        "accuracy_before_da"     : round(acc_before_da, 4),
        "accuracy_after_da"      : round(acc_after_da, 4),
        "da_net_improvement"     : round(acc_after_da - acc_before_da, 4),
        "challenge_rate"         : round(challenge_rate, 4),
        "n_challenged"           : len(challenged),
        "n_improved_by_da"       : len(improved),
        "n_worsened_by_da"       : len(worsened),
        "improvement_rate_of_challenges": round(len(improved)/len(challenged), 4) if challenged else 0,
        "details"                : results,
    }


# ── Evaluation 5: Memory Audit Effectiveness ──────────────────────────────────

def eval_memory_audit() -> dict:
    """
    Measures how well the Memory Architect detects distortions
    in the AI's own stored summaries.

    Injects known distorted and clean summaries, then checks:
        - True positive rate (distorted summaries correctly flagged)
        - False positive rate (clean summaries incorrectly flagged)
        - Precision, Recall, F1 for the audit itself
    """
    print_section("EVAL 5: Memory Audit Effectiveness")

    TEST_USER = "eval_audit_test_99"

    distorted_summaries = [
        "User is always anxious and never handles stress well.",
        "User is completely broken and will never improve.",
        "User is a total failure at relationships.",
        "User definitely hates social situations — this is permanent.",
        "User catastrophizes every single setback without exception.",
        "User will always struggle with overgeneralization — it's who they are.",
        "User never shows any resilience whatsoever.",
    ]

    clean_summaries = [
        "User showed anxiety in 2 of 5 sessions around work deadlines.",
        "User has been working on reframing negative self-talk this month.",
        "User responded well to Socratic questioning in this session.",
        "User mentioned feeling overwhelmed but identified one coping strategy.",
        "Session focused on examining evidence for and against a core belief.",
        "User made progress identifying triggers for emotional reasoning patterns.",
        "User discussed a recent setback and explored proportional responses.",
    ]

    print(f"Injecting {len(distorted_summaries)} distorted + {len(clean_summaries)} clean summaries...")
    clear_user_memory(TEST_USER)

    all_summaries = distorted_summaries + clean_summaries
    random.shuffle(all_summaries)

    for s in all_summaries:
        add_session_inference(TEST_USER, s, session_id="eval_audit")
        time.sleep(0.5)

    print("Running Memory Architect audit...")
    audit_results = audit_ai_memory(TEST_USER, n_memories=20)

    # Score each result
    tp = fp = fn = tn = 0
    detected     = []
    missed       = []
    false_pos    = []
    true_neg     = []

    for r in audit_results:
        content = r.get("memory_content", "")
        verdict = r.get("verdict", "CLEAN")

        is_actually_distorted = any(d[:40] in content for d in distorted_summaries)
        is_actually_clean     = any(c[:40] in content for c in clean_summaries)

        if verdict == "DISTORTED" and is_actually_distorted:
            tp += 1
            detected.append(content[:70])
        elif verdict == "DISTORTED" and is_actually_clean:
            fp += 1
            false_pos.append(content[:70])
        elif verdict == "CLEAN" and is_actually_distorted:
            fn += 1
            missed.append(content[:70])
        elif verdict == "CLEAN" and is_actually_clean:
            tn += 1
            true_neg.append(content[:70])

    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall       = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1           = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n── Results ──")
    print(f"  True positives  (caught distorted) : {tp}/{len(distorted_summaries)}  ({tp/len(distorted_summaries)*100:.1f}%)")
    print(f"  False negatives (missed distorted) : {fn}/{len(distorted_summaries)}")
    print(f"  False positives (flagged clean)    : {fp}/{len(clean_summaries)}  ({fp/len(clean_summaries)*100:.1f}%)")
    print(f"  True negatives  (kept clean)       : {tn}/{len(clean_summaries)}")
    print(f"  Precision  : {precision:.3f}")
    print(f"  Recall     : {recall:.3f}")
    print(f"  F1 score   : {f1:.3f}")
    print(f"  Specificity: {specificity:.3f}")

    if detected:
        print(f"\n  Correctly flagged:")
        for d in detected: print(f"    ✓ {d}...")
    if missed:
        print(f"\n  Missed:")
        for m in missed: print(f"    ✗ {m}...")
    if false_pos:
        print(f"\n  Falsely flagged:")
        for f in false_pos: print(f"    ! {f}...")

    clear_user_memory(TEST_USER)

    return {
        "n_distorted_injected": len(distorted_summaries),
        "n_clean_injected"    : len(clean_summaries),
        "true_positives"      : tp,
        "false_negatives"     : fn,
        "false_positives"     : fp,
        "true_negatives"      : tn,
        "precision"           : round(precision, 4),
        "recall"              : round(recall, 4),
        "f1_score"            : round(f1, 4),
        "specificity"         : round(specificity, 4),
        "detected"            : detected,
        "missed"              : missed,
        "false_positive_list" : false_pos,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, labels: list, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(14, 11))
        im      = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        thresh = cm_arr.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                       fontsize=9, color="white" if cm_arr[i, j] > thresh else "black")
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title("Cognitive Distortion Classification — Confusion Matrix", fontsize=13, pad=20)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib not installed — skipping plot (pip install matplotlib)")


def plot_calibration_curve(calib: dict, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bin_confs = calib.get("bin_confidences", [])
        bin_accs  = calib.get("bin_accuracies", [])

        if not bin_confs:
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
        ax.plot(bin_confs, bin_accs, "o-", color="#6B5CE7", linewidth=2,
                markersize=8, label="Model calibration")
        ax.fill_between(bin_confs, bin_accs, bin_confs,
                       alpha=0.15, color="#6B5CE7", label="Calibration gap")
        ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
        ax.set_ylabel("Fraction Correct", fontsize=12)
        ax.set_title(f"Confidence Calibration Curve  (ECE={calib['ece']:.3f})", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib not installed — skipping plot")


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary_report(results: dict, path: str):
    c  = results.get("classification", {})
    cb = results.get("calibration", {})
    m  = results.get("memory_comparison", {})
    da = results.get("da_override_impact", {})
    a  = results.get("memory_audit", {})

    lines = [
        "=" * 65,
        "  COGNITIVE GUARDRAIL AI — EVALUATION SUMMARY REPORT",
        f"  Generated: {results['timestamp']}",
        "=" * 65,
        "",
        "1. CLASSIFICATION ACCURACY",
        "-" * 40,
        f"   Samples evaluated : {c.get('n_samples', 'N/A')}",
        f"   Accuracy          : {c.get('accuracy', 0)*100:.1f}%",
        f"   Macro F1          : {c.get('macro_f1', 0):.3f}",
        f"   Macro Precision   : {c.get('macro_precision', 0):.3f}",
        f"   Macro Recall      : {c.get('macro_recall', 0):.3f}",
        f"   Cohen's Kappa     : {c.get('cohen_kappa', 0):.3f}  ({c.get('kappa_interp', '')})",
        "",
        "2. CONFIDENCE CALIBRATION",
        "-" * 40,
        f"   Avg confidence    : {cb.get('avg_confidence', 0):.3f}",
        f"   Avg accuracy      : {cb.get('avg_accuracy', 0):.3f}",
        f"   Overconfidence    : {cb.get('overconfidence', 0):+.3f}",
        f"   ECE (lower=better): {cb.get('ece', 0):.3f}",
        f"   Status            : {cb.get('calibration_status', 'N/A')}",
        "",
        "3. MEMORY vs NO-MEMORY COMPARISON",
        "-" * 40,
        f"   Samples           : {m.get('n_samples', 'N/A')}",
        f"   Accuracy no memory: {m.get('accuracy_no_memory', 0)*100:.1f}%",
        f"   Accuracy w/ memory: {m.get('accuracy_with_memory', 0)*100:.1f}%",
        f"   Delta             : {m.get('accuracy_delta', 0)*100:+.1f}%",
        f"   Avg conf delta    : {m.get('avg_conf_delta', 0):+.3f}",
        f"   Labels changed    : {m.get('labels_changed_pct', 0):.1f}% of sessions",
        f"   Improved by memory: {m.get('improved_by_memory', 'N/A')}",
        f"   Worsened by memory: {m.get('worsened_by_memory', 'N/A')}",
        "",
        "4. DEVIL'S ADVOCATE IMPACT",
        "-" * 40,
        f"   Samples           : {da.get('n_samples', 'N/A')}",
        f"   Accuracy before DA: {da.get('accuracy_before_da', 0)*100:.1f}%",
        f"   Accuracy after DA : {da.get('accuracy_after_da', 0)*100:.1f}%",
        f"   Net improvement   : {da.get('da_net_improvement', 0)*100:+.1f}%",
        f"   Challenge rate    : {da.get('challenge_rate', 0)*100:.1f}%",
        f"   Improved of challenged: {da.get('improvement_rate_of_challenges', 0)*100:.1f}%",
        "",
        "5. MEMORY AUDIT EFFECTIVENESS",
        "-" * 40,
        f"   Distorted injected: {a.get('n_distorted_injected', 'N/A')}",
        f"   Clean injected    : {a.get('n_clean_injected', 'N/A')}",
        f"   Precision         : {a.get('precision', 0):.3f}",
        f"   Recall            : {a.get('recall', 0):.3f}",
        f"   F1 score          : {a.get('f1_score', 0):.3f}",
        f"   Specificity       : {a.get('specificity', 0):.3f}",
        f"   True positives    : {a.get('true_positives', 'N/A')}/{a.get('n_distorted_injected', 'N/A')}",
        f"   False positives   : {a.get('false_positives', 'N/A')}/{a.get('n_clean_injected', 'N/A')}",
        "",
        "=" * 65,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {path}")
    print("\n" + "\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_section("COGNITIVE GUARDRAIL AI — FULL EVALUATION SUITE")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config  : {MAX_SAMPLES} classification samples, {MEMORY_COMPARE_N} memory compare, {DA_EVAL_N} DA eval")

    print("\nLoading dataset...")
    df = load_merged()
    print(f"Dataset: {len(df)} samples across {df['Dominant Distortion'].nunique()} labels")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples"      : MAX_SAMPLES,
            "memory_compare_n" : MEMORY_COMPARE_N,
            "da_eval_n"        : DA_EVAL_N,
            "test_size"        : TEST_SIZE,
        }
    }

    # Run all evaluations
    print("\n[1/5] Classification accuracy...")
    r1 = eval_classification(df)
    all_results["classification"] = r1

    print("\n[2/5] Confidence calibration...")
    r2 = eval_calibration(r1)
    all_results["calibration"] = r2

    print("\n[3/5] Memory vs No-Memory comparison...")
    r3 = eval_memory_vs_no_memory(df)
    all_results["memory_comparison"] = r3

    print("\n[4/5] Devil's Advocate override impact...")
    r4 = eval_da_override_impact(df)
    all_results["da_override_impact"] = r4

    print("\n[5/5] Memory audit effectiveness...")
    r5 = eval_memory_audit()
    all_results["memory_audit"] = r5

    # Save outputs
    print_section("SAVING OUTPUTS")

    results_path = ROOT_DIR / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    if r1.get("confusion_matrix"):
        plot_confusion_matrix(
            cm     = r1["confusion_matrix"],
            labels = r1["confusion_labels"],
            path   = str(ROOT_DIR / "eval_confusion_matrix.png")
        )

    if r2.get("bin_confidences"):
        plot_calibration_curve(
            calib = r2,
            path  = str(ROOT_DIR / "eval_calibration_curve.png")
        )

    write_summary_report(
        results = all_results,
        path    = str(ROOT_DIR / "eval_summary_report.txt")
    )

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
