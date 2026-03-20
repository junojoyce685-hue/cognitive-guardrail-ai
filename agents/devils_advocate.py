"""
devils_advocate.py
------------------
The second agent in the pipeline.

Takes the Analyst's output and challenges it by:
1. Pulling relevant user facts from ChromaDB (fact vault)
2. Asking Groq to argue against or validate the Analyst's classification
3. Returning AGREE or CHALLENGE with full reasoning

If CHALLENGE is returned, the pipeline loops back to the Analyst
with the counter-argument (max 2 loops).

Returns a structured dict with:
    - verdict         : "AGREE" or "CHALLENGE"
    - reasoning       : why it agrees or challenges
    - counter_argument: specific pushback (empty if AGREE)
    - suggested_label : alternative label if challenging (empty if AGREE)
"""

import json
import re
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from config import GROQ_API_KEY

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000

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

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Devil's Advocate agent working inside a cognitive distortion detection system.

Your job is NOT to be harsh — it is to ensure the Analyst's classification is accurate and fair to the patient.

You will receive:
1. The patient's original statement
2. The Analyst's classification (label, explanation, confidence)
3. Relevant facts about this patient from memory (if any)

YOUR TASK:
- Ask yourself: "Is the Analyst correct, or did they mislabel a valid emotion as a distortion?"
- Ask yourself: "Does the patient's history (from memory facts) support or contradict this label?"
- Ask yourself: "Could this be a different distortion type instead?"

WHEN TO AGREE:
- The label is clearly supported by the statement
- The confidence is high (above 0.75)
- No contradicting facts from memory

WHEN TO CHALLENGE:
- The emotion expressed might be valid and proportional, not distorted
- A different distortion label fits better
- Memory facts suggest this is out of character for this patient
- Confidence is low (below 0.55) and the reasoning feels weak

IMPORTANT RULES:
- Be fair and protect the patient from being mislabeled
- If you challenge, always suggest an alternative label from the valid taxonomy
- Never be dismissive of the patient's emotions
- Respond ONLY with a valid JSON object. No extra text, no markdown, no backticks.

VALID LABELS YOU CAN SUGGEST:
All-or-nothing thinking, Overgeneralization, Mental filter, Should statements,
Labeling, Personalization, Magnification, Emotional Reasoning, Mind Reading,
Fortune-telling, No distortion

OUTPUT FORMAT (strict JSON):
{
    "verdict": "<AGREE or CHALLENGE>",
    "reasoning": "<explanation of your decision>",
    "counter_argument": "<specific pushback to send back to Analyst, empty string if AGREE>",
    "suggested_label": "<alternative label if challenging, empty string if AGREE>"
}"""


# ── Devil's Advocate Function ─────────────────────────────────────────────────

def run_devils_advocate(
    user_input: str,
    analyst_result: dict,
    memory_facts: list = None,
) -> dict:
    """
    Challenges or validates the Analyst's distortion classification.

    Args:
        user_input      : original patient statement
        analyst_result  : output dict from run_analyst()
        memory_facts    : list of relevant fact strings from ChromaDB (can be empty)

    Returns:
        dict with keys: verdict, reasoning, counter_argument, suggested_label
    """
    memory_facts = memory_facts or []

    # ── Build memory facts block ──────────────────────────────────────────────
    if memory_facts:
        facts_block = "Relevant facts about this patient from memory:\n"
        for i, fact in enumerate(memory_facts, 1):
            facts_block += f"  {i}. {fact}\n"
    else:
        facts_block = "No prior memory facts available for this patient.\n"

    # ── Build user message ────────────────────────────────────────────────────
    user_message = f"""Here is the case to evaluate:

PATIENT STATEMENT:
\"{user_input}\"

ANALYST CLASSIFICATION:
- Label         : {analyst_result.get('label')}
- Secondary     : {analyst_result.get('secondary_label', 'None')}
- Distorted part: {analyst_result.get('distorted_part', 'None')}
- Explanation   : {analyst_result.get('explanation')}
- Reality check : {analyst_result.get('reality_check')}
- Confidence    : {analyst_result.get('confidence')}

{facts_block}

Should you AGREE with the Analyst or CHALLENGE this classification?
Respond with a JSON object only."""

    # ── Call Groq API ─────────────────────────────────────────────────────────
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
    )

    raw_text = response.choices[0].message.content.strip()

    # ── Parse JSON response ───────────────────────────────────────────────────
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Devil's Advocate: Could not parse JSON:\n{raw_text}")

    # ── Validate verdict ──────────────────────────────────────────────────────
    if result.get("verdict") not in ("AGREE", "CHALLENGE"):
        print(f"[devils_advocate] WARNING: Unknown verdict '{result.get('verdict')}' — defaulting to AGREE")
        result["verdict"] = "AGREE"

    # ── Validate suggested label if challenging ───────────────────────────────
    if result.get("verdict") == "CHALLENGE":
        suggested = result.get("suggested_label", "")
        if suggested and suggested not in VALID_LABELS:
            print(f"[devils_advocate] WARNING: Invalid suggested label '{suggested}' — clearing it")
            result["suggested_label"] = ""

    # ── Ensure all keys exist ─────────────────────────────────────────────────
    result.setdefault("reasoning",       "")
    result.setdefault("counter_argument","")
    result.setdefault("suggested_label", "")

    print(f"[devils_advocate] Verdict: {result['verdict']}")
    if result["verdict"] == "CHALLENGE":
        print(f"[devils_advocate] Suggested label: {result['suggested_label']}")
        print(f"[devils_advocate] Counter: {result['counter_argument'][:80]}...")

    return result


# ── Consensus Loop (used by pipeline.py) ──────────────────────────────────────

def run_consensus_loop(
    user_input: str,
    analyst_result: dict,
    df,
    memory_facts: list = None,
    max_loops: int = 2,
) -> tuple[dict, dict, int]:
    """
    Runs the full AGREE/CHALLENGE loop between Analyst and Devil's Advocate.

    Args:
        user_input      : original patient statement
        analyst_result  : initial output from run_analyst()
        df              : merged DataFrame (needed to re-run analyst)
        memory_facts    : facts from ChromaDB
        max_loops       : maximum number of challenge loops (default 2)

    Returns:
        tuple of:
            - final analyst_result  (dict)
            - final devils_result   (dict)
            - loop_count            (int) — how many loops were needed
    """
    from agents.analyst import run_analyst, SYSTEM_PROMPT as BASE_ANALYST_PROMPT

    loop_count    = 0
    devils_result = None

    while loop_count < max_loops:
        # Run Devil's Advocate
        devils_result = run_devils_advocate(
            user_input     = user_input,
            analyst_result = analyst_result,
            memory_facts   = memory_facts,
        )

        # If AGREE → done
        if devils_result["verdict"] == "AGREE":
            print(f"[consensus] AGREED after {loop_count + 1} round(s)")
            break

        # If CHALLENGE → re-run Analyst with counter-argument injected
        loop_count += 1
        print(f"[consensus] CHALLENGE round {loop_count} — re-running Analyst...")

        counter    = devils_result["counter_argument"]
        suggestion = devils_result["suggested_label"]

        hint = f"""
IMPORTANT NOTE FROM REVIEWER:
A challenge was raised about your previous classification.
Counter-argument: {counter}
Please reconsider — could this be '{suggestion}' instead?
Re-evaluate carefully before responding.
"""
        patched_prompt = BASE_ANALYST_PROMPT + hint

        analyst_result = run_analyst(
            user_input      = user_input,
            df              = df,
            k               = 5,
            override_prompt = patched_prompt,
        )

    if loop_count == max_loops and devils_result["verdict"] == "CHALLENGE":
        print(f"[consensus] Max loops reached — keeping last Analyst result")

    return analyst_result, devils_result, loop_count


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from training.csv_loader import load_merged
    from agents.analyst import run_analyst

    df = load_merged()

    test_cases = [
        {
            "input" : "I always ruin everything, nobody will ever love me.",
            "facts" : ["Patient has a supportive partner", "Patient mentioned close friends last session"],
        },
        {
            "input" : "I feel like my boss hates me, he barely spoke to me today.",
            "facts" : [],
        },
        {
            "input" : "I had a hard week but I managed to get through it.",
            "facts" : [],
        },
    ]

    for case in test_cases:
        print(f"\n{'─'*60}")
        print(f"Input: {case['input']}")

        analyst_result = run_analyst(case["input"], df)
        print(f"Analyst label: {analyst_result['label']} ({analyst_result['confidence']})")

        final_analyst, final_devils, loops = run_consensus_loop(
            user_input     = case["input"],
            analyst_result = analyst_result,
            df             = df,
            memory_facts   = case["facts"],
        )

        print(f"Final label  : {final_analyst['label']}")
        print(f"Verdict      : {final_devils['verdict']}")
        print(f"Loops needed : {loops}")
        print(f"DA Reasoning : {final_devils['reasoning'][:100]}...")