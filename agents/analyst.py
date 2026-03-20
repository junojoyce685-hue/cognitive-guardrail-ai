"""
analyst.py
----------
The first agent in the pipeline.

Takes raw user input and classifies it into a cognitive distortion
using Groq API + few-shot examples retrieved from the annotated CSV.

Returns a structured dict with:
    - label           : dominant distortion label
    - secondary_label : secondary distortion (if any)
    - distorted_part  : the specific phrase that contains the distortion
    - explanation     : why this is a distortion
    - reality_check   : a gentle reframing question
    - confidence      : float between 0 and 1
"""

import json
import re
import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from config import GROQ_API_KEY
from training.few_shot_builder import (
    build_analyst_examples,
    format_analyst_prompt,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL            = "llama-3.3-70b-versatile"
MAX_TOKENS       = 1000
DISTORTIONS_PATH = ROOT_DIR / "data" / "distortions.json"

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

# ── Load distortion definitions once ─────────────────────────────────────────
with open(DISTORTIONS_PATH, "r", encoding="utf-8") as f:
    DISTORTIONS_DATA = json.load(f)

DISTORTION_DEFINITIONS = "\n".join([
    f"- {d['label']}: {d['definition']}"
    for d in DISTORTIONS_DATA["distortions"]
])

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a clinical psychologist specializing in Cognitive Behavioral Therapy (CBT).
Your task is to analyze a patient's statement and identify the presence of cognitive distortions.

You have been trained on the following cognitive distortion taxonomy:
{DISTORTION_DEFINITIONS}

INSTRUCTIONS:
1. Read the patient's statement carefully.
2. Use the annotated examples provided to guide your classification.
3. Identify the most dominant cognitive distortion present, if any.
4. If a secondary distortion is also clearly present, note it.
5. Extract the specific phrase or sentence that contains the distortion.
6. Provide a brief, empathetic explanation of why it is a distortion.
7. Provide a gentle reality-check question to help the patient reflect.
8. Assign a confidence score between 0.0 and 1.0.

IMPORTANT RULES:
- If no distortion is detected, set label to "No distortion" and confidence accordingly.
- Never be harsh or judgmental in explanations.
- The reality_check must be a question, not a statement.
- Respond ONLY with a valid JSON object. No extra text, no markdown, no backticks.

OUTPUT FORMAT (strict JSON):
{{
    "label": "<dominant distortion label or No distortion>",
    "secondary_label": "<secondary label or empty string>",
    "distorted_part": "<exact phrase from patient text or empty string>",
    "explanation": "<brief empathetic explanation>",
    "reality_check": "<a gentle reflective question>",
    "confidence": <float between 0.0 and 1.0>
}}"""


# ── Analyst Function ──────────────────────────────────────────────────────────

def run_analyst(
    user_input: str,
    df,
    k: int = 5,
    override_prompt: str = None,
) -> dict:
    """
    Classifies a patient's statement for cognitive distortions.

    Args:
        user_input      : raw text from the user/patient
        df              : merged DataFrame from csv_loader.load_merged()
        k               : number of few-shot examples to retrieve
        override_prompt : optional patched system prompt from meta reviewer

    Returns:
        dict with keys: label, secondary_label, distorted_part,
                        explanation, reality_check, confidence
    """
    # Step 1: Retrieve few-shot examples
    examples       = build_analyst_examples(user_input, df, k=k, balanced=True)
    few_shot_block = format_analyst_prompt(examples)

    # Step 2: Build user message
    user_message = f"""{few_shot_block}
Now analyze the following new patient statement:

Patient said: \"{user_input}\"

Respond with a JSON object only."""

    # Step 3: Call Groq API
    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or GROQ_API_KEY)
    system   = override_prompt if override_prompt else SYSTEM_PROMPT

    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ]
    )

    raw_text = response.choices[0].message.content.strip()

    # Step 4: Parse JSON response
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Analyst: Could not parse JSON from response:\n{raw_text}")

    # Step 5: Validate label
    if result.get("label") not in VALID_LABELS:
        print(f"[analyst] WARNING: Unknown label '{result.get('label')}' — defaulting to 'No distortion'")
        result["label"] = "No distortion"

    # Step 6: Ensure all keys exist with defaults
    result.setdefault("secondary_label", "")
    result.setdefault("distorted_part",  "")
    result.setdefault("explanation",     "")
    result.setdefault("reality_check",   "")
    result.setdefault("confidence",      0.5)

    print(f"[analyst] Label: {result['label']} | Confidence: {result['confidence']}")
    return result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from training.csv_loader import load_merged

    df = load_merged()

    test_inputs = [
        "I always ruin everything I touch, nobody will ever love me.",
        "I feel like such a failure because I didn't finish my assignment.",
        "My boss didn't say good morning today, he must be planning to fire me.",
        "I had a pretty good day today, got some work done and felt okay.",
    ]

    for text in test_inputs:
        print(f"\n{'─'*60}")
        print(f"Input     : {text}")
        result = run_analyst(text, df)
        print(f"Label     : {result['label']}")
        print(f"Secondary : {result['secondary_label']}")
        print(f"Distorted : {result['distorted_part']}")
        print(f"Explain   : {result['explanation']}")
        print(f"Reality   : {result['reality_check']}")
        print(f"Confidence: {result['confidence']}")