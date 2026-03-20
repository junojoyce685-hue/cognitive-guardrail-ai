"""
responder.py
------------
The third agent in the pipeline.

Takes the final agreed-upon Analyst output and generates a
therapist-style response to the user using:
    - Few-shot examples from therapist_responses.csv
    - The confirmed distortion label and explanation
    - Relevant session inferences from local JSON store (if any)

Returns a structured dict with:
    - response        : the therapist-style reply to show the user
    - technique       : the CBT technique used in the response
    - follow_up       : a gentle follow-up question to keep dialogue going
    - flagged         : True if the input suggests crisis/self-harm risk
    - flag_reason     : reason for flagging (empty if not flagged)
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
from training.few_shot_builder import (
    build_responder_examples,
    format_responder_prompt,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1500

# Crisis keywords — if detected, response is flagged for human review
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "don't want to live",
    "self harm", "self-harm", "hurt myself", "cutting", "overdose",
    "no reason to live", "better off dead", "want to die",
]

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a warm, empathetic AI assistant trained in Cognitive Behavioral Therapy (CBT) techniques.

Your role is to respond to a patient's statement in the style of a supportive therapist.
You have been given:
1. The patient's original statement
2. The identified cognitive distortion (if any)
3. Examples of how real therapists responded to similar statements

YOUR RESPONSE STYLE:
- Always validate the patient's feelings first before addressing the distortion
- Never be dismissive, preachy, or lecture the patient
- Use Socratic questioning — guide them to their own insights rather than telling them what to think
- Keep the response warm, conversational, and human — not clinical or robotic
- Be concise — 3 to 5 sentences for the main response
- End with one gentle follow-up question to keep the dialogue open

CBT TECHNIQUES YOU CAN USE:
- Socratic questioning     : ask questions that gently challenge assumptions
- Thought reframing        : offer an alternative perspective
- Evidence examination     : "What evidence supports or contradicts this thought?"
- Decatastrophizing        : gently reduce magnified fears
- Behavioral activation    : suggest a small concrete action
- Self-compassion nudge    : encourage the patient to treat themselves as they would a friend

SAFETY RULE:
If the patient's statement contains any indication of self-harm, suicidal ideation,
or immediate crisis — set flagged to true and flag_reason accordingly.
Do not attempt to handle crisis situations — just flag them clearly.

IMPORTANT:
- Respond ONLY with a valid JSON object. No extra text, no markdown, no backticks.

OUTPUT FORMAT (strict JSON):
{
    "response"   : "<warm therapist-style response to the patient>",
    "technique"  : "<CBT technique used>",
    "follow_up"  : "<one gentle follow-up question>",
    "flagged"    : <true or false>,
    "flag_reason": "<reason if flagged, empty string if not>"
}"""


# ── Crisis Check ──────────────────────────────────────────────────────────────

def _check_for_crisis(text: str) -> tuple[bool, str]:
    """
    Quick keyword scan for crisis signals before calling the API.
    Returns (is_crisis, reason).
    """
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True, f"Crisis keyword detected: '{keyword}'"
    return False, ""


# ── Responder Function ────────────────────────────────────────────────────────

def run_responder(
    user_input: str,
    analyst_result: dict,
    df,
    recent_inferences: list = None,
    k: int = 3,
) -> dict:
    """
    Generates a therapist-style response to the patient's statement.

    Args:
        user_input         : original patient statement
        analyst_result     : final output from run_analyst() after consensus
        df                 : merged DataFrame from csv_loader.load_merged()
        recent_inferences  : list of recent inference strings (optional)
        k                  : number of few-shot examples to retrieve

    Returns:
        dict with keys: response, technique, follow_up, flagged, flag_reason
    """
    recent_inferences = recent_inferences or []

    # ── Step 1: Crisis check (fast, no API call needed) ───────────────────────
    is_crisis, crisis_reason = _check_for_crisis(user_input)
    if is_crisis:
        print(f"[responder] CRISIS FLAG: {crisis_reason}")
        return {
            "response"   : "I hear that you're going through something very difficult right now. "
                           "What you're feeling matters, and you deserve real support. "
                           "Please reach out to a crisis helpline or a trusted person in your life — "
                           "you don't have to face this alone.",
            "technique"  : "Crisis referral",
            "follow_up"  : "Is there someone safe you can reach out to right now?",
            "flagged"    : True,
            "flag_reason": crisis_reason,
        }

    # ── Step 2: Retrieve few-shot therapist examples ──────────────────────────
    examples       = build_responder_examples(user_input, df, k=k, balanced=True)
    few_shot_block = format_responder_prompt(examples)

    # ── Step 3: Build context from analyst result ─────────────────────────────
    if analyst_result.get("label") != "No distortion":
        distortion_context = f"""The following cognitive distortion has been identified in this statement:
- Distortion type : {analyst_result.get('label')}
- Distorted part  : {analyst_result.get('distorted_part', '')}
- Explanation     : {analyst_result.get('explanation', '')}
- Reality check   : {analyst_result.get('reality_check', '')}""".strip()
    else:
        distortion_context = "No cognitive distortion was detected in this statement."

    # ── Step 4: Build session history context ─────────────────────────────────
    history_context = ""
    if recent_inferences:
        history_context = "Relevant observations from this patient's history:\n"
        for i, inf in enumerate(recent_inferences[:3], 1):
            history_context += f"  {i}. {inf}\n"

    # ── Step 5: Build full user message ───────────────────────────────────────
    user_message = f"""{few_shot_block}

DISTORTION ANALYSIS:
{distortion_context}

{history_context}

Now respond to the following patient statement:
Patient said: \"{user_input}\"

Respond with a JSON object only."""

    # ── Step 6: Call Groq API ─────────────────────────────────────────────────
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

    # ── Step 7: Parse JSON response ───────────────────────────────────────────
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Responder: Could not parse JSON:\n{raw_text}")

    # ── Step 8: Ensure all keys exist ─────────────────────────────────────────
    result.setdefault("response",    "I hear you. Can you tell me more about what you're experiencing?")
    result.setdefault("technique",   "")
    result.setdefault("follow_up",   "")
    result.setdefault("flagged",     False)
    result.setdefault("flag_reason", "")

    # ── Step 9: Double check flagging even if model missed it ─────────────────
    if not result["flagged"] and is_crisis:
        result["flagged"]     = True
        result["flag_reason"] = crisis_reason

    print(f"[responder] Technique: {result['technique']} | Flagged: {result['flagged']}")
    return result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from training.csv_loader import load_merged
    from agents.analyst import run_analyst
    from agents.devils_advocate import run_consensus_loop

    df = load_merged()

    test_cases = [
        {
            "input"      : "I always ruin every relationship I have. Nobody could ever truly love me.",
            "inferences" : ["User has mentioned a caring partner in previous sessions."],
        },
        {
            "input"      : "My boss ignored my email. He definitely hates me and is going to fire me.",
            "inferences" : [],
        },
        {
            "input"      : "I had a hard week but I got through it. Feeling a bit better today.",
            "inferences" : [],
        },
    ]

    for case in test_cases:
        print(f"\n{'─'*60}")
        print(f"Input: {case['input']}")

        analyst_result           = run_analyst(case["input"], df)
        final_analyst, _, loops  = run_consensus_loop(
            user_input     = case["input"],
            analyst_result = analyst_result,
            df             = df,
        )

        result = run_responder(
            user_input        = case["input"],
            analyst_result    = final_analyst,
            df                = df,
            recent_inferences = case["inferences"],
        )

        print(f"Label     : {final_analyst['label']}")
        print(f"Technique : {result['technique']}")
        print(f"Response  : {result['response']}")
        print(f"Follow-up : {result['follow_up']}")
        print(f"Flagged   : {result['flagged']}")
        if result["flagged"]:
            print(f"Flag reason: {result['flag_reason']}")