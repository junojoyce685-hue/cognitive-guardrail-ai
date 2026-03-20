# cognitive-guardrail-ai
analyzes user thoughts, detects biased thinking patterns, challenges them, and generates improved responses while learning from past mistakes

-> Distorted thinking often lacks structured feedback, and standard chatbots offer unvalidated, single-pass responses.

# Note
For research and educational use only; not intended as a replacement for professional therapy.

# Key Features:
Multi-stage reasoning pipeline instead of single-pass generation
Adversarial validation via a Devil’s Advocate module
Response auditing for empathy, safety, and accuracy
Feedback-driven meta-learning loop
Memory filtering for high-confidence knowledge retention

User Input
   ↓
Analyst → Devil’s Advocate → Responder
   ↓
Response Auditor (async)
   ↓
Memory Architect
   ↓
Meta-Cognitive Reviewer

# Modules:
Analyst: Classifies distortions using embeddings and few-shot data
Devil’s Advocate: Challenges predictions to reduce errors
Responder: Generates context-aware responses
Auditor: Evaluates response quality
Memory Architect: Stores validated inferences
Meta Reviewer: Improves system behavior over time

# Example

Input:
“I always mess things up and nothing ever works out for me.”

Output:
Distortion: Overgeneralization (0.78)
Response: Encourages identifying exceptions to challenge the pattern

# Technical Approach

Sentence embeddings + cosine similarity for classification
Few-shot learning with curated datasets
Retrieval-Augmented Generation using ChromaDB
Feedback-based prompt refinement
Asynchronous response evaluation

# Limitations

Partial reliance on LLM-based evaluation
Dependent on dataset quality and coverage
Not a substitute for professional mental health support

# Tech Stack
Python, Streamlit, Sentence Transformers, ChromaDB, LLM APIs, CSV datasets
