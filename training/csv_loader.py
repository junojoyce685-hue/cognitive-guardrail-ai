"""
csv_loader.py
-------------
Loads, validates, and merges the two CSV datasets:
- Annotated_data.csv   → patient questions + distortion labels
- Therapist_responses.csv → therapist answers

Joined on Id_Number to produce a unified DataFrame.
"""

import pandas as pd
import json
import os
from pathlib import Path


# ── Paths (adjust if your folder structure differs) ──────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent  # project root
DATA_DIR         = BASE_DIR / "data"

ANNOTATED_PATH   = DATA_DIR / "Annotated_data.csv"
THERAPIST_PATH   = DATA_DIR / "Therapist_responses.csv"
DISTORTIONS_PATH = DATA_DIR / "distortions.json"


# ── Expected columns ──────────────────────────────────────────────────────────
ANNOTATED_REQUIRED = [
    "Id_Number",
    "Patient Question",
    "Distorted part",
    "Dominant Distortion",
]

THERAPIST_REQUIRED = [
    "Id_Number",
    "Question",
    "Answer",
]

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


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_annotated(path: Path = ANNOTATED_PATH) -> pd.DataFrame:
    """
    Load and validate Annotated_data.csv.
    Returns a clean DataFrame with standardized column names.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotated data not found at: {path}")

    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Validate required columns
    missing = [c for c in ANNOTATED_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Annotated CSV missing columns: {missing}")

    # Normalize text fields
    df["Patient Question"]    = df["Patient Question"].astype(str).str.strip()
    df["Distorted part"]      = df["Distorted part"].fillna("").astype(str).str.strip()
    df["Dominant Distortion"] = df["Dominant Distortion"].fillna("No distortion").astype(str).str.strip()

    # Optional secondary distortion column
    if "Secondary Distortion (Optional)" in df.columns:
        df["Secondary Distortion (Optional)"] = (
            df["Secondary Distortion (Optional)"].fillna("").astype(str).str.strip()
        )
    else:
        df["Secondary Distortion (Optional)"] = ""

    # Validate labels
    invalid = df[~df["Dominant Distortion"].isin(VALID_LABELS)]["Dominant Distortion"].unique()
    if len(invalid) > 0:
        print(f"[WARNING] Unknown distortion labels found: {invalid}")

    print(f"[csv_loader] Annotated data loaded: {len(df)} rows")
    return df


def load_therapist(path: Path = THERAPIST_PATH) -> pd.DataFrame:
    """
    Load and validate Therapist_responses.csv.
    Returns a clean DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Therapist responses not found at: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in THERAPIST_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Therapist CSV missing columns: {missing}")

    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"]   = df["Answer"].astype(str).str.strip()

    print(f"[csv_loader] Therapist responses loaded: {len(df)} rows")
    return df


def load_distortions(path: Path = DISTORTIONS_PATH) -> dict:
    """
    Load distortions.json reference file.
    Returns the full dict with labels, definitions, examples.
    """
    if not path.exists():
        raise FileNotFoundError(f"distortions.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[csv_loader] Distortions loaded: {len(data['distortions'])} labels")
    return data


# ── Merger ────────────────────────────────────────────────────────────────────

def load_merged() -> pd.DataFrame:
    """
    Joins Annotated_data.csv and Therapist_responses.csv on Id_Number.

    Returns a unified DataFrame with columns:
        Id_Number | Patient Question | Distorted part |
        Dominant Distortion | Secondary Distortion (Optional) |
        Therapist Answer
    """
    annotated  = load_annotated()
    therapist  = load_therapist()

    # Rename therapist Answer to avoid confusion
    therapist = therapist.rename(columns={"Answer": "Therapist Answer"})

    # Drop redundant Question column from therapist (already in annotated)
    therapist = therapist[["Id_Number", "Therapist Answer"]]

    merged = pd.merge(annotated, therapist, on="Id_Number", how="inner")

    print(f"[csv_loader] Merged dataset: {len(merged)} rows")
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_by_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Filter merged DataFrame by a specific dominant distortion label."""
    return df[df["Dominant Distortion"] == label].reset_index(drop=True)


def get_label_counts(df: pd.DataFrame) -> pd.Series:
    """Return count of each distortion label in the dataset."""
    return df["Dominant Distortion"].value_counts()


def get_row_by_id(df: pd.DataFrame, Id_Number: int) -> pd.Series:
    """Fetch a single row by Id_Number."""
    row = df[df["Id_Number"] == Id_Number]
    if row.empty:
        raise ValueError(f"Id_Number {Id_Number} not found in dataset.")
    return row.iloc[0]


def get_distortion_definition(distortions: dict, label: str) -> dict:
    """Look up a distortion label in distortions.json and return its full entry."""
    for d in distortions["distortions"]:
        if d["label"].lower() == label.lower():
            return d
    raise ValueError(f"Label '{label}' not found in distortions.json")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    merged      = load_merged()
    distortions = load_distortions()

    print("\n── Label Distribution ──")
    print(get_label_counts(merged).to_string())

    print("\n── Sample Row ──")
    sample = merged.iloc[0]
    print(f"  ID            : {sample['Id_Number']}")
    print(f"  Question      : {sample['Patient Question'][:80]}...")
    print(f"  Distortion    : {sample['Dominant Distortion']}")
    print(f"  Distorted part: {sample['Distorted part'][:80]}...")
    print(f"  Therapist Ans : {sample['Therapist Answer'][:80]}...")

    print("\n── Distortion Lookup ──")
    entry = get_distortion_definition(distortions, "Overgeneralization")