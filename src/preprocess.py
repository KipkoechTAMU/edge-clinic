"""
preprocess.py
-------------
Cleans and standardizes raw train (Excel) and test (CSV) vignette data.

Usage:
    python preprocess.py \
        --train data/raw/train.xlsx \
        --test  data/raw/test.csv \
        --out   data/processed/

Outputs:
    data/processed/train_clean.csv
    data/processed/test_clean.csv
    data/processed/model_baselines_train.csv   # GPT4, LLAMA, GEMINI kept separately
    data/processed/model_baselines_test.csv
"""

import argparse
import re
import os
import pandas as pd


# ---------------------------------------------------------------------------
# Column config
# ---------------------------------------------------------------------------

KEEP_COLS = [
    "Master_Index",
    "County",
    "Health level",
    "Years of Experience",
    "Nursing Competency",
    "Clinical Panel",
    "Prompt",
    "Clinician",
]

BASELINE_COLS = [
    "Master_Index",
    "GPT4.0",
    "LLAMA",
    "GEMINI",
]

# Facility type normalization map
# Maps raw Excel values -> clean token used in model prefix
FACILITY_MAP = {
    "sub county hospitals and nursing homes": "sub_county_hospital",
    "national referral hospitals": "national_referral_hospital",
    "county referral hospitals": "county_referral_hospital",
    "primary care": "primary_care",
    "dispensaries and health centres": "dispensary_health_centre",
}

# Medical abbreviation expansion
# Built from observations in the dataset
ABBREV_MAP = {
    r"\bdka\b": "diabetic ketoacidosis",
    r"\bpud\b": "peptic ulcer disease",
    r"\bgcs\b": "glasgow coma scale",
    r"\bspo2\b": "oxygen saturation",
    r"\bhr\b": "heart rate",
    r"\brr\b": "respiratory rate",
    r"\bbp\b": "blood pressure",
    r"\btemp\b": "temperature",
    r"\biv\b": "intravenous",
    r"\bim\b": "intramuscular",
    r"\bivf\b": "intravenous fluids",
    r"\bnacl\b": "sodium chloride",
    r"\bcbc\b": "complete blood count",
    r"\buecs\b": "urea electrolytes creatinine",
    r"\blfts\b": "liver function tests",
    r"\babg\b": "arterial blood gas",
    r"\bvbg\b": "venous blood gas",
    r"\becg\b": "electrocardiogram",
    r"\bicu\b": "intensive care unit",
    r"\bhdu\b": "high dependency unit",
    r"\bnsaid\b": "non steroidal anti inflammatory drug",
    r"\bnsaids\b": "non steroidal anti inflammatory drugs",
    r"\bppi\b": "proton pump inhibitor",
    r"\bppis\b": "proton pump inhibitors",
    r"\btig\b": "tetanus immunoglobulin",
    r"\bdtp\b": "diphtheria tetanus pertussis",
    r"\btbsa\b": "total body surface area",
    r"\bco\b": "carbon monoxide",
    r"\bcohb\b": "carboxyhemoglobin",
    r"\bhbot\b": "hyperbaric oxygen therapy",
    r"\bdm\b": "diabetes mellitus",
    r"\buti\b": "urinary tract infection",
    r"\bbun\b": "blood urea nitrogen",
    r"\brbs\b": "random blood sugar",
    r"\bhba1c\b": "glycated haemoglobin",
    r"\bdre\b": "digital rectal examination",
    r"\bkub\b": "kidney ureter bladder",
}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def expand_abbreviations(text: str) -> str:
    """Expand known medical abbreviations (case-insensitive)."""
    for pattern, expansion in ABBREV_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Full text cleaning pipeline for Prompt and Clinician columns."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = expand_abbreviations(text)
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Normalize punctuation: collapse dashes, bullets, arrows
    text = re.sub(r"[•\-–—►▶]+", " ", text)
    # Collapse whitespace
    text = normalize_whitespace(text)
    return text


def build_prefix(facility: str, experience: int | float) -> str:
    """
    Build the structured prefix prepended to each prompt.
    Example: 'facility: national_referral_hospital | experience: 17 years | '
    """
    facility_clean = FACILITY_MAP.get(
        str(facility).strip().lower(), str(facility).strip().lower().replace(" ", "_")
    )
    try:
        exp_str = f"{int(experience)} years"
    except (ValueError, TypeError):
        exp_str = str(experience)
    return f"facility: {facility_clean} | experience: {exp_str} | "


def build_model_input(row: pd.Series) -> str:
    """Combine prefix + cleaned prompt into the final model input string."""
    prefix = build_prefix(row["Health level"], row["Years of Experience"])
    return prefix + row["prompt_clean"]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_train(path: str) -> pd.DataFrame:
    """Load train Excel file. Handles both .xlsx and tab-separated .xml exports."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".csv", ".tsv", ".txt", ".xml"):
        # The file shared looks tab-separated despite .xml extension
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return df


def load_test(path: str) -> pd.DataFrame:
    """Load test CSV file."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline on a dataframe.

    Returns:
        clean_df    — model-ready dataframe
        baseline_df — GPT4/LLAMA/GEMINI responses kept separately
    """
    print(f"\n[{split}] Raw shape: {df.shape}")

    # --- Separate baseline columns ---
    existing_baseline_cols = [c for c in BASELINE_COLS if c in df.columns]
    baseline_df = df[existing_baseline_cols].copy() if existing_baseline_cols else pd.DataFrame()

    # --- Select and validate keep columns ---
    existing_keep = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing expected columns: {missing}")
    df = df[existing_keep].copy()

    # --- Drop rows with missing Prompt or Clinician ---
    before = len(df)
    df = df.dropna(subset=["Prompt", "Clinician"])
    dropped = before - len(df)
    if dropped:
        print(f"  [WARN] Dropped {dropped} rows with missing Prompt or Clinician")

    # --- Clean text ---
    df["prompt_clean"] = df["Prompt"].apply(clean_text)
    df["clinician_clean"] = df["Clinician"].apply(clean_text)

    # --- Build model input (prefix + prompt) ---
    df["model_input"] = df.apply(build_model_input, axis=1)

    # --- Validate no empty targets ---
    empty_targets = (df["clinician_clean"].str.strip() == "").sum()
    if empty_targets:
        print(f"  [WARN] {empty_targets} rows have empty Clinician response after cleaning")

    # --- Final column selection ---
    clean_df = df[[
        "Master_Index",
        "County",
        "Health level",
        "Years of Experience",
        "Nursing Competency",
        "Clinical Panel",
        "model_input",       # input to T5
        "clinician_clean",   # target for T5
    ]].rename(columns={
        "model_input": "input_text",
        "clinician_clean": "target_text",
    })

    print(f"  [{split}] Clean shape: {clean_df.shape}")
    print(f"  [{split}] Sample input:\n    {clean_df['input_text'].iloc[0][:200]}")
    print(f"  [{split}] Sample target:\n    {clean_df['target_text'].iloc[0][:200]}")

    return clean_df, baseline_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess edge-clinic vignette data")
    parser.add_argument("--train", required=True, help="Path to train Excel file")
    parser.add_argument("--test",  required=True, help="Path to test CSV file")
    parser.add_argument("--out",   required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Train ---
    train_raw = load_train(args.train)
    train_clean, train_baselines = preprocess(train_raw, split="TRAIN")
    train_clean.to_csv(os.path.join(args.out, "train_clean.csv"), index=False)
    if not train_baselines.empty:
        train_baselines.to_csv(os.path.join(args.out, "model_baselines_train.csv"), index=False)

    # --- Test ---
    test_raw = load_test(args.test)
    test_clean, test_baselines = preprocess(test_raw, split="TEST")
    test_clean.to_csv(os.path.join(args.out, "test_clean.csv"), index=False)
    if not test_baselines.empty:
        test_baselines.to_csv(os.path.join(args.out, "model_baselines_test.csv"), index=False)

    print("\n[DONE] Processed files written to:", args.out)


if __name__ == "__main__":
    main()