"""
tokenize.py
-----------
Tokenizes cleaned vignette data using T5's tokenizer and saves
HuggingFace Dataset objects to disk for use in fine-tuning.

Usage:
    python src/data/tokenize.py \
        --train data/processed/train_clean.csv \
        --test  data/processed/test_clean.csv \
        --out   data/processed/tokenized/

Outputs:
    data/processed/tokenized/train/   # HuggingFace Dataset
    data/processed/tokenized/test/    # HuggingFace Dataset
    data/processed/tokenized/stats.json  # token length stats for debugging
"""

import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME       = "t5-small"
MAX_INPUT_LEN    = 512   # T5-small hard limit
MAX_TARGET_LEN   = 256   # increased from 128 — mean target length is 161 tokens


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_batch(batch, tokenizer):
    """
    Tokenize a batch of input_text and target_text.
    - input_text  -> model inputs (encoder)
    - target_text -> labels (decoder); empty string for test set
    """
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )

    # For test set target_text is empty — still tokenize to keep schema consistent
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )

    # Replace padding token id in labels with -100 so loss ignores padding
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, tokenizer) -> dict:
    """
    Compute token length statistics to verify truncation isn't too aggressive.
    Prints a warning if more than 10% of inputs are being truncated.
    """
    input_lengths  = [len(tokenizer(t)["input_ids"]) for t in df["input_text"]]
    target_lengths = [len(tokenizer(t)["input_ids"]) for t in df["target_text"] if t]

    truncated_inputs  = sum(1 for l in input_lengths  if l > MAX_INPUT_LEN)
    truncated_targets = sum(1 for l in target_lengths if l > MAX_TARGET_LEN)

    stats = {
        "input_length": {
            "min":  min(input_lengths),
            "max":  max(input_lengths),
            "mean": round(sum(input_lengths) / len(input_lengths), 1),
            "truncated": truncated_inputs,
            "truncated_pct": round(100 * truncated_inputs / len(input_lengths), 1),
        },
        "target_length": {
            "min":  min(target_lengths) if target_lengths else 0,
            "max":  max(target_lengths) if target_lengths else 0,
            "mean": round(sum(target_lengths) / len(target_lengths), 1) if target_lengths else 0,
            "truncated": truncated_targets,
            "truncated_pct": round(100 * truncated_targets / len(target_lengths), 1) if target_lengths else 0,
        },
    }

    if stats["input_length"]["truncated_pct"] > 10:
        print(f"  [WARN] {stats['input_length']['truncated_pct']}% of inputs exceed "
              f"{MAX_INPUT_LEN} tokens and will be truncated. "
              f"Consider increasing MAX_INPUT_LEN or shortening the prefix.")

    if stats["target_length"]["truncated_pct"] > 10:
        print(f"  [WARN] {stats['target_length']['truncated_pct']}% of targets exceed "
              f"{MAX_TARGET_LEN} tokens and will be truncated. "
              f"Consider increasing MAX_TARGET_LEN.")

    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_split(df: pd.DataFrame, tokenizer, split: str, out_dir: str) -> dict:
    """Tokenize a split and save as HuggingFace Dataset."""
    print(f"\n[{split}] Rows: {len(df)}")

    # Fill empty target_text (test set) with empty string
    df["target_text"] = df["target_text"].fillna("").astype(str)

    # Compute and print stats
    print(f"  [{split}] Computing token length stats...")
    stats = compute_stats(df, tokenizer)
    print(f"  [{split}] Input  lengths — min: {stats['input_length']['min']}, "
          f"max: {stats['input_length']['max']}, "
          f"mean: {stats['input_length']['mean']}, "
          f"truncated: {stats['input_length']['truncated']} ({stats['input_length']['truncated_pct']}%)")
    print(f"  [{split}] Target lengths — min: {stats['target_length']['min']}, "
          f"max: {stats['target_length']['max']}, "
          f"mean: {stats['target_length']['mean']}, "
          f"truncated: {stats['target_length']['truncated']} ({stats['target_length']['truncated_pct']}%)")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenize in batches
    dataset = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=32,
        desc=f"Tokenizing {split}",
    )

    # Set format to PyTorch tensors for training
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Save to disk
    split_out = os.path.join(out_dir, split.lower())
    dataset.save_to_disk(split_out)
    print(f"  [{split}] Saved to: {split_out}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Tokenize edge-clinic cleaned data")
    parser.add_argument("--train", required=True, help="Path to train_clean.csv")
    parser.add_argument("--test",  required=True, help="Path to test_clean.csv")
    parser.add_argument("--out",   required=True, help="Output directory for tokenized datasets")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # Load cleaned CSVs
    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)

    # Process splits
    all_stats = {}
    all_stats["train"] = process_split(train_df, tokenizer, split="TRAIN", out_dir=args.out)
    all_stats["test"]  = process_split(test_df,  tokenizer, split="TEST",  out_dir=args.out)

    # Save stats
    stats_path = os.path.join(args.out, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[DONE] Token stats saved to: {stats_path}")
    print(f"[DONE] Tokenized datasets written to: {args.out}")


if __name__ == "__main__":
    main()