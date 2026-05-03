"""
lora_inference.py
-----------------
Runs inference on the test set using the LoRA fine-tuned T5-small model
and computes ROUGE scores to compare against the zero-shot baseline.

Usage:
    python src/lora_inference.py \
        --test       data/processed/test_clean.csv \
        --train      data/processed/train_clean.csv \
        --adapter    checkpoints/lora-t5-small \
        --out        results/

Outputs:
    results/lora_predictions.csv   # input, prediction, reference per row
    results/lora_rouge.json        # ROUGE-1, ROUGE-2, ROUGE-L scores
"""

import argparse
import json
import os

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME     = "t5-small"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 256
BATCH_SIZE     = 8


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_predictions(df: pd.DataFrame, model, tokenizer, device: str) -> list[str]:
    """Run inference on all rows in df, return list of generated strings."""
    predictions = []
    total = len(df)

    model.eval()
    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            batch_df = df.iloc[start:start + BATCH_SIZE]
            inputs = tokenizer(
                list(batch_df["input_text"]),
                max_length=MAX_INPUT_LEN,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_TARGET_LEN,
                num_beams=4,
                early_stopping=True,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)

            done = min(start + BATCH_SIZE, total)
            print(f"  Processed {done}/{total} examples", end="\r")

    print()
    return predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L averaged across all examples."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return {
        "rouge1": round(sum(scores["rouge1"]) / len(scores["rouge1"]), 4),
        "rouge2": round(sum(scores["rouge2"]) / len(scores["rouge2"]), 4),
        "rougeL": round(sum(scores["rougeL"]) / len(scores["rougeL"]), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuned T5-small inference")
    parser.add_argument("--test",    required=True, help="Path to test_clean.csv")
    parser.add_argument("--train",   required=True, help="Path to train_clean.csv (for references)")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter weights dir")
    parser.add_argument("--out",     required=True, help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load data ---
    test_df  = pd.read_csv(args.test)
    train_df = pd.read_csv(args.train)
    print(f"Test  rows : {len(test_df)}")
    print(f"Train rows : {len(train_df)}")

    # --- References (same logic as baseline.py) ---
    has_test_labels = (
        "target_text" in test_df.columns
        and test_df["target_text"].notna().any()
        and (test_df["target_text"].str.strip() != "").any()
    )

    if has_test_labels:
        print("Test labels found — evaluating against test targets.")
        references = test_df["target_text"].fillna("").tolist()
        eval_df = test_df
    else:
        print("No test labels — using first 100 train targets as proxy references.")
        references = train_df["target_text"].fillna("").tolist()[:len(test_df)]
        eval_df = test_df

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # --- Load base model + LoRA adapter ---
    print(f"Loading base model : {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.to(device)
    print("LoRA adapter loaded successfully.")

    # --- Generate predictions ---
    print(f"\nGenerating predictions (batch size={BATCH_SIZE})...")
    predictions = generate_predictions(eval_df, model, tokenizer, device)

    # --- Compute ROUGE ---
    print("Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)

    print("\n--- LoRA Fine-tuned T5-small ROUGE Scores ---")
    print(f"  ROUGE-1 : {rouge_scores['rouge1']}")
    print(f"  ROUGE-2 : {rouge_scores['rouge2']}")
    print(f"  ROUGE-L : {rouge_scores['rougeL']}")

    # --- Load baseline for comparison ---
    baseline_path = os.path.join(args.out, "baseline_rouge.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print("\n--- Improvement over zero-shot baseline ---")
        print(f"  ROUGE-1 : {baseline['rouge1']} → {rouge_scores['rouge1']} "
              f"(+{round(rouge_scores['rouge1'] - baseline['rouge1'], 4)})")
        print(f"  ROUGE-2 : {baseline['rouge2']} → {rouge_scores['rouge2']} "
              f"(+{round(rouge_scores['rouge2'] - baseline['rouge2'], 4)})")
        print(f"  ROUGE-L : {baseline['rougeL']} → {rouge_scores['rougeL']} "
              f"(+{round(rouge_scores['rougeL'] - baseline['rougeL'], 4)})")

    # --- Save predictions ---
    results_df = eval_df[["Master_Index", "input_text"]].copy()
    results_df["prediction"] = predictions
    results_df["reference"]  = references
    pred_path = os.path.join(args.out, "lora_predictions.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to : {pred_path}")

    # --- Save ROUGE scores ---
    rouge_path = os.path.join(args.out, "lora_rouge.json")
    with open(rouge_path, "w") as f:
        json.dump({"model": MODEL_NAME, "mode": "lora_finetuned", **rouge_scores}, f, indent=2)
    print(f"ROUGE scores saved to: {rouge_path}")


if __name__ == "__main__":
    main()