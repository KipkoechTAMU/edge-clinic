"""
rag_inference.py
----------------
Combines RAG retrieval with LoRA fine-tuned T5-small for inference.
For each test vignette, retrieves top-k similar training cases,
constructs an augmented prompt, and generates a clinical response.

Usage:
    python src/rag_inference.py \
        --test       data/processed/test_clean.csv \
        --train      data/processed/train_clean.csv \
        --adapter    checkpoints/lora-t5-small \
        --index      data/processed/faiss.index \
        --emb-index  data/processed/embedding_index.csv \
        --out        results/ \
        --top-k      3

Outputs:
    results/rag_predictions.csv   # input, augmented prompt, prediction, reference
    results/rag_rouge.json        # ROUGE-1, ROUGE-2, ROUGE-L scores
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Import retriever from retrieve.py in same src/ directory
sys.path.append(os.path.dirname(__file__))
from retrieve import VignetteRetriever


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

def generate_predictions(
    df: pd.DataFrame,
    augmented_prompts: list[str],
    model,
    tokenizer,
    device: str,
) -> list[str]:
    """Run inference on augmented prompts, return list of generated strings."""
    predictions = []
    total = len(augmented_prompts)

    model.eval()
    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            batch_prompts = augmented_prompts[start:start + BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts,
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
    parser = argparse.ArgumentParser(description="RAG + LoRA T5-small inference")
    parser.add_argument("--test",      required=True, help="Path to test_clean.csv")
    parser.add_argument("--train",     required=True, help="Path to train_clean.csv")
    parser.add_argument("--adapter",   required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--index",     required=True, help="Path to faiss.index")
    parser.add_argument("--emb-index", required=True, help="Path to embedding_index.csv")
    parser.add_argument("--out",       required=True, help="Output directory")
    parser.add_argument("--top-k",     type=int, default=3, help="Number of cases to retrieve")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load data ---
    test_df  = pd.read_csv(args.test)
    train_df = pd.read_csv(args.train)
    print(f"Test  rows : {len(test_df)}")
    print(f"Train rows : {len(train_df)}")

    # --- References ---
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

    # --- Load retriever ---
    print(f"\nLoading retriever (top-k={args.top_k})...")
    retriever = VignetteRetriever(
        index_path=args.index,
        emb_index_path=args.emb_index,
        top_k=args.top_k,
    )

    # --- Build augmented prompts for all test vignettes ---
    print("\nBuilding augmented prompts...")
    augmented_prompts = []
    for i, row in eval_df.iterrows():
        augmented = retriever.build_augmented_prompt(row["input_text"])
        augmented_prompts.append(augmented)
        if (i + 1) % 20 == 0:
            print(f"  Built {i + 1}/{len(eval_df)} prompts", end="\r")
    print(f"  Built {len(augmented_prompts)} augmented prompts")

    # Log average augmented prompt length
    avg_len = sum(len(p.split()) for p in augmented_prompts) / len(augmented_prompts)
    print(f"  Average augmented prompt length: {avg_len:.0f} words")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # --- Load base model + LoRA adapter ---
    print(f"Loading base model : {MODEL_NAME}")
    tokenizer  = T5Tokenizer.from_pretrained(MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.to(device)

    # --- Generate predictions ---
    print(f"\nGenerating predictions (batch size={BATCH_SIZE})...")
    predictions = generate_predictions(eval_df, augmented_prompts, model, tokenizer, device)

    # --- Compute ROUGE ---
    print("Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)

    print("\n--- RAG + LoRA T5-small ROUGE Scores ---")
    print(f"  ROUGE-1 : {rouge_scores['rouge1']}")
    print(f"  ROUGE-2 : {rouge_scores['rouge2']}")
    print(f"  ROUGE-L : {rouge_scores['rougeL']}")

    # --- Compare against all previous results ---
    print("\n--- Full Comparison ---")
    print(f"  {'Model':<30} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
    print(f"  {'-'*56}")

    for fname, label in [
        ("baseline_rouge.json",  "Zero-shot T5-small"),
        ("lora_rouge.json",      "T5-small + LoRA"),
        ("rag_rouge.json",       "T5-small + RAG + LoRA"),
    ]:
        fpath = os.path.join(args.out, fname)
        if fname == "rag_rouge.json":
            scores = rouge_scores
        elif os.path.exists(fpath):
            with open(fpath) as f:
                scores = json.load(f)
        else:
            continue
        print(f"  {label:<30} {scores['rouge1']:>8} {scores['rouge2']:>8} {scores['rougeL']:>8}")

    # --- Save predictions ---
    results_df = eval_df[["Master_Index", "input_text"]].copy()
    results_df["augmented_prompt"] = augmented_prompts
    results_df["prediction"]       = predictions
    results_df["reference"]        = references
    pred_path = os.path.join(args.out, "rag_predictions.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to : {pred_path}")

    # --- Save ROUGE scores ---
    rouge_path = os.path.join(args.out, "rag_rouge.json")
    with open(rouge_path, "w") as f:
        json.dump({"model": MODEL_NAME, "mode": "rag_lora", **rouge_scores}, f, indent=2)
    print(f"ROUGE scores saved to: {rouge_path}")


if __name__ == "__main__":
    main()