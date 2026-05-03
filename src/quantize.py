"""
quantize.py
-----------
Merges LoRA adapter weights into T5-small base model, applies INT8
dynamic quantization, and benchmarks size and latency improvements.

Usage:
    python src/quantize.py \
        --adapter  checkpoints/lora-t5-small \
        --test     data/processed/test_clean.csv \
        --out      checkpoints/quantized-t5-small \
        --results  results/

Outputs:
    checkpoints/quantized-t5-small/   # quantized model weights
    results/quantization_results.json # size, latency, ROUGE comparison
"""

import argparse
import json
import os
import time

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
N_SAMPLES      = 20    # number of test examples for latency benchmarking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_size_mb(model) -> float:
    """Compute total size of model parameters in MB."""
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in model.parameters()
    )
    return round(total_bytes / (1024 ** 2), 2)


def get_saved_size_mb(path: str) -> float:
    """Compute total size of all files in a directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return round(total / (1024 ** 2), 2)


def generate_single(text: str, model, tokenizer, device: str) -> tuple[str, float]:
    """
    Generate a single prediction and return (prediction, latency_ms).
    Used for latency benchmarking.
    """
    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=4,
            early_stopping=True,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction, latency_ms


def benchmark_latency(df: pd.DataFrame, model, tokenizer, device: str, n: int) -> dict:
    """
    Run inference on n samples and return latency statistics in ms.
    Warms up with 2 runs before measuring.
    """
    samples = df["input_text"].tolist()[:n]

    # Warmup
    for text in samples[:2]:
        generate_single(text, model, tokenizer, device)

    # Benchmark
    latencies = []
    for text in samples:
        _, lat = generate_single(text, model, tokenizer, device)
        latencies.append(lat)

    return {
        "min_ms":  round(min(latencies), 1),
        "max_ms":  round(max(latencies), 1),
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "n":       n,
    }


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
    parser = argparse.ArgumentParser(description="Quantize LoRA fine-tuned T5-small")
    parser.add_argument("--adapter",  required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--test",     required=True, help="Path to test_clean.csv")
    parser.add_argument("--train",    required=True, help="Path to train_clean.csv for references")
    parser.add_argument("--out",      required=True, help="Output dir for quantized model")
    parser.add_argument("--results",  required=True, help="Output dir for results JSON")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    # --- Load data ---
    test_df  = pd.read_csv(args.test)
    train_df = pd.read_csv(args.train)
    references = train_df["target_text"].fillna("").tolist()[:len(test_df)]

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # --- Load base model + LoRA adapter ---
    print(f"Loading base model: {MODEL_NAME}")
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter from: {args.adapter}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    # --- Merge LoRA weights into base model ---
    print("\nMerging LoRA adapter into base model...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    float32_size = get_model_size_mb(merged_model)
    print(f"  Float32 model size (in memory): {float32_size} MB")

    # --- Benchmark float32 latency on CPU ---
    print(f"\nBenchmarking float32 latency on CPU ({N_SAMPLES} samples)...")
    float32_latency = benchmark_latency(test_df, merged_model, tokenizer, "cpu", N_SAMPLES)
    print(f"  Mean latency: {float32_latency['mean_ms']} ms")
    print(f"  Min  latency: {float32_latency['min_ms']} ms")
    print(f"  Max  latency: {float32_latency['max_ms']} ms")

    # --- Compute float32 ROUGE on subset ---
    print(f"\nComputing float32 ROUGE on {N_SAMPLES} samples...")
    float32_preds = []
    for text in test_df["input_text"].tolist()[:N_SAMPLES]:
        pred, _ = generate_single(text, merged_model, tokenizer, "cpu")
        float32_preds.append(pred)
    float32_rouge = compute_rouge(float32_preds, references[:N_SAMPLES])
    print(f"  ROUGE-1: {float32_rouge['rouge1']}")
    print(f"  ROUGE-2: {float32_rouge['rouge2']}")
    print(f"  ROUGE-L: {float32_rouge['rougeL']}")

    # --- Apply INT8 dynamic quantization ---
    print("\nApplying INT8 dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        merged_model,
        {torch.nn.Linear},   # quantize all Linear layers
        dtype=torch.qint8,
    )
    quantized_model.eval()

    int8_size = get_model_size_mb(quantized_model)
    print(f"  INT8 model size (in memory): {int8_size} MB")
    print(f"  Size reduction: {round((1 - int8_size / float32_size) * 100, 1)}%")

    # --- Benchmark INT8 latency on CPU ---
    print(f"\nBenchmarking INT8 latency on CPU ({N_SAMPLES} samples)...")
    int8_latency = benchmark_latency(test_df, quantized_model, tokenizer, "cpu", N_SAMPLES)
    print(f"  Mean latency: {int8_latency['mean_ms']} ms")
    print(f"  Min  latency: {int8_latency['min_ms']} ms")
    print(f"  Max  latency: {int8_latency['max_ms']} ms")
    print(f"  Speedup: {round(float32_latency['mean_ms'] / int8_latency['mean_ms'], 2)}x")

    # --- Compute INT8 ROUGE on subset ---
    print(f"\nComputing INT8 ROUGE on {N_SAMPLES} samples...")
    int8_preds = []
    for text in test_df["input_text"].tolist()[:N_SAMPLES]:
        pred, _ = generate_single(text, quantized_model, tokenizer, "cpu")
        int8_preds.append(pred)
    int8_rouge = compute_rouge(int8_preds, references[:N_SAMPLES])
    print(f"  ROUGE-1: {int8_rouge['rouge1']}")
    print(f"  ROUGE-2: {int8_rouge['rouge2']}")
    print(f"  ROUGE-L: {int8_rouge['rougeL']}")

    # --- Save quantized model ---
    print(f"\nSaving quantized model to: {args.out}")
    torch.save(quantized_model.state_dict(), os.path.join(args.out, "quantized_model.pt"))
    tokenizer.save_pretrained(args.out)
    saved_size = get_saved_size_mb(args.out)
    print(f"  Saved size on disk: {saved_size} MB")

    # --- Summary ---
    print("\n--- Quantization Summary ---")
    print(f"  {'Metric':<30} {'Float32':>12} {'INT8':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Size in memory (MB)':<30} {float32_size:>12} {int8_size:>12}")
    print(f"  {'Mean latency (ms)':<30} {float32_latency['mean_ms']:>12} {int8_latency['mean_ms']:>12}")
    print(f"  {'ROUGE-1':<30} {float32_rouge['rouge1']:>12} {int8_rouge['rouge1']:>12}")
    print(f"  {'ROUGE-2':<30} {float32_rouge['rouge2']:>12} {int8_rouge['rouge2']:>12}")
    print(f"  {'ROUGE-L':<30} {float32_rouge['rougeL']:>12} {int8_rouge['rougeL']:>12}")

    # --- Save results ---
    results = {
        "float32": {
            "size_mb": float32_size,
            "latency": float32_latency,
            "rouge":   float32_rouge,
        },
        "int8": {
            "size_mb": int8_size,
            "latency": int8_latency,
            "rouge":   int8_rouge,
        },
        "size_reduction_pct":  round((1 - int8_size / float32_size) * 100, 1),
        "speedup":             round(float32_latency['mean_ms'] / int8_latency['mean_ms'], 2),
        "rouge1_drop":         round(float32_rouge['rouge1'] - int8_rouge['rouge1'], 4),
    }
    results_path = os.path.join(args.results, "quantization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()