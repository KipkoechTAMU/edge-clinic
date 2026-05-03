"""
finetune.py
-----------
Fine-tunes T5-small with LoRA adapters on the clinical vignette dataset.
Designed to run on HPRC via SLURM but works locally too.

Usage:
    python src/finetune.py \
        --train     data/processed/tokenized/train \
        --out       checkpoints/lora-t5-small \
        --log       results/training_log.json \
        --epochs    10 \
        --batch     8 \
        --lr        3e-4

Outputs:
    checkpoints/lora-t5-small/   # LoRA adapter weights
    results/training_log.json    # loss per epoch for plotting
"""

import argparse
import json
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "t5-small"

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],   # inject into attention query and value
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)


# ---------------------------------------------------------------------------
# Loss logger callback
# ---------------------------------------------------------------------------

class LossLoggerCallback(TrainerCallback):
    """
    Captures training and evaluation loss per epoch and saves to JSON.
    Passed to Trainer as a custom callback.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.records  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step}
        if "loss" in logs:
            record["train_loss"] = round(logs["loss"], 4)
        if "eval_loss" in logs:
            record["eval_loss"] = round(logs["eval_loss"], 4)
        if record.keys() - {"step"}:  # only save if there's actual loss data
            self.records.append(record)
            with open(self.log_path, "w") as f:
                json.dump(self.records, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune T5-small")
    parser.add_argument("--train",   required=True, help="Path to tokenized train dataset")
    parser.add_argument("--out",     required=True, help="Output dir for LoRA adapter weights")
    parser.add_argument("--log",     required=True, help="Path to save training log JSON")
    parser.add_argument("--epochs",  type=int,   default=10,   help="Number of training epochs")
    parser.add_argument("--batch",   type=int,   default=8,    help="Per-device train batch size")
    parser.add_argument("--lr",      type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup",  type=int,   default=50,   help="Warmup steps")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load tokenized dataset ---
    print(f"\nLoading tokenized dataset from: {args.train}")
    train_dataset = load_from_disk(args.train)
    print(f"  Train examples: {len(train_dataset)}")

    # --- Load base model and tokenizer ---
    print(f"\nLoading base model: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # --- Apply LoRA ---
    print("\nApplying LoRA adapters...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model       = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data collator ---
    # Handles dynamic padding within each batch
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,          # keep only last 2 checkpoints to save disk
        fp16=(device == "cuda"),     # mixed precision on GPU
        predict_with_generate=False,
        report_to="none",            # disable wandb/tensorboard
        load_best_model_at_end=False,
    )

    # --- Loss logger ---
    loss_logger = LossLoggerCallback(log_path=args.log)

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[loss_logger],
    )

    # --- Train ---
    print(f"\nStarting training: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    trainer.train()

    # --- Save LoRA adapter weights ---
    print(f"\nSaving LoRA adapter weights to: {args.out}")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"Training log saved to: {args.log}")
    print("\n[DONE] Fine-tuning complete.")


if __name__ == "__main__":
    main()