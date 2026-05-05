# edge-clinic

Clinical response prediction for Kenyan frontline nurses using RAG + LoRA fine-tuned T5-small, optimized for edge deployment on resource-constrained hardware.

## Project Overview

This project trains a lightweight NLP model to generate clinical responses to nursing vignettes from Kenyan healthcare settings. The pipeline combines:

- **LoRA fine-tuning** of T5-small on 400 labeled clinical vignettes
- **RAG (Retrieval Augmented Generation)** using FAISS for context retrieval
- **INT8 quantization** for edge deployment (72.8% size reduction)


## Repository Structure

```
edge-clinic/
├── data/
│   ├── raw/                          # original train.xlsx and test_raw.csv (not committed)
│   ├── processed/                    # cleaned CSVs and tokenized datasets (not committed)
│   └── augmented/                    # augmented training data (not committed)
├── src/
│   ├── preprocess.py                 # clean and normalize raw data
│   ├── tokenize_data.py              # tokenize with T5 tokenizer
│   ├── embed.py                      # embed training vignettes with MiniLM
│   ├── build_index.py                # build FAISS index
│   ├── retrieve.py                   # RAG retrieval and prompt construction
│   ├── baseline.py                   # zero-shot T5-small inference
│   ├── finetune.py                   # LoRA fine-tuning
│   ├── lora_inference.py             # LoRA fine-tuned inference + ROUGE
│   ├── rag_inference.py              # RAG + LoRA inference + ROUGE
│   └── quantize.py                   # INT8 quantization + benchmarking
├── slurm/
│   └── train.sh                      # HPRC SLURM job script
├── checkpoints/                      # model weights (not committed)
├── results/                          # ROUGE scores and logs (not committed)
├── report/                           # IEEE-format report
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/KipkoechTAMU/edge-clinic.git
cd edge-clinic
pip install -r requirements.txt
```

## Reproducing Results

Run each step in order from the project root.

**1. Preprocess raw data**
```bash
python src/preprocess.py \
    --train data/raw/train.xlsx \
    --test  data/raw/test_raw.csv \
    --out   data/processed/
```

**2. Tokenize**
```bash
python src/tokenize_data.py \
    --train data/processed/train_clean.csv \
    --test  data/processed/test_clean.csv \
    --out   data/processed/tokenized/
```

**3. Build RAG index**
```bash
python src/embed.py \
    --train data/processed/train_clean.csv \
    --out   data/processed/

python src/build_index.py \
    --embeddings data/processed/embeddings.npy \
    --out        data/processed/
```

**4. Zero-shot baseline**
```bash
python src/baseline.py \
    --test  data/processed/test_clean.csv \
    --train data/processed/train_clean.csv \
    --out   results/
```

**5. LoRA fine-tuning**
```bash
python src/finetune.py \
    --train   data/processed/tokenized/train \
    --out     checkpoints/lora-t5-small \
    --log     results/training_log.json \
    --epochs  20 \
    --batch   8 \
    --lr      3e-4
```

**6. LoRA inference**
```bash
python src/lora_inference.py \
    --test    data/processed/test_clean.csv \
    --train   data/processed/train_clean.csv \
    --adapter checkpoints/lora-t5-small \
    --out     results/
```

**7. RAG + LoRA inference**
```bash
python src/rag_inference.py \
    --test      data/processed/test_clean.csv \
    --train     data/processed/train_clean.csv \
    --adapter   checkpoints/lora-t5-small \
    --index     data/processed/faiss.index \
    --emb-index data/processed/embedding_index.csv \
    --out       results/ \
    --top-k     3
```

**8. Quantization**
```bash
python src/quantize.py \
    --adapter  checkpoints/lora-t5-small \
    --test     data/processed/test_clean.csv \
    --train    data/processed/train_clean.csv \
    --out      checkpoints/quantized-t5-small \
    --results  results/
```


## Hardware

- **Training:** NVIDIA RTX 4050 Laptop GPU (6.4GB VRAM)
- **Target deployment:** NVIDIA Jetson Nano (4GB RAM)

## Citation

Dataset: Kenyan clinical nursing vignettes collected from frontline nurses across Uasin Gishu and Kiambu counties.