"""
embed.py
--------
Embeds all training vignettes using a sentence transformer model.
Produces a numpy array of vectors and an index mapping back to Master_Index.

These embeddings are consumed by index.py to build the FAISS index
used for retrieval at inference time.

Usage:
    python src/embed.py \
        --train data/processed/train_clean.csv \
        --out   data/processed/

Outputs:
    data/processed/embeddings.npy        # shape: (400, 384)
    data/processed/embedding_index.csv   # maps row position -> Master_Index + input_text
"""

import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2: fast, lightweight, strong semantic similarity performance
# embedding dim: 384
# good fit for short-to-medium clinical text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE      = 32


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_vignettes(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed a list of texts into vectors.
    Returns a numpy array of shape (n, embedding_dim).
    """
    print(f"  Embedding {len(texts)} vignettes in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize for cosine similarity via dot product
    )
    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed training vignettes for RAG")
    parser.add_argument("--train", required=True, help="Path to train_clean.csv")
    parser.add_argument("--out",   required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load training data ---
    print("Loading training data...")
    train_df = pd.read_csv(args.train)
    print(f"  Rows: {len(train_df)}")

    texts = train_df["input_text"].tolist()

    # --- Load embedding model ---
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # --- Embed ---
    print("\nEmbedding vignettes...")
    embeddings = embed_vignettes(texts, model)
    print(f"  Embedding shape: {embeddings.shape}")  # should be (400, 384)

    # --- Save embeddings ---
    emb_path = os.path.join(args.out, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"\nEmbeddings saved to : {emb_path}")

    # --- Save index mapping ---
    # Maps each row position back to Master_Index and stores input_text
    # so retrieve.py can reconstruct the full context without reloading the CSV
    index_df = train_df[["Master_Index", "input_text", "target_text"]].reset_index(drop=True)
    index_df.index.name = "embedding_row"
    index_path = os.path.join(args.out, "embedding_index.csv")
    index_df.to_csv(index_path)
    print(f"Embedding index saved to: {index_path}")

    # --- Sanity check ---
    print("\n--- Sanity Check ---")
    print(f"  Embeddings shape : {embeddings.shape}")
    print(f"  Index rows       : {len(index_df)}")
    print(f"  Sample vector    : {embeddings[0][:5]}...")  # first 5 dims of first vector
    print(f"  Vector norm      : {np.linalg.norm(embeddings[0]):.4f}")  # should be ~1.0 if normalized


if __name__ == "__main__":
    main()