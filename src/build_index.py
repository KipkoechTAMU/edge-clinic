"""
index.py
--------
Builds a FAISS flat index from pre-computed training vignette embeddings.
Saves the index to disk for use by retrieve.py at inference time.

Usage:
    python src/index.py \
        --embeddings data/processed/embeddings.npy \
        --out        data/processed/

Outputs:
    data/processed/faiss.index   # FAISS flat index, 400 vectors x 384 dims
"""

import argparse
import os

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384   # must match all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
    parser.add_argument("--out",        required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load embeddings ---
    print("Loading embeddings...")
    embeddings = np.load(args.embeddings).astype("float32")  # FAISS requires float32
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    # --- Validate dimensions ---
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}. "
            f"Check that embed.py used the correct model."
        )

    # --- Build FAISS index ---
    # IndexFlatIP: exact inner product search
    # Since embeddings are L2-normalized (done in embed.py),
    # inner product == cosine similarity — higher score = more similar
    print("\nBuilding FAISS flat index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  Vectors in index: {index.ntotal}")

    # --- Sanity check: query with first vector, expect itself as top result ---
    print("\nSanity check: querying index with first training vector...")
    query = embeddings[0:1]  # shape (1, 384)
    scores, indices = index.search(query, k=3)
    print(f"  Top-3 indices : {indices[0].tolist()}")
    print(f"  Top-3 scores  : {[round(s, 4) for s in scores[0].tolist()]}")
    print(f"  ✓ Top result is index 0 (itself): {indices[0][0] == 0}")

    # --- Save index ---
    index_path = os.path.join(args.out, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"\nFAISS index saved to: {index_path}")
    print(f"Index size on disk  : {os.path.getsize(index_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()