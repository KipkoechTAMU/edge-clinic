"""
retrieve.py
-----------
Retrieval component of the RAG pipeline.
Given a test vignette, finds the top-k most similar training vignettes
using the FAISS index and constructs an augmented prompt for T5.

Can be used as a module (imported by rag_inference.py) or run standalone
to inspect retrieval quality on a few examples.

Usage (standalone):
    python src/retrieve.py \
        --test       data/processed/test_clean.csv \
        --index      data/processed/faiss.index \
        --emb-index  data/processed/embedding_index.csv \
        --top-k      3

Outputs (standalone):
    Prints augmented prompts for the first 3 test vignettes to stdout
    for manual inspection of retrieval quality.
"""

import argparse

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # must match embed.py
TOP_K           = 3                      # number of similar cases to retrieve
MAX_CONTEXT_LEN = 300                    # max chars per retrieved example in prompt


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class VignetteRetriever:
    """
    Loads FAISS index and embedding index once, then answers retrieve() calls.
    Designed to be instantiated once and reused across all test vignettes.
    """

    def __init__(self, index_path: str, emb_index_path: str, top_k: int = TOP_K):
        print(f"Loading FAISS index from: {index_path}")
        self.index   = faiss.read_index(index_path)
        self.emb_df  = pd.read_csv(emb_index_path, index_col="embedding_row")
        self.top_k   = top_k
        self.model   = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  Vectors in index : {self.index.ntotal}")
        print(f"  Index rows       : {len(self.emb_df)}")

    def retrieve(self, query_text: str) -> list[dict]:
        """
        Embed query_text, search FAISS, return top-k training examples.

        Returns:
            List of dicts with keys: master_index, score, input_text, target_text
        """
        # Embed the query
        query_vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Search FAISS
        scores, indices = self.index.search(query_vec, k=self.top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.emb_df.iloc[idx]
            results.append({
                "master_index": row["Master_Index"],
                "score":        round(float(score), 4),
                "input_text":   row["input_text"],
                "target_text":  row["target_text"],
            })

        return results

    def build_augmented_prompt(self, query_text: str) -> str:
        """
        Retrieve top-k similar cases and construct the augmented prompt.

        Format:
            similar cases:
            case 1: <truncated input> response: <truncated clinician response>
            case 2: ...
            case 3: ...
            query: <query_text>

        The full string is the input_text fed into T5 during RAG inference.
        """
        retrieved = self.retrieve(query_text)

        context_parts = []
        for i, case in enumerate(retrieved, start=1):
            # Truncate to keep prompt within T5's 512 token limit
            input_snippet    = case["input_text"][:MAX_CONTEXT_LEN].strip()
            response_snippet = case["target_text"][:MAX_CONTEXT_LEN].strip() if case["target_text"] else ""
            context_parts.append(
                f"case {i}: {input_snippet} response: {response_snippet}"
            )

        context_block  = " | ".join(context_parts)
        augmented      = f"similar cases: {context_block} | query: {query_text}"
        return augmented


# ---------------------------------------------------------------------------
# Standalone inspection
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect RAG retrieval quality")
    parser.add_argument("--test",      required=True, help="Path to test_clean.csv")
    parser.add_argument("--index",     required=True, help="Path to faiss.index")
    parser.add_argument("--emb-index", required=True, help="Path to embedding_index.csv")
    parser.add_argument("--top-k",     type=int, default=TOP_K, help="Number of cases to retrieve")
    parser.add_argument("--n",         type=int, default=3, help="Number of test examples to inspect")
    args = parser.parse_args()

    # Load retriever
    retriever = VignetteRetriever(
        index_path=args.index,
        emb_index_path=args.emb_index,
        top_k=args.top_k,
    )

    # Load test data
    test_df = pd.read_csv(args.test)
    print(f"\nInspecting retrieval for first {args.n} test vignettes...\n")

    for i, row in test_df.head(args.n).iterrows():
        query = row["input_text"]
        print(f"{'='*80}")
        print(f"TEST VIGNETTE {i+1} ({row['Master_Index']})")
        print(f"Query (truncated): {query[:200]}...")

        # Show retrieved cases
        retrieved = retriever.retrieve(query)
        print(f"\nTop-{args.top_k} retrieved training cases:")
        for j, case in enumerate(retrieved, start=1):
            print(f"  [{j}] {case['master_index']} | score: {case['score']}")
            print(f"       input   : {case['input_text'][:150]}...")
            print(f"       response: {case['target_text'][:150]}...")

        # Show augmented prompt length
        augmented = retriever.build_augmented_prompt(query)
        print(f"\nAugmented prompt length: {len(augmented.split())} words")
        print()


if __name__ == "__main__":
    main()