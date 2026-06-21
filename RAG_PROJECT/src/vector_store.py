# ============================================================
# vector_store.py
# PURPOSE: Store chunk embeddings and search for the most relevant ones.
#
# REAL-WORLD EQUIVALENT: This is what Pinecone / FAISS / ChromaDB do.
# We're doing the same thing with plain numpy so you can see
# exactly what's happening under the hood.
#
# HOW SEARCH WORKS:
# 1. User asks a question → embed it into a vector Q
# 2. Compare Q to every stored chunk vector using cosine similarity
# 3. Return the top-K chunks with the highest similarity scores
# ============================================================

import numpy as np


def _cosine_similarity(vec_a, vec_b) -> float:
    """
    Measures the angle between two vectors.
    Returns a float in [-1, 1]:
      1.0  → vectors point in the same direction (very similar text)
      0.0  → vectors are perpendicular (unrelated text)
     -1.0  → vectors point in opposite directions

    Formula: cos(θ) = (A · B) / (||A|| * ||B||)
    """
    dot    = np.dot(vec_a, vec_b)           # dot product
    norm_a = np.linalg.norm(vec_a)          # magnitude of A
    norm_b = np.linalg.norm(vec_b)          # magnitude of B

    if norm_a == 0 or norm_b == 0:
        return 0.0  # guard against zero-length vectors

    return float(dot / (norm_a * norm_b))


class VectorStore:
    """
    A simple in-memory store: a list of chunks + a list of their embeddings.
    Index i in self.chunks corresponds to index i in self.embeddings.
    """

    def __init__(self):
        self.chunks: list = []          # raw text strings
        self.embeddings: list = []      # numpy arrays, one per chunk

    def add(self, chunks: list, embeddings) -> None:
        """
        Populate the store with chunks and their pre-computed embeddings.
        Call this once after loading and embedding the document.
        """
        self.chunks = chunks
        self.embeddings = embeddings
        print(f"[VectorStore] Indexed {len(chunks)} chunks.")

    def search(self, query_embedding, top_k: int = 3) -> list:
        """
        Find the top_k chunks most similar to the query_embedding.

        Returns a list of dicts:
          [{"text": "...", "score": 0.87}, ...]
        sorted from most to least relevant.
        """
        if not self.chunks:
            print("[VectorStore] No chunks stored yet!")
            return []

        # Score every stored chunk against the query
        scored = []
        for idx, chunk_emb in enumerate(self.embeddings):
            score = _cosine_similarity(query_embedding, chunk_emb)
            scored.append((score, idx))

        # Sort descending: highest similarity first
        scored.sort(reverse=True)

        # Build the result list with text + score
        results = []
        for score, idx in scored[:top_k]:
            results.append({
                "text":  self.chunks[idx],
                "score": round(score, 4),   # rounded for readability
            })

        return results