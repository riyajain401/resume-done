"""
rag_engine.py
--------------
A small, dependency-light RAG pipeline:

  documents -> chunk -> embed -> FAISS index -> similarity search

Embeddings come from OpenAI's text-embedding-3-small in "live" mode.
In "mock" mode (no API key needed - used for local testing/demo) we
fall back to a deterministic hashing-based embedding so the whole
pipeline still runs end-to-end and can be unit tested without network
access or API credits.
"""

from __future__ import annotations
import hashlib
import numpy as np
import faiss

EMBED_DIM = 1536  # matches text-embedding-3-small


def _mock_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding: same text -> same vector, and
    textually similar strings land closer together than random ones
    because we hash overlapping n-grams into the same dimensions."""
    vec = np.zeros(dim, dtype="float32")
    words = text.lower().split()
    for i in range(len(words)):
        gram = " ".join(words[i : i + 2])
        h = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class VectorStore:
    """Thin wrapper around a FAISS flat index plus the source chunks,
    so we can map similarity hits back to readable text."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
        self.chunks: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[str], metadatas: list[dict]):
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        k = min(k, len(self.chunks)) or 1
        scores, idxs = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(score),
                }
            )
        return results


class RagEngine:
    def __init__(self, openai_client=None, mode: str = "mock"):
        self.openai_client = openai_client
        self.mode = mode
        self.store = VectorStore()

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.mode == "live" and self.openai_client is not None:
            resp = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            vectors = np.array([d.embedding for d in resp.data], dtype="float32")
        else:
            vectors = np.stack([_mock_embed(t) for t in texts])
        # normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def index_document(self, chunks: list[str], source: str):
        if not chunks:
            return
        embeddings = self.embed(chunks)
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.store.add(embeddings, chunks, metadatas)

    def retrieve(self, query: str, k: int = 5):
        query_emb = self.embed([query])[0]
        return self.store.search(query_emb, k=k)
