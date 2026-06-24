"""NOMAD external reconstructive memory bus.

Phase 0: exact retrieval + compression rerank.

Storage: document chunks live on CPU (dict of chunk_id -> text). Vectors are
computed lazily via the model's embedding and cached. Top-K retrieval uses
a mixed scoring function:

    score(chunk_i | x) = λ_exact * R_exact(chunk_i, x)
                        + λ_gzip  * R_gzip(chunk_i, x)

Phase 1+ will add: semantic (embedding cosine), kNN (hidden-state proximity),
trace (temporal recency). Phase 1+ will also add logit bias and hidden adapter.
"""

from __future__ import annotations

import gzip
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MemoryChunk:
    """A single chunk in external memory."""

    chunk_id: str
    text: str
    tokens: list[int] | None = None
    embedding: Tensor | None = None  # [D] mean-pooled embedding
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exact match scoring
# ---------------------------------------------------------------------------


def exact_match_score(query: str, chunk: str) -> float:
    """Normalized exact substring match score.

    Returns 1.0 if chunk contains query verbatim; 0.0 otherwise.
    For token-based scoring, we compute n-gram overlap.
    """
    if not query or not chunk:
        return 0.0

    # Exact substring
    if query in chunk:
        return 1.0

    # Token overlap (word-level)
    query_tokens = set(query.lower().split())
    chunk_tokens = set(chunk.lower().split())
    if not query_tokens:
        return 0.0

    overlap = len(query_tokens & chunk_tokens)
    return overlap / len(query_tokens)


def exact_match_score_ngram(query: str, chunk: str, n: int = 4) -> float:
    """Character n-gram overlap score for fuzzy matching."""
    if not query or not chunk:
        return 0.0

    def ngrams(s: str, k: int) -> set[str]:
        s = s.lower()
        return {s[i : i + k] for i in range(max(0, len(s) - k + 1))}

    q_grams = ngrams(query, n)
    c_grams = ngrams(chunk, n)

    if not q_grams:
        return 0.0

    return len(q_grams & c_grams) / len(q_grams)


# ---------------------------------------------------------------------------
# Compression gain scoring (GzipT-inspired)
# ---------------------------------------------------------------------------


def compression_size(text: str) -> int:
    """Compressed size in bytes using gzip."""
    return len(gzip.compress(text.encode("utf-8", errors="replace")))


def compression_gain(
    query: str,
    chunk: str,
    continuation: str = "",
) -> float:
    """Compression gain: how much chunk helps compress query+continuation.

    G_i = C(query) + C(continuation) - C(chunk + query + continuation)

    Positive = chunk helped (it shares statistical structure).
    Normalized by C(query) for comparability.
    """
    if not query or not chunk:
        return 0.0

    c_query = compression_size(query)
    c_cont = compression_size(continuation) if continuation else 0
    c_joint = compression_size(chunk + query + continuation)

    raw_gain = c_query + c_cont - c_joint
    return raw_gain / max(1, c_query)


def compression_rerank_score(
    query: str,
    chunk: str,
    baseline_size: int | None = None,
) -> float:
    """Compression-based relevance score.

    Lower joint compression = more shared structure = higher score.
    Score = (baseline - joint) / baseline, clamped to [0, 1].
    """
    if not query or not chunk:
        return 0.0

    joint = compression_size(chunk + "\n" + query)
    if baseline_size is None:
        baseline = compression_size(query) + compression_size(chunk)
    else:
        baseline = baseline_size

    if baseline <= 0:
        return 0.0

    gain = (baseline - joint) / baseline
    return max(0.0, min(1.0, gain))


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------


class ExternalMemory:
    """CPU-resident document chunk store with retrieval.

    Phase 0: exact + compression scoring.
    Embeddings are computed lazily and cached on CPU.

    Usage:
        mem = ExternalMemory(hidden_size=256)
        mem.insert("doc1_chunk0", "The capital of France is Paris.")
        scores = mem.retrieve("What is the capital of France?", top_k=4)
        vectors = mem.get_vectors(scores)  # [K, D]
    """

    def __init__(
        self,
        hidden_size: int = 256,
        *,
        lambda_exact: float = 0.7,
        lambda_gzip: float = 0.3,
    ) -> None:
        self.hidden_size = hidden_size
        self.lambda_exact = lambda_exact
        self.lambda_gzip = lambda_gzip

        self._chunks: dict[str, MemoryChunk] = {}
        self._embed_fn: Callable[[str], Tensor] | None = None

    def set_embed_fn(self, fn: Callable[[str], Tensor]) -> None:
        """Set the function used to embed chunks (typically model.embed_tokens)."""
        self._embed_fn = fn

    def insert(
        self,
        chunk_id: str,
        text: str,
        tokens: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a chunk."""
        chunk = MemoryChunk(
            chunk_id=chunk_id,
            text=text,
            tokens=tokens,
            metadata=metadata or {},
        )
        self._chunks[chunk_id] = chunk

    def insert_batch(
        self,
        items: list[tuple[str, str]],
    ) -> None:
        """Insert multiple (chunk_id, text) pairs."""
        for chunk_id, text in items:
            self.insert(chunk_id, text)

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk by ID."""
        self._chunks.pop(chunk_id, None)

    def __len__(self) -> int:
        return len(self._chunks)

    def _embed_chunk(self, chunk: MemoryChunk):
        """Lazily compute embedding for a chunk."""
        if chunk.embedding is not None:
            return
        if self._embed_fn is None:
            # No embed function yet; store zero
            chunk.embedding = torch.zeros(self.hidden_size)
            return
        # Simple mean-pooling of token embeddings
        tokens = chunk.tokens
        if tokens is not None:
            ids = torch.tensor(tokens, dtype=torch.long)
            with torch.no_grad():
                embs = self._embed_fn(ids)  # [T, D]
                chunk.embedding = embs.mean(dim=0).cpu()
        else:
            chunk.embedding = torch.zeros(self.hidden_size)

    def _score_chunk(self, query: str, chunk: MemoryChunk) -> float:
        """Compute mixed retrieval score for a chunk."""
        # Phase 0: exact + compression
        exact = exact_match_score(query, chunk.text)
        gzip_score = compression_rerank_score(query, chunk.text)
        return self.lambda_exact * exact + self.lambda_gzip * gzip_score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Retrieve top-K chunks for a query string.

        Returns list of (chunk_id, score) sorted by descending score.
        """
        if not self._chunks:
            return []

        scored = [
            (cid, self._score_chunk(query, chunk))
            for cid, chunk in self._chunks.items()
        ]

        scored = [(cid, s) for cid, s in scored if s > min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_by_tokens(
        self,
        token_query: str,
        *,
        top_k: int = 4,
    ) -> tuple[list[str], Tensor]:
        """Retrieve and return chunk IDs + embedding vectors [K, D]."""
        results = self.retrieve(token_query, top_k=top_k)
        if not results:
            return [], torch.zeros(0, self.hidden_size)

        chunk_ids = [cid for cid, _ in results]
        vectors = self.get_vectors(chunk_ids)
        return chunk_ids, vectors

    def get_vectors(self, chunk_ids: list[str]) -> Tensor:
        """Get embedding vectors for chunk IDs. Lazily computes embeddings."""
        vectors = []
        for cid in chunk_ids:
            chunk = self._chunks.get(cid)
            if chunk is None:
                vectors.append(torch.zeros(self.hidden_size))
                continue
            self._embed_chunk(chunk)
            vectors.append(chunk.embedding)
        if not vectors:
            return torch.zeros(0, self.hidden_size)
        return torch.stack(vectors)

    def get_chunk_texts(self, chunk_ids: list[str]) -> list[str]:
        """Get raw text for chunk IDs."""
        return [
            self._chunks[cid].text if cid in self._chunks else ""
            for cid in chunk_ids
        ]

    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        total_chars = sum(len(c.text) for c in self._chunks.values())
        embedded = sum(
            1 for c in self._chunks.values() if c.embedding is not None
        )
        return {
            "num_chunks": len(self._chunks),
            "total_chars": total_chars,
            "embedded_chunks": embedded,
        }


# ---------------------------------------------------------------------------
# Batch memory reader (for training)
# ---------------------------------------------------------------------------


class BatchMemoryReader:
    """Precompute external memory reads for a batch of text sequences.

    Given a list of text sequences and an ExternalMemory, computes
    per-position memory vectors by querying the memory with the prefix
    context up to that position.

    For Phase 0, we use a simple sliding window of the last N tokens
    as the query.
    """

    def __init__(
        self,
        memory: ExternalMemory,
        *,
        query_window: int = 32,
        top_k: int = 4,
    ) -> None:
        self.memory = memory
        self.query_window = query_window
        self.top_k = top_k

    def read_batch(
        self,
        texts: list[str],
        *,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Compute external memory reads for each position in each sequence.

        Args:
            texts: list of full text sequences (one per batch item)

        Returns:
            memory_reads: [B, T, D] external memory vectors
        """
        if not self.memory._chunks:
            # No memory loaded; return zeros
            max_len = max(len(t.split()) for t in texts)
            return torch.zeros(len(texts), max_len, self.memory.hidden_size)

        # For simplicity in Phase 0: use the full text as query
        # In practice, this would be the prefix up to position t
        batch = len(texts)
        # Get memory for each sequence
        all_reads = []
        max_len = 0
        for text in texts:
            words = text.split()
            max_len = max(max_len, len(words))
            reads = []
            for t in range(1, len(words) + 1):
                prefix = " ".join(words[max(0, t - self.query_window) : t])
                _, vectors = self.memory.retrieve_by_tokens(
                    prefix, top_k=self.top_k
                )
                if vectors.shape[0] > 0:
                    # Mean pool top-K memory vectors
                    reads.append(vectors.mean(dim=0))  # [D]
                else:
                    reads.append(torch.zeros(self.memory.hidden_size))
            # Pad to max_len
            while len(reads) < max_len:
                reads.append(torch.zeros(self.memory.hidden_size))
            all_reads.append(torch.stack(reads))

        return torch.stack(all_reads).to(device)


# ---------------------------------------------------------------------------
# Utility: ingest text file into memory
# ---------------------------------------------------------------------------


def ingest_text_file(
    memory: ExternalMemory,
    filepath: str,
    *,
    chunk_size_chars: int = 256,
    chunk_overlap: int = 32,
) -> int:
    """Ingest a text file into external memory, splitting into chunks.

    Returns number of chunks added.
    """
    with open(filepath, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Simple character-based chunking with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
            if chunk_id not in memory._chunks:
                memory.insert(chunk_id, chunk_text)
                chunks.append(chunk_id)
        start += chunk_size_chars - chunk_overlap

    return len(chunks)


def ingest_text(
    memory: ExternalMemory,
    text: str,
    *,
    chunk_id_prefix: str = "doc",
    chunk_size_chars: int = 256,
    chunk_overlap: int = 32,
) -> int:
    """Ingest raw text into external memory, splitting into chunks."""
    count = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{chunk_id_prefix}_{hashlib.md5(chunk_text.encode()).hexdigest()[:12]}"
            if chunk_id not in memory._chunks:
                memory.insert(chunk_id, chunk_text)
                count += 1
        start += chunk_size_chars - chunk_overlap
    return count
