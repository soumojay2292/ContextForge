"""Index chunk embeddings with FAISS and retrieve the chunks most similar to a query."""

from collections.abc import Sequence
from dataclasses import dataclass

import faiss
import numpy as np

from src.ingestion.chunking import DocumentChunk
from src.retrieval.embeddings import EmbeddingModel


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and how similar it is to the query.

    Attributes:
        chunk: The retrieved chunk, including its source and page number.
        score: Cosine similarity to the query, from -1 to 1 (higher is more similar).
    """

    chunk: DocumentChunk
    score: float


class VectorStore:
    """FAISS inner-product index over chunk embeddings.

    Embeddings are L2-normalized before indexing and searching, so inner-product
    scores are cosine similarities. FAISS numbers vectors 0, 1, 2, ... in the
    order they are added, and chunks are stored in the same order, so a FAISS
    result id is the position of its chunk.

    Args:
        embedding_model: Embeds chunks when they are added and queries when searching.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.dimension: int = embedding_model.dimension
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks: list[DocumentChunk] = []

    def __len__(self) -> int:
        return self._index.ntotal

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Embed chunks and add them to the index."""
        self.add_embeddings(chunks, self.embedding_model.embed_chunks(chunks))

    def add_embeddings(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks whose embeddings were already computed.

        Raises:
            ValueError: If embeddings is not shaped (len(chunks), dimension).
        """
        # Copy, because faiss.normalize_L2 works in place and needs C-ordered float32.
        vectors = np.array(embeddings, dtype=np.float32, order="C")
        expected_shape = (len(chunks), self.dimension)
        if vectors.shape != expected_shape:
            raise ValueError(f"Expected embeddings of shape {expected_shape}, got {vectors.shape}")
        if not chunks:
            return

        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return up to k chunks most similar to a text query, most similar first.

        Raises:
            ValueError: If the query is empty or k is not a positive integer.
        """
        return self.search_by_embedding(self.embedding_model.embed_query(query), k)

    def search_by_embedding(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Return up to k chunks most similar to a query embedding, most similar first.

        Returns fewer than k results when the index holds fewer than k chunks,
        and an empty list when the index is empty.

        Raises:
            ValueError: If k is not a positive integer, or query_embedding is not
                shaped (dimension,).
        """
        _validate_k(k)
        query = np.array(query_embedding, dtype=np.float32, order="C")
        if query.shape != (self.dimension,):
            raise ValueError(f"Expected a query embedding of shape ({self.dimension},), got {query.shape}")
        if len(self) == 0:
            return []

        query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        # Asking FAISS for more results than it has pads them with id -1, so cap k.
        scores, ids = self._index.search(query, min(int(k), len(self)))
        return [SearchResult(chunk=self._chunks[i], score=float(score)) for score, i in zip(scores[0], ids[0])]


def _validate_k(k: int) -> None:
    # bool is a subclass of int, but k=True is a mistake rather than k=1.
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
