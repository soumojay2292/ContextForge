"""Decide from retrieval scores whether the indexed documents can answer a question."""

from dataclasses import dataclass
from enum import Enum

from src.retrieval.vector_store import SearchResult, VectorStore

# Cosine similarity of the best-matching chunk at or above which a question counts
# as covered by the documents. Measured with all-MiniLM-L6-v2 on the project's test
# document at chunk sizes 250 and 1000: 12 questions the document answers scored
# 0.25 to 0.84, and 12 unrelated questions scored -0.09 to 0.09. 0.2 falls in that
# gap, nearer the answerable side, because the top score of unrelated questions
# rises as more chunks are indexed. Recalibrate when changing the embedding model.
DEFAULT_SIMILARITY_THRESHOLD = 0.2
DEFAULT_TOP_K = 5


class RoutingStatus(str, Enum):
    """Why a question was or was not judged answerable from the documents."""

    SUPPORTED = "supported"
    BELOW_THRESHOLD = "below_threshold"
    NO_DOCUMENTS = "no_documents"


@dataclass(frozen=True)
class RoutingDecision:
    """The outcome of routing one question.

    Attributes:
        status: Whether the question is supported, and if not, why.
        reason: Human-readable explanation of the status.
        results: Retrieved chunks with scores, most similar first. Kept for
            unsupported questions too, as the closest matches found.
        top_score: Similarity of the best match, or None if nothing was searched.
    """

    status: RoutingStatus
    reason: str
    results: tuple[SearchResult, ...]
    top_score: float | None

    @property
    def supported(self) -> bool:
        """Whether the documents hold enough evidence to answer the question."""
        return self.status is RoutingStatus.SUPPORTED


class QueryRouter:
    """Retrieves chunks for a question and judges whether they are relevant enough.

    Only supported questions should be passed on for answering; for the rest,
    the documents most likely do not cover the topic.

    Args:
        vector_store: Index of the document chunks to search.
        similarity_threshold: Minimum top cosine similarity, from -1 to 1, for
            a question to be supported.
        k: Number of chunks to retrieve.

    Raises:
        ValueError: If similarity_threshold is not a number from -1 to 1, or k
            is not a positive integer.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        k: int = DEFAULT_TOP_K,
    ):
        # bool is a subclass of int, but True/False here are mistakes, not 1/0.
        if isinstance(similarity_threshold, bool) or not isinstance(similarity_threshold, (int, float)) or not (
            -1.0 <= similarity_threshold <= 1.0
        ):
            raise ValueError(f"similarity_threshold must be a number from -1 to 1, got {similarity_threshold!r}")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        self.vector_store = vector_store
        self.similarity_threshold = float(similarity_threshold)
        self.k = k

    def route(self, question: str) -> RoutingDecision:
        """Search for question and decide whether the documents support it.

        Raises:
            ValueError: If question is empty.
        """
        if not question.strip():
            raise ValueError("Question must not be empty")
        if len(self.vector_store) == 0:
            return RoutingDecision(
                status=RoutingStatus.NO_DOCUMENTS,
                reason="No documents have been indexed.",
                results=(),
                top_score=None,
            )

        results = tuple(self.vector_store.search(question, k=self.k))
        top_score = results[0].score
        if top_score >= self.similarity_threshold:
            return RoutingDecision(
                status=RoutingStatus.SUPPORTED,
                reason=f"Top similarity {top_score:.3f} meets the threshold of {self.similarity_threshold:.3f}.",
                results=results,
                top_score=top_score,
            )
        return RoutingDecision(
            status=RoutingStatus.BELOW_THRESHOLD,
            reason=(
                f"Top similarity {top_score:.3f} is below the threshold of {self.similarity_threshold:.3f}, "
                "so the documents most likely do not cover this question."
            ),
            results=results,
            top_score=top_score,
        )
