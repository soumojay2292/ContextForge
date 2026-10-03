import math
import re

import numpy as np
import pytest

from src.ingestion.chunking import DocumentChunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.query_router import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    QueryRouter,
    RoutingDecision,
    RoutingStatus,
)
from src.retrieval.vector_store import SearchResult, VectorStore


class KeywordEmbedder:
    """Stand-in for EmbeddingModel with one dimension per keyword.

    Similarity depends only on shared keywords, so scores are predictable: a
    query with none of the keywords scores 0 against every chunk. Every embedded
    query is recorded, to check when the router avoids searching.
    """

    KEYWORDS = ("pdf", "chunk", "embedding", "faiss")
    dimension = len(KEYWORDS)

    def __init__(self):
        self.queries: list[str] = []

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        vectors = [self._embed(chunk.text) for chunk in chunks]
        return np.array(vectors, dtype=np.float32).reshape(len(chunks), self.dimension)

    def embed_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z]+", text.lower())
        vector = np.array([words.count(keyword) for keyword in self.KEYWORDS], dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


PDF_CHUNK = DocumentChunk(text="Loading a PDF reads every page of the pdf.", source="manual.pdf", page_number=1, chunk_index=0)
CHUNK_CHUNK = DocumentChunk(text="Each chunk is a piece of one page.", source="manual.pdf", page_number=2, chunk_index=1)
EMBEDDING_CHUNK = DocumentChunk(text="An embedding is a vector of numbers.", source="notes.pdf", page_number=5, chunk_index=3)
ALL_CHUNKS = [PDF_CHUNK, CHUNK_CHUNK, EMBEDDING_CHUNK]

SUPPORTED_QUESTION = "How do I load a PDF?"  # scores 1.0 against PDF_CHUNK
OFF_TOPIC_QUESTION = "What is the capital of Japan?"  # no keywords: scores 0.0 against everything
PARTIAL_QUESTION = "Does the pdf use faiss?"  # half its keywords match PDF_CHUNK: scores 1/sqrt(2)


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def store(embedder: KeywordEmbedder) -> VectorStore:
    vector_store = VectorStore(embedder)
    vector_store.add_chunks(ALL_CHUNKS)
    return vector_store


@pytest.fixture
def empty_store(embedder: KeywordEmbedder) -> VectorStore:
    return VectorStore(embedder)


# --- Supported queries ------------------------------------------------------


def test_relevant_question_is_supported(store: VectorStore):
    decision = QueryRouter(store).route(SUPPORTED_QUESTION)

    assert isinstance(decision, RoutingDecision)
    assert decision.supported
    assert decision.status is RoutingStatus.SUPPORTED
    assert decision.top_score == pytest.approx(1.0)
    assert decision.results[0].chunk == PDF_CHUNK


def test_supported_reason_reports_score_and_threshold(store: VectorStore):
    decision = QueryRouter(store, similarity_threshold=0.5).route(SUPPORTED_QUESTION)

    assert decision.reason == "Top similarity 1.000 meets the threshold of 0.500."


def test_results_are_search_results_ranked_by_score(store: VectorStore):
    decision = QueryRouter(store).route(PARTIAL_QUESTION)

    assert all(isinstance(result, SearchResult) for result in decision.results)
    scores = [result.score for result in decision.results]
    assert scores == sorted(scores, reverse=True)
    assert decision.top_score == scores[0]


def test_k_sets_how_many_chunks_are_retrieved(store: VectorStore):
    assert len(QueryRouter(store).route(SUPPORTED_QUESTION).results) == 3  # default k=5, only 3 chunks
    assert len(QueryRouter(store, k=1).route(SUPPORTED_QUESTION).results) == 1


# --- Unsupported queries ----------------------------------------------------


def test_unrelated_question_is_unsupported(store: VectorStore):
    decision = QueryRouter(store).route(OFF_TOPIC_QUESTION)

    assert not decision.supported
    assert decision.status is RoutingStatus.BELOW_THRESHOLD
    assert decision.top_score == pytest.approx(0.0)
    assert decision.reason.startswith("Top similarity 0.000 is below the threshold of 0.200")


def test_partial_match_below_a_strict_threshold_is_unsupported(store: VectorStore):
    decision = QueryRouter(store, similarity_threshold=0.8).route(PARTIAL_QUESTION)

    assert decision.top_score == pytest.approx(1 / math.sqrt(2))
    assert not decision.supported


def test_unsupported_decision_still_returns_closest_matches(store: VectorStore):
    decision = QueryRouter(store).route(OFF_TOPIC_QUESTION)

    assert len(decision.results) == 3
    assert {result.chunk for result in decision.results} == set(ALL_CHUNKS)


# --- Threshold boundaries and configuration ---------------------------------


def test_score_exactly_at_threshold_is_supported(store: VectorStore):
    top_score = QueryRouter(store).route(PARTIAL_QUESTION).top_score

    assert QueryRouter(store, similarity_threshold=top_score).route(PARTIAL_QUESTION).supported


def test_score_just_below_threshold_is_unsupported(store: VectorStore):
    top_score = QueryRouter(store).route(PARTIAL_QUESTION).top_score
    threshold = float(np.nextafter(top_score, 1.0))

    decision = QueryRouter(store, similarity_threshold=threshold).route(PARTIAL_QUESTION)

    assert not decision.supported
    assert decision.status is RoutingStatus.BELOW_THRESHOLD


def test_score_just_above_threshold_is_supported(store: VectorStore):
    top_score = QueryRouter(store).route(PARTIAL_QUESTION).top_score
    threshold = float(np.nextafter(top_score, -1.0))

    assert QueryRouter(store, similarity_threshold=threshold).route(PARTIAL_QUESTION).supported


def test_threshold_extremes(store: VectorStore):
    assert QueryRouter(store, similarity_threshold=-1.0).route(OFF_TOPIC_QUESTION).supported
    assert QueryRouter(store, similarity_threshold=1.0).route(SUPPORTED_QUESTION).supported
    assert not QueryRouter(store, similarity_threshold=1.0).route(PARTIAL_QUESTION).supported


def test_defaults_are_the_documented_constants(store: VectorStore):
    router = QueryRouter(store)

    assert router.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD == 0.2
    assert router.k == DEFAULT_TOP_K


@pytest.mark.parametrize("threshold", [1.5, -1.01, math.nan, math.inf, "0.5", None, True])
def test_invalid_threshold_raises_value_error(store: VectorStore, threshold):
    with pytest.raises(ValueError, match="similarity_threshold"):
        QueryRouter(store, similarity_threshold=threshold)


@pytest.mark.parametrize("k", [0, -1, 2.5, True, None])
def test_invalid_k_raises_value_error(store: VectorStore, k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        QueryRouter(store, k=k)


# --- Empty stores -----------------------------------------------------------


def test_empty_store_reports_no_documents_without_searching(empty_store: VectorStore, embedder: KeywordEmbedder):
    decision = QueryRouter(empty_store).route(SUPPORTED_QUESTION)

    assert not decision.supported
    assert decision.status is RoutingStatus.NO_DOCUMENTS
    assert decision.reason == "No documents have been indexed."
    assert decision.results == ()
    assert decision.top_score is None
    assert embedder.queries == []


def test_empty_store_ignores_threshold(empty_store: VectorStore):
    assert QueryRouter(empty_store, similarity_threshold=-1.0).route(SUPPORTED_QUESTION).status is RoutingStatus.NO_DOCUMENTS


# --- Invalid input ----------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises_value_error_without_searching(store: VectorStore, embedder: KeywordEmbedder, question: str):
    with pytest.raises(ValueError, match="Question must not be empty"):
        QueryRouter(store).route(question)
    assert embedder.queries == []


def test_empty_question_is_rejected_even_when_store_is_empty(empty_store: VectorStore):
    with pytest.raises(ValueError, match="Question must not be empty"):
        QueryRouter(empty_store).route("  ")


# --- Metadata preservation --------------------------------------------------


def test_supported_results_keep_chunk_metadata(store: VectorStore):
    decision = QueryRouter(store).route("What is an embedding?")
    top = decision.results[0].chunk

    assert top == EMBEDDING_CHUNK
    assert (top.source, top.page_number, top.chunk_index, top.text) == (
        "notes.pdf", 5, 3, "An embedding is a vector of numbers.",
    )


def test_unsupported_results_keep_chunk_metadata(store: VectorStore):
    decision = QueryRouter(store).route(OFF_TOPIC_QUESTION)

    assert sorted((r.chunk.source, r.chunk.page_number, r.chunk.chunk_index) for r in decision.results) == [
        ("manual.pdf", 1, 0), ("manual.pdf", 2, 1), ("notes.pdf", 5, 3),
    ]


def test_metadata_survives_chunks_added_in_batches(embedder: KeywordEmbedder):
    vector_store = VectorStore(embedder)
    vector_store.add_chunks([PDF_CHUNK])
    vector_store.add_chunks([CHUNK_CHUNK, EMBEDDING_CHUNK])

    assert QueryRouter(vector_store).route("Tell me about each chunk").results[0].chunk == CHUNK_CHUNK


# --- With the real embedding model and the default threshold ----------------

DOCUMENT_CHUNKS = [
    DocumentChunk(
        text="ContextForge Test Document 1. Introduction ContextForge is an intelligent document research assistant "
        "designed to retrieve relevant information from uploaded documents and provide grounded answers.",
        source="test_document.pdf", page_number=1, chunk_index=0,
    ),
    DocumentChunk(
        text="2. Document Processing The document processing pipeline extracts text from PDF files, preserves page "
        "information, cleans the extracted text, and divides the content into meaningful chunks.",
        source="test_document.pdf", page_number=1, chunk_index=1,
    ),
    DocumentChunk(
        text="3. Semantic Retrieval Semantic retrieval represents text as numerical embeddings and compares a user "
        "query with document chunks. Relevant chunks can then be supplied to a question answering or generation component.",
        source="test_document.pdf", page_number=1, chunk_index=2,
    ),
    DocumentChunk(
        text="4. Testing A reliable document assistant should test successful document loading as well as invalid files, "
        "empty documents, page ordering, and preservation of source information.",
        source="test_document.pdf", page_number=1, chunk_index=3,
    ),
]


@pytest.fixture(scope="module")
def document_router(model: EmbeddingModel) -> QueryRouter:
    vector_store = VectorStore(model)
    vector_store.add_chunks(DOCUMENT_CHUNKS)
    return QueryRouter(vector_store)


@pytest.mark.parametrize(
    ("question", "expected_chunk_index"),
    [
        ("What does the document processing pipeline do?", 1),
        ("How does semantic retrieval work?", 2),
        ("What should a reliable document assistant test?", 3),
        ("Why is the extracted text cleaned?", 1),
    ],
)
def test_real_model_supports_questions_the_document_answers(document_router: QueryRouter, question: str, expected_chunk_index: int):
    decision = document_router.route(question)

    assert decision.supported
    assert decision.results[0].chunk.chunk_index == expected_chunk_index
    assert decision.results[0].chunk.source == "test_document.pdf"


@pytest.mark.parametrize(
    "question",
    ["What is the capital of Japan?", "How do I change a car tyre?", "Explain quantum entanglement.", "What's a good pasta recipe?"],
)
def test_real_model_rejects_unrelated_questions(document_router: QueryRouter, question: str):
    decision = document_router.route(question)

    assert not decision.supported
    assert decision.top_score < DEFAULT_SIMILARITY_THRESHOLD
