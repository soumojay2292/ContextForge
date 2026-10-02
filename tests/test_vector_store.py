import re

import numpy as np
import pytest

from src.ingestion.chunking import DocumentChunk, chunk_pages
from src.ingestion.pdf_loader import PageData
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import SearchResult, VectorStore


class KeywordEmbedder:
    """Stand-in for EmbeddingModel with one dimension per keyword.

    Similarity depends only on which keywords two texts share, so scores are
    predictable, and no model has to be loaded.
    """

    KEYWORDS = ("pdf", "chunk", "embedding", "faiss")
    dimension = len(KEYWORDS)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        vectors = [self._embed(chunk.text) for chunk in chunks]
        return np.array(vectors, dtype=np.float32).reshape(len(chunks), self.dimension)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query must not be empty")
        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z]+", text.lower())
        vector = np.array([words.count(keyword) for keyword in self.KEYWORDS], dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


def make_chunk(text: str, source: str = "doc.pdf", page_number: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, page_number=page_number, chunk_index=chunk_index)


PDF_CHUNK = make_chunk("Loading a PDF reads every page of the pdf.", page_number=1)
CHUNK_CHUNK = make_chunk("Each chunk is a piece of one page.", page_number=2)
EMBEDDING_CHUNK = make_chunk("An embedding is a vector of numbers.", page_number=3)
ALL_CHUNKS = [PDF_CHUNK, CHUNK_CHUNK, EMBEDDING_CHUNK]


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(KeywordEmbedder())


@pytest.fixture
def filled_store(store: VectorStore) -> VectorStore:
    store.add_chunks(ALL_CHUNKS)
    return store


def unit(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# --- Indexing ---------------------------------------------------------------


def test_new_store_is_empty_with_model_dimension(store: VectorStore):
    assert len(store) == 0
    assert store.dimension == KeywordEmbedder.dimension


def test_add_chunks_indexes_every_chunk(store: VectorStore):
    store.add_chunks(ALL_CHUNKS)

    assert len(store) == 3


def test_adding_in_batches_accumulates(store: VectorStore):
    store.add_chunks(ALL_CHUNKS[:2])
    store.add_chunks(ALL_CHUNKS[2:])

    assert len(store) == 3


def test_embeddings_are_normalized_before_indexing(store: VectorStore):
    store.add_embeddings([PDF_CHUNK, EMBEDDING_CHUNK], np.array([[3, 0, 0, 0], [0, 0, 2, 0]], dtype=np.float32))

    results = store.search_by_embedding(unit(5, 0, 0, 0), k=1)

    assert results[0].chunk == PDF_CHUNK
    assert results[0].score == pytest.approx(1.0)


def test_add_embeddings_does_not_modify_the_callers_array(store: VectorStore):
    embeddings = np.array([[3, 0, 0, 0]], dtype=np.float32)

    store.add_embeddings([PDF_CHUNK], embeddings)

    np.testing.assert_array_equal(embeddings, [[3, 0, 0, 0]])


@pytest.mark.parametrize(
    "embeddings",
    [
        np.zeros((2, 4), dtype=np.float32),  # fewer rows than chunks
        np.zeros((3, 5), dtype=np.float32),  # wrong dimension
        np.zeros(4, dtype=np.float32),  # not a matrix
    ],
)
def test_add_embeddings_rejects_wrong_shape(store: VectorStore, embeddings: np.ndarray):
    with pytest.raises(ValueError, match="shape"):
        store.add_embeddings(ALL_CHUNKS, embeddings)

    assert len(store) == 0


# --- Search -----------------------------------------------------------------


def test_search_returns_search_results(filled_store: VectorStore):
    results = filled_store.search("pdf")

    assert all(isinstance(result, SearchResult) for result in results)
    assert all(isinstance(result.score, float) for result in results)


def test_search_puts_the_best_match_first(filled_store: VectorStore):
    results = filled_store.search("How does an embedding work?")

    assert results[0].chunk == EMBEDDING_CHUNK
    assert results[0].score == pytest.approx(1.0)


def test_search_returns_at_most_k_results(filled_store: VectorStore):
    assert len(filled_store.search("pdf", k=2)) == 2


def test_k_larger_than_index_returns_every_chunk_once(filled_store: VectorStore):
    results = filled_store.search("pdf", k=10)

    assert sorted(result.chunk.page_number for result in results) == [1, 2, 3]


def test_search_by_embedding_matches_text_search(filled_store: VectorStore):
    by_text = filled_store.search("chunk", k=3)
    by_vector = filled_store.search_by_embedding(unit(0, 1, 0, 0), k=3)

    assert [r.chunk for r in by_text] == [r.chunk for r in by_vector]


# --- Ranking ----------------------------------------------------------------


def test_results_are_ranked_by_cosine_similarity(store: VectorStore):
    names = ["opposite", "orthogonal", "close", "exact"]
    store.add_embeddings(
        [make_chunk(name, chunk_index=i) for i, name in enumerate(names)],
        np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0.8, 0.6, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
    )

    results = store.search_by_embedding(unit(1, 0, 0, 0), k=4)

    assert [r.chunk.text for r in results] == ["exact", "close", "orthogonal", "opposite"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.8, 0.0, -1.0], abs=1e-6)


def test_partial_keyword_match_ranks_between_full_and_no_match(filled_store: VectorStore):
    filled_store.add_chunks([make_chunk("A pdf chunk.", page_number=4)])

    results = filled_store.search("pdf", k=4)

    assert [r.chunk.page_number for r in results[:2]] == [1, 4]
    assert results[1].score == pytest.approx(1 / np.sqrt(2))
    assert results[2].score == pytest.approx(0.0)


def test_scores_are_in_descending_order(filled_store: VectorStore):
    scores = [r.score for r in filled_store.search("pdf chunk", k=3)]

    assert scores == sorted(scores, reverse=True)


# --- Metadata preservation --------------------------------------------------


def test_results_keep_full_chunk_metadata(store: VectorStore):
    chunk = make_chunk("The faiss index.", source="reports/q3.pdf", page_number=12, chunk_index=4)
    store.add_chunks([PDF_CHUNK, chunk])

    top = store.search("faiss", k=1)[0]

    assert top.chunk.source == "reports/q3.pdf"
    assert top.chunk.page_number == 12
    assert top.chunk.chunk_index == 4
    assert top.chunk.text == "The faiss index."


def test_chunks_added_in_separate_batches_map_back_correctly(store: VectorStore):
    first = [
        make_chunk("About pdf files.", source="a.pdf", page_number=1),
        make_chunk("About one chunk.", source="a.pdf", page_number=2),
    ]
    second = [
        make_chunk("About an embedding.", source="b.pdf", page_number=5, chunk_index=3),
        make_chunk("About faiss.", source="b.pdf", page_number=6),
    ]
    store.add_chunks(first)
    store.add_chunks(second)

    for chunk in first + second:
        assert store.search(chunk.text, k=1)[0].chunk == chunk


# --- Empty index and invalid input ------------------------------------------


def test_search_on_empty_store_returns_nothing(store: VectorStore):
    assert store.search("pdf") == []
    assert store.search_by_embedding(unit(1, 0, 0, 0)) == []


def test_adding_no_chunks_leaves_store_empty(store: VectorStore):
    store.add_chunks([])

    assert len(store) == 0
    assert store.search("pdf") == []


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3", None])
@pytest.mark.parametrize("fixture_name", ["store", "filled_store"])
def test_invalid_k_raises_value_error(request: pytest.FixtureRequest, fixture_name: str, k):
    vector_store: VectorStore = request.getfixturevalue(fixture_name)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        vector_store.search("pdf", k=k)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        vector_store.search_by_embedding(unit(1, 0, 0, 0), k=k)


def test_numpy_integer_k_is_accepted(filled_store: VectorStore):
    assert len(filled_store.search("pdf", k=np.int64(2))) == 2


@pytest.mark.parametrize("fixture_name", ["store", "filled_store"])
def test_empty_query_raises_value_error(request: pytest.FixtureRequest, fixture_name: str):
    vector_store: VectorStore = request.getfixturevalue(fixture_name)

    with pytest.raises(ValueError, match="empty"):
        vector_store.search("   ")


@pytest.mark.parametrize("query_embedding", [unit(1, 0, 0), unit(1, 0, 0, 0, 0), np.ones((1, 4), dtype=np.float32)])
def test_query_embedding_with_wrong_shape_raises_value_error(filled_store: VectorStore, query_embedding: np.ndarray):
    with pytest.raises(ValueError, match="shape"):
        filled_store.search_by_embedding(query_embedding)


# --- With the real embedding model ------------------------------------------


def test_real_model_retrieves_the_relevant_page(model: EmbeddingModel):
    pages = [
        PageData(page_number=1, text="Bake the bread at 220 degrees for thirty minutes.", source="cookbook.pdf"),
        PageData(page_number=2, text="The pipeline extracts text from PDF files and splits it into chunks.", source="contextforge.pdf"),
        PageData(page_number=3, text="The football match ended in a draw after extra time.", source="sports.pdf"),
    ]
    store = VectorStore(model)
    store.add_chunks(chunk_pages(pages))

    results = store.search("How is text extracted from PDF documents?", k=3)

    assert len(store) == 3
    assert (results[0].chunk.source, results[0].chunk.page_number) == ("contextforge.pdf", 2)
    assert results[0].score > results[1].score


def test_real_model_scores_are_cosine_similarities(model: EmbeddingModel):
    store = VectorStore(model)
    store.add_chunks([make_chunk("Semantic retrieval compares embeddings."), make_chunk("Bread needs flour.")])

    results = store.search("Semantic retrieval compares embeddings.", k=2)

    assert store.dimension == model.dimension
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert all(-1.0 <= r.score <= 1.0 + 1e-4 for r in results)
