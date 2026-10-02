import numpy as np
import pytest

from src.ingestion.chunking import DocumentChunk, chunk_pages
from src.ingestion.pdf_loader import PageData
from src.retrieval.embeddings import EmbeddingModel

MINILM_DIMENSION = 384


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    # Loading the model is slow, so every test in this module shares one instance.
    return EmbeddingModel()


def make_chunks(*texts: str) -> list[DocumentChunk]:
    return [
        DocumentChunk(text=text, source="doc.pdf", page_number=1, chunk_index=index)
        for index, text in enumerate(texts)
    ]


# --- Shape ------------------------------------------------------------------


def test_dimension_matches_the_model(model: EmbeddingModel):
    assert model.dimension == MINILM_DIMENSION


def test_chunk_embeddings_have_one_row_per_chunk(model: EmbeddingModel):
    embeddings = model.embed_chunks(make_chunks("First chunk.", "Second chunk.", "Third chunk."))

    assert embeddings.shape == (3, model.dimension)
    assert embeddings.dtype == np.float32


def test_single_chunk_is_still_two_dimensional(model: EmbeddingModel):
    embeddings = model.embed_chunks(make_chunks("Only chunk."))

    assert embeddings.shape == (1, model.dimension)


def test_query_embedding_is_a_vector_of_model_dimension(model: EmbeddingModel):
    embedding = model.embed_query("What does ContextForge do?")

    assert embedding.shape == (model.dimension,)
    assert embedding.dtype == np.float32


def test_embeddings_are_unit_length(model: EmbeddingModel):
    chunk_embeddings = model.embed_chunks(make_chunks("Some text.", "Other text."))
    query_embedding = model.embed_query("A question?")

    np.testing.assert_allclose(np.linalg.norm(chunk_embeddings, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(query_embedding), 1.0, rtol=1e-5)


# --- Consistency ------------------------------------------------------------


def test_same_text_gives_same_embedding(model: EmbeddingModel):
    first = model.embed_chunks(make_chunks("Repeatable text."))
    second = model.embed_chunks(make_chunks("Repeatable text."))

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_chunk_embedding_does_not_depend_on_its_batch(model: EmbeddingModel):
    alone = model.embed_chunks(make_chunks("Target sentence."))
    in_batch = model.embed_chunks(make_chunks("A much longer neighbouring sentence with many more words.", "Target sentence."))

    np.testing.assert_allclose(alone[0], in_batch[1], atol=1e-4)


def test_query_and_chunk_share_a_vector_space(model: EmbeddingModel):
    text = "Semantic retrieval compares embeddings."

    chunk_embedding = model.embed_chunks(make_chunks(text))[0]
    query_embedding = model.embed_query(text)

    assert float(chunk_embedding @ query_embedding) == pytest.approx(1.0, abs=1e-4)


# --- Empty input ------------------------------------------------------------


def test_no_chunks_gives_empty_array_with_model_dimension(model: EmbeddingModel):
    embeddings = model.embed_chunks([])

    assert embeddings.shape == (0, model.dimension)
    assert embeddings.dtype == np.float32


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_raises_value_error(model: EmbeddingModel, query: str):
    with pytest.raises(ValueError, match="empty"):
        model.embed_query(query)


# --- Multiple chunks --------------------------------------------------------


def test_rows_follow_chunk_order(model: EmbeddingModel):
    forward = model.embed_chunks(make_chunks("Apples are fruit.", "Rust is a language."))
    reverse = model.embed_chunks(make_chunks("Rust is a language.", "Apples are fruit."))

    np.testing.assert_allclose(forward, reverse[::-1], atol=1e-4)


def test_more_chunks_than_batch_size(model: EmbeddingModel):
    chunks = make_chunks(*(f"Chunk number {i} talks about topic {i}." for i in range(model.batch_size + 8)))

    embeddings = model.embed_chunks(chunks)

    assert embeddings.shape == (len(chunks), model.dimension)


def test_query_is_closest_to_the_relevant_chunk(model: EmbeddingModel):
    chunks = make_chunks(
        "Bake the bread at 220 degrees for thirty minutes.",
        "The pipeline extracts text from PDF files and splits it into chunks.",
        "The football match ended in a draw after extra time.",
    )

    scores = model.embed_chunks(chunks) @ model.embed_query("How is text extracted from PDF documents?")

    assert int(np.argmax(scores)) == 1


def test_embeds_chunks_produced_by_the_chunker(model: EmbeddingModel):
    pages = [
        PageData(page_number=1, text="ContextForge loads PDFs. It keeps page numbers.", source="doc.pdf"),
        PageData(page_number=2, text="Chunks are embedded next.", source="doc.pdf"),
    ]
    chunks = chunk_pages(pages, chunk_size=30, chunk_overlap=0)

    embeddings = model.embed_chunks(chunks)

    assert len(chunks) == 3
    assert embeddings.shape == (3, model.dimension)
