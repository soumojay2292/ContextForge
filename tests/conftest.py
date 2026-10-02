import pytest

from src.retrieval.embeddings import EmbeddingModel


@pytest.fixture(scope="session")
def model() -> EmbeddingModel:
    # Loading the model is slow and memory-hungry, so all tests share one instance.
    return EmbeddingModel()
