"""Turn document chunks and queries into vectors with a Sentence Transformers model."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.chunking import DocumentChunk

# Small (22M parameters) and fast on CPU; produces 384-dimensional embeddings.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Embeds document chunks and queries into the same vector space.

    Embeddings are float32 and L2-normalized, so the dot product of two
    embeddings equals their cosine similarity.

    Args:
        model_name: Hugging Face name or local path of a Sentence Transformers model.
        device: Device to run on, e.g. "cpu" or "cuda". None picks automatically.
        batch_size: Number of chunks encoded at once.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self._model.get_embedding_dimension()

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        """Embed the text of each chunk.

        Returns:
            Array of shape (len(chunks), dimension), where row i is the embedding
            of chunks[i]. Empty input gives an array of shape (0, dimension).
        """
        if not chunks:
            # The model returns shape (0,) for empty input; keep the column dimension.
            return np.empty((0, self.dimension), dtype=np.float32)

        # encode_document/encode_query apply the model's document/query prompts
        # when it defines them (MiniLM does not, but models like E5 do).
        embeddings = self._model.encode_document(
            [chunk.text for chunk in chunks],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query.

        Returns:
            Array of shape (dimension,).

        Raises:
            ValueError: If the query is empty or only whitespace.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        embedding = self._model.encode_query(query, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embedding, dtype=np.float32)
