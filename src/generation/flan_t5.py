"""Generate answers grounded in retrieved document chunks with FLAN-T5."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.ingestion.chunking import DocumentChunk

DEFAULT_GENERATION_MODEL_NAME = "google/flan-t5-small"

# Returned instead of the model's output when it says the context has no answer.
NOT_AVAILABLE_ANSWER = "The answer is not available in the provided documents."

PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    'If the context does not contain the answer, reply "not available".\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Normalized model outputs that mean "no answer". "unanswerable" is what FLAN-T5
# was trained to say for unanswerable reading-comprehension questions.
_NOT_AVAILABLE_REPLIES = frozenset({"not available", "unanswerable"})


@dataclass(frozen=True)
class GeneratedAnswer:
    """An answer generated from retrieved chunks.

    Attributes:
        text: The generated answer, or NOT_AVAILABLE_ANSWER.
        sources: The chunks included in the prompt, most relevant first.
    """

    text: str
    sources: tuple[DocumentChunk, ...]

    @property
    def is_available(self) -> bool:
        """Whether the model found an answer in the context."""
        return self.text != NOT_AVAILABLE_ANSWER


def format_context(chunks: Sequence[DocumentChunk]) -> str:
    """Number each chunk's text, in order, separated by blank lines."""
    return "\n\n".join(f"[{number}] {chunk.text.strip()}" for number, chunk in enumerate(chunks, start=1))


def build_prompt(question: str, chunks: Sequence[DocumentChunk]) -> str:
    """Build the instruction prompt that asks for an answer from the chunks only."""
    return PROMPT_TEMPLATE.format(context=format_context(chunks), question=question.strip())


class FlanT5Generator:
    """Answers questions from retrieved chunks using a FLAN-T5 model.

    The model is loaded on the first call to generate(), not when the generator
    is created.

    Args:
        model_name: Hugging Face name or local path of a sequence-to-sequence model.
        device: Device to run on, e.g. "cpu" or "cuda". None picks automatically.
        max_input_tokens: Longest prompt, in tokens. FLAN-T5 was trained on 512.
        max_new_tokens: Longest answer, in tokens.
        num_beams: Beam search width; 1 means greedy decoding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATION_MODEL_NAME,
        device: str | None = None,
        max_input_tokens: int = 512,
        max_new_tokens: int = 64,
        num_beams: int = 1,
    ):
        self.model_name = model_name
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, question: str, chunks: Sequence[DocumentChunk]) -> GeneratedAnswer:
        """Generate a concise answer to question from the chunks.

        Chunks should be ordered most relevant first. They are added to the
        prompt in that order until it is full, and only those that fit are
        returned as sources. If even the first chunk does not fit, its
        beginning is used.

        Raises:
            ValueError: If question is empty or too long, or no chunk contains any text.
        """
        if not question.strip():
            raise ValueError("Question must not be empty")
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            raise ValueError("Context must include at least one chunk with text")

        self._load()
        prompt_chunks, sources = self._fit_chunks(question, chunks)
        output = self._generate_text(build_prompt(question, prompt_chunks))
        text = NOT_AVAILABLE_ANSWER if _says_not_available(output) else output
        return GeneratedAnswer(text=text, sources=tuple(sources))

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imported here, not at module level, because importing PyTorch is slow.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self._device).eval()

    def _fit_chunks(
        self, question: str, chunks: list[DocumentChunk]
    ) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        """Choose the chunks to put in the prompt so it fits in max_input_tokens.

        Returns:
            (chunks to format into the prompt, original chunks they came from).
            These differ only when the first chunk had to be shortened.
        """
        fitted: list[DocumentChunk] = []
        for chunk in chunks:
            if self._count_tokens(build_prompt(question, [*fitted, chunk])) > self.max_input_tokens:
                break
            fitted.append(chunk)
        if fitted:
            return fitted, fitted

        # Even the most relevant chunk is too long on its own: keep as much of
        # its beginning as fits. Re-tokenizing a cut text can add a token or two,
        # so shrink the budget by any overflow until the prompt fits.
        first = chunks[0]
        budget = self.max_input_tokens - self._count_tokens(build_prompt(question, [replace(first, text="")]))
        while budget > 0:
            shortened = replace(first, text=self._truncate(first.text, budget))
            overflow = self._count_tokens(build_prompt(question, [shortened])) - self.max_input_tokens
            if overflow <= 0:
                return [shortened], [first]
            budget -= overflow
        raise ValueError(f"Question is too long for max_input_tokens={self.max_input_tokens}")

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text)["input_ids"])

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Cut text after its first max_tokens tokens."""
        offsets = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        return text if len(offsets) <= max_tokens else text[: offsets[max_tokens - 1][1]]

    def _generate_text(self, prompt: str) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs.to(self._device),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _says_not_available(output: str) -> bool:
    # Drop punctuation and quotes only: answers can be pure numbers, like "2024".
    normalized = re.sub(r"[^\w\s]", "", output).strip().lower()
    return not normalized or normalized in _NOT_AVAILABLE_REPLIES
