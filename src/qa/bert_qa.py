"""Extract answers to questions from document chunks with a BERT model fine-tuned for QA."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from src.ingestion.chunking import DocumentChunk

# BERT architecture (12 layers, hidden size 384) fine-tuned on SQuAD 2.0. SQuAD 2.0
# includes unanswerable questions, so the model can predict that there is no answer.
DEFAULT_QA_MODEL_NAME = "deepset/minilm-uncased-squad2"


@dataclass(frozen=True)
class ExtractedAnswer:
    """An answer span copied out of one chunk.

    Attributes:
        text: The answer, exactly as it appears in the chunk.
        score: Model confidence from 0 to 1 (start probability x end probability).
        chunk: The chunk the answer came from, including its source and page number.
        start_char: Index in chunk.text where the answer starts.
        end_char: Index in chunk.text just past the end of the answer.
    """

    text: str
    score: float
    chunk: DocumentChunk
    start_char: int
    end_char: int


@dataclass(frozen=True)
class _Window:
    """A run of one chunk's tokens short enough to fit in a single model input."""

    chunk: DocumentChunk
    token_ids: list[int]
    offsets: list[tuple[int, int]]  # (start, end) character span of each token in chunk.text


class BertQA:
    """Finds the span of text in a set of chunks that best answers a question.

    Works with BERT-style extractive QA models, whose input is
    "[CLS] question [SEP] context [SEP]".

    Args:
        model_name: Hugging Face name or local path of an extractive QA model.
        device: Device to run on, e.g. "cpu" or "cuda". None picks automatically.
        max_length: Maximum tokens per model input (question + context + special tokens).
        doc_stride: Tokens shared by neighbouring windows when a chunk is too long
            for one input, so an answer near a window edge is not cut in half.
        max_answer_tokens: Longest answer, in tokens, that will be returned.
        min_score: Answers scoring below this are treated as no answer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QA_MODEL_NAME,
        device: str | None = None,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_tokens: int = 30,
        min_score: float = 0.1,
    ):
        # Imported here, not at module level, because importing PyTorch is slow.
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_tokens = max_answer_tokens
        self.min_score = min_score
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            # Only fast tokenizers report character offsets, which map answers back to the text.
            raise ValueError(f"{model_name} has no fast tokenizer")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()

    def answer(self, question: str, chunks: Sequence[DocumentChunk]) -> ExtractedAnswer | None:
        """Return the best answer to question found in chunks.

        Returns:
            The highest-scoring answer across all chunks, or None if the model
            predicts that no chunk answers the question, or if the best answer
            scores below min_score.

        Raises:
            ValueError: If question is empty or too long, or no chunk contains any text.
        """
        if not question.strip():
            raise ValueError("Question must not be empty")
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            raise ValueError("Context must include at least one chunk with text")

        question_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        # [CLS] + question + [SEP] come before the context and [SEP] after it.
        context_start = len(question_ids) + 2
        window_size = self.max_length - context_start - 1
        if window_size <= self.doc_stride:
            raise ValueError(f"Question is too long ({len(question_ids)} tokens) for max_length={self.max_length}")

        contexts = self.tokenizer(
            [chunk.text for chunk in chunks], add_special_tokens=False, return_offsets_mapping=True
        )
        windows = [
            _Window(chunk=chunk, token_ids=token_ids[start:end], offsets=offsets[start:end])
            for chunk, token_ids, offsets in zip(chunks, contexts["input_ids"], contexts["offset_mapping"])
            for start, end in _window_ranges(len(token_ids), window_size, self.doc_stride)
        ]
        start_logits, end_logits = self._predict(question_ids, windows)

        best: ExtractedAnswer | None = None
        for window, window_start_logits, window_end_logits in zip(windows, start_logits, end_logits):
            context_mask = np.zeros(len(window_start_logits), dtype=bool)
            context_mask[context_start : context_start + len(window.token_ids)] = True
            span = self._best_span(window_start_logits, window_end_logits, context_mask)
            if span is None:
                continue

            start_token, end_token, score = span
            if best is None or score > best.score:
                start_char = window.offsets[start_token - context_start][0]
                end_char = window.offsets[end_token - context_start][1]
                best = ExtractedAnswer(
                    text=window.chunk.text[start_char:end_char],
                    score=score,
                    chunk=window.chunk,
                    start_char=start_char,
                    end_char=end_char,
                )

        if best is None or best.score < self.min_score:
            return None
        return best

    def _predict(self, question_ids: list[int], windows: list[_Window]) -> tuple[np.ndarray, np.ndarray]:
        """Run the model on every window; return start and end logits, one row per window."""
        import torch

        rows = [
            [self.tokenizer.cls_token_id, *question_ids, self.tokenizer.sep_token_id, *window.token_ids, self.tokenizer.sep_token_id]
            for window in windows
        ]
        width = max(len(row) for row in rows)
        input_ids = np.full((len(rows), width), self.tokenizer.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        # Segment ids tell BERT which tokens are the question (0) and which the context (1).
        token_type_ids = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row
            attention_mask[i, : len(row)] = 1
            token_type_ids[i, len(question_ids) + 2 : len(row)] = 1

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        inputs = {
            name: torch.from_numpy(values).to(self.device)
            for name, values in inputs.items()
            if name in self.tokenizer.model_input_names  # e.g. DistilBERT takes no token_type_ids
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy()

    def _best_span(
        self, start_logits: np.ndarray, end_logits: np.ndarray, context_mask: np.ndarray
    ) -> tuple[int, int, float] | None:
        """Return (start token, end token, score) of the best answer in one window.

        Returns None if the model prefers "no answer" to every span in the window.
        """
        # Probabilities cover the context tokens plus [CLS] at position 0, which
        # SQuAD 2.0 models point at to say the window holds no answer.
        candidates = context_mask.copy()
        candidates[0] = True
        start_probs = _softmax(np.where(candidates, start_logits, -np.inf))
        end_probs = _softmax(np.where(candidates, end_logits, -np.inf))
        no_answer_score = start_probs[0] * end_probs[0]

        # Score every span that lies inside the context, ends at or after its
        # start, and is at most max_answer_tokens long.
        length = len(context_mask)
        valid = np.outer(context_mask, context_mask)
        valid &= np.triu(np.ones((length, length), dtype=bool))
        valid &= np.tril(np.ones((length, length), dtype=bool), k=self.max_answer_tokens - 1)
        span_scores = np.where(valid, np.outer(start_probs, end_probs), 0.0)

        start, end = np.unravel_index(np.argmax(span_scores), span_scores.shape)
        score = float(span_scores[start, end])
        if score <= no_answer_score:
            return None
        return int(start), int(end), score


def _window_ranges(num_tokens: int, window_size: int, stride: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) token ranges covering num_tokens, each overlapping the previous by stride."""
    start = 0
    while True:
        end = min(start + window_size, num_tokens)
        yield start, end
        if end == num_tokens:
            return
        start = end - stride


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()
