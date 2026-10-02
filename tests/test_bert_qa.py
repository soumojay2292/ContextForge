from types import SimpleNamespace

import numpy as np
import pytest

from src.ingestion.chunking import DocumentChunk
from src.qa.bert_qa import BertQA, ExtractedAnswer, _window_ranges


@pytest.fixture(scope="module")
def qa() -> BertQA:
    # Loading the model is slow, so every test in this module shares one instance.
    return BertQA()


def make_chunk(text: str, source: str = "doc.pdf", page_number: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, page_number=page_number, chunk_index=chunk_index)


FACT_CHUNK = make_chunk(
    "ContextForge was built in 2024 by a research team in Berlin. It answers questions about PDF documents."
)
MIXED_CHUNKS = [
    make_chunk("Bake the bread at 220 degrees for thirty minutes.", source="cookbook.pdf", page_number=4),
    make_chunk("ContextForge was built in 2024 by a research team in Berlin.", source="contextforge.pdf", page_number=7, chunk_index=2),
    make_chunk("The football match ended in a draw after extra time.", source="sports.pdf", page_number=1),
]


# --- Answer extraction ------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where was ContextForge built?", "Berlin"),
        ("When was ContextForge built?", "2024"),
        ("What does ContextForge answer questions about?", "PDF documents"),
    ],
)
def test_extracts_the_answer_span(qa: BertQA, question: str, expected: str):
    answer = qa.answer(question, [FACT_CHUNK])

    assert isinstance(answer, ExtractedAnswer)
    assert answer.text == expected


def test_answer_offsets_point_into_the_chunk_text(qa: BertQA):
    answer = qa.answer("What does ContextForge answer questions about?", [FACT_CHUNK])

    # The model is uncased, but the answer is copied from the original text, keeping "PDF".
    assert FACT_CHUNK.text[answer.start_char : answer.end_char] == answer.text == "PDF documents"


def test_answer_score_is_a_probability_above_min_score(qa: BertQA):
    answer = qa.answer("Where was ContextForge built?", [FACT_CHUNK])

    assert qa.min_score <= answer.score <= 1.0


def test_finds_the_answer_in_the_right_chunk(qa: BertQA):
    assert qa.answer("Where was ContextForge built?", MIXED_CHUNKS).text == "Berlin"
    assert qa.answer("How long should the bread bake?", MIXED_CHUNKS).text == "thirty minutes"


@pytest.mark.parametrize("position", ["start", "middle", "end"])
def test_finds_answers_anywhere_in_a_chunk_longer_than_one_window(qa: BertQA, position: str):
    filler = " ".join(f"Sentence {i} is filler about nothing in particular." for i in range(60))
    fact = "The secret code is 4417."
    text = {"start": f"{fact} {filler} {filler}", "middle": f"{filler} {fact} {filler}", "end": f"{filler} {filler} {fact}"}[position]
    chunk = make_chunk(text)

    assert len(qa.tokenizer(text, add_special_tokens=False)["input_ids"]) > 2 * qa.max_length
    assert qa.answer("What is the secret code?", [chunk]).text == "4417"


# --- Metadata preservation --------------------------------------------------


def test_answer_keeps_the_source_chunk_metadata(qa: BertQA):
    answer = qa.answer("Where was ContextForge built?", MIXED_CHUNKS)

    assert answer.chunk == MIXED_CHUNKS[1]
    assert (answer.chunk.source, answer.chunk.page_number, answer.chunk.chunk_index) == ("contextforge.pdf", 7, 2)


def test_blank_chunks_are_skipped(qa: BertQA):
    chunks = [make_chunk("   ", page_number=1), make_chunk(FACT_CHUNK.text, page_number=2)]

    answer = qa.answer("Where was ContextForge built?", chunks)

    assert answer.text == "Berlin"
    assert answer.chunk.page_number == 2


# --- Invalid input ----------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises_value_error(qa: BertQA, question: str):
    with pytest.raises(ValueError, match="Question must not be empty"):
        qa.answer(question, [FACT_CHUNK])


@pytest.mark.parametrize("chunks", [[], [make_chunk("")], [make_chunk("  "), make_chunk("\n")]])
def test_empty_context_raises_value_error(qa: BertQA, chunks: list[DocumentChunk]):
    with pytest.raises(ValueError, match="Context must include"):
        qa.answer("Where was ContextForge built?", chunks)


def test_question_too_long_for_the_model_raises_value_error(qa: BertQA):
    question = "Where " + "really " * qa.max_length + "was ContextForge built?"

    with pytest.raises(ValueError, match="too long"):
        qa.answer(question, [FACT_CHUNK])


# --- No answer and low confidence -------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["What is the capital of Japan?", "How much does ContextForge cost?", "Who won the football match?"],
)
def test_unanswerable_question_returns_none(qa: BertQA, question: str):
    assert qa.answer(question, [FACT_CHUNK]) is None


def test_unanswerable_across_several_chunks_returns_none(qa: BertQA):
    assert qa.answer("What is the capital of Japan?", MIXED_CHUNKS) is None


def test_answer_below_min_score_returns_none(qa: BertQA, monkeypatch: pytest.MonkeyPatch):
    answer = qa.answer("Where was ContextForge built?", [FACT_CHUNK])
    monkeypatch.setattr(qa, "min_score", answer.score + 0.001)

    assert qa.answer("Where was ContextForge built?", [FACT_CHUNK]) is None


# --- Span selection (synthetic logits, no model needed) ---------------------


def best_span(start_logits, end_logits, context_positions, max_answer_tokens=30, length=8):
    context_mask = np.zeros(length, dtype=bool)
    context_mask[list(context_positions)] = True
    fake_qa = SimpleNamespace(max_answer_tokens=max_answer_tokens)
    return BertQA._best_span(fake_qa, np.array(start_logits, float), np.array(end_logits, float), context_mask)


def logits_peaking_at(*positions: int, length: int = 8) -> list[float]:
    logits = [0.0] * length
    for position in positions:
        logits[position] = 10.0
    return logits


def test_best_span_uses_the_most_likely_start_and_end():
    span = best_span(logits_peaking_at(4), logits_peaking_at(5), context_positions=range(3, 7))

    assert span[:2] == (4, 5)
    assert 0.5 < span[2] <= 1.0


def test_best_span_never_ends_before_it_starts():
    start, end, _ = best_span(logits_peaking_at(5), logits_peaking_at(4), context_positions=range(3, 7))

    assert start <= end


def test_best_span_respects_max_answer_tokens():
    start, end, _ = best_span(logits_peaking_at(3), logits_peaking_at(6), context_positions=range(3, 7), max_answer_tokens=2)

    assert end - start + 1 <= 2


def test_best_span_ignores_tokens_outside_the_context():
    # Position 1 is a question token: despite the highest logits it cannot be the answer.
    logits = [0.0, 10.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]

    span = best_span(logits, logits, context_positions=range(3, 7))

    assert span[:2] == (4, 4)


def test_best_span_returns_none_when_cls_is_most_likely():
    assert best_span(logits_peaking_at(0), logits_peaking_at(0), context_positions=range(3, 7)) is None


# --- Windowing --------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (0, [(0, 0)]),
        (5, [(0, 5)]),
        (10, [(0, 10)]),
        (25, [(0, 10), (6, 16), (12, 22), (18, 25)]),
    ],
)
def test_window_ranges(num_tokens: int, expected: list[tuple[int, int]]):
    assert list(_window_ranges(num_tokens, window_size=10, stride=4)) == expected


def test_window_ranges_cover_every_token_with_stride_overlap():
    windows = list(_window_ranges(1000, window_size=371, stride=128))

    assert windows[0][0] == 0 and windows[-1][1] == 1000
    assert all(previous_end - start == 128 for (_, previous_end), (start, _) in zip(windows, windows[1:]))
