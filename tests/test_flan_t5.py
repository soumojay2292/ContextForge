import pytest

from src.generation.flan_t5 import (
    DEFAULT_GENERATION_MODEL_NAME,
    NOT_AVAILABLE_ANSWER,
    FlanT5Generator,
    GeneratedAnswer,
    _says_not_available,
    build_prompt,
    format_context,
)
from src.ingestion.chunking import DocumentChunk


def make_chunk(text: str, source: str = "doc.pdf", page_number: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, page_number=page_number, chunk_index=chunk_index)


FACT_CHUNKS = [
    make_chunk("ContextForge was built in 2024 by a research team in Berlin. It answers questions about PDF documents.")
]
MIXED_CHUNKS = [
    make_chunk("Bake the bread at 220 degrees for thirty minutes.", source="cookbook.pdf", page_number=4),
    make_chunk("ContextForge was built in 2024 by a research team in Berlin.", source="contextforge.pdf", page_number=7, chunk_index=2),
    make_chunk("The football match ended in a draw after extra time.", source="sports.pdf", page_number=1),
]
DOCUMENT_CHUNKS = [
    make_chunk(
        "ContextForge Test Document 1. Introduction ContextForge is an intelligent document research assistant "
        "designed to retrieve relevant information from uploaded documents and provide grounded answers.",
        source="test_document.pdf", chunk_index=0,
    ),
    make_chunk(
        "2. Document Processing The document processing pipeline extracts text from PDF files, preserves page "
        "information, cleans the extracted text, and divides the content into meaningful chunks.",
        source="test_document.pdf", chunk_index=1,
    ),
    make_chunk(
        "3. Semantic Retrieval Semantic retrieval represents text as numerical embeddings and compares a user "
        "query with document chunks. Relevant chunks can then be supplied to a question answering or generation component.",
        source="test_document.pdf", chunk_index=2,
    ),
    make_chunk(
        "4. Testing A reliable document assistant should test successful document loading as well as invalid files, "
        "empty documents, page ordering, and preservation of source information.",
        source="test_document.pdf", chunk_index=3,
    ),
]


@pytest.fixture(scope="module")
def generator() -> FlanT5Generator:
    # Loading the model is slow, so every test in this module shares one instance.
    return FlanT5Generator()


# --- Context formatting and prompt ------------------------------------------


def test_format_context_numbers_chunks_in_order():
    context = format_context([make_chunk("First."), make_chunk("  Second.  "), make_chunk("Third.")])

    assert context == "[1] First.\n\n[2] Second.\n\n[3] Third."


def test_prompt_tells_the_model_to_use_only_the_context():
    prompt = build_prompt("Where was it built?", FACT_CHUNKS)

    assert "using only the context" in prompt
    assert 'If the context does not contain the answer, reply "not available"' in prompt


def test_prompt_puts_context_before_question_and_ends_with_answer_cue():
    prompt = build_prompt("  Where was it built?  ", MIXED_CHUNKS)

    assert prompt.index("[1] Bake the bread") < prompt.index("[3] The football") < prompt.index("Question: Where was it built?\n")
    assert prompt.endswith("Answer:")


# --- Not-available detection ------------------------------------------------


@pytest.mark.parametrize("output", ["not available", "Not available.", '"not available"', "unanswerable", "", "  "])
def test_says_not_available_recognizes_refusals(output: str):
    assert _says_not_available(output)


@pytest.mark.parametrize("output", ["2024", "Berlin", "30 minutes", "not available in Berlin"])
def test_says_not_available_keeps_real_answers(output: str):
    assert not _says_not_available(output)


def test_generated_answer_reports_availability():
    assert GeneratedAnswer(text="Berlin", sources=()).is_available
    assert not GeneratedAnswer(text=NOT_AVAILABLE_ANSWER, sources=()).is_available


# --- Lazy, configurable loading ---------------------------------------------


def test_model_is_not_loaded_until_first_generation():
    generator = FlanT5Generator(model_name="some/other-model", device="cpu", max_input_tokens=256, max_new_tokens=16, num_beams=4)

    assert not generator.is_loaded
    assert (generator.model_name, generator.device, generator.max_input_tokens, generator.max_new_tokens, generator.num_beams) == (
        "some/other-model", "cpu", 256, 16, 4,
    )


def test_default_model_is_flan_t5_small():
    assert FlanT5Generator().model_name == DEFAULT_GENERATION_MODEL_NAME == "google/flan-t5-small"


# --- Invalid input (checked before the model loads) -------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises_value_error_without_loading(question: str):
    generator = FlanT5Generator()

    with pytest.raises(ValueError, match="Question must not be empty"):
        generator.generate(question, FACT_CHUNKS)
    assert not generator.is_loaded


@pytest.mark.parametrize("chunks", [[], [make_chunk("")], [make_chunk("  "), make_chunk("\n")]])
def test_empty_chunks_raise_value_error_without_loading(chunks: list[DocumentChunk]):
    generator = FlanT5Generator()

    with pytest.raises(ValueError, match="Context must include"):
        generator.generate("Where was ContextForge built?", chunks)
    assert not generator.is_loaded


def test_question_too_long_for_the_prompt_raises_value_error(generator: FlanT5Generator):
    question = "Where " + "really " * generator.max_input_tokens + "was ContextForge built?"

    with pytest.raises(ValueError, match="too long"):
        generator.generate(question, FACT_CHUNKS)


# --- Generation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "chunks", "expected"),
    [
        ("Where was ContextForge built?", FACT_CHUNKS, "Berlin"),
        ("When was ContextForge built?", FACT_CHUNKS, "2024"),
        ("Who built ContextForge?", FACT_CHUNKS, "research team"),
        ("At what temperature should the bread bake?", MIXED_CHUNKS, "220"),
        ("How did the football match end?", MIXED_CHUNKS, "draw"),
        ("What does semantic retrieval represent text as?", DOCUMENT_CHUNKS, "numerical embeddings"),
        ("What does the pipeline extract text from?", DOCUMENT_CHUNKS, "PDF"),
    ],
)
def test_generates_answer_from_context(generator: FlanT5Generator, question: str, chunks, expected: str):
    answer = generator.generate(question, chunks)

    assert generator.is_loaded
    assert answer.is_available
    assert expected.lower() in answer.text.lower()


@pytest.mark.parametrize(
    ("question", "chunks"),
    [
        ("How much does ContextForge cost?", FACT_CHUNKS),
        ("Who is the CEO of ContextForge?", FACT_CHUNKS),
        ("Who scored the winning goal?", MIXED_CHUNKS),
        ("What is the capital of Japan?", MIXED_CHUNKS),
        ("What programming language is ContextForge written in?", DOCUMENT_CHUNKS),
    ],
)
def test_says_answer_is_not_available_when_context_lacks_it(generator: FlanT5Generator, question: str, chunks):
    answer = generator.generate(question, chunks)

    assert answer.text == NOT_AVAILABLE_ANSWER
    assert not answer.is_available


def test_answer_length_is_limited_by_max_new_tokens(generator: FlanT5Generator, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generator, "max_new_tokens", 4)

    answer = generator.generate("What is ContextForge?", DOCUMENT_CHUNKS)

    assert len(generator._tokenizer(answer.text, add_special_tokens=False)["input_ids"]) <= 4


# --- Source preservation ----------------------------------------------------


def test_sources_are_the_supplied_chunks_in_order(generator: FlanT5Generator):
    answer = generator.generate("Where was ContextForge built?", MIXED_CHUNKS)

    assert answer.sources == tuple(MIXED_CHUNKS)
    assert [(c.source, c.page_number, c.chunk_index) for c in answer.sources] == [
        ("cookbook.pdf", 4, 0), ("contextforge.pdf", 7, 2), ("sports.pdf", 1, 0),
    ]


def test_sources_are_kept_when_the_answer_is_not_available(generator: FlanT5Generator):
    answer = generator.generate("Who scored the winning goal?", MIXED_CHUNKS)

    assert answer.sources == tuple(MIXED_CHUNKS)


def test_blank_chunks_are_left_out_of_sources(generator: FlanT5Generator):
    answer = generator.generate("Where was ContextForge built?", [make_chunk("   ", page_number=1), *FACT_CHUNKS])

    assert answer.sources == tuple(FACT_CHUNKS)


def test_only_chunks_that_fit_the_prompt_are_sources(generator: FlanT5Generator):
    chunks = [
        make_chunk(" ".join(f"Point {i} of section {section} says nothing new." for i in range(20)), page_number=section)
        for section in range(1, 6)
    ]

    answer = generator.generate("What do the sections say?", chunks)
    prompt = build_prompt("What do the sections say?", answer.sources)

    assert 0 < len(answer.sources) < len(chunks)
    assert answer.sources == tuple(chunks[: len(answer.sources)])
    assert generator._count_tokens(prompt) <= generator.max_input_tokens
    assert generator._count_tokens(build_prompt("What do the sections say?", chunks[: len(answer.sources) + 1])) > generator.max_input_tokens


def test_oversized_first_chunk_is_shortened_but_kept_as_source(generator: FlanT5Generator):
    huge = make_chunk("The report covers 2024. " + " ".join(f"Filler sentence {i} adds nothing." for i in range(300)), page_number=9)

    prompt_chunks, sources = generator._fit_chunks("What year does the report cover?", [huge])
    answer = generator.generate("What year does the report cover?", [huge])

    assert generator._count_tokens(build_prompt("What year does the report cover?", prompt_chunks)) <= generator.max_input_tokens
    assert prompt_chunks[0].text.startswith("The report covers 2024.")
    assert len(prompt_chunks[0].text) < len(huge.text)
    assert sources == [huge]
    assert answer.sources == (huge,)
