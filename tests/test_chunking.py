from pathlib import Path

import pymupdf
import pytest

from src.ingestion.chunking import (
    DocumentChunk,
    chunk_pages,
    chunk_text,
    clean_text,
    split_sentences,
)
from src.ingestion.pdf_loader import PageData, load_pdf

# 20 sentences of 21 characters each: "Sentence 00 is short."
SENTENCES = [f"Sentence {i:02d} is short." for i in range(20)]
TEXT = " ".join(SENTENCES)


# --- Cleaning ---------------------------------------------------------------


def test_clean_text_collapses_whitespace_and_newlines():
    assert clean_text("  Hello\n\nworld.\t  This   is\r\na test.  ") == "Hello world. This is a test."


def test_clean_text_rejoins_words_hyphenated_across_lines():
    assert clean_text("The infor-\nmation was retrieved.") == "The information was retrieved."


def test_clean_text_keeps_hyphens_within_a_line():
    assert clean_text("A state-of-the-art system.") == "A state-of-the-art system."


def test_clean_text_normalizes_ligatures():
    assert clean_text("The ﬁrst ﬂow") == "The first flow"


def test_clean_text_removes_invisible_characters():
    assert clean_text("Zero​width\x00 and soft­hyphen") == "Zerowidth and softhyphen"


@pytest.mark.parametrize("text", ["", "   ", "\n\t\n"])
def test_clean_text_of_blank_input_is_empty(text: str):
    assert clean_text(text) == ""


# --- Sentence splitting -----------------------------------------------------


def test_split_sentences_on_terminal_punctuation():
    assert split_sentences("It works. Does it? Yes!") == ["It works.", "Does it?", "Yes!"]


def test_split_sentences_ignores_abbreviations_and_decimals():
    text = "Dr. Smith measured 3.14 units, e.g. in Fig. 2 of the paper. It matched."

    assert split_sentences(text) == [
        "Dr. Smith measured 3.14 units, e.g. in Fig. 2 of the paper.",
        "It matched.",
    ]


def test_split_sentences_keeps_numbered_heading_with_its_text():
    text = "Answers are grounded. 2. Document Processing The pipeline extracts text."

    assert split_sentences(text) == [
        "Answers are grounded.",
        "2. Document Processing The pipeline extracts text.",
    ]


def test_split_sentences_of_empty_text_is_empty():
    assert split_sentences("") == []


# --- Chunk size and overlap -------------------------------------------------


@pytest.mark.parametrize(("chunk_size", "chunk_overlap"), [(50, 0), (100, 30), (200, 80)])
def test_chunks_never_exceed_chunk_size(chunk_size: int, chunk_overlap: int):
    chunks = chunk_text(TEXT, chunk_size, chunk_overlap)

    assert chunks
    assert all(len(chunk) <= chunk_size for chunk in chunks)


def test_chunks_are_made_of_whole_sentences():
    chunks = chunk_text(TEXT, chunk_size=100, chunk_overlap=30)

    for chunk in chunks:
        assert all(sentence in SENTENCES for sentence in split_sentences(chunk))


def test_without_overlap_chunks_reassemble_the_original_text():
    chunks = chunk_text(TEXT, chunk_size=100, chunk_overlap=0)

    assert len(chunks) > 1
    assert " ".join(chunks) == TEXT


def test_overlap_repeats_trailing_sentences_of_previous_chunk():
    # 21-char sentences: a 100-char chunk holds 4, and 30 chars of overlap carries 1.
    chunks = chunk_text(TEXT, chunk_size=100, chunk_overlap=30)

    for previous, current in zip(chunks, chunks[1:]):
        assert split_sentences(previous)[-1] == split_sentences(current)[0]


def test_overlap_never_exceeds_chunk_overlap():
    chunk_overlap = 50
    chunks = chunk_text(TEXT, chunk_size=100, chunk_overlap=chunk_overlap)

    for previous, current in zip(chunks, chunks[1:]):
        previous_sentences = split_sentences(previous)
        shared = [s for s in split_sentences(current) if s in previous_sentences]
        assert len(" ".join(shared)) <= chunk_overlap


def test_every_sentence_appears_in_some_chunk():
    chunks = chunk_text(TEXT, chunk_size=100, chunk_overlap=30)

    assert {s for chunk in chunks for s in split_sentences(chunk)} == set(SENTENCES)


def test_sentence_longer_than_chunk_size_is_split_between_words():
    long_sentence = " ".join(["word"] * 60) + "."

    chunks = chunk_text(long_sentence, chunk_size=50, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert " ".join(chunks) == long_sentence


@pytest.mark.parametrize(("chunk_size", "chunk_overlap"), [(0, 0), (-10, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_chunk_parameters_raise_value_error(chunk_size: int, chunk_overlap: int):
    with pytest.raises(ValueError):
        chunk_text(TEXT, chunk_size, chunk_overlap)
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size, chunk_overlap)


# --- Empty text -------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_chunk_text_of_empty_text_is_empty(text: str):
    assert chunk_text(text) == []


def test_chunk_pages_with_no_pages_is_empty():
    assert chunk_pages([]) == []


def test_empty_page_produces_no_chunks():
    pages = [PageData(page_number=1, text="  \n ", source="doc.pdf")]

    assert chunk_pages(pages) == []


# --- Metadata and multiple pages --------------------------------------------


def test_chunks_keep_source_and_page_number():
    pages = [PageData(page_number=7, text=TEXT, source="reports/annual.pdf")]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.source == "reports/annual.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)


def test_chunk_pages_cleans_page_text():
    pages = [PageData(page_number=1, text="The infor-\nmation is\n\nhere.", source="doc.pdf")]

    assert chunk_pages(pages)[0].text == "The information is here."


def test_chunk_index_restarts_on_each_page():
    pages = [
        PageData(page_number=1, text=TEXT, source="doc.pdf"),
        PageData(page_number=2, text=TEXT, source="doc.pdf"),
    ]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=0)

    for page_number in (1, 2):
        indexes = [c.chunk_index for c in chunks if c.page_number == page_number]
        assert indexes == list(range(len(indexes)))


def test_chunks_from_multiple_pages_stay_on_their_own_page():
    pages = [
        PageData(page_number=1, text="Alpha is on page one. It has two sentences.", source="doc.pdf"),
        PageData(page_number=2, text="", source="doc.pdf"),
        PageData(page_number=3, text="Gamma is on page three.", source="doc.pdf"),
    ]

    chunks = chunk_pages(pages)

    assert [(c.page_number, c.text) for c in chunks] == [
        (1, "Alpha is on page one. It has two sentences."),
        (3, "Gamma is on page three."),
    ]


def test_chunks_from_multiple_documents_keep_their_source():
    pages = [
        PageData(page_number=1, text="From the first file.", source="a.pdf"),
        PageData(page_number=1, text="From the second file.", source="b.pdf"),
    ]

    chunks = chunk_pages(pages)

    assert [(c.source, c.page_number, c.text) for c in chunks] == [
        ("a.pdf", 1, "From the first file."),
        ("b.pdf", 1, "From the second file."),
    ]


def test_loaded_pdf_chunks_keep_page_numbers(tmp_path: Path):
    doc = pymupdf.open()
    for text in ["Page one text. More on page one.", "Page two text."]:
        doc.new_page().insert_text((72, 72), text)
    pdf_path = tmp_path / "doc.pdf"
    doc.save(pdf_path)
    doc.close()

    chunks = chunk_pages(load_pdf(pdf_path))

    assert [(c.page_number, c.text) for c in chunks] == [
        (1, "Page one text. More on page one."),
        (2, "Page two text."),
    ]
    assert all(c.source == str(pdf_path) for c in chunks)
