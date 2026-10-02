"""Clean extracted PDF text and split it into sentence-based chunks."""

import re
import textwrap
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from src.ingestion.pdf_loader import PageData

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# A word split across lines with a hyphen or soft hyphen, e.g. "infor-\nmation".
_HYPHENATED_LINE_BREAK = re.compile(r"(?<=[A-Za-z])[-­]\s*\n\s*(?=[a-z])")
# Control characters, soft hyphens and zero-width characters left behind by PDFs.
_INVISIBLE_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f­​﻿]")
_WHITESPACE = re.compile(r"\s+")
# Whitespace after ., ! or ? (optionally followed by a closing quote or bracket)
# when the next character can start a sentence.
_SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+(?=[\"'“‘(\[]?[A-Z0-9])"
)
# Words whose trailing period does not end a sentence.
_ABBREVIATIONS = frozenset(
    {"dr", "e.g", "etc", "fig", "i.e", "inc", "jr", "ltd", "mr", "mrs", "ms", "no", "prof", "sr", "st", "vol", "vs"}
)


@dataclass(frozen=True)
class DocumentChunk:
    """A piece of a page's text, with the metadata needed to cite it.

    Attributes:
        text: Cleaned chunk text.
        source: Path to the PDF the chunk came from.
        page_number: 1-based page the chunk came from.
        chunk_index: 0-based position of the chunk within its page.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int


def clean_text(text: str) -> str:
    """Normalize raw PDF text into a single line of clean prose.

    Applies Unicode NFKC normalization (e.g. the "ﬁ" ligature becomes "fi"),
    rejoins words hyphenated across line breaks, removes invisible control
    characters, and collapses all whitespace, including newlines, to single spaces.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHENATED_LINE_BREAK.sub("", text)
    text = _INVISIBLE_CHARS.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences ending in ., ! or ?.

    A period after a common abbreviation ("e.g.", "Dr."), a single initial ("J.")
    or a bare number (the "2." of a numbered heading) does not end a sentence.
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    for piece in _SENTENCE_BOUNDARY.split(text.strip()):
        if sentences and _ends_mid_sentence(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Group whole sentences into chunks of at most chunk_size characters.

    Each chunk after the first starts with trailing sentences of the previous
    chunk, totalling at most chunk_overlap characters. Overlap is made of whole
    sentences, so it can be shorter than chunk_overlap, or empty when the last
    sentence alone is longer. A sentence longer than chunk_size is split
    between words.

    Raises:
        ValueError: If chunk_size is not positive, or chunk_overlap is negative
            or not smaller than chunk_size.
    """
    _validate_chunk_params(chunk_size, chunk_overlap)

    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= chunk_size:
            units.append(sentence)
        else:
            units.extend(textwrap.wrap(sentence, chunk_size, break_on_hyphens=False))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _joined_length([*current, unit]) > chunk_size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, chunk_overlap)
            # Give up overlap sentences that would push the next chunk past chunk_size.
            while current and _joined_length([*current, unit]) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: Iterable[PageData],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Clean and chunk each page, keeping each chunk's source and page number.

    Chunks never span pages, so every chunk maps to exactly one page.
    Pages with no text produce no chunks.

    Raises:
        ValueError: If chunk_size or chunk_overlap is invalid (see chunk_text).
    """
    _validate_chunk_params(chunk_size, chunk_overlap)

    chunks: list[DocumentChunk] = []
    for page in pages:
        page_chunks = chunk_text(clean_text(page.text), chunk_size, chunk_overlap)
        chunks.extend(
            DocumentChunk(text=text, source=page.source, page_number=page.page_number, chunk_index=index)
            for index, text in enumerate(page_chunks)
        )
    return chunks


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be at least 0 and less than chunk_size ({chunk_size}), got {chunk_overlap}"
        )


def _ends_mid_sentence(sentence: str) -> bool:
    if not sentence.endswith("."):
        return False
    last_word = sentence.rsplit(maxsplit=1)[-1][:-1].lstrip("\"'“‘([").lower()
    return last_word in _ABBREVIATIONS or last_word.isdigit() or (len(last_word) == 1 and last_word.isalpha())


def _overlap_tail(sentences: list[str], chunk_overlap: int) -> list[str]:
    """Return the most trailing sentences whose joined length fits in chunk_overlap."""
    tail: list[str] = []
    for sentence in reversed(sentences):
        if _joined_length([sentence, *tail]) > chunk_overlap:
            break
        tail.insert(0, sentence)
    return tail


def _joined_length(parts: list[str]) -> int:
    return len(" ".join(parts))
