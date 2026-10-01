"""Load PDF files and extract their text page by page using PyMuPDF."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


class InvalidPDFError(Exception):
    """Raised when a file exists but cannot be read as a PDF."""


@dataclass(frozen=True)
class PageData:
    """Text extracted from a single PDF page.

    Attributes:
        page_number: 1-based page number, matching what a PDF viewer shows.
        text: Raw text extracted from the page (empty if the page has none).
        source: Path to the PDF the page came from.
    """

    page_number: int
    text: str
    source: str


def load_pdf(file_path: str | Path) -> list[PageData]:
    """Extract text from every page of a PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        One PageData per page, in document order.

    Raises:
        FileNotFoundError: If no file exists at file_path.
        InvalidPDFError: If the path is not a file, or the file is empty,
            corrupted, not a PDF, or password-protected.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise InvalidPDFError(f"Path is not a file: {path}")

    try:
        doc = pymupdf.open(path, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise InvalidPDFError(f"Could not open PDF {path}: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidPDFError(f"PDF is password-protected: {path}")

        return [
            PageData(page_number=index + 1, text=page.get_text(), source=str(path))
            for index, page in enumerate(doc)
        ]
