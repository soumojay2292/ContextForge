from pathlib import Path

import pymupdf
import pytest

from src.ingestion.pdf_loader import InvalidPDFError, PageData, load_pdf


def make_pdf(path: Path, page_texts: list[str], **save_options) -> Path:
    """Write a PDF with one page per entry in page_texts (empty string = blank page)."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path, **save_options)
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "sample.pdf", ["First page", "Second page", "Third page"])


def test_returns_one_page_data_per_page(sample_pdf: Path):
    pages = load_pdf(sample_pdf)

    assert len(pages) == 3
    assert all(isinstance(page, PageData) for page in pages)


def test_page_numbers_are_one_based_and_in_order(sample_pdf: Path):
    pages = load_pdf(sample_pdf)

    assert [page.page_number for page in pages] == [1, 2, 3]


def test_text_is_extracted_from_the_matching_page(sample_pdf: Path):
    pages = load_pdf(sample_pdf)

    assert [page.text.strip() for page in pages] == ["First page", "Second page", "Third page"]


def test_source_is_recorded_on_every_page(sample_pdf: Path):
    pages = load_pdf(sample_pdf)

    assert all(page.source == str(sample_pdf) for page in pages)


def test_accepts_string_path(sample_pdf: Path):
    pages = load_pdf(str(sample_pdf))

    assert len(pages) == 3


def test_blank_page_keeps_its_page_number(tmp_path: Path):
    pdf = make_pdf(tmp_path / "with_blank.pdf", ["Intro", "", "Outro"])

    pages = load_pdf(pdf)

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert pages[1].text.strip() == ""
    assert pages[2].text.strip() == "Outro"


def test_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "does_not_exist.pdf")


def test_directory_raises_invalid_pdf(tmp_path: Path):
    with pytest.raises(InvalidPDFError, match="not a file"):
        load_pdf(tmp_path)


def test_empty_file_raises_invalid_pdf(tmp_path: Path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    with pytest.raises(InvalidPDFError):
        load_pdf(empty)


def test_non_pdf_file_raises_invalid_pdf(tmp_path: Path):
    not_a_pdf = tmp_path / "notes.pdf"
    not_a_pdf.write_text("This is plain text, not a PDF.")

    with pytest.raises(InvalidPDFError):
        load_pdf(not_a_pdf)


def test_corrupted_pdf_raises_invalid_pdf(tmp_path: Path):
    corrupted = tmp_path / "corrupted.pdf"
    corrupted.write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog")

    with pytest.raises(InvalidPDFError):
        load_pdf(corrupted)


def test_password_protected_pdf_raises_invalid_pdf(tmp_path: Path):
    pdf = make_pdf(
        tmp_path / "locked.pdf",
        ["Secret"],
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user",
        owner_pw="owner",
    )

    with pytest.raises(InvalidPDFError, match="password-protected"):
        load_pdf(pdf)
