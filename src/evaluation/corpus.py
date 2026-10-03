"""Build the evaluation corpus PDFs from their plain-text sources.

Each source is an ASCII text file in data/eval/corpus_sources/ whose first line
is the document title. Pages are separated by a line containing only PAGE_BREAK.
Rebuild every PDF in data/eval/corpus/ with:

    python -m src.evaluation.corpus
"""

from pathlib import Path

import pymupdf

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
SOURCES_DIR = EVAL_DIR / "corpus_sources"
CORPUS_DIR = EVAL_DIR / "corpus"

PAGE_BREAK = "=== PAGE BREAK ==="

_PAGE_WIDTH, _PAGE_HEIGHT = pymupdf.paper_size("a4")
_MARGIN = 72
_FONT_SIZE = 11


def read_source_pages(source_path: Path) -> list[str]:
    """Return the text of each page in a source file.

    Raises:
        ValueError: If any page is empty.
    """
    pages: list[list[str]] = [[]]
    for line in Path(source_path).read_text(encoding="utf-8").splitlines():
        if line.strip() == PAGE_BREAK:
            pages.append([])
        else:
            pages[-1].append(line)

    texts = ["\n".join(lines).strip() for lines in pages]
    if not all(texts):
        raise ValueError(f"{Path(source_path).name} has an empty page")
    return texts


def build_pdf(source_path: Path, pdf_path: Path) -> None:
    """Write a PDF with one page per source page.

    Raises:
        ValueError: If a source page has more text than fits on one PDF page.
    """
    pages = read_source_pages(source_path)
    doc = pymupdf.open()
    for number, text in enumerate(pages, start=1):
        page = doc.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
        text_area = pymupdf.Rect(_MARGIN, _MARGIN, _PAGE_WIDTH - _MARGIN, _PAGE_HEIGHT - _MARGIN)
        if page.insert_textbox(text_area, text, fontsize=_FONT_SIZE, fontname="helv") < 0:
            raise ValueError(f"{Path(source_path).name}: page {number} does not fit on one PDF page")

    # Fixed metadata and no random file ID, so a source always produces the same bytes.
    doc.set_metadata({"title": pages[0].splitlines()[0], "creationDate": "", "modDate": ""})
    doc.save(pdf_path, garbage=4, deflate=True, no_new_id=True)
    doc.close()


def build_corpus(sources_dir: Path = SOURCES_DIR, corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    """Build a PDF in corpus_dir for every source in sources_dir; return their paths."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = []
    for source_path in sorted(Path(sources_dir).glob("*.txt")):
        pdf_path = Path(corpus_dir) / f"{source_path.stem}.pdf"
        build_pdf(source_path, pdf_path)
        pdf_paths.append(pdf_path)
    return pdf_paths


if __name__ == "__main__":
    for path in build_corpus():
        print(f"Built {path}")
