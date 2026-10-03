from dataclasses import replace
from pathlib import Path

import pytest

from src.evaluation.corpus import CORPUS_DIR, PAGE_BREAK, SOURCES_DIR, build_pdf, read_source_pages
from src.evaluation.dataset import (
    ANSWERABLE_CATEGORIES,
    FAILURE_TAGS,
    UNANSWERABLE_CATEGORIES,
    EvalQuestion,
    Evidence,
    load_questions,
    validate_dataset,
)
from src.ingestion.chunking import chunk_pages, clean_text
from src.ingestion.pdf_loader import load_pdf

SOURCES = sorted(SOURCES_DIR.glob("*.txt"))


@pytest.fixture(scope="module")
def questions() -> list[EvalQuestion]:
    return load_questions()


# --- Corpus -----------------------------------------------------------------


def test_corpus_has_three_to_five_documents_of_two_to_three_pages():
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))

    assert 3 <= len(pdfs) <= 5
    assert all(2 <= len(load_pdf(pdf)) <= 3 for pdf in pdfs)


def test_every_source_has_a_pdf_and_every_pdf_a_source():
    assert {source.stem for source in SOURCES} == {pdf.stem for pdf in CORPUS_DIR.glob("*.pdf")}


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: path.stem)
def test_sources_are_ascii(source: Path):
    # The built-in PDF fonts only render a limited character set reliably.
    source.read_text(encoding="utf-8").encode("ascii")


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: path.stem)
def test_pdf_pages_contain_exactly_their_source_text(source: Path):
    pdf_pages = load_pdf(CORPUS_DIR / f"{source.stem}.pdf")

    assert [clean_text(page.text) for page in pdf_pages] == [clean_text(text) for text in read_source_pages(source)]


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: path.stem)
def test_building_a_pdf_twice_gives_identical_bytes(source: Path, tmp_path: Path):
    build_pdf(source, tmp_path / "first.pdf")
    build_pdf(source, tmp_path / "second.pdf")

    assert (tmp_path / "first.pdf").read_bytes() == (tmp_path / "second.pdf").read_bytes()


def test_source_with_an_empty_page_is_rejected(tmp_path: Path):
    source = tmp_path / "broken.txt"
    source.write_text(f"Title\nSome text.\n{PAGE_BREAK}\n\n{PAGE_BREAK}\nMore text.\n")

    with pytest.raises(ValueError, match="empty page"):
        read_source_pages(source)


def test_page_too_long_for_one_pdf_page_is_rejected(tmp_path: Path):
    source = tmp_path / "long.txt"
    source.write_text("Title\n" + "A line of text.\n" * 200)

    with pytest.raises(ValueError, match="does not fit"):
        build_pdf(source, tmp_path / "long.pdf")


# --- The committed dataset --------------------------------------------------


def test_dataset_passes_validation(questions: list[EvalQuestion]):
    assert validate_dataset(questions) == []


def test_dataset_size_and_balance(questions: list[EvalQuestion]):
    answerable = sum(question.answerable for question in questions)

    assert 45 <= len(questions) <= 55
    assert 28 <= answerable <= 32
    assert 18 <= len(questions) - answerable <= 22


@pytest.mark.parametrize("split", ["dev", "test"])
def test_every_category_appears_in_each_split(questions: list[EvalQuestion], split: str):
    categories = {question.category for question in questions if question.split == split}

    assert categories == ANSWERABLE_CATEGORIES | UNANSWERABLE_CATEGORIES


def test_every_failure_tag_is_used(questions: list[EvalQuestion]):
    assert {tag for question in questions for tag in question.failure_tags} == set(FAILURE_TAGS)


@pytest.mark.parametrize(("chunk_size", "chunk_overlap"), [(250, 0), (250, 100), (500, 100), (1000, 200), (2000, 0)])
def test_every_evidence_passage_lies_inside_one_chunk(questions: list[EvalQuestion], chunk_size: int, chunk_overlap: int):
    chunks = [
        chunk
        for pdf in sorted(CORPUS_DIR.glob("*.pdf"))
        for chunk in chunk_pages(load_pdf(pdf), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]

    missing = [
        (question.id, evidence.text)
        for question in questions
        for evidence in question.evidence
        if not any(
            clean_text(evidence.text) in chunk.text
            and Path(chunk.source).name == evidence.source
            and chunk.page_number == evidence.page
            for chunk in chunks
        )
    ]
    assert missing == []


# --- Validation rules, on a tiny corpus -------------------------------------

LONG_SENTENCE = "This sentence is deliberately long " + "and keeps going " * 16 + "until it finally ends here."


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    source = tmp_path / "tiny.txt"
    source.write_text(
        "Tiny Manual\nThe widget is blue. The gadget weighs 3 kilograms.\n"
        f"{PAGE_BREAK}\n"
        f"The widget is blue. Setup has two steps: plug it in and press start. {LONG_SENTENCE}\n"
    )
    build_pdf(source, tmp_path / "tiny.pdf")
    return tmp_path


def make_question(**changes) -> EvalQuestion:
    question = EvalQuestion(
        id="t1",
        question="How heavy is the gadget?",
        answerable=True,
        category="numeric",
        answers=("3 kilograms",),
        evidence=(Evidence(source="tiny.pdf", page=1, text="The gadget weighs 3 kilograms."),),
        failure_tags=(),
        split="dev",
    )
    return replace(question, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"category": "yes_no", "answers": ("yes",)},
        {"answers": ("6 kilograms",), "failure_tags": ("requires_reasoning",)},
        {
            "category": "list",
            "answers": ("plug it in and press start",),
            "list_items": ("plug it in", "press start"),
            "evidence": (Evidence(source="tiny.pdf", page=2, text="Setup has two steps: plug it in and press start."),),
        },
        {"category": "unanswerable_off_topic", "answerable": False, "answers": (), "evidence": ()},
    ],
    ids=["numeric", "yes_no", "requires_reasoning", "list", "unanswerable"],
)
def test_valid_questions_pass(tiny_corpus: Path, changes: dict):
    assert validate_dataset([make_question(**changes)], tiny_corpus) == []


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    [
        ({"evidence": (Evidence("tiny.pdf", 1, "The gadget weighs 4 kilograms."),)}, "evidence not found"),
        ({"evidence": (Evidence("tiny.pdf", 2, "The gadget weighs 3 kilograms."),)}, "evidence not found on tiny.pdf page 2"),
        ({"evidence": (Evidence("tiny.pdf", 3, "The gadget weighs 3 kilograms."),)}, "has no page 3"),
        ({"evidence": (Evidence("tiny.pdf", "1", "The gadget weighs 3 kilograms."),)}, "has no page '1'"),
        ({"evidence": (Evidence("other.pdf", 1, "The gadget weighs 3 kilograms."),)}, "is not in the corpus"),
        ({"evidence": (Evidence("tiny.pdf", 1, "The widget is blue."),), "answers": ("blue",)}, "occurs 2 times"),
        ({"evidence": (Evidence("tiny.pdf", 1, "blue. The gadget weighs 3 kilograms"),)}, "spans more than one sentence"),
        ({"evidence": (Evidence("tiny.pdf", 2, "until it finally ends here"),), "answers": ("ends here",)}, "longer than 250"),
        ({"answers": ("5 kilograms",)}, "does not appear in the evidence"),
        ({"answers": ()}, "need non-empty answers"),
        ({"answers": ("3 kilograms", " ")}, "need non-empty answers"),
        ({"evidence": ()}, "need evidence"),
        ({"question": "  "}, "question is empty"),
        ({"failure_tags": ("made_up",)}, "unknown failure tags"),
        ({"failure_tags": ("distractor", "distractor")}, "repeated failure tags"),
        ({"split": "train"}, "unknown split"),
        ({"category": "trivia"}, "unknown category"),
        ({"answerable": False}, "requires answerable=true"),
        ({"category": "unanswerable_off_topic"}, "requires answerable=false"),
        ({"category": "unanswerable_near_domain", "answerable": False}, "must have no answers"),
        ({"category": "yes_no", "answers": ("maybe",)}, "yes_no answers"),
        ({"category": "list", "list_items": ()}, "at least two list_items"),
        ({"category": "list", "list_items": ("3 kilograms", "4 kilograms")}, "list_items not in the evidence"),
        ({"list_items": ("3 kilograms", "gadget")}, "only list questions"),
        ({"category": "multi_section"}, "at least two evidence passages"),
    ],
)
def test_validation_reports_each_problem(tiny_corpus: Path, changes: dict, expected_error: str):
    errors = validate_dataset([make_question(**changes)], tiny_corpus)

    assert any(expected_error in error for error in errors), errors


def test_validation_reports_duplicate_ids_and_questions(tiny_corpus: Path):
    errors = validate_dataset([make_question(), make_question(question="HOW HEAVY is the gadget?")], tiny_corpus)

    assert "t1: duplicate id" in errors
    assert "duplicate question 'how heavy is the gadget?'" in errors


# --- Loading ----------------------------------------------------------------

VALID_LINE = (
    '{"id": "t1", "question": "Q?", "answerable": false, "category": "unanswerable_off_topic", '
    '"answers": [], "evidence": [], "failure_tags": [], "split": "dev"}'
)


def test_load_questions_parses_lines_and_skips_blank_ones(tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    path.write_text(f"{VALID_LINE}\n\n{VALID_LINE.replace('t1', 't2')}\n")

    assert [question.id for question in load_questions(path)] == ["t1", "t2"]


@pytest.mark.parametrize(
    ("bad_line", "expected_error"),
    [
        ("{not json", r":2:"),
        (VALID_LINE.replace('"split": "dev"', '"splt": "dev"'), r"missing fields \['split'\]"),
        (VALID_LINE.replace('"split": "dev"', '"split": "dev", "extra": 1'), r"unknown fields \['extra'\]"),
    ],
)
def test_load_questions_rejects_malformed_lines(tmp_path: Path, bad_line: str, expected_error: str):
    path = tmp_path / "questions.jsonl"
    path.write_text(f"{VALID_LINE}\n{bad_line}\n")

    with pytest.raises(ValueError, match=expected_error):
        load_questions(path)
