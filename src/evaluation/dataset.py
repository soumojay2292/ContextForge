"""Load the evaluation questions and check them against the corpus.

Each line of data/eval/questions.jsonl is one question. Evidence is stored as
exact text from a corpus PDF, never as chunk ids, so the same labels work for
any chunk size or overlap. Validate the dataset with:

    python -m src.evaluation.dataset
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.corpus import CORPUS_DIR, EVAL_DIR
from src.ingestion.chunking import clean_text, split_sentences
from src.ingestion.pdf_loader import load_pdf

QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"

ANSWERABLE_CATEGORIES = frozenset({"factoid", "numeric", "list", "yes_no", "paraphrase", "multi_section"})
UNANSWERABLE_CATEGORIES = frozenset({"unanswerable_off_topic", "unanswerable_near_domain"})
SPLITS = frozenset({"dev", "test"})

FAILURE_TAGS = {
    "paraphrase": "The question words things differently from the document.",
    "exact_token": "The answer hinges on a code or identifier, such as E17 or SHA-256.",
    "distractor": "A similar fact with a different value appears elsewhere in the corpus.",
    "number_format": "The document writes the number in words; answers may use digits.",
    "cross_document": "The evidence comes from two documents.",
    "cross_page": "The evidence comes from two pages of one document.",
    "late_in_page": "The evidence is at the end of a long page.",
    "hyphenation": "The evidence contains a word hyphenated across a line break in the PDF.",
    "requires_reasoning": "The answer is computed from the evidence rather than copied from it.",
    "false_premise": "The question assumes something the documents do not say.",
    "vague": "The question is too underspecified to retrieve anything relevant.",
}

# Evidence must lie inside one sentence at most this long. The chunker never
# splits a sentence shorter than chunk_size, so every evidence passage then lies
# whole inside one chunk for any chunk_size of at least this many characters.
MAX_EVIDENCE_SENTENCE_CHARS = 250

_REQUIRED_FIELDS = {"id", "question", "answerable", "category", "answers", "evidence", "failure_tags", "split"}
_OPTIONAL_FIELDS = {"list_items", "notes"}


@dataclass(frozen=True)
class Evidence:
    """A passage that supports an answer.

    Attributes:
        source: File name of a PDF in the corpus directory.
        page: 1-based page number in that PDF.
        text: Exact text from that page (compared after clean_text).
    """

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class EvalQuestion:
    """One evaluation question with its gold answers and evidence.

    Attributes:
        id: Unique identifier, such as "q001".
        question: The question as a user would ask it.
        answerable: Whether the corpus contains the answer.
        category: One of ANSWERABLE_CATEGORIES or UNANSWERABLE_CATEGORIES.
        answers: Accepted answers. The first is copied from the evidence, except
            for yes_no questions ("yes" or "no") and requires_reasoning questions.
            The rest are alternative phrasings.
        evidence: Passages that support the answer; empty if unanswerable.
        failure_tags: Keys of FAILURE_TAGS describing what makes the question hard.
        split: "dev" for tuning, "test" for reporting.
        list_items: For list questions, each item the answer should contain.
        notes: Free-text remarks for people reading the dataset.
    """

    id: str
    question: str
    answerable: bool
    category: str
    answers: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    failure_tags: tuple[str, ...]
    split: str
    list_items: tuple[str, ...] = ()
    notes: str = ""


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """Read questions from a JSON Lines file, skipping blank lines.

    Raises:
        ValueError: If a line is not valid JSON or has missing or unknown fields.
    """
    questions = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(_parse_question(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return questions


def validate_dataset(questions: list[EvalQuestion], corpus_dir: Path = CORPUS_DIR) -> list[str]:
    """Check questions against each other and against the corpus PDFs.

    Returns:
        One message per problem found; an empty list means the dataset is valid.
    """
    corpus = {pdf.name: [clean_text(page.text) for page in load_pdf(pdf)] for pdf in sorted(Path(corpus_dir).glob("*.pdf"))}
    errors = []
    for question_id, count in Counter(q.id for q in questions).items():
        if count > 1:
            errors.append(f"{question_id}: duplicate id")
    for text, count in Counter(q.question.strip().lower() for q in questions).items():
        if count > 1:
            errors.append(f"duplicate question {text!r}")
    for question in questions:
        errors.extend(f"{question.id}: {problem}" for problem in _question_problems(question, corpus))
    return errors


def _parse_question(record: dict) -> EvalQuestion:
    missing = _REQUIRED_FIELDS - record.keys()
    unknown = record.keys() - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if missing:
        raise ValueError(f"missing fields {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    return EvalQuestion(
        id=record["id"],
        question=record["question"],
        answerable=record["answerable"],
        category=record["category"],
        answers=tuple(record["answers"]),
        evidence=tuple(Evidence(source=e["source"], page=e["page"], text=e["text"]) for e in record["evidence"]),
        failure_tags=tuple(record["failure_tags"]),
        split=record["split"],
        list_items=tuple(record.get("list_items", ())),
        notes=record.get("notes", ""),
    )


def _question_problems(question: EvalQuestion, corpus: dict[str, list[str]]) -> list[str]:
    problems = []
    if not question.question.strip():
        problems.append("question is empty")
    if question.split not in SPLITS:
        problems.append(f"unknown split {question.split!r}")
    unknown_tags = sorted(set(question.failure_tags) - FAILURE_TAGS.keys())
    if unknown_tags:
        problems.append(f"unknown failure tags {unknown_tags}")
    if len(set(question.failure_tags)) != len(question.failure_tags):
        problems.append("repeated failure tags")

    if question.category in UNANSWERABLE_CATEGORIES:
        if question.answerable is not False:
            problems.append(f"category {question.category} requires answerable=false")
        if question.answers or question.evidence or question.list_items:
            problems.append("unanswerable questions must have no answers, evidence or list_items")
        return problems
    if question.category not in ANSWERABLE_CATEGORIES:
        return [*problems, f"unknown category {question.category!r}"]
    if question.answerable is not True:
        problems.append(f"category {question.category} requires answerable=true")

    if not question.answers or not all(answer.strip() for answer in question.answers):
        problems.append("answerable questions need non-empty answers")
    if not question.evidence:
        problems.append("answerable questions need evidence")
    for evidence in question.evidence:
        problems.extend(_evidence_problems(evidence, corpus))

    evidence_text = " ".join(clean_text(evidence.text) for evidence in question.evidence).lower()
    if question.category == "yes_no":
        if question.answers not in (("yes",), ("no",)):
            problems.append('yes_no answers must be ["yes"] or ["no"]')
    elif question.answers and "requires_reasoning" not in question.failure_tags:
        if clean_text(question.answers[0]).lower() not in evidence_text:
            problems.append(f"first answer {question.answers[0]!r} does not appear in the evidence")

    if question.category == "list":
        if len(question.list_items) < 2:
            problems.append("list questions need at least two list_items")
        missing = [item for item in question.list_items if clean_text(item).lower() not in evidence_text]
        if missing:
            problems.append(f"list_items not in the evidence: {missing}")
    elif question.list_items:
        problems.append("only list questions may have list_items")

    if question.category == "multi_section" and len(question.evidence) < 2:
        problems.append("multi_section questions need at least two evidence passages")
    return problems


def _evidence_problems(evidence: Evidence, corpus: dict[str, list[str]]) -> list[str]:
    if evidence.source not in corpus:
        return [f"evidence source {evidence.source!r} is not in the corpus"]
    pages = corpus[evidence.source]
    if isinstance(evidence.page, bool) or not isinstance(evidence.page, int) or not 1 <= evidence.page <= len(pages):
        return [f"{evidence.source} has no page {evidence.page!r}"]
    text = clean_text(evidence.text)
    if not text:
        return ["evidence text is empty"]
    page_text = pages[evidence.page - 1]
    if text not in page_text:
        return [f"evidence not found on {evidence.source} page {evidence.page}: {text!r}"]

    problems = []
    occurrences = sum(page.count(text) for document in corpus.values() for page in document)
    if occurrences > 1:
        problems.append(f"evidence occurs {occurrences} times in the corpus, so retrieval hits would be ambiguous: {text!r}")
    sentences = [sentence for sentence in split_sentences(page_text) if text in sentence]
    if not sentences:
        problems.append(f"evidence spans more than one sentence: {text!r}")
    elif min(len(sentence) for sentence in sentences) > MAX_EVIDENCE_SENTENCE_CHARS:
        problems.append(f"evidence is in a sentence longer than {MAX_EVIDENCE_SENTENCE_CHARS} characters: {text!r}")
    return problems


def main() -> int:
    questions = load_questions()
    errors = validate_dataset(questions)
    answerable = sum(question.answerable for question in questions)
    print(f"{len(questions)} questions: {answerable} answerable, {len(questions) - answerable} unanswerable")
    print("By split:", dict(sorted(Counter(q.split for q in questions).items())))
    print("By category and split:")
    for (category, split), count in sorted(Counter((q.category, q.split) for q in questions).items()):
        print(f"  {category:26s} {split:5s} {count}")
    print("By failure tag:", dict(sorted(Counter(t for q in questions for t in q.failure_tags).items())))
    print(f"{len(errors)} validation errors")
    for error in errors:
        print(f"  {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
