"""Per-question evaluation records, their JSON Lines format, and failure diagnosis.

Nothing here loads a model, so reports can read saved runs cheaply.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.dataset import Evidence
from src.ingestion.chunking import DocumentChunk

GOLD_CONTEXT = "gold_context"
RETRIEVED_CONTEXT = "retrieved_context"

# End-to-end outcomes: what a user would have received.
ANSWERED = "answered"
REFUSED_BY_ROUTER = "refused_by_router"
REFUSED_BY_READER = "refused_by_reader"
ERROR = "error"

# Diagnoses. For answerable questions, the first failing stage of the pipeline.
CORRECT = "correct"
RETRIEVAL_MISS = "retrieval_miss"  # the evidence was not among the retrieved chunks
ROUTER_REJECT = "router_reject"  # the evidence was retrieved but the router refused the question
CONTEXT_DROPPED = "context_dropped"  # the evidence was retrieved but did not fit in the reader's input
READER_ERROR = "reader_error"  # the reader is wrong even when given the gold chunks
READER_DISTRACTED = "reader_distracted"  # right with the gold chunks, wrong with the retrieved ones
# For unanswerable questions, REFUSED_BY_ROUTER and REFUSED_BY_READER are correct outcomes.
FALSE_ACCEPT = "false_accept"  # an unanswerable question received an answer


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk.

    Attributes:
        rank: 1-based position in the results.
        chunk_id: Identifier from chunk_id(); the text is in the run's index.jsonl.
        passages: Indexes of the question's evidence passages this chunk contains.
    """

    rank: int
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    score: float
    passages: tuple[int, ...]


@dataclass(frozen=True)
class RetrievalRecord:
    """What retrieval returned for one question.

    The hit, rank and reciprocal-rank fields are None for unanswerable
    questions, which have no evidence to find.

    Attributes:
        passage_ranks: For each evidence passage, the rank of the first chunk
            containing it, or None if no retrieved chunk does.
        hit_any_at_k: Whether any evidence passage is in the top k chunks.
        hit_all_at_k: Whether every evidence passage is in the top k chunks.
    """

    results: tuple[RetrievedChunk, ...]
    top_score: float | None
    passage_ranks: tuple[int | None, ...]
    hit_any_at_1: bool | None
    hit_any_at_3: bool | None
    hit_any_at_5: bool | None
    hit_all_at_1: bool | None
    hit_all_at_3: bool | None
    hit_all_at_5: bool | None
    reciprocal_rank: float | None

    @property
    def all_evidence_retrieved(self) -> bool:
        return all(rank is not None for rank in self.passage_ranks)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalRecord":
        return cls(
            **{
                **data,
                "results": tuple(RetrievedChunk(**{**result, "passages": tuple(result["passages"])}) for result in data["results"]),
                "passage_ranks": tuple(data["passage_ranks"]),
            }
        )


@dataclass(frozen=True)
class RoutingRecord:
    status: str
    supported: bool
    reason: str
    top_score: float | None
    threshold: float
    k: int


@dataclass(frozen=True)
class ReaderRecord:
    """One reader's attempt at one question with one kind of context.

    Metric fields are None where they do not apply: answer metrics for
    unanswerable questions, list_item_recall outside list questions, and
    lexical_support for refusals and extractive readers.

    Attributes:
        reader: "bert" or "flan".
        mode: GOLD_CONTEXT or RETRIEVED_CONTEXT.
        context_chunk_ids: The chunks the reader was given, in order.
        used_chunk_ids: The chunks it actually read; FLAN-T5 reads only those
            that fit in its input.
        passages_in_used_context: For each evidence passage, whether a used
            chunk contains it; None for unanswerable questions.
        prediction: The reader's raw output; None when BERT found no answer.
        refused: Whether the prediction is a refusal.
        confidence: BERT's answer score.
        answer_chunk_id: The chunk BERT's answer came from.
        correct: For answerable questions: answer_containment, or for list
            questions list_item_recall == 1.0. Refusals are never correct.
        error: Set, and every result field empty, when the reader call failed.
    """

    reader: str
    mode: str
    context_chunk_ids: tuple[str, ...]
    used_chunk_ids: tuple[str, ...]
    passages_in_used_context: tuple[bool, ...] | None
    prediction: str | None
    refused: bool
    confidence: float | None
    answer_chunk_id: str | None
    exact_match: bool | None
    token_f1: float | None
    answer_containment: bool | None
    list_item_recall: float | None
    lexical_support: float | None
    correct: bool | None
    seconds: float
    error: str | None

    @classmethod
    def from_dict(cls, data: dict) -> "ReaderRecord":
        passages = data["passages_in_used_context"]
        return cls(
            **{
                **data,
                "context_chunk_ids": tuple(data["context_chunk_ids"]),
                "used_chunk_ids": tuple(data["used_chunk_ids"]),
                "passages_in_used_context": None if passages is None else tuple(passages),
            }
        )


@dataclass(frozen=True)
class EndToEndRecord:
    """What the full pipeline (router, then reader) would have returned.

    Attributes:
        outcome: ANSWERED, REFUSED_BY_ROUTER, REFUSED_BY_READER or ERROR.
        correct: For answerable questions, whether the answer was correct; for
            unanswerable ones, whether the question was refused. None on error.
    """

    reader: str
    outcome: str
    prediction: str | None
    correct: bool | None


@dataclass(frozen=True)
class QuestionRecord:
    """Everything recorded about one evaluation question."""

    id: str
    question: str
    split: str
    category: str
    answerable: bool
    failure_tags: tuple[str, ...]
    answers: tuple[str, ...]
    list_items: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    retrieval: RetrievalRecord | None
    routing: RoutingRecord | None
    readers: tuple[ReaderRecord, ...]
    end_to_end: tuple[EndToEndRecord, ...]
    diagnosis: dict[str, str]
    errors: tuple[str, ...]

    def reader_record(self, reader: str, mode: str) -> ReaderRecord | None:
        return next((record for record in self.readers if record.reader == reader and record.mode == mode), None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QuestionRecord":
        return cls(
            **{
                **data,
                "failure_tags": tuple(data["failure_tags"]),
                "answers": tuple(data["answers"]),
                "list_items": tuple(data["list_items"]),
                "evidence": tuple(Evidence(**evidence) for evidence in data["evidence"]),
                "retrieval": None if data["retrieval"] is None else RetrievalRecord.from_dict(data["retrieval"]),
                "routing": None if data["routing"] is None else RoutingRecord(**data["routing"]),
                "readers": tuple(ReaderRecord.from_dict(record) for record in data["readers"]),
                "end_to_end": tuple(EndToEndRecord(**record) for record in data["end_to_end"]),
                "errors": tuple(data["errors"]),
            }
        )


def chunk_id(chunk: DocumentChunk) -> str:
    """Identify a chunk by file name, page and position, e.g. "manual.pdf:p3:c1"."""
    return f"{Path(chunk.source).name}:p{chunk.page_number}:c{chunk.chunk_index}"


def derive_end_to_end(
    reader: str, answerable: bool, routing: RoutingRecord | None, retrieved: ReaderRecord | None
) -> EndToEndRecord:
    """Combine the routing decision with the reader's retrieved-context result.

    A supported question goes to the reader with exactly the retrieved chunks,
    so the retrieved-context result is the pipeline's answer and no new
    inference is needed.
    """
    if routing is None:
        return EndToEndRecord(reader=reader, outcome=ERROR, prediction=None, correct=None)
    if not routing.supported:
        return EndToEndRecord(reader=reader, outcome=REFUSED_BY_ROUTER, prediction=None, correct=not answerable)
    if retrieved is None or retrieved.error is not None:
        return EndToEndRecord(reader=reader, outcome=ERROR, prediction=None, correct=None)
    if retrieved.refused:
        return EndToEndRecord(reader=reader, outcome=REFUSED_BY_READER, prediction=retrieved.prediction, correct=not answerable)
    correct = bool(retrieved.correct) if answerable else False
    return EndToEndRecord(reader=reader, outcome=ANSWERED, prediction=retrieved.prediction, correct=correct)


def diagnose(
    answerable: bool,
    retrieval: RetrievalRecord | None,
    routing: RoutingRecord | None,
    gold: ReaderRecord | None,
    retrieved: ReaderRecord | None,
    end_to_end: EndToEndRecord,
) -> str:
    """Name the outcome of one question for one reader.

    Unanswerable questions get REFUSED_BY_ROUTER, REFUSED_BY_READER or
    FALSE_ACCEPT. Answerable ones get CORRECT, or else the first stage that
    failed: retrieval, routing, the reader's input limit, and finally the
    reader itself, split by whether it can answer from the gold chunks.
    ERROR means a failed stage left too little to decide.
    """
    if end_to_end.outcome == ERROR:
        return ERROR
    if not answerable:
        return end_to_end.outcome if end_to_end.outcome in (REFUSED_BY_ROUTER, REFUSED_BY_READER) else FALSE_ACCEPT
    if end_to_end.correct:
        return CORRECT
    if not retrieval.all_evidence_retrieved:
        return RETRIEVAL_MISS
    if not routing.supported:
        return ROUTER_REJECT
    if not all(retrieved.passages_in_used_context):
        return CONTEXT_DROPPED
    if gold is None or gold.error is not None:
        return ERROR
    return READER_DISTRACTED if gold.correct else READER_ERROR


def write_records(path: Path, records: Sequence[QuestionRecord]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict()) + "\n")


def load_records(path: Path) -> list[QuestionRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [QuestionRecord.from_dict(json.loads(line)) for line in lines if line.strip()]
