"""Run the evaluation questions through each pipeline stage and save the raw results.

Stages, each loading its model only while it runs:

1. Index the evaluation corpus with the default chunking and embedding model.
2. Route every question once; the router's top-k results are the retrieval.
3. Ask BERT, then FLAN-T5, every question twice: with the gold chunks (the
   chunks containing the evidence) and with the retrieved chunks.
4. Derive end-to-end outcomes and a diagnosis per reader, without inference.

Only the existing default configuration is used. Run with:

    python -m src.evaluation.runner [--split dev|test] [--ids q001,q028] [--readers bert,flan]

Results are written to reports/evaluation/runs/<timestamp>/: run.json,
index.jsonl (every chunk's text) and records.jsonl (one record per question).
"""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.evaluation.corpus import CORPUS_DIR
from src.evaluation.dataset import QUESTIONS_PATH, EvalQuestion, load_questions, validate_dataset
from src.evaluation.metrics import (
    answer_containment,
    exact_match,
    hit_at_k,
    is_refusal,
    is_relevant,
    lexical_support,
    list_item_recall,
    reciprocal_rank,
    token_f1,
)
from src.evaluation.records import (
    GOLD_CONTEXT,
    RETRIEVED_CONTEXT,
    QuestionRecord,
    ReaderRecord,
    RetrievalRecord,
    RetrievedChunk,
    RoutingRecord,
    chunk_id,
    derive_end_to_end,
    diagnose,
    write_records,
)
from src.generation.flan_t5 import FlanT5Generator
from src.ingestion.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DocumentChunk, chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.qa.bert_qa import BertQA
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.query_router import DEFAULT_TOP_K, QueryRouter
from src.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "evaluation" / "runs"
READERS = ("bert", "flan")
SPLITS = ("all", "dev", "test")

# Hit@5 needs the top 5 chunks, which the router retrieves by default.
RETRIEVAL_DEPTH = 5

_PACKAGES = ("torch", "transformers", "sentence-transformers", "faiss-cpu", "PyMuPDF", "numpy")


@dataclass(frozen=True)
class ReaderOutput:
    """A reader's answer in a form shared by BERT and FLAN-T5.

    Attributes:
        prediction: The answer text, or None when BERT found no answer.
        used_chunks: The chunks the reader actually read.
        confidence: BERT's answer score.
        answer_chunk: The chunk BERT's answer came from.
    """

    prediction: str | None
    used_chunks: tuple[DocumentChunk, ...]
    confidence: float | None = None
    answer_chunk: DocumentChunk | None = None


Reader = Callable[[str, list[DocumentChunk]], ReaderOutput]


def bert_reader(bert: BertQA) -> Reader:
    def read(question: str, chunks: list[DocumentChunk]) -> ReaderOutput:
        answer = bert.answer(question, chunks)
        if answer is None:
            return ReaderOutput(prediction=None, used_chunks=tuple(chunks))
        return ReaderOutput(answer.text, tuple(chunks), confidence=answer.score, answer_chunk=answer.chunk)

    return read


def flan_reader(flan: FlanT5Generator) -> Reader:
    def read(question: str, chunks: list[DocumentChunk]) -> ReaderOutput:
        answer = flan.generate(question, chunks)
        return ReaderOutput(prediction=answer.text, used_chunks=answer.sources)

    return read


@dataclass(frozen=True)
class EvaluationRun:
    """The result of run_evaluation, ready to save."""

    records: list[QuestionRecord]
    index_chunks: list[DocumentChunk]
    settings: dict
    timings: dict[str, float]


@dataclass(frozen=True)
class _Retrieved:
    retrieval: RetrievalRecord | None
    routing: RoutingRecord | None
    chunks: list[DocumentChunk]
    error: str | None


def build_index_chunks(corpus_dir: Path = CORPUS_DIR) -> list[DocumentChunk]:
    """Chunk every corpus PDF with the default chunk size and overlap, in document order."""
    pdfs = sorted(Path(corpus_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {corpus_dir}")
    return [chunk for pdf in pdfs for chunk in chunk_pages(load_pdf(pdf))]


def gold_context(question: EvalQuestion, index_chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    """The chunks a perfect retriever would return for a question.

    For each evidence passage, in the dataset's order, this is the chunk that
    contains it. When overlapping chunks both contain a passage, the earlier
    chunk on the page is used, so the choice never depends on retrieval. A
    chunk holding several passages appears once. Unanswerable questions have
    no gold context.

    Raises:
        ValueError: If no chunk contains one of the evidence passages.
    """
    context: list[DocumentChunk] = []
    for passage in question.evidence:
        containing = [chunk for chunk in index_chunks if is_relevant(chunk, passage)]
        if not containing:
            raise ValueError(f"{question.id}: no chunk contains the evidence {passage.text!r}")
        first = min(containing, key=lambda chunk: chunk.chunk_index)
        if first not in context:
            context.append(first)
    return context


def run_evaluation(
    questions: Sequence[EvalQuestion],
    corpus_dir: Path = CORPUS_DIR,
    *,
    readers: Sequence[str] = READERS,
    make_embedder: Callable[[], EmbeddingModel] = EmbeddingModel,
    make_bert: Callable[[], BertQA] = BertQA,
    make_flan: Callable[[], FlanT5Generator] = FlanT5Generator,
    log: Callable[[str], None] = lambda message: None,
) -> EvaluationRun:
    """Evaluate questions stage by stage, holding at most one model at a time.

    The make_* arguments create each model when its stage starts; tests pass
    fakes. A failing question or reader call is recorded and the run goes on.

    Raises:
        ValueError: If readers names an unknown reader.
    """
    unknown = sorted(set(readers) - set(READERS))
    if unknown:
        raise ValueError(f"Unknown readers {unknown}; choose from {list(READERS)}")
    if DEFAULT_TOP_K < RETRIEVAL_DEPTH:
        raise RuntimeError(f"The router retrieves {DEFAULT_TOP_K} chunks, but Hit@{RETRIEVAL_DEPTH} needs {RETRIEVAL_DEPTH}")

    settings: dict = {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}
    timings: dict[str, float] = {}

    log("Stage 1/3: indexing the corpus, then retrieving and routing every question")
    start = time.perf_counter()
    index_chunks = build_index_chunks(corpus_dir)
    embedder = make_embedder()
    store = VectorStore(embedder)
    store.add_chunks(index_chunks)
    router = QueryRouter(store)
    settings.update(
        embedding_model=getattr(embedder, "model_name", None),
        similarity_threshold=router.similarity_threshold,
        router_k=router.k,
    )
    retrieved = {question.id: _retrieve_and_route(question, router) for question in questions}
    del router, store, embedder
    gc.collect()
    timings["index_retrieval_routing"] = time.perf_counter() - start
    log(f"  {len(index_chunks)} chunks indexed, {len(questions)} questions routed")

    gold: dict[str, list[DocumentChunk]] = {}
    gold_errors: dict[str, str] = {}
    for question in questions:
        if question.answerable:
            try:
                gold[question.id] = gold_context(question, index_chunks)
            except ValueError as exc:
                gold_errors[question.id] = str(exc)

    reader_records: dict[str, list[ReaderRecord]] = {question.id: [] for question in questions}
    specs = {"bert": (make_bert, bert_reader, False), "flan": (make_flan, flan_reader, True)}
    for stage, name in enumerate(readers, start=2):
        make_model, wrap, generative = specs[name]
        log(f"Stage {stage}/3: {name} with gold and retrieved contexts")
        start = time.perf_counter()
        try:
            model = make_model()
        except Exception as exc:  # Recorded on every question, so the other reader's results survive.
            error = f"{name} failed to load: {type(exc).__name__}: {exc}"
            for question in questions:
                modes = (GOLD_CONTEXT, RETRIEVED_CONTEXT) if question.answerable else (RETRIEVED_CONTEXT,)
                reader_records[question.id].extend(_failed_reader_record(name, mode, (), error) for mode in modes)
            timings[name] = time.perf_counter() - start
            log(f"  {error}")
            continue

        read = wrap(model)
        for question in questions:
            if question.answerable:
                if question.id in gold_errors:
                    record = _failed_reader_record(name, GOLD_CONTEXT, (), gold_errors[question.id])
                else:
                    record = _evaluate_reader(name, GOLD_CONTEXT, read, question, gold[question.id], generative)
                reader_records[question.id].append(record)
            found = retrieved[question.id]
            if found.error is not None:
                record = _failed_reader_record(name, RETRIEVED_CONTEXT, (), "no retrieved context: retrieval failed")
            else:
                record = _evaluate_reader(name, RETRIEVED_CONTEXT, read, question, found.chunks, generative)
            reader_records[question.id].append(record)
            log(f"  {name} {question.id}")
        settings[name] = _reader_settings(model)
        del read, model
        gc.collect()
        timings[name] = time.perf_counter() - start

    records = [_assemble(question, retrieved[question.id], reader_records[question.id], readers) for question in questions]
    return EvaluationRun(records=records, index_chunks=index_chunks, settings=settings, timings=timings)


def evaluate(
    questions_path: Path = QUESTIONS_PATH,
    corpus_dir: Path = CORPUS_DIR,
    output_dir: Path | None = None,
    *,
    split: str = "all",
    ids: Sequence[str] | None = None,
    readers: Sequence[str] = READERS,
    make_embedder: Callable[[], EmbeddingModel] = EmbeddingModel,
    make_bert: Callable[[], BertQA] = BertQA,
    make_flan: Callable[[], FlanT5Generator] = FlanT5Generator,
    log: Callable[[str], None] = lambda message: None,
) -> Path:
    """Validate the dataset, evaluate the selected questions and save the run.

    Returns:
        The directory the run was saved to.

    Raises:
        ValueError: If the dataset is invalid or the selection matches no questions.
        FileExistsError: If output_dir already exists.
    """
    questions = load_questions(questions_path)
    errors = validate_dataset(questions, corpus_dir)
    if errors:
        raise ValueError("The evaluation dataset is invalid:\n" + "\n".join(f"  {error}" for error in errors))
    selected = _select_questions(questions, split, ids)

    run = run_evaluation(
        selected,
        corpus_dir,
        readers=readers,
        make_embedder=make_embedder,
        make_bert=make_bert,
        make_flan=make_flan,
        log=log,
    )
    output_dir = Path(output_dir) if output_dir else RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_run(run, output_dir, _run_metadata(run, questions_path, corpus_dir, split, ids))
    return output_dir


def save_run(run: EvaluationRun, output_dir: Path, run_metadata: dict) -> None:
    """Write run.json, index.jsonl and records.jsonl into a new directory.

    Raises:
        FileExistsError: If output_dir already exists, so earlier runs are never overwritten.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as file:
        for chunk in run.index_chunks:
            entry = {
                "chunk_id": chunk_id(chunk),
                "source": Path(chunk.source).name,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            file.write(json.dumps(entry) + "\n")
    write_records(output_dir / "records.jsonl", run.records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ContextForge with its default configuration.")
    parser.add_argument("--split", choices=SPLITS, default="all", help="Which questions to run (default: all).")
    parser.add_argument("--ids", help="Comma-separated question ids to run, e.g. q001,q028.")
    parser.add_argument("--readers", default=",".join(READERS), help="Comma-separated readers: bert, flan.")
    parser.add_argument("--output-dir", type=Path, help="Where to save the run (default: reports/evaluation/runs/<timestamp>).")
    args = parser.parse_args(argv)

    try:
        output_dir = evaluate(
            output_dir=args.output_dir,
            split=args.split,
            ids=args.ids.split(",") if args.ids else None,
            readers=args.readers.split(","),
            log=print,
        )
    except (ValueError, FileExistsError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Saved the run to {output_dir}")
    return 0


def _retrieve_and_route(question: EvalQuestion, router: QueryRouter) -> _Retrieved:
    try:
        decision = router.route(question.question)
    except Exception as exc:  # Recorded so that one failing question does not stop the run.
        return _Retrieved(None, None, [], f"retrieval and routing failed: {type(exc).__name__}: {exc}")

    chunks = [result.chunk for result in decision.results]
    results = tuple(
        RetrievedChunk(
            rank=rank,
            chunk_id=chunk_id(result.chunk),
            source=Path(result.chunk.source).name,
            page=result.chunk.page_number,
            chunk_index=result.chunk.chunk_index,
            score=result.score,
            passages=tuple(index for index, passage in enumerate(question.evidence) if is_relevant(result.chunk, passage)),
        )
        for rank, result in enumerate(decision.results, start=1)
    )
    routing = RoutingRecord(
        status=decision.status.value,
        supported=decision.supported,
        reason=decision.reason,
        top_score=decision.top_score,
        threshold=router.similarity_threshold,
        k=router.k,
    )
    return _Retrieved(_retrieval_record(question, chunks, results, decision.top_score), routing, chunks, None)


def _retrieval_record(
    question: EvalQuestion, chunks: list[DocumentChunk], results: tuple[RetrievedChunk, ...], top_score: float | None
) -> RetrievalRecord:
    if not question.answerable:
        return RetrievalRecord(
            results=results,
            top_score=top_score,
            passage_ranks=(),
            hit_any_at_1=None,
            hit_any_at_3=None,
            hit_any_at_5=None,
            hit_all_at_1=None,
            hit_all_at_3=None,
            hit_all_at_5=None,
            reciprocal_rank=None,
        )
    evidence = question.evidence
    return RetrievalRecord(
        results=results,
        top_score=top_score,
        passage_ranks=tuple(
            next((result.rank for result in results if index in result.passages), None) for index in range(len(evidence))
        ),
        hit_any_at_1=hit_at_k(chunks, evidence, 1),
        hit_any_at_3=hit_at_k(chunks, evidence, 3),
        hit_any_at_5=hit_at_k(chunks, evidence, 5),
        hit_all_at_1=hit_at_k(chunks, evidence, 1, require_all=True),
        hit_all_at_3=hit_at_k(chunks, evidence, 3, require_all=True),
        hit_all_at_5=hit_at_k(chunks, evidence, 5, require_all=True),
        reciprocal_rank=reciprocal_rank(chunks, evidence),
    )


def _evaluate_reader(
    reader: str, mode: str, read: Reader, question: EvalQuestion, context: Sequence[DocumentChunk], generative: bool
) -> ReaderRecord:
    start = time.perf_counter()
    try:
        output = read(question.question, list(context))
    except Exception as exc:  # Recorded so that one failing call does not stop the run.
        return _failed_reader_record(reader, mode, context, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    seconds = time.perf_counter() - start

    refused = is_refusal(output.prediction)
    # A refusal is scored as an empty answer, so its wording cannot earn credit.
    scored = "" if refused else output.prediction
    answerable = question.answerable
    is_list = question.category == "list"
    containment = answer_containment(scored, question.answers) if answerable else None
    item_recall = list_item_recall(scored, question.list_items) if answerable and is_list else None
    if not answerable:
        correct = None
    elif is_list:
        correct = item_recall == 1.0
    else:
        correct = containment

    return ReaderRecord(
        reader=reader,
        mode=mode,
        context_chunk_ids=tuple(chunk_id(chunk) for chunk in context),
        used_chunk_ids=tuple(chunk_id(chunk) for chunk in output.used_chunks),
        passages_in_used_context=(
            tuple(any(is_relevant(chunk, passage) for chunk in output.used_chunks) for passage in question.evidence)
            if answerable
            else None
        ),
        prediction=output.prediction,
        refused=refused,
        confidence=output.confidence,
        answer_chunk_id=chunk_id(output.answer_chunk) if output.answer_chunk is not None else None,
        exact_match=exact_match(scored, question.answers) if answerable else None,
        token_f1=token_f1(scored, question.answers) if answerable else None,
        answer_containment=containment,
        list_item_recall=item_recall,
        lexical_support=(
            lexical_support(output.prediction, [chunk.text for chunk in output.used_chunks])
            if generative and not refused
            else None
        ),
        correct=correct,
        seconds=seconds,
        error=None,
    )


def _failed_reader_record(
    reader: str, mode: str, context: Sequence[DocumentChunk], error: str, seconds: float = 0.0
) -> ReaderRecord:
    return ReaderRecord(
        reader=reader,
        mode=mode,
        context_chunk_ids=tuple(chunk_id(chunk) for chunk in context),
        used_chunk_ids=(),
        passages_in_used_context=None,
        prediction=None,
        refused=False,
        confidence=None,
        answer_chunk_id=None,
        exact_match=None,
        token_f1=None,
        answer_containment=None,
        list_item_recall=None,
        lexical_support=None,
        correct=None,
        seconds=seconds,
        error=error,
    )


def _assemble(
    question: EvalQuestion, retrieved: _Retrieved, reader_records: list[ReaderRecord], readers: Sequence[str]
) -> QuestionRecord:
    end_to_end = []
    diagnosis = {}
    for name in readers:
        gold = next((r for r in reader_records if r.reader == name and r.mode == GOLD_CONTEXT), None)
        from_retrieval = next((r for r in reader_records if r.reader == name and r.mode == RETRIEVED_CONTEXT), None)
        outcome = derive_end_to_end(name, question.answerable, retrieved.routing, from_retrieval)
        end_to_end.append(outcome)
        diagnosis[name] = diagnose(question.answerable, retrieved.retrieval, retrieved.routing, gold, from_retrieval, outcome)

    errors = [retrieved.error] if retrieved.error else []
    errors += [f"{record.reader} {record.mode}: {record.error}" for record in reader_records if record.error]
    return QuestionRecord(
        id=question.id,
        question=question.question,
        split=question.split,
        category=question.category,
        answerable=question.answerable,
        failure_tags=question.failure_tags,
        answers=question.answers,
        list_items=question.list_items,
        evidence=question.evidence,
        retrieval=retrieved.retrieval,
        routing=retrieved.routing,
        readers=tuple(reader_records),
        end_to_end=tuple(end_to_end),
        diagnosis=diagnosis,
        errors=tuple(errors),
    )


def _select_questions(questions: Sequence[EvalQuestion], split: str, ids: Sequence[str] | None) -> list[EvalQuestion]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {list(SPLITS)}")
    selected = [question for question in questions if split == "all" or question.split == split]
    if ids:
        unknown = sorted(set(ids) - {question.id for question in questions})
        if unknown:
            raise ValueError(f"Unknown question ids {unknown}")
        selected = [question for question in selected if question.id in set(ids)]
    if not selected:
        raise ValueError("No questions match the selection")
    return selected


def _reader_settings(model: object) -> dict:
    names = ("model_name", "max_length", "doc_stride", "max_answer_tokens", "min_score", "max_input_tokens", "max_new_tokens", "num_beams")
    return {name: getattr(model, name) for name in names if hasattr(model, name)}


def _run_metadata(run: EvaluationRun, questions_path: Path, corpus_dir: Path, split: str, ids: Sequence[str] | None) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(),
        "python": sys.version.split()[0],
        "packages": {name: _package_version(name) for name in _PACKAGES},
        "settings": run.settings,
        "selection": {"split": split, "ids": list(ids) if ids else None, "questions": len(run.records)},
        "dataset": {"path": _display_path(questions_path), "sha256": _sha256(questions_path)},
        "corpus": {
            "path": _display_path(corpus_dir),
            "pdfs": {pdf.name: _sha256(pdf) for pdf in sorted(Path(corpus_dir).glob("*.pdf"))},
            "chunks": len(run.index_chunks),
        },
        "timings_seconds": {stage: round(seconds, 2) for stage, seconds in run.timings.items()},
        "questions_with_errors": sum(bool(record.errors) for record in run.records),
    }


def _git_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "uncommitted_changes": None if status is None else bool(status)}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    # Paths inside the project are stored relative to it, to keep user names out of saved runs.
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


if __name__ == "__main__":
    sys.exit(main())
