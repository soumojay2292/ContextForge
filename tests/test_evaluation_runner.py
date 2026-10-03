import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.corpus import CORPUS_DIR, PAGE_BREAK, build_pdf
from src.evaluation.dataset import EvalQuestion, Evidence
from src.evaluation.records import (
    ANSWERED,
    CONTEXT_DROPPED,
    CORRECT,
    ERROR,
    FALSE_ACCEPT,
    GOLD_CONTEXT,
    READER_DISTRACTED,
    READER_ERROR,
    REFUSED_BY_READER,
    REFUSED_BY_ROUTER,
    RETRIEVAL_MISS,
    RETRIEVED_CONTEXT,
    ROUTER_REJECT,
    EndToEndRecord,
    ReaderRecord,
    RetrievalRecord,
    RoutingRecord,
    chunk_id,
    derive_end_to_end,
    diagnose,
    load_records,
    write_records,
)
from src.evaluation.runner import build_index_chunks, evaluate, gold_context, run_evaluation
from src.generation.flan_t5 import NOT_AVAILABLE_ANSWER, GeneratedAnswer
from src.ingestion.chunking import DocumentChunk, chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.qa.bert_qa import ExtractedAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- A small corpus: two documents, three short pages each, one chunk per page.

PAGES = {
    "alpha": [
        "Alpha Manual\nThe widget is blue. The widget weighs 3 kilograms.",
        "Outdoor use\nCan the widget be used outdoors? No, the widget is for indoor use only.",
        "Storage\nThe widget must be stored dry. Spare parts are kept in Oslo. The kit contains a cable, a charger and a manual.",
    ],
    "beta": [
        "Beta Report\nThe beta trial used the widget in 12 schools.",
        "Beta costs\nThe beta trial cost 500 euros. A red gadget was cheaper.",
        "Beta outcome\nThe trial team recommends the widget for all schools.",
    ],
}
A1, A2, A3, B1, B2, B3 = ("alpha", 1), ("alpha", 2), ("alpha", 3), ("beta", 1), ("beta", 2), ("beta", 3)
NOISE = "noise"


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("corpus")
    for name, pages in PAGES.items():
        source = directory / f"{name}.txt"
        source.write_text(f"\n{PAGE_BREAK}\n".join(pages) + "\n")
        build_pdf(source, directory / f"{name}.pdf")
        source.unlink()
    return directory


class ScriptedEmbedder:
    """Stand-in for EmbeddingModel whose similarity scores are set by each test.

    Every page has its own dimension, so a chunk's vector points at its page.
    A question's vector is given in `queries` as weights per page, plus an
    optional NOISE weight that matches no chunk and lowers every score. Other
    questions embed to zero and score 0 against every chunk.
    """

    def __init__(self, queries: dict[str, dict], fail_on: tuple[str, ...] = ()):
        self.keys = [A1, A2, A3, B1, B2, B3, NOISE]
        self.dimension = len(self.keys)
        self.queries = queries
        self.fail_on = fail_on
        self.model_name = "scripted-embedder"

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        vectors = np.zeros((len(chunks), self.dimension), dtype=np.float32)
        for row, chunk in enumerate(chunks):
            vectors[row, self.keys.index((Path(chunk.source).stem, chunk.page_number))] = 1.0
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        if query in self.fail_on:
            raise RuntimeError("fake embedding failure")
        vector = np.zeros(self.dimension, dtype=np.float32)
        for key, weight in self.queries.get(query, {}).items():
            vector[self.keys.index(key)] = weight
        return vector


class FakeBert:
    """Extractive stand-in for BertQA: a scripted span is returned only if a context chunk contains it.

    A span in `distractors` wins whenever its chunk is in the context, as if the
    model were misled by it.
    """

    def __init__(self, spans: dict[str, str], distractors: dict[str, str] | None = None, fail_on: tuple[str, ...] = ()):
        self.spans = spans
        self.distractors = distractors or {}
        self.fail_on = fail_on
        self.calls = 0

    def answer(self, question: str, chunks: list[DocumentChunk]) -> ExtractedAnswer | None:
        self.calls += 1
        if question in self.fail_on:
            raise RuntimeError("fake BERT failure")
        for spans in (self.distractors, self.spans):
            span = spans.get(question)
            for chunk in chunks:
                if span and span in chunk.text:
                    start = chunk.text.index(span)
                    return ExtractedAnswer(text=span, score=0.9, chunk=chunk, start_char=start, end_char=start + len(span))
        return None


class FakeFlan:
    """Generative stand-in for FlanT5Generator that reads only its first `capacity` chunks.

    A scripted answer is given only if a read chunk contains it; `free_answers`
    are generated regardless of the context.
    """

    def __init__(self, answers: dict[str, str], free_answers: dict[str, str] | None = None, capacity: int = 2):
        self.answers = answers
        self.free_answers = free_answers or {}
        self.capacity = capacity
        self.calls = 0

    def generate(self, question: str, chunks: list[DocumentChunk]) -> GeneratedAnswer:
        self.calls += 1
        used = tuple(chunks[: self.capacity])
        if question in self.free_answers:
            return GeneratedAnswer(text=self.free_answers[question], sources=used)
        answer = self.answers.get(question)
        if answer and any(answer.lower() in chunk.text.lower() for chunk in used):
            return GeneratedAnswer(text=answer, sources=used)
        return GeneratedAnswer(text=NOT_AVAILABLE_ANSWER, sources=used)


def evidence(page: tuple[str, int], text: str) -> Evidence:
    return Evidence(source=f"{page[0]}.pdf", page=page[1], text=text)


def make_question(question_id, text, category, answers=(), passages=(), list_items=()) -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        question=text,
        answerable=bool(passages),
        category=category,
        answers=tuple(answers),
        evidence=tuple(passages),
        failure_tags=(),
        split="test",
        list_items=tuple(list_items),
    )


# --- Scenario: one question per outcome, with scripted retrieval and readers.

SCENARIO = [
    # (question, query weights, expected BERT diagnosis, expected FLAN diagnosis)
    (make_question("correct", "What colour is the widget?", "factoid", ["blue"], [evidence(A1, "The widget is blue.")]),
     {A1: 1.0}, CORRECT, CORRECT),
    (make_question("yes_no_extractable", "Can the widget be used outdoors?", "yes_no", ["no"],
                   [evidence(A2, "No, the widget is for indoor use only.")]),
     {A2: 1.0}, CORRECT, CORRECT),
    (make_question("yes_no_not_extractable", "Must the widget be stored dry?", "yes_no", ["yes"],
                   [evidence(A3, "The widget must be stored dry.")]),
     {A3: 1.0}, READER_ERROR, CORRECT),
    (make_question("list", "What does the kit contain?", "list", ["a cable, a charger and a manual"],
                   [evidence(A3, "The kit contains a cable, a charger and a manual.")], ["cable", "charger", "manual"]),
     {A3: 1.0}, READER_ERROR, CORRECT),
    # Five other pages outscore the evidence page, so it falls outside the top 5.
    (make_question("miss", "How many schools used the widget in the beta trial?", "factoid", ["12 schools"],
                   [evidence(B1, "The beta trial used the widget in 12 schools.")]),
     {A1: 1.0, A2: 1.0, A3: 1.0, B2: 1.0, B3: 1.0}, RETRIEVAL_MISS, RETRIEVAL_MISS),
    # The evidence ranks first, but noise keeps its score (0.15) under the 0.2 threshold.
    (make_question("router_reject", "How much did the beta trial cost?", "numeric", ["500 euros"],
                   [evidence(B2, "The beta trial cost 500 euros.")]),
     {B2: 0.15, NOISE: 0.99}, ROUTER_REJECT, ROUTER_REJECT),
    # The evidence ranks third: BERT reads it, FLAN only reads the top two.
    (make_question("dropped", "What does the trial team recommend?", "factoid", ["the widget for all schools"],
                   [evidence(B3, "The trial team recommends the widget for all schools.")]),
     {A1: 1.0, A2: 0.9, B3: 0.8}, CORRECT, CONTEXT_DROPPED),
    # The retrieved distractor page misleads BERT; the gold context does not contain it.
    (make_question("distracted", "Where are spare parts kept?", "factoid", ["Oslo"],
                   [evidence(A3, "Spare parts are kept in Oslo.")]),
     {A3: 1.0, B2: 0.5}, READER_DISTRACTED, CORRECT),
    (make_question("multi_section", "What colour is the widget used in the beta trial?", "multi_section", ["blue"],
                   [evidence(B1, "The beta trial used the widget in 12 schools."), evidence(A1, "The widget is blue.")]),
     {B1: 1.0, A1: 0.9}, CORRECT, CORRECT),
    (make_question("refused_by_router", "What is the capital of Japan?", "unanswerable_off_topic"),
     {}, REFUSED_BY_ROUTER, REFUSED_BY_ROUTER),
    (make_question("refused_by_reader", "Who designed the widget?", "unanswerable_near_domain"),
     {A1: 1.0}, REFUSED_BY_READER, REFUSED_BY_READER),
    (make_question("false_accept", "How long is the widget warranty?", "unanswerable_near_domain"),
     {A1: 1.0}, FALSE_ACCEPT, FALSE_ACCEPT),
    (make_question("bert_fails", "Why is the widget blue?", "factoid", ["blue"], [evidence(A1, "The widget is blue.")]),
     {A1: 1.0}, ERROR, CORRECT),
]
QUESTIONS = [question for question, *_ in SCENARIO]
QUERIES = {question.question: weights for question, weights, *_ in SCENARIO}
BERT_SPANS = {
    "What colour is the widget?": "blue",
    "Can the widget be used outdoors?": "No",
    "Must the widget be stored dry?": "yes",  # not in the text, so BERT cannot extract it
    "What does the kit contain?": "a cable, a charger",  # only part of the list
    "How many schools used the widget in the beta trial?": "12 schools",
    "How much did the beta trial cost?": "500 euros",
    "What does the trial team recommend?": "the widget for all schools",
    "Where are spare parts kept?": "Oslo",
    "What colour is the widget used in the beta trial?": "blue",
}
BERT_DISTRACTORS = {"Where are spare parts kept?": "500 euros", "How long is the widget warranty?": "3 kilograms"}
FLAN_ANSWERS = {**BERT_SPANS, "Can the widget be used outdoors?": "No", "Why is the widget blue?": "blue"}
FLAN_FREE_ANSWERS = {
    "Must the widget be stored dry?": "Yes",
    "What does the kit contain?": "a manual, a charger and a cable",  # every item, in another order
    "How long is the widget warranty?": "two years",
}


@pytest.fixture(scope="module")
def scenario(corpus_dir: Path):
    events: list[str] = []
    bert = FakeBert(BERT_SPANS, BERT_DISTRACTORS, fail_on=("Why is the widget blue?",))
    flan = FakeFlan(FLAN_ANSWERS, FLAN_FREE_ANSWERS)

    def make(name, model):
        def factory():
            events.append(name)
            return model

        return factory

    run = run_evaluation(
        QUESTIONS,
        corpus_dir,
        make_embedder=make("embedder", ScriptedEmbedder(QUERIES)),
        make_bert=make("bert", bert),
        make_flan=make("flan", flan),
    )
    return {"run": run, "records": {record.id: record for record in run.records}, "events": events, "bert": bert, "flan": flan}


# --- Index ------------------------------------------------------------------


def test_index_uses_default_chunking_in_document_order(corpus_dir: Path):
    chunks = build_index_chunks(corpus_dir)

    assert [chunk_id(chunk) for chunk in chunks] == [
        "alpha.pdf:p1:c0", "alpha.pdf:p2:c0", "alpha.pdf:p3:c0", "beta.pdf:p1:c0", "beta.pdf:p2:c0", "beta.pdf:p3:c0",
    ]


def test_evaluation_corpus_index_matches_default_chunk_pages():
    expected = [chunk for pdf in sorted(CORPUS_DIR.glob("*.pdf")) for chunk in chunk_pages(load_pdf(pdf), 1000, 200)]

    assert build_index_chunks(CORPUS_DIR) == expected


def test_index_of_an_empty_directory_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_index_chunks(tmp_path)


# --- Gold context -----------------------------------------------------------


def page_chunk(text: str, source: str = "manual.pdf", page: int = 1, index: int = 0) -> DocumentChunk:
    return DocumentChunk(text=text, source=f"corpus/{source}", page_number=page, chunk_index=index)


def test_gold_context_is_the_chunk_containing_the_evidence():
    chunks = [page_chunk("Intro text."), page_chunk("The battery lasts 18 months.", page=2)]
    question = make_question("q", "Battery?", "numeric", ["18 months"], [Evidence("manual.pdf", 2, "The battery lasts 18 months.")])

    assert gold_context(question, chunks) == [chunks[1]]


def test_gold_context_uses_the_earlier_of_two_overlapping_chunks():
    earlier = page_chunk("Setup is easy. The battery lasts 18 months.", index=0)
    later = page_chunk("The battery lasts 18 months. Reset with the button.", index=1)
    question = make_question("q", "Battery?", "numeric", ["18 months"], [Evidence("manual.pdf", 1, "The battery lasts 18 months.")])

    assert gold_context(question, [later, earlier]) == [earlier]


def test_gold_context_for_multi_section_keeps_dataset_order():
    report = page_chunk("The pilot used the S2 sensor.", source="report.pdf")
    manual = page_chunk("The S2 battery lasts 18 months.", source="manual.pdf")
    question = make_question(
        "q", "Battery of the pilot sensor?", "multi_section", ["18 months"],
        [Evidence("report.pdf", 1, "The pilot used the S2 sensor."), Evidence("manual.pdf", 1, "The S2 battery lasts 18 months.")],
    )

    assert gold_context(question, [manual, report]) == [report, manual]


def test_gold_context_lists_a_chunk_shared_by_two_passages_once():
    shared = page_chunk("The budget was 84,000. The cost was 79,500.")
    question = make_question(
        "q", "Savings?", "multi_section", ["4,500"],
        [Evidence("manual.pdf", 1, "The budget was 84,000."), Evidence("manual.pdf", 1, "The cost was 79,500.")],
    )

    assert gold_context(question, [shared]) == [shared]


def test_unanswerable_question_has_no_gold_context():
    assert gold_context(make_question("q", "Capital of Japan?", "unanswerable_off_topic"), [page_chunk("Text.")]) == []


def test_gold_context_reports_evidence_found_in_no_chunk():
    question = make_question("q", "Battery?", "numeric", ["18 months"], [Evidence("manual.pdf", 1, "Not in any chunk.")])

    with pytest.raises(ValueError, match="no chunk contains the evidence"):
        gold_context(question, [page_chunk("The battery lasts 18 months.")])


# --- Diagnosis of every scenario --------------------------------------------


@pytest.mark.parametrize(
    ("question_id", "bert_label", "flan_label"),
    [(question.id, bert_label, flan_label) for question, _, bert_label, flan_label in SCENARIO],
)
def test_scenario_diagnoses(scenario, question_id: str, bert_label: str, flan_label: str):
    assert scenario["records"][question_id].diagnosis == {"bert": bert_label, "flan": flan_label}


def test_scenario_covers_every_diagnosis_label(scenario):
    labels = {label for record in scenario["run"].records for label in record.diagnosis.values()}

    assert labels == {
        CORRECT, RETRIEVAL_MISS, ROUTER_REJECT, CONTEXT_DROPPED, READER_ERROR, READER_DISTRACTED,
        REFUSED_BY_ROUTER, REFUSED_BY_READER, FALSE_ACCEPT, ERROR,
    }


# --- Retrieval and routing --------------------------------------------------


def test_multi_section_retrieval_records_each_passage(scenario):
    retrieval = scenario["records"]["multi_section"].retrieval

    assert retrieval.passage_ranks == (1, 2)
    assert retrieval.results[0].passages == (0,)
    assert retrieval.results[1].passages == (1,)
    assert (retrieval.hit_any_at_1, retrieval.hit_all_at_1, retrieval.hit_all_at_3) == (True, False, True)
    assert retrieval.reciprocal_rank == 1.0


def test_retrieval_miss_records_the_missing_passage(scenario):
    retrieval = scenario["records"]["miss"].retrieval

    assert retrieval.passage_ranks == (None,)
    assert (retrieval.hit_any_at_5, retrieval.reciprocal_rank) == (False, 0.0)
    assert len(retrieval.results) == 5


def test_routing_record_keeps_status_reason_and_threshold(scenario):
    routing = scenario["records"]["router_reject"].routing

    assert (routing.status, routing.supported, routing.threshold, routing.k) == ("below_threshold", False, 0.2, 5)
    assert routing.top_score == pytest.approx(0.15, abs=0.001)
    assert "below the threshold" in routing.reason


def test_unanswerable_questions_have_scores_but_no_retrieval_metrics(scenario):
    record = scenario["records"]["refused_by_router"]

    assert record.retrieval.top_score == 0.0
    assert record.retrieval.passage_ranks == ()
    assert record.retrieval.hit_any_at_5 is None and record.retrieval.reciprocal_rank is None
    assert len(record.retrieval.results) == 5


# --- Reader contexts --------------------------------------------------------


def test_retrieved_context_is_the_retrieval_results_in_rank_order(scenario):
    record = scenario["records"]["dropped"]
    retrieved_ids = tuple(result.chunk_id for result in record.retrieval.results)
    bert = record.reader_record("bert", RETRIEVED_CONTEXT)
    flan = record.reader_record("flan", RETRIEVED_CONTEXT)

    assert bert.context_chunk_ids == flan.context_chunk_ids == retrieved_ids
    assert bert.used_chunk_ids == retrieved_ids
    assert flan.used_chunk_ids == retrieved_ids[:2]
    assert (bert.passages_in_used_context, flan.passages_in_used_context) == ((True,), (False,))


def test_gold_context_reader_records_use_the_evidence_chunks(scenario):
    record = scenario["records"]["multi_section"]

    for reader in ("bert", "flan"):
        gold = record.reader_record(reader, GOLD_CONTEXT)
        assert gold.context_chunk_ids == ("beta.pdf:p1:c0", "alpha.pdf:p1:c0")
        assert gold.passages_in_used_context == (True, True)


def test_unanswerable_questions_are_only_read_with_retrieved_context(scenario):
    record = scenario["records"]["refused_by_reader"]

    assert [(r.reader, r.mode) for r in record.readers] == [("bert", RETRIEVED_CONTEXT), ("flan", RETRIEVED_CONTEXT)]
    assert all(r.correct is None and r.exact_match is None and r.passages_in_used_context is None for r in record.readers)


# --- Scoring ----------------------------------------------------------------


def test_bert_answers_a_yes_no_question_when_the_answer_is_an_extractable_span(scenario):
    bert = scenario["records"]["yes_no_extractable"].reader_record("bert", RETRIEVED_CONTEXT)

    assert (bert.prediction, bert.refused, bert.answer_containment, bert.correct) == ("No", False, True, True)
    assert bert.answer_chunk_id == "alpha.pdf:p2:c0"


def test_bert_without_an_extractable_yes_no_span_is_scored_like_any_refusal(scenario):
    record = scenario["records"]["yes_no_not_extractable"]

    for mode in (GOLD_CONTEXT, RETRIEVED_CONTEXT):
        bert = record.reader_record("bert", mode)
        assert (bert.prediction, bert.refused, bert.correct, bert.token_f1) == (None, True, False, 0.0)


def test_list_questions_are_correct_only_with_every_item(scenario):
    record = scenario["records"]["list"]
    bert = record.reader_record("bert", RETRIEVED_CONTEXT)
    flan = record.reader_record("flan", RETRIEVED_CONTEXT)

    assert (bert.list_item_recall, bert.correct) == (pytest.approx(2 / 3), False)
    # Every item in a different order: not contained verbatim, but correct by item recall.
    assert (flan.answer_containment, flan.list_item_recall, flan.correct) == (False, 1.0, True)


def test_correct_answers_get_full_answer_metrics(scenario):
    bert = scenario["records"]["correct"].reader_record("bert", RETRIEVED_CONTEXT)

    assert (bert.exact_match, bert.token_f1, bert.answer_containment, bert.correct) == (True, 1.0, True, True)
    assert bert.confidence == 0.9 and bert.list_item_recall is None


def test_refusals_get_no_credit_from_their_wording(scenario):
    flan = scenario["records"]["dropped"].reader_record("flan", RETRIEVED_CONTEXT)

    assert flan.prediction == NOT_AVAILABLE_ANSWER
    assert (flan.refused, flan.exact_match, flan.token_f1, flan.answer_containment, flan.correct) == (True, False, 0.0, False, False)


def test_lexical_support_is_recorded_for_generated_answers_only(scenario):
    record = scenario["records"]["false_accept"]

    assert record.reader_record("flan", RETRIEVED_CONTEXT).lexical_support == 0.0  # "two years" is in no chunk
    assert record.reader_record("bert", RETRIEVED_CONTEXT).lexical_support is None
    assert scenario["records"]["correct"].reader_record("flan", RETRIEVED_CONTEXT).lexical_support == 1.0


# --- End to end -------------------------------------------------------------


def test_end_to_end_follows_routing_and_the_retrieved_context_answer(scenario):
    for record in scenario["run"].records:
        for outcome in record.end_to_end:
            retrieved = record.reader_record(outcome.reader, RETRIEVED_CONTEXT)
            if not record.routing.supported:
                assert (outcome.outcome, outcome.prediction) == (REFUSED_BY_ROUTER, None)
            elif retrieved.error:
                assert outcome.outcome == ERROR
            else:
                assert outcome.prediction == retrieved.prediction


def test_end_to_end_needs_no_extra_inference(scenario):
    answerable = sum(question.answerable for question in QUESTIONS)
    # One gold-context call per answerable question and one retrieved-context call per question.
    assert scenario["bert"].calls == scenario["flan"].calls == answerable + len(QUESTIONS)


def test_router_reject_keeps_what_the_reader_would_have_answered(scenario):
    record = scenario["records"]["router_reject"]

    assert [outcome.outcome for outcome in record.end_to_end] == [REFUSED_BY_ROUTER, REFUSED_BY_ROUTER]
    assert record.reader_record("bert", RETRIEVED_CONTEXT).correct is True


# --- Model loading and failures ---------------------------------------------


def test_each_model_is_created_once_in_stage_order(scenario):
    assert scenario["events"] == ["embedder", "bert", "flan"]


def test_settings_record_the_default_configuration(scenario):
    settings = scenario["run"].settings

    assert (settings["chunk_size"], settings["chunk_overlap"]) == (1000, 200)
    assert (settings["similarity_threshold"], settings["router_k"]) == (0.2, 5)
    assert settings["embedding_model"] == "scripted-embedder"


def test_failed_reader_call_is_recorded_and_the_run_continues(scenario):
    record = scenario["records"]["bert_fails"]

    for mode in (GOLD_CONTEXT, RETRIEVED_CONTEXT):
        bert = record.reader_record("bert", mode)
        assert bert.error == "RuntimeError: fake BERT failure"
        assert bert.prediction is None and bert.correct is None
    assert record.errors == (
        "bert gold_context: RuntimeError: fake BERT failure",
        "bert retrieved_context: RuntimeError: fake BERT failure",
    )
    assert record.reader_record("flan", RETRIEVED_CONTEXT).correct is True


def test_reader_that_fails_to_load_is_recorded_for_every_question(corpus_dir: Path):
    def broken_flan():
        raise MemoryError("not enough memory")

    run = run_evaluation(
        QUESTIONS[:2],
        corpus_dir,
        make_embedder=lambda: ScriptedEmbedder(QUERIES),
        make_bert=lambda: FakeBert(BERT_SPANS),
        make_flan=broken_flan,
    )

    for record in run.records:
        assert record.diagnosis == {"bert": CORRECT, "flan": ERROR}
        flan_records = [r for r in record.readers if r.reader == "flan"]
        assert flan_records and all(r.error == "flan failed to load: MemoryError: not enough memory" for r in flan_records)


def test_retrieval_failure_is_recorded_and_gold_context_still_runs(corpus_dir: Path):
    question = QUESTIONS[0]

    run = run_evaluation(
        [question],
        corpus_dir,
        make_embedder=lambda: ScriptedEmbedder(QUERIES, fail_on=(question.question,)),
        make_bert=lambda: FakeBert(BERT_SPANS),
        make_flan=lambda: FakeFlan(FLAN_ANSWERS),
    )
    record = run.records[0]

    assert record.retrieval is None and record.routing is None
    assert record.errors[0] == "retrieval and routing failed: RuntimeError: fake embedding failure"
    assert record.diagnosis == {"bert": ERROR, "flan": ERROR}
    assert record.reader_record("bert", GOLD_CONTEXT).correct is True
    assert record.reader_record("bert", RETRIEVED_CONTEXT).error == "no retrieved context: retrieval failed"


def test_running_one_reader_does_not_load_the_other(corpus_dir: Path):
    def forbidden():
        raise AssertionError("BERT should not be loaded")

    run = run_evaluation(
        QUESTIONS[:1],
        corpus_dir,
        readers=("flan",),
        make_embedder=lambda: ScriptedEmbedder(QUERIES),
        make_bert=forbidden,
        make_flan=lambda: FakeFlan(FLAN_ANSWERS),
    )

    assert run.records[0].diagnosis == {"flan": CORRECT}
    assert {r.reader for r in run.records[0].readers} == {"flan"}


def test_unknown_reader_is_rejected(corpus_dir: Path):
    with pytest.raises(ValueError, match="Unknown readers"):
        run_evaluation(QUESTIONS[:1], corpus_dir, readers=("gpt",))


# --- derive_end_to_end and diagnose on hand-built records -------------------


def reader_record(*, mode=RETRIEVED_CONTEXT, prediction="blue", refused=False, correct=True, passages=(True,), error=None):
    return ReaderRecord(
        reader="bert", mode=mode, context_chunk_ids=(), used_chunk_ids=(), passages_in_used_context=passages,
        prediction=prediction, refused=refused, confidence=None, answer_chunk_id=None, exact_match=None,
        token_f1=None, answer_containment=None, list_item_recall=None, lexical_support=None, correct=correct,
        seconds=0.0, error=error,
    )


def routing_record(supported: bool) -> RoutingRecord:
    return RoutingRecord(status="supported" if supported else "below_threshold", supported=supported, reason="",
                         top_score=0.5, threshold=0.2, k=5)


def retrieval_record(passage_ranks: tuple) -> RetrievalRecord:
    return RetrievalRecord(results=(), top_score=0.5, passage_ranks=passage_ranks, hit_any_at_1=None, hit_any_at_3=None,
                           hit_any_at_5=None, hit_all_at_1=None, hit_all_at_3=None, hit_all_at_5=None, reciprocal_rank=None)


@pytest.mark.parametrize(
    ("answerable", "routing", "retrieved", "expected"),
    [
        (True, None, reader_record(), (ERROR, None, None)),
        (True, routing_record(False), reader_record(), (REFUSED_BY_ROUTER, None, False)),
        (False, routing_record(False), reader_record(), (REFUSED_BY_ROUTER, None, True)),
        (True, routing_record(True), reader_record(error="boom"), (ERROR, None, None)),
        (True, routing_record(True), None, (ERROR, None, None)),
        (True, routing_record(True), reader_record(prediction=None, refused=True, correct=False), (REFUSED_BY_READER, None, False)),
        (False, routing_record(True), reader_record(prediction=None, refused=True, correct=None), (REFUSED_BY_READER, None, True)),
        (True, routing_record(True), reader_record(), (ANSWERED, "blue", True)),
        (True, routing_record(True), reader_record(correct=False), (ANSWERED, "blue", False)),
        (False, routing_record(True), reader_record(correct=None), (ANSWERED, "blue", False)),
    ],
)
def test_derive_end_to_end(answerable, routing, retrieved, expected):
    outcome = derive_end_to_end("bert", answerable, routing, retrieved)

    assert (outcome.outcome, outcome.prediction, outcome.correct) == expected


WRONG = EndToEndRecord(reader="bert", outcome=ANSWERED, prediction="red", correct=False)


@pytest.mark.parametrize(
    ("answerable", "retrieval", "routing", "gold", "retrieved", "end_to_end", "expected"),
    [
        (True, None, None, None, None, EndToEndRecord("bert", ERROR, None, None), ERROR),
        (False, None, None, None, None, EndToEndRecord("bert", REFUSED_BY_ROUTER, None, True), REFUSED_BY_ROUTER),
        (False, None, None, None, None, EndToEndRecord("bert", REFUSED_BY_READER, None, True), REFUSED_BY_READER),
        (False, None, None, None, None, EndToEndRecord("bert", ANSWERED, "red", False), FALSE_ACCEPT),
        (True, None, None, None, None, EndToEndRecord("bert", ANSWERED, "blue", True), CORRECT),
        (True, retrieval_record((1, None)), routing_record(True), None, None, WRONG, RETRIEVAL_MISS),
        (True, retrieval_record((1,)), routing_record(False), None, None,
         EndToEndRecord("bert", REFUSED_BY_ROUTER, None, False), ROUTER_REJECT),
        (True, retrieval_record((3,)), routing_record(True), None, reader_record(passages=(False,)), WRONG, CONTEXT_DROPPED),
        (True, retrieval_record((1,)), routing_record(True), reader_record(mode=GOLD_CONTEXT, correct=False),
         reader_record(correct=False), WRONG, READER_ERROR),
        (True, retrieval_record((1,)), routing_record(True), reader_record(mode=GOLD_CONTEXT, correct=True),
         reader_record(correct=False), WRONG, READER_DISTRACTED),
        (True, retrieval_record((1,)), routing_record(True), reader_record(mode=GOLD_CONTEXT, error="boom"),
         reader_record(correct=False), WRONG, ERROR),
        (True, retrieval_record((1,)), routing_record(True), None, reader_record(correct=False), WRONG, ERROR),
    ],
    ids=[
        "pipeline error", "unanswerable refused by router", "unanswerable refused by reader", "false accept",
        "correct", "multi-section passage missing", "router reject", "context dropped", "reader error",
        "reader distracted", "gold run failed", "gold run missing",
    ],
)
def test_diagnose(answerable, retrieval, routing, gold, retrieved, end_to_end, expected):
    assert diagnose(answerable, retrieval, routing, gold, retrieved, end_to_end) == expected


# --- Saving runs ------------------------------------------------------------


def test_records_round_trip_through_json_lines(scenario, tmp_path: Path):
    path = tmp_path / "records.jsonl"

    write_records(path, scenario["run"].records)

    assert load_records(path) == scenario["run"].records
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(QUESTIONS)


def write_dataset(path: Path, questions: list[EvalQuestion]) -> None:
    lines = []
    for question in questions:
        lines.append(json.dumps({
            "id": question.id, "question": question.question, "answerable": question.answerable,
            "category": question.category, "answers": list(question.answers),
            "evidence": [{"source": e.source, "page": e.page, "text": e.text} for e in question.evidence],
            "failure_tags": [], "split": question.split,
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_evaluate_saves_run_index_and_records_without_overwriting(corpus_dir: Path, tmp_path: Path):
    questions_path = tmp_path / "questions.jsonl"
    write_dataset(questions_path, [QUESTIONS[0], QUESTIONS[9]])  # "correct" and "refused_by_router"
    output_dir = tmp_path / "run"

    def run_once():
        return evaluate(
            questions_path, corpus_dir, output_dir,
            make_embedder=lambda: ScriptedEmbedder(QUERIES),
            make_bert=lambda: FakeBert(BERT_SPANS),
            make_flan=lambda: FakeFlan(FLAN_ANSWERS),
        )

    assert run_once() == output_dir
    metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    index = [json.loads(line) for line in (output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    records = load_records(output_dir / "records.jsonl")

    assert metadata["selection"] == {"split": "all", "ids": None, "questions": 2}
    assert metadata["settings"]["chunk_size"] == 1000 and metadata["corpus"]["chunks"] == 6
    assert set(metadata["corpus"]["pdfs"]) == {"alpha.pdf", "beta.pdf"}
    assert metadata["dataset"]["path"] == "questions.jsonl"  # outside the project: file name only
    assert [entry["chunk_id"] for entry in index][:2] == ["alpha.pdf:p1:c0", "alpha.pdf:p2:c0"]
    assert index[0]["text"] == "Alpha Manual The widget is blue. The widget weighs 3 kilograms."
    assert [record.diagnosis for record in records] == [
        {"bert": CORRECT, "flan": CORRECT}, {"bert": REFUSED_BY_ROUTER, "flan": REFUSED_BY_ROUTER},
    ]
    with pytest.raises(FileExistsError):
        run_once()


def test_evaluate_runs_only_the_selected_questions(corpus_dir: Path, tmp_path: Path):
    questions_path = tmp_path / "questions.jsonl"
    write_dataset(questions_path, [QUESTIONS[0], QUESTIONS[9]])

    output_dir = evaluate(
        questions_path, corpus_dir, tmp_path / "run", ids=["correct"],
        make_embedder=lambda: ScriptedEmbedder(QUERIES),
        make_bert=lambda: FakeBert(BERT_SPANS),
        make_flan=lambda: FakeFlan(FLAN_ANSWERS),
    )

    assert [record.id for record in load_records(output_dir / "records.jsonl")] == ["correct"]
    with pytest.raises(ValueError, match="Unknown question ids"):
        evaluate(questions_path, corpus_dir, tmp_path / "other", ids=["nope"])


def test_evaluate_rejects_an_invalid_dataset_before_loading_any_model(corpus_dir: Path, tmp_path: Path):
    questions_path = tmp_path / "questions.jsonl"
    broken = make_question("broken", "What colour?", "factoid", ["blue"], [evidence(A1, "The widget is green.")])
    write_dataset(questions_path, [broken])

    def forbidden():
        raise AssertionError("no model should be loaded")

    with pytest.raises(ValueError, match="evidence not found"):
        evaluate(questions_path, corpus_dir, tmp_path / "run", make_embedder=forbidden, make_bert=forbidden, make_flan=forbidden)
    assert not (tmp_path / "run").exists()


# --- Dependencies -----------------------------------------------------------


def test_importing_the_runner_loads_no_model_library():
    heavy = ["torch", "transformers", "sentence_transformers"]
    code = f"import sys, src.evaluation.runner; print([m for m in {heavy!r} if m in sys.modules])"

    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
