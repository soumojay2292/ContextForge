import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.corpus import CORPUS_DIR
from src.evaluation.dataset import Evidence, load_questions
from src.evaluation.metrics import (
    REFUSAL_PHRASES,
    answer_containment,
    auroc,
    exact_match,
    false_accept_rate,
    false_refusal_rate,
    false_reject_rate,
    hit_at_k,
    is_refusal,
    is_relevant,
    lexical_support,
    list_item_recall,
    mean_reciprocal_rank,
    normalize_answer,
    reciprocal_rank,
    refusal_rate,
    token_f1,
)
from src.generation.flan_t5 import NOT_AVAILABLE_ANSWER
from src.ingestion.chunking import DocumentChunk, chunk_pages
from src.ingestion.pdf_loader import load_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def chunk(text: str, source: str = "corpus/manual.pdf", page_number: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(text=text, source=source, page_number=page_number, chunk_index=chunk_index)


BATTERY = Evidence(source="manual.pdf", page=1, text="The battery lasts 18 months.")
RESET = Evidence(source="manual.pdf", page=2, text="Hold the button for 15 seconds.")

BATTERY_CHUNK = chunk("Specs. The battery lasts 18 months. It weighs 160 grams.", page_number=1)
RESET_CHUNK = chunk("Reset. Hold the button for 15 seconds. The light flashes.", page_number=2, chunk_index=1)
OTHER_CHUNK = chunk("The housing is white.", page_number=1, chunk_index=2)


# --- is_relevant ------------------------------------------------------------


def test_chunk_containing_evidence_on_the_right_page_is_relevant():
    assert is_relevant(BATTERY_CHUNK, BATTERY)


def test_evidence_matches_after_cleaning():
    messy = chunk("Specs. The   battery\nlasts 18 months.", source="D:/data/eval/corpus/manual.pdf")

    assert is_relevant(messy, BATTERY)


def test_hyphenated_line_break_in_chunk_still_matches():
    hyphenated = chunk("Every update is verified with SHA-256 check-\nsums before it is installed.", page_number=3)

    assert is_relevant(hyphenated, Evidence("manual.pdf", 3, "verified with SHA-256 checksums"))


@pytest.mark.parametrize(
    "candidate",
    [
        chunk("The battery lasts 18 months.", page_number=2),  # wrong page
        chunk("The battery lasts 18 months.", source="corpus/report.pdf"),  # wrong document
        chunk("Specs. The battery lasts"),  # passage cut off by a chunk boundary
        OTHER_CHUNK,
    ],
    ids=["wrong page", "wrong source", "partial passage", "unrelated"],
)
def test_chunk_without_the_full_evidence_on_the_right_page_is_not_relevant(candidate: DocumentChunk):
    assert not is_relevant(candidate, BATTERY)


def test_empty_evidence_text_is_rejected():
    with pytest.raises(ValueError, match="Evidence text must not be empty"):
        is_relevant(BATTERY_CHUNK, Evidence("manual.pdf", 1, "  \n "))


# --- hit_at_k ---------------------------------------------------------------


@pytest.mark.parametrize(("k", "expected"), [(1, False), (2, False), (3, True), (10, True)])
def test_hit_at_k_depends_on_the_rank_of_the_evidence(k: int, expected: bool):
    ranked = [OTHER_CHUNK, RESET_CHUNK, BATTERY_CHUNK]  # BATTERY is at rank 3

    assert hit_at_k(ranked, [BATTERY], k) is expected


def test_hit_at_k_with_no_retrieved_chunks_is_a_miss():
    assert hit_at_k([], [BATTERY], k=5) is False


def test_hit_at_k_any_versus_all_evidence():
    ranked = [BATTERY_CHUNK, OTHER_CHUNK, RESET_CHUNK]

    assert hit_at_k(ranked, [BATTERY, RESET], k=1) is True
    assert hit_at_k(ranked, [BATTERY, RESET], k=1, require_all=True) is False
    assert hit_at_k(ranked, [BATTERY, RESET], k=3, require_all=True) is True


@pytest.mark.parametrize("k", [0, -1, 1.5, True, None])
def test_hit_at_k_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        hit_at_k([BATTERY_CHUNK], [BATTERY], k)


def test_hit_at_k_rejects_questions_without_evidence():
    with pytest.raises(ValueError, match="at least one evidence passage"):
        hit_at_k([BATTERY_CHUNK], [], k=1)


# --- reciprocal_rank and mean_reciprocal_rank -------------------------------


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        ([BATTERY_CHUNK, OTHER_CHUNK], 1.0),
        ([OTHER_CHUNK, BATTERY_CHUNK], 0.5),
        ([OTHER_CHUNK, OTHER_CHUNK, OTHER_CHUNK, BATTERY_CHUNK], 0.25),
        ([OTHER_CHUNK, RESET_CHUNK], 0.0),
        ([], 0.0),
    ],
    ids=["rank 1", "rank 2", "rank 4", "not retrieved", "nothing retrieved"],
)
def test_reciprocal_rank(ranked: list[DocumentChunk], expected: float):
    assert reciprocal_rank(ranked, [BATTERY]) == expected


def test_reciprocal_rank_uses_the_first_relevant_chunk():
    # Overlapping chunks can repeat a passage; only the best rank counts.
    assert reciprocal_rank([OTHER_CHUNK, BATTERY_CHUNK, BATTERY_CHUNK], [BATTERY]) == 0.5


def test_reciprocal_rank_with_several_passages_uses_whichever_comes_first():
    assert reciprocal_rank([OTHER_CHUNK, RESET_CHUNK, BATTERY_CHUNK], [BATTERY, RESET]) == 0.5


def test_reciprocal_rank_rejects_questions_without_evidence():
    with pytest.raises(ValueError, match="at least one evidence passage"):
        reciprocal_rank([BATTERY_CHUNK], [])


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == 0.5
    assert mean_reciprocal_rank([0.25]) == 0.25


def test_mean_reciprocal_rank_of_nothing_is_undefined():
    with pytest.raises(ValueError, match="No reciprocal ranks"):
        mean_reciprocal_rank([])


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_mean_reciprocal_rank_rejects_out_of_range_values(value: float):
    with pytest.raises(ValueError, match="between 0 and 1"):
        mean_reciprocal_rank([1.0, value])


# --- Retrieval metrics on the real dataset ----------------------------------


@pytest.mark.parametrize(("chunk_size", "chunk_overlap"), [(250, 50), (1000, 200)])
def test_retrieval_metrics_work_with_dataset_evidence(chunk_size: int, chunk_overlap: int):
    corpus_chunks = [
        corpus_chunk
        for pdf in sorted(CORPUS_DIR.glob("*.pdf"))
        for corpus_chunk in chunk_pages(load_pdf(pdf), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]

    for question in load_questions():
        if not question.answerable:
            continue
        relevant = [c for c in corpus_chunks if any(is_relevant(c, passage) for passage in question.evidence)]
        irrelevant = [c for c in corpus_chunks if c not in relevant]

        # Ranking every relevant chunk first must give a perfect score...
        assert reciprocal_rank(relevant + irrelevant, question.evidence) == 1.0, question.id
        assert hit_at_k(relevant + irrelevant, question.evidence, k=len(relevant), require_all=True), question.id
        # ...and leaving them out entirely must give zero.
        assert reciprocal_rank(irrelevant, question.evidence) == 0.0, question.id


# --- normalize_answer -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Cat!", "cat"),
        ("  A quick,  brown\nfox. ", "quick brown fox"),
        ("an apple and the theatre", "apple and theatre"),  # only whole-word articles go
        ("EUR 79,500", "eur 79500"),
        ("SHA-256 checksums", "sha256 checksums"),
        ("22%", "22"),
        ("", ""),
        ("The.", ""),
    ],
)
def test_normalize_answer(text: str, expected: str):
    assert normalize_answer(text) == expected


# --- exact_match ------------------------------------------------------------


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match("Chief Information Security Officer.", ["the chief information security officer"])


def test_exact_match_uses_the_best_of_several_references():
    assert exact_match("Frankfurt", ["a second data centre in Frankfurt", "Frankfurt"])


def test_exact_match_requires_the_whole_answer():
    assert not exact_match("in Frankfurt", ["Frankfurt"])
    assert not exact_match("", ["Frankfurt"])


# --- token_f1 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("Frankfurt", "Frankfurt", 1.0),
        # [red, cat] vs [red, dog]: 1 shared word, precision 1/2, recall 1/2 -> F1 1/2.
        ("a red cat", "the red dog", 0.5),
        # [red] vs [red, dog]: precision 1, recall 1/2 -> F1 2/3.
        ("red", "red dog", 2 / 3),
        # [red, red] vs [red]: a repeated word counts once -> precision 1/2, recall 1 -> F1 2/3.
        ("red red", "red", 2 / 3),
        ("blue", "red dog", 0.0),
        # SQuAD convention for empty answers.
        ("", "", 1.0),
        ("", "red", 0.0),
        ("red", "the", 0.0),
    ],
)
def test_token_f1(prediction: str, reference: str, expected: float):
    assert token_f1(prediction, [reference]) == pytest.approx(expected)


def test_token_f1_uses_the_best_reference():
    assert token_f1("red", ["blue", "red dog", "red"]) == 1.0


# --- answer_containment -----------------------------------------------------


def test_containment_finds_a_reference_inside_a_longer_answer():
    assert answer_containment("The battery lasts 18 months.", ["18 months"])


def test_containment_matches_whole_words_only():
    assert not answer_containment("Built in 2024.", ["2"])
    assert not answer_containment("eighteen", ["eight"])


def test_containment_accepts_any_reference():
    assert not answer_containment("Bake for 30 minutes", ["thirty minutes"])
    assert answer_containment("Bake for 30 minutes", ["thirty minutes", "30 minutes"])


def test_reference_that_normalizes_to_nothing_is_never_contained():
    assert not answer_containment("the answer", ["the"])


# --- list_item_recall -------------------------------------------------------

EQUIPMENT = ["laptop", "monitor", "headset"]  # q018
INCIDENT_STEPS = ["contain the incident", "assess the impact", "notify affected customers", "review the lessons learned"]  # q016
MEASUREMENTS = ["carbon dioxide", "temperature", "relative humidity", "fine particulate matter"]  # q017


@pytest.mark.parametrize(
    ("prediction", "expected_items", "expected"),
    [
        ("a laptop, a monitor and a headset", EQUIPMENT, 1.0),
        ("Headset, monitor and laptop.", EQUIPMENT, 1.0),  # order does not matter
        ("LAPTOP; Monitor!", EQUIPMENT, 2 / 3),  # case and punctuation ignored
        ("a laptop", EQUIPMENT, 1 / 3),
        ("a phone and a desk", EQUIPMENT, 0.0),
        ("", EQUIPMENT, 0.0),
        # Multi-word items: BERT returning only part of the list gets 2 of 4.
        ("contain the incident, assess the impact", INCIDENT_STEPS, 0.5),
        # "the" is dropped on both sides, so it need not match exactly.
        ("Contain incident, assess impact, notify affected customers, review the lessons learned.", INCIDENT_STEPS, 1.0),
        # The Orbit S1 distractor answer names 2 of the S2's 4 measurements.
        ("carbon dioxide and temperature", MEASUREMENTS, 0.5),
    ],
)
def test_list_item_recall(prediction: str, expected_items: list[str], expected: float):
    assert list_item_recall(prediction, expected_items) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("prediction", "item"),
    [
        ("two fans", "fan"),  # inside a longer word
        ("built in 2024", "2"),
        ("eighteen months", "eight"),
        ("the start", "art"),
        ("humidity relative to outdoors", "relative humidity"),  # words present, but not as a run
    ],
)
def test_list_item_recall_matches_whole_words_only(prediction: str, item: str):
    assert list_item_recall(prediction, [item]) == 0.0


def test_list_item_that_normalizes_to_nothing_is_never_found():
    # "the" normalizes to "", so only "laptop" of the two items can be found.
    assert list_item_recall("the laptop", ["laptop", "the"]) == 0.5


def test_list_item_recall_needs_at_least_one_item():
    with pytest.raises(ValueError, match="At least one expected list item"):
        list_item_recall("a laptop", [])


def test_list_item_recall_rejects_a_bare_string_as_items():
    with pytest.raises(ValueError, match="not a single string"):
        list_item_recall("a laptop", "laptop")


def test_dataset_list_answers_contain_all_their_items():
    list_questions = [question for question in load_questions() if question.category == "list"]

    assert list_questions
    for question in list_questions:
        assert list_item_recall(question.answers[0], question.list_items) == 1.0, question.id


# --- Reference validation, shared by the answer metrics ---------------------


@pytest.mark.parametrize("metric", [exact_match, token_f1, answer_containment])
def test_answer_metrics_need_at_least_one_reference(metric):
    with pytest.raises(ValueError, match="At least one reference"):
        metric("Frankfurt", [])


@pytest.mark.parametrize("metric", [exact_match, token_f1, answer_containment])
def test_answer_metrics_reject_a_bare_string_as_references(metric):
    with pytest.raises(ValueError, match="not a single string"):
        metric("Frankfurt", "Frankfurt")


# --- is_refusal and refusal_rate --------------------------------------------


@pytest.mark.parametrize(
    "prediction",
    [None, "", "   ", NOT_AVAILABLE_ANSWER, "not available", "Not available.", '"Unanswerable"', "No answer"],
)
def test_refusals_are_detected(prediction):
    assert is_refusal(prediction)


@pytest.mark.parametrize("prediction", ["Berlin", "2024", "not available in Berlin", "No", "no answer was given by the team"])
def test_answers_are_not_refusals(prediction: str):
    assert not is_refusal(prediction)


def test_refusal_phrases_are_configurable():
    assert is_refusal("I don't know", refusal_phrases=["I don't know"])
    assert not is_refusal("not available", refusal_phrases=["I don't know"])


def test_default_refusal_phrases_include_the_generator_message():
    assert NOT_AVAILABLE_ANSWER in REFUSAL_PHRASES


def test_refusal_rate():
    assert refusal_rate([True, False, False, True]) == 0.5
    assert refusal_rate([False]) == 0.0


def test_refusal_rate_of_nothing_is_undefined():
    with pytest.raises(ValueError, match="No predictions"):
        refusal_rate([])


# --- false_refusal_rate, false_reject_rate and false_accept_rate ------------

#                       q0     q1     q2     q3     q4
ANSWERABLE = [True, True, True, False, False]
REFUSED = [True, False, False, True, False]  # q0 wrongly refused, q4 wrongly answered
SUPPORTED = [False, True, True, True, False]  # q0 wrongly rejected, q3 wrongly accepted


def test_false_refusal_rate_counts_refused_answerable_questions():
    assert false_refusal_rate(REFUSED, ANSWERABLE) == pytest.approx(1 / 3)


def test_false_reject_rate_counts_unsupported_answerable_questions():
    assert false_reject_rate(SUPPORTED, ANSWERABLE) == pytest.approx(1 / 3)


def test_false_accept_rate_for_routing_counts_supported_unanswerable_questions():
    assert false_accept_rate(SUPPORTED, ANSWERABLE) == 0.5


def test_false_accept_rate_for_answers_counts_answered_unanswerable_questions():
    answered = [not refused for refused in REFUSED]

    assert false_accept_rate(answered, ANSWERABLE) == 0.5


def test_false_refusal_rate_equals_false_reject_rate_of_answered():
    answered = [not refused for refused in REFUSED]

    assert false_refusal_rate(REFUSED, ANSWERABLE) == false_reject_rate(answered, ANSWERABLE)


def test_perfect_and_worst_error_rates():
    assert false_reject_rate(ANSWERABLE, ANSWERABLE) == 0.0
    assert false_accept_rate(ANSWERABLE, ANSWERABLE) == 0.0
    accept_everything = [True] * len(ANSWERABLE)
    assert false_reject_rate(accept_everything, ANSWERABLE) == 0.0
    assert false_accept_rate(accept_everything, ANSWERABLE) == 1.0


@pytest.mark.parametrize("rate", [false_refusal_rate, false_reject_rate])
def test_rates_over_answerable_questions_need_some(rate):
    with pytest.raises(ValueError, match="No answerable questions"):
        rate([True, False], [False, False])


def test_false_accept_rate_needs_unanswerable_questions():
    with pytest.raises(ValueError, match="No unanswerable questions"):
        false_accept_rate([True, False], [True, True])


@pytest.mark.parametrize("rate", [false_refusal_rate, false_reject_rate, false_accept_rate, auroc])
def test_two_argument_metrics_reject_different_lengths(rate):
    with pytest.raises(ValueError, match="equal length"):
        rate([True, False, True], [True, False])


@pytest.mark.parametrize("rate", [false_refusal_rate, false_reject_rate, false_accept_rate])
def test_error_rates_of_nothing_are_undefined(rate):
    with pytest.raises(ValueError, match="No"):
        rate([], [])


# --- lexical_support --------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "sources", "expected"),
    [
        ("18 months", ["The battery lasts 18 months."], 1.0),
        ("The Berlin office.", ["Our office in Berlin"], 1.0),  # articles and punctuation ignored
        ("30 minutes", ["Bake for thirty minutes."], 0.5),  # "30" is not in the source
        ("Paris", ["The office is in Berlin."], 0.0),
        ("Berlin 2024", ["Built in 2024.", "Located in Berlin."], 1.0),  # words may come from different sources
        ("2", ["Built in 2024."], 0.0),  # whole words only
        ("Berlin", [], 0.0),
    ],
)
def test_lexical_support(answer: str, sources: list[str], expected: float):
    assert lexical_support(answer, sources) == expected


def test_lexical_support_counts_repeated_answer_words_each_time():
    # [berlin, berlin, paris]: two of three words are supported.
    assert lexical_support("Berlin Berlin Paris", ["Berlin"]) == pytest.approx(2 / 3)


@pytest.mark.parametrize("answer", ["", "   ", "the", "!?"])
def test_lexical_support_needs_an_answer_with_words(answer: str):
    with pytest.raises(ValueError, match="no words"):
        lexical_support(answer, ["The battery lasts 18 months."])


# --- auroc ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "answerable", "expected"),
    [
        ([0.9, 0.8, 0.2, 0.1], [True, True, False, False], 1.0),  # perfectly separated
        ([0.1, 0.2, 0.8, 0.9], [True, True, False, False], 0.0),  # perfectly inverted
        ([0.5, 0.5, 0.5, 0.5], [True, True, False, False], 0.5),  # all tied
        # Answerable 0.8, 0.4 vs unanswerable 0.6, 0.2: 0.8 beats both, 0.4 beats only 0.2 -> 3/4.
        ([0.8, 0.4, 0.6, 0.2], [True, True, False, False], 0.75),
        # Answerable 0.5 vs unanswerable 0.5, 0.1: one tie (1/2) and one win (1) -> 1.5/2.
        ([0.5, 0.5, 0.1], [True, False, False], 0.75),
        ([0.3, -0.2], [True, False], 1.0),  # negative cosine similarities are fine
    ],
)
def test_auroc(scores: list[float], answerable: list[bool], expected: float):
    assert auroc(scores, answerable) == pytest.approx(expected)


def test_auroc_does_not_depend_on_order():
    scores = [0.8, 0.4, 0.6, 0.2, 0.7]
    answerable = [True, True, False, False, True]
    order = [4, 2, 0, 3, 1]

    assert auroc(scores, answerable) == auroc([scores[i] for i in order], [answerable[i] for i in order])


@pytest.mark.parametrize(
    ("scores", "answerable"),
    [([0.9, 0.8], [True, True]), ([0.1, 0.2], [False, False]), ([], [])],
    ids=["only answerable", "only unanswerable", "empty"],
)
def test_auroc_needs_both_classes(scores: list[float], answerable: list[bool]):
    with pytest.raises(ValueError, match="at least one answerable and one unanswerable"):
        auroc(scores, answerable)


def test_auroc_rejects_nan_scores():
    with pytest.raises(ValueError, match="NaN"):
        auroc([0.9, math.nan], [True, False])


# --- Dependencies -----------------------------------------------------------


def test_importing_metrics_loads_no_model_libraries():
    heavy = ["torch", "transformers", "faiss", "sentence_transformers"]
    code = f"import sys, src.evaluation.metrics; print([m for m in {heavy!r} if m in sys.modules])"

    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
