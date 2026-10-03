"""Metrics for scoring retrieval, answers, refusals and routing.

Every function is deterministic, pure Python and loads no model; nothing here
imports PyTorch, Transformers, FAISS or Sentence Transformers. Functions that
average over questions raise ValueError when the average is undefined, for
example a false-refusal rate over a set with no answerable questions.

Routing and answering share the same error rates, phrased as "accepted":
for the router, accepted means supported; for an answering model, accepted
means it gave an answer instead of refusing.
"""

import math
import re
import string
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from src.evaluation.dataset import Evidence
from src.generation.flan_t5 import NOT_AVAILABLE_ANSWER
from src.ingestion.chunking import DocumentChunk, clean_text

# Normalized predictions equal to one of these count as refusals.
REFUSAL_PHRASES = (NOT_AVAILABLE_ANSWER, "not available", "unanswerable", "no answer")

_PUNCTUATION = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


# --- Retrieval --------------------------------------------------------------


def is_relevant(chunk: DocumentChunk, evidence: Evidence) -> bool:
    """Whether a chunk contains an evidence passage.

    The chunk must come from the evidence's PDF and page and contain its text,
    compared after clean_text. Matching on text instead of chunk ids keeps the
    labels valid for any chunk size or overlap.

    Raises:
        ValueError: If the evidence text is empty.
    """
    passage = clean_text(evidence.text)
    if not passage:
        raise ValueError("Evidence text must not be empty")
    return (
        Path(chunk.source).name == evidence.source
        and chunk.page_number == evidence.page
        and passage in clean_text(chunk.text)
    )


def hit_at_k(
    chunks: Sequence[DocumentChunk], evidence: Sequence[Evidence], k: int, require_all: bool = False
) -> bool:
    """Whether the top k retrieved chunks contain the evidence.

    Args:
        chunks: Retrieved chunks, most similar first.
        evidence: The question's evidence passages.
        k: How many of the top chunks to look at.
        require_all: If False, retrieving any one passage counts as a hit. If
            True, every passage must be retrieved, which is what multi_section
            questions need in order to be answered.

    Raises:
        ValueError: If k is not a positive integer, or evidence is empty.
    """
    _check_k(k)
    _check_evidence(evidence)
    top_chunks = chunks[:k]
    found = [any(is_relevant(chunk, passage) for chunk in top_chunks) for passage in evidence]
    return all(found) if require_all else any(found)


def reciprocal_rank(chunks: Sequence[DocumentChunk], evidence: Sequence[Evidence]) -> float:
    """1 / rank of the first chunk containing any evidence passage, or 0.0 if none does.

    Raises:
        ValueError: If evidence is empty.
    """
    _check_evidence(evidence)
    for rank, chunk in enumerate(chunks, start=1):
        if any(is_relevant(chunk, passage) for passage in evidence):
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Average of per-question reciprocal ranks.

    Raises:
        ValueError: If reciprocal_ranks is empty or a value is outside 0 to 1.
    """
    if any(not 0.0 <= value <= 1.0 for value in reciprocal_ranks):
        raise ValueError("Reciprocal ranks must be between 0 and 1")
    return _mean(reciprocal_ranks, "reciprocal ranks")


# --- Answer quality ---------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation and the articles a, an and the, and collapse whitespace.

    This matches the official SQuAD evaluation script. Removing punctuation
    joins some tokens: "79,500" becomes "79500" and "SHA-256" becomes "sha256".
    """
    text = text.lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: Sequence[str]) -> bool:
    """Whether the prediction equals any reference after normalize_answer.

    Raises:
        ValueError: If references is empty or a single string.
    """
    _check_references(references)
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(reference) for reference in references)


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Best SQuAD-style token F1 between the prediction and any reference.

    F1 combines precision (share of predicted words that are in the reference)
    and recall (share of reference words that were predicted), counting
    repeated words at most as often as they occur in both.

    Raises:
        ValueError: If references is empty or a single string.
    """
    _check_references(references)
    return max(_token_f1(prediction, reference) for reference in references)


def answer_containment(prediction: str, references: Sequence[str]) -> bool:
    """Whether the prediction contains any reference as a run of whole words.

    Both sides are normalized first, so "The battery lasts 18 months." contains
    "18 months", but "2024" does not contain "2". A reference that normalizes
    to nothing, such as "the", is never contained.

    Raises:
        ValueError: If references is empty or a single string.
    """
    _check_references(references)
    padded_prediction = f" {normalize_answer(prediction)} "
    return any(
        normalized and f" {normalized} " in padded_prediction
        for normalized in (normalize_answer(reference) for reference in references)
    )


def list_item_recall(prediction: str, expected_items: Sequence[str]) -> float:
    """Share of the expected list items that appear in the prediction.

    Each item is matched like answer_containment: as a run of whole words after
    normalize_answer, in any order. So "notify affected customers" is found in
    "Contain the incident and notify affected customers.", but "fan" is not
    found in "fans" and "2" is not found in "2024". An item that normalizes to
    nothing, such as "the", is never found.

    Raises:
        ValueError: If expected_items is empty or a single string.
    """
    if isinstance(expected_items, str):
        # A bare string would be read as one item per character.
        raise ValueError("expected_items must be a sequence of strings, not a single string")
    if not expected_items:
        raise ValueError("At least one expected list item is needed")
    padded_prediction = f" {normalize_answer(prediction)} "
    found = sum(
        bool(normalized) and f" {normalized} " in padded_prediction
        for normalized in (normalize_answer(item) for item in expected_items)
    )
    return found / len(expected_items)


# --- Grounding and refusals -------------------------------------------------


def is_refusal(prediction: str | None, refusal_phrases: Sequence[str] = REFUSAL_PHRASES) -> bool:
    """Whether a prediction declines to answer.

    None (BertQA found no answer), empty text, and text that normalizes to one
    of refusal_phrases (FlanT5Generator's not-available message, for example)
    count as refusals. An answer that only contains a refusal phrase, such as
    "not available in Berlin", does not.
    """
    if prediction is None:
        return True
    normalized = normalize_answer(prediction)
    return not normalized or normalized in {normalize_answer(phrase) for phrase in refusal_phrases}


def refusal_rate(refused: Sequence[bool]) -> float:
    """Share of predictions that are refusals.

    Raises:
        ValueError: If refused is empty.
    """
    return _mean([bool(value) for value in refused], "predictions")


def false_refusal_rate(refused: Sequence[bool], answerable: Sequence[bool]) -> float:
    """Share of answerable questions that were refused.

    The same as false_reject_rate with accepted = not refused.

    Raises:
        ValueError: If the lengths differ, or no question is answerable.
    """
    return false_reject_rate([not value for value in refused], answerable)


def lexical_support(answer: str, sources: Sequence[str]) -> float:
    """Share of the answer's words that appear anywhere in the source texts.

    Words are compared after normalize_answer. 1.0 means every word of the
    answer occurs in the sources; lower values flag words the model added,
    such as an invented name or a number written differently ("30" for
    "thirty"). It cannot catch a wrong answer copied from the sources.

    Raises:
        ValueError: If the answer has no words after normalization.
    """
    words = normalize_answer(answer).split()
    if not words:
        raise ValueError("Answer has no words to check")
    source_words = {word for source in sources for word in normalize_answer(source).split()}
    return sum(word in source_words for word in words) / len(words)


# --- Routing (and answering) error rates ------------------------------------


def false_reject_rate(accepted: Sequence[bool], answerable: Sequence[bool]) -> float:
    """Share of answerable questions that were not accepted.

    Raises:
        ValueError: If the lengths differ, or no question is answerable.
    """
    _check_same_length(accepted, answerable)
    rejected = [not was_accepted for was_accepted, is_answerable in zip(accepted, answerable) if is_answerable]
    return _mean(rejected, "answerable questions")


def false_accept_rate(accepted: Sequence[bool], answerable: Sequence[bool]) -> float:
    """Share of unanswerable questions that were accepted.

    Raises:
        ValueError: If the lengths differ, or every question is answerable.
    """
    _check_same_length(accepted, answerable)
    wrongly_accepted = [bool(was_accepted) for was_accepted, is_answerable in zip(accepted, answerable) if not is_answerable]
    return _mean(wrongly_accepted, "unanswerable questions")


def auroc(scores: Sequence[float], answerable: Sequence[bool]) -> float:
    """Area under the ROC curve for telling answerable from unanswerable questions by score.

    This is the probability that a randomly chosen answerable question scores
    higher than a randomly chosen unanswerable one, with ties counting half.
    1.0 means some threshold separates them perfectly, 0.5 is no better than
    chance, and the value does not depend on any particular threshold.

    Raises:
        ValueError: If the lengths differ, a score is NaN, or there is not at
            least one answerable and one unanswerable question.
    """
    _check_same_length(scores, answerable)
    if any(math.isnan(score) for score in scores):
        raise ValueError("Scores must not be NaN")
    positives = [score for score, is_answerable in zip(scores, answerable) if is_answerable]
    negatives = [score for score, is_answerable in zip(scores, answerable) if not is_answerable]
    if not positives or not negatives:
        raise ValueError("AUROC needs at least one answerable and one unanswerable question")

    # Compare every answerable score with every unanswerable one: a win counts 1, a tie 0.5.
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


# --- Helpers ----------------------------------------------------------------


def _token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        # SQuAD convention: two empty answers agree; an empty and a non-empty one do not.
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _check_k(k: int) -> None:
    # bool is a subclass of int, but k=True is a mistake rather than k=1.
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _check_evidence(evidence: Sequence[Evidence]) -> None:
    if not evidence:
        raise ValueError("Retrieval metrics need at least one evidence passage; unanswerable questions have none")


def _check_references(references: Sequence[str]) -> None:
    if isinstance(references, str):
        # A bare string would be read as one reference per character.
        raise ValueError("references must be a sequence of strings, not a single string")
    if not references:
        raise ValueError("At least one reference answer is needed")


def _check_same_length(first: Sequence, second: Sequence) -> None:
    if len(first) != len(second):
        raise ValueError(f"Expected sequences of equal length, got {len(first)} and {len(second)}")


def _mean(values: Sequence[float], what: str) -> float:
    if not values:
        raise ValueError(f"No {what} to average over")
    return sum(values) / len(values)
