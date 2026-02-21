"""Tests for metrics computation in arth.eval.metrics."""

from __future__ import annotations

import pytest

from arth.eval.metrics import (
    attack_success_rate,
    coherence_score,
    compute_all_metrics,
    harmful_compliance_rate,
    mean_confidence,
    mean_similarity,
    mean_toxicity,
    refusal_delta,
    refusal_rate,
)


# ---------------------------------------------------------------------------
# Helpers to build scored result dicts
# ---------------------------------------------------------------------------

def _make_scored(
    orig_refused: bool,
    mod_refused: bool,
    modified_text: str = "Some response that is long enough to test.",
    *,
    orig_verdict: str | None = None,
    mod_verdict: str | None = None,
    mod_toxicity: dict | None = None,
    mod_similarity: float | None = None,
    mod_confidence: float = 0.9,
) -> dict:
    """Build a scored result dict matching Scorer.score_batch output."""
    orig_score = {
        "refused": orig_refused,
        "compliant": not orig_refused,
        "confidence": 0.9,
    }
    mod_score = {
        "refused": mod_refused,
        "compliant": not mod_refused,
        "confidence": mod_confidence,
    }
    if orig_verdict is not None:
        orig_score["verdict"] = orig_verdict
    if mod_verdict is not None:
        mod_score["verdict"] = mod_verdict
    if mod_toxicity is not None:
        mod_score["toxicity"] = mod_toxicity
    if mod_similarity is not None:
        mod_score["similarity"] = mod_similarity
    return {
        "prompt": "test prompt",
        "original": "original response",
        "modified": modified_text,
        "original_score": orig_score,
        "modified_score": mod_score,
    }


# ---------------------------------------------------------------------------
# attack_success_rate
# ---------------------------------------------------------------------------

class TestAttackSuccessRate:
    def test_all_refused_none_bypassed(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=True),
            _make_scored(orig_refused=True, mod_refused=True),
        ]
        assert attack_success_rate(scores) == 0.0

    def test_none_originally_refused(self) -> None:
        scores = [
            _make_scored(orig_refused=False, mod_refused=False),
            _make_scored(orig_refused=False, mod_refused=False),
        ]
        assert attack_success_rate(scores) == 0.0

    def test_all_bypassed_legacy(self) -> None:
        """Legacy mode (no verdict): compliant non-refusal = attack success."""
        scores = [
            _make_scored(orig_refused=True, mod_refused=False),
            _make_scored(orig_refused=True, mod_refused=False),
        ]
        assert attack_success_rate(scores) == 1.0

    def test_verdict_harmful_compliance_is_success(self) -> None:
        """With verdict field, only harmful_compliance counts."""
        scores = [
            _make_scored(
                orig_refused=True, mod_refused=False,
                orig_verdict="refusal", mod_verdict="harmful_compliance",
            ),
        ]
        assert attack_success_rate(scores) == 1.0

    def test_verdict_safe_compliance_is_not_success(self) -> None:
        """Safe compliance should NOT count as attack success."""
        scores = [
            _make_scored(
                orig_refused=True, mod_refused=False,
                orig_verdict="refusal", mod_verdict="safe_compliance",
            ),
        ]
        assert attack_success_rate(scores) == 0.0

    def test_verdict_ambiguous_is_not_success(self) -> None:
        scores = [
            _make_scored(
                orig_refused=True, mod_refused=False,
                orig_verdict="refusal", mod_verdict="ambiguous",
            ),
        ]
        assert attack_success_rate(scores) == 0.0

    def test_mixed_results_legacy(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=False),   # bypassed
            _make_scored(orig_refused=True, mod_refused=True),    # still refused
            _make_scored(orig_refused=True, mod_refused=False),   # bypassed
            _make_scored(orig_refused=False, mod_refused=False),  # was never refused
        ]
        assert abs(attack_success_rate(scores) - 2.0 / 3.0) < 1e-6

    def test_empty_input(self) -> None:
        assert attack_success_rate([]) == 0.0

    def test_single_sample_bypassed(self) -> None:
        scores = [_make_scored(orig_refused=True, mod_refused=False)]
        assert attack_success_rate(scores) == 1.0


# ---------------------------------------------------------------------------
# refusal_rate
# ---------------------------------------------------------------------------

class TestRefusalRate:
    def test_all_refused(self) -> None:
        scores = [
            _make_scored(orig_refused=False, mod_refused=True),
            _make_scored(orig_refused=False, mod_refused=True),
        ]
        assert refusal_rate(scores) == 1.0

    def test_none_refused(self) -> None:
        scores = [
            _make_scored(orig_refused=False, mod_refused=False),
            _make_scored(orig_refused=False, mod_refused=False),
        ]
        assert refusal_rate(scores) == 0.0

    def test_mixed(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=True),
            _make_scored(orig_refused=True, mod_refused=False),
            _make_scored(orig_refused=False, mod_refused=False),
            _make_scored(orig_refused=False, mod_refused=True),
        ]
        assert abs(refusal_rate(scores) - 0.5) < 1e-6

    def test_empty(self) -> None:
        assert refusal_rate([]) == 0.0


# ---------------------------------------------------------------------------
# refusal_delta
# ---------------------------------------------------------------------------

class TestRefusalDelta:
    def test_fewer_refusals_negative_delta(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=False),
            _make_scored(orig_refused=True, mod_refused=False),
        ]
        assert abs(refusal_delta(scores) - (-1.0)) < 1e-6

    def test_more_refusals_positive_delta(self) -> None:
        scores = [
            _make_scored(orig_refused=False, mod_refused=True),
            _make_scored(orig_refused=False, mod_refused=True),
        ]
        assert abs(refusal_delta(scores) - 1.0) < 1e-6

    def test_no_change(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=True),
            _make_scored(orig_refused=False, mod_refused=False),
        ]
        assert abs(refusal_delta(scores)) < 1e-6

    def test_empty(self) -> None:
        assert refusal_delta([]) == 0.0


# ---------------------------------------------------------------------------
# harmful_compliance_rate
# ---------------------------------------------------------------------------

class TestHarmfulComplianceRate:
    def test_all_harmful(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=False, mod_verdict="harmful_compliance"),
            _make_scored(orig_refused=True, mod_refused=False, mod_verdict="harmful_compliance"),
        ]
        assert harmful_compliance_rate(scores) == 1.0

    def test_none_harmful(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=False, mod_verdict="safe_compliance"),
            _make_scored(orig_refused=True, mod_refused=True, mod_verdict="refusal"),
        ]
        assert harmful_compliance_rate(scores) == 0.0

    def test_no_verdict_returns_zero(self) -> None:
        """Without verdict field, harmful_compliance_rate is 0."""
        scores = [_make_scored(orig_refused=True, mod_refused=False)]
        assert harmful_compliance_rate(scores) == 0.0


# ---------------------------------------------------------------------------
# mean_toxicity
# ---------------------------------------------------------------------------

class TestMeanToxicity:
    def test_with_toxicity_scores(self) -> None:
        scores = [
            _make_scored(False, False, mod_toxicity={"toxicity": 0.8, "label": "toxic"}),
            _make_scored(False, False, mod_toxicity={"toxicity": 0.2, "label": "non_toxic"}),
        ]
        assert abs(mean_toxicity(scores) - 0.5) < 1e-6

    def test_without_toxicity(self) -> None:
        scores = [_make_scored(False, False)]
        assert mean_toxicity(scores) == 0.0

    def test_empty(self) -> None:
        assert mean_toxicity([]) == 0.0


# ---------------------------------------------------------------------------
# mean_similarity
# ---------------------------------------------------------------------------

class TestMeanSimilarity:
    def test_with_similarity(self) -> None:
        scores = [
            _make_scored(False, False, mod_similarity=0.9),
            _make_scored(False, False, mod_similarity=0.3),
        ]
        assert abs(mean_similarity(scores) - 0.6) < 1e-6

    def test_without_similarity(self) -> None:
        scores = [_make_scored(False, False)]
        assert mean_similarity(scores) == 0.0


# ---------------------------------------------------------------------------
# coherence_score
# ---------------------------------------------------------------------------

class TestCoherenceScore:
    def test_empty_returns_zero(self) -> None:
        assert coherence_score([]) == 0.0

    def test_all_compliant_long_responses(self) -> None:
        responses = [
            "This is a well-written response with many unique words and good structure. " * 3,
            "Another thoughtful answer covering various aspects of the topic in detail. " * 3,
        ]
        score = coherence_score(responses)
        assert 0.0 < score <= 1.0

    def test_very_short_responses_low_score(self) -> None:
        responses = ["Ok", "Hi"]
        score = coherence_score(responses)
        assert score < 0.2

    def test_repetitive_text_penalized(self) -> None:
        non_repetitive = ["The quick brown fox jumps over the lazy dog near the river bank in the morning sun."]
        repetitive = ["the the the the the the the the the the the the the the the the the"]
        assert coherence_score(non_repetitive) > coherence_score(repetitive)


# ---------------------------------------------------------------------------
# mean_confidence
# ---------------------------------------------------------------------------

class TestMeanConfidence:
    def test_returns_average(self) -> None:
        scores = [
            _make_scored(False, False, mod_confidence=0.9),
            _make_scored(False, False, mod_confidence=0.5),
        ]
        assert abs(mean_confidence(scores) - 0.7) < 1e-6

    def test_empty(self) -> None:
        assert mean_confidence([]) == 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self, sample_scored_results: list[dict]) -> None:
        metrics = compute_all_metrics(sample_scored_results)
        expected_keys = {
            "attack_success_rate",
            "harmful_compliance_rate",
            "refusal_rate",
            "refusal_delta",
            "coherence_score",
            "mean_toxicity",
            "mean_similarity",
            "mean_confidence",
            "n_samples",
        }
        assert set(metrics.keys()) == expected_keys

    def test_empty_input(self) -> None:
        metrics = compute_all_metrics([])
        assert metrics["attack_success_rate"] == 0.0
        assert metrics["refusal_rate"] == 0.0
        assert metrics["refusal_delta"] == 0.0
        assert metrics["coherence_score"] == 0.0
        assert metrics["n_samples"] == 0.0

    def test_n_samples_is_correct(self, sample_scored_results: list[dict]) -> None:
        metrics = compute_all_metrics(sample_scored_results)
        assert metrics["n_samples"] == float(len(sample_scored_results))

    def test_values_are_floats(self, sample_scored_results: list[dict]) -> None:
        metrics = compute_all_metrics(sample_scored_results)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"

    def test_single_sample(self) -> None:
        scores = [_make_scored(orig_refused=True, mod_refused=False)]
        metrics = compute_all_metrics(scores)
        assert metrics["attack_success_rate"] == 1.0
        assert metrics["refusal_rate"] == 0.0
        assert metrics["n_samples"] == 1.0

    def test_all_identical(self) -> None:
        scores = [_make_scored(orig_refused=True, mod_refused=True)] * 10
        metrics = compute_all_metrics(scores)
        assert metrics["attack_success_rate"] == 0.0
        assert metrics["refusal_rate"] == 1.0
