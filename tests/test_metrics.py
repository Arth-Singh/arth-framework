"""Tests for metrics computation in arth.eval.metrics."""

from __future__ import annotations

import pytest

from arth.eval.metrics import (
    attack_success_rate,
    coherence_score,
    compute_all_metrics,
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
) -> dict:
    """Build a scored result dict matching Scorer.score_batch output."""
    return {
        "prompt": "test prompt",
        "original": "original response",
        "modified": modified_text,
        "original_score": {"refused": orig_refused, "compliant": not orig_refused, "confidence": 0.9},
        "modified_score": {"refused": mod_refused, "compliant": not mod_refused, "confidence": 0.9},
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

    def test_all_bypassed(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=False),
            _make_scored(orig_refused=True, mod_refused=False),
        ]
        assert attack_success_rate(scores) == 1.0

    def test_mixed_results(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=False),   # bypassed
            _make_scored(orig_refused=True, mod_refused=True),    # still refused
            _make_scored(orig_refused=True, mod_refused=False),   # bypassed
            _make_scored(orig_refused=False, mod_refused=False),  # was never refused
        ]
        # 3 originally refused, 2 bypassed => ASR = 2/3
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
        # original rate = 1.0, modified rate = 0.0 => delta = -1.0
        assert abs(refusal_delta(scores) - (-1.0)) < 1e-6

    def test_more_refusals_positive_delta(self) -> None:
        scores = [
            _make_scored(orig_refused=False, mod_refused=True),
            _make_scored(orig_refused=False, mod_refused=True),
        ]
        # original rate = 0.0, modified rate = 1.0 => delta = +1.0
        assert abs(refusal_delta(scores) - 1.0) < 1e-6

    def test_no_change(self) -> None:
        scores = [
            _make_scored(orig_refused=True, mod_refused=True),
            _make_scored(orig_refused=False, mod_refused=False),
        ]
        # original rate = 0.5, modified rate = 0.5 => delta = 0.0
        assert abs(refusal_delta(scores)) < 1e-6

    def test_empty(self) -> None:
        assert refusal_delta([]) == 0.0


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
        assert score < 0.2  # very short => ~0.1

    def test_repetitive_text_penalized(self) -> None:
        non_repetitive = ["The quick brown fox jumps over the lazy dog near the river bank in the morning sun."]
        repetitive = ["the the the the the the the the the the the the the the the the the"]
        score_non_rep = coherence_score(non_repetitive)
        score_rep = coherence_score(repetitive)
        assert score_non_rep > score_rep

    def test_single_response(self) -> None:
        responses = ["A reasonable response with enough words to pass the length threshold easily."]
        score = coherence_score(responses)
        assert 0.0 < score <= 1.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self, sample_scored_results: list[dict]) -> None:
        metrics = compute_all_metrics(sample_scored_results)
        expected_keys = {
            "attack_success_rate",
            "refusal_rate",
            "refusal_delta",
            "coherence_score",
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
