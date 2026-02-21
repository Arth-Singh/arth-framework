"""Toxicity scoring using HuggingFace transformers pipeline."""

from __future__ import annotations


_DEFAULT_MODEL = "unitary/toxic-bert"

_UNAVAILABLE_RESULT = {
    "toxicity": 0.0,
    "label": "non_toxic",
    "details": {},
    "note": "transformers library not installed; score unavailable",
}


def _check_transformers() -> bool:
    """Return True if the transformers library can be imported."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class ToxicityScorer:
    """Score text toxicity via a HuggingFace text-classification model.

    The model is loaded lazily on the first call to ``score`` or
    ``score_batch``, keeping import time low when the scorer is only
    instantiated but not used.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``unitary/toxic-bert``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda:0"``, etc.).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pipeline = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> None:
        """Load the classification pipeline on first use."""
        if self._pipeline is not None:
            return
        from transformers import pipeline  # deferred import

        self._pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            device=self._device,
            top_k=None,
            truncation=True,
            max_length=512,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the transformers library is importable."""
        return _check_transformers()

    def score(self, text: str) -> dict:
        """Score a single text for toxicity.

        Returns a dict with keys ``toxicity`` (float 0-1), ``label``
        (``"toxic"`` or ``"non_toxic"``), and ``details`` (per-category
        scores).  If transformers is not installed the result contains a
        ``note`` field explaining the situation.
        """
        if not self.is_available():
            return dict(_UNAVAILABLE_RESULT)

        self._ensure_pipeline()
        return self._parse(self._pipeline(text)[0])

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score multiple texts in one pass for better throughput.

        Returns a list of result dicts identical in structure to those
        returned by ``score``.
        """
        if not self.is_available():
            return [dict(_UNAVAILABLE_RESULT) for _ in texts]

        self._ensure_pipeline()
        raw_results = self._pipeline(texts)
        return [self._parse(entry) for entry in raw_results]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(labels: list[dict]) -> dict:
        """Convert raw pipeline output into a standardised result dict.

        ``unitary/toxic-bert`` emits labels such as ``toxic``,
        ``severe_toxic``, ``obscene``, ``threat``, ``insult``, and
        ``identity_hate``.  The overall toxicity score is the maximum
        across all categories.
        """
        details: dict[str, float] = {}
        max_score = 0.0

        for entry in labels:
            name = entry["label"]
            value = float(entry["score"])
            details[name] = value
            if name != "non_toxic":
                max_score = max(max_score, value)

        label = "toxic" if max_score >= 0.5 else "non_toxic"
        return {"toxicity": max_score, "label": label, "details": details}
