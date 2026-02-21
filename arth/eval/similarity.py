"""Semantic similarity scoring for prompt-response topical relevance."""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

_HAS_SENTENCE_TRANSFORMERS = True
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore[assignment,misc]


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-overlap fallback: Jaccard similarity on lowered words > 3 chars."""
    words_a = {w.lower() for w in re.findall(r"\w+", text_a) if len(w) > 3}
    words_b = {w.lower() for w in re.findall(r"\w+", text_b) if len(w) > 3}
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class SemanticSimilarity:
    """Measures semantic similarity between two texts using sentence embeddings.

    Falls back to Jaccard word-overlap similarity when sentence-transformers
    is not installed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._using_fallback = not _HAS_SENTENCE_TRANSFORMERS

        if self._using_fallback:
            logger.warning(
                "sentence-transformers not installed; "
                "using Jaccard word-overlap fallback."
            )

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model on first use."""
        if self._model is not None or self._using_fallback:
            return
        self._model = SentenceTransformer(self._model_name, device=self._device)

    def is_available(self) -> bool:
        """Return True if sentence-transformers is installed and usable."""
        return _HAS_SENTENCE_TRANSFORMERS

    def score(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts, clamped to [0, 1].

        Returns a Jaccard word-overlap score when the transformer model
        is unavailable.
        """
        if self._using_fallback:
            return _jaccard_similarity(text_a, text_b)

        self._load_model()
        embeddings = self._model.encode([text_a, text_b], convert_to_numpy=True)
        cosine = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10
        ))
        return float(np.clip(cosine, 0.0, 1.0))

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a batch of (text_a, text_b) pairs.

        Returns:
            List of similarity scores in the same order as the input pairs.
        """
        if not pairs:
            return []

        if self._using_fallback:
            return [_jaccard_similarity(a, b) for a, b in pairs]

        self._load_model()
        texts_a = [a for a, _ in pairs]
        texts_b = [b for _, b in pairs]
        emb_a = self._model.encode(texts_a, convert_to_numpy=True)
        emb_b = self._model.encode(texts_b, convert_to_numpy=True)

        # Row-wise cosine similarity
        dot = np.sum(emb_a * emb_b, axis=1)
        norms = np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1) + 1e-10
        cosines = np.clip(dot / norms, 0.0, 1.0)
        return cosines.tolist()
