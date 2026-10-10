"""Sentiment scoring for filing text.

Two backends, deliberately:

* ``lexicon`` -- word counting, no ML dependencies, runs in milliseconds.
* ``finbert`` -- a transformer fine-tuned on financial text, needs torch.

Having both is the point. Running the identical event set through both answers
a question worth asking: does a 110M-parameter transformer actually beat
counting words for this task? Reporting that comparison is more interesting
than either number alone, and it is cheap once the pipeline exists.

On the word list: the standard reference is the Loughran-McDonald finance
sentiment dictionary, which is distributed by its authors rather than bundled
here. Point ``--lexicon-csv`` at their file to use it. Without it, a small
built-in list is used; it is weaker and the README says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A deliberately small, hand-written fallback. Not a substitute for the
# Loughran-McDonald dictionary -- just enough to run end to end without it.
FALLBACK_POSITIVE = {
    "achieved", "advantage", "beneficial", "better", "boost", "strong", "strength",
    "exceeded", "exceeds", "favorable", "gain", "gains", "good", "greater", "growth",
    "improved", "improvement", "increase", "increased", "outperform", "positive",
    "profit", "profitable", "record", "rose", "strengthen", "succeed", "success",
    "successful", "surpassed", "upgrade", "upside", "win", "won",
}
FALLBACK_NEGATIVE = {
    "adverse", "adversely", "against", "breach", "ceased", "challenging", "claims",
    "closure", "concern", "decline", "declined", "decrease", "decreased", "default",
    "deficiency", "delay", "delayed", "deteriorate", "difficult", "discontinued",
    "dispute", "down", "downgrade", "doubt", "fail", "failed", "failure", "fell",
    "impairment", "investigation", "lawsuit", "litigation", "loss", "losses",
    "negative", "restructuring", "shortfall", "terminate", "terminated", "unable",
    "uncertain", "uncertainty", "weak", "weakness", "worse", "writedown",
}

# Words that flip the polarity of the next few tokens. "not profitable" scoring
# as positive is the classic failure mode of bag-of-words sentiment.
NEGATORS = {"no", "not", "never", "none", "nor", "cannot", "without", "unable", "failed"}
NEGATION_SCOPE = 3

_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class SentimentScore:
    """Score in [-1, 1], plus the counts behind it so it can be audited."""

    score: float
    positive: int
    negative: int
    tokens: int
    backend: str

    @property
    def polarity(self) -> str:
        if self.score > 0.01:
            return "positive"
        if self.score < -0.01:
            return "negative"
        return "neutral"


def load_lexicon(csv_path: Path | None) -> tuple[set[str], set[str]]:
    """Load a Loughran-McDonald style CSV, or fall back to the built-in list.

    The LM master dictionary has a ``Word`` column plus ``Positive``/``Negative``
    columns holding the year a word entered that category (0 means never).
    """
    if csv_path is None:
        return set(FALLBACK_POSITIVE), set(FALLBACK_NEGATIVE)

    import csv

    positive: set[str] = set()
    negative: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            lowered = {key.lower(): value for key, value in row.items() if key}
            word = (lowered.get("word") or "").strip().lower()
            if not word:
                continue
            if (lowered.get("positive") or "0").strip() not in {"0", ""}:
                positive.add(word)
            if (lowered.get("negative") or "0").strip() not in {"0", ""}:
                negative.add(word)
    if not positive and not negative:
        raise ValueError(f"no sentiment words parsed from {csv_path}")
    return positive, negative


class LexiconScorer:
    """Negation-aware word-count sentiment."""

    backend = "lexicon"

    def __init__(self, positive: set[str] | None = None, negative: set[str] | None = None):
        self.positive = positive if positive is not None else set(FALLBACK_POSITIVE)
        self.negative = negative if negative is not None else set(FALLBACK_NEGATIVE)

    def score(self, text: str) -> SentimentScore:
        tokens = _WORD.findall((text or "").lower())
        pos = neg = 0
        for i, token in enumerate(tokens):
            is_positive = token in self.positive
            is_negative = token in self.negative
            if not (is_positive or is_negative):
                continue
            window = tokens[max(0, i - NEGATION_SCOPE) : i]
            if any(word in NEGATORS for word in window):
                is_positive, is_negative = is_negative, is_positive
            pos += is_positive
            neg += is_negative

        total = pos + neg
        value = (pos - neg) / total if total else 0.0
        return SentimentScore(
            score=float(value), positive=pos, negative=neg, tokens=len(tokens), backend=self.backend
        )


class FinBertScorer:
    """ProsusAI/finbert. Requires torch and transformers."""

    backend = "finbert"
    MODEL = "ProsusAI/finbert"
    MAX_TOKENS = 512

    def __init__(self, device: str | None = None, batch_size: int = 16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        # finbert label order is (positive, negative, neutral)
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]

    def score(self, text: str) -> SentimentScore:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        results: list[SentimentScore] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [t or "" for t in texts[start : start + self.batch_size]]
            encoded = self.tokenizer(
                chunk, truncation=True, max_length=self.MAX_TOKENS, padding=True, return_tensors="pt"
            ).to(self.device)
            with self.torch.no_grad():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
            for text, row in zip(chunk, probabilities):
                scores = dict(zip(self.labels, row))
                # Net sentiment: P(positive) - P(negative), leaving neutral mass out.
                value = float(scores.get("positive", 0.0) - scores.get("negative", 0.0))
                results.append(
                    SentimentScore(
                        score=value,
                        positive=int(round(100 * scores.get("positive", 0.0))),
                        negative=int(round(100 * scores.get("negative", 0.0))),
                        tokens=len(text.split()),
                        backend=self.backend,
                    )
                )
        return results


def build_scorer(name: str, lexicon_csv: Path | None = None):
    if name == "lexicon":
        positive, negative = load_lexicon(lexicon_csv)
        return LexiconScorer(positive, negative)
    if name == "finbert":
        return FinBertScorer()
    raise ValueError(f"unknown scorer {name!r}; expected 'lexicon' or 'finbert'")
