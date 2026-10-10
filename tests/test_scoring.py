"""Tests for the lexicon scorer. FinBERT is not tested here: it needs torch
and a model download, and it is exercised by the end-to-end run instead."""

from __future__ import annotations

import pytest

from filing_alpha.scoring import LexiconScorer, build_scorer, load_lexicon


@pytest.fixture
def scorer():
    return LexiconScorer()


class TestLexiconScorer:
    def test_positive_text_scores_positive(self, scorer):
        result = scorer.score("Revenue growth exceeded guidance and margins improved.")
        assert result.score > 0
        assert result.polarity == "positive"
        assert result.positive > 0

    def test_negative_text_scores_negative(self, scorer):
        result = scorer.score("The company reported a material impairment and a widening loss.")
        assert result.score < 0
        assert result.polarity == "negative"

    def test_neutral_text_scores_zero(self, scorer):
        result = scorer.score("The registrant filed this report pursuant to the rules.")
        assert result.score == 0.0
        assert result.polarity == "neutral"

    def test_empty_text_is_safe(self, scorer):
        for value in ("", None):
            result = scorer.score(value)
            assert result.score == 0.0
            assert result.tokens == 0

    def test_score_is_bounded(self, scorer):
        assert scorer.score("growth gains profit success " * 50).score == pytest.approx(1.0)
        assert scorer.score("loss decline failure weakness " * 50).score == pytest.approx(-1.0)

    def test_negation_flips_polarity(self, scorer):
        """'not profitable' must not read as positive -- the classic bag-of-words bug."""
        plain = scorer.score("The segment was profitable.")
        negated = scorer.score("The segment was not profitable.")
        assert plain.score > 0
        assert negated.score < 0

    def test_negation_only_reaches_nearby_words(self, scorer):
        """A negator should not flip a sentiment word far later in the sentence."""
        result = scorer.score("no comment was provided although results improved")
        assert result.score > 0

    def test_counts_are_reported_for_auditing(self, scorer):
        result = scorer.score("growth and profit but also loss")
        assert result.positive == 2
        assert result.negative == 1
        assert result.tokens == 6
        assert result.backend == "lexicon"


class TestLoadLexicon:
    def test_falls_back_without_a_csv(self):
        positive, negative = load_lexicon(None)
        assert "growth" in positive
        assert "impairment" in negative

    def test_parses_a_loughran_mcdonald_style_csv(self, tmp_path):
        csv_path = tmp_path / "lm.csv"
        csv_path.write_text(
            "Word,Negative,Positive\nEXCELLENT,0,2009\nLITIGATION,2009,0\nTHE,0,0\n",
            encoding="utf-8",
        )
        positive, negative = load_lexicon(csv_path)
        assert positive == {"excellent"}
        assert negative == {"litigation"}

    def test_empty_csv_raises(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("Word,Negative,Positive\nTHE,0,0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no sentiment words"):
            load_lexicon(csv_path)


class TestBuildScorer:
    def test_builds_the_lexicon_backend(self):
        assert build_scorer("lexicon").backend == "lexicon"

    def test_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown scorer"):
            build_scorer("vader")
