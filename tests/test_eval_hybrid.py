"""Tests for the hybrid arm combiner (eval/hybrid.py). Pure logic -- no
model, no GPU, no dataset -- so this is buildable and verifiable before a
fine-tuned checkpoint exists to supply the real fallback side."""

import pytest

from computeruse.eval.hybrid import combine
from computeruse.eval.uia_only import UiaArmResult


def _uia(example_id, available, hit, app="notepad", richness="rich", split="dev"):
    return UiaArmResult(
        example_id=example_id, app=app, app_richness=richness, split=split,
        available=available, hit=hit,
    )


def test_hybrid_uses_uia_when_available_regardless_of_grounder():
    uia_results = [_uia("e1", available=True, hit=True)]
    # grounder disagrees -- hybrid must still trust UIA since it resolved
    grounder_hits = {"e1": False}
    records = combine(uia_results, grounder_hits)
    assert records[0].hit is True
    assert records[0].arm == "hybrid"


def test_hybrid_falls_back_to_grounder_when_uia_unavailable():
    uia_results = [_uia("e1", available=False, hit=False)]
    grounder_hits = {"e1": True}
    records = combine(uia_results, grounder_hits)
    assert records[0].hit is True


def test_hybrid_falls_back_and_still_misses_if_grounder_missed_too():
    uia_results = [_uia("e1", available=False, hit=False)]
    grounder_hits = {"e1": False}
    records = combine(uia_results, grounder_hits)
    assert records[0].hit is False


def test_hybrid_raises_rather_than_silently_defaulting_on_missing_fallback():
    uia_results = [_uia("e1", available=False, hit=False)]
    with pytest.raises(KeyError):
        combine(uia_results, grounder_hits={})


def test_hybrid_preserves_app_richness_and_split_from_the_uia_result():
    uia_results = [_uia("e1", available=True, hit=True, app="paint", richness="weak", split="test_same_app")]
    records = combine(uia_results, grounder_hits={})
    assert records[0].app == "paint"
    assert records[0].app_richness == "weak"
    assert records[0].split == "test_same_app"
