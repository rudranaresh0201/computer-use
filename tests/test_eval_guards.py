import pytest

from computeruse.eval import guards
from computeruse.web.browser import WebElement

VIEWPORT = (1000, 1000)  # boxes in 0-1000 units already
SAVE = WebElement("e1", "button", "Save", (100, 100, 50, 50))     # center (125, 125)
ICON = WebElement("e2", "button", "", (300, 100, 50, 50))         # center (325, 125), no name
ICON2 = WebElement("e3", "button", "", (500, 100, 50, 50))        # center (525, 125), no name
TREE = [SAVE, ICON, ICON2]


@pytest.mark.parametrize("rule", guards.GUARDS)
def test_none_hands_over_to_vision(rule):
    point = guards.decide(rule, {"ref": None}, (530, 120), TREE, VIEWPORT)
    assert point == ((525, 125) if rule == "llm_snap" else (530, 120))


def test_baseline_trusts_any_pick_named_trusts_only_names():
    nameless_pick = {"ref": "e2"}
    vision = (530, 120)  # on ICON2 -- vision disagrees with the LLM's nameless pick
    assert guards.decide("llm_guarded", nameless_pick, vision, TREE, VIEWPORT) == (325, 125)
    assert guards.decide("llm_named", nameless_pick, vision, TREE, VIEWPORT) == (530, 120)
    assert guards.decide("llm_named", {"ref": "e1"}, vision, TREE, VIEWPORT) == (125, 125)


def test_agree_trusts_a_nameless_pick_only_when_vision_lands_on_it():
    assert guards.decide("llm_agree", {"ref": "e2"}, (310, 140), TREE, VIEWPORT) == (325, 125)
    assert guards.decide("llm_agree", {"ref": "e2"}, (530, 120), TREE, VIEWPORT) == (530, 120)


def test_unknown_ref_counts_as_none():
    assert guards.decide("llm_guarded", {"ref": "e99"}, (1, 2), TREE, VIEWPORT) == (1, 2)


def test_choose_prefers_the_simplest_rule_within_one_point():
    assert guards.choose({"llm_guarded": 0.80, "llm_named": 0.845, "llm_agree": 0.85, "llm_snap": 0.84}) == "llm_named"
    assert guards.choose({"llm_guarded": 0.80, "llm_named": 0.82, "llm_agree": 0.85, "llm_snap": 0.84}) == "llm_agree"
