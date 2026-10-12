"""The sandbox test set (sandbox/) and the tree/vision/hybrid scoring
(eval/compare.py): the generated conditions really hold on a rendered page,
and each arm answers by its stated rule."""

from __future__ import annotations

import json

import pytest

from computeruse.eval import compare
from computeruse.sandbox.build import shares_a_word
from computeruse.web.browser import WebElement

VIEWPORT = (1000, 1000)  # so pixel boxes read directly as 0-1000 units in these tests


def el(name, box, role="button", ref="e1"):
    return WebElement(ref, role, name, box)


# -- conditions ----------------------------------------------------------------


def test_a_reworded_instruction_may_not_share_a_word_or_stem_with_the_label():
    assert shares_a_word("send this to the printer", "Print")  # same 4-letter stem
    assert shares_a_word("click Save", "Save")
    assert not shares_a_word("keep my work", "Save")


# -- the tree arm --------------------------------------------------------------


def test_a_name_inside_the_instruction_is_a_full_match_and_a_paraphrase_is_not():
    assert compare.name_score("press the Save button", "Save") == 1.0
    assert compare.name_score("reset the Privacy settings", "Reset") == 1.0
    assert compare.name_score("keep my work", "Save") < compare.MATCH_THRESHOLD
    assert compare.name_score("anything", "") == 0.0


def test_tree_answer_takes_the_first_of_equal_matches_and_gives_up_below_threshold():
    elements = [el("Reset", (0, 0, 10, 10), ref="e1"), el("Reset", (50, 0, 10, 10), ref="e2"), el("Save", (90, 0, 10, 10))]
    assert compare.tree_answer("click Reset in the Billing panel", elements).ref == "e1"
    assert compare.tree_answer("keep my work", elements) is None
    assert compare.tree_answer("click Save", [el("Save", (0, 0, 10, 10), role="heading")]) is None  # not clickable


def test_snap_moves_a_point_into_the_element_it_falls_in_or_near():
    icon = el("", (100, 100, 20, 20), ref="icon")
    toolbar = el("Tools", (0, 90, 400, 40), ref="bar")
    assert compare.snap((105, 105), [toolbar, icon], VIEWPORT) == (110, 110)  # smallest containing element
    assert compare.snap((128, 112), [icon], VIEWPORT) == (110, 110)  # 8 units away: within the radius
    assert compare.snap((300, 300), [icon], VIEWPORT) == (300, 300)  # nothing near: unchanged
    assert compare.snap(None, [icon], VIEWPORT) is None


def test_each_arm_answers_by_its_own_rule():
    item = {"id": "p1-c2", "condition": "C2", "instruction": "click the gear icon", "gt_box_norm": [100, 100, 120, 120]}
    elements = [el("", (100, 100, 20, 20), ref="gear"), el("Settings", (0, 500, 80, 30), role="link", ref="nav")]
    scored = compare.score_item(item, elements, vlm_point=(124, 110), vlm_zero_shot_point=(600, 600), viewport=VIEWPORT)
    assert scored["hits"] == {"tree": False, "vlm_zero_shot": False, "vlm": False, "fallback": False, "snap": True}
    assert scored["tree_matched"] is None  # "gear" names nothing in the tree
    # the vlm missed by 4 units; snapping to the icon's exact box turned it into a hit


def test_report_has_every_arm_and_condition_and_the_hypotheses():
    scored = [
        {"id": f"i{c}{k}", "condition": c, "hits": {a: (a == "vlm") == (c != "C1") for a in compare.ARMS}}
        for c in compare.CONDITIONS for k in range(4)
    ]
    out = compare.report(scored)
    assert set(out["table"]) == set(compare.ARMS)
    assert out["table"]["vlm"]["C1"]["accuracy"] == 0.0 and out["table"]["vlm"]["C2"]["accuracy"] == 1.0
    assert out["hypotheses"]["H4 C1: tree - vlm (>= 0 expected)"]["diff"] == 1.0


# -- the generated pages, rendered for real --------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    pytest.importorskip("playwright")
    from computeruse.sandbox.build import build

    out = tmp_path_factory.mktemp("compare")
    try:
        manifest = build(out, n_pages=3, first_seed=500)
    except Exception as e:
        pytest.skip(f"Chrome not available: {e}")
    return out, manifest


def test_every_page_yields_all_five_conditions_checked_on_the_rendered_tree(built):
    out, manifest = built
    assert manifest["dropped"] == []
    assert manifest["by_condition"] == {c: 3 for c in compare.CONDITIONS}
    items = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()]
    assert all((out / i["image"]).exists() and (out / i["tree"]).exists() for i in items)
    assert all(0 <= v <= 1000 for i in items for v in i["gt_box_norm"])


def test_canvas_targets_are_invisible_to_the_tree_but_drawn_on_screen(built):
    from computeruse.sandbox.build import visible_in_pixels

    out, _ = built
    for item in (json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()):
        if item["condition"] == "C4":
            tree = (out / item["tree"]).read_text(encoding="utf-8")
            assert f'"{item["label"]}"' not in tree
            assert visible_in_pixels((out / item["image"]).read_bytes(), item["gt_box_px"])
