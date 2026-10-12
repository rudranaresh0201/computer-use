"""Test B (eval/real_tree.py): damaging a real screen's UIA tree on purpose."""

from computeruse.eval import real_tree


def row(id_, name, bbox, screenshot="s1"):
    return {"id": id_, "screenshot_id": screenshot, "instruction": f"Select {name}", "app": "calc",
            "element": {"name": name, "bbox_real": bbox}, "real_size": [1000, 1000], "scale_x": 1.0, "scale_y": 1.0}


def test_damage_blanks_or_removes_only_the_target():
    elements = [real_tree.element_of(row("a", "Nine", [0, 0, 100, 100])), real_tree.element_of(row("b", "Eight", [200, 0, 300, 100]))]
    assert real_tree.damage(elements, "a", "intact") == elements
    assert [e.name for e in real_tree.damage(elements, "a", "nameless")] == ["", "Eight"]
    assert [e.ref for e in real_tree.damage(elements, "a", "absent")] == ["b"]


def test_each_condition_is_scored_with_the_same_vlm_answer():
    labels = [row("a", "Nine", [0, 0, 100, 100]), row("b", "Eight", [200, 0, 300, 100])]
    predictions = [{"example_id": "a", "split": "test_same_app", "predicted_point": [104, 50],
                    "ground_truth_bbox_norm": [0, 0, 100, 100]}]
    scored = {s["condition"]: s["hits"] for s in real_tree.run(labels, predictions)}
    assert scored["intact"]["tree"] and not scored["nameless"]["tree"] and not scored["absent"]["tree"]
    assert not scored["intact"]["vlm"]  # 4 units outside the box...
    assert scored["intact"]["snap"] and scored["nameless"]["snap"]  # ...snapped into it while the box is in the tree
    assert not scored["absent"]["snap"]  # nothing of the target left to snap to
