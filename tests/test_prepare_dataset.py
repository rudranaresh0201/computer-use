"""
Tests for training/prepare_dataset.py. The one thing that must never be
wrong here is the real-pixel -> scaled-pixel -> normalized coordinate
conversion (see the module docstring for why) -- everything else is
straightforward partition/IO logic.
"""

import json
from pathlib import Path

from computeruse.training.prepare_dataset import (
    SPLITS,
    load_labels,
    normalized_center,
    prepare,
    to_training_example,
    write_splits,
)


def _sample(sample_id: str, split: str, app: str = "notepad") -> dict:
    return {
        "id": sample_id,
        "screenshot_id": f"{app}_0000",
        "screenshot_path": f"images/{app}/{app}_0000.png",
        "real_size": [1000, 1000],
        "scaled_size": [500, 500],
        "scale_x": 2.0,
        "scale_y": 2.0,
        "element": {
            "name": "Save",
            "control_type": "Button",
            "automation_id": "SaveButton",
            "bbox_real": [400, 400, 600, 600],
        },
        "instruction": "Click the Save button",
        "app": app,
        "app_richness": "rich",
        "split": split,
        "session_id": "2026-07-13T00-00-00",
        "source": "uia_auto_label",
    }


def test_normalized_center_converts_real_to_scaled_then_normalizes():
    # real bbox center (500, 500) -> scaled center (250, 250) (divide by
    # scale 2.0) -> normalized against scaled_size (500, 500) -> (500, 500)
    # on the 0-1000 range. Chosen so real/scaled/normalized don't
    # accidentally coincide and mask a bug.
    norm_x, norm_y = normalized_center(
        bbox_real=[400, 400, 600, 600],
        real_size=(1000, 1000),
        scale_x=2.0,
        scale_y=2.0,
    )
    assert (norm_x, norm_y) == (500, 500)


def test_normalized_center_handles_no_resize():
    # scale 1.0 means real_size == scaled_size (small screenshot, no
    # resize applied in perception/screenshot.py).
    norm_x, norm_y = normalized_center(
        bbox_real=[0, 0, 200, 200],
        real_size=(800, 800),
        scale_x=1.0,
        scale_y=1.0,
    )
    assert (norm_x, norm_y) == (125, 125)


def test_to_training_example_builds_expected_fields():
    example = to_training_example(_sample("notepad_0000_e0", "train"))
    assert example.id == "notepad_0000_e0"
    assert example.image_path == "images/notepad/notepad_0000.png"
    assert example.prompt == "Click the Save button"
    assert example.target == "(500,500)"
    assert example.app == "notepad"
    assert example.split == "train"


def test_prepare_partitions_by_existing_split_field(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    samples = [
        _sample("a", "train"),
        _sample("b", "train"),
        _sample("c", "dev"),
        _sample("d", "test_same_app"),
        _sample("e", "test_held_out_app"),
    ]
    with labels_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    by_split = prepare(tmp_path)

    assert {e.id for e in by_split["train"]} == {"a", "b"}
    assert {e.id for e in by_split["dev"]} == {"c"}
    assert {e.id for e in by_split["test_same_app"]} == {"d"}
    assert {e.id for e in by_split["test_held_out_app"]} == {"e"}


def test_prepare_never_invents_a_split_not_in_the_data(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(json.dumps(_sample("a", "train")) + "\n", encoding="utf-8")

    by_split = prepare(tmp_path)

    assert set(by_split.keys()) == set(SPLITS)


def test_load_labels_skips_blank_lines(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(_sample("a", "train")) + "\n\n" + json.dumps(_sample("b", "dev")) + "\n",
        encoding="utf-8",
    )
    samples = load_labels(labels_path)
    assert len(samples) == 2


def test_write_splits_writes_one_file_per_split(tmp_path):
    by_split = prepare_from_samples = {
        "train": [to_training_example(_sample("a", "train"))],
        "dev": [],
        "test_same_app": [],
        "test_held_out_app": [],
    }
    out_dir = tmp_path / "splits"
    write_splits(by_split, out_dir)

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "dev.jsonl").exists()

    lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["id"] == "a"
    assert written["target"] == "(500,500)"

    assert (out_dir / "dev.jsonl").read_text(encoding="utf-8") == ""
