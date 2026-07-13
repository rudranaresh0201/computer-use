"""
Tests for training/dataset.py. mask_prompt_tokens and collate_fn are pure
tensor/list logic -- tested directly, no network or real model needed.
GroundingDataset itself needs the real Qwen2-VL processor (network
download) to produce real pixel_values, so it's verified separately as a
one-off check against a real collected image, not in this suite.
"""

import torch

from computeruse.training.dataset import IGNORE_INDEX, collate_fn, mask_prompt_tokens, resolve_path


def test_resolve_path_prefers_the_real_nested_path(tmp_path):
    nested = tmp_path / "images" / "calculator" / "calculator_0000.png"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"fake png")

    assert resolve_path(tmp_path, "images/calculator/calculator_0000.png") == nested


def test_resolve_path_falls_back_to_flat_basename_lookup(tmp_path):
    # simulates Kaggle's uploader flattening images/calculator/foo.png -> foo.png
    flat = tmp_path / "calculator_0000.png"
    flat.write_bytes(b"fake png")

    found = resolve_path(tmp_path, "images/calculator/calculator_0000.png")
    assert found == flat


def test_resolve_path_raises_clearly_when_neither_exists(tmp_path):
    try:
        resolve_path(tmp_path, "images/calculator/calculator_0000.png")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "calculator_0000.png" in str(exc)


def test_mask_prompt_tokens_masks_everything_before_prompt_len():
    input_ids = [1, 2, 3, 4, 5]
    labels = mask_prompt_tokens(input_ids, prompt_len=3)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]


def test_mask_prompt_tokens_zero_prompt_len_masks_nothing():
    input_ids = [1, 2, 3]
    labels = mask_prompt_tokens(input_ids, prompt_len=0)
    assert labels == [1, 2, 3]


def test_mask_prompt_tokens_full_length_masks_everything():
    input_ids = [1, 2, 3]
    labels = mask_prompt_tokens(input_ids, prompt_len=3)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_mask_prompt_tokens_rejects_prompt_len_longer_than_sequence():
    try:
        mask_prompt_tokens([1, 2, 3], prompt_len=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def _fake_item(seq_len: int, pixel_rows: int = 4) -> dict:
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.tensor([IGNORE_INDEX] * (seq_len - 1) + [99], dtype=torch.long),
        "pixel_values": torch.zeros((pixel_rows, 2), dtype=torch.float),
        "image_grid_thw": torch.tensor([1, 2, pixel_rows // 2]),
    }


def test_collate_fn_pads_to_batch_max_length():
    batch = [_fake_item(3), _fake_item(5)]
    out = collate_fn(batch, pad_token_id=0)

    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    assert out["labels"].shape == (2, 5)
    # shorter sequence padded on the right with pad_token_id / 0 / IGNORE_INDEX
    assert out["input_ids"][0].tolist() == [0, 1, 2, 0, 0]
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert out["labels"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 99, IGNORE_INDEX, IGNORE_INDEX]


def test_collate_fn_concatenates_pixel_values_not_pads():
    batch = [_fake_item(3, pixel_rows=4), _fake_item(3, pixel_rows=6)]
    out = collate_fn(batch, pad_token_id=0)

    assert out["pixel_values"].shape == (10, 2)  # 4 + 6 rows concatenated
    assert out["image_grid_thw"].shape == (2, 3)  # stacked, one row per image
