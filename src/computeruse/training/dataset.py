"""
Turns training/prepare_dataset.py's split files into real model-ready
tensors: image pixels, token ids, and *masked* labels for causal-LM SFT.

The one genuinely easy-to-get-wrong detail here (same spirit as Day 0's
screenshot coordinate-scaling note): loss must only be computed on the
assistant's target-coordinate tokens, not on the system prompt, the
instruction, or the image tokens -- those are always given as context,
never something the model needs to learn to reproduce. Masking them with
-100 (torch's CrossEntropyLoss ignore_index) is what makes this
supervised fine-tuning instead of "teach the model to also memorize its
own prompt."

A second, less obvious pitfall this module avoids: computing how many
tokens the *prompt* occupies by tokenizing the prompt text with the plain
tokenizer would undercount it. Qwen2-VL's real processor expands the
single literal "<|image_pad|>" placeholder in the chat-template text into
many image tokens (one per visual patch, depending on image resolution --
see the Naive Dynamic Resolution mechanism in the Qwen2-VL paper). So the
prompt length must be computed by running the *same processor* over the
*same image* on the prompt-only text, not by tokenizing text alone --
otherwise the mask boundary lands in the wrong place relative to the
actual expanded token sequence.

A third, environment-specific pitfall: Kaggle's web dataset uploader was
observed (2026-07-13, twice, with a correctly-built zip using forward
slashes) to flatten nested folders -- `images/calculator/foo.png` lands
as just `foo.png` at the dataset root, same for `splits/*.jsonl`. Rather
than depend on an upload UI we don't control behaving, `resolve_path`
below tries the real nested relative path first and falls back to a
basename-only lookup under `dataset_root` if that path doesn't exist --
safe because every screenshot filename already embeds its app name
(`calculator_0000.png`), so basenames are unique across the whole
dataset even without the folder structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .chat_format import to_conversation
from .prepare_dataset import TrainingExample

IGNORE_INDEX = -100


def mask_prompt_tokens(input_ids: list[int], prompt_len: int) -> list[int]:
    """labels = input_ids with everything before `prompt_len` replaced by
    IGNORE_INDEX. `prompt_len` must already account for the real,
    processor-expanded token count (see module docstring), not a plain
    tokenizer count."""
    if prompt_len > len(input_ids):
        raise ValueError(
            f"prompt_len ({prompt_len}) exceeds the full sequence length "
            f"({len(input_ids)}) -- the prompt and full text were not "
            "processed consistently."
        )
    return [IGNORE_INDEX] * prompt_len + list(input_ids[prompt_len:])


def resolve_path(dataset_root: Path, relative_path: str) -> Path:
    """Prefer the real nested path; fall back to a basename-only lookup
    under dataset_root if the platform flattened the folder structure
    (see module docstring). Raises with both attempted paths if neither
    exists, so a genuinely missing file fails loudly, not silently."""
    nested = dataset_root / relative_path
    if nested.exists():
        return nested

    flat = dataset_root / Path(relative_path).name
    if flat.exists():
        return flat

    raise FileNotFoundError(
        f"could not find {relative_path!r} under {dataset_root} -- tried "
        f"{nested} (nested) and {flat} (flat fallback)"
    )


def _load_examples(jsonl_path: Path) -> list[TrainingExample]:
    examples = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(TrainingExample(**json.loads(line)))
    return examples


class GroundingDataset(Dataset):
    """One split file (train.jsonl, dev.jsonl, ...) as a torch Dataset.
    `processor` is Qwen2-VL's real AutoProcessor (image + text -> model
    inputs); `tokenizer` is `processor.tokenizer`, kept separate only
    because building the prompt-only text needs `apply_chat_template`
    directly."""

    def __init__(self, jsonl_path: Path, dataset_root: Path, processor, tokenizer):
        self.examples = _load_examples(jsonl_path)
        self.dataset_root = dataset_root
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        image_path = resolve_path(self.dataset_root, example.image_path)
        image = Image.open(image_path).convert("RGB")
        conversation = to_conversation(example)

        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        prompt_text = self.tokenizer.apply_chat_template(
            conversation[:-1], tokenize=False, add_generation_prompt=True
        )

        full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        # same image through the same processor, so the image-token
        # expansion matches exactly -- see module docstring.
        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        input_ids = full_inputs["input_ids"][0]
        labels = mask_prompt_tokens(input_ids.tolist(), prompt_len)

        return {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"][0],
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"][0],
            "labels": torch.tensor(labels),
        }


def collate_fn(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Pads input_ids/attention_mask/labels to the batch's max length.
    pixel_values/image_grid_thw are concatenated, not padded -- Qwen2-VL
    represents images as a flat sequence of visual patches with
    image_grid_thw recording each image's own (t, h, w) grid, which is
    exactly what lets the model tell the images apart after
    concatenation, so no padding is needed on that side."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.stack([item["image_grid_thw"] for item in batch], dim=0),
    }
