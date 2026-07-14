"""
LoRA fine-tuning entrypoint for Qwen2-VL-2B-Instruct on our GUI-grounding
dataset. Intended to run on Kaggle/Colab (ADR-0003's $0-budget
commitment) -- this machine has no GPU, so `main()` is never exercised by
the test suite; only the config/wiring functions below it are.

Hyperparameters start from Mehul's Re:Fine Phase 1 (CodeT5 + LoRA) config
as a real reference point (see training/lora_config.py's docstring),
adjusted per the hypothesis doc's explicit rule: LoRA config and stopping
criterion get fixed *before* looking at held-out numbers, and the dev
split (not held-out) absorbs any tuning -- so these defaults are a
starting point to tune against dev loss, not a final answer.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

from .dataset import GroundingDataset, collate_fn, resolve_path
from .lora_config import build_lora_config

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def build_training_args(output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # PyTorch's default non-reentrant checkpoint mode strictly requires
        # the backward-pass recomputation to save the exact same number of
        # intermediate tensors as the original forward pass. Qwen2-VL's
        # architecture doesn't guarantee that under this model's forward
        # path (observed: 151 tensors saved on the real forward vs. 28 on
        # recomputation -- a torch/transformers compatibility gap, not a
        # bug in this training loop). The older reentrant mode doesn't do
        # this strict check and is the documented workaround.
        gradient_checkpointing_kwargs={"use_reentrant": True},
        learning_rate=5e-4,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
    )


def build_trainer(dataset_root: Path, output_dir: Path) -> Trainer:
    """Wires the real model + LoRA adapter + our GroundingDataset/collate_fn
    into a HF Trainer. Not called by tests -- downloads the ~4GB base
    model and needs a GPU to run in reasonable time; this function exists
    so the Kaggle notebook is `from computeruse.training.train_lora import
    build_trainer; build_trainer(...).train()`, not a copy-pasted script."""
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()
    # gradient checkpointing needs at least one input tensor with
    # requires_grad=True to build a backward graph -- with the base model
    # frozen (only the tiny LoRA adapters are trainable), nothing upstream
    # of the input embeddings satisfies that by default.
    model.enable_input_require_grads()

    train_dataset = GroundingDataset(
        resolve_path(dataset_root, "splits/train.jsonl"), dataset_root, processor, processor.tokenizer
    )
    eval_dataset = GroundingDataset(
        resolve_path(dataset_root, "splits/dev.jsonl"), dataset_root, processor, processor.tokenizer
    )

    return Trainer(
        model=model,
        args=build_training_args(output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
    )


def main(
    dataset_root: Path = Path("data/gui_grounding"),
    output_dir: Path = Path("runs/lora_grounder"),
) -> None:
    trainer = build_trainer(dataset_root, output_dir)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    main()
