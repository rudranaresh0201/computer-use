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


def _disable_peft_torchao_dispatch() -> None:
    """Kaggle's base image ships an old torchao (0.10.0 as of 2026-07-16)
    that peft's LoRA dispatcher rejects outright -- `is_torchao_available()`
    raises ImportError for a too-old version instead of just returning
    False, even though this project never uses torchao-quantized models.
    Uninstalling torchao from the shell (`pip uninstall torchao`) turned
    out not to reliably fix this on Kaggle (a second site-packages copy
    survives the uninstall), so this patches the check at the source
    instead: force peft's torchao dispatcher to report "unavailable" so it
    falls through to the plain LoRA dispatch path we actually want. Safe to
    no-op if peft's internals change shape (older/newer peft without this
    module) -- we only need this on the exact peft version Kaggle installs.
    """
    try:
        from peft.tuners.lora import torchao as _peft_torchao

        _peft_torchao.is_torchao_available = lambda: False
    except (ImportError, AttributeError):
        pass


_disable_peft_torchao_dispatch()


def cast_trainable_params_to_fp32(model):
    """Keep the frozen base model in fp16 (it never receives an optimizer
    update, and fp16 is what makes a 2B model fit a 15GB T4), but hold the
    *trainable* LoRA params in fp32.

    Why this is not optional: Adam's update magnitudes here are ~1e-4
    relative to weights of ~1e-2, and fp16's smallest normal value is
    ~6e-5. Storing the adapter in fp16 rounds a large share of those
    updates to zero, so the model appears to train (loss looks sane --
    the loss is dominated by the frozen base) while the adapter barely
    moves. Which updates survive depends on batch order, which is what
    made 2026-07-19's rerun look like random lr instability. This plus
    fp16=True in TrainingArguments (autocast + GradScaler, see below) is
    the standard PEFT mixed-precision recipe; we had neither.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    return model


def build_training_args(output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        # Real mixed precision, not "the weights happen to be fp16".
        # fp16=True is what makes Trainer wrap the step in torch.autocast
        # AND attach a GradScaler; without it, the backward pass runs
        # unscaled in fp16 and small gradients underflow to zero with no
        # warning. T4 has no usable bf16 (no bf16 tensor cores), so fp16 +
        # GradScaler is the correct choice on this hardware, not bf16.
        fp16=True,
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
        # NOTE (2026-07-19, second pass): the run-to-run instability this
        # comment blames on lr was at least partly the fp16 underflow bug
        # fixed above -- no GradScaler was attached, so an unpredictable
        # subset of adapter updates was being dropped every run. 1e-4 is
        # kept as a reasonable value on its own merits, but if the fixed
        # run now underfits, raising lr back toward 3e-4 is the first
        # thing to try, since the original reason for lowering it is gone.
        #
        # 5e-4 over a single epoch (63 steps on the 542-row v2 dataset) is
        # too little gradient signal for a stable result -- the 2026-07-19
        # rerun landed noticeably worse than the run before it purely from
        # random init/batch-order variance, confirmed by re-verifying the
        # dataset was identical both times. Checked against the literature
        # before picking a new value: SeeClick (arxiv 2401.10935), the
        # closest real published GUI-grounding LoRA fine-tune, uses 3e-5 --
        # roughly 15x lower than our original 5e-4. Landed on 1e-4 as a
        # middle point: meaningfully lower than 5e-4 (which was likely
        # too aggressive and part of why results were noisy run-to-run),
        # but higher than SeeClick's 3e-5 since our dataset is far smaller
        # (542 rows vs. their ~1M-scale corpus) and needs more signal per
        # step to move at all. More epochs gives the model repeated,
        # gentler passes over the same data instead of a few large, noisy
        # ones. seed pins the run so a repeat is reproducible, not another
        # random draw.
        learning_rate=1e-4,
        num_train_epochs=3,
        # A LoRA adapter starts at exactly zero (B is zero-initialized), so
        # the first few steps take the full lr against a model that hasn't
        # moved at all yet. A short warmup costs ~2 steps and removes that
        # initial shock, which matters more here than usual because the
        # whole run is only ~189 steps.
        warmup_ratio=0.03,
        seed=42,
        data_seed=42,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Without this, a 3-epoch run keeps epoch 3 even if epoch 2 had the
        # better dev loss -- i.e. it silently reports the overfit checkpoint
        # on a 252-example dataset, where overfitting by epoch 3 is likely.
        # Selecting on dev (never on either test split) is exactly what the
        # hypothesis doc permits.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
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
    # Must happen after get_peft_model (the adapters don't exist before it)
    # and before the Trainer is built, so the optimizer is constructed over
    # fp32 params -- see cast_trainable_params_to_fp32's docstring.
    cast_trainable_params_to_fp32(model)
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
