"""
Tests for training/train_lora.py. build_training_args is pure config, no
model/network needed. build_trainer/main download the real ~4GB base
model and need a GPU to run in reasonable time -- not exercised here,
consistent with this being a Kaggle/Colab entrypoint, not something this
machine runs.
"""

from pathlib import Path

import pytest
import torch

from computeruse.training.train_lora import (
    build_training_args,
    cast_trainable_params_to_fp32,
    find_last_checkpoint,
    resolve_precision,
    torch_dtype_for,
)


def test_training_args_match_documented_starting_hyperparameters():
    # revised 2026-07-19 after the v2 rerun showed real run-to-run
    # instability at lr=5e-4/1 epoch -- see build_training_args' own
    # comment for the SeeClick-literature-informed reasoning behind these
    # specific values.
    args = build_training_args(Path("runs/test"))
    assert args.learning_rate == 1e-4
    assert args.num_train_epochs == 3
    assert args.seed == 42
    assert args.data_seed == 42


def test_training_args_use_gradient_accumulation_to_fit_a_free_tier_gpu():
    # per_device_train_batch_size=4 OOM'd on a 15GB Kaggle T4 (fp16 weights
    # + activations at batch 4 didn't fit). batch_size=1 with
    # gradient_accumulation_steps=4 keeps the same effective batch size
    # with a much smaller peak memory footprint; gradient_checkpointing
    # trades compute for memory on top of that.
    args = build_training_args(Path("runs/test"))
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.gradient_checkpointing is True


def test_training_args_use_reentrant_checkpointing_for_qwen2vl_compatibility():
    # non-reentrant (torch's default) checkpointing raised CheckpointError
    # on Qwen2-VL: forward saved 151 tensors, recomputation only 28. The
    # reentrant mode doesn't do that strict recomputation check.
    args = build_training_args(Path("runs/test"))
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_training_args_evaluates_and_saves_every_100_steps():
    # Changed from "epoch" (~393 steps here) to steps=100 after losing two
    # runs the same day (2026-07-23) to Kaggle session death before the
    # first epoch boundary -- see build_training_args' save_steps comment.
    # load_best_model_at_end requires save points to be a subset of eval
    # points, so both moved together.
    args = build_training_args(Path("runs/test"))
    assert args.eval_strategy == "steps"
    assert args.eval_steps == 100
    assert args.save_strategy == "steps"
    assert args.save_steps == 100


def test_find_last_checkpoint_returns_none_when_output_dir_missing(tmp_path):
    assert find_last_checkpoint(tmp_path / "does_not_exist") is None


def test_find_last_checkpoint_returns_none_when_no_checkpoints(tmp_path):
    (tmp_path / "some_other_file.txt").write_text("x")
    assert find_last_checkpoint(tmp_path) is None


def test_find_last_checkpoint_picks_highest_step_number(tmp_path):
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-300").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    assert find_last_checkpoint(tmp_path) == str(tmp_path / "checkpoint-300")


def test_training_args_enable_real_mixed_precision():
    # The bug this guards (found 2026-07-19): the model was loaded with
    # torch_dtype=float16 but fp16 was never set here, so Trainer attached
    # no GradScaler and the backward pass ran unscaled in fp16. Small
    # gradients underflowed to zero silently -- training "worked" (loss
    # looked sane, dominated by the frozen base) while the adapter barely
    # moved, and which updates survived depended on batch order. Exactly
    # one of fp16/bf16 must be on, and it must match how the model was
    # loaded (torch_dtype_for).
    args = build_training_args(Path("runs/test"), precision="fp16")
    assert args.fp16 is True
    assert args.bf16 is False


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="TrainingArguments(bf16=True) refuses to construct without an Ampere+ GPU "
    "-- which is itself the guard that a non-bf16 pod fails loudly instead of silently",
)
def test_training_args_bf16_mode_turns_fp16_off():
    # On Ampere/Ada (a rented pod, not Kaggle's Turing T4) bf16 is the
    # right mode: fp32's exponent range means gradients can't underflow,
    # so no GradScaler is involved at all. Setting both would be
    # incoherent -- autocast has one compute dtype.
    args = build_training_args(Path("runs/test"), precision="bf16")
    assert args.bf16 is True
    assert args.fp16 is False


def test_resolve_precision_falls_back_to_fp16_without_a_bf16_gpu():
    # Turing/CPU has no bf16 tensor cores; claiming bf16 there would run
    # emulated and slower, not faster.
    assert resolve_precision() in {"fp16", "bf16"}


def test_torch_dtype_matches_the_precision_mode():
    assert torch_dtype_for("fp16") is torch.float16
    assert torch_dtype_for("bf16") is torch.bfloat16


def test_torch_dtype_rejects_an_unknown_precision():
    # a typo here would otherwise silently load the base model in fp32 and
    # OOM, or mismatch autocast's compute dtype.
    try:
        torch_dtype_for("fp8")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown precision")


def test_effective_batch_size_is_preserved_when_trading_batch_for_accumulation():
    # batch=2/accum=2 on a 24GB card is the same experiment as
    # batch=1/accum=4 on a 15GB T4 -- lr was reasoned about at an
    # effective batch of 4, and only the product must stay fixed.
    t4 = build_training_args(Path("runs/test"), per_device_batch_size=1, gradient_accumulation_steps=4)
    pod = build_training_args(Path("runs/test"), per_device_batch_size=2, gradient_accumulation_steps=2)
    assert t4.per_device_train_batch_size * t4.gradient_accumulation_steps == 4
    assert pod.per_device_train_batch_size * pod.gradient_accumulation_steps == 4


def test_training_args_select_the_best_dev_checkpoint_not_the_last():
    # 3 epochs over 252 examples can overfit by the last epoch; without
    # this, that overfit checkpoint is what gets reported. Selection is on
    # dev only, never on either test split.
    args = build_training_args(Path("runs/test"))
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False


def test_cast_trainable_params_promotes_only_the_trainable_ones():
    # mirrors the real shape: a frozen fp16 "base" plus a trainable fp16
    # "adapter". Only the adapter should come back as fp32 -- promoting the
    # frozen base too would blow up memory on a 15GB T4 for no benefit.
    frozen = torch.nn.Linear(4, 4).half()
    frozen.requires_grad_(False)
    adapter = torch.nn.Linear(4, 4).half()
    model = torch.nn.Sequential(frozen, adapter)

    cast_trainable_params_to_fp32(model)

    assert all(p.dtype == torch.float16 for p in frozen.parameters())
    assert all(p.dtype == torch.float32 for p in adapter.parameters())


def test_cast_trainable_params_keeps_them_trainable():
    # a careless implementation that rebuilds the tensor instead of
    # assigning .data can drop requires_grad, which would silently freeze
    # the entire adapter -- a worse version of the bug being fixed.
    layer = torch.nn.Linear(4, 4).half()
    cast_trainable_params_to_fp32(layer)
    assert all(p.requires_grad for p in layer.parameters())


def test_training_args_keeps_unused_columns_for_custom_collator():
    # our collate_fn expects pixel_values/image_grid_thw, which the
    # default HF column-pruning would otherwise silently drop.
    args = build_training_args(Path("runs/test"))
    assert args.remove_unused_columns is False


def test_training_args_max_steps_defaults_to_full_three_epoch_schedule():
    # -1 is HF's sentinel for "ignore max_steps, use num_train_epochs" --
    # local/unconstrained runs must be unaffected by the Kaggle-budget
    # override added 2026-08-10.
    args = build_training_args(Path("runs/test"))
    assert args.max_steps == -1
    assert args.num_train_epochs == 3


def test_training_args_max_steps_override_hard_caps_the_run():
    # added 2026-08-10: a fixed-hours Kaggle session needs the run to stop
    # (and save_model) well inside its wall-clock budget, without
    # shrinking the dataset itself. HF's Trainer treats max_steps > 0 as
    # taking precedence over num_train_epochs.
    args = build_training_args(Path("runs/test"), max_steps=700)
    assert args.max_steps == 700
