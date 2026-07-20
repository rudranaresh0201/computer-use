"""
Tests for training/train_lora.py. build_training_args is pure config, no
model/network needed. build_trainer/main download the real ~4GB base
model and need a GPU to run in reasonable time -- not exercised here,
consistent with this being a Kaggle/Colab entrypoint, not something this
machine runs.
"""

from pathlib import Path

import torch

from computeruse.training.train_lora import build_training_args, cast_trainable_params_to_fp32


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


def test_training_args_evaluates_and_saves_every_epoch():
    args = build_training_args(Path("runs/test"))
    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"


def test_training_args_enable_real_mixed_precision():
    # The bug this guards (found 2026-07-19): the model was loaded with
    # torch_dtype=float16 but fp16 was never set here, so Trainer attached
    # no GradScaler and the backward pass ran unscaled in fp16. Small
    # gradients underflowed to zero silently -- training "worked" (loss
    # looked sane, dominated by the frozen base) while the adapter barely
    # moved, and which updates survived depended on batch order. Do not
    # remove fp16=True without also changing how the model is loaded.
    args = build_training_args(Path("runs/test"))
    assert args.fp16 is True


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
