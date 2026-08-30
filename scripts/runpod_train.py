"""
Phase 3 training entrypoint for a rented GPU pod (RunPod), as opposed to
notebooks/phase3_lora_training.ipynb which targets Kaggle.

Why a script and not the notebook: every run lost so far died with its
*session*, not with training -- four Kaggle runs and one Colab run, all
with healthy loss curves and no weights at the end (see docs/journal.md,
2026-07-23 and 2026-08-10). A notebook cell is bound to a browser
connection; this is meant to be launched under nohup/tmux so closing the
laptop cannot kill it, writing checkpoints to a persistent volume that
outlives the pod itself. That combination -- not any change to the
training loop -- is what makes a run survivable.

Usage on the pod (see docs/runpod-setup.md for the full runbook):

    nohup python scripts/runpod_train.py \
        --dataset-root /workspace/data/gui_grounding \
        --output-dir /workspace/runs/lora_grounder \
        > /workspace/runs/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from computeruse.training.train_lora import (
    build_trainer,
    find_last_checkpoint,
    resolve_precision,
)


def resolve_dataset_root(root: Path) -> Path:
    """Find the directory that actually holds labels.jsonl, and insist
    there is exactly one.

    Same guard as the Kaggle notebook's cell 9, for the same reason: with
    two copies of the dataset mounted (a stale upload alongside the
    current one), training silently runs against whichever is found first
    and the logs look identical either way. That is a whole wasted run
    with nothing to indicate it happened.
    """
    if (root / "labels.jsonl").exists():
        return root
    candidates = sorted(root.rglob("labels.jsonl"))
    if not candidates:
        raise SystemExit(
            f"no labels.jsonl found under {root} -- did the dataset unzip there? "
            f"expected {root}/labels.jsonl plus {root}/splits/ and {root}/images/"
        )
    if len(candidates) > 1:
        raise SystemExit(
            f"found {len(candidates)} labels.jsonl under {root}: {candidates}. "
            "Delete the stale copy before training -- running against the wrong "
            "one costs a full GPU run and looks identical in the logs."
        )
    return candidates[0].parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace/data/gui_grounding"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/runs/lora_grounder"))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="hard cap on steps. Omit for the full 3-epoch schedule -- on a "
        "persistent volume there is no session deadline to budget against.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="per-device batch. 2 fits a 24GB card comfortably; drop to 1 "
        "(and raise --grad-accum to 4) if you see CUDA out of memory.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="gradient accumulation steps. batch-size * grad-accum must stay "
        "at 4 -- that is the effective batch every hyperparameter was tuned at.",
    )
    args = parser.parse_args()

    effective = args.batch_size * args.grad_accum
    if effective != 4:
        raise SystemExit(
            f"batch-size * grad-accum = {effective}, expected 4. Changing the "
            "effective batch size changes the experiment (lr=1e-4 was chosen "
            "against a batch of 4); trading one factor for the other is free."
        )

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible -- check the pod actually has a GPU attached")

    dataset_root = resolve_dataset_root(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset:   {dataset_root}")
    print(f"output:    {args.output_dir}")
    print(f"gpu:       {torch.cuda.get_device_name(0)}")
    print(f"precision: {resolve_precision()}")
    print(f"batch:     {args.batch_size} x {args.grad_accum} accum = {effective} effective")

    trainer = build_trainer(
        dataset_root,
        args.output_dir,
        max_steps=args.max_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )

    resume_from = find_last_checkpoint(args.output_dir)
    if resume_from:
        print(f"resuming from {resume_from}")
    else:
        print("no checkpoint found -- starting fresh from step 0")

    started = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - started

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"\ntrained in {elapsed / 3600:.2f}h")
    print(f"adapter saved to {final_dir}")
    print("Next: python scripts/runpod_eval.py --adapter", final_dir)


if __name__ == "__main__":
    main()
