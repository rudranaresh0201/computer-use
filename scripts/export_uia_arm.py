"""
Run the UIA-only ablation arm on Windows and save its per-example results
to JSON, so the GPU pod can join it against the model arms without needing
pywinauto.

Why this is a separate step rather than part of runpod_eval.py: the arm
itself is CPU/text-only and takes seconds, but it lives in
`eval/uia_only.py`, which imports `dataset.collector` ->
`perception.uia` -> `pywinauto`. That chain is Windows-only and will not
import on a Linux pod at all. Splitting it out keeps the arm running where
it can run, and turns its result into a versioned artifact rather than
something silently recomputed differently on two machines.

Run this locally (Windows), before or during the pod session:

    python scripts/export_uia_arm.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from computeruse.eval import uia_only
from computeruse.training.prepare_dataset import SPLITS

EVAL_SPLITS = tuple(s for s in SPLITS if s != "train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gui_grounding"))
    parser.add_argument("--out", type=Path, default=Path("runs/uia_arm_results.json"))
    args = parser.parse_args()

    results = uia_only.run_arm(args.dataset_root, splits=EVAL_SPLITS)
    resolved = sum(r.available for r in results)
    hits = sum(r.hit for r in results)

    payload = {
        "splits": list(EVAL_SPLITS),
        "results": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"examples scored: {len(results)}")
    print(f"UIA resolved a candidate on: {resolved} ({resolved / len(results):.1%})")
    print(f"click accuracy: {hits / len(results):.1%}")
    print(f"written to {args.out}")
    print(
        "\nNote (see eval/uia_only.py's docstring): this arm cannot score low by "
        "construction -- the dataset only contains elements UIA already saw, so "
        "the correct element is always in the candidate pool. Treat it as a "
        "ceiling, not as evidence that UIA solves grounding in the wild."
    )


if __name__ == "__main__":
    main()
