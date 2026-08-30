"""
The Phase 3 ablation, run end to end on a GPU pod: all four arms from
docs/research/gui-grounding-hypothesis.md, scored on the same example set,
joined into one report.

    uia_only    -- text matching against the captured UIA tree (CPU only)
    zero_shot   -- base Qwen2-VL-2B, no fine-tuning (the H1 control)
    fine_tuned  -- base + our LoRA adapter (the thing being tested)
    hybrid      -- UIA when it resolves, fine_tuned when it doesn't (H2)

All four run over the same three splits (dev, test_same_app,
test_held_out_app) so hybrid.combine can join them per example -- it
raises KeyError rather than silently defaulting if the two sides ever
disagree about which examples they scored.

Usage on the pod:

    python scripts/runpod_eval.py \
        --dataset-root /workspace/data/gui_grounding \
        --adapter /workspace/runs/lora_grounder/final \
        --out /workspace/runs/ablation_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from computeruse.dataset.registry import load_registry
from computeruse.eval import hybrid
from computeruse.eval.report import EvalRecord, build_report
from computeruse.eval.vlm_grounder import (
    MODEL_ID,
    evaluate_arm,
    summarize_diagnostics,
    to_eval_records,
)
from computeruse.training.dataset import resolve_path
from computeruse.training.prepare_dataset import SPLITS, TrainingExample
from computeruse.training.train_lora import resolve_precision, torch_dtype_for

EVAL_SPLITS = tuple(s for s in SPLITS if s != "train")

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class UiaRow:
    """Local mirror of eval.uia_only.UiaArmResult, rebuilt from the JSON
    scripts/export_uia_arm.py writes on Windows.

    Not imported from uia_only on purpose: that module reaches
    dataset.collector -> perception.uia -> pywinauto, which cannot import
    on Linux. hybrid.combine only reads attributes, so a structurally
    identical object is all it needs.
    """

    example_id: str
    app: str
    app_richness: str
    split: str
    available: bool
    hit: bool


def load_uia_results(path: Path) -> list[UiaRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["splits"]) != EVAL_SPLITS:
        raise SystemExit(
            f"{path} was exported over splits {payload['splits']}, but this run "
            f"scores {list(EVAL_SPLITS)}. Re-export before joining -- hybrid "
            "requires both sides to cover the same examples."
        )
    return [UiaRow(**row) for row in payload["results"]]


def resolve_apps_yaml(dataset_root: Path) -> Path:
    """apps.yaml is the *registry* (tracked source), not generated data --
    .gitignore keeps labels/images out of git but keeps this in, and the
    dataset upload zip is built strictly from labels.jsonl's referenced
    paths, so it contains no apps.yaml at all. Look in the repo first and
    fall back to the dataset root, rather than assuming
    `dataset_root / "apps.yaml"` (which the Kaggle notebook's zero-shot
    cell does -- it would have raised only after the model download).
    """
    candidates = (
        REPO_ROOT / "data" / "gui_grounding" / "apps.yaml",
        dataset_root / "apps.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "apps.yaml not found in the repo or the dataset root -- it carries the "
        "per-app richness labels that H2's rich-tree/weak-tree slice needs. "
        f"Looked in: {[str(c) for c in candidates]}"
    )


def load_examples(dataset_root: Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for split in EVAL_SPLITS:
        path = resolve_path(dataset_root, f"splits/{split}.jsonl")
        examples += [
            TrainingExample(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return examples


def load_labels_by_id(dataset_root: Path) -> dict[str, dict]:
    path = resolve_path(dataset_root, "labels.jsonl")
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def run_vlm_arm(model, processor, examples, labels_by_id, dataset_root, name):
    print(f"\n=== {name}: {len(examples)} examples ===")
    results = evaluate_arm(
        model, processor, processor.tokenizer, examples, labels_by_id, dataset_root
    )
    diagnostics = summarize_diagnostics(results)
    print(f"  parse rate:      {diagnostics['parse_rate']:.1%}")
    print(f"  median distance: {diagnostics['median_center_distance']:.0f} / 1000 units")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace/data/gui_grounding"))
    parser.add_argument("--adapter", type=Path, default=Path("/workspace/runs/lora_grounder/final"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/runs/ablation_report.json"))
    parser.add_argument(
        "--uia-results",
        type=Path,
        default=REPO_ROOT / "runs" / "uia_arm_results.json",
        help="JSON from scripts/export_uia_arm.py, run on Windows. Without it "
        "the uia_only and hybrid arms are skipped (the two model arms still run).",
    )
    parser.add_argument(
        "--skip-fine-tuned",
        action="store_true",
        help="run only the arms that need no adapter (uia_only, zero_shot) -- "
        "useful to bank the baselines while training is still running.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not (dataset_root / "labels.jsonl").exists():
        candidates = sorted(dataset_root.rglob("labels.jsonl"))
        if len(candidates) != 1:
            raise SystemExit(
                f"expected exactly one labels.jsonl under {dataset_root}, found {candidates}"
            )
        dataset_root = candidates[0].parent

    richness_by_app = {c.name: c.richness for c in load_registry(resolve_apps_yaml(dataset_root))}
    examples = load_examples(dataset_root)
    labels_by_id = load_labels_by_id(dataset_root)
    records = []

    # --- arm 1: UIA-only. Computed on Windows (it needs pywinauto's
    # element model) and carried here as JSON -- see export_uia_arm.py. ---
    uia_results: list[UiaRow] = []
    if args.uia_results.exists():
        uia_results = load_uia_results(args.uia_results)
        records += [
            EvalRecord(
                example_id=r.example_id,
                arm="uia_only",
                app=r.app,
                app_richness=r.app_richness,
                split=r.split,
                hit=r.hit,
            )
            for r in uia_results
        ]
        resolved = sum(r.available for r in uia_results)
        print(f"=== uia_only: {len(uia_results)} examples from {args.uia_results} ===")
        print(f"  UIA resolved a candidate on {resolved} of them")
    else:
        print(
            f"!! {args.uia_results} not found -- skipping the uia_only and hybrid "
            "arms. Run scripts/export_uia_arm.py on Windows and copy the JSON over "
            "to get all four."
        )

    precision = resolve_precision()
    dtype = torch_dtype_for(precision)
    print(f"\nloading {MODEL_ID} in {precision} on {torch.cuda.get_device_name(0)}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype).to("cuda")
    base.eval()

    # --- arm 2: zero-shot. The H1 control: same base weights, no adapter. ---
    zero_shot = run_vlm_arm(base, processor, examples, labels_by_id, dataset_root, "zero_shot")
    records += to_eval_records(zero_shot, arm="zero_shot", richness_by_app=richness_by_app)

    if not args.skip_fine_tuned:
        if not args.adapter.exists():
            raise SystemExit(f"adapter not found at {args.adapter} -- has training finished?")
        # --- arm 3: fine-tuned. Same base weights, adapter applied on top,
        # so any difference from zero_shot is attributable to our data. ---
        print(f"\napplying LoRA adapter from {args.adapter}")
        tuned = PeftModel.from_pretrained(base, str(args.adapter))
        tuned.eval()
        fine_tuned = run_vlm_arm(
            tuned, processor, examples, labels_by_id, dataset_root, "fine_tuned"
        )
        records += to_eval_records(fine_tuned, arm="fine_tuned", richness_by_app=richness_by_app)

        # --- arm 4: hybrid. Not a third set of predictions -- a per-example
        # choice between the two already scored above. ---
        if uia_results:
            grounder_hits = {r.example_id: r.hit for r in fine_tuned}
            records += hybrid.combine(uia_results, grounder_hits)

    report = build_report(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("ABLATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
