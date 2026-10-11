"""
The ablation's decisions and its report, as pure functions -- everything
between "examples were scored" and "here is what H1-H3 say", testable
without a GPU. The Kaggle notebook (notebooks/phase3_eval_kaggle.ipynb)
does the scoring; this module decides what the scores mean.

Protocol (docs/research/gui-grounding-hypothesis.md):
1. Choose the adapter on dev only: every saved snapshot is scored on one
   fixed, seeded dev subset, and the best click accuracy wins. Zero-shot
   is scored on the same subset as step 0 of the learning curve, never as
   a candidate.
2. Score zero-shot and the chosen adapter on dev + both test splits,
   once. Nothing is chosen after a test number has been seen.
3. Report the four arms sliced by arm, richness and split, each accuracy
   with a 95% bootstrap interval, and H1/H3 as paired differences.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from computeruse.eval import hybrid
from computeruse.eval.report import (
    EvalRecord,
    accuracy,
    build_report,
    ci_by_arm_and_split,
    paired_difference,
)
from computeruse.eval.vlm_grounder import (
    ZERO_SHOT,
    ArmResult,
    format_counts,
    summarize_diagnostics,
    to_eval_records,
)
from computeruse.training.prepare_dataset import SPLITS, TrainingExample

EVAL_SPLITS = tuple(s for s in SPLITS if s != "train")
FINAL = "final"
_SNAPSHOT_RE = re.compile(r"^adapters/(step-\d{5})/adapter_model\.safetensors$")


@dataclass
class UiaRow:
    """eval.uia_only.UiaArmResult, rebuilt from the JSON
    scripts/export_uia_arm.py writes on Windows. Not imported from uia_only
    on purpose: that module reaches pywinauto, which cannot import on
    Linux. hybrid.combine only reads attributes, so a structurally
    identical object is all it needs."""

    example_id: str
    app: str
    app_richness: str
    split: str
    available: bool
    hit: bool


def load_uia_results(path: Path) -> list[UiaRow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(payload["splits"]) != EVAL_SPLITS:
        raise ValueError(
            f"{path} was exported over splits {payload['splits']}, but the ablation "
            f"scores {list(EVAL_SPLITS)}. Re-export before joining -- hybrid needs "
            "both sides to cover the same examples."
        )
    return [UiaRow(**row) for row in payload["results"]]


# -- which adapters exist, and choosing one --------------------------------


def adapter_arms(repo_files: list[str]) -> list[str]:
    """The adapters the training run left on the Hub, oldest first:
    "step-00100" ... then "final". A snapshot counts only with both of its
    files present."""
    files = set(repo_files)
    snapshots = sorted(
        m.group(1)
        for f in files
        if (m := _SNAPSHOT_RE.match(f)) and f"adapters/{m.group(1)}/adapter_config.json" in files
    )
    has_final = {f"{FINAL}/adapter_model.safetensors", f"{FINAL}/adapter_config.json"} <= files
    return snapshots + ([FINAL] if has_final else [])


def adapter_path(arm: str, download_root: Path) -> Path:
    return Path(download_root) / (FINAL if arm == FINAL else f"adapters/{arm}")


def step_of(arm: str, final_step: int) -> int:
    if arm == ZERO_SHOT:
        return 0
    if arm == FINAL:
        return final_step
    return int(arm.removeprefix("step-"))


def dev_selection_subset(dev: list[TrainingExample], size: int, seed: int = 0) -> list[TrainingExample]:
    """A fixed dev subset for choosing the adapter. Sorted by id before
    sampling, so it depends only on (ids, size, seed) and never on file
    order -- a resumed session must pick exactly the same examples."""
    ordered = sorted(dev, key=lambda e: e.id)
    if size >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, size), key=lambda e: e.id)


def learning_curve(dev_results: dict[str, list[ArmResult]], final_step: int) -> list[dict]:
    """Dev click accuracy at each saved step; zero-shot is step 0 (a LoRA
    adapter starts as an exact no-op, so the untrained model IS step 0)."""
    rows = []
    for arm, results in dev_results.items():
        diagnostics = summarize_diagnostics(results)
        rows.append(
            {
                "arm": arm,
                "step": step_of(arm, final_step),
                "n": len(results),
                "accuracy": sum(r.hit for r in results) / len(results) if results else 0.0,
                "parse_rate": diagnostics["parse_rate"],
                "median_center_distance": (
                    diagnostics["median_center_distance"] if diagnostics["parse_rate"] > 0 else None
                ),
            }
        )
    return sorted(rows, key=lambda r: r["step"])


def choose_adapter(curve: list[dict]) -> str:
    """Highest dev click accuracy among the adapters -- the metric the
    hypotheses are about, not dev loss, which scores "137" and "927" as
    equally wrong when the answer is "127". Ties go to the smaller median
    center distance, then to the later step. Zero-shot is never a
    candidate: it is the baseline the chosen adapter is compared against."""
    candidates = [row for row in curve if row["arm"] != ZERO_SHOT]
    if not candidates:
        raise ValueError("no adapters were scored on dev -- nothing to choose from")

    def key(row: dict) -> tuple:
        distance = row["median_center_distance"]
        return row["accuracy"], -(distance if distance is not None else math.inf), row["step"]

    return max(candidates, key=key)["arm"]


# -- the report -------------------------------------------------------------


def _hits(results: list[ArmResult], split: str) -> dict[str, bool]:
    return {r.example_id: r.hit for r in results if r.split == split}


def assemble_report(
    *,
    zero_shot: list[ArmResult],
    fine_tuned: list[ArmResult],
    chosen_arm: str,
    richness_by_app: dict[str, str],
    uia_rows: Optional[list[UiaRow]] = None,
    curve: Optional[list[dict]] = None,
    selection: Optional[dict] = None,
    n_boot: int = 2000,
) -> dict:
    """Everything H1-H3 need from the two model arms (scored on the same
    examples) and, when given, the UIA arm exported on Windows. The UIA
    rows are restricted to the examples the model arms scored; the run
    itself checks that this is all of them."""
    ids = {r.example_id for r in zero_shot}
    if ids != {r.example_id for r in fine_tuned}:
        raise ValueError("zero_shot and fine_tuned scored different examples")

    records: list[EvalRecord] = to_eval_records(zero_shot, ZERO_SHOT, richness_by_app)
    records += to_eval_records(fine_tuned, "fine_tuned", richness_by_app)
    uia_resolved = None
    if uia_rows is not None:
        uia_scored = [u for u in uia_rows if u.example_id in ids]
        missing = ids - {u.example_id for u in uia_scored}
        if missing:
            raise ValueError(f"{len(missing)} scored examples have no UIA result, e.g. {sorted(missing)[:3]}")
        records += [
            EvalRecord(u.example_id, "uia_only", u.app, u.app_richness, u.split, u.hit) for u in uia_scored
        ]
        records += hybrid.combine(uia_scored, {r.example_id: r.hit for r in fine_tuned})
        uia_resolved = {"resolved": sum(u.available for u in uia_scored), "of": len(uia_scored)}

    report = build_report(records)
    report["ci95_by_arm_and_split"] = ci_by_arm_and_split(records, n_boot=n_boot)
    report["h1_test_same_app"] = paired_difference(
        _hits(fine_tuned, "test_same_app"), _hits(zero_shot, "test_same_app"), n_boot=n_boot
    )
    report["h3_test_held_out_app"] = paired_difference(
        _hits(fine_tuned, "test_held_out_app"), _hits(zero_shot, "test_held_out_app"), n_boot=n_boot
    )
    # H3's other half: held-out apps should be harder than held-out screenshots.
    ft_records = [r for r in records if r.arm == "fine_tuned"]
    report["h3_fine_tuned_same_vs_held_out"] = {
        split: accuracy([r for r in ft_records if r.split == split])
        for split in ("test_same_app", "test_held_out_app")
    }
    report["diagnostics"] = {
        name: {
            split: {**summarize_diagnostics(subset), "formats": format_counts(subset)}
            for split in EVAL_SPLITS
            if (subset := [r for r in results if r.split == split])
        }
        for name, results in ((ZERO_SHOT, zero_shot), ("fine_tuned", fine_tuned))
    }
    report["chosen_adapter"] = chosen_arm
    report["uia_resolved"] = uia_resolved
    report["selection"] = {**(selection or {}), "curve": curve or []}
    return report
