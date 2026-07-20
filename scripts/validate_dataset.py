"""
Hard gate between collection and training.

`inspect_dataset.py` reports what the dataset *contains*. This script asks a
different question: is any of it wrong? It exits non-zero when it finds a
defect that would silently corrupt a training run, so it can be run before
every upload and treated as a build failure rather than a report to skim.

Every check here exists because the defect it looks for actually happened
and cost real GPU runs (docs/journal.md, 2026-07-19):

- WRONG WINDOW: the collector labeled whatever window was in the foreground,
  with nothing verifying it was the target app. When focus was stolen
  mid-run, the *code editor's* elements were written as "file_explorer" and
  "audacity". 196 of 542 rows -- 36% of the dataset, and 100% of one
  held-out app -- were wrong this way, and three training runs were spent
  before anyone rendered a bounding box onto an image and looked at it.
- OUT OF BOUNDS: a coordinate-space mistake (screen-absolute boxes stored
  against a cropped image, say) puts targets outside the picture entirely.
  Unlearnable, and invisible in the loss.
- CHROME ONLY: six screenshots contained nothing but Minimize/Restore/Close,
  because the app never actually rendered its content. They train the model
  that every instruction means "top right corner".
- AMBIGUOUS: one instruction pointing at two different boxes in the same
  image. The model can only learn to average them.

Usage:
    python scripts/validate_dataset.py [dataset_root]
"""

from __future__ import annotations

import collections
import getpass
import json
import os
import sys
from pathlib import Path

from PIL import Image

# Element names that belong to a developer environment, not to any app in
# the registry. Several of these in one screenshot means the collector
# walked the wrong window. Kept as a fingerprint rather than a title check
# because labels.jsonl records no window title -- if that changes, prefer
# checking the title directly.
_DEV_ENV_FINGERPRINT = {
    "explorer", "outline", "timeline", "problems", "output", "terminal",
    "selection", "go", "run", "journal.md", "pyproject.toml", "readme.md",
    ".venv", ".pytest_cache", "learning.md", "roadmap.md", "scripts",
    "tests", "src", "docs", "notebooks", "chat", "sessions",
    "describe what to build", "toggle panel", "customize layout",
}
_FINGERPRINT_THRESHOLD = 3

# Pure window chrome -- present on every app, at the same place, and
# therefore near-worthless as grounding signal in quantity.
_CHROME_NAMES = {"Close", "Minimize", "Maximize", "Restore", "System"}

_MAX_CHROME_FRACTION = 0.20
_MAX_APP_ROW_FRACTION = 0.20


def load(labels_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pii_terms() -> list[str]:
    terms = [getpass.getuser()]
    terms.extend(
        t.strip() for t in os.environ.get("COMPUTERUSE_DATASET_REDACT_TERMS", "").split(",")
        if t.strip()
    )
    return [t.lower() for t in terms if t]


def check_wrong_window(by_shot: dict[str, list[dict]]) -> list[str]:
    errors = []
    for sid, rows in sorted(by_shot.items()):
        names = {r["element"]["name"].strip().lower() for r in rows}
        hits = names & _DEV_ENV_FINGERPRINT
        if len(hits) >= _FINGERPRINT_THRESHOLD:
            errors.append(
                f"WRONG WINDOW: {sid} (app={rows[0]['app']}, {len(rows)} rows) looks like "
                f"a developer-environment window, not the target app -- matched "
                f"{sorted(hits)[:6]}"
            )
    return errors


def check_boxes_in_bounds(samples: list[dict], dataset_root: Path) -> list[str]:
    errors = []
    sizes: dict[str, tuple[int, int]] = {}
    for s in samples:
        path = dataset_root / s["screenshot_path"]
        if not path.exists():
            errors.append(f"MISSING IMAGE: {s['id']} -> {s['screenshot_path']}")
            continue
        if s["screenshot_path"] not in sizes:
            sizes[s["screenshot_path"]] = Image.open(path).size
        scaled_w, scaled_h = sizes[s["screenshot_path"]]
        if (scaled_w, scaled_h) != tuple(s["scaled_size"]):
            errors.append(
                f"SIZE MISMATCH: {s['screenshot_path']} is {scaled_w}x{scaled_h} on disk "
                f"but labels say {tuple(s['scaled_size'])}"
            )
        real_w, real_h = s["real_size"]
        left, top, right, bottom = s["element"]["bbox_real"]
        if left < 0 or top < 0 or right > real_w or bottom > real_h:
            errors.append(
                f"OUT OF BOUNDS: {s['id']} box {s['element']['bbox_real']} is outside "
                f"the {real_w}x{real_h} captured frame"
            )
        if right <= left or bottom <= top:
            errors.append(f"DEGENERATE BOX: {s['id']} box {s['element']['bbox_real']}")
    return errors


def check_ambiguous(by_shot: dict[str, list[dict]]) -> list[str]:
    errors = []
    for sid, rows in sorted(by_shot.items()):
        seen: dict[str, set] = collections.defaultdict(set)
        for r in rows:
            seen[r["instruction"]].add(tuple(r["element"]["bbox_real"]))
        for instruction, boxes in seen.items():
            if len(boxes) > 1:
                errors.append(
                    f"AMBIGUOUS: {sid} has {instruction!r} pointing at {len(boxes)} "
                    f"different boxes -- the model can only learn their average"
                )
    return errors


def check_split_leakage(samples: list[dict]) -> list[str]:
    splits_per_shot: dict[str, set] = collections.defaultdict(set)
    for s in samples:
        splits_per_shot[s["screenshot_id"]].add(s["split"])
    return [
        f"SPLIT LEAKAGE: {sid} appears in {sorted(v)} -- splits must be screenshot-level"
        for sid, v in sorted(splits_per_shot.items())
        if len(v) > 1
    ]


def check_pii(samples: list[dict]) -> list[str]:
    terms = _pii_terms()
    errors = []
    for s in samples:
        low = s["instruction"].lower()
        for term in terms:
            if term in low:
                errors.append(f"PII: {s['id']} instruction contains {term!r}: {s['instruction']!r}")
                break
    return errors


def check_composition(samples: list[dict], by_shot: dict[str, list[dict]]) -> list[str]:
    """Not corruption -- skew. Warnings, since the right threshold is a
    judgement call, but each one materially changes what the model learns."""
    warnings = []
    chrome = [s for s in samples if s["element"]["name"] in _CHROME_NAMES]
    if samples and len(chrome) / len(samples) > _MAX_CHROME_FRACTION:
        warnings.append(
            f"CHROME HEAVY: {len(chrome)}/{len(samples)} "
            f"({len(chrome)/len(samples):.0%}) of rows are window chrome "
            f"(Close/Minimize/...), which sits at the same place in every app"
        )
    for sid, rows in sorted(by_shot.items()):
        if all(r["element"]["name"] in _CHROME_NAMES for r in rows):
            warnings.append(
                f"CHROME ONLY: {sid} ({rows[0]['app']}) contains nothing but window "
                f"chrome -- the app's content was never captured"
            )
    per_app = collections.Counter(s["app"] for s in samples)
    for app, n in per_app.most_common(3):
        if samples and n / len(samples) > _MAX_APP_ROW_FRACTION:
            warnings.append(
                f"APP SKEW: {app} is {n}/{len(samples)} ({n/len(samples):.0%}) of all "
                f"rows -- that share of the gradient goes to one app"
            )
    return warnings


def main() -> int:
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/gui_grounding")
    samples = load(dataset_root / "labels.jsonl")
    by_shot: dict[str, list[dict]] = collections.defaultdict(list)
    for s in samples:
        by_shot[s["screenshot_id"]].append(s)

    print(f"{len(samples)} rows across {len(by_shot)} screenshots in {dataset_root}")
    print()

    errors: list[str] = []
    errors += check_wrong_window(by_shot)
    errors += check_boxes_in_bounds(samples, dataset_root)
    errors += check_ambiguous(by_shot)
    errors += check_split_leakage(samples)
    errors += check_pii(samples)
    warnings = check_composition(samples, by_shot)

    for w in warnings:
        print(f"  warn  {w}")
    if warnings:
        print()
    for e in errors:
        print(f"  FAIL  {e}")

    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s).")
        print("Do NOT upload or train on this dataset until these are resolved.")
        return 1
    print(f"PASSED: 0 errors, {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
