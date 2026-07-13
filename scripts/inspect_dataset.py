"""
Inspects data/gui_grounding/labels.jsonl for correctness and distribution --
run after every collection session, not just once. Checks the two things a
training run can't recover from if wrong (a broken image reference, a
mislabeled split) plus the distributional facts the hypothesis doc needs
verified empirically rather than assumed at design time (per-app richness,
per split, per control_type).

Run: .venv/Scripts/python.exe scripts/inspect_dataset.py [dataset_root]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "gui_grounding"


def load_samples(labels_path: Path) -> list[dict]:
    samples = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  ! malformed JSON at line {line_no}: {exc}")
    return samples


def check_integrity(samples: list[dict], dataset_root: Path) -> list[str]:
    """Things that would silently corrupt training if left unchecked."""
    issues = []
    seen_ids = set()
    missing_images = set()

    for s in samples:
        if s["id"] in seen_ids:
            issues.append(f"duplicate sample id: {s['id']}")
        seen_ids.add(s["id"])

        if not s["instruction"].strip():
            issues.append(f"empty instruction: {s['id']}")

        bbox = s["element"]["bbox_real"]
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            issues.append(f"degenerate bbox {bbox}: {s['id']}")

        img_path = dataset_root / s["screenshot_path"]
        if img_path not in missing_images and not img_path.exists():
            issues.append(f"missing image file: {s['screenshot_path']}")
            missing_images.add(img_path)

    return issues


def report(samples: list[dict]) -> None:
    print(f"\ntotal labeled examples: {len(samples)}")

    screenshot_ids = {s["screenshot_id"] for s in samples}
    print(f"total screenshots:      {len(screenshot_ids)}")

    print("\nby split:")
    for split, count in Counter(s["split"] for s in samples).most_common():
        print(f"  {split:20s} {count}")

    print("\nby app (examples / screenshots / avg elements per screenshot):")
    per_app_screens: dict[str, set] = defaultdict(set)
    per_app_examples: Counter = Counter()
    per_app_richness: dict[str, str] = {}
    for s in samples:
        per_app_examples[s["app"]] += 1
        per_app_screens[s["app"]].add(s["screenshot_id"])
        per_app_richness[s["app"]] = s["app_richness"]
    for app in sorted(per_app_examples):
        n_screens = len(per_app_screens[app])
        avg = per_app_examples[app] / n_screens if n_screens else 0
        print(
            f"  {app:20s} {per_app_examples[app]:4d} examples / "
            f"{n_screens:2d} screenshots / {avg:5.1f} avg elements "
            f"(declared richness: {per_app_richness[app]})"
        )

    print("\nby control_type:")
    for ctype, count in Counter(s["element"]["control_type"] for s in samples).most_common():
        print(f"  {ctype:15s} {count}")

    print("\nrichness sanity check (avg elements/screenshot, declared 'rich' should exceed 'weak'):")
    rich_avgs, weak_avgs = [], []
    for app in sorted(per_app_examples):
        n_screens = len(per_app_screens[app])
        avg = per_app_examples[app] / n_screens if n_screens else 0
        (rich_avgs if per_app_richness[app] == "rich" else weak_avgs).append(avg)
    if rich_avgs and weak_avgs:
        print(f"  rich apps avg: {sum(rich_avgs)/len(rich_avgs):.1f}   weak apps avg: {sum(weak_avgs)/len(weak_avgs):.1f}")
    else:
        print("  (need at least one collected app from each richness label to compare)")


def main() -> None:
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    labels_path = dataset_root / "labels.jsonl"

    if not labels_path.exists():
        print(f"no labels.jsonl at {labels_path} -- nothing collected yet")
        sys.exit(1)

    samples = load_samples(labels_path)
    if not samples:
        print("labels.jsonl exists but is empty")
        sys.exit(1)

    issues = check_integrity(samples, dataset_root)
    report(samples)

    print(f"\nintegrity issues: {len(issues)}")
    for issue in issues[:20]:
        print(f"  ! {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more")

    target_total = 210  # design doc's projected screenshot count across all 14 apps
    n_screens = len({s["screenshot_id"] for s in samples})
    print(
        f"\nreadiness: {n_screens}/{target_total} target screenshots collected "
        f"({n_screens / target_total:.0%}). Not ready to train -- collection is "
        f"a small subset of the full 14-app registry so far."
        if n_screens < target_total
        else "\nreadiness: target screenshot count reached."
    )


if __name__ == "__main__":
    main()
