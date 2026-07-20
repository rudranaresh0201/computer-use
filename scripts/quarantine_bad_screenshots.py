"""
Remove specific screenshots (and their label rows) from the dataset, and
reset the progress ledger so the orchestrator re-collects exactly those
states on the next run.

Written 2026-07-19 for the wrong-window contamination found by
`validate_dataset.py`: the collector labeled the development editor's
elements as "file_explorer" and "audacity" whenever focus was stolen
mid-run. Those screenshots are unsalvageable -- the picture is of the wrong
application -- so they're removed rather than repaired, and re-collected
with the now-verifying collector.

Everything is backed up before deletion, and the script prints exactly what
it will do. Nothing is guessed: the screenshots to drop are named
explicitly on the command line, not inferred.

Usage:
    python scripts/quarantine_bad_screenshots.py --list
    python scripts/quarantine_bad_screenshots.py --dry-run file_explorer_0000 ...
    python scripts/quarantine_bad_screenshots.py --apply file_explorer_0000 ...
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import time
from pathlib import Path

DATASET_ROOT = Path("data/gui_grounding")


def load_rows(labels_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshots", nargs="*", help="screenshot_id values to remove")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--apply", action="store_true", help="actually delete (default is dry-run)")
    parser.add_argument("--list", action="store_true", help="list screenshots and exit")
    args = parser.parse_args()

    root = args.dataset_root
    labels_path = root / "labels.jsonl"
    rows = load_rows(labels_path)
    by_shot: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_shot[r["screenshot_id"]].append(r)

    if args.list:
        for sid, rs in sorted(by_shot.items()):
            print(f"{sid:26s} app={rs[0]['app']:18s} split={rs[0]['split']:18s} rows={len(rs)}")
        return 0

    targets = set(args.screenshots)
    unknown = targets - set(by_shot)
    if unknown:
        print(f"ERROR: unknown screenshot_id(s): {sorted(unknown)}")
        return 1
    if not targets:
        parser.error("name at least one screenshot_id, or pass --list")

    # which (app, state) pairs must be re-collected? The ledger keys states
    # by name, which labels.jsonl doesn't record -- so reset every state of
    # each affected app and let the orchestrator redo that app in full.
    affected_apps = sorted({by_shot[s][0]["app"] for s in targets})
    doomed_rows = [r for r in rows if r["screenshot_id"] in targets]
    keep_rows = [r for r in rows if r["screenshot_id"] not in targets]
    images = sorted({root / r["screenshot_path"] for r in doomed_rows})

    print(f"screenshots to remove : {sorted(targets)}")
    print(f"label rows to remove  : {len(doomed_rows)} (of {len(rows)}; {len(keep_rows)} kept)")
    print(f"image files to remove : {len(images)}")
    print(f"apps to re-collect    : {affected_apps}")
    print()

    ledger_path = root / "progress.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    to_clear = [s for s in ledger["completed_states"] if s.split(":", 1)[0] in affected_apps]
    print(f"ledger entries to clear ({len(to_clear)}):")
    for entry in to_clear:
        print(f"    {entry}")
    print()

    if not args.apply:
        print("DRY RUN -- nothing changed. Re-run with --apply to do it.")
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%S")
    backup = root / f"_quarantine_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, backup / "labels.jsonl")
    shutil.copy2(ledger_path, backup / "progress.json")
    for image in images:
        if image.exists():
            shutil.copy2(image, backup / image.name)
    print(f"backed up labels, ledger and {len(images)} images to {backup}")

    labels_path.write_text(
        "".join(json.dumps(r) + "\n" for r in keep_rows), encoding="utf-8"
    )
    for image in images:
        if image.exists():
            image.unlink()

    ledger["completed_states"] = sorted(set(ledger["completed_states"]) - set(to_clear))
    # leave next_seq alone: it must keep counting up so a re-collected state
    # gets a fresh screenshot_id instead of overwriting a file another row
    # still references.
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    print(f"removed {len(doomed_rows)} rows and {len(images)} images")
    print(f"cleared {len(to_clear)} ledger entries")
    print()
    print("Next: re-run collection for", ",".join(affected_apps))
    print("  python scripts/run_dataset_collection.py --apps " + ",".join(affected_apps))
    print("Then: python scripts/validate_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
