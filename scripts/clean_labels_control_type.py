"""
One-time cleanup for data/gui_grounding/labels.jsonl after collector.py's
control-type fix (2026-07-14): the collector used to label Window/Text/Pane
elements too, which meant a window, its sub-panes, and a text label could
all share one UIA name (e.g. "Calculator") and collapse to the identical
generic "Click {name}" instruction pointing at genuinely different boxes --
this is what caused the v1 model to output the same coordinate regardless of
the actual screenshot for that instruction (see docs/journal.md, v1 training
run). filter_elements() now excludes those control types going forward, but
the 8 apps collected before the fix already have the bad rows baked into
labels.jsonl. Re-walking those apps to regenerate them from scratch is not
necessary -- every row already carries its element's control_type, so this
script just re-applies the same rule retroactively, offline, using the
collector's own template dict as the single source of truth for "which
control types are real instruction targets."

Writes a timestamped backup of the original file before overwriting, since
labels.jsonl represents real collection work that isn't cheap to redo.

Run: .venv/Scripts/python.exe scripts/clean_labels_control_type.py [dataset_root]
"""

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from computeruse.dataset.collector import _INSTRUCTION_TEMPLATES

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "gui_grounding"


def load_samples(labels_path: Path) -> list[dict]:
    samples = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> None:
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    labels_path = dataset_root / "labels.jsonl"

    if not labels_path.exists():
        print(f"no labels.jsonl at {labels_path}")
        sys.exit(1)

    samples = load_samples(labels_path)
    allowed_types = set(_INSTRUCTION_TEMPLATES)

    kept = [s for s in samples if s["element"]["control_type"] in allowed_types]
    dropped = [s for s in samples if s["element"]["control_type"] not in allowed_types]

    print(f"total rows before: {len(samples)}")
    print(f"kept (real instruction target types): {len(kept)}")
    print(f"dropped (no real template -- Window/Text/Pane/...): {len(dropped)}")

    if dropped:
        print("\ndropped, by control_type:")
        for ctype, count in Counter(s["element"]["control_type"] for s in dropped).most_common():
            print(f"  {ctype:15s} {count}")
        print("\ndropped, by app:")
        for app, count in Counter(s["app"] for s in dropped).most_common():
            print(f"  {app:15s} {count}")

    if not dropped:
        print("\nnothing to clean -- labels.jsonl already only contains real instruction targets.")
        return

    backup_path = labels_path.with_name(
        f"labels.jsonl.pre_cleanup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.bak"
    )
    shutil.copy2(labels_path, backup_path)
    print(f"\nbacked up original to {backup_path}")

    with labels_path.open("w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample) + "\n")
    print(f"wrote {len(kept)} cleaned rows to {labels_path}")
    print("\nnext: regenerate splits with prepare_dataset.py's main()")


if __name__ == "__main__":
    main()
