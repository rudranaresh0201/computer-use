"""
Draw every label's bounding box onto its screenshot, so a human can look at
the data instead of trusting a row count.

This is the check that found the 2026-07-19 disaster. `inspect_dataset.py`
reported "0 integrity issues" on a dataset where 36% of rows were the code
editor's UI labeled as other applications, because it verified schema and
file existence -- not whether the picture showed the app the label claimed.
Numbers cannot catch that. A rendered box can, instantly.

Run it after every collection, per app, before uploading anything:

    python scripts/render_labels.py --app calculator
    python scripts/render_labels.py --app calculator --since 4
    python scripts/render_labels.py --all

Output goes to runs/label_review/<app>/. Red box = labeled target, green
dot = the exact point the model is trained to predict (the box centre).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "gui_grounding"
OUT_ROOT = REPO_ROOT / "runs" / "label_review"


def render(rows: list[dict], dataset_root: Path, out_path: Path, half: bool) -> None:
    first = rows[0]
    image = Image.open(dataset_root / first["screenshot_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x, scale_y = first["scale_x"], first["scale_y"]

    for row in rows:
        left, top, right, bottom = row["element"]["bbox_real"]
        box = (left / scale_x, top / scale_y, right / scale_x, bottom / scale_y)
        draw.rectangle(box, outline=(255, 0, 0), width=3)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 255, 0))

    if half:
        image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", help="only render this app")
    parser.add_argument("--all", action="store_true", help="render every app")
    parser.add_argument(
        "--since",
        type=int,
        default=None,
        help="only screenshots with a sequence number >= this (skip already-reviewed ones)",
    )
    parser.add_argument("--full-size", action="store_true", help="don't halve the output")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()

    if not args.app and not args.all:
        parser.error("pass --app <name> or --all")

    rows = [
        json.loads(line)
        for line in (args.dataset_root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.app:
        rows = [r for r in rows if r["app"] == args.app]
    if args.since is not None:
        rows = [r for r in rows if int(r["screenshot_id"].rsplit("_", 1)[1]) >= args.since]

    if not rows:
        print("no rows matched")
        return 1

    by_shot: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_shot[row["screenshot_id"]].append(row)

    print(f"rendering {len(by_shot)} screenshots ({len(rows)} boxes)")
    for sid, shot_rows in sorted(by_shot.items()):
        app = shot_rows[0]["app"]
        out_path = OUT_ROOT / app / f"{sid}.png"
        render(shot_rows, args.dataset_root, out_path, half=not args.full_size)
        # Windows consoles default to cp1252; real UIA names contain
        # characters it can't encode (Calculator's fullwidth plus, U+FF0B),
        # which crashed this script mid-render. Never let a print kill a
        # review run.
        names = [
            r["element"]["name"].encode("ascii", "replace").decode("ascii")
            for r in shot_rows
        ][:5]
        print(f"  {sid:26s} {len(shot_rows):3d} boxes  e.g. {names}")

    print()
    print(f"open them: {OUT_ROOT}")
    print("Check: are the boxes on the app named in the path, and on things")
    print("that are actually VISIBLE (not hidden behind an open menu)?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
