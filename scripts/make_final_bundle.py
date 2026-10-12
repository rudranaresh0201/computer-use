"""
Pack the inputs of the last GPU run into one zip for a Kaggle dataset
(`compuse-final`). Each part is a folder with items.jsonl, the images, and
bundle.json naming it; the notebook (phase3_final_kaggle.ipynb) scores every
part it finds.

    guard_dev   sandbox DEV pages (seeds 100-159) -- the grounder's answers
                here are what a better guard is designed on; never test pages
    realtest    Test C: real apps, boxes drawn by a person (runs/realtest,
                classified against the full UIA tree)

Zip paths are written with forward slashes (Kaggle rejects backslashes).

    python scripts/make_final_bundle.py [out.zip]
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from computeruse.realtest.classify import classify

GUARD_DEV = Path("runs/guard_dev")
REALTEST = Path("runs/realtest")


def parts() -> dict[str, tuple[list[dict], Path]]:
    found = {}
    items = [json.loads(line) for line in (GUARD_DEV / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert json.loads((GUARD_DEV / "manifest.json").read_text())["first_seed"] < 1000, "guard_dev must be dev pages"
    found["guard_dev"] = (items, GUARD_DEV)
    if REALTEST.is_dir():
        real, coverage = classify(REALTEST)
        if real:
            found["realtest"] = (real, REALTEST)
            print("realtest coverage:", json.dumps(coverage, indent=1))
    return found


def main(out: Path) -> None:
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, (items, root) in parts().items():
            z.writestr(f"{name}/bundle.json", json.dumps({"name": name, "items": len(items)}))
            z.writestr(f"{name}/items.jsonl", "".join(json.dumps(i) + "\n" for i in items))
            for image in sorted({i["image"] for i in items}):
                z.write(root / image, Path(name, image).as_posix())
            by = {}
            for i in items:
                by[i["condition"]] = by.get(i["condition"], 0) + 1
            print(f"{name}: {len(items)} items {by}")
    with zipfile.ZipFile(out) as z:
        assert not any("\\" in n for n in z.namelist()), "backslash in a zip path"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop" / "compuse-final.zip")
