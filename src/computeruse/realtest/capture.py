"""
Tests C and D, step 1: `computeruse capture <app>` saves what an agent would
get from a real Windows app -- the screenshot and the FULL UI Automation
tree of the foreground window, unnamed elements included.

The v3 dataset kept only UIA's named elements (they were the labels), so it
can't say whether something a person can click is missing from the tree.
These captures can: a person draws boxes on the screenshot (label.py), and
classify.py checks each box against this tree -- named, unnamed, or absent.

The window's title and class are saved with the tree, because the one time
this project trusted "the foreground window is the app I think it is",
36% of a dataset turned out to be the wrong app (journal, 2026-07-19).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from computeruse.perception import screenshot

OUT_DIR = Path("runs/realtest")


def walk(node, out: list[dict[str, Any]], depth: int = 0, parent: int = -1, max_elements: int = 1500, max_depth: int = 25) -> None:
    """Every element with a non-empty box, depth-first, with its depth and
    parent index so the tree's nesting survives the trip to JSON."""
    if len(out) >= max_elements or depth > max_depth:
        return
    index = parent
    try:
        rect = node.rectangle()
        if rect.width() > 0 and rect.height() > 0:
            index = len(out)
            out.append({
                "i": index, "parent": parent, "depth": depth,
                "name": node.window_text() or "",
                "control_type": node.element_info.control_type or "",
                "automation_id": getattr(node.element_info, "automation_id", None) or None,
                "rect": [rect.left, rect.top, rect.right, rect.bottom],
            })
    except Exception:  # noqa: BLE001 -- one bad node must not end the walk
        pass
    try:
        children = node.children()
    except Exception:  # noqa: BLE001
        return
    for child in children:
        if len(out) >= max_elements:
            return
        walk(child, out, depth + 1, index, max_elements, max_depth)


def capture(app: str, out_dir: Path = OUT_DIR, delay: int = 5, log: Callable[[str], None] = print) -> Path:
    from pywinauto import Desktop

    out_dir.mkdir(parents=True, exist_ok=True)
    for left in range(delay, 0, -1):
        log(f"  {left}... click on the {app} window now")
        time.sleep(1)
    window = Desktop(backend="uia").window(active_only=True, visible_only=True).wrapper_object()
    elements: list[dict[str, Any]] = []
    walk(window, elements)
    shot = screenshot.capture()
    rect = window.rectangle()
    stem = f"{app}_{time.strftime('%Y%m%d-%H%M%S')}"
    (out_dir / f"{stem}.png").write_bytes(shot.png_bytes)
    record = {
        "id": stem, "app": app, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window": {"title": window.window_text() or "", "class_name": window.class_name() or "",
                   "rect": [rect.left, rect.top, rect.right, rect.bottom]},
        "real_size": list(shot.real_size), "scaled_size": list(shot.scaled_size),
        "scale_x": shot.scale_x, "scale_y": shot.scale_y, "elements": elements,
    }
    (out_dir / f"{stem}.uia.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    named = sum(1 for e in elements if e["name"].strip())
    log(f"saved {stem}: window {record['window']['title']!r} ({record['window']['class_name']}), "
        f"{len(elements)} UIA elements, {named} with a name")
    return out_dir / f"{stem}.png"
