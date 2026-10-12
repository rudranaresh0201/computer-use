"""
Tests C and D, step 3: what does UIA know about each box a person drew?

    named    an element in the tree covers the box and has a name
    unnamed  an element covers it, with no name (icon buttons, custom controls)
    absent   nothing in the tree covers it (drawn surfaces, Electron with
             accessibility off, custom toolkits)

"Covers" (fixed before any labels exist, 2026-10-12): the element's box
overlaps the person's box with IoU >= 0.5, or contains the box's center
while being at most 4x its area -- loose enough for a hand-drawn box,
strict enough that a window-sized container never counts as "the button".

Test D is the share of each status per app. Test C is the grounder and the
tree methods clicking these targets, scored against the person's box.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from computeruse.web.browser import WebElement

ROLE_OF = {
    "Button": "button", "SplitButton": "button", "MenuItem": "menuitem", "Hyperlink": "link", "Edit": "textbox",
    "ComboBox": "combobox", "CheckBox": "checkbox", "RadioButton": "radio", "TabItem": "tab", "ListItem": "option",
    "TreeItem": "treeitem", "Slider": "slider", "Spinner": "spinbutton", "DataItem": "option",
}


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def covers(element_rect, box) -> bool:
    if _iou(element_rect, box) >= 0.5:
        return True
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ex1, ey1, ex2, ey2 = element_rect
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2 and area(element_rect) <= 4 * max(area(box), 1.0)


def status_of(box_real, elements: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    covering = [e for e in elements if covers(e["rect"], box_real)]
    if not covering:
        return "absent", None
    best = max(covering, key=lambda e: (bool(e["name"].strip()), _iou(e["rect"], box_real)))
    return ("named" if best["name"].strip() else "unnamed"), best


def tree_elements(capture: dict[str, Any]) -> list[WebElement]:
    """The UIA tree as the tree methods (eval/compare.py) read it, in the
    0-1000 space of the screenshot."""
    sx, sy = capture["scale_x"], capture["scale_y"]
    w, h = capture["scaled_size"]
    out = []
    for e in capture["elements"]:
        x1, y1, x2, y2 = e["rect"]
        nx1, ny1, nx2, ny2 = x1 / sx / w * 1000, y1 / sy / h * 1000, x2 / sx / w * 1000, y2 / sy / h * 1000
        out.append(WebElement(f"u{e['i']}", ROLE_OF.get(e["control_type"], e["control_type"].lower() or "generic"),
                              e["name"].strip(), (nx1, ny1, nx2 - nx1, ny2 - ny1)))
    return out


def tree_text(capture: dict[str, Any]) -> str:
    """The tree as an LLM reads it, nesting by indentation (eval/tree_llm.py's format)."""
    lines = []
    for e, el in zip(capture["elements"], tree_elements(capture)):
        name = f' "{el.name}"' if el.name else ""
        lines.append(f'{"  " * e["depth"]}- {el.role}{name} [ref={el.ref}]')
    return "\n".join(lines)


def classify(folder: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items, per_app = [], defaultdict(Counter)
    for meta in sorted(folder.glob("*.uia.json")):
        cid = meta.name[: -len(".uia.json")]
        labels_path = folder / f"{cid}.labels.json"
        if not labels_path.exists():
            continue
        capture = json.loads(meta.read_text(encoding="utf-8"))
        sx, sy = capture["scale_x"], capture["scale_y"]
        w, h = capture["scaled_size"]
        for k, label in enumerate(json.loads(labels_path.read_text(encoding="utf-8"))):
            x1, y1, x2, y2 = label["box"]
            status, element = status_of((x1 * sx, y1 * sy, x2 * sx, y2 * sy), capture["elements"])
            per_app[capture["app"]][status] += 1
            items.append({
                "id": f"{cid}_l{k}", "capture": cid, "app": capture["app"], "condition": status,
                "instruction": label["instruction"],
                "gt_box_norm": [round(x1 / w * 1000), round(y1 / h * 1000), round(x2 / w * 1000), round(y2 / h * 1000)],
                "image": f"{cid}.png", "uia_name": element["name"] if element else None,
                "uia_type": element["control_type"] if element else None,
            })
    coverage = {}
    for app, counts in sorted(per_app.items()):
        total = sum(counts.values())
        coverage[app] = {"labels": total, **{s: counts[s] for s in ("named", "unnamed", "absent")},
                         "exposed": (counts["named"] + counts["unnamed"]) / total, "named_share": counts["named"] / total}
    return items, coverage
