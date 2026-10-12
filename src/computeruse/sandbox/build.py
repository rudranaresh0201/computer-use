"""
Render generated pages (pages.py) in Chrome and write a test set where every
item's condition has been checked against the real accessibility tree.

    <out>/items.jsonl         one target per line: instruction, true box, condition
    <out>/pages/<id>.png      the screenshot (VIEWPORT, no scrolling)
    <out>/pages/<id>.tree.txt the accessibility tree with boxes -- what a tree method gets
    <out>/manifest.json       seeds, counts, and every item dropped and why

True boxes come from the DOM (getBoundingClientRect of the marked element)
or, for canvas targets, from where we drew them -- never from the tree, and
never from a model. Coordinates are also given normalized to 0-1000 over
the screenshot, the space the grounding model answers in.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from computeruse.sandbox.pages import VIEWPORT, Page, Target, generate
from computeruse.web.browser import Browser, WebElement, parse_snapshot

STOPWORDS = {"click", "press", "the", "a", "an", "on", "in", "of", "to", "button", "my", "this", "it", "these",
             "settings", "panel", "i", "need", "what", "just", "did", "me"}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def shares_a_word(instruction: str, label: str) -> bool:
    """Same word, or same 4-letter stem ("print" / "printer")."""
    a, b = content_words(instruction), content_words(label)
    return any(x == y or (len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4]) for x in a for y in b)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    w, h = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(element: WebElement) -> tuple[float, float, float, float]:
    x, y, w, h = element.box
    return x, y, x + w, y + h


def tree_element_at(elements: list[WebElement], box: tuple[float, float, float, float]) -> WebElement | None:
    """The tree element covering this box, if the tree has one."""
    best = max((e for e in elements if e.box), key=lambda e: _iou(_xyxy(e), box), default=None)
    return best if best is not None and _iou(_xyxy(best), box) > 0.6 else None


def check(target: Target, box: tuple[float, float, float, float], elements: list[WebElement], snapshot: str) -> str | None:
    """None if the target really is its condition on the rendered page; otherwise why not."""
    found = tree_element_at(elements, box)
    if target.condition == "C1":
        if found is None or found.name != target.label:
            return f"C1 target is not in the tree as {target.label!r}"
    elif target.condition == "C2":
        if found is None or found.name:
            return f"C2 target should be in the tree with no name, got {found.name if found else None!r}"
    elif target.condition == "C3":
        if found is None or found.name != target.label:
            return f"C3 target is not in the tree as {target.label!r}"
        if shares_a_word(target.instruction, target.label):
            return f"C3 instruction {target.instruction!r} shares a word with {target.label!r}"
    elif target.condition == "C4":
        if found is not None:
            return f"C4 target is visible to the tree as {found.role} {found.name!r}"
        if re.search(rf'"{re.escape(target.label)}"|: {re.escape(target.label)}\b', snapshot):
            return f"C4 label {target.label!r} appears in the tree"
    elif target.condition == "C5":
        if found is None or found.name != target.label:
            return f"C5 target is not in the tree as {target.label!r}"
        if sum(e.name == target.label for e in elements) < 2:
            return f"C5 label {target.label!r} is not ambiguous in the tree"
    return None


def visible_in_pixels(png: bytes, box: tuple[float, float, float, float], min_std: float = 12.0) -> bool:
    """A target must be something you can see: its box can't be one flat
    color. Catches a target that is in the DOM but didn't render (the canvas
    bug of 2026-10-12: the tree check passed, the screenshot was empty)."""
    import io

    from PIL import Image, ImageStat

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = Image.open(io.BytesIO(png)).convert("L").crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
    return ImageStat.Stat(crop).stddev[0] >= min_std


def _true_box(web: Browser, target: Target) -> tuple[float, float, float, float]:
    if target.canvas_rect is not None:
        cx, cy = web.page.evaluate("(() => { const r = document.getElementById('tools').getBoundingClientRect(); return [r.x, r.y]; })()")
        x, y, w, h = target.canvas_rect
        return cx + x, cy + y, cx + x + w, cy + y + h
    x, y, w, h = web.page.evaluate(
        "(id) => { const r = document.querySelector(`[data-target=\"${id}\"]`).getBoundingClientRect();"
        " return [r.x, r.y, r.width, r.height]; }",
        target.target_id,
    )
    return x, y, x + w, y + h


def normalize(box: tuple[float, float, float, float]) -> list[int]:
    w, h = VIEWPORT
    x1, y1, x2, y2 = box
    return [round(x1 / w * 1000), round(y1 / h * 1000), round(x2 / w * 1000), round(y2 / h * 1000)]


def build(out: Path, n_pages: int, first_seed: int = 0, browser: Browser | None = None) -> dict[str, Any]:
    out = Path(out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    items, dropped = [], []
    own = browser is None
    web = browser or Browser(headless=True, viewport=VIEWPORT).__enter__()
    try:
        for seed in range(first_seed, first_seed + n_pages):
            page: Page = generate(seed)
            web.page.set_content(page.html)
            web.page.wait_for_timeout(50)  # let the canvas script draw
            snapshot = web.page.locator("body").aria_snapshot(mode="ai", boxes=True)
            elements = parse_snapshot(snapshot)
            png = web.page.screenshot()
            (out / "pages" / f"{page.page_id}.png").write_bytes(png)
            (out / "pages" / f"{page.page_id}.tree.txt").write_text(snapshot, encoding="utf-8")
            for target in page.targets:
                box = _true_box(web, target)
                problem = check(target, box, elements, snapshot)
                if problem is None and not visible_in_pixels(png, box):
                    problem = f"{target.condition} target is not visible in the screenshot"
                if problem:
                    dropped.append({"id": target.target_id, "why": problem})
                    continue
                items.append({
                    "id": target.target_id, "page_id": page.page_id, "condition": target.condition,
                    "instruction": target.instruction, "label": target.label,
                    "gt_box_px": [round(v, 1) for v in box], "gt_box_norm": normalize(box),
                    "image": f"pages/{page.page_id}.png", "tree": f"pages/{page.page_id}.tree.txt",
                })
    finally:
        if own:
            web.__exit__(None, None, None)
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    counts: dict[str, int] = {}
    for item in items:
        counts[item["condition"]] = counts.get(item["condition"], 0) + 1
    manifest = {"pages": n_pages, "first_seed": first_seed, "viewport": list(VIEWPORT), "items": len(items),
                "by_condition": dict(sorted(counts.items())), "dropped": dropped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["build", "check", "content_words", "shares_a_word", "normalize", "asdict"]
