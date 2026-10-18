"""
Checkers for real apps: each reads the app's own saved state and gives (passed, detail). The brain's
"done" is a claim; this is the verdict.

    excalidraw   the scene excalidraw.com keeps in localStorage["excalidraw"] (its own autosave):
                 every expected label is written somewhere, and every expected connection is an
                 arrow BOUND to the two shapes carrying those labels -- an arrow that merely ends
                 near a box does not count, nor does a label floating outside its box.

    computeruse agent "Draw ..." --site excalidraw --check excalidraw \\
        --expect "Brain,Router,Eyes" --connect "Brain>Router,Router>Eyes"
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

EXCALIDRAW_SCENE = "JSON.parse(localStorage.getItem('excalidraw') || '[]')"


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def excalidraw_verdict(elements: list[dict[str, Any]], labels: list[str],
                       connect: list[tuple[str, str]]) -> tuple[bool, str]:
    live = [e for e in elements if not e.get("isDeleted")]
    by_id = {e["id"]: e for e in live if "id" in e}
    texts = [e for e in live if e.get("type") == "text"]

    def label_of(eid: str) -> str:
        """A shape's label is its bound text; a bare text element is its own label."""
        e = by_id.get(eid)
        if e is None:
            return ""
        if e.get("type") == "text":
            return _norm(e.get("text", ""))
        return " ".join(_norm(t.get("text", "")) for t in texts if t.get("containerId") == eid)

    written = [_norm(t.get("text", "")) for t in texts]
    missing = [lab for lab in labels if not any(_norm(lab) in w for w in written)]
    links = []
    for a in live:
        if a.get("type") != "arrow":
            continue
        s, e = (a.get("startBinding") or {}).get("elementId"), (a.get("endBinding") or {}).get("elementId")
        if s and e and s != e and s in by_id and e in by_id:
            links.append((label_of(s), label_of(e)))
    unlinked = [f"{x}>{y}" for x, y in connect
                if not any((_norm(x) in s and _norm(y) in e) or (_norm(y) in s and _norm(x) in e) for s, e in links)]
    shapes = sum(1 for e in live if e.get("type") in ("rectangle", "ellipse", "diamond"))
    arrows = sum(1 for e in live if e.get("type") == "arrow")
    detail = (f"{shapes} shapes, {arrows} arrows ({len(links)} bound at both ends), labels {[_norm(w) for w in written]}"
              + (f"; missing labels {missing}" if missing else "") + (f"; not connected {unlinked}" if unlinked else ""))
    return not missing and not unlinked, detail


def excalidraw_checker(page: Callable[[], Any], labels: list[str], connect: list[tuple[str, str]]):
    def check() -> tuple[bool, str]:
        time.sleep(1.0)  # its autosave is debounced: let the last edit land
        return excalidraw_verdict(page().evaluate(EXCALIDRAW_SCENE), labels, connect)
    return check


def excalidraw_fresh(page) -> None:
    """Start from an empty board (the agent's profile keeps the last drawing)."""
    page.evaluate("localStorage.removeItem('excalidraw')")
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(1500)


def parse_connect(spec: str) -> list[tuple[str, str]]:
    pairs = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, sep, b = part.partition(">")
        if not sep or not a.strip() or not b.strip():
            raise ValueError(f"--connect wants A>B pairs, got {part!r}")
        pairs.append((a.strip(), b.strip()))
    return pairs
