"""
Checkers for real apps: each reads the app's own saved state and gives (passed, detail). The brain's
"done" is a claim; this is the verdict.

    excalidraw   the scene excalidraw.com keeps in localStorage["excalidraw"] (its own autosave):
                 every expected label is written somewhere, and every expected connection is an
                 arrow BOUND to the two shapes carrying those labels -- an arrow that merely ends
                 near a box does not count, nor does a label floating outside its box.

    computeruse agent "Draw ..." --site excalidraw --check excalidraw \\
        --expect "Brain,Router,Eyes" --connect "Brain>Router,Router>Eyes"

    tinkercad    the design's real geometry: the checker exports everything in the design as STL
                 (Tinkercad's own Export dialog), splits the mesh into separate solids, and checks
                 that a round solid stands on top of a box: its bottom within 1 mm of the box's top
                 and its centre over the box's footprint. Shape kind comes from the mesh (a box is
                 12 triangles), never from the agent.
"""

from __future__ import annotations

import re
import struct
import time
from collections.abc import Callable
from pathlib import Path
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


def read_stl(data: bytes) -> list[tuple[tuple[float, float, float], ...]]:
    """Triangles of a binary or ASCII STL, each three (x, y, z) vertices."""
    if data[:5].lower() == b"solid" and b"facet" in data[:400]:
        nums = [tuple(map(float, m)) for m in re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)]
        return [tuple(nums[i:i + 3]) for i in range(0, len(nums) - 2, 3)]
    (n,) = struct.unpack_from("<I", data, 80)
    tris = []
    for i in range(n):
        v = struct.unpack_from("<12f", data, 84 + 50 * i)
        tris.append(((v[3], v[4], v[5]), (v[6], v[7], v[8]), (v[9], v[10], v[11])))
    return tris


def solids(tris) -> list[dict[str, Any]]:
    """Split a mesh into connected solids (triangles sharing a vertex); each with its triangle count
    and bounding box. Shapes that merely touch stay separate: they share no vertices."""
    parent = list(range(len(tris)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple, int] = {}
    for t, tri in enumerate(tris):
        for v in tri:
            key = tuple(round(c, 3) for c in v)
            if key in seen:
                a, b = find(seen[key]), find(t)
                if a != b:
                    parent[a] = b
            else:
                seen[key] = t
    groups: dict[int, list] = {}
    for t, tri in enumerate(tris):
        groups.setdefault(find(t), []).append(tri)
    out = []
    for g in groups.values():
        xs, ys, zs = zip(*(v for tri in g for v in tri))
        out.append({"triangles": len(g), "min": (min(xs), min(ys), min(zs)), "max": (max(xs), max(ys), max(zs))})
    return sorted(out, key=lambda s: s["min"][2])


def stacked_verdict(parts: list[dict[str, Any]], tol: float = 1.0) -> tuple[bool, str]:
    """A round solid (more than a box's 12 triangles) standing on a box that stands on the workplane."""
    boxes = [p for p in parts if p["triangles"] == 12 and abs(p["min"][2]) <= tol]
    rounds = [p for p in parts if p["triangles"] > 12]
    desc = "; ".join(f"{p['triangles']} tris, z {p['min'][2]:.1f} to {p['max'][2]:.1f}" for p in parts)
    for b in boxes:
        for r in rounds:
            cx, cy = (r["min"][0] + r["max"][0]) / 2, (r["min"][1] + r["max"][1]) / 2
            if (abs(r["min"][2] - b["max"][2]) <= tol and b["min"][0] <= cx <= b["max"][0]
                    and b["min"][1] <= cy <= b["max"][1]):
                return True, f"{len(parts)} solids, a round one sits on the box: {desc}"
    return False, f"{len(parts)} solids, nothing round stands on a box: {desc or 'empty design'}"


def tinkercad_export_stl(page, folder: Path) -> bytes:
    """Everything in the design, through Tinkercad's own Export dialog."""
    page.keyboard.press("Escape")
    page.get_by_text("Export", exact=True).first.click()
    page.wait_for_timeout(1500)
    everything = page.get_by_text(re.compile(r"Everything in the design", re.I))
    if everything.count():
        everything.first.click()
    with page.expect_download(timeout=60000) as dl:
        page.get_by_text(".STL", exact=True).first.click()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "design.stl"
    dl.value.save_as(str(path))
    page.keyboard.press("Escape")
    return path.read_bytes()


def tinkercad_checker(page: Callable[[], Any], folder: Path = Path("runs/agent/live")):
    def check() -> tuple[bool, str]:
        time.sleep(1.0)
        return stacked_verdict(solids(read_stl(tinkercad_export_stl(page(), folder))))
    return check


def tinkercad_fresh(page) -> None:
    """A new, empty 3D design (it appears in the signed-in account's Designs)."""
    page.goto("https://www.tinkercad.com/dashboard", wait_until="domcontentloaded")
    page.wait_for_timeout(5000)
    page.get_by_role("button", name="Create").first.click()
    page.wait_for_timeout(1500)
    page.get_by_role("menuitem", name=re.compile("3D Design", re.I)).first.click()
    page.wait_for_url(re.compile(r"/things/.+/edit"), timeout=60000)
    page.wait_for_timeout(12000)  # the WebGL editor loads well after the URL changes


def circuit_fresh(page) -> None:
    """A new, empty Tinkercad Circuit (it appears in the signed-in account's Designs)."""
    page.goto("https://www.tinkercad.com/dashboard", wait_until="domcontentloaded")
    page.wait_for_timeout(5000)
    page.get_by_role("button", name="Create").first.click()
    page.wait_for_timeout(1500)
    page.get_by_role("menuitem", name=re.compile("Circuit", re.I)).first.click()
    page.wait_for_url(re.compile(r"/things/.+/editel"), timeout=60000)
    page.wait_for_timeout(10000)
    page.keyboard.press("Escape")  # the "new component" tip


def dismiss_tips(page) -> None:
    """Close Tinkercad's onboarding bubbles ("Discover pre-made circuits...", "NEW micro:bit component")."""
    for text in ("Discover pre-made circuits", "NEW micro:bit component"):
        try:
            tip = page.get_by_text(text).first
            if tip.is_visible():
                tip.locator("xpath=ancestor::*[.//button][1]//button").first.click(timeout=3000)
                page.wait_for_timeout(500)
        except Exception:  # noqa: BLE001 -- a bubble that will not close is the brain's to handle
            pass


def circuit_outro(page) -> None:
    """After the verdict, for the video: the schematic, then the parts list, then back to the board."""
    for view in ("Schematic", "Components", "Circuits"):
        page.get_by_role("button", name=view).first.click()
        page.wait_for_timeout(4000)


def parse_connect(spec: str) -> list[tuple[str, str]]:
    pairs = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, sep, b = part.partition(">")
        if not sep or not a.strip() or not b.strip():
            raise ValueError(f"--connect wants A>B pairs, got {part!r}")
        pairs.append((a.strip(), b.strip()))
    return pairs
