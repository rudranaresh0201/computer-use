"""
The hero tasks, each with an INDEPENDENT checker: it does not read the agent's own check()
receipts, it looks at the result itself -- the PNG's pixels, the Sheet's CSV export, the STL file.
That is what makes `computeruse bench-task <task> --runs 10` an honest pass rate (the
reliability paper, arxiv 2604.17849: 78% pass@10 can be 36% pass-all-10, so we report both).

Every run gets a tag (e.g. "7f2a") written into its file or document name, so a leftover from an
earlier run can never pass for this one.
"""

from __future__ import annotations

import csv
import io
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

Verdict = tuple[bool, str]


@dataclass
class Task:
    name: str
    target: str
    goal: Callable[[str], str]  # tag -> the natural-language command
    check: Callable[["Context"], Verdict]


@dataclass
class Context:
    tag: str
    started: float  # time.time() when the run began: anything older is not this run's
    home: Path
    pages: Callable[[], list[Any]] = lambda: []  # the agent Chrome's open pages (for web checks)
    fetch: Callable[[str], str] = lambda url: ""  # GET with the agent's sign-in (Docs/Sheets export)


# --- pure checks (tested without apps) ---------------------------------------------------------

def find_fresh(home: Path, name: str, started: float, folders=("Pictures", "Downloads", "Documents", "Desktop")
               ) -> Optional[Path]:
    """A file called `name` (case-insensitive) saved after `started`, in the usual folders."""
    want = name.lower()
    for folder in folders:
        for root in (home / folder, home / "OneDrive" / folder):
            if not root.is_dir():
                continue
            for p in root.iterdir():
                if p.is_file() and p.name.lower() == want and p.stat().st_mtime >= started - 1:
                    return p
    return None


def drawing_verdict(png: bytes, min_ink: float = 0.03, min_colors: int = 3) -> Verdict:
    """A house with a sun and a tree: enough of the canvas is drawn on, in several colours.
    Cannot tell a house from a scribble -- the receipts and the recording do that -- but it does
    catch the real failures: an empty canvas, one stray line, a file that is not an image."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(png)).convert("RGB")
    except Exception as e:  # noqa: BLE001 -- any decode failure is a failed task
        return False, f"not a readable image ({e.__class__.__name__})"
    small = image.resize((200, 150))
    pixels = list(small.getdata())
    ink = [p for p in pixels if min(p) < 235]  # not white
    share = len(ink) / len(pixels)
    # colours that cover at least 0.5% of the canvas, bucketed so anti-aliasing is one colour
    buckets: dict[tuple[int, int, int], int] = {}
    for r, g, b in ink:
        key = (r // 64, g // 64, b // 64)
        buckets[key] = buckets.get(key, 0) + 1
    colors = sum(1 for n in buckets.values() if n >= 0.005 * len(pixels))
    detail = f"{share:.0%} of the canvas drawn on, {colors} colours"
    return share >= min_ink and colors >= min_colors, detail


def sheet_verdict(csv_text: str, min_items: int = 4) -> Verdict:
    """An expense table: at least `min_items` rows of (item, amount), and a total row whose number
    equals the sum of the amounts above it."""
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(csv_text)) if any(c.strip() for c in r)]
    amounts, total = [], None
    for r in rows:
        nums = [_number(c) for c in r[1:] if _number(c) is not None]
        if not nums:
            continue
        if re.search(r"total", " ".join(r), re.IGNORECASE):
            total = nums[-1]
        else:
            amounts.append(nums[0])
    if len(amounts) < min_items:
        return False, f"{len(amounts)} expense rows (need {min_items})"
    if total is None:
        return False, f"{len(amounts)} expense rows but no Total row"
    if abs(total - sum(amounts)) > 0.01:
        return False, f"Total says {total:g} but the rows add up to {sum(amounts):g}"
    return True, f"{len(amounts)} expenses, Total {total:g} matches"


def _number(cell: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.\-]", "", cell.replace(",", ""))
    try:
        return float(cleaned) if cleaned not in ("", "-", ".") else None
    except ValueError:
        return None


def doc_verdict(text: str, must_mention: str = "total", min_words: int = 80) -> Verdict:
    words = len(text.split())
    if words < min_words:
        return False, f"{words} words (need {min_words})"
    if must_mention.lower() not in text.lower():
        return False, f"{words} words, but it never mentions {must_mention!r}"
    return True, f"{words} words, mentions {must_mention!r}"


def stl_verdict(data: bytes) -> Verdict:
    """A real mesh: ASCII STL with facets, or binary STL whose triangle count fits its size."""
    if data[:5].lower() == b"solid" and b"facet" in data[:4096]:
        return True, f"ASCII STL, {data.count(b'facet normal')} triangles"
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if n > 0 and len(data) == 84 + 50 * n:
            return True, f"binary STL, {n} triangles"
    return False, f"not an STL mesh ({len(data)} bytes)"


# --- the tasks --------------------------------------------------------------------------------

def _paint(ctx: Context) -> Verdict:
    path = find_fresh(ctx.home, f"house_{ctx.tag}.png", ctx.started)
    if path is None:
        return False, f"no house_{ctx.tag}.png saved during the run"
    ok, detail = drawing_verdict(path.read_bytes())
    return ok, f"{path.name}: {detail}"


def _google_file(ctx: Context, kind: str, fmt: str) -> tuple[Optional[str], str]:
    """The export text of the Doc/Sheet whose title carries this run's tag, from the agent's tabs."""
    for page in ctx.pages():
        m = re.search(rf"docs\.google\.com/{kind}/d/([\w-]+)", page.url)
        if m and ctx.tag in page.title():
            return ctx.fetch(f"https://docs.google.com/{kind}/d/{m.group(1)}/export?format={fmt}"), page.title()
    return None, ""


def _sheets(ctx: Context) -> Verdict:
    text, title = _google_file(ctx, "spreadsheets", "csv")
    if text is None:
        return False, f"no open Sheet titled with {ctx.tag!r}"
    ok, detail = sheet_verdict(text)
    return ok, f"{title}: {detail}"


def _docs(ctx: Context) -> Verdict:
    text, title = _google_file(ctx, "document", "txt")
    if text is None:
        return False, f"no open Doc titled with {ctx.tag!r}"
    ok, detail = doc_verdict(text)
    return ok, f"{title}: {detail}"


def _tinkercad(ctx: Context) -> Verdict:
    folder = ctx.home / "Downloads"
    fresh = [p for p in folder.glob("*.stl") if p.stat().st_mtime >= ctx.started - 1] if folder.is_dir() else []
    if not fresh:
        return False, "no .stl downloaded during the run"
    path = max(fresh, key=lambda p: p.stat().st_mtime)
    ok, detail = stl_verdict(path.read_bytes())
    return ok, f"{path.name}: {detail}"


TASKS = {t.name: t for t in (
    Task("paint", "paint", lambda tag: (
        f"In Paint, draw a simple house with a roof, a sun and a tree, in colour. "
        f"Save it as house_{tag}.png in Pictures."), _paint),
    Task("sheets", "sheets", lambda tag: (
        f"Create a new Google Sheet named 'Week expenses {tag}'. Put 5 expenses from a student's week "
        f"in columns Item and Amount, then a Total row that adds them with a SUM formula."), _sheets),
    Task("docs", "docs", lambda tag: (
        f"Create a new Google Doc titled 'Expense summary {tag}' with a short paragraph (100-150 words) "
        f"summarising a student's week of spending: where the money went and the total."), _docs),
    Task("tinkercad", "tinkercad", lambda tag: (
        "In Tinkercad, create a new 3D design: a flat rounded keychain tag with a hole for the ring, "
        "then export it as STL."), _tinkercad),
)}


def new_tag() -> str:
    import uuid

    return uuid.uuid4().hex[:4]


def now() -> float:
    return time.time()
