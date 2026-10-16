"""
The web surface: websites in the agent's Chrome, through Playwright. Same rules as the desktop
(session.py), different eyes and hands:

  eyes   Chrome's accessibility tree (web/browser.py's AI-mode snapshot, refs like e7) and a
         screenshot of the page's viewport in CSS pixels;
  hands  the page's own input (page.mouse / page.keyboard / aria-ref locators) -- not the OS mouse,
         so the agent's Chrome can sit behind other windows and the person's cursor is left alone.

Screenshot pixels ARE page CSS pixels here (scale="css"), so x, y need no mapping. Canvas-drawn
apps (Sheets' grid, Docs' page, Tinkercad's workplane) are where the tree goes blind and x, y
-- the vision channel, later the trained grounder -- does the work.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from .session import THUMB, Move, Refused, Shot, Step, differs

SITES = {
    # name: (where it starts, what the model is told it is for)
    "sheets": ("https://docs.google.com/spreadsheets/u/0/",
               "Google Sheets: spreadsheets, formulas, charts (the grid is drawn on a canvas: cells by x, y or keys)"),
    "docs": ("https://docs.google.com/document/u/0/", "Google Docs: write and format documents"),
    "gmail": ("https://mail.google.com/mail/u/0/#inbox",
              "Gmail: read mail and write drafts; sending waits for the person's approval"),
    "tinkercad": ("https://www.tinkercad.com/dashboard",
                  "Tinkercad: 3D design in the browser (the workplane is WebGL: shapes by drag and keys)"),
    "excalidraw": ("https://excalidraw.com/", "Excalidraw: whiteboard diagrams (the canvas is invisible to the tree)"),
}
SIGN_IN = ("https://accounts.google.com/", "https://www.tinkercad.com/login")

KEY_NAMES = {
    "ctrl": "Control", "control": "Control", "alt": "Alt", "shift": "Shift", "win": "Meta", "cmd": "Meta",
    "meta": "Meta", "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab",
    "space": "Space", "backspace": "Backspace", "delete": "Delete", "del": "Delete", "home": "Home", "end": "End",
    "pageup": "PageUp", "pagedown": "PageDown", "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft",
    "right": "ArrowRight",
}


def playwright_keys(keys: str) -> str:
    """'ctrl+shift+s' -> 'Control+Shift+s'; 'f2' -> 'F2'; letters keep their case."""
    parts = []
    for k in keys.replace(" ", "").split("+"):
        low = k.lower()
        parts.append(KEY_NAMES.get(low) or (low.upper() if re.fullmatch(r"f\d{1,2}", low) else k))
    return "+".join(parts)


@dataclass
class WebSurface:
    sites: dict[str, tuple[str, str]]  # name -> (start URL, one-line description)
    page: Callable[[], Any]  # -> the Playwright Page the agent works in
    settle: Callable[[], None] = lambda: None  # wait after an action (tests pass a no-op)

    kind: str = "web"
    elements: dict[str, Any] = field(default_factory=dict)
    shot: Optional[Shot] = None
    viewport: tuple[int, int, int, int] = (0, 0, 1280, 800)  # scrollX, scrollY, width, height

    @property
    def observed(self) -> bool:
        return self.shot is not None

    def describe(self, target: str) -> str:
        return self.sites[target][1]

    def open(self, target: str) -> str:
        url = self.sites[target][0]
        page = self.page()
        here = urlparse(page.url).netloc
        if not here or here != urlparse(url).netloc:  # already on the site (signed in, mid-task): stay
            page.goto(url, wait_until="domcontentloaded")
        page.bring_to_front()
        self.shot = None
        return f"{target} is open in the agent's Chrome ({page.url}). Observe it next."

    def ready(self) -> Optional[str]:
        try:
            page = self.page()
            return "The agent's Chrome tab is closed. Open the target again." if page.is_closed() else None
        except Exception as e:  # noqa: BLE001 -- the browser went away: say so, don't crash the server
            return f"The agent's Chrome is not reachable ({type(e).__name__}). Open the target again."

    def observe(self) -> tuple[str, Shot]:
        from .web.agent import render_observation
        from .web.browser import PageView, parse_snapshot

        page = self.page()
        snapshot = page.locator("body").aria_snapshot(mode="ai", boxes=True)
        view = PageView(url=page.url, title=page.title(), snapshot=snapshot, elements=parse_snapshot(snapshot))
        self.elements = {e.ref: e for e in view.elements}
        sx, sy, w, h = (int(v) for v in page.evaluate("[scrollX, scrollY, innerWidth, innerHeight]"))
        self.viewport = (sx, sy, w, h)
        self.shot = self._shoot(page, (w, h))
        tree = render_observation(view, self.viewport)
        text = (f"{tree}\n\nScreenshot: {w}x{h} pixels = the page's visible area; x, y for act() are in these pixels.\n"
                "Act on elements by ref (e7), only named ones.")
        return text, self.shot

    def state(self) -> tuple[str, bytes]:
        """The page as a before/after comparison sees it: URL + tree (text changes show up here even
        when they are a few pixels) and a thumbnail (canvas drawing shows up only here)."""
        page = self.page()
        tree = page.locator("body").aria_snapshot(mode="ai").replace(" [active]", "")
        _, _, w, h = (int(v) for v in page.evaluate("[scrollX, scrollY, innerWidth, innerHeight]"))
        return f"{page.url} | {tree}", self._shoot(page, (w, h)).small

    def changed(self, before: tuple[str, bytes], after: tuple[str, bytes]) -> bool:
        return before[0] != after[0] or differs(before[1], after[1])

    def verify(self, kind: str, value: str) -> Optional[tuple[str, str]]:
        page = self.page()
        if kind == "url":
            if value and value.lower() in page.url.lower():
                return "verified", f"the page is at {page.url[:80]}"
            return "contradicted", f"the page is at {page.url[:80]}, not {value!r}"
        if kind == "text":
            haystack = page.locator("body").aria_snapshot(mode="ai") + "\n" + page.title()
            if value and value.lower() in haystack.lower():
                return "verified", f"the page shows {value[:60]!r}"
            return "contradicted", f"{value!r} is not on the page (as the accessibility tree reads it)"
        return None

    def resolve(self, action, element, x, y, to_x, to_y, text, keys, direction, amount) -> Move:
        page = self.page()
        if action == "key":
            if not keys:
                raise Refused("key needs keys, e.g. keys='ctrl+s'.")
            combo = playwright_keys(keys)
            return Move(self._run(lambda: page.keyboard.press(combo)), Step("key", f"key {keys}", "api"))
        if action == "type" and element is None:
            if not text:
                raise Refused("type needs text.")
            return Move(self._run(lambda: page.keyboard.type(text, delay=15)), Step("type", f"type {text[:40]!r}", "api"))
        if action == "scroll":
            dy = (1 if direction == "down" else -1) * int(amount) * 120
            point = self._xy(x, y) if x is not None and y is not None else None

            def scroll():
                if point:
                    page.mouse.move(*point)
                page.mouse.wheel(0, dy)

            return Move(self._run(scroll), Step("scroll", f"scroll {direction}", "vision" if point else "api",
                                                self._frac(*point) if point else None))
        if action == "drag":
            if None in (x, y, to_x, to_y):
                raise Refused("drag needs x, y and to_x, to_y (screenshot pixels).")
            a, b = self._xy(x, y), self._xy(to_x, to_y)

            def drag():
                page.mouse.move(*a)
                page.mouse.down()
                page.mouse.move(*b, steps=12)
                page.mouse.up()

            return Move(self._run(drag), Step("drag", f"drag ({x}, {y}) -> ({to_x}, {to_y})", "vision",
                                              self._frac((x + to_x) / 2, (y + to_y) / 2)))
        clicks = {"click": 1, "double_click": 2, "right_click": 1}[action] if action != "type" else 1
        button = "right" if action == "right_click" else "left"
        if element is not None:
            e = self._named(element)
            locator = page.locator(f"aria-ref={e.ref}")
            frac = self._box_frac(e.box)
            label = f'{e.ref} {e.role} "{e.name}"'
            if action == "type":
                def type_into():
                    locator.click(timeout=5000)
                    page.keyboard.type(text or "", delay=15)

                return Move(self._run(type_into), Step("type", f"type {(text or '')[:40]!r} into {label}", "tree", frac))
            return Move(self._run(lambda: locator.click(button=button, click_count=clicks, timeout=5000)),
                        Step("click", f"{action.replace('_', ' ')} {label}", "tree", frac))
        if x is None or y is None:
            raise Refused("Give an element ref or x, y (screenshot pixels).")
        px, py = self._xy(x, y)
        if action == "type":
            def click_type():
                page.mouse.click(px, py)
                page.keyboard.type(text or "", delay=15)

            return Move(self._run(click_type), Step("type", f"type {(text or '')[:40]!r} at ({x}, {y})", "vision", self._frac(x, y)))
        return Move(self._run(lambda: page.mouse.click(px, py, button=button, click_count=clicks)),
                    Step("click", f"{action.replace('_', ' ')} ({x}, {y})", "vision", self._frac(x, y)))

    # --- internals --------------------------------------------------------------

    def _run(self, do: Callable[[], None]) -> Callable[[], tuple[bool, str]]:
        def run() -> tuple[bool, str]:
            try:
                do()
            except Exception as e:  # noqa: BLE001 -- Playwright errors become a reason the model reads
                return False, (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
            self.settle()
            return True, ""

        return run

    def _named(self, ref: str):
        e = self.elements.get(ref)
        if e is None:
            raise Refused(f"No element {ref!r} in the latest observation. Observe again.")
        if not e.name.strip():
            raise Refused(f"{ref} has no name: the tree can't say which element it is. "
                          "Find it on the screenshot and act with x, y instead.")
        return e

    def _xy(self, x, y) -> tuple[int, int]:
        w, h = self.shot.size
        if not (0 <= x < w and 0 <= y < h):
            raise Refused(f"({x}, {y}) is outside the {w}x{h} screenshot.")
        return int(x), int(y)

    def _frac(self, x, y) -> tuple[float, float]:
        w, h = self.shot.size
        return round(x / w, 4), round(y / h, 4)

    def _box_frac(self, box) -> Optional[tuple[float, float]]:
        if not box:
            return None
        sx, sy, w, h = self.viewport
        bx, by, bw, bh = box
        return round((bx - sx + bw / 2) / w, 4), round((by - sy + bh / 2) / h, 4)

    @staticmethod
    def _shoot(page, size: tuple[int, int]) -> Shot:
        from PIL import Image

        jpeg = page.screenshot(type="jpeg", quality=78, scale="css")
        small = Image.open(io.BytesIO(jpeg)).convert("L").resize(THUMB).tobytes()
        return Shot(jpeg, size, lambda x, y: (x, y), small)
