"""
Chrome's eyes and hands, through Playwright: the web counterpart of
perception/uia.py (what can I see?) and action/controller.py (do it).

Eyes. Windows exposes each element through UI Automation; a browser exposes
the same thing as its accessibility tree. Playwright's AI-mode snapshot
prints that tree with a reference on every element:

    - textbox "Search products" [ref=e6]
    - button "Search" [ref=e7]

role + accessible name + a handle to act on -- the same triple UIA gives
(control_type, name, rect). Chrome computes the names itself (a <label>
becomes its input's name), which is the point of using the tree rather
than scraping HTML. And like UIA it has blind spots: whatever is drawn
into a <canvas> is simply absent -- the case the grounding model exists for.

Hands. Act on a ref (`aria-ref=e7` is a Playwright locator), never on a
CSS selector the brain would have to invent. Each action reports whether
the page actually changed, because "no exception" and "had the intended
effect" are different claims (journal, 2026-07-10).

Chrome is launched with a fresh, empty profile -- never the user's own,
with its logins and cookies -- so whatever drives it can only reach what
it logs into itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Roles a user (or a brain) can act on. Everything else in the tree is
# context: headings, text, regions.
INTERACTIVE_ROLES = frozenset(
    {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio",
        "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "option", "switch",
        "slider", "spinbutton", "treeitem",
    }
)
TYPEABLE_ROLES = frozenset({"textbox", "searchbox", "combobox", "spinbutton"})

# One snapshot line: "- role "name" [attr] [ref=e7] [box=x,y,w,h]: text"
_LINE_RE = re.compile(r'^\s*-\s+(?P<role>[a-z]+)(?:\s+"(?P<name>(?:[^"\\]|\\.)*)")?(?P<attrs>.*)$')
# After a navigation Playwright prefixes refs with a frame id: e7, then f1e321.
_REF_PATTERN = r"(?:f\d+)?e\d+"
_REF_RE = re.compile(rf"\[ref=({_REF_PATTERN})\]")
_BOX_RE = re.compile(r"\[box=(-?\d+),(-?\d+),(\d+),(\d+)\]")

ACTION_TIMEOUT_MS = 5_000  # a stale or covered ref fails fast instead of hanging the loop


@dataclass(frozen=True)
class WebElement:
    ref: str
    role: str
    name: str
    box: Optional[tuple[int, int, int, int]] = None  # x, y, width, height in page pixels

    @property
    def interactive(self) -> bool:
        return self.role in INTERACTIVE_ROLES

    def describe(self) -> str:
        return f'{self.ref:<5} {self.role:<10} "{self.name}"' if self.name else f"{self.ref:<5} {self.role}"


@dataclass
class PageView:
    """What the browser shows right now, as text a person or a model can read."""

    url: str
    title: str
    snapshot: str
    elements: list[WebElement] = field(default_factory=list)

    def interactive(self) -> list[WebElement]:
        return [e for e in self.elements if e.interactive]

    def find(self, ref: str) -> Optional[WebElement]:
        return next((e for e in self.elements if e.ref == ref), None)


@dataclass
class ActResult:
    ok: bool
    detail: str = ""
    page_changed: bool = False


def parse_snapshot(snapshot: str) -> list[WebElement]:
    """Every element in an AI-mode snapshot that carries a ref."""
    elements = []
    for line in snapshot.splitlines():
        match = _LINE_RE.match(line)
        ref = _REF_RE.search(line)
        if match is None or ref is None:
            continue
        box = _BOX_RE.search(line)
        name = (match.group("name") or "").replace('\\"', '"')
        elements.append(
            WebElement(
                ref=ref.group(1),
                role=match.group("role"),
                name=name,
                box=tuple(int(v) for v in box.groups()) if box else None,
            )
        )
    return elements


class Browser:
    """One Chrome window. Use as a context manager:

        with Browser() as web:
            web.goto("https://en.wikipedia.org")
            view = web.observe()
            web.type(view_ref, "Alan Turing", submit=True)
    """

    def __init__(self, headless: bool = False, channel: str = "chrome", viewport: tuple[int, int] = (1280, 800)):
        self._headless = headless
        self._channel = channel
        self._viewport = viewport
        self._playwright = None
        self._browser = None
        self.page = None

    def __enter__(self) -> "Browser":
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(channel=self._channel, headless=self._headless)
        context = self._browser.new_context(viewport={"width": self._viewport[0], "height": self._viewport[1]})
        self.page = context.new_page()
        return self

    def __exit__(self, *exc) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    # -- eyes ---------------------------------------------------------------

    def observe(self, boxes: bool = True) -> PageView:
        snapshot = self.page.locator("body").aria_snapshot(mode="ai", boxes=boxes)
        return PageView(url=self.page.url, title=self.page.title(), snapshot=snapshot, elements=parse_snapshot(snapshot))

    def screenshot(self) -> bytes:
        return self.page.screenshot()

    # -- hands --------------------------------------------------------------

    def goto(self, url: str) -> ActResult:
        if "://" not in url and not url.startswith(("about:", "data:")):
            url = "https://" + url
        return self._act(lambda: self.page.goto(url, wait_until="domcontentloaded"), f"opened {url}")

    def click(self, ref: str) -> ActResult:
        return self._act(lambda: self._locate(ref).click(timeout=ACTION_TIMEOUT_MS), f"clicked {ref}")

    def type(self, ref: str, text: str, submit: bool = False) -> ActResult:
        def run():
            target = self._locate(ref)
            target.fill(text, timeout=ACTION_TIMEOUT_MS)
            if submit:
                target.press("Enter", timeout=ACTION_TIMEOUT_MS)

        return self._act(run, f"typed {text!r} into {ref}" + (" and pressed Enter" if submit else ""))

    def press(self, key: str) -> ActResult:
        return self._act(lambda: self.page.keyboard.press(key), f"pressed {key}")

    def scroll(self, direction: str = "down", pixels: int = 600) -> ActResult:
        dy = {"down": pixels, "up": -pixels}.get(direction)
        if dy is None:
            return ActResult(False, f"scroll direction must be up or down, got {direction!r}")
        return self._act(lambda: self.page.mouse.wheel(0, dy), f"scrolled {direction}")

    def back(self) -> ActResult:
        return self._act(lambda: self.page.go_back(), "went back")

    def _fingerprint(self) -> str:
        """The tree as a before/after comparison should see it: focus moving
        to the clicked button ([active]) is not the page changing."""
        tree = self.page.locator("body").aria_snapshot(mode="ai").replace(" [active]", "")
        return f"{self.page.url}\n{tree}"

    def _locate(self, ref: str):
        if not re.fullmatch(_REF_PATTERN, ref):
            raise ValueError(f"{ref!r} is not a ref -- refs look like e7 or f1e321")
        return self.page.locator(f"aria-ref={ref}")

    def _act(self, do, done_message: str) -> ActResult:
        """Run an action, wait for the page to settle, and report whether
        anything visible to the tree actually changed."""
        from playwright.sync_api import Error as PlaywrightError

        before = self._fingerprint()
        try:
            do()
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=ACTION_TIMEOUT_MS)
            except PlaywrightError:
                pass  # still loading is fine; the snapshot below shows where it got to
        except (PlaywrightError, ValueError) as e:
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            return ActResult(False, first_line)
        changed = self._fingerprint() != before
        return ActResult(True, done_message if changed else done_message + " (the page did not change)", changed)
