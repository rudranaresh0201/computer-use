"""
The desktop surface: Windows apps through UI Automation (the tree) and the mouse and keyboard.
session.py owns the plan and the proof; this module only answers "what is on screen" and
"how do I do this action here", under the rules the research earned:

  * a tree element is acted on only if it has a NAME (llm_named, Tests A/B) -- else use x, y;
  * x, y are pixels of the screenshot the model saw, mapped back to the real screen
    (the Phase 1 incident: the resized image is not the screen);
  * the app must still be in front, or it is brought back, or nothing happens.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .action.schema import Action, ActionType
from .session import Move, Refused, Shot, Step, ascii_only

IMAGE_EDGE = 1280  # long edge of the screenshot the model sees


@dataclass
class DesktopSurface:
    apps: dict[str, tuple[list[str], str]]  # name -> (launch command, a word its window title contains)
    tree: Callable[[], list] = None  # -> [UIElement] of the foreground window
    shoot: Callable[[], Shot] = None
    execute: Callable[[Action], Any] = None  # -> ActionResult
    foreground: Callable[[], str] = None  # -> foreground window title
    focus: Callable[[str], bool] = None  # bring a window whose title contains this word to the front
    launch: Callable[[str], str] = None  # app name -> its window title word, once it is open
    descriptions: dict[str, str] = field(default_factory=dict)

    kind: str = "desktop"
    window: Optional[str] = None  # title word of the app being driven
    elements: dict[str, Any] = field(default_factory=dict)
    shot: Optional[Shot] = None

    @property
    def observed(self) -> bool:
        return self.shot is not None

    def describe(self, target: str) -> str:
        return self.descriptions.get(target, "Windows app")

    def open(self, target: str) -> str:
        self.window = self.launch(target)
        self.shot = None
        return f"{target} is open and in front (window title contains {self.window!r}). Observe it next."

    def ready(self) -> Optional[str]:
        if self.window.lower() in self.foreground().lower():
            return None
        if self.focus(self.window) and self.window.lower() in self.foreground().lower():
            return None
        return f"{self.window!r} is not in front ({self.foreground()!r} is), and it could not be brought back. Stopping is safer."

    def observe(self) -> tuple[str, Shot]:
        from .brain.llm_desktop import render_elements

        listing, self.elements = render_elements(self.tree())
        self.shot = self.shoot()
        w, h = self.shot.size
        text = (f"Window: {self.foreground()}\nScreenshot: {w}x{h} pixels; x, y for act() are in these pixels.\n"
                f"Elements (act on them by id; only named ones):\n{listing}")
        return text, self.shot

    def small(self) -> bytes:
        return self.shoot().small

    def verify(self, kind: str, value: str) -> Optional[tuple[str, str]]:
        if kind != "text":
            return None
        names = [getattr(e, "name", "") or "" for e in self.tree()] + [self.foreground()]
        hit = next((n for n in names if value and value.lower() in n.lower()), None)
        if hit is None:
            return "contradicted", f"{value!r} is not in the window"
        return "verified", f"the window shows {hit.strip()[:60]!r}"

    def resolve(self, action, element, x, y, to_x, to_y, text, keys, direction, amount) -> Move:
        run = self._runner
        if action == "key":
            if not keys:
                raise Refused("key needs keys, e.g. keys='ctrl+s'.")
            return Move(run(Action(ActionType.KEY, text=keys)), Step("key", f"key {keys}", "api"))
        if action == "type" and element is None:
            if not text:
                raise Refused("type needs text.")
            ascii_only(text)
            return Move(run(Action(ActionType.TYPE, text=text)), Step("type", f"type {text[:40]!r}", "api"))
        if action == "scroll":
            target = self._point(element, x, y, required=False)
            act = Action(ActionType.SCROLL, coordinate=target[0] if target else None,
                         scroll_direction=direction, scroll_amount=int(amount))
            return Move(run(act), Step("scroll", f"scroll {direction}", target[1] if target else "api",
                                       target[2] if target else None))
        if action == "drag":
            if None in (x, y, to_x, to_y):
                raise Refused("drag needs x, y and to_x, to_y (screenshot pixels).")
            start, _, _ = self._point(None, x, y)
            end, _, _ = self._point(None, to_x, to_y)
            return Move(run(Action(ActionType.LEFT_CLICK_DRAG, start_coordinate=start, coordinate=end)),
                        Step("drag", f"drag ({x}, {y}) -> ({to_x}, {to_y})", "vision", self._frac((x + to_x) / 2, (y + to_y) / 2)))
        where, channel, frac = self._point(element, x, y)
        label = self._describe(element, x, y)
        if action == "type":
            ascii_only(text or "")
            # click into the field, then type: one step for the person watching
            return Move(run(Action(ActionType.LEFT_CLICK, coordinate=where), Action(ActionType.TYPE, text=text or "")),
                        Step("type", f"type {(text or '')[:40]!r} into {label}", channel, frac))
        kind = {"click": ActionType.LEFT_CLICK, "double_click": ActionType.DOUBLE_CLICK, "right_click": ActionType.RIGHT_CLICK}[action]
        return Move(run(Action(kind, coordinate=where)), Step("click", f"{action.replace('_', ' ')} {label}", channel, frac))

    # --- internals --------------------------------------------------------------

    def _runner(self, *actions: Action) -> Callable[[], tuple[bool, str]]:
        def run() -> tuple[bool, str]:
            for a in actions:
                result = self.execute(a)
                if not getattr(result, "success", False):
                    return False, getattr(result, "error", "") or "failed"
            return True, ""

        return run

    def _point(self, element, x, y, required: bool = True):
        if element is not None:
            e = self.elements.get(element)
            if e is None:
                raise Refused(f"No element {element!r} in the latest observation. Observe again.")
            if not (getattr(e, "name", "") or "").strip():
                raise Refused(f"{element} has no name: the tree can't say which element it is. "
                              "Find it on the screenshot and act with x, y instead.")
            cx, cy = e.center
            return (cx, cy), "tree", self._frac_real(cx, cy)
        if x is None or y is None:
            if required:
                raise Refused("Give an element id or x, y (screenshot pixels).")
            return None
        w, h = self.shot.size
        if not (0 <= x < w and 0 <= y < h):
            raise Refused(f"({x}, {y}) is outside the {w}x{h} screenshot.")
        return self.shot.to_real(int(x), int(y)), "vision", self._frac(x, y)

    def _describe(self, element, x, y) -> str:
        if element is not None:
            e = self.elements[element]
            return f'{element} {e.control_type} "{e.name}"'
        return f"({x}, {y})"

    def _frac(self, x, y) -> tuple[float, float]:
        w, h = self.shot.size
        return round(x / w, 4), round(y / h, 4)

    def _frac_real(self, rx, ry) -> Optional[tuple[float, float]]:
        # where a tree element sits on the screenshot: invert to_real using the image corners
        w, h = self.shot.size
        x1, y1 = self.shot.to_real(w - 1, h - 1)
        x0, y0 = self.shot.to_real(0, 0)
        if x1 == x0 or y1 == y0:
            return None
        return round((rx - x0) / (x1 - x0), 4), round((ry - y0) / (y1 - y0), 4)


# --- the real machine -----------------------------------------------------------------

DESCRIPTIONS = {
    "calculator": "Windows Calculator: arithmetic, conversions",
    "notepad_pp": "Notepad++: write, edit and save text files",
    "paint": "Microsoft Paint: draw shapes and lines on a canvas, fill colours, save images",
}


def real_desktop() -> DesktopSurface:
    """The desktop surface wired to this Windows machine: UIA, mss screenshots, pyautogui."""
    from PIL import Image

    from .action.controller import ActionController
    from .desktop_run import APPS, foreground_title, open_app
    from .perception.screenshot import capture
    from .perception.uia import get_foreground_window_tree
    from .safety import ActionLogger, SafetyConfig, SafetyGuard

    controller = ActionController(SafetyGuard(SafetyConfig(), ActionLogger(Path("logs/mcp_actions.jsonl"))))

    def shoot() -> Shot:
        s = capture(max_long_edge=IMAGE_EDGE)
        image = Image.open(io.BytesIO(s.png_bytes)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=80)
        small = image.convert("L").resize((96, 54)).tobytes()
        return Shot(buf.getvalue(), s.scaled_size, s.to_real_coords, small)

    def focus(word: str) -> bool:
        from pywinauto import Desktop

        for w in Desktop(backend="uia").windows(visible_only=True):
            if word.lower() in (w.window_text() or "").lower():
                try:
                    w.set_focus()
                    time.sleep(0.4)
                    return True
                except Exception:  # noqa: BLE001
                    return False
        return False

    def launch(app: str) -> str:
        if focus(APPS[app][1]):
            return APPS[app][1]
        return open_app(app)[1]

    return DesktopSurface(apps=APPS, tree=get_foreground_window_tree, shoot=shoot, execute=controller.execute,
                          foreground=foreground_title, focus=focus, launch=launch, descriptions=DESCRIPTIONS)
