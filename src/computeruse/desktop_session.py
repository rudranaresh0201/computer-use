"""
The desktop as a set of tools a reasoning model can call: open an app, observe it, act on it,
and prove a checkpoint happened. `mcp_server.py` exposes these over MCP; this module is plain
Python so every rule here is tested without a model, a mouse or a screen.

How a target is found -- the research result, enforced here, not left to the prompt:

    keys / typing into the focused app      channel "api"     no pixels, no guessing
    an element from the tree, BY NAME       channel "tree"    exact, cheap
    x, y read off the screenshot            channel "vision"  the model's own eyes
    (locate(), our trained grounder, joins as channel "eyes" in a later piece)

Acting on a tree element that has no name is refused: the tree knows a button exists but not
which one it is, and "tree first" with nameless elements is exactly the naive hybrid that lost
to vision alone in Tests A and B. The model is told to use coordinates instead.

A checkpoint is ticked only by `check`, and only when a verifier finds the evidence (a file on
disk, text in the window, a changed screen). The UI hears about steps, receipts and screens
through `events` (a JSON-lines file the sidecar tails), so the model's tool results stay plain.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .action.schema import Action, ActionType

ACTIONS = ("click", "double_click", "right_click", "drag", "type", "key", "scroll", "wait")
SEARCH_DIRS = ("Pictures", "Documents", "Desktop", "Downloads")
IMAGE_EDGE = 1280  # long edge of the screenshot the model sees


@dataclass
class Shot:
    """One screenshot: what the model saw, and how to map its pixels back to the screen."""

    jpeg: bytes
    size: tuple[int, int]  # the image the model sees
    to_real: Callable[[int, int], tuple[int, int]]
    small: bytes = b""  # grayscale thumbnail, for "did the screen change"


@dataclass
class Step:
    kind: str
    label: str
    channel: str
    point: Optional[tuple[float, float]] = None  # where it pointed, as fractions of the screen


@dataclass
class DesktopSession:
    apps: dict[str, tuple[list[str], str]]  # name -> (launch command, a word its window title contains)
    tree: Callable[[], list] = None  # -> [UIElement] of the foreground window
    shoot: Callable[[], Shot] = None
    execute: Callable[[Action], Any] = None  # -> ActionResult
    foreground: Callable[[], str] = None  # -> foreground window title
    focus: Callable[[str], bool] = None  # bring a window whose title contains this word to the front
    launch: Callable[[str], str] = None  # app name -> its window title word, once it is open
    events: Callable[[str, dict], None] = lambda method, params: None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    home: Path = field(default_factory=Path.home)

    app: Optional[str] = None
    window: Optional[str] = None  # title word of the app being driven
    plan: list[str] = field(default_factory=list)
    done: set[int] = field(default_factory=set)
    elements: dict[str, Any] = field(default_factory=dict)
    shot: Optional[Shot] = None
    steps: int = 0

    # --- tools ---------------------------------------------------------------

    def targets(self) -> str:
        return "Apps you can drive: " + ", ".join(sorted(self.apps))

    def open(self, app: str) -> str:
        if app not in self.apps:
            return f"Unknown app {app!r}. Choose one of: {', '.join(sorted(self.apps))}."
        self.window = self.launch(app)
        self.app = app
        self.events("started", {"app": app, "goal": ""})
        return f"{app} is open and in front (window title contains {self.window!r}). Observe it next."

    def set_plan(self, steps: list[str]) -> str:
        self.plan = [str(s).strip() for s in steps if str(s).strip()][:12]
        self.done = set()
        self.events("plan", {"app": self.app, "goal": "", "steps": self.plan})
        return "Plan recorded:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.plan, 1))

    def observe(self) -> tuple[str, bytes]:
        problem = self._front()
        if problem:
            return problem, b""
        from .brain.llm_desktop import render_elements

        listing, self.elements = render_elements(self.tree())
        self.shot = self.shoot()
        w, h = self.shot.size
        self.events("screen", {"src": "data:image/jpeg;base64," + base64.b64encode(self.shot.jpeg).decode()})
        text = (f"Window: {self.foreground()}\nScreenshot: {w}x{h} pixels; x, y for act() are in these pixels.\n"
                f"Elements (act on them by id; only named ones):\n{listing}")
        return text, self.shot.jpeg

    def act(self, action: str, element: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None,
            to_x: Optional[int] = None, to_y: Optional[int] = None, text: Optional[str] = None,
            keys: Optional[str] = None, direction: str = "down", amount: int = 3, seconds: float = 1.0) -> str:
        if action not in ACTIONS:
            return f"Unknown action {action!r}. Use one of: {', '.join(ACTIONS)}."
        if action == "wait":
            self.sleep(max(0.0, min(float(seconds), 10.0)))
            return f"Waited {seconds:g}s. Observe to see the screen now."
        problem = self._front()
        if problem:
            return problem
        if self.shot is None:
            return "Observe first: actions refer to the latest observation."

        try:
            plan = self._resolve(action, element, x, y, to_x, to_y, text, keys, direction, amount)
        except _Refused as e:
            return str(e)
        before = self.shoot().small
        result = self.execute(plan.action)
        if not getattr(result, "success", False):
            return f"{plan.step.label}: failed ({getattr(result, 'error', 'unknown error')})."
        self.sleep(0.6)
        changed = _differs(before, self.shoot().small)
        self.steps += 1
        params = {"n": self.steps, "kind": plan.step.kind, "label": plan.step.label, "note": "",
                  "channel": plan.step.channel}
        if plan.step.point:
            params["x"], params["y"] = plan.step.point
        self.events("step", params)
        return (f"{plan.step.label} [{plan.step.channel}]. The screen {'changed' if changed else 'did NOT change'}."
                + ("" if changed else " If that was unexpected, observe and try another way."))

    def check(self, step: int, kind: str, value: str = "", min_bytes: int = 1) -> str:
        if not 1 <= step <= len(self.plan):
            return f"Step {step} is not in the plan (1..{len(self.plan)}). Record a plan first."
        verdict, summary = self._verify(kind, value, min_bytes)
        self.events("receipt", {"step": step, "verdict": verdict, "summary": summary})
        if verdict != "verified":
            return f"NOT verified: {summary}. The step is not done; fix it and check again."
        self.done.add(step)
        self.events("check", {"step": step})
        left = [i for i in range(1, len(self.plan) + 1) if i not in self.done]
        return f"Step {step} verified: {summary}." + (f" Remaining steps: {left}." if left else " Every step is verified.")

    # --- internals --------------------------------------------------------------

    def _front(self) -> Optional[str]:
        if not self.window:
            return "No app is open yet: call open(app) first."
        if self.window.lower() in self.foreground().lower():
            return None
        if self.focus(self.window) and self.window.lower() in self.foreground().lower():
            return None
        return f"{self.window!r} is not in front ({self.foreground()!r} is), and it could not be brought back. Stopping is safer."

    def _resolve(self, action, element, x, y, to_x, to_y, text, keys, direction, amount) -> "_Plan":
        if action == "key":
            if not keys:
                raise _Refused("key needs keys, e.g. keys='ctrl+s'.")
            return _Plan(Action(ActionType.KEY, text=keys), Step("key", f"key {keys}", "api"))
        if action == "type" and element is None:
            if not text:
                raise _Refused("type needs text.")
            _ascii(text)
            return _Plan(Action(ActionType.TYPE, text=text), Step("type", f"type {text[:40]!r}", "api"))
        if action == "scroll":
            target = self._point(element, x, y, required=False)
            act = Action(ActionType.SCROLL, coordinate=target[0] if target else None,
                         scroll_direction=direction, scroll_amount=int(amount))
            return _Plan(act, Step("scroll", f"scroll {direction}", target[1] if target else "api",
                                   target[2] if target else None))
        if action == "drag":
            if None in (x, y, to_x, to_y):
                raise _Refused("drag needs x, y and to_x, to_y (screenshot pixels).")
            start, _, _ = self._point(None, x, y)
            end, _, mid = self._point(None, to_x, to_y)
            frac = self._frac((x + to_x) / 2, (y + to_y) / 2)
            return _Plan(Action(ActionType.LEFT_CLICK_DRAG, start_coordinate=start, coordinate=end),
                         Step("drag", f"drag ({x}, {y}) -> ({to_x}, {to_y})", "vision", frac))
        where, channel, frac = self._point(element, x, y)
        label = self._describe(element, x, y)
        if action == "type":
            _ascii(text or "")
            # click into the field first, then type: one step for the person watching
            self.execute(Action(ActionType.LEFT_CLICK, coordinate=where))
            return _Plan(Action(ActionType.TYPE, text=text or ""), Step("type", f"type {text[:40]!r} into {label}", channel, frac))
        kind = {"click": ActionType.LEFT_CLICK, "double_click": ActionType.DOUBLE_CLICK, "right_click": ActionType.RIGHT_CLICK}[action]
        return _Plan(Action(kind, coordinate=where), Step("click", f"{action.replace('_', ' ')} {label}", channel, frac))

    def _point(self, element, x, y, required: bool = True):
        if element is not None:
            e = self.elements.get(element)
            if e is None:
                raise _Refused(f"No element {element!r} in the latest observation. Observe again.")
            if not (getattr(e, "name", "") or "").strip():
                raise _Refused(f"{element} has no name: the tree can't say which element it is. "
                               "Find it on the screenshot and act with x, y instead.")
            cx, cy = e.center
            return (cx, cy), "tree", self._frac_real(cx, cy)
        if x is None or y is None:
            if required:
                raise _Refused("Give an element id or x, y (screenshot pixels).")
            return None
        w, h = self.shot.size
        if not (0 <= x < w and 0 <= y < h):
            raise _Refused(f"({x}, {y}) is outside the {w}x{h} screenshot.")
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
        # where a tree element sits on the screenshot: invert to_real on the image corners
        w, h = self.shot.size
        x1, y1 = self.shot.to_real(w - 1, h - 1)
        x0, y0 = self.shot.to_real(0, 0)
        if x1 == x0 or y1 == y0:
            return None
        return round((rx - x0) / (x1 - x0), 4), round((ry - y0) / (y1 - y0), 4)

    def _verify(self, kind: str, value: str, min_bytes: int) -> tuple[str, str]:
        if kind == "file":
            path = self._find_file(value)
            if path is None:
                return "contradicted", f"no file named {value!r} in {', '.join(SEARCH_DIRS)}"
            size, age = path.stat().st_size, self.clock() - path.stat().st_mtime
            if size < min_bytes:
                return "contradicted", f"{path.name} exists but is {size} bytes"
            if age > 900:
                return "contradicted", f"{path.name} exists but was last saved {age / 60:.0f} min ago, before this task"
            return "verified", f"{path.name} exists, {size / 1024:.0f} KB, saved {age:.0f}s ago"
        if kind == "text":
            names = [getattr(e, "name", "") or "" for e in self.tree()] + [self.foreground()]
            hit = next((n for n in names if value and value.lower() in n.lower()), None)
            if hit is None:
                return "contradicted", f"{value!r} is not in the window"
            return "verified", f"the window shows {hit.strip()[:60]!r}"
        if kind == "changed":
            if self.shot is None:
                return "contradicted", "nothing observed yet"
            if _differs(self.shot.small, self.shoot().small):
                return "verified", "the screen changed since the last observation"
            return "contradicted", "the screen looks the same as at the last observation"
        return "contradicted", f"unknown check kind {kind!r} (use file, text or changed)"

    def _find_file(self, value: str) -> Optional[Path]:
        p = Path(os.path.expandvars(value)).expanduser()
        if p.is_absolute():
            return p if p.is_file() else None
        for d in SEARCH_DIRS:
            for root in (self.home / d, self.home / "OneDrive" / d):
                if (root / p).is_file():
                    return root / p
        return None


@dataclass
class _Plan:
    action: Action
    step: Step


class _Refused(Exception):
    pass


def _ascii(text: str) -> None:
    if any(ord(c) > 126 for c in text):
        raise _Refused("type can only send plain ASCII text for now.")


def _differs(a: bytes, b: bytes, threshold: float = 1.5) -> bool:
    """Mean absolute difference of two same-size grayscale thumbnails, in grey levels."""
    if not a or not b or len(a) != len(b):
        return a != b
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) > threshold


# --- the real machine -----------------------------------------------------------------

def real_session(events: Callable[[str, dict], None]) -> DesktopSession:
    """A session wired to this Windows machine: UIA, mss screenshots, pyautogui."""
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

    return DesktopSession(apps=APPS, tree=get_foreground_window_tree, shoot=shoot, execute=controller.execute,
                          foreground=foreground_title, focus=focus, launch=launch, events=events)


def file_events(path: Optional[str]) -> Callable[[str, dict], None]:
    """Events for the window, appended to a JSON-lines file the sidecar tails. No path: dropped."""
    if not path:
        return lambda method, params: None

    def emit(method: str, params: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": method, "params": params}) + "\n")

    return emit
