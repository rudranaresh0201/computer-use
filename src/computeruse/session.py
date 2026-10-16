"""
One task, across apps and websites: the plan, the actions, and the proof. `mcp_server.py` exposes
a Session over MCP; the Session hands each action to the *surface* that owns the open target:

    DesktopSurface (desktop_session.py)   Windows apps: UI Automation + mouse/keyboard
    WebSurface     (web_surface.py)       websites in the agent's Chrome: accessibility tree + page input

so "fill a Sheet, write a Doc, draft a Gmail" is one plan with one set of receipts.

How a target is found -- the research result, enforced by the surfaces, not left to a prompt:

    keys / typing into the focused field       channel "api"     no pixels, no guessing
    an element from the tree, BY NAME          channel "tree"    exact, cheap
    x, y read off the screenshot               channel "vision"  the model's own eyes
    (locate(), our trained grounder, joins as channel "eyes" in a later piece)

Acting on a tree element that has no name is refused: "tree first" with nameless elements is the
naive hybrid that lost to vision alone in Tests A and B. A plan step is ticked only by `check`, and
only when a verifier finds evidence. The window hears about steps, receipts and screens through
`events` (a JSON-lines file the sidecar tails), so the model's tool results stay plain text.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

ACTIONS = ("click", "double_click", "right_click", "drag", "type", "key", "scroll", "wait")
SEARCH_DIRS = ("Pictures", "Documents", "Desktop", "Downloads")
FRESH_SECONDS = 900  # a file older than this was not saved by this task


@dataclass
class Shot:
    """One screenshot: what the model saw, and how to map its pixels back to where input goes."""

    jpeg: bytes
    size: tuple[int, int]  # the image the model sees
    to_real: Callable[[int, int], tuple[int, int]]
    small: bytes = b""  # grayscale thumbnail, for "did the screen change"


@dataclass
class Step:
    kind: str
    label: str
    channel: str
    point: Optional[tuple[float, float]] = None  # where it pointed, as fractions of the screenshot


@dataclass
class Move:
    """A resolved action: run() does it and returns (ok, error); step is how it is reported."""

    run: Callable[[], tuple[bool, str]]
    step: Step


class Refused(Exception):
    """An action a surface will not take, with the reason the model should read."""


class Surface(Protocol):
    kind: str  # "desktop" | "web"
    observed: bool

    def describe(self, target: str) -> str: ...
    def open(self, target: str) -> str: ...
    def ready(self) -> Optional[str]: ...
    def observe(self) -> tuple[str, Shot]: ...
    def resolve(self, action: str, element: Optional[str], x, y, to_x, to_y, text, keys, direction, amount) -> Move: ...
    def small(self) -> bytes: ...
    def verify(self, kind: str, value: str) -> Optional[tuple[str, str]]: ...


@dataclass
class Session:
    surfaces: dict[str, Any]  # target name -> the surface that drives it (one surface may own many targets)
    events: Callable[[str, dict], None] = lambda method, params: None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    home: Path = field(default_factory=Path.home)

    target: Optional[str] = None
    surface: Any = None
    plan: list[str] = field(default_factory=list)
    done: set[int] = field(default_factory=set)
    baseline: bytes = b""  # the screen when the current plan step started
    steps: int = 0

    # --- tools ---------------------------------------------------------------

    def targets(self) -> str:
        lines = []
        for kind, title in (("desktop", "Desktop apps"), ("web", "Websites (in the agent's Chrome)")):
            names = sorted(t for t, s in self.surfaces.items() if s.kind == kind)
            if names:
                lines.append(f"{title}:\n" + "\n".join(f"- {t}: {self.surfaces[t].describe(t)}" for t in names))
        return "\n".join(lines) or "Nothing to drive."

    def open(self, target: str) -> str:
        surface = self.surfaces.get(target)
        if surface is None:
            return f"Unknown target {target!r}. Choose one of: {', '.join(sorted(self.surfaces))}."
        message = surface.open(target)
        self.target, self.surface = target, surface
        self.events("started", {"app": target})
        return message

    def set_plan(self, steps: list[str]) -> str:
        self.plan = [str(s).strip() for s in steps if str(s).strip()][:12]
        self.done = set()
        self.baseline = self.surface.small() if self.surface else b""
        self.events("plan", {"app": self.target, "steps": self.plan})
        return "Plan recorded:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.plan, 1))

    def observe(self) -> tuple[str, bytes]:
        problem = self._problem()
        if problem:
            return problem, b""
        text, shot = self.surface.observe()
        self.events("screen", {"src": "data:image/jpeg;base64," + base64.b64encode(shot.jpeg).decode()})
        return text, shot.jpeg

    def act(self, action: str, element: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None,
            to_x: Optional[int] = None, to_y: Optional[int] = None, text: Optional[str] = None,
            keys: Optional[str] = None, direction: str = "down", amount: int = 3, seconds: float = 1.0) -> str:
        if action not in ACTIONS:
            return f"Unknown action {action!r}. Use one of: {', '.join(ACTIONS)}."
        if action == "wait":
            self.sleep(max(0.0, min(float(seconds), 10.0)))
            return f"Waited {seconds:g}s. Observe to see the screen now."
        problem = self._problem()
        if problem:
            return problem
        if not self.surface.observed:
            return "Observe first: actions refer to the latest observation."
        try:
            move = self.surface.resolve(action, element, x, y, to_x, to_y, text, keys, direction, amount)
        except Refused as e:
            return str(e)
        before = self.surface.small()
        ok, error = move.run()
        if not ok:
            return f"{move.step.label}: failed ({error or 'unknown error'})."
        self.sleep(0.6)
        changed = differs(before, self.surface.small())
        self.steps += 1
        params = {"n": self.steps, "kind": move.step.kind, "label": move.step.label, "note": "",
                  "channel": move.step.channel}
        if move.step.point:
            params["x"], params["y"] = move.step.point
        self.events("step", params)
        return (f"{move.step.label} [{move.step.channel}]. The screen {'changed' if changed else 'did NOT change'}."
                + ("" if changed else " If that was unexpected, observe and try another way."))

    def check(self, step: int, kind: str, value: str = "", min_bytes: int = 1) -> str:
        if not 1 <= step <= len(self.plan):
            return f"Step {step} is not in the plan (1..{len(self.plan)}). Record a plan first."
        verdict, summary = self._verify(kind, value, min_bytes)
        self.events("receipt", {"step": step, "verdict": verdict, "summary": summary})
        if verdict != "verified":
            return f"NOT verified: {summary}. The step is not done; fix it and check again."
        self.done.add(step)
        self.baseline = self.surface.small() if self.surface else b""  # the next step is judged from here
        self.events("check", {"step": step})
        left = [i for i in range(1, len(self.plan) + 1) if i not in self.done]
        return f"Step {step} verified: {summary}." + (f" Remaining steps: {left}." if left else " Every step is verified.")

    # --- internals --------------------------------------------------------------

    def _problem(self) -> Optional[str]:
        if self.surface is None:
            return "Nothing is open yet: call open(target) first."
        return self.surface.ready()

    def _verify(self, kind: str, value: str, min_bytes: int) -> tuple[str, str]:
        if kind == "file":
            return self._verify_file(value, min_bytes)
        if kind == "changed":
            if not self.baseline or self.surface is None:
                return "contradicted", "no starting screen for this step (open something and record a plan first)"
            if differs(self.baseline, self.surface.small()):
                return "verified", "the screen changed since this step started"
            return "contradicted", "the screen looks the same as when this step started"
        found = self.surface.verify(kind, value) if self.surface else None
        return found or ("contradicted", f"unknown check kind {kind!r} for {self.target or 'this target'}")

    def _verify_file(self, value: str, min_bytes: int) -> tuple[str, str]:
        path = self._find_file(value)
        if path is None:
            return "contradicted", f"no file named {value!r} in {', '.join(SEARCH_DIRS)}"
        size, age = path.stat().st_size, self.clock() - path.stat().st_mtime
        if size < min_bytes:
            return "contradicted", f"{path.name} exists but is {size} bytes"
        if age > FRESH_SECONDS:
            return "contradicted", f"{path.name} exists but was last saved {age / 60:.0f} min ago, before this task"
        return "verified", f"{path.name} exists, {size / 1024:.0f} KB, saved {age:.0f}s ago"

    def _find_file(self, value: str) -> Optional[Path]:
        p = Path(os.path.expandvars(value)).expanduser()
        if p.is_absolute():
            return p if p.is_file() else None
        for d in SEARCH_DIRS:
            for root in (self.home / d, self.home / "OneDrive" / d):
                if (root / p).is_file():
                    return root / p
        return None


def differs(a: bytes, b: bytes, threshold: float = 1.5) -> bool:
    """Mean absolute difference of two same-size grayscale thumbnails, in grey levels."""
    if not a or not b or len(a) != len(b):
        return a != b
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) > threshold


def ascii_only(text: str) -> None:
    if any(ord(c) > 126 for c in text):
        raise Refused("type can only send plain ASCII text here.")


def file_events(path: Optional[str]) -> Callable[[str, dict], None]:
    """Events for the window, appended to a JSON-lines file the sidecar tails. No path: dropped."""
    if not path:
        return lambda method, params: None

    def emit(method: str, params: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": method, "params": params}) + "\n")

    return emit
