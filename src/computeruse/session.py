"""
One task, across apps and websites: the plan, the actions, and the proof. `mcp_server.py` exposes
a Session over MCP; the Session hands each action to the *surface* that owns the open target:

    DesktopSurface (desktop_session.py)   Windows apps: UI Automation + mouse/keyboard
    WebSurface     (web_surface.py)       websites in the agent's Chrome: accessibility tree + page input

so "fill a Sheet, write a Doc, draft a Gmail" is one plan with one set of receipts.

How a target is found -- the research result, enforced by the surfaces, not left to a prompt:

    keys / typing into the focused field       channel "api"     no pixels, no guessing
    an element from the tree, BY NAME          channel "tree"    exact, cheap
    x, y our grounder found: locate()          channel "eyes"    for what the tree does not name
    x, y read off the screenshot               channel "vision"  the model's own eyes

Acting on a tree element that has no name is refused: "tree first" with nameless elements is the
naive hybrid that lost to vision alone in Tests A and B. A plan step is ticked only by `check`, and
only when a verifier finds evidence. The window hears about steps, receipts and screens through
`events` (a JSON-lines file the sidecar tails), so the model's tool results stay plain text.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
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
    def state(self) -> Any: ...  # whatever this surface compares to tell "the screen changed"
    def changed(self, before: Any, after: Any) -> bool: ...
    def verify(self, kind: str, value: str) -> Optional[tuple[str, str]]: ...


@dataclass
class Session:
    surfaces: dict[str, Any]  # target name -> the surface that drives it (one surface may own many targets)
    events: Callable[[str, dict], None] = lambda method, params: None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    home: Path = field(default_factory=Path.home)
    # asks the person about an irreversible action; True = go ahead. Default: no one to ask, so no.
    approve: Callable[[str], bool] = lambda reason: False
    # the grounder: (screenshot jpeg, description) -> (point in its pixels or None, seconds). None: no eyes.
    eyes: Optional[Callable[[bytes, str], tuple[Optional[tuple[int, int]], float]]] = None

    target: Optional[str] = None
    surface: Any = None
    plan: list[str] = field(default_factory=list)
    done: set[int] = field(default_factory=set)
    baseline: Any = None  # the surface's state when the current plan step started
    steps: int = 0
    shot: Optional[Shot] = None  # the latest screenshot the model saw
    located: dict = field(default_factory=dict)  # (x, y) the eyes returned since that screenshot -> description

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
        self.baseline = self.surface.state() if self.surface else None
        self.events("plan", {"app": self.target, "steps": self.plan})
        return "Plan recorded:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.plan, 1))

    def observe(self) -> tuple[str, bytes]:
        problem = self._problem()
        if problem:
            return problem, b""
        text, shot = self.surface.observe()
        self.shot, self.located = shot, {}
        self.events("screen", {"src": "data:image/jpeg;base64," + base64.b64encode(shot.jpeg).decode()})
        return text, shot.jpeg

    def locate(self, description: str) -> tuple[str, bytes]:
        """Our grounder finds `description` in the latest screenshot. Returns the point and a zoomed
        crop with a ring on it, so the model can confirm the eyes before it clicks: on apps they never
        trained on they are right about half the time (the paper's held-out number)."""
        problem = self._problem()
        if problem:
            return problem, b""
        if self.eyes is None:
            return "The eyes are not running here: read x, y off the screenshot yourself.", b""
        if self.shot is None or not self.surface.observed:
            return "Observe first: the eyes look at the latest screenshot.", b""
        point, seconds = self.ground_with_eyes(description)
        if point is None:
            return f"The eyes found nothing readable for {description!r} ({seconds:.1f}s). Use the screenshot.", b""
        x, y = point
        return (f"The eyes put {description!r} at x={x}, y={y} ({seconds:.1f}s). The zoom shows a ring there: "
                f"if it sits on the right thing, act with exactly these x, y; if not, use your own reading "
                f"of the screenshot."), zoom(self.shot.jpeg, (x, y))

    def ground_with_eyes(self, description: str) -> tuple[Optional[tuple[int, int]], float]:
        """Our grounder on the latest screenshot: the point in its pixels (or None) and the seconds it
        took. The point is remembered, so an act() on exactly it is reported as channel "eyes"."""
        if self.eyes is None or self.shot is None:
            return None, 0.0
        point, seconds = self.eyes(self.shot.jpeg, description)
        if point is not None:
            x, y = point
            self.located[(x, y)] = description
            w, h = self.shot.size
            self.events("locate", {"what": description, "x": x / w, "y": y / h, "seconds": round(seconds, 1)})
        return point, seconds

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
        if move.step.channel == "vision" and (x, y) in self.located:
            move.step.channel = "eyes"  # the coordinates came from our grounder, not the model's reading
        risk = getattr(self.surface, "risk", None)
        reason = risk(action, element, x, y, text, keys) if risk else None
        if reason and not self.approve(reason):
            return (f"The person did not approve: {reason}. Do not do it another way; "
                    "finish what you can without it and say what is left for them.")
        before = self.surface.state()
        ok, error = move.run()
        if not ok:
            return f"{move.step.label}: failed ({error or 'unknown error'})."
        self.sleep(0.6)
        changed = self.surface.changed(before, self.surface.state())
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
        self.baseline = self.surface.state() if self.surface else None  # the next step is judged from here
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
            if self.baseline is None or self.surface is None:
                return "contradicted", "no starting screen for this step (open something and record a plan first)"
            if self.surface.changed(self.baseline, self.surface.state()):
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


def zoom(jpeg: bytes, point: tuple[int, int], box: tuple[int, int] = (320, 200), scale: int = 2) -> bytes:
    """A crop of the screenshot around `point`, enlarged, with a ring on the point."""
    import io

    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    bw, bh = min(box[0], image.width), min(box[1], image.height)
    left = max(0, min(point[0] - bw // 2, image.width - bw))
    top = max(0, min(point[1] - bh // 2, image.height - bh))
    crop = image.crop((left, top, left + bw, top + bh)).resize((bw * scale, bh * scale), Image.LANCZOS)
    cx, cy, r = (point[0] - left) * scale, (point[1] - top) * scale, 14
    draw = ImageDraw.Draw(crop)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(251, 146, 60), width=3)
    out = io.BytesIO()
    crop.save(out, "JPEG", quality=85)
    return out.getvalue()


THUMB = (160, 90)  # grayscale thumbnail size used to compare screens


def differs(a: bytes, b: bytes, mean_threshold: float = 1.5, strong: int = 48, strong_pixels: int = 3) -> bool:
    """Two same-size grayscale thumbnails differ if the average moved, OR a few pixels moved a lot:
    a short line of new text barely moves the average (seen in the web tests), but it does
    flip a handful of pixels hard."""
    if not a or not b or len(a) != len(b):
        return a != b
    deltas = [abs(x - y) for x, y in zip(a, b)]
    return sum(deltas) / len(deltas) > mean_threshold or sum(d > strong for d in deltas) >= strong_pixels


def ascii_only(text: str) -> None:
    if any(ord(c) > 126 for c in text):
        raise Refused("type can only send plain ASCII text here.")


RISKY_WORDS = re.compile(
    r"\b(buy|pay|purchase|checkout|place order|order now|delete|remove|send|post|publish|submit order|"
    r"unsubscribe|sign out|log out|transfer|confirm payment)\b", re.IGNORECASE)
SECRET_FIELDS = re.compile(r"pass(word|code)|\bpin\b|card number|cvv", re.IGNORECASE)


def file_approver(events_path: Optional[str], emit: Callable[[str, dict], None], timeout: float = 90.0,
                  poll: float = 0.25, clock: Callable[[], float] = time.time,
                  sleep: Callable[[float], None] = time.sleep) -> Callable[[str], bool]:
    """Ask the person through the app: emit `approval`, then wait for approvals/<id>.json (written by
    the sidecar when they click) next to the events file. No events file means no app to ask: no."""
    if not events_path:
        return lambda reason: False
    folder = Path(events_path).parent / "approvals"

    def approve(reason: str) -> bool:
        aid = uuid.uuid4().hex[:8]
        emit("approval", {"id": aid, "action": reason})
        answer, deadline = folder / f"{aid}.json", clock() + timeout
        while clock() < deadline:
            if answer.exists():
                ok = json.loads(answer.read_text(encoding="utf-8")).get("ok") is True
                emit("approval-done", {"id": aid, "ok": ok})
                return ok
            sleep(poll)
        emit("approval-done", {"id": aid, "ok": False, "timeout": True})
        return False

    return approve


def file_events(path: Optional[str]) -> Callable[[str, dict], None]:
    """Events for the window, appended to a JSON-lines file the sidecar tails. No path: dropped."""
    if not path:
        return lambda method, params: None

    def emit(method: str, params: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"method": method, "params": params}) + "\n")

    return emit
