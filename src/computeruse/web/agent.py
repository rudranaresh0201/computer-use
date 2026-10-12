"""
A web agent: a model reads what's on screen and picks one action at a time.

    look   -- the part of the accessibility tree that is on screen right now
    decide -- the model calls exactly one tool: click / type / press / scroll /
              goto / back / done
    act    -- web/browser.py does it and reports whether the page changed
    repeat -- until the model calls done, or max_steps

Each call to the model is self-contained: goal + a short list of the steps
so far + the current screen. Old screens are never resent, so a call costs
about the same at step 12 as at step 1 -- what keeps a free tier (Groq:
~8K tokens/minute) usable.

Every step is logged (action, result, tokens, seconds); an episode is one
JSON line. Those lines are the raw material for the research numbers:
success rate, steps, tokens, and how often the agent says "done" when it
isn't.

Safety: actions that spend money, delete, send, or touch a password are
shown to the person first and skipped unless they say yes. Chrome runs with
a fresh profile (browser.py), so nothing is logged in to begin with.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from computeruse.brain.chat import ChatModel, ToolCall
from computeruse.web.browser import ActResult, Browser, PageView

MAX_OBSERVATION_CHARS = 6_000  # ~1.5K tokens of screen per step
HISTORY_STEPS = 12

SYSTEM_PROMPT = """\
You operate a web browser to achieve the user's goal.
Each turn you see the goal, the steps you already took, and the part of the page currently on screen,
as an accessibility tree: one element per line, `role "name" [ref=e12]`. Indentation shows nesting.
Act by calling exactly ONE tool per turn, using a ref from the CURRENT screen.
- To search: type into the search box with submit=true.
- If what you need is not on screen, scroll down, or go to a better page.
- Read the page text to find answers; don't guess.
- When the goal is achieved, call done with a short answer or summary.
- If the goal is impossible, call done and say why.
Never enter passwords or payment details."""

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "click", "description": "Click an element on the current screen.",
        "parameters": {"type": "object", "properties": {"ref": {"type": "string", "description": "e.g. e12"}},
                       "required": ["ref"]}}},
    {"type": "function", "function": {
        "name": "type", "description": "Replace the text in a text box. submit=true presses Enter afterwards.",
        "parameters": {"type": "object", "properties": {
            "ref": {"type": "string"}, "text": {"type": "string"}, "submit": {"type": "boolean"}},
            "required": ["ref", "text"]}}},
    {"type": "function", "function": {
        "name": "press", "description": "Press a key, e.g. Enter, Escape, Tab, PageDown.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "scroll", "description": "Scroll the page to see more.",
        "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["down", "up"]}},
                       "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "goto", "description": "Open a URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "back", "description": "Go back to the previous page.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "done", "description": "Finish: give the answer, or say what was done / why it is impossible.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]

_BOX_ATTR = re.compile(r"\s*\[box=(-?\d+),(-?\d+),(\d+),(\d+)\]")
_NOISE_ATTRS = re.compile(r"\s*\[(?:cursor=pointer|active)\]")
_BARE_CONTAINER = re.compile(r"^\s*-\s+generic(?:\s+\[ref=[^\]]+\])?:?\s*$")
_RISKY_WORDS = re.compile(
    r"\b(buy|pay|purchase|checkout|place order|order now|delete|remove|send|post|publish|submit order|"
    r"unsubscribe|sign out|log out|transfer|confirm payment)\b",
    re.IGNORECASE,
)


def render_observation(view: PageView, viewport: tuple[int, int, int, int], max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """The accessibility tree, cut down to what is on screen. A line with a
    box is shown if its box overlaps the viewport; a line without one (text,
    attributes) follows the nearest element above it in the tree."""
    _, top, _, height = viewport
    bottom = top + height
    shown: list[str] = []
    stack: list[tuple[int, bool]] = []  # (indent, visible) of open boxed ancestors
    for line in view.snapshot.splitlines():
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        box = _BOX_ATTR.search(line)
        if box:
            y, h = int(box.group(2)), int(box.group(4))
            visible = h > 0 and y < bottom and y + h > top
            stack.append((indent, visible))
        else:
            visible = stack[-1][1] if stack else True
        clean = _NOISE_ATTRS.sub("", _BOX_ATTR.sub("", line))
        if not visible or clean.strip().startswith("- /url:") or _BARE_CONTAINER.match(clean):
            continue
        shown.append(clean)
    text = "\n".join(shown)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n... (more on this screen than fits; scroll to see the rest)"
    return f"Title: {view.title}\nURL: {view.url}\n\n{text or '(nothing on screen -- try scrolling or waiting)'}"


def risk_of(name: str, arguments: dict[str, Any], view: PageView) -> Optional[str]:
    """Why this action needs a person's yes, or None."""
    element = view.find(str(arguments.get("ref", ""))) if "ref" in arguments else None
    label = element.name if element else ""
    if name == "type" and re.search(r"pass(word|code)|\bpin\b|card number|cvv", label, re.IGNORECASE):
        return f"typing into {label!r}"
    if name in ("click", "type") and _RISKY_WORDS.search(label):
        return f"{name} on {label!r}"
    if name == "press" and arguments.get("key") == "Enter" and _RISKY_WORDS.search(view.title):
        return f"pressing Enter on a page titled {view.title!r}"
    return None


@dataclass
class Step:
    index: int
    action: str
    arguments: dict[str, Any]
    ok: bool
    detail: str
    page_changed: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_seconds: float = 0.0
    thought: str = ""


@dataclass
class Episode:
    goal: str
    model: str
    status: str = "running"  # done | max_steps | stuck | error
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    start_url: str = ""
    final_url: str = ""
    video: Optional[str] = None
    error: str = ""

    @property
    def tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    def to_json(self) -> dict[str, Any]:
        record = asdict(self)
        record["tokens"] = self.tokens
        return record


class WebAgent:
    def __init__(
        self,
        browser: Browser,
        model: ChatModel,
        max_steps: int = 15,
        confirm: Callable[[str], bool] = lambda reason: False,
        on_step: Callable[[Step], None] = lambda step: None,
        log_path: Optional[Path] = None,
        max_consecutive_failures: int = 3,
    ) -> None:
        self.browser = browser
        self.model = model
        self.max_steps = max_steps
        self.confirm = confirm
        self.on_step = on_step
        self.log_path = log_path
        self.max_consecutive_failures = max_consecutive_failures

    def run(self, goal: str, start_url: Optional[str] = None) -> Episode:
        episode = Episode(goal=goal, model=self.model.name, start_url=start_url or "")
        started = time.monotonic()
        failures = 0
        try:
            if start_url:
                self.browser.goto(start_url)
            for index in range(1, self.max_steps + 1):
                view = self.browser.observe()
                screen = render_observation(view, self.browser.viewport_rect())
                call = self.model.next_call(self._messages(goal, episode.steps, screen), TOOLS)
                step = self._execute(index, call, view)
                episode.steps.append(step)
                self.on_step(step)
                if step.action == "done":
                    episode.status, episode.answer = "done", str(call.arguments.get("answer", ""))
                    break
                failures = 0 if step.ok else failures + 1
                if failures >= self.max_consecutive_failures:
                    episode.status = "stuck"
                    break
            else:
                episode.status = "max_steps"
        except Exception as e:  # noqa: BLE001 -- an episode records its own failure
            episode.status, episode.error = "error", f"{type(e).__name__}: {e}"
        episode.seconds = round(time.monotonic() - started, 2)
        episode.final_url = self.browser.page.url if self.browser.page is not None else ""
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(episode.to_json()) + "\n")
        return episode

    @staticmethod
    def _messages(goal: str, steps: list[Step], screen: str) -> list[dict[str, Any]]:
        history = "\n".join(
            f"{s.index}. {s.action} {json.dumps(s.arguments)} -> {'ok' if s.ok else 'FAILED'}: {s.detail}"
            for s in steps[-HISTORY_STEPS:]
        ) or "(none yet)"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Goal: {goal}\n\nSteps so far:\n{history}\n\nThe screen now:\n{screen}\n\n"
                                        "Call exactly one tool."},
        ]

    def _execute(self, index: int, call: ToolCall, view: PageView) -> Step:
        name, args = call.name, call.arguments
        step = Step(index, name or "(no tool call)", args, False, "", False,
                    call.usage.get("prompt_tokens", 0), call.usage.get("completion_tokens", 0),
                    round(call.latency_s, 2), call.text[:300])
        if not name:
            step.detail = "the model answered without calling a tool"
            return step
        if name == "done":
            step.ok, step.detail = True, "finished"
            return step
        if "ref" in args and view.find(str(args["ref"])) is None:
            step.detail = f"{args['ref']} is not on the current screen -- use a ref from the screen shown"
            return step
        reason = risk_of(name, args, view)
        if reason and not self.confirm(reason):
            step.detail = f"not done: a person did not approve {reason}"
            return step
        result = self._dispatch(name, args)
        step.ok, step.detail, step.page_changed = result.ok, result.detail, result.page_changed
        return step

    def _dispatch(self, name: str, args: dict[str, Any]) -> ActResult:
        web = self.browser
        if name == "click":
            return web.click(str(args.get("ref", "")))
        if name == "type":
            return web.type(str(args.get("ref", "")), str(args.get("text", "")), submit=bool(args.get("submit")))
        if name == "press":
            return web.press(str(args.get("key", "")))
        if name == "scroll":
            return web.scroll(str(args.get("direction", "down")))
        if name == "goto":
            return web.goto(str(args.get("url", "")))
        if name == "back":
            return web.back()
        return ActResult(False, f"unknown tool {name!r}")
