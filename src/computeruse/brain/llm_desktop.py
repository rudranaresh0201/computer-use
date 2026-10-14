"""
A real Brain for the desktop LangGraph loop (orchestrator/graph.py): an LLM
reads the foreground window's UIA tree as a numbered list and picks one
action. The same idea as the web agent (web/agent.py), on Windows apps:

    [u3] Button "Seven"          click(id="u3")   -> a mouse click on its center
    [u9] Edit "Text editor"      type(text=...)   -> keystrokes into the focused control
                                 key(keys="ctrl+s"), scroll(direction), done(answer)

Two guards, both because the hands are a real mouse and keyboard:
- the foreground window must still be the app the run started on; if focus
  moves (a terminal, a pop-up from another program), the brain ends the run
  instead of clicking or typing into the wrong window;
- an id that is not on screen, or a reply that is not a tool call, becomes a
  no-op with a note the model sees next step -- and the graph's
  consecutive-failure breaker ends a run that keeps doing it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..action.schema import Action, ActionType
from ..perception.screenshot import Screenshot
from ..perception.uia import UIElement
from .base import Brain
from .chat import ChatModel, ToolCall

MAX_OBSERVATION_CHARS = 6_000
HISTORY_STEPS = 12
INTERACTIVE = {"Button", "SplitButton", "MenuItem", "Hyperlink", "Edit", "ComboBox", "CheckBox", "RadioButton",
               "TabItem", "ListItem", "TreeItem", "Slider", "Spinner", "DataItem", "MenuBar", "Document"}
READABLE = {"Text", "Header", "HeaderItem", "StatusBar", "TitleBar"}

SYSTEM_PROMPT = """\
You operate a Windows application for a person, one action at a time.
Each turn you see the goal, what you did so far, and the elements of the app window from its accessibility
tree, one per line as [id] Role "name". Call exactly one tool.
- click an element by its id; type text into whatever has focus (click a text field first); key for shortcuts.
- Prefer the keyboard when the app takes typed input: one type action beats many clicks
  (Calculator: type 123*45= ; then read the display).
- Only use ids from the current list; the list changes after every action.
- When the goal is achieved, or you can read the answer on screen, call done with the answer.
- If the goal cannot be done in this app, call done and say why."""

TOOLS = [
    {"type": "function", "function": {"name": "click", "description": "Click an element by its id.",
                                      "parameters": {"type": "object", "properties": {"id": {"type": "string"}},
                                                     "required": ["id"]}}},
    {"type": "function", "function": {"name": "type", "description": "Type text into the focused control.",
                                      "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                                                     "required": ["text"]}}},
    {"type": "function", "function": {"name": "key", "description": "Press keys, e.g. enter, escape, ctrl+s.",
                                      "parameters": {"type": "object", "properties": {"keys": {"type": "string"}},
                                                     "required": ["keys"]}}},
    {"type": "function", "function": {"name": "scroll", "description": "Scroll the window.",
                                      "parameters": {"type": "object", "properties": {
                                          "direction": {"type": "string", "enum": ["up", "down"]}},
                                                     "required": ["direction"]}}},
    {"type": "function", "function": {"name": "done", "description": "Finish, with the answer or what was done.",
                                      "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                                                     "required": ["answer"]}}},
]
CHECK_TOOL = {"type": "function", "function": {
    "name": "check", "description": "Mark a plan step finished, once the window shows it is done.",
    "parameters": {"type": "object", "properties": {"step": {"type": "integer"}}, "required": ["step"]}}}
PLAN_PROMPT = """
There is a plan; when the window shows a step is finished, call check with its number, then carry on.
Checking costs no action. Mark every step before you call done."""
MAX_CHECKS_PER_TURN = 3


def render_elements(elements: list[UIElement]) -> tuple[str, dict[str, UIElement]]:
    """The window as the model reads it: interactive elements (named or not) and
    named text, numbered; repeats of the same role+name at the same spot dropped."""
    lines, ids, seen = [], {}, set()
    for i, e in enumerate(elements):
        role = e.control_type
        name = " ".join(e.name.split())[:80]
        if role not in INTERACTIVE and not (role in READABLE and name):
            continue
        key = (role, name, e.center)
        if key in seen:
            continue
        seen.add(key)
        ref = f"u{i}"
        ids[ref] = e
        lines.append(f'[{ref}] {role} "{name}"' if name else f"[{ref}] {role} (no name)")
    text = "\n".join(lines)
    if len(text) > MAX_OBSERVATION_CHARS:
        text = text[:MAX_OBSERVATION_CHARS].rsplit("\n", 1)[0] + "\n... (more elements not shown)"
    return text or "(the accessibility tree shows no elements for this window)", ids


class LLMDesktopBrain(Brain):
    def __init__(
        self,
        chat: ChatModel,
        window_title: str,
        foreground_title: Callable[[], str],
        on_step: Callable[[str, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        plan: list[str] | None = None,
        on_check: Callable[[int], None] | None = None,
    ) -> None:
        self.chat = chat
        self.window_title = window_title.lower()
        self._foreground_title = foreground_title
        self._on_step = on_step or (lambda action, note: None)
        self._should_stop = should_stop or (lambda: False)
        self.plan = list(plan or [])
        self.checked: set[int] = set()
        self._on_check = on_check or (lambda step: None)
        self.history: list[str] = []
        self.tokens = 0

    def decide(self, goal: str, screenshot: Screenshot, ui_elements: list[UIElement],
               action_history: list[Action]) -> Action:
        if self._should_stop():
            return self._finish("STOPPED: stopped by the person")
        title = self._foreground_title()
        if self.window_title not in title.lower():
            return self._finish(f"STOPPED: the foreground window is {title!r}, not the app this run started on")

        screen, ids = render_elements(ui_elements)
        tools = TOOLS + [CHECK_TOOL] if self.plan else TOOLS
        system = SYSTEM_PROMPT + PLAN_PROMPT if self.plan else SYSTEM_PROMPT
        for _ in range(MAX_CHECKS_PER_TURN + 1):
            done_so_far = "\n".join(self.history[-HISTORY_STEPS:]) or "(nothing yet)"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Goal: {goal}\n{self._plan_text()}\nSo far:\n{done_so_far}\n\n"
                                            f"Window {title!r}:\n{screen}"},
            ]
            call = self.chat.next_call(messages, tools)
            self.tokens += sum(call.usage.values())
            if call.name != "check" or not self.plan:
                return self._to_action(call, ids)
            self._check(call.arguments.get("step"))  # no mouse, no screen change: ask again, same turn
        return self._noop("check", "too many checks in one turn; take an action next")

    def _plan_text(self) -> str:
        if not self.plan:
            return ""
        lines = [f"  {'[x]' if i in self.checked else '[ ]'} {i}. {step}" for i, step in enumerate(self.plan, 1)]
        return "\nPlan:\n" + "\n".join(lines) + "\n"

    def _check(self, step) -> None:
        try:
            n = int(step)
        except (TypeError, ValueError):
            n = 0
        if 1 <= n <= len(self.plan) and n not in self.checked:
            self.checked.add(n)
            self.history.append(f"{len(self.history) + 1}. checked plan step {n}")
            self._on_check(n)

    def _to_action(self, call: ToolCall, ids: dict[str, UIElement]) -> Action:
        args: dict[str, Any] = call.arguments
        if call.name == "click":
            element = ids.get(str(args.get("id", "")))
            if element is None:
                return self._noop(f"click {args.get('id')!r}", "refused: that id is not in the current list")
            label = f'click {args["id"]} {element.control_type} "{element.name[:40]}"'
            return self._act(Action(ActionType.LEFT_CLICK, coordinate=element.center), label)
        if call.name == "type" and args.get("text"):
            return self._act(Action(ActionType.TYPE, text=str(args["text"])), f"type {args['text']!r}")
        if call.name == "key" and args.get("keys"):
            return self._act(Action(ActionType.KEY, text=str(args["keys"]).lower()), f"key {args['keys']}")
        if call.name == "scroll" and args.get("direction") in ("up", "down"):
            return self._act(Action(ActionType.SCROLL, scroll_direction=args["direction"], scroll_amount=5),
                             f"scroll {args['direction']}")
        if call.name == "done":
            return self._finish(str(args.get("answer", "")))
        said = (call.text or "").strip()[:120]
        return self._noop("(no action)", f"the reply was not a valid tool call: {said!r}")

    def _act(self, action: Action, label: str) -> Action:
        self.history.append(f"{len(self.history) + 1}. {label}")
        self._on_step(label, "")
        return action

    def _noop(self, label: str, note: str) -> Action:
        self.history.append(f"{len(self.history) + 1}. {label} -- {note}")
        self._on_step(label, note)
        return Action(ActionType.WAIT, duration=0.2)

    def _finish(self, answer: str) -> Action:
        self._on_step("done", answer)
        return Action(ActionType.DONE, text=answer)
