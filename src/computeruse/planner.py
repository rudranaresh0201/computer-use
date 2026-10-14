"""
Before the agent moves: turn what the person typed into a plan.

    "what's 123 times 45"  ->  app: calculator
                               goal: "Compute 123 x 45 and read the result"
                               steps: ["Enter 123", "Multiply by 45", "Read the result"]

One tool call to the same chat model the brain uses. The steps are the
checklist the window shows; the brain ticks them (its `check` tool) as it
works, so "done in order" is visible. If no app fits, the plan says so and
the run ends before anything is clicked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .brain.chat import ChatModel

APP_DESCRIPTIONS = {
    "calculator": "Windows Calculator: arithmetic, conversions",
    "notepad_pp": "Notepad++: write, edit or save text",
    "paint": "Microsoft Paint: draw shapes, lines, text on a canvas",
}
MAX_STEPS = 6

SYSTEM_PROMPT = """\
You plan a task for an agent that operates one Windows application with the mouse and keyboard.
Pick the app that fits, restate the goal precisely, and list 2-6 short steps a person would see happen
on screen, in order. Steps are checkpoints, not individual clicks ("Enter 123", not "click 1, click 2").
If none of the apps can do the task, use app "none" and put the reason in goal."""


@dataclass
class Plan:
    app: Optional[str]  # None: no app can do this
    goal: str
    steps: list[str] = field(default_factory=list)
    tokens: int = 0


def plan_tool(apps: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": "plan", "description": "The plan for the task.",
        "parameters": {"type": "object", "properties": {
            "app": {"type": "string", "enum": [*apps, "none"]},
            "goal": {"type": "string", "description": "The task, stated precisely for the agent"},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "2-6 checkpoints, in order"},
        }, "required": ["app", "goal", "steps"]}}}


def make_plan(chat: ChatModel, command: str, apps: list[str], app: Optional[str] = None) -> Plan:
    """The plan for `command`. `app` set: the person already chose it. A reply that isn't a
    usable plan falls back to the command itself as a one-step plan, never a crash."""
    menu = "\n".join(f"- {a}: {APP_DESCRIPTIONS.get(a, a)}" for a in apps)
    ask = f"Apps:\n{menu}\n\nTask: {command}"
    if app:
        ask += f"\n\nThe person chose the app: {app}. Use it."
    call = chat.next_call([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ask}],
                          [plan_tool(apps)])
    tokens = sum(call.usage.values())
    args = call.arguments if call.name == "plan" else {}
    chosen = app or args.get("app")
    goal = str(args.get("goal") or command).strip()
    if chosen == "none":
        return Plan(None, goal, [], tokens)
    if chosen not in apps:
        chosen = app or (apps[0] if len(apps) == 1 else None)
        if chosen is None:
            return Plan(None, f"I couldn't tell which app to use for: {command}", [], tokens)
    steps = [str(s).strip() for s in args.get("steps") or [] if str(s).strip()][:MAX_STEPS]
    return Plan(chosen, goal, steps or [command], tokens)
