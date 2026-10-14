"""
`computeruse desktop-run "<goal>" --app calculator`: the LangGraph loop
(orchestrator/graph.py) on a real Windows app, with an LLM brain.

    perceive (screenshot + UIA tree) -> decide (LLMDesktopBrain) -> act (mouse/keyboard) -> back to decide

The app is opened first and brought to the front; the brain stops the run if
anything else takes the front. Kill switch: slam the mouse into a screen
corner (pyautogui's fail-safe). Every action is logged to logs/desktop_run.jsonl.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

APPS = {
    # name: (command, a word the window title contains)
    "calculator": (["calc.exe"], "Calculator"),
    "notepad_pp": ([r"C:\Program Files\Notepad++\notepad++.exe", "-multiInst", "-nosession"], "Notepad++"),
    "paint": (["mspaint.exe"], "Paint"),
}


def foreground_title() -> str:
    from pywinauto import Desktop

    try:
        return Desktop(backend="uia").window(active_only=True).wrapper_object().window_text() or ""
    except Exception:  # noqa: BLE001 -- no active window is an answer, not a crash
        return ""


def open_app(name: str, wait: float = 20.0):
    from pywinauto import Desktop

    command, title = APPS[name]
    subprocess.Popen(command)
    deadline = time.time() + wait
    while time.time() < deadline:
        time.sleep(1.0)
        for w in Desktop(backend="uia").windows(visible_only=True):
            if title.lower() in (w.window_text() or "").lower():
                w.set_focus()
                time.sleep(1.0)
                return w, title
    raise RuntimeError(f"{name} did not open a window with {title!r} in its title within {wait:.0f}s")


def run_task(
    goal: str,
    app: str,
    model: str,
    max_steps: int,
    on_step: Callable[[str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """One agent run, no printing: open the app, loop perceive/decide/act,
    and report how it ended. Both the terminal (main) and the app's sidecar
    (sidecar.py) call this. Raises RuntimeError if the brain can't start
    (no GROQ_API_KEY) or the app never opens."""
    from .action.controller import ActionController
    from .brain.chat import OpenAICompatibleChat
    from .brain.llm_desktop import LLMDesktopBrain
    from .orchestrator.graph import build_graph
    from .safety import ActionLogger, SafetyConfig, SafetyGuard

    chat = OpenAICompatibleChat.groq(model=model)
    _, title = open_app(app)
    brain = LLMDesktopBrain(chat, title, foreground_title, on_step=on_step, should_stop=should_stop)
    controller = ActionController(SafetyGuard(SafetyConfig(), ActionLogger(Path("logs/desktop_run.jsonl"))))
    graph = build_graph(brain, controller)
    start = time.time()
    state = graph.invoke(
        {"goal": goal, "screenshot": None, "ui_elements": [], "action_history": [], "done": False,
         "step_count": 0, "consecutive_failures": 0},
        config={"recursion_limit": 2 * max_steps + 5},
    )
    last = state["action_history"][-1] if state["action_history"] else None
    if state["done"] and last is not None and not (last.text or "").startswith("STOPPED"):
        status, answer = "done", last.text or ""
    elif last is not None and (last.text or "").startswith("STOPPED"):
        status, answer = "stopped", last.text or ""
    else:
        status = "stopped"
        answer = (f"STOPPED after {len(state['action_history'])} actions "
                  f"({state['consecutive_failures']} failures in a row)")
    return {"status": status, "answer": answer, "actions": len(state["action_history"]),
            "tokens": brain.tokens, "seconds": round(time.time() - start, 1), "brain": chat.name}


def main(goal: str, app: str, model: str, max_steps: int) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print(f"goal:  {goal}\napp:   {app}\nbrain: {model} (Groq)\n(kill switch: move the mouse into a screen corner)\n")
    count = [0]

    def show(label: str, note: str) -> None:
        count[0] += 1
        print(f"{count[0]:>3}. {label}" + (f"\n     {note}" if note else ""))

    try:
        result = run_task(goal, app, model, max_steps, on_step=show)
    except RuntimeError as e:
        print(e)
        return 2
    print()
    print(f"DONE: {result['answer']}" if result["status"] == "done" else result["answer"])
    print(f"{result['actions']} actions, {result['tokens'] / 1000:.1f}k tokens, {result['seconds']:.0f}s")
    return 0 if result["status"] == "done" else 1
