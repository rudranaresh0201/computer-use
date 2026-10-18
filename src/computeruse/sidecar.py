"""
`computeruse sidecar`: the agent as a child process the app's window talks to.

The window (app/, Electron) is the host; this process is the sidecar. They
speak JSON-RPC 2.0, one message per line, over stdin/stdout -- the same
framing as an MCP stdio server:

    window -> sidecar   {"jsonrpc":"2.0","id":1,"method":"run","params":{"goal":"...","app":"calculator"}}
    sidecar -> window   {"jsonrpc":"2.0","id":1,"result":{"started":true}}
    sidecar -> window   {"jsonrpc":"2.0","method":"step","params":{"n":1,"kind":"click","label":"...","note":""}}
    sidecar -> window   {"jsonrpc":"2.0","method":"finished","params":{"status":"done","answer":"5535",...}}
    window -> sidecar   {"jsonrpc":"2.0","id":2,"method":"stop"}

Methods: ping, apps, run, stop. A run happens on a thread so `stop` can be
read while it works; the brain checks the stop flag before every decision.
`run` without an app is planned first (planner.py): a `plan` notification
carries the app and the checklist, and `check` notifications tick it.
stdout carries protocol messages only: anything else that prints (a library,
a stray debug line) is sent to stderr, or the window would read it as a
broken message.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from typing import Any, TextIO

# runner(goal, app, on_step, should_stop, plan=[...], on_check=fn) -> {"status", "answer", ...}
Runner = Callable[..., dict]
# planner(command, app or None) -> {"app": str | None, "goal": str, "steps": [str]}
Planner = Callable[[str, "str | None"], dict]

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, BUSY = -32700, -32600, -32601, -32602, -32000


def step_kind(label: str, note: str) -> str:
    """What the robot should act out for one step: click / type / key / scroll /
    done, or noop for a step the brain refused (an unknown id, a bad reply)."""
    if label == "done":
        return "done"
    if note:
        return "noop"
    word = label.split(" ", 1)[0]
    return word if word in ("click", "type", "key", "scroll") else "noop"


class Sidecar:
    def __init__(self, runner: Runner, apps: list[str], out: TextIO, planner: Planner | None = None,
                 brain: str = "", app_optional: bool = False,
                 extra: "dict[str, Callable[[dict], dict]] | None" = None) -> None:
        self._runner = runner
        self._extra = extra or {}  # more methods, e.g. approve for the claude brain
        self._brain = brain
        self._app_optional = app_optional  # the brain picks the app itself (claude)
        self._planner = planner
        self._apps = apps
        self._out = out
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        # a thread made but not yet started (between the reply and start) counts as running
        return self._thread is not None and (self._thread.is_alive() or self._thread.ident is None)

    def send(self, message: dict) -> None:
        with self._write_lock:
            self._out.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
            self._out.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"method": method, "params": params})

    def handle(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self.send({"id": None, "error": {"code": PARSE_ERROR, "message": "not JSON"}})
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
            self.send({"id": msg.get("id") if isinstance(msg, dict) else None,
                       "error": {"code": INVALID_REQUEST, "message": "expected an object with a method"}})
            return
        has_id = "id" in msg
        try:
            result, then = self._dispatch(msg["method"], msg.get("params") or {})
        except _RpcError as e:
            if has_id:
                self.send({"id": msg["id"], "error": {"code": e.code, "message": str(e)}})
            return
        if has_id:  # a message without an id is a notification: no reply
            self.send({"id": msg["id"], "result": result})
        if then is not None:  # after the reply, so "started" never arrives before the answer to "run"
            then()

    def _dispatch(self, method: str, params: dict[str, Any]) -> tuple[dict, Callable[[], None] | None]:
        """The reply for one request, plus anything to start once it is sent."""
        if method == "ping":
            return {"pong": True, "brain": self._brain}, None
        if method == "apps":
            return {"apps": self._apps}, None
        if method == "stop":
            self._stop.set()
            return {"stopping": self.running}, None
        if method == "run":
            goal, app = params.get("goal"), params.get("app")
            if not isinstance(goal, str) or not goal.strip():
                raise _RpcError(INVALID_PARAMS, "run needs a non-empty goal")
            if app is None and self._planner is None and not self._app_optional:
                raise _RpcError(INVALID_PARAMS, "run needs an app (this sidecar has no planner)")
            if app is not None and app not in self._apps:
                raise _RpcError(INVALID_PARAMS, f"app must be one of {self._apps}")
            if self.running:
                raise _RpcError(BUSY, "a run is already in progress")
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(goal.strip(), app), daemon=True)
            return {"started": True}, self._thread.start
        if method in self._extra:
            try:
                return self._extra[method](params), None
            except ValueError as e:
                raise _RpcError(INVALID_PARAMS, str(e)) from e
        raise _RpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")

    def _run(self, goal: str, app: str | None) -> None:
        count = [0]

        def on_step(label: str, note: str) -> None:
            count[0] += 1
            self.notify("step", {"n": count[0], "kind": step_kind(label, note), "label": label, "note": note})

        def on_check(step: int) -> None:
            self.notify("check", {"step": step})

        try:
            steps: list[str] = []
            if self._planner is not None:
                self.notify("planning", {"goal": goal})
                plan = self._planner(goal, app)
                if not plan.get("app"):
                    self.notify("finished", {"status": "stopped", "actions": 0,
                                             "answer": f"STOPPED: {plan.get('goal') or 'no app fits'}"})
                    return
                goal, app, steps = plan.get("goal") or goal, plan["app"], list(plan.get("steps") or [])
                self.notify("plan", {"app": app, "goal": goal, "steps": steps})
            if self._planner is None and app is None:
                self.notify("planning", {"goal": goal})  # the brain plans and opens the app itself
            else:
                self.notify("started", {"goal": goal, "app": app})
            result = self._runner(goal, app, on_step, self._stop.is_set, plan=steps, on_check=on_check)
        except Exception as e:  # noqa: BLE001 -- any failure must reach the window as an ending, not silence
            result = {"status": "error", "answer": f"{type(e).__name__}: {e}"}
        self.notify("finished", result)

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def serve(self, inp: TextIO) -> None:
        for line in inp:
            if line.strip():
                self.handle(line)
        # stdin closed: the window is gone, so end any run rather than leave the mouse moving
        self._stop.set()
        self.wait()


DEMO_PLAN = {"app": "calculator", "goal": "Compute 123 x 45 in Calculator and read the result",
             "steps": ["Enter 123", "Multiply by 45", "Read the result"]}
DEMO_STEPS = [  # (label, note, the plan step this action finishes, 0 = none)
    ('click u57 Button "One"', "", 0), ('click u58 Button "Two"', "", 0), ('click u59 Button "Three"', "", 1),
    ('click u41 Button "Multiply by"', "", 0), ('click u60 Button "Four"', "", 0), ('click u61 Button "Five"', "", 0),
    ('click u53 Button "Equals"', "", 2), ("done", "123 x 45 = 5535", 3),
]


def demo_planner(command: str, app: str | None) -> dict:
    return dict(DEMO_PLAN)


def demo_runner(goal: str, app: str, on_step: Callable[[str, str], None], should_stop: Callable[[], bool],
                plan: list[str] | None = None, on_check: Callable[[int], None] | None = None,
                pause: float = 1.2) -> dict:
    """A scripted run with the real message shapes and no mouse or keyboard: for building
    the window and for showing it off. Same answer whatever the goal says."""
    import time

    start = time.time()
    for n, (label, note, finishes) in enumerate(DEMO_STEPS, 1):
        time.sleep(pause)
        if should_stop():
            on_step("done", "STOPPED: stopped by the person")
            return {"status": "stopped", "answer": "STOPPED: stopped by the person", "actions": n,
                    "tokens": 0, "seconds": round(time.time() - start, 1), "brain": "demo"}
        if finishes and on_check is not None and label == "done":
            on_check(finishes)  # the brain ticks the last step, then says done
        on_step(label, note)
        if finishes and on_check is not None and label != "done":
            on_check(finishes)
    return {"status": "done", "answer": DEMO_STEPS[-1][1], "actions": len(DEMO_STEPS), "tokens": 0,
            "seconds": round(time.time() - start, 1), "brain": "demo"}


SHOWCASE_PLAN = {"app": "paint", "goal": "Draw a house with a sun and a tree in Paint, then save it as house.png",
                 "steps": ["Pick the rectangle tool", "Draw the house", "Add the roof", "Draw the sun",
                           "Draw the tree", "Save as house.png"]}
# (thought, label, kind, channel, (x, y) as fractions of the screen or None, plan step it finishes, receipt)
SHOWCASE_STEPS = [
    ("Paint is open. The toolbar is in the accessibility tree, so I can pick the rectangle by name.",
     'click u112 Button "Rectangle"', "click", "tree", (0.44, 0.09), 1, "tool selected: Rectangle (tree state)"),
    ("The canvas is a single element to the tree: drawing has to be coordinates. House body first, centred low.",
     "drag (0.38, 0.52) -> (0.62, 0.82)", "drag", "vision", (0.5, 0.67), 2, "canvas changed in the box drawn"),
    ("Roof: two lines meeting above the house. The line tool is named too.",
     'click u109 Button "Line"', "click", "tree", (0.41, 0.09), 0, None),
    ("Left slope, then right slope.",
     "drag (0.36, 0.52) -> (0.50, 0.36)", "drag", "vision", (0.43, 0.44), 0, None),
    ("", "drag (0.50, 0.36) -> (0.64, 0.52)", "drag", "vision", (0.57, 0.44), 3, "roof lines present"),
    ("The sun: an unnamed icon in the shapes gallery, so the tree can't tell me which one is the oval. Asking the eyes.",
     'locate "oval shape tool"', "locate", "eyes", (0.47, 0.11), 0, None),
    ("", "drag (0.76, 0.18) -> (0.88, 0.32)", "drag", "vision", (0.82, 0.25), 4, "circle drawn top right"),
    ("Tree: a trunk and a round top on the left of the house.",
     "drag (0.16, 0.62) -> (0.20, 0.82)", "drag", "vision", (0.18, 0.72), 0, None),
    ("", "drag (0.10, 0.44) -> (0.26, 0.64)", "drag", "vision", (0.18, 0.54), 5, "tree drawn left of the house"),
    ("Saving: Ctrl+S opens the dialog, the file name box is named.",
     "key ctrl+s", "key", "api", None, 0, None),
    ("", 'type u7 Edit "File name" house.png', "type", "tree", (0.42, 0.78), 0, None),
    ("", "key enter", "key", "api", None, 6, "house.png exists, 48 KB, saved 1 s ago"),
]


def _screen_data_url(path: "str | None", width: int = 960) -> "str | None":
    """A screenshot as a small JPEG data URL for the window's live-screen panel."""
    if not path:
        return None
    try:
        import base64
        import io

        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.thumbnail((width, width))
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except OSError:
        return None


def showcase_runner(goal: str, app: str, on_step: Callable[[str, str], None], should_stop: Callable[[], bool],
                    plan: list[str] | None = None, on_check: Callable[[int], None] | None = None,
                    emit: Callable[[str, dict], None] | None = None, pause: float = 1.4,
                    screen: "str | None" = None) -> dict:
    """The demo the window shows off: every event a real run sends (thoughts, the channel each
    action used, where it pointed, receipts, the screen), scripted, with no mouse or keyboard."""
    import time

    emit = emit or (lambda method, params: None)
    image = _screen_data_url(screen)
    if image:
        emit("screen", {"src": image})
    start = time.time()
    for n, (thought, label, kind, channel, point, finishes, receipt) in enumerate(SHOWCASE_STEPS, 1):
        if thought:
            emit("thought", {"text": thought})
            time.sleep(pause * 0.8)
        time.sleep(pause * 0.6)
        if should_stop():
            return {"status": "stopped", "answer": "STOPPED: stopped by the person", "actions": n - 1,
                    "tokens": 0, "seconds": round(time.time() - start, 1), "brain": "demo"}
        params = {"n": n, "kind": kind, "label": label, "note": "", "channel": channel}
        if point:
            params["x"], params["y"] = point
        emit("step", params)
        if finishes:
            if receipt:
                emit("receipt", {"step": finishes, "verdict": "verified", "summary": receipt})
            if on_check is not None:
                on_check(finishes)
    answer = "Saved house.png: a house with a roof, a sun and a tree"
    return {"status": "done", "answer": answer, "actions": len(SHOWCASE_STEPS), "tokens": 0,
            "seconds": round(time.time() - start, 1), "brain": "demo"}


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def approve(run_dir: dict, params: dict) -> dict:
    """The person's answer to an `approval` event, written where the MCP server is waiting for it
    (session.file_approver): <run dir>/approvals/<id>.json."""
    import re

    aid, ok = str(params.get("id", "")), params.get("ok") is True
    if not re.fullmatch(r"[0-9a-f]{8}", aid):
        raise ValueError("approve needs the 8-hex id from the approval event")
    if run_dir.get("path") is None:
        raise ValueError("no run is waiting for an approval")
    folder = run_dir["path"] / "approvals"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{aid}.json").write_text(json.dumps({"ok": ok}), encoding="utf-8")
    return {"recorded": True, "ok": ok}


def loop_runner(goal: str, app: "str | None", notify: Callable[[str, dict], None], should_stop: Callable[[], bool],
                run_dir: dict, run_goal=None) -> dict:
    """Our loop (agent/runner.py) behind the window: the brain only decides; the router, our grounder,
    the hands and the checker are ours. The window's approve() answers land in run_dir/approvals."""
    import os
    import tempfile
    from pathlib import Path

    from .session import file_approver

    if run_goal is None:
        from .agent.runner import run_goal
    run_dir["path"] = Path(tempfile.mkdtemp(prefix="cu-loop-"))

    def emit(method: str, params: dict) -> None:
        if method in ("finished", "run"):  # the sidecar sends the one ending, with the checker's verdict
            return
        notify(method, params)

    approve_fn = file_approver(str(run_dir["path"] / "events.jsonl"), notify)
    _, summary = run_goal(goal, app, emit, brain_kind=os.environ.get("CU_LOOP_BRAIN", "claude"),
                          model=os.environ.get("CU_CLAUDE_MODEL") or None, approve=approve_fn, should_stop=should_stop,
                          max_turns=int(os.environ.get("CU_MAX_TURNS", "40")))
    return summary


def main(model: str, max_steps: int, demo: bool = False, brain: str = "groq") -> None:
    from .brain.chat import OpenAICompatibleChat
    from .desktop_run import APPS, run_task
    from .planner import make_plan

    protocol = sys.stdout
    run_dir: dict = {"path": None}  # the claude brain's current run folder (events, approvals)
    sys.stdout = sys.stderr  # every stray print() now lands on stderr, not in the protocol stream
    for stream in (protocol, sys.stderr, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    def runner(goal: str, app: str, on_step: Callable[[str, str], None], should_stop: Callable[[], bool],
               plan: list[str] | None = None, on_check: Callable[[int], None] | None = None) -> dict:
        if demo:
            from pathlib import Path

            shots = sorted(Path("runs/realtest").glob("paint_*.png"))  # a real Paint screenshot, if captured
            return showcase_runner(goal, app, on_step, should_stop, plan=plan, on_check=on_check,
                                   emit=sidecar.notify, screen=str(shots[-1]) if shots else None)
        if brain == "loop":
            return loop_runner(goal, app, sidecar.notify, should_stop, run_dir)
        if brain == "claude":
            from .claude_brain import run_claude

            import os

            import tempfile
            from pathlib import Path

            run_dir["path"] = Path(tempfile.mkdtemp(prefix="cu-claude-"))
            return run_claude(goal, app, sidecar.notify, should_stop, model=os.environ.get("CU_CLAUDE_MODEL") or None,
                              workdir=run_dir["path"])
        return run_task(goal, app, model, max_steps, on_step=on_step, should_stop=should_stop,
                        plan=plan, on_check=on_check)

    def planner(command: str, app: str | None) -> dict:
        if demo:
            return dict(SHOWCASE_PLAN)
        p = make_plan(OpenAICompatibleChat.groq(model=model), command, list(APPS), app)
        return {"app": p.app, "goal": p.goal, "steps": p.steps}

    import os

    label = ("demo" if demo else f"our loop · {os.environ.get('CU_LOOP_BRAIN', 'claude')}" if brain == "loop"
             else "claude" if brain == "claude" else f"groq · {model}")
    print(f"sidecar ready (brain: {label}); reading JSON-RPC on stdin", file=sys.stderr)
    claude = brain in ("claude", "loop") and not demo  # both pick the target themselves and drive websites too
    targets = list(APPS)
    if claude:  # the claude brain drives websites too (web_surface.SITES), through the agent's Chrome
        from .web_surface import SITES

        targets += list(SITES)
    sidecar = Sidecar(runner, targets, protocol, planner=None if claude else planner, brain=label, app_optional=claude,
                      extra={"approve": lambda params: approve(run_dir, params)})
    sidecar.serve(sys.stdin)
