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
    def __init__(self, runner: Runner, apps: list[str], out: TextIO, planner: Planner | None = None) -> None:
        self._runner = runner
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
            return {"pong": True}, None
        if method == "apps":
            return {"apps": self._apps}, None
        if method == "stop":
            self._stop.set()
            return {"stopping": self.running}, None
        if method == "run":
            goal, app = params.get("goal"), params.get("app")
            if not isinstance(goal, str) or not goal.strip():
                raise _RpcError(INVALID_PARAMS, "run needs a non-empty goal")
            if app is None and self._planner is None:
                raise _RpcError(INVALID_PARAMS, "run needs an app (this sidecar has no planner)")
            if app is not None and app not in self._apps:
                raise _RpcError(INVALID_PARAMS, f"app must be one of {self._apps}")
            if self.running:
                raise _RpcError(BUSY, "a run is already in progress")
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(goal.strip(), app), daemon=True)
            return {"started": True}, self._thread.start
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


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def main(model: str, max_steps: int, demo: bool = False) -> None:
    from .brain.chat import OpenAICompatibleChat
    from .desktop_run import APPS, run_task
    from .planner import make_plan

    protocol = sys.stdout
    sys.stdout = sys.stderr  # every stray print() now lands on stderr, not in the protocol stream
    for stream in (protocol, sys.stderr, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    def runner(goal: str, app: str, on_step: Callable[[str, str], None], should_stop: Callable[[], bool],
               plan: list[str] | None = None, on_check: Callable[[int], None] | None = None) -> dict:
        if demo:
            return demo_runner(goal, app, on_step, should_stop, plan=plan, on_check=on_check)
        return run_task(goal, app, model, max_steps, on_step=on_step, should_stop=should_stop,
                        plan=plan, on_check=on_check)

    def planner(command: str, app: str | None) -> dict:
        if demo:
            return demo_planner(command, app)
        p = make_plan(OpenAICompatibleChat.groq(model=model), command, list(APPS), app)
        return {"app": p.app, "goal": p.goal, "steps": p.steps}

    print(f"sidecar ready (brain: {'demo, no mouse' if demo else model}); reading JSON-RPC on stdin", file=sys.stderr)
    Sidecar(runner, list(APPS), protocol, planner=planner).serve(sys.stdin)
