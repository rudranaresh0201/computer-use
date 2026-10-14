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

# runner(goal, app, on_step, should_stop) -> {"status", "answer", ...}
Runner = Callable[[str, str, Callable[[str, str], None], Callable[[], bool]], dict]

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
    def __init__(self, runner: Runner, apps: list[str], out: TextIO) -> None:
        self._runner = runner
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
            if app not in self._apps:
                raise _RpcError(INVALID_PARAMS, f"app must be one of {self._apps}")
            if self.running:
                raise _RpcError(BUSY, "a run is already in progress")
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(goal.strip(), app), daemon=True)
            return {"started": True}, self._thread.start
        raise _RpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")

    def _run(self, goal: str, app: str) -> None:
        count = [0]

        def on_step(label: str, note: str) -> None:
            count[0] += 1
            self.notify("step", {"n": count[0], "kind": step_kind(label, note), "label": label, "note": note})

        self.notify("started", {"goal": goal, "app": app})
        try:
            result = self._runner(goal, app, on_step, self._stop.is_set)
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


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def main(model: str, max_steps: int) -> None:
    from .desktop_run import APPS, run_task

    protocol = sys.stdout
    sys.stdout = sys.stderr  # every stray print() now lands on stderr, not in the protocol stream
    for stream in (protocol, sys.stderr, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    def runner(goal: str, app: str, on_step: Callable[[str, str], None], should_stop: Callable[[], bool]) -> dict:
        return run_task(goal, app, model, max_steps, on_step=on_step, should_stop=should_stop)

    print(f"sidecar ready (brain: {model}); reading JSON-RPC on stdin", file=sys.stderr)
    Sidecar(runner, list(APPS), protocol).serve(sys.stdin)
