"""
The sidecar's `claude` brain: Claude Code, headless, drives our MCP server.

    window --run--> sidecar --spawns--> claude -p "<goal>" --output-format stream-json ...
                                            └── stdio MCP ──> computeruse mcp  (the hands; writes CU_EVENTS)

Two streams reach the window, and they never mix:
  * claude's stdout (stream-json): its sentences become `thought` notifications, the final
    `result` becomes `finished`.
  * the MCP server's events file (CU_EVENTS, JSON lines): `plan`, `started`, `step`, `screen`,
    `receipt`, `check`, forwarded as they are written.
The model's tool results stay plain text; the window still sees every step, screen and receipt.
Stop kills the whole process tree (claude -> node -> the MCP server), so nothing keeps clicking.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

TOOLS = ("targets", "open", "plan", "observe", "act", "check")
APPEND_PROMPT = (
    "You are the reasoning of a desktop agent shown in a 3D app. The person reads every sentence you write "
    "as your thinking, live, so before each tool call write ONE short plain sentence about what you are doing "
    "and why. Finish with one sentence saying what was done.")

Emit = Callable[[str, dict], None]


def thoughts_from(line: str) -> list[str]:
    """The sentences in one stream-json line (assistant text blocks), for the thinking panel."""
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return []
    if msg.get("type") != "assistant":
        return []
    return [b["text"].strip() for b in msg.get("message", {}).get("content", [])
            if b.get("type") == "text" and b.get("text", "").strip()]


def result_from(line: str) -> Optional[dict]:
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    if msg.get("type") != "result":
        return None
    ok = msg.get("subtype") == "success" and not msg.get("is_error")
    usage = msg.get("usage") or {}
    return {"status": "done" if ok else "error", "answer": (msg.get("result") or "").strip(),
            "turns": msg.get("num_turns"), "tokens": usage.get("output_tokens", 0)}


class EventTail:
    """Follows the MCP server's events file and forwards each new line, from a thread."""

    def __init__(self, path: Path, emit: Emit, poll: float = 0.1) -> None:
        self.path, self.emit, self.poll = path, emit, poll
        self.count = {"step": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "EventTail":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._drain()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain()
            time.sleep(self.poll)

    def _drain(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            f.seek(getattr(self, "_pos", 0))
            while True:
                line = f.readline()
                if not line or not line.endswith("\n"):  # a half-written line waits for the next pass
                    break
                self._pos = f.tell()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                method = event.get("method")
                if method:
                    if method in self.count:
                        self.count[method] += 1
                    self.emit(method, event.get("params") or {})


def claude_exe() -> str:
    """The real claude executable. On Windows `claude` is a .cmd shim, and cmd.exe would re-parse
    the prompt (quotes, newlines, & and | in a person's task): call the binary it forwards to."""
    shim = shutil.which("claude.cmd")
    if shim:
        real = Path(shim).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if real.exists():
            return str(real)
    return shutil.which("claude") or "claude"


def claude_command(goal: str, app: Optional[str], mcp_config: Path, model: Optional[str] = None) -> list[str]:
    exe = claude_exe()
    prompt = goal if not app else f"{goal}\n\n(Use the app: {app}.)"
    cmd = [exe, "-p", prompt, "--mcp-config", str(mcp_config), "--strict-mcp-config", "--tools", "",
           "--allowedTools", ",".join(f"mcp__computeruse__{t}" for t in TOOLS),
           "--append-system-prompt", APPEND_PROMPT,
           "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    return cmd


def mcp_config(events: Path) -> dict:
    server = Path(sys.executable).with_name("computeruse.exe")
    command = [str(server), "mcp"] if server.exists() else [sys.executable, "-m", "computeruse.cli", "mcp"]
    return {"mcpServers": {"computeruse": {"command": command[0], "args": command[1:], "env": {"CU_EVENTS": str(events)}}}}


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()


def run_claude(goal: str, app: Optional[str], emit: Emit, should_stop: Callable[[], bool],
               model: Optional[str] = None, command: Optional[list[str]] = None,
               workdir: Optional[Path] = None) -> dict:
    """One task, start to finish. Returns the `finished` payload."""
    work = Path(workdir or tempfile.mkdtemp(prefix="cu-claude-"))
    events = work / "events.jsonl"
    events.write_text("", encoding="utf-8")
    config = work / "mcp.json"
    config.write_text(json.dumps(mcp_config(events)), encoding="utf-8")
    cmd = command or claude_command(goal, app, config, model)

    start = time.time()
    tail = EventTail(events, emit).start()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "CU_EVENTS": str(events)},
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    stopped = threading.Event()

    def watch_stop() -> None:
        while proc.poll() is None:
            if should_stop():
                stopped.set()
                kill_tree(proc)
                return
            time.sleep(0.2)

    threading.Thread(target=watch_stop, daemon=True).start()
    result: Optional[dict] = None
    for line in proc.stdout:
        for text in thoughts_from(line):
            emit("thought", {"text": text})
        result = result_from(line) or result
    proc.wait()
    tail.stop()
    seconds = round(time.time() - start, 1)
    actions = tail.count["step"]
    if stopped.is_set():
        return {"status": "stopped", "answer": "STOPPED: stopped by the person", "actions": actions,
                "seconds": seconds, "brain": "claude"}
    if result is None:
        err = (proc.stderr.read() or "").strip().splitlines()[-1:] if proc.stderr else []
        return {"status": "error", "answer": f"claude ended without a result ({err[0] if err else f'exit {proc.returncode}'})",
                "actions": actions, "seconds": seconds, "brain": "claude"}
    return {**result, "actions": actions, "seconds": seconds, "brain": "claude"}
