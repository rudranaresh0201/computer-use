"""
The brains: they read the situation and say what to do next as one JSON intent (intent.py).
They do not click, do not choose the channel, and do not decide whether the task succeeded.

    ClaudeCodeBrain   `claude -p` with every tool turned off: a pure reasoner, one call per step,
                      the screenshot sent as an image (stream-json input). Runs on the person's
                      Claude Code subscription, no API key.
    ChatBrain         any OpenAI-compatible chat model (Groq's free gpt-oss-120b): text only, so it
                      gets the tree and our grounder is its only sight -- arm "paper_blind".
    ScriptedBrain     fixed replies, for tests and for exercising the loop without a model.

Each call is stateless: the brain gets the goal, the plan, a short history of what happened, the
observation and (if its arm lets it see) the screenshot. Context stays bounded however long the
task runs, and every step can be replayed from the episode log.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .intent import BadIntent, Intent, parse_intent

HISTORY_STEPS = 15


@dataclass
class ArmView:
    """What a brain is shown and allowed to say, per arm (router.ARMS)."""

    sees_tree: bool
    sees_image: bool
    how: str  # the part of the prompt that says how to point at things


HOW_PAPER = """\
To act on something: if the element list has it WITH A NAME, use "element": "<id>". If it is not in the
list, or its entry has no name, describe it in "target" ("the floppy-disk save icon in the toolbar",
"the + button next to Music") -- a trained grounder finds it on the screen. Never give x, y for a click;
for drawing (drag) you may give x, y / to_x, to_y in screenshot pixels, or describe both ends with
"target" and "to_target"."""

ARM_VIEWS = {
    "paper": ArmView(True, True, HOW_PAPER),
    "paper_blind": ArmView(True, False, HOW_PAPER.replace(
        " for drawing (drag) you may give x, y / to_x, to_y in screenshot pixels, or describe both ends with",
        " you cannot see the screen, so for drag describe both ends with")),
    "vision": ArmView(False, True, """\
To act on something, give "x" and "y": the pixel in the screenshot (its size is stated) to click. For
drag also give "to_x", "to_y"."""),
    "tree": ArmView(True, False, """\
To act on something, use "element": "<id>" of a NAMED element in the list, or put its exact name in
"target". Elements without a name cannot be used."""),
    "both": ArmView(True, True, """\
To act on something: a NAMED element from the list ("element": "<id>"), or "x", "y" in screenshot
pixels read off the screenshot. Elements without a name cannot be used by id."""),
}

SYSTEM = """\
You are the reasoning of a computer-use agent. Each turn you see the goal, the plan, what happened so
far and the current screen; you answer with ONE JSON object and nothing else:

{"thought": "<one sentence: what you see and why this action>", "action": "<action>", ...fields}

actions:
  plan    "steps": ["...", ...]          2-8 short checkpoints, in order. Do this first.
  click | double_click | right_click     on a place (see below)
  type    "text": "..."                  into the focused field, or add a place to click into first
  key     "keys": "ctrl+s"               shortcuts, enter, tab, escape
  scroll  "direction": "up"|"down"       optionally at a place
  drag    from a place to another        drawing, sliders
  wait    "seconds": 1-10
  check   "step": <n>, "evidence": {"kind": "text"|"url"|"file"|"changed", "value": "..."}
          a plan step counts as done only when its check is verified by the system
  done    "claim": "what was achieved"   only when every step is verified, or say why it cannot be done

{how}

Rules: one action per turn. Prefer keys when the app takes them. After each action you get the new
screen: if it did not change as expected, try another way instead of repeating. Never type passwords,
never pay or buy; sending, deleting and posting wait for the person's approval."""


@dataclass
class View:
    goal: str
    arm: str
    plan: list[str]
    done_steps: set[int]
    history: list[str]
    note: str  # the result of the last action / why the last reply was refused
    observation: str  # the element list + window/page header
    screenshot: bytes = b""  # JPEG
    size: tuple[int, int] = (0, 0)


def prompt_text(view: View) -> str:
    arm = ARM_VIEWS[view.arm]
    plan = ("\n".join(f"  {'[x]' if i in view.done_steps else '[ ]'} {i}. {s}" for i, s in enumerate(view.plan, 1))
            if view.plan else "  (no plan yet: make one first)")
    history = "\n".join(view.history[-HISTORY_STEPS:]) or "  (nothing yet)"
    parts = [f"Goal: {view.goal}", f"Plan:\n{plan}", f"So far:\n{history}"]
    if view.note:
        parts.append(f"Last result: {view.note}")
    if arm.sees_tree:
        parts.append(f"Screen (accessibility tree):\n{view.observation}")
    elif view.size[0]:
        parts.append(f"Screen: the screenshot below, {view.size[0]}x{view.size[1]} pixels.")
    if arm.sees_image and view.screenshot:
        parts.append("The screenshot is attached.")
    parts.append("Answer with one JSON object.")
    return "\n\n".join(parts)


def system_prompt(arm: str) -> str:
    return SYSTEM.replace("{how}", ARM_VIEWS[arm].how)


@dataclass
class Decision:
    intent: Optional[Intent]
    raw: str
    tokens: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""  # why there is no intent (bad reply, the brain process failed)


class Brain(Protocol):
    name: str

    def decide(self, view: View) -> Decision: ...


def _decide(reply: str, tokens: dict[str, int], seconds: float) -> Decision:
    try:
        return Decision(parse_intent(reply), reply, tokens, seconds)
    except BadIntent as e:
        return Decision(None, reply, tokens, seconds, f"your reply was not a usable action: {e}")


# --- Claude Code, headless, as a pure brain ----------------------------------------------------

def claude_brain_command(arm: str, model: Optional[str] = None, exe: Optional[str] = None) -> list[str]:
    from ..claude_brain import claude_exe

    cmd = [exe or claude_exe(), "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--system-prompt", system_prompt(arm), "--tools", "", "--strict-mcp-config", "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    return cmd


def claude_message(view: View) -> str:
    """One stream-json user message: the text, plus the screenshot as an image block if the arm sees."""
    content = [{"type": "text", "text": prompt_text(view)}]
    if ARM_VIEWS[view.arm].sees_image and view.screenshot:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.b64encode(view.screenshot).decode()}})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}})


def read_result(stdout: str) -> tuple[Optional[str], dict[str, int], str]:
    """(final text, token usage, error) from claude's stream-json output."""
    text, usage, error = None, {}, ""
    for line in stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") != "result":
            continue
        if msg.get("is_error") or msg.get("subtype") != "success":
            error = str(msg.get("result") or msg.get("subtype") or "error")[:300]
        text = msg.get("result") or ""
        u = msg.get("usage") or {}
        usage = {k: int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                   "cache_read_input_tokens", "output_tokens")}
    return text, usage, error


@dataclass
class ClaudeCodeBrain:
    arm: str = "paper"
    model: Optional[str] = None
    timeout: float = 180.0
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run  # tests pass a fake
    name: str = "claude"

    def decide(self, view: View) -> Decision:
        start = time.time()
        workdir = tempfile.mkdtemp(prefix="cu-brain-")  # no project CLAUDE.md, nothing to read
        try:
            done = self.run(claude_brain_command(self.arm, self.model), input=claude_message(view) + "\n",
                            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=workdir,
                            timeout=self.timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return Decision(None, "", {}, time.time() - start, f"the brain took longer than {self.timeout:.0f}s")
        text, usage, error = read_result(done.stdout or "")
        seconds = time.time() - start
        if text is None:
            tail = (done.stderr or "").strip().splitlines()[-1:] or [f"exit {done.returncode}"]
            return Decision(None, "", usage, seconds, f"claude gave no result ({tail[0][:200]})")
        if error:
            return Decision(None, text, usage, seconds, f"claude reported an error: {error}")
        return _decide(text, usage, seconds)


# --- any OpenAI-compatible chat model (Groq) ---------------------------------------------------

@dataclass
class ChatBrain:
    chat: object  # brain.chat.OpenAICompatibleChat (or anything with .complete(messages))
    arm: str = "paper_blind"
    name: str = "chat"

    def __post_init__(self) -> None:
        if ARM_VIEWS[self.arm].sees_image:
            raise ValueError(f"ChatBrain is text only; arm {self.arm!r} needs a model that sees images")
        self.name = getattr(self.chat, "name", self.name)

    def decide(self, view: View) -> Decision:
        start = time.time()
        messages = [{"role": "system", "content": system_prompt(self.arm)},
                    {"role": "user", "content": prompt_text(view)}]
        try:
            reply, usage = self.chat.complete(messages)
        except Exception as e:  # noqa: BLE001 -- a failed call is a failed step, not a crashed run
            return Decision(None, "", {}, time.time() - start, f"the model call failed ({str(e)[:200]})")
        tokens = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
        return _decide(reply, tokens, time.time() - start)


# --- scripted ----------------------------------------------------------------------------------

@dataclass
class ScriptedBrain:
    """Replies from a list; each may be a JSON string or a function of the View (to react to it)."""

    replies: list
    name: str = "scripted"
    seen: list = field(default_factory=list)

    def decide(self, view: View) -> Decision:
        self.seen.append(view)
        if not self.replies:
            return _decide('{"action": "done", "claim": "(script ran out)"}', {}, 0.0)
        reply = self.replies.pop(0)
        return _decide(reply(view) if callable(reply) else reply, {}, 0.0)


def make_brain(kind: str, arm: str, model: Optional[str] = None) -> Brain:
    if kind == "claude":
        return ClaudeCodeBrain(arm=arm, model=model or os.environ.get("CU_CLAUDE_MODEL"))
    if kind == "groq":
        from ..brain.chat import GROQ_DEFAULT_MODEL, OpenAICompatibleChat

        return ChatBrain(OpenAICompatibleChat.groq(model or GROQ_DEFAULT_MODEL, timeout=90.0), arm=arm)
    raise ValueError(f"unknown brain {kind!r} (claude | groq)")
