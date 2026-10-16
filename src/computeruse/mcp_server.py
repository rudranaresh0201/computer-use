"""
`computeruse mcp`: the desktop as an MCP server (stdio). Any MCP host -- Claude Code, Claude
Desktop, Cursor -- becomes the reasoning; this server is the eyes and hands, and the rules.

    claude -p "Draw a house in Paint" --mcp-config '{"mcpServers":{"computeruse":{"command":"computeruse","args":["mcp"]}}}'

The tools are thin: every rule lives in desktop_session.py (tested without a screen). All
Windows work runs on ONE worker thread: UI Automation is COM, and COM objects belong to the
thread that created them, while the SDK runs tools wherever it likes.
If CU_EVENTS names a file, steps / receipts / screens are appended to it for the app window.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

INSTRUCTIONS = """\
You operate desktop apps and websites for the person, through these tools only. targets() lists them.

Loop: open(target) -> plan(steps) -> observe() -> act(...) -> observe() ... -> check(step, ...) for every step.

How to act, in this order of preference:
1. Keys: shortcuts and typing into the focused app (act action="key"/"type"): exact, no pixels.
2. Named elements from observe()'s list (act element="u12"): exact. Elements without a name are refused.
3. Coordinates read off the screenshot (act x=, y=): for canvases and unnamed icons. Drawing is drag.

Observe after every action or two; never act on a stale picture. A plan step counts as done only when
check() verifies it with evidence (a saved file, text in the window, a changed screen). Say what you are
about to do in one short sentence before each action. If something refuses or fails twice, try another way."""


def build_server(session=None):
    from mcp.server.mcpserver import Image, MCPServer

    from .session import file_events

    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="desktop")
    state = {"session": session}

    def on_worker(fn, *args, **kwargs):
        def call():
            if state["session"] is None:  # made on the worker thread, so COM lives there
                import comtypes

                comtypes.CoInitialize()
                state["session"] = build_session(file_events(os.environ.get("CU_EVENTS")))
            return fn(state["session"], *args, **kwargs)

        return asyncio.get_running_loop().run_in_executor(worker, call)

    server = MCPServer("computeruse", instructions=INSTRUCTIONS)

    @server.tool(description="List the desktop apps and websites this server can drive.")
    async def targets() -> str:
        return await on_worker(lambda s: s.targets())

    @server.tool(description="Open (or bring to the front) a target from targets(), e.g. 'paint' or 'sheets'. "
                             "Call it first, and again to switch to another app or site.")
    async def open(target: str) -> str:  # noqa: A001 -- the tool's name is what the model sees
        return await on_worker(lambda s: s.open(target))

    @server.tool(description="Record the task as 2-8 short checkpoints, in order (e.g. 'Draw the house', "
                             "'Save as house.png'). Each one is later proven with check().")
    async def plan(steps: list[str]) -> str:
        return await on_worker(lambda s: s.set_plan(steps))

    @server.tool(description="See the app now: a screenshot plus its accessibility elements, numbered (u12). "
                             "x, y in act() are pixels of THIS screenshot.")
    async def observe() -> list:
        text, jpeg = await on_worker(lambda s: s.observe())
        return [text, Image(data=jpeg, format="jpeg")] if jpeg else [text]

    @server.tool(description=(
        "Do one thing. action: click | double_click | right_click | drag | type | key | scroll | wait.\n"
        "Target with element='u12' (a NAMED element from observe) or x, y (screenshot pixels).\n"
        "drag: x, y -> to_x, to_y. type: text (into element if given, else the focused field). "
        "key: keys like 'ctrl+s' or 'enter'. scroll: direction up/down, amount. wait: seconds."))
    async def act(action: str, element: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None,
                  to_x: Optional[int] = None, to_y: Optional[int] = None, text: Optional[str] = None,
                  keys: Optional[str] = None, direction: str = "down", amount: int = 3, seconds: float = 1.0) -> str:
        return await on_worker(lambda s: s.act(action, element, x, y, to_x, to_y, text, keys, direction, amount, seconds))

    @server.tool(description=(
        "Prove a plan step (1-based) is done. kind: 'file' (value = file name or path; it must exist, be "
        "non-empty and freshly saved), 'text' (value must appear in the window), or 'changed' (the screen "
        "changed since this step started). Only a verified check ticks the step."))
    async def check(step: int, kind: str, value: str = "", min_bytes: int = 1) -> str:
        return await on_worker(lambda s: s.check(step, kind, value, min_bytes))

    return server


def build_session(events):
    from .desktop_session import real_desktop
    from .session import Session

    desktop = real_desktop()
    return Session(surfaces={name: desktop for name in desktop.apps}, events=events)


def main() -> None:
    build_server().run("stdio")
