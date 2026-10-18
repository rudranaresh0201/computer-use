"""The MCP layer: the seven tools exist, and calls reach the session on its one worker thread."""

import asyncio
import threading

from computeruse.claude_brain import TOOLS
from computeruse.mcp_server import build_server


class FakeSession:
    def __init__(self):
        self.threads = set()

    def _note(self):
        self.threads.add(threading.get_ident())

    def targets(self):
        self._note()
        return "Desktop apps:\n- paint: draw"

    def open(self, target):
        self._note()
        return f"{target} is open"

    def observe(self):
        self._note()
        return "Window: Paint", b"\xff\xd8jpeg"

    def locate(self, description):
        self._note()
        return f"found {description} at x=3, y=4", b"\xff\xd8zoom"

    def act(self, *args):
        self._note()
        return f"did {args[0]}"


def call(server, name, args):
    return asyncio.run(server.call_tool(name, args))


def test_the_seven_tools_are_registered_and_claude_may_call_each():
    names = {t.name for t in asyncio.run(build_server(session=FakeSession()).list_tools())}
    assert names == {"targets", "open", "plan", "observe", "locate", "act", "check"} == set(TOOLS)


def test_calls_reach_the_session_and_observe_returns_text_and_an_image():
    fake = FakeSession()
    server = build_server(session=fake)
    assert "paint is open" in str(call(server, "open", {"target": "paint"}))
    result = call(server, "observe", {})
    kinds = [c.type for c in result.content]
    assert kinds == ["text", "image"]
    found = call(server, "locate", {"description": "the Brush tool"})
    assert [c.type for c in found.content] == ["text", "image"] and "x=3, y=4" in str(found)
    assert "did click" in str(call(server, "act", {"action": "click", "x": 1, "y": 2}))
    assert len(fake.threads) == 1  # every call on the same worker thread: COM objects stay valid
