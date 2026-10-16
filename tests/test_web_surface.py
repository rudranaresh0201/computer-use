"""The web surface through a Session, on a local page in headless Chrome: the tree channel by name,
the nameless refusal, the vision channel on a canvas the tree can't see, keys, and the checks."""

from __future__ import annotations

from urllib.parse import quote

import pytest

from computeruse.session import Session
from computeruse.web_surface import WebSurface, playwright_keys

pytest.importorskip("playwright")

PAGE = """<html><head><title>Notes</title></head><body style="margin:0">
<label for=t>Title</label><input id=t>
<button onclick="document.getElementById('out').textContent='saved: '+document.getElementById('t').value">Save</button>
<button onclick="document.getElementById('out').textContent='icon pressed'"><svg width=16 height=16></svg></button>
<p id=out></p>
<div role=button style="position:absolute;left:600px;top:20px;width:80px;height:30px" onclick="document.getElementById('out').textContent='SENT'">Send</div>
<canvas id=c width=400 height=200 style="display:block;border:1px solid #000"></canvas>
<p id=hits></p>
<script>
const c = document.getElementById('c'), g = c.getContext('2d');
c.addEventListener('mousedown', e => { g.fillRect(e.offsetX - 5, e.offsetY - 5, 10, 10);
  document.getElementById('hits').textContent = 'canvas hit at ' + e.offsetX + ',' + e.offsetY; });
document.addEventListener('keydown', e => { if (e.ctrlKey && e.key === 's') { e.preventDefault();
  document.getElementById('out').textContent = 'ctrl+s pressed'; } });
</script></body></html>"""
URL = "data:text/html," + quote(PAGE)


@pytest.fixture(scope="module")
def page():
    from computeruse.web.browser import Browser

    try:
        browser = Browser(headless=True).__enter__()
    except Exception as e:  # no Chrome on this machine
        pytest.skip(f"Chrome not available: {e}")
    yield browser.page
    browser.__exit__(None, None, None)


@pytest.fixture
def session(page):
    events = []
    web = WebSurface(sites={"notes": (URL, "a test page")}, page=lambda: page)
    s = Session(surfaces={"notes": web}, events=lambda m, p: events.append((m, p)), sleep=lambda s: None)
    page.goto("about:blank")
    s.open("notes")
    s.events_seen = events
    return s


def ref_of(session, role, name):
    return next(e.ref for e in session.surface.elements.values() if e.role == role and e.name == name)


def test_keys_are_translated_for_playwright():
    assert playwright_keys("ctrl+shift+s") == "Control+Shift+s"
    assert playwright_keys("enter") == "Enter" and playwright_keys("f2") == "F2" and playwright_keys("down") == "ArrowDown"


def test_observe_reads_the_tree_and_returns_the_viewport_screenshot(session):
    text, jpeg = session.observe()
    assert 'button "Save"' in text and "1280x800" in text and jpeg[:2] == b"\xff\xd8"
    assert any(m == "screen" for m, _ in session.events_seen)


def test_typing_into_a_named_field_and_clicking_a_named_button_go_through_the_tree(session):
    session.observe()
    session.act("type", element=ref_of(session, "textbox", "Title"), text="Weekly expenses")
    reply = session.act("click", element=ref_of(session, "button", "Save"))
    assert "changed" in reply
    steps = [p for m, p in session.events_seen if m == "step"]
    assert [p["channel"] for p in steps] == ["tree", "tree"] and 0 < steps[-1]["x"] < 1
    session.set_plan(["Save the title"])
    assert "verified" in session.check(1, "text", "saved: Weekly expenses")


def test_a_nameless_icon_button_is_refused(session):
    session.observe()
    nameless = next(e.ref for e in session.surface.elements.values() if e.role == "button" and not e.name)
    assert "has no name" in session.act("click", element=nameless)


def test_the_canvas_is_reached_by_coordinates_on_the_vision_channel(session):
    text, _ = session.observe()
    assert "canvas hit" not in text
    session.act("click", x=60, y=120)  # inside the canvas: the tree has nothing to name there
    session.set_plan(["Mark the canvas"])
    assert "verified" in session.check(1, "text", "canvas hit at")
    assert [p["channel"] for m, p in session.events_seen if m == "step"][-1] == "vision"


def test_keys_reach_the_page(session):
    session.observe()
    session.act("key", keys="ctrl+s")
    session.set_plan(["Shortcut"])
    assert "verified" in session.check(1, "text", "ctrl+s pressed")
    assert "verified" in session.check(1, "url", "data:text/html")


def test_a_send_button_is_gated_even_when_clicked_by_coordinates(session):
    asked = []
    session.approve = lambda reason: asked.append(reason) or False
    session.observe()
    reply = session.act("click", x=640, y=35)  # the Send button's spot, not its ref
    assert "did not approve" in reply and asked and "Send" in asked[0]
    session.set_plan(["x"])
    assert "NOT verified" in session.check(1, "text", "SENT")  # it really did not happen
    session.approve = lambda reason: True
    session.act("click", element=ref_of(session, "button", "Send"))
    assert "verified" in session.check(1, "text", "SENT")
