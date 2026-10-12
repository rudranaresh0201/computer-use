"""The web agent loop (web/agent.py) and the model client (brain/chat.py),
driven by a scripted model on a local page in headless Chrome -- the whole
loop with no API key and no budget."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from computeruse.brain.chat import OpenAICompatibleChat, ScriptedChat, parse_completion
from computeruse.web.agent import WebAgent, render_observation, risk_of
from computeruse.web.browser import Browser, PageView, parse_snapshot

pytest.importorskip("playwright")

SHOP = """<html><head><title>Tiny shop</title></head><body>
<label for=q>Search products</label><input id=q>
<button onclick="document.getElementById('out').textContent='3 results for '+document.getElementById('q').value">Search</button>
<p id=out></p>
<button onclick="document.getElementById('out').textContent='ORDER PLACED'">Buy now</button>
</body></html>"""


@pytest.fixture(scope="module")
def web():
    try:
        browser = Browser(headless=True).__enter__()
    except Exception as e:
        pytest.skip(f"Chrome not available: {e}")
    yield browser
    browser.__exit__(None, None, None)


@pytest.fixture
def shop(web):
    web.page.set_content(SHOP)
    return web


def ref_of(web, name):
    return next(e.ref for e in web.observe().elements if e.name == name)


# -- the screen the model sees ------------------------------------------------


def test_the_model_only_sees_what_is_on_screen_without_layout_noise():
    snapshot = "\n".join([
        '- generic [ref=e1] [box=0,0,800,3000]:',
        '  - button "Top" [ref=e2] [cursor=pointer] [box=0,10,50,20]',
        '  - paragraph [ref=e3] [box=0,2000,800,40]:',
        '    - text: far below',
        '  - link "Bottom" [ref=e4] [box=0,2900,50,20]:',
        '    - /url: "#x"',
    ])
    view = PageView("http://x", "X", snapshot, parse_snapshot(snapshot))
    top = render_observation(view, (0, 0, 800, 600))
    assert 'button "Top" [ref=e2]' in top and "far below" not in top and "Bottom" not in top
    assert "box=" not in top and "cursor" not in top and "generic" not in top
    scrolled = render_observation(view, (0, 1900, 800, 600))
    assert "far below" in scrolled and "Top" not in scrolled


def test_long_screens_are_cut_with_a_hint_to_scroll():
    snapshot = "\n".join(f'- button "b{i}" [ref=e{i}] [box=0,0,10,10]' for i in range(2000))
    view = PageView("http://x", "X", snapshot, parse_snapshot(snapshot))
    screen = render_observation(view, (0, 0, 800, 600), max_chars=500)
    assert len(screen) < 700 and "scroll to see the rest" in screen


# -- the loop -------------------------------------------------------------------


def test_the_agent_completes_a_task_and_logs_the_episode(shop, tmp_path):
    box, button = ref_of(shop, "Search products"), ref_of(shop, "Search")
    model = ScriptedChat([
        ("type", {"ref": box, "text": "red shoes"}),
        ("click", {"ref": button}),
        ("done", {"answer": "3 results"}),
    ])
    log = tmp_path / "episodes.jsonl"
    episode = WebAgent(shop, model, log_path=log).run("search for red shoes")
    assert episode.status == "done" and episode.answer == "3 results"
    assert [s.action for s in episode.steps] == ["type", "click", "done"]
    assert episode.steps[1].page_changed
    assert "3 results for red shoes" in shop.observe().snapshot
    record = json.loads(log.read_text().splitlines()[0])
    assert record["status"] == "done" and len(record["steps"]) == 3 and record["tokens"] > 0
    # each call sends the goal, the steps so far, and the current screen -- never old screens
    last_prompt = model.seen[-1][1]["content"]
    assert "Goal: search for red shoes" in last_prompt and "2. click" in last_prompt


def test_a_ref_that_is_not_on_screen_is_refused_and_the_agent_continues(shop):
    episode = WebAgent(shop, ScriptedChat([("click", {"ref": "e999"}), ("done", {"answer": "x"})])).run("g")
    assert not episode.steps[0].ok and "not on the current screen" in episode.steps[0].detail
    assert episode.status == "done"


def test_buying_needs_a_persons_yes(shop):
    buy = ref_of(shop, "Buy now")
    denied = WebAgent(shop, ScriptedChat([("click", {"ref": buy}), ("done", {"answer": "x"})]),
                      confirm=lambda reason: False).run("buy it")
    assert "did not approve" in denied.steps[0].detail
    assert "ORDER PLACED" not in shop.observe().snapshot
    asked = []
    WebAgent(shop, ScriptedChat([("click", {"ref": buy}), ("done", {"answer": "x"})]),
             confirm=lambda reason: asked.append(reason) or True).run("buy it")
    assert asked == ["click on 'Buy now'"] and "ORDER PLACED" in shop.observe().snapshot


def test_three_failures_in_a_row_stop_the_episode_as_stuck(shop):
    model = ScriptedChat([("", {}), ("click", {"ref": "e999"}), ("", {}), ("done", {"answer": "never"})])
    assert WebAgent(shop, model).run("g").status == "stuck"


def test_running_out_of_steps_is_recorded(shop):
    model = ScriptedChat([("scroll", {"direction": "down"})] * 5)
    episode = WebAgent(shop, model, max_steps=3).run("g")
    assert episode.status == "max_steps" and len(episode.steps) == 3


def test_risk_rules():
    snapshot = '- textbox "Password" [ref=e1]\n- button "Delete account" [ref=e2]\n- button "Search" [ref=e3]'
    view = PageView("u", "t", snapshot, parse_snapshot(snapshot))
    assert risk_of("type", {"ref": "e1", "text": "x"}, view)
    assert risk_of("click", {"ref": "e2"}, view)
    assert risk_of("click", {"ref": "e3"}, view) is None


# -- the model client -------------------------------------------------------------


def test_parse_completion_reads_tool_calls_json_in_text_and_prose():
    call = parse_completion({"choices": [{"message": {"tool_calls": [{"function": {
        "name": "click", "arguments": '{"ref": "e7"}'}}]}}], "usage": {"prompt_tokens": 900, "completion_tokens": 12}})
    assert (call.name, call.arguments, call.usage["prompt_tokens"]) == ("click", {"ref": "e7"}, 900)
    texted = parse_completion({"choices": [{"message": {"content": 'I will {"tool": "scroll", "arguments": {"direction": "down"}}'}}]})
    assert (texted.name, texted.arguments) == ("scroll", {"direction": "down"})
    assert parse_completion({"choices": [{"message": {"content": "Let me think..."}}]}).name == ""


def test_rate_limits_are_waited_out_not_crashed_on():
    replies = [
        SimpleNamespace(status_code=429, headers={"retry-after": "3"}, text="slow down"),
        SimpleNamespace(status_code=200, headers={}, text="", json=lambda: {
            "choices": [{"message": {"tool_calls": [{"function": {"name": "done", "arguments": '{"answer":"ok"}'}}]}}]}),
    ]
    waited = []
    chat = OpenAICompatibleChat("m", "http://x", "k", sleep=waited.append, post=lambda *a, **k: replies.pop(0))
    assert chat.next_call([], []).name == "done" and waited == [3.0]


def test_a_missing_groq_key_says_how_to_set_it(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="setx GROQ_API_KEY"):
        OpenAICompatibleChat.groq()
