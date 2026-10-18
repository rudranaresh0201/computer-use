"""The sandbox suite in a real headless Chrome, through the whole loop: every task's checker fails a
wrong or empty attempt and passes a right one. The brain is scripted and the eyes are an oracle that
reads true positions from the page -- this tests the suite and the plumbing, not any model."""

import json
import re

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from computeruse.agent.bench import run_episode, summarize  # noqa: E402
from computeruse.agent.brains import ScriptedBrain  # noqa: E402
from computeruse.agent.suite import VIEWPORT  # noqa: E402


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Chromium for Playwright ({e.__class__.__name__})")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    pg = browser.new_page(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]})
    yield pg
    pg.close()


def ref(view, name):
    m = re.search(rf'"{re.escape(name)}"[^\n]*\[ref=(\w+)\]', view.observation)
    assert m, f"{name!r} not in:\n{view.observation}"
    return m.group(1)


def j(**kw):
    return json.dumps(kw)


def oracle(page):
    """Where things really are, read from the DOM / the canvas geometry (viewport pixels)."""
    def eyes(jpeg, description):
        d = description.lower()
        if "save" in d:
            p = page.evaluate("(() => { const b = document.querySelector('[data-k=save]').getBoundingClientRect();"
                              " return [b.x + b.width / 2, b.y + b.height / 2]; })()")
        elif "+" in d or "-" in d:
            chan = re.search(r"(music|voice|alerts)", d).group(1).title()
            p = page.evaluate("""(c) => { const cv = document.getElementById('cv').getBoundingClientRect();
                const i = chans.indexOf(c); return [cv.x + (%s), cv.y + 40 + i * 70 + 17]; }""" %
                              ("440" if "+" in d else "140"), chan)
        elif "red" in d or "blue" in d:
            p = page.evaluate("(w) => { const cv = document.getElementById('cv').getBoundingClientRect();"
                              " const q = w === 'red' ? R : B; return [cv.x + q[0], cv.y + q[1]]; }",
                              "red" if "red" in d else "blue")
        else:
            return None, 0.0
        return (round(p[0]), round(p[1])), 0.01
    return eyes


def run(page, task, replies, arm="paper", eyes=True):
    return run_episode(page, task, 1, ScriptedBrain(list(replies)), arm, oracle(page) if eyes else None)


def test_every_checker_fails_an_untouched_page(page):
    for task in ("signup", "note", "mixer", "archive", "connect"):
        ep = run(page, task, [j(action="done", claim="did it")])
        assert not ep.passed and ep.false_success, task  # said done, the page disagrees


def test_signup_by_named_fields(page):
    def fill(view):
        goal = view.goal
        name = re.search(r"Subscribe (.+?) \(", goal).group(1)
        return j(action="type", element=ref(view, "Full name"), text=name)

    def email(view):
        return j(action="type", element=ref(view, "Email address"), text=re.search(r"\((.+?)\)", view.goal).group(1))

    ep = run(page, "signup", [fill, email,
                              lambda v: j(action="click", element=ref(v, "Email me product updates")),
                              lambda v: j(action="click", element=ref(v, "Sign up")),
                              j(action="done", claim="subscribed")])
    assert ep.passed, ep.checker
    assert ep.channels() == {"tree": 4}


def test_note_save_icon_needs_the_eyes_and_ctrl_s_does_not_work(page):
    text_of = lambda v: re.search(r'"(.+)"', v.goal).group(1)  # noqa: E731
    ep = run(page, "note", [lambda v: j(action="type", element=ref(v, "Note"), text=text_of(v)),
                            j(action="key", keys="ctrl+s"), j(action="done", claim="saved")])
    assert not ep.passed  # the shortcut is off: only the icon saves
    ep = run(page, "note", [lambda v: j(action="type", element=ref(v, "Note"), text=text_of(v)),
                            j(action="click", target="the floppy-disk save icon"), j(action="done", claim="saved")])
    assert ep.passed and ep.channels().get("eyes") == 1, ep.checker


def test_mixer_canvas_buttons_through_the_eyes(page):
    def plus(view):
        ch = re.search(r"Set the (\w+) volume", view.goal).group(1)
        return j(action="click", target=f"the + button for {ch}")

    def times(view):  # how many presses: target minus the start value drawn for that channel
        return None

    ep = run(page, "mixer", [plus, plus, plus, j(action="done", claim="set")])
    want = int(re.search(r"volume to (\d+)", ep.goal).group(1))
    vols = page.evaluate("vol")
    ch = re.search(r"Set the (\w+) volume", ep.goal).group(1)
    assert ep.channels() == {"eyes": 3}
    assert ep.passed == (vols[ch] == want)


def test_archive_needs_the_right_one_of_three_same_named_buttons(page):
    def wrong(view):
        return j(action="click", element=re.findall(r'"Archive"[^\n]*\[ref=(\w+)\]', view.observation)[0])

    ep = run(page, "archive", [wrong, j(action="done", claim="archived")])
    person = re.search(r"Archive (.+)'s order", ep.goal).group(1)
    first_row = page.evaluate("document.querySelector('tr').children[1].textContent")
    assert ep.passed == (first_row == person)


def test_connect_by_describing_both_ends(page):
    ep = run(page, "connect", [j(action="drag", target="the red dot", to_target="the blue dot"),
                               j(action="done", claim="drew it")], arm="paper_blind")
    assert ep.passed and ep.channels() == {"eyes": 1}, ep.checker


def test_summary_counts_pass_false_success_and_channels(page):
    rows = [{"task": "a", **run(page, "note", [j(action="done", claim="x")]).to_json()},
            {"task": "b", "passed": True, "false_success": False, "claim": "y", "channels": {"tree": 2},
             "tokens": {"input_tokens": 10}, "actions": 2, "seconds": 3.0, "status": "done"}]
    s = summarize(rows)
    assert s["episodes"] == 2 and s["pass_rate"] == 0.5 and s["false_success"] == 1 and s["channels"] == {"tree": 2}
