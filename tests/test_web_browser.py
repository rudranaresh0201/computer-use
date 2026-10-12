"""Chrome's eyes and hands (web/browser.py), on a local page in headless
Chrome: what the accessibility tree shows, what it can't see, and whether
actions report real effects."""

from __future__ import annotations

import pytest

from computeruse.web.browser import Browser, parse_snapshot
from computeruse.web.shell import run_command

pytest.importorskip("playwright")

SHOP = """<html><head><title>Tiny shop</title></head><body><h1>Shop</h1>
<nav><a href="#home">Home</a> <a href="#cart">Cart</a></nav>
<label for=q>Search products</label><input id=q placeholder="e.g. shoes">
<button onclick="document.getElementById('out').textContent='searched: '+document.getElementById('q').value">Search</button>
<button style="display:none">Hidden</button>
<button onclick="">Does nothing</button>
<p id=out></p>
<div role=button aria-label="Close dialog" tabindex=0>x</div>
<canvas id=c width=200 height=50></canvas>
<script>document.getElementById('c').getContext('2d').fillText('Buy now', 10, 30)</script>
</body></html>"""


@pytest.fixture(scope="module")
def web():
    try:
        browser = Browser(headless=True).__enter__()
    except Exception as e:  # no Chrome on this machine
        pytest.skip(f"Chrome not available: {e}")
    yield browser
    browser.__exit__(None, None, None)


@pytest.fixture
def shop(web):
    web.page.set_content(SHOP)
    return web


def by_name(view, name):
    return next(e for e in view.elements if e.name == name)


def test_parse_snapshot_reads_role_name_ref_and_box():
    snapshot = '- textbox "Search products" [ref=e6] [box=111,84,177,21]:\n  - text: hi\n- button "Say \\"hi\\"" [ref=e7]'
    first, second = parse_snapshot(snapshot)
    assert (first.ref, first.role, first.name, first.box) == ("e6", "textbox", "Search products", (111, 84, 177, 21))
    assert (second.ref, second.name, second.box) == ("e7", 'Say "hi"', None)


def test_refs_after_a_navigation_carry_a_frame_prefix():
    # Seen on en.wikipedia.org after a search: refs became f1e321, not e321.
    (link,) = parse_snapshot('- link "Alan Turing" [ref=f1e321] [cursor=pointer]')
    assert link.ref == "f1e321" and link.interactive


def test_acting_works_on_a_page_reached_by_navigation(web):
    pages = {
        "/first": '<title>First</title><a href="/second">next</a>',
        "/second": "<title>Second</title><button onclick=\"document.title='Clicked'\">Go on</button>",
    }
    web.page.route(
        "http://site.test/**",
        lambda route: route.fulfill(content_type="text/html", body=pages[route.request.url.removeprefix("http://site.test")]),
    )
    assert web.goto("http://site.test/first").ok
    assert web.click(by_name(web.observe(), "next").ref).ok
    second = web.observe()
    assert second.title == "Second"
    button = by_name(second, "Go on")
    assert web.click(button.ref).ok, button.ref
    assert web.page.title() == "Clicked"


def test_the_tree_names_things_the_way_chrome_computes_them(shop):
    view = shop.observe()
    names = {(e.role, e.name) for e in view.interactive()}
    assert ("textbox", "Search products") in names  # the <label> became the input's name
    assert ("button", "Close dialog") in names  # an ARIA div counts as a button
    assert ("link", "Cart") in names
    assert all(e.box is not None for e in view.interactive())  # pixel boxes, like UIA rects
    assert view.title == "Tiny shop"


def test_hidden_elements_and_canvas_drawings_are_invisible_to_the_tree(shop):
    view = shop.observe()
    assert "Hidden" not in [e.name for e in view.elements]
    # "Buy now" is drawn into a <canvas>: on screen, but not in the tree --
    # the web's version of an app UIA can't see.
    assert "Buy now" not in view.snapshot


def test_type_and_click_change_the_page_and_say_so(shop):
    view = shop.observe()
    typed = shop.type(by_name(view, "Search products").ref, "red shoes")
    clicked = shop.click(by_name(view, "Search").ref)
    assert typed.ok and clicked.ok and clicked.page_changed
    assert "searched: red shoes" in shop.observe().snapshot


def test_a_click_with_no_effect_is_reported_as_such(shop):
    result = shop.click(by_name(shop.observe(), "Does nothing").ref)
    assert result.ok and not result.page_changed
    assert "did not change" in result.detail


def test_a_bad_or_missing_ref_fails_fast_with_a_readable_reason(shop):
    shop.observe()
    missing = shop.click("e999")
    assert not missing.ok and missing.detail
    nonsense = shop.click("#q")
    assert not nonsense.ok and "not a ref" in nonsense.detail


def test_the_shell_drives_the_same_browser(shop, capsys):
    view = shop.observe()
    search_box = by_name(view, "Search products").ref
    assert run_command(shop, "look")
    assert run_command(shop, f"type {search_box} blue hat")
    assert run_command(shop, f"click {by_name(view, 'Search').ref}")
    out = capsys.readouterr().out
    assert "Search products" in out and "typed 'blue hat'" in out and "clicked" in out
    assert run_command(shop, "quit") is False
