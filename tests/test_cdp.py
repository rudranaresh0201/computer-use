"""The agent's Chrome over CDP: start it on a profile of its own, attach, drive a tab, detach.
Headless with a throwaway profile and port, so it never touches the real agent profile."""

import socket

import pytest

from computeruse.web import cdp

pytest.importorskip("playwright")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def agent(tmp_path):
    try:
        cdp.chrome_path()
    except FileNotFoundError:
        pytest.skip("Chrome not installed")
    chrome = cdp.AgentChrome(port=free_port(), profile=tmp_path / "profile", headless=True)
    yield chrome
    chrome.close(quit_chrome=True)


def test_nothing_listens_before_launch():
    assert cdp.endpoint(free_port()) is None


def test_attach_starts_chrome_and_gives_a_working_tab(agent):
    page = agent.page()
    assert cdp.endpoint(agent.port) is not None
    page.goto("data:text/html,<title>hello</title><button>Go</button>")
    assert page.title() == "hello"
    assert agent.page() is page  # the same tab on the next call


def test_a_closed_tab_is_replaced_not_fatal(agent):
    first = agent.page()
    first.goto("data:text/html,<p>one</p>")
    first.context.new_page()  # keep the browser alive with another tab
    first.close()
    assert not agent.page().is_closed()


def test_the_profile_is_the_agents_own(agent, tmp_path):
    agent.page()
    assert (tmp_path / "profile").is_dir() and any((tmp_path / "profile").iterdir())
