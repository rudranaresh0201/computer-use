"""
The agent's own Chrome: a real Chrome with a dedicated profile, attached over CDP.

Why not let Playwright launch Chrome (as web/browser.py does)? Google and Autodesk refuse to sign
in inside a browser that automation started ("This browser or app may not be secure"). So:

    1. `computeruse chrome` starts an ordinary Chrome on a profile of its own
       (%LOCALAPPDATA%\\computeruse\\chrome -- never the person's everyday profile), with
       --remote-debugging-port on 127.0.0.1. The person signs in once, by hand.
    2. The MCP server attaches to that Chrome with Playwright's connect_over_cdp and drives a tab.
       Sign-ins live in the profile and survive restarts.

Chrome refuses remote debugging on the default profile, which is the right rule: the debugging
port can drive whatever that profile is signed into, so it must be a profile made for the agent.
The port listens on localhost only; while that Chrome runs, local programs could attach to it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

PORT = 9333  # not 9222, which other debugging tools use by default
PROFILE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "computeruse" / "chrome"
CHROME_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
)


def chrome_path() -> str:
    for p in CHROME_PATHS:
        if Path(p).is_file():
            return p
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found
    raise FileNotFoundError("Chrome is not installed where expected")


def endpoint(port: int = PORT) -> Optional[str]:
    """The DevTools websocket URL if a Chrome is listening on this port, else None."""
    import requests

    try:
        r = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1)
        return r.json().get("webSocketDebuggerUrl") if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def launch(port: int = PORT, profile: Path = PROFILE, urls: tuple[str, ...] = (), headless: bool = False,
           wait: float = 20.0) -> subprocess.Popen:
    """Start the agent's Chrome and wait until its DevTools port answers."""
    profile.mkdir(parents=True, exist_ok=True)
    args = [chrome_path(), f"--remote-debugging-port={port}", "--remote-debugging-address=127.0.0.1",
            f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check", "--window-size=1366,900"]
    if headless:
        args.append("--headless=new")
    proc = subprocess.Popen(args + list(urls or ("about:blank",)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait
    while time.time() < deadline:
        if endpoint(port):
            return proc
        if proc.poll() is not None and not endpoint(port):
            # Chrome hands the launch to an instance already running this profile, then exits
            time.sleep(0.5)
            if endpoint(port):
                return proc
        time.sleep(0.3)
    raise TimeoutError(f"Chrome did not open its debugging port {port} within {wait:.0f}s "
                       "(another Chrome may be using the agent profile without the port: close it and retry)")


class AgentChrome:
    """Attach to (or start) the agent's Chrome and hand out the tab the agent works in.
    Sync Playwright: create and use it on ONE thread (the MCP server's worker thread)."""

    def __init__(self, port: int = PORT, profile: Path = PROFILE, headless: bool = False) -> None:
        self.port, self.profile, self.headless = port, profile, headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self.proc: Optional[subprocess.Popen] = None

    def page(self):
        if self._page is not None and not self._page.is_closed() and self._browser.is_connected():
            return self._page
        self._connect()
        context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        pages = [p for p in context.pages if not p.is_closed()]
        # the agent's tab: the last ordinary page, or a fresh one -- never DevTools or an extension page
        normal = [p for p in pages if p.url.startswith(("http", "about:", "data:"))]
        self._page = normal[-1] if normal else context.new_page()
        return self._page

    def _connect(self) -> None:
        if self._browser is not None and self._browser.is_connected():
            return
        if not endpoint(self.port):
            self.proc = launch(self.port, self.profile, headless=self.headless)
        from playwright.sync_api import sync_playwright

        if self._playwright is None:
            self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}")

    def close(self, quit_chrome: bool = False) -> None:
        """Detach. The person's signed-in Chrome keeps running unless quit_chrome."""
        if self._browser is not None:
            if quit_chrome:
                self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        if quit_chrome and self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
