"""
One goal, start to finish, on any target: the shared runner behind `computeruse agent` and the app
window's sidecar (`computeruse sidecar --brain loop`).

    goal ──► pick the target (one short LLM call, or the person's choice)
         ──► open it, and start it clean when we know how (a new Tinkercad design or circuit, an empty board)
         ──► our loop (loop.py): the brain decides, the router grounds, the session acts and verifies
         ──► the target's checker gives the verdict, read from the application, never from the brain

Targets are the Windows apps of desktop_run.APPS (UI Automation + screenshots, the real mouse) and the
websites of web_surface.SITES (the agent's own Chrome, the page's own input). Both go through one
Session, so the approval gate, the change check and the channel report are the same everywhere.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .checks import (circuit_fresh, circuit_outro, excalidraw_checker, excalidraw_fresh, tinkercad_checker,
                     tinkercad_fresh)
from .loop import Agent, Episode
from .router import Router

OUT = Path("runs/agent/live")
CIRCUIT_WORDS = re.compile(r"\b(circuit|led|battery|resistor|arduino|breadboard|wire|simulat\w*|pcb)\b", re.I)


def target_list() -> dict[str, str]:
    """Every target the agent can open: name -> what it is for."""
    from ..desktop_run import APPS
    from ..web_surface import SITES

    apps = {name: f"Windows app: {title}" for name, (_, title) in APPS.items()}
    return apps | {name: desc for name, (_, desc) in SITES.items()}


def guess_target(goal: str, targets: dict[str, str]) -> Optional[str]:
    """The target named in the goal itself ("... in Paint", "a Tinkercad circuit"), if any."""
    low = goal.lower()
    for name in targets:
        spoken = re.escape(name.replace("_pp", "++").replace("_", " "))
        if re.search(rf"(?<!\w){spoken}(?!\w)", low) or name in low:  # lookarounds: \b fails after "++"
            return name
    if CIRCUIT_WORDS.search(goal):
        return "tinkercad" if "tinkercad" in targets else None
    return None


def pick_target(goal: str, brain_kind: str, model: Optional[str] = None,
                targets: Optional[dict[str, str]] = None) -> Optional[str]:
    """The goal's own words first; otherwise one short question to the brain's model."""
    from .brains import ask_text

    targets = targets or target_list()
    named = guess_target(goal, targets)
    if named:
        return named
    menu = "\n".join(f"  {k}: {v}" for k, v in targets.items())
    reply = ask_text(brain_kind, f"Goal: {goal}\n\nWhich ONE of these targets is needed to do it?\n{menu}\n\n"
                                 "Answer with only the target's name, or NONE.", model)
    word = re.findall(r"[a-z_+]+", reply.lower())
    return next((w for w in word if w in targets), None)


@dataclass
class Prepared:
    checker: Optional[Callable[[], tuple[bool, str]]] = None
    outro: Optional[Callable[[Any], None]] = None
    mode: str = ""
    hint: str = ""  # how this app works, the way a person would be told before using it


CIRCUIT_HINT = """How Tinkercad Circuits works: the white area on the left is the board. To add a part, DRAG its tile
from the components panel on the right onto the board: "target" = the tile (e.g. "the Coin Cell 3V Battery tile"),
and give "to_x", "to_y" yourself for where it lands: a point on the white board, well below the toolbar. Empty
places have nothing to find, so always give x, y for them. Put the two parts side by side with space between.
Do not use Zoom To Fit until both parts are on the board (with one part it zooms that part to fill the view).
To wire, drag from one terminal (the small metal end of a lead or pin) to the other: give x, y and to_x, to_y
yourself from the screenshot. "Start Simulation" is at the top right; a lit LED glows."""
TINKERCAD_3D_HINT = """How Tinkercad 3D works: the blue grid is the workplane. To add a shape, DRAG its tile from the
shapes panel on the right onto the workplane; dropping a shape onto another one puts it on top."""


def prepare(target: str, goal: str, page: Optional[Callable[[], Any]], folder: Path, fresh: bool = True,
            expect: list[str] = (), connect: list[tuple[str, str]] = ()) -> Prepared:
    """Start the target clean and choose its checker, when we know one for it."""
    if target == "tinkercad" and page:
        if CIRCUIT_WORDS.search(goal):
            if fresh:
                circuit_fresh(page())
            return Prepared(None, circuit_outro, "circuit", CIRCUIT_HINT)
        if fresh:
            tinkercad_fresh(page())
        return Prepared(tinkercad_checker(page, folder), None, "3d", TINKERCAD_3D_HINT)
    if target == "excalidraw" and page:
        if fresh:
            excalidraw_fresh(page())
        return Prepared(excalidraw_checker(page, list(expect), list(connect)) if expect or connect else None, None,
                        "board")
    return Prepared()


def make_session(events: Callable[[str, dict], None], approve: Callable[[str], bool], eyes=None,
                 extra_sites: Optional[dict[str, tuple[str, str]]] = None):
    """Desktop apps and websites behind one Session; the agent's Chrome starts only when a site is used."""
    from ..desktop_session import real_desktop
    from ..session import Session
    from ..web.cdp import AgentChrome
    from ..web_surface import SITES, WebSurface

    desktop = real_desktop()
    chrome = AgentChrome()
    sites = SITES | (extra_sites or {})
    web = WebSurface(sites, page=chrome.page, settle=lambda: time.sleep(0.5))
    surfaces = {name: desktop for name in desktop.apps} | {name: web for name in sites}
    return Session(surfaces=surfaces, events=events, approve=approve, eyes=eyes), chrome


LEFT_SHARE = 0.70  # CU_LAYOUT=split: the target on the left 70%, the app window (main.js) on the right


def place_target(target: str, web: bool, chrome) -> None:
    """For recording: put the target's window on the left of the screen, next to the app window."""
    import os

    if os.environ.get("CU_LAYOUT") != "split":
        return
    try:
        if web:
            page = chrome.page()
            w, h = page.evaluate("[screen.availWidth, screen.availHeight]")  # CSS px, the unit CDP bounds use
            cdp = page.context.new_cdp_session(page)
            wid = cdp.send("Browser.getWindowForTarget")["windowId"]
            cdp.send("Browser.setWindowBounds", {"windowId": wid, "bounds": {"windowState": "normal"}})
            cdp.send("Browser.setWindowBounds", {"windowId": wid, "bounds": {"left": 0, "top": 0,
                                                                            "width": int(w * LEFT_SHARE), "height": h}})
            return
        import ctypes
        from ctypes import wintypes

        from pywinauto import Desktop

        from ..desktop_run import APPS

        rect = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
        title = APPS[target][1].lower()
        for win in Desktop(backend="uia").windows(visible_only=True):
            if title in (win.window_text() or "").lower():
                ctypes.windll.user32.ShowWindow(win.handle, 9)  # SW_RESTORE: a maximised window ignores moves
                ctypes.windll.user32.MoveWindow(win.handle, rect.left, rect.top,
                                                int((rect.right - rect.left) * LEFT_SHARE), rect.bottom - rect.top, True)
                return
    except Exception:  # noqa: BLE001 -- a layout nicety must never stop a run
        pass


def eyes_from_env():
    """Our grounder on a GPU if CU_EYES_URL / CU_EYES_TOKEN are set (eyes_remote.py), else none."""
    import os

    if not (os.environ.get("CU_EYES_URL") and os.environ.get("CU_EYES_TOKEN")):
        return None
    from ..eyes_remote import RemoteEyes

    return RemoteEyes.from_env()


def split_goal(goal: str, targets: dict[str, str]) -> list[tuple[str, Optional[str]]]:
    """ "Build the LED circuit in Tinkercad, then open Calculator and work out the resistor" -> one part per
    target, in order. A "then" that stays in the same app ("... then run the simulation") stays in its part."""
    pieces = [p.strip(" ,.") for p in re.split(r",?\s+(?:and\s+)?then\s+", goal, flags=re.I) if p.strip(" ,.")]
    parts: list[tuple[str, Optional[str]]] = []
    for piece in pieces:
        t = guess_target(piece, targets)
        if parts and (t is None or t == parts[-1][1]):
            parts[-1] = (f"{parts[-1][0]}, then {piece}", parts[-1][1])
        else:
            parts.append((piece, t))
    return parts if len(parts) > 1 else [(goal, None)]


def run_goal(goal: str, target: Optional[str], events: Callable[[str, dict], None], brain_kind: str = "claude",
             arm: str = "paper", model: Optional[str] = None, eyes: Any = "env",
             approve: Callable[[str], bool] = lambda reason: False, should_stop: Callable[[], bool] = lambda: False,
             fresh: bool = True, max_turns: int = 40, expect: list[str] = (), connect: list[tuple[str, str]] = (),
             session_factory: Callable[..., Any] = make_session,
             extra_sites: Optional[dict[str, tuple[str, str]]] = None, brain=None,
             targets: Optional[dict[str, str]] = None) -> tuple[Optional[Episode], dict]:
    """Run a goal; one that spans apps ("..., then open Calculator and ...") runs as parts, in order, each
    told what the earlier parts found, and passes only if every part passes."""
    parts = split_goal(goal, targets or target_list()) if target is None else [(goal, target)]
    kw = dict(brain_kind=brain_kind, arm=arm, model=model, eyes=eyes, approve=approve, should_stop=should_stop,
              fresh=fresh, max_turns=max_turns, expect=expect, connect=connect, session_factory=session_factory,
              extra_sites=extra_sites, brain=brain)
    if len(parts) == 1:
        return _run_one(parts[0][0], parts[0][1], events, **kw)
    start, found, summaries, ep = time.time(), [], [], None
    for i, (part, part_target) in enumerate(parts, 1):
        if should_stop():
            break
        events("thought", {"text": f"Part {i} of {len(parts)}: {part}"})
        told = part + (f" (Earlier parts of this task found: {'; '.join(found)})" if found else "")
        ep, s = _run_one(told, part_target, events, **kw)
        summaries.append(s)
        found.append(s.get("answer", ""))
        if not s.get("passed", s.get("status") == "done"):
            break
    passed = len(summaries) == len(parts) and all(s.get("passed", s.get("status") == "done") for s in summaries)
    answer = " | ".join(f"{i}. {s.get('answer', '')}" for i, s in enumerate(summaries, 1))
    return ep, {"status": "done" if passed else "stopped", "answer": answer, "passed": passed,
                "actions": sum(s.get("actions", 0) for s in summaries), "seconds": round(time.time() - start, 1),
                "tokens": sum(s.get("tokens", 0) for s in summaries), "brain": summaries[-1].get("brain", "") if summaries else "",
                "parts": summaries, "trace": summaries[-1].get("trace", "") if summaries else ""}


def _run_one(goal: str, target: Optional[str], events: Callable[[str, dict], None], brain_kind: str = "claude",
             arm: str = "paper", model: Optional[str] = None, eyes: Any = "env",
             approve: Callable[[str], bool] = lambda reason: False, should_stop: Callable[[], bool] = lambda: False,
             fresh: bool = True, max_turns: int = 40, expect: list[str] = (), connect: list[tuple[str, str]] = (),
             session_factory: Callable[..., Any] = make_session,
             extra_sites: Optional[dict[str, tuple[str, str]]] = None, brain=None) -> tuple[Optional[Episode], dict]:
    """One target: open it, start it clean, run our loop, return the episode and a summary for the window."""
    from .brains import make_brain

    start = time.time()
    events("planning", {"goal": goal})
    target = target or pick_target(goal, brain_kind, model)
    if not target:
        return None, {"status": "stopped", "answer": "STOPPED: no target fits this goal", "actions": 0}
    eyes = eyes_from_env() if eyes == "env" else eyes
    session, chrome = session_factory(events, approve, eyes, extra_sites)
    try:
        opened = session.open(target)
        events("started", {"goal": goal, "app": target})
        folder = OUT / time.strftime("%Y%m%d-%H%M%S")
        web = getattr(session.surface, "kind", "") == "web"
        # set the app up at full width, then move it: a narrow Tinkercad dashboard hides "Create" in a menu
        prep = prepare(target, goal, chrome.page if web else None, folder, fresh, expect, connect)
        place_target(target, web, chrome)
        if target == "tinkercad" and web:
            from .checks import dismiss_tips

            chrome.page().wait_for_timeout(1500)
            dismiss_tips(chrome.page())
        arm = brain_arm(brain_kind, arm)
        brain = brain or make_brain(brain_kind, arm, model)
        router = Router(arm, (lambda d: session.ground_with_eyes(d)) if eyes else None)
        told = f"{goal}\n\n{prep.hint}" if prep.hint else goal
        agent = Agent(session, brain, router, told, checker=prep.checker, should_stop=should_stop, max_turns=max_turns)
        ep = agent.run()
        ep.target = target
        folder.mkdir(parents=True, exist_ok=True)
        import json

        (folder / "episode.json").write_text(json.dumps(ep.to_json(), indent=2), encoding="utf-8")
        if prep.outro and web:
            try:
                prep.outro(chrome.page())
            except Exception as e:  # noqa: BLE001 -- the outro is for the video; the result is saved
                events("thought", {"text": f"(outro skipped: {e.__class__.__name__})"})
        verdict = ep.checker[1] if ep.checker else ""
        answer = ep.claim or opened
        if ep.checker:
            answer = f"{'PASSED' if ep.passed else 'NOT PASSED'}: {verdict}" + (f" (the brain said: {ep.claim})" if ep.claim else "")
        status = "done" if ep.passed else ("stopped" if ep.status in ("stopped", "gave_up", "unverified") else
                                           "error" if ep.status == "error" else "stopped")
        return ep, {"status": status, "answer": answer, "actions": sum(1 for t in ep.turns if t.channel),
                    "seconds": round(time.time() - start, 1), "tokens": sum(ep.tokens().values()),
                    "brain": brain.name, "passed": ep.passed, "checker": verdict, "channels": ep.channels(),
                    "trace": str(folder / "episode.json")}
    finally:
        chrome.close()


def brain_arm(brain_kind: str, arm: str) -> str:
    """A text-only brain cannot be shown the screenshot: it gets the blind version of the paper's rules."""
    return arm if brain_kind == "claude" else ("paper_blind" if arm in ("paper", "paper_blind") else arm)
