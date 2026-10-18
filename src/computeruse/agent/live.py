"""
`computeruse agent`: our loop on a real website, in the agent's own signed-in Chrome (web/cdp.py),
with the panel open for watching and recording.

    computeruse agent "Draw a diagram: boxes Brain, Router, Eyes; arrows Brain->Router->Eyes" \\
        --site excalidraw --brain claude --arm paper --eyes remote \\
        --check excalidraw --expect "Brain,Router,Eyes" --connect "Brain>Router,Router>Eyes"

The brain only decides (one JSON intent a turn). Perceiving, the paper's routing rules, our
grounder, the hands, the approval gate and the checker are ours. Every run is saved to
runs/agent/live/<stamp>.json (full trace) and runs/agent/live/<stamp>.events.jsonl (what the panel
showed, for re-rendering the video's captions later).
"""

from __future__ import annotations

import json
import time
import webbrowser
from pathlib import Path
from typing import Optional

from .checks import (circuit_fresh, circuit_outro, excalidraw_checker, excalidraw_fresh, tinkercad_checker,
                     tinkercad_fresh)
from .loop import Agent
from .router import Router

OUT = Path("runs/agent/live")


def console_approve(reason: str) -> bool:
    try:
        return input(f"\n  APPROVE? {reason} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_live(goal: str, site: Optional[str] = None, url: Optional[str] = None, brain_kind: str = "claude",
             arm: str = "paper", eyes_kind: str = "remote", model: Optional[str] = None,
             check: Optional[str] = None, expect: list[str] = (), connect: list[tuple[str, str]] = (),
             fresh: bool = True, panel: bool = True, max_turns: int = 40, hold: bool = True):
    from ..session import Session
    from ..web.cdp import AgentChrome
    from ..web_surface import SITES, WebSurface
    from .brains import make_brain

    if bool(site) == bool(url):
        raise SystemExit("give --site or --url")
    name = site or "page"
    sites = {name: SITES[site]} if site else {name: (url, f"the web page at {url}")}

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = (OUT / f"{stamp}.events.jsonl").open("w", encoding="utf-8")
    view = None
    if panel:
        from .panel import Panel

        view = Panel().start()
        webbrowser.open(view.url)
        print(f"panel: {view.url}")

    def events(m: str, p: dict) -> None:
        if m != "screen":  # screenshots are large; the trace keeps the rest
            log.write(json.dumps({"t": round(time.time(), 2), "event": m, **p}) + "\n")
            log.flush()
        if view:
            view(m, p)

    eyes = None
    if eyes_kind == "remote":
        from ..eyes_remote import RemoteEyes

        eyes = RemoteEyes.from_env()
        print(f"eyes: {eyes.url} {eyes.health()}")

    chrome = AgentChrome()
    surface = WebSurface(sites, page=chrome.page, settle=lambda: time.sleep(0.5))
    session = Session(surfaces={name: surface}, events=events, approve=console_approve, eyes=eyes)
    session.open(name)
    checker, outro = None, None
    if check == "excalidraw":
        if fresh:
            excalidraw_fresh(chrome.page())
        checker = excalidraw_checker(chrome.page, list(expect), list(connect))
    elif check == "tinkercad":
        if fresh:
            tinkercad_fresh(chrome.page())
        checker = tinkercad_checker(chrome.page, OUT / stamp)
    elif check == "circuit":
        if fresh:
            circuit_fresh(chrome.page())
        outro = circuit_outro  # after the verdict: the schematic and the parts list, for the video
    elif check:
        raise SystemExit(f"unknown --check {check!r} (excalidraw, tinkercad, circuit)")
    session.shot = None  # the fresh page replaced what open() saw

    if not goal:
        if not view:
            raise SystemExit("give a goal, or keep the panel on to type one there")
        print("type the goal into the panel ...")
        goal = view.ask("Build a circuit where a coin-cell battery lights a red LED, then run the simulation.")

    brain = make_brain(brain_kind, arm, model)
    router = Router(arm, (lambda d: session.ground_with_eyes(d)) if eyes else None)
    agent = Agent(session, brain, router, goal, checker=checker, max_turns=max_turns)
    ep = agent.run()
    ep.target = url or site
    path = OUT / f"{stamp}.json"
    path.write_text(json.dumps(ep.to_json(), indent=2), encoding="utf-8")
    log.close()
    if outro:
        try:
            outro(chrome.page())
        except Exception as e:  # noqa: BLE001 -- the outro is for the video; the result is already saved
            print(f"outro skipped: {e.__class__.__name__}: {str(e)[:120]}")

    print(f"\n{'PASSED' if ep.passed else 'NOT PASSED'}  status={ep.status}  actions={sum(1 for t in ep.turns if t.channel)}"
          f"  channels={ep.channels()}  {ep.seconds:.0f}s")
    print(f"brain says: {ep.claim or '(no claim)'}")
    if ep.checker:
        print(f"checker:    {ep.checker[1]}")
    print(f"trace: {path}")
    if view and hold:
        try:
            input("\nThe panel stays up for the recording. Press Enter to close it.")
        except EOFError:
            pass
        view.stop()
    chrome.close()
    return ep
