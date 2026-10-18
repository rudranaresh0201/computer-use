"""
`computeruse agent`: one goal on any target, with the panel open for watching and recording.

    computeruse agent --record                                   # type the goal in the panel; it picks the target
    computeruse agent "Light a red LED from a coin cell, then run the simulation" --site tinkercad
    computeruse agent "Draw a house with a sun in Paint and save it as house.png" --app paint
    computeruse agent "..." --url https://example.com

The goal comes first, then the target opens (runner.py picks it when none is given): prompt, then
the app at work, then the verdict. The brain only decides (one JSON intent a turn). Perceiving, the
paper's routing rules, our grounder, the hands, the approval gate and the checker are ours. Every run
is saved under runs/agent/live/ (episode trace, the panel's events, and with --record the video).
"""

from __future__ import annotations

import json
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional

from .runner import OUT, run_goal

PLACEHOLDER = "Build a circuit in Tinkercad where a coin-cell battery lights a red LED, then run the simulation."


def console_approve(reason: str) -> bool:
    try:
        return input(f"\n  APPROVE? {reason} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def start_recording(path: Path, fps: int = 20):
    """The whole screen to an mp4 (ffmpeg's gdigrab, from the imageio-ffmpeg wheel: no install, no admin).
    Stop it by writing "q" to its stdin, which lets it finish the file properly."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "gdigrab", "-framerate", str(fps), "-draw_mouse", "0",
           "-i", "desktop", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recording(proc) -> None:
    if proc and proc.poll() is None:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()


def run_live(goal: str = "", site: Optional[str] = None, url: Optional[str] = None, app: Optional[str] = None,
             brain_kind: str = "claude", arm: str = "paper", eyes_kind: str = "remote", model: Optional[str] = None,
             expect: list[str] = (), connect: list[tuple[str, str]] = (), fresh: bool = True, panel: bool = True,
             max_turns: int = 40, hold: bool = True, record: bool = False):
    if sum(map(bool, (site, url, app))) > 1:
        raise SystemExit("give at most one of --site, --app, --url")
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

    recorder = None
    if record:
        recorder = start_recording(OUT / f"{stamp}.mp4")
        print(f"recording the screen to {OUT / (stamp + '.mp4')}")
    if not goal:
        if not view:
            raise SystemExit("give a goal, or keep the panel on to type one there")
        print("type the goal into the panel ...")
        goal = view.ask(PLACEHOLDER)

    extra = {"page": (url, f"the web page at {url}")} if url else None
    ep, summary = run_goal(goal, site or app or ("page" if url else None), events, brain_kind, arm, model, eyes,
                           approve=console_approve, fresh=fresh, max_turns=max_turns, expect=list(expect),
                           connect=list(connect), extra_sites=extra)
    log.close()
    if recorder:
        time.sleep(3)  # end on the verdict for a beat
        stop_recording(recorder)
        print(f"video: {OUT / (stamp + '.mp4')}")

    print(f"\n{summary['answer']}")
    if ep:
        print(f"status={ep.status}  actions={summary['actions']}  channels={ep.channels()}  {ep.seconds:.0f}s")
        print(f"trace: {summary['trace']}")
    if view and hold:
        try:
            input("\nThe panel stays up. Press Enter to close it.")
        except EOFError:
            pass
        view.stop()
    return ep
