"""
Record a demo take: the app window docked right (our loop, Claude deciding), the target app or the agent's
Chrome on the left, the whole screen captured to runs/agent/live/demo-<stamp>.mp4.

    .venv\\Scripts\\python scripts\\record_demo.py --delay 45

1. waits --delay seconds (time to minimise everything else on screen),
2. starts the screen recording and opens the window (CU_LAYOUT=split, CU_BRAIN=loop),
3. the person types the goal in the window and presses Run,
4. when the person closes the window after the verdict, the recording stops and the file is kept.

Needs CU_EYES_URL / CU_EYES_TOKEN in the environment for our grounder. Brain: CU_LOOP_BRAIN (claude).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from computeruse.agent.live import start_recording, stop_recording  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=int, default=45)
    ap.add_argument("--brain", default=os.environ.get("CU_LOOP_BRAIN", "claude"))
    ap.add_argument("--max-minutes", type=float, default=20)
    ap.add_argument("--goal", default="", help="typed into the window by itself; the take then ends on its own")
    ap.add_argument("--hold", type=float, default=10, help="seconds to stay on the verdict before stopping")
    args = ap.parse_args()
    if not (os.environ.get("CU_EYES_URL") and os.environ.get("CU_EYES_TOKEN")):
        raise SystemExit("set CU_EYES_URL and CU_EYES_TOKEN (the eyes server prints both)")
    print(f"starting in {args.delay}s: minimise everything except what should be in the video", flush=True)
    time.sleep(args.delay)
    out = ROOT / "runs" / "agent" / "live" / f"demo-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    recorder = start_recording(out)
    env = os.environ | {"CU_LAYOUT": "split", "CU_BRAIN": "loop", "CU_LOOP_BRAIN": args.brain}
    if args.goal:
        env["CU_AUTORUN"] = args.goal
    electron = ROOT / "app" / "node_modules" / ".bin" / "electron.cmd"
    log = out.with_suffix(".log")
    window = subprocess.Popen([str(electron), "."], cwd=ROOT / "app", env=env,
                              stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    print(f"recording to {out}", flush=True)
    deadline = time.time() + args.max_minutes * 60
    while window.poll() is None and time.time() < deadline:
        time.sleep(2)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if args.goal and "[agent] finished" in text:  # the window printed the verdict: hold on it, then end
            print("verdict reached", flush=True)
            time.sleep(args.hold)
            break
    if window.poll() is None:  # electron.cmd starts electron.exe: end the whole tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(window.pid)], capture_output=True)
    stop_recording(recorder)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
