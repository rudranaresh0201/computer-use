"""
Phase 0 proof-of-concept: no LLM involved.

Proves the two lowest-level primitives work on this machine before anything
else gets built on top of them:
  1. "eyes"  - capture a screenshot of the real desktop (mss)
  2. "hands" - move the mouse and click programmatically (pyautogui)

Run: .venv/Scripts/python.exe scripts/poc_hands_and_eyes.py
"""

import time
from pathlib import Path

import mss
import mss.tools
import pyautogui

OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "poc"


def take_screenshot() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "screenshot.png"
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # primary monitor
        shot = sct.grab(monitor)
        mss.tools.to_png(shot.rgb, shot.size, output=str(out_path))
    return out_path


def move_and_click_center() -> tuple[int, int]:
    screen_w, screen_h = pyautogui.size()
    x, y = screen_w // 2, screen_h // 2
    pyautogui.moveTo(x, y, duration=0.5)
    return x, y


def main() -> None:
    print("[eyes] capturing screenshot...")
    path = take_screenshot()
    print(f"[eyes] saved screenshot to {path}")

    print("[hands] moving mouse to screen center in 1s...")
    time.sleep(1)
    x, y = move_and_click_center()
    print(f"[hands] mouse moved to ({x}, {y}). Not clicking to avoid side effects.")

    print("PoC OK: screenshot capture + mouse control both work on this machine.")


if __name__ == "__main__":
    main()
