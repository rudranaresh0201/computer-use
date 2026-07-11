"""
Phase 1 demo: proves the real harness works end-to-end, not just unit tests.

Opens Notepad (fresh, untitled, nothing saved), reads its UI Automation
tree, uses that tree to find the text editing area precisely (not a guessed
pixel coordinate), types into it through the full ActionController +
SafetyGuard stack, takes a before/after screenshot, and prints everything
so it can be pasted into the Phase 1 journal entry as evidence.

Run: .venv/Scripts/python.exe scripts/phase1_demo.py
"""

import subprocess
import sys
import time
from pathlib import Path

from pywinauto import Desktop

from computeruse.action.controller import ActionController
from computeruse.action.schema import Action, ActionType
from computeruse.perception import screenshot as screenshot_mod
from computeruse.perception.uia import get_foreground_window_tree
from computeruse.safety import ActionLogger, SafetyConfig, SafetyGuard

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "phase1_demo"
TARGET_FILE = RUN_DIR / "demo_target.txt"
# Windows 11 Notepad is single-instance and can silently refocus an existing
# tab instead of opening a fresh blank one (see the 2026-07-10 incident in
# docs/journal.md). So instead of guessing at a generic "Untitled" title, we
# create our own uniquely-named file and verify the window title contains
# *that exact filename* — a target we control end to end, not a guess.
TARGET_TITLE_MARKER = TARGET_FILE.name


def wait_for_target_notepad_window(timeout: float = 5.0, poll_interval: float = 0.2):
    """Poll until the Notepad window for our specific target file is the
    foreground window, or give up. Returns the pid, or None. This is the
    "verify before you act" check the incident writeup in docs/journal.md
    is about — never assume a launched app has focus, confirm it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            win = Desktop(backend="uia").window(active_only=True, visible_only=True)
            wrapper = win.wrapper_object()
            title = wrapper.window_text()
            if "Notepad" in title and TARGET_TITLE_MARKER in title:
                return wrapper.process_id()
        except Exception:
            pass
        time.sleep(poll_interval)
    return None


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("== 1. Screenshot + scaling ==")
    before = screenshot_mod.capture(save_path=RUN_DIR / "before.png")
    print(f"real_size={before.real_size} scaled_size={before.scaled_size} "
          f"scale=({before.scale_x:.3f}, {before.scale_y:.3f})")

    print(f"\n== 2. Launching Notepad on a target file we control: {TARGET_FILE} ==")
    TARGET_FILE.write_text("", encoding="utf-8")  # pre-create it so Notepad opens with no dialog
    subprocess.Popen(["notepad.exe", str(TARGET_FILE)])

    print(f"Verifying the Notepad window for '{TARGET_TITLE_MARKER}' actually has focus before doing anything else...")
    pid = wait_for_target_notepad_window()
    if pid is None:
        print(
            f"ABORT: could not confirm the '{TARGET_TITLE_MARKER}' Notepad window got focus "
            "within the timeout. Refusing to type blindly into whatever window "
            "happens to be focused. This is the exact failure mode from the "
            "2026-07-10 incident (see docs/journal.md) — fail loud, not silent."
        )
        sys.exit(1)
    print(f"confirmed: target Notepad window (pid={pid}) has focus")

    print("\n== 3. UI Automation tree of the foreground window ==")
    elements = get_foreground_window_tree()
    print(f"found {len(elements)} elements")
    edit_el = None
    for el in elements:
        marker = ""
        if el.control_type in ("Edit", "Document") and edit_el is None:
            edit_el = el
            marker = "  <-- using this as the text area"
        print(f"  [{el.control_type}] '{el.name}' rect={el.rect}{marker}")

    config = SafetyConfig(allowed_actions=set(ActionType), max_actions_per_minute=60)
    logger = ActionLogger(RUN_DIR / "actions.jsonl")
    guard = SafetyGuard(config, logger)
    controller = ActionController(guard)

    print("\n== 4. Executing actions through ActionController + SafetyGuard ==")
    if edit_el is None:
        print(
            "ABORT: no Edit/Document element found on the verified Notepad window. "
            "Refusing to type without a confirmed target (this is where the vision "
            "fallback from ADR-0001 would kick in, in Phase 3)."
        )
        sys.exit(1)

    x, y = edit_el.center
    result = controller.execute(Action(type=ActionType.LEFT_CLICK, coordinate=(x, y)))
    print(f"left_click at UIA-derived coordinate {(x, y)}: success={result.success} error={result.error}")

    # Re-verify focus didn't move after the click before we type — cheap and
    # exactly the discipline the incident showed was missing.
    pid_after_click = wait_for_target_notepad_window(timeout=1.0)
    if pid_after_click != pid:
        print("ABORT: focus moved away from the target Notepad window after the click. Not typing.")
        sys.exit(1)

    result = controller.execute(
        Action(type=ActionType.TYPE, text="computer-use Phase 1 demo: hands + eyes + UIA all work.")
    )
    print(f"type: success={result.success} error={result.error}")

    print("\n== 5. After screenshot ==")
    after = screenshot_mod.capture(save_path=RUN_DIR / "after.png")
    print(f"saved to {after.path}")

    print(f"\nAction log: {RUN_DIR / 'actions.jsonl'}")
    print("Notepad left open for visual confirmation — close it manually when you've checked it.")
    print("\nPhase 1 demo complete.")


if __name__ == "__main__":
    main()
