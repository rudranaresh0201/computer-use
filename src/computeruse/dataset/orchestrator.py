"""
Dataset orchestration layer — the "future collection-run script" that
collector.py's own module docstring said would be a separate caller. Walks
the app registry (apps.yaml), launches/attaches to each app, drives it
through the registry's predefined states, and calls the existing collector
unchanged at each state.

Deliberately not shaped like the Phase 2 LangGraph agent loop: there's no
Brain deciding what to do next. Every action here is fixed in advance in the
registry, because we already know which states we want labeled — see
docs/journal.md for the fuller reasoning. What it does share with Phase 2 is
the "hands": driving steps are dispatched through the same Action /
ActionController / SafetyGuard stack Phase 1 built, not a new way of
touching the OS.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pywinauto import Desktop

from ..action.controller import ActionController
from ..action.schema import Action, ActionType
from ..perception.uia import UIElement, get_foreground_window_tree
from .collector import collect_from_current_window, write_samples
from .registry import AppConfig, AppState, DriveStep, GeometryVariant, load_registry

logger = logging.getLogger("computeruse.dataset.orchestrator")

WINDOW_FIND_TIMEOUT = 15.0
WINDOW_POLL_INTERVAL = 0.5
STEP_SETTLE_SECONDS = 0.15  # let the UI catch up between individual input events

DWMWA_CLOAKED = 14

# Win32 constants for _apply_geometry's SetWindowPos call
SW_RESTORE = 9
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010


class OrchestrationError(Exception):
    pass


def _wait_for_stable_rect(window, timeout: float = 3.0, poll: float = 0.1) -> None:
    """Packaged/UWP-hosted apps (e.g. Paint, MSPaintApp) report a zero or
    garbage bounding rect from UIA for a brief window right after
    set_focus() succeeds -- the window is confirmed foreground before its
    accessibility tree has finished laying out, so every descendant's rect
    (and therefore collector.py's on-screen filter, which depends on real
    rects) is briefly wrong. Confirmed by direct measurement: Paint's window
    rect read immediately after set_focus() was (0,0,0,0); the same read
    after waiting was the real (191,107,1727,971) -- see docs/journal.md,
    2026-07-16. Poll until the rect is non-degenerate instead of guessing a
    fixed sleep, since different apps' packaged UIA providers settle at
    different speeds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            rect = window.element_info.rectangle
            if rect.width() > 0 and rect.height() > 0:
                return
        except Exception:
            pass
        time.sleep(poll)


def _is_cloaked(hwnd: int) -> bool:
    """True if DWM is hiding this window from the desktop even though it may
    still report the classic WS_VISIBLE flag. Windows 11 rewrote Task
    Manager as a packaged/WinUI3 app that uses DWM cloaking to control
    on-screen visibility instead of WS_VISIBLE, so pywinauto's
    visible_only=True filter (which checks the classic flag) discards a
    window that is genuinely on screen -- see docs/journal.md, 2026-07-15."""
    try:
        cloaked = wintypes.DWORD()
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            DWMWA_CLOAKED,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return bool(cloaked.value)
    except Exception:
        return False


@dataclass
class ProgressLedger:
    """Persists which (app, state) pairs are already collected, and the next
    free screenshot sequence number per app, so a multi-app run can be
    interrupted (crash, an app that needs installing first, closing the
    laptop) and resumed later without re-collecting a state or overwriting a
    previous run's images under the same screenshot_id."""

    path: Path
    completed: set[str] = field(default_factory=set)
    next_seq: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ProgressLedger":
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            completed=set(data.get("completed_states", [])),
            next_seq=dict(data.get("next_seq", {})),
        )

    def is_done(self, app: str, state: str) -> bool:
        return f"{app}:{state}" in self.completed

    def peek_seq(self, app: str) -> int:
        return self.next_seq.get(app, 0)

    def mark_done(self, app: str, state: str) -> None:
        self.completed.add(f"{app}:{state}")
        self.next_seq[app] = self.peek_seq(app) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"completed_states": sorted(self.completed), "next_seq": self.next_seq},
                indent=2,
            ),
            encoding="utf-8",
        )


def _process_already_running(exe_name: str) -> bool:
    """Best-effort check (via tasklist, no extra dependency) for whether a
    process with this exact image name is already running."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return exe_name.lower() in result.stdout.lower()
    except Exception:
        return False


def _launch(config: AppConfig) -> None:
    if config.single_instance and _process_already_running(config.launch.target):
        raise OrchestrationError(
            f"{config.launch.target} is already running and {config.name} is "
            f"registered single_instance -- launching now would attach to "
            f"that existing window (and whatever real content it holds) "
            f"instead of a fresh one. Close it first, then re-run."
        )
    if config.launch.method == "exe":
        subprocess.Popen([config.launch.target, *config.launch.args])
    elif config.launch.method == "shell":
        os.startfile(config.launch.target)  # ms-settings: URIs, .msc files
    else:
        raise OrchestrationError(f"unknown launch method: {config.launch.method!r}")


def _window_matches(window, title_contains: str, window_class_contains: Optional[str]) -> bool:
    """A window is accepted if it matches the title OR (when given) the
    window class -- additive, since some apps' titles are unreliable (see
    window_class_contains's docstring in registry.py)."""
    try:
        if title_contains.lower() in (window.window_text() or "").lower():
            return True
    except Exception:
        pass
    if window_class_contains:
        try:
            if window_class_contains.lower() in (window.class_name() or "").lower():
                return True
        except Exception:
            pass
    return False


def _verify_foreground(title_contains: str, window_class_contains: Optional[str] = None) -> bool:
    try:
        active = Desktop(backend="uia").window(active_only=True, visible_only=False)
        if _is_cloaked(active.handle):
            return False
        return _window_matches(active, title_contains, window_class_contains)
    except Exception:
        return False


def _find_and_focus_window(
    title_contains: str,
    timeout: float = WINDOW_FIND_TIMEOUT,
    window_class_contains: Optional[str] = None,
):
    """Poll for a window matching `title_contains` or `window_class_contains`,
    focus it, then verify it actually became the foreground window before
    returning — the direct fix for the 2026-07-10 Notepad incident (a script
    that typed into a target it never confirmed had focus). Raises rather
    than guessing if nothing is ever confirmed foreground within the timeout."""
    deadline = time.monotonic() + timeout
    last_seen_titles: list[str] = []

    while time.monotonic() < deadline:
        try:
            candidates = Desktop(backend="uia").windows(visible_only=False)
        except Exception:
            candidates = []

        last_seen_titles = []
        for window in candidates:
            try:
                title = window.window_text()
            except Exception:
                continue
            try:
                if _is_cloaked(window.handle):
                    continue
            except Exception:
                pass
            last_seen_titles.append(title)
            if _window_matches(window, title_contains, window_class_contains):
                try:
                    window.set_focus()
                except Exception:
                    continue
                if _verify_foreground(title_contains, window_class_contains):
                    _wait_for_stable_rect(window)
                    return window
        time.sleep(WINDOW_POLL_INTERVAL)

    raise OrchestrationError(
        f"no window matching title={title_contains!r} class={window_class_contains!r} "
        f"became foreground within {timeout}s (last seen titles: {last_seen_titles})"
    )


def _find_element(
    name: Optional[str] = None,
    control_type: Optional[str] = None,
    automation_id: Optional[str] = None,
) -> Optional[UIElement]:
    """automation_id, when given, is authoritative and skips the name check
    entirely — some native controls (Control Panel's search box) expose a
    placeholder `name` like "\\r" instead of real text, so matching on name
    there would never succeed even though the element is real and stable."""
    for element in get_foreground_window_tree():
        if automation_id is not None:
            if element.automation_id != automation_id:
                continue
        elif name is not None and element.name.strip() != name:
            continue
        if control_type is not None and element.control_type != control_type:
            continue
        return element
    return None


def _drive_step(step: DriveStep, controller: ActionController) -> None:
    if step.kind == "key":
        controller.execute(Action(type=ActionType.KEY, text=step.text))
    elif step.kind == "type":
        controller.execute(Action(type=ActionType.TYPE, text=step.text))
    elif step.kind == "wait":
        time.sleep(step.duration_seconds)
    elif step.kind == "click_element":
        element = _find_element(step.name, step.control_type, step.automation_id)
        if element is None:
            raise OrchestrationError(
                f"click_element could not find an element matching "
                f"name={step.name!r} control_type={step.control_type!r} "
                f"automation_id={step.automation_id!r} in the current window"
            )
        controller.execute(Action(type=ActionType.LEFT_CLICK, coordinate=element.center))
    else:
        raise OrchestrationError(f"unknown drive step kind: {step.kind!r}")
    time.sleep(STEP_SETTLE_SECONDS)


def _apply_geometry(window, variant: GeometryVariant) -> None:
    """Best-effort resize/move before capturing a geometry variant.

    Deliberately best-effort: a maximized window ignores move_window until
    it's restored, and some packaged apps (Settings, Task Manager) enforce a
    minimum size or refuse outright. A variant that can't be applied is
    still worth collecting -- it just yields the app's natural geometry
    instead of the requested one, which is a duplicate screenshot at worst,
    never a wrong label. The rect is re-read from the OS after the move and
    re-verified for stability, so whatever geometry actually took effect is
    what gets cropped and labeled.
    """
    if variant.width is None or variant.height is None:
        return
    # pywinauto's UIA wrapper has no move_window (that's the win32 backend's
    # API) -- confirmed live 2026-07-19, when every variant silently failed
    # with AttributeError and 24 "different sizes" all came out identical.
    # Go straight to the Win32 API via the window handle instead.
    try:
        hwnd = window.handle
    except Exception as exc:
        logger.warning("geometry variant %s: no window handle: %s", variant.name, exc)
        return
    try:
        # SW_RESTORE first: a maximized window ignores SetWindowPos's
        # size/position until it is un-maximized.
        ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
        ok = ctypes.windll.user32.SetWindowPos(
            hwnd,
            0,  # hWndInsertAfter: ignored because of SWP_NOZORDER
            int(variant.left),
            int(variant.top),
            int(variant.width),
            int(variant.height),
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
        if not ok:
            logger.warning(
                "geometry variant %s: SetWindowPos refused (app may enforce a "
                "minimum size)", variant.name
            )
            return
    except Exception as exc:
        logger.warning("geometry variant %s could not be applied: %s", variant.name, exc)
        return
    _wait_for_stable_rect(window)
    time.sleep(0.3)  # let the app finish reflowing its layout after a resize


def _run_state(
    app_config: AppConfig,
    state: AppState,
    controller: ActionController,
    dataset_root: Path,
    session_id: str,
    screenshot_seq: int,
    variant: Optional[GeometryVariant] = None,
) -> int:
    # Geometry first, then the drive steps. Resizing after the steps would
    # leave any dialog they opened at its original size and position while
    # the parent window moved out from under it -- the variant would change
    # the background and not the thing being labeled.
    if variant is not None:
        window = _find_and_focus_window(
            app_config.window_title_contains,
            timeout=WINDOW_FIND_TIMEOUT,
            window_class_contains=app_config.window_class_contains,
        )
        _apply_geometry(window, variant)

    for step in state.steps:
        _drive_step(step, controller)
    time.sleep(state.settle_seconds)

    # Re-confirm focus immediately before labeling, not just once per app
    # before the state loop. Drive steps open dialogs, apps crash after
    # launch (GIMP exits 5-15s in), and a human touching the machine steals
    # focus -- any of which used to mean the next state got labeled from
    # whatever window happened to be in front. See docs/journal.md 2026-07-19.
    if not _verify_foreground(
        app_config.window_title_contains, app_config.window_class_contains
    ):
        _find_and_focus_window(
            app_config.window_title_contains,
            timeout=WINDOW_FIND_TIMEOUT,
            window_class_contains=app_config.window_class_contains,
        )

    samples = collect_from_current_window(
        app=app_config.name,
        app_richness=app_config.richness,
        split=state.split,
        session_id=session_id,
        screenshot_seq=screenshot_seq,
        dataset_root=dataset_root,
        # the collector re-verifies independently -- belt and braces, since
        # focus can be lost in the milliseconds between the check above and
        # the actual screen grab
        expect_title_contains=app_config.window_title_contains,
        expect_class_contains=app_config.window_class_contains,
    )
    write_samples(samples, dataset_root / "labels.jsonl")
    return len(samples)


def run(
    registry_path: Path,
    dataset_root: Path,
    controller: ActionController,
    session_id: str,
    apps_filter: Optional[list[str]] = None,
    resume: bool = True,
) -> ProgressLedger:
    """Walk the registry once: for each app with pending states, launch it,
    drive it through each not-yet-collected state, and invoke the collector.
    A failure launching one app, or driving one state, is logged and skipped
    rather than aborting the whole run — a third-party app that isn't
    installed yet (VLC, GIMP, ...) shouldn't block every other app, and one
    broken state shouldn't cost the rest of that app's states."""
    apps = load_registry(registry_path)
    if apps_filter is not None:
        apps = [a for a in apps if a.name in apps_filter]

    ledger_path = dataset_root / "progress.json"
    ledger = ProgressLedger.load(ledger_path) if resume else ProgressLedger(ledger_path)

    for app_config in apps:
        # expand each state into its geometry variants (or a single None
        # variant when the app declares none), then drop the already-done
        pending = []
        for state in app_config.states:
            variants = state.geometry_variants or [None]
            for variant in variants:
                unit = state.name if variant is None else f"{state.name}@{variant.name}"
                if not ledger.is_done(app_config.name, unit):
                    pending.append((state, variant))
        if not pending:
            logger.info(
                "app=%s all %d states already collected, skipping",
                app_config.name,
                len(app_config.states),
            )
            continue

        logger.info(
            "app=%s launching (%d/%d states pending)",
            app_config.name,
            len(pending),
            len(app_config.states),
        )
        try:
            _launch(app_config)
            _find_and_focus_window(
                app_config.window_title_contains,
                timeout=app_config.launch_timeout_seconds,
                window_class_contains=app_config.window_class_contains,
            )
        except Exception as exc:
            logger.error("app=%s failed to launch/focus, skipping app: %s", app_config.name, exc)
            continue

        for state, variant in pending:
            # one ledger key per (state, variant) so an interrupted run
            # resumes at the right variant instead of re-collecting the
            # whole state or skipping its remaining sizes
            unit = state.name if variant is None else f"{state.name}@{variant.name}"
            seq = ledger.peek_seq(app_config.name)
            try:
                count = _run_state(
                    app_config, state, controller, dataset_root, session_id, seq, variant
                )
                ledger.mark_done(app_config.name, unit)
                logger.info(
                    "app=%s state=%s collected %d samples (screenshot_seq=%d)",
                    app_config.name,
                    unit,
                    count,
                    seq,
                )
            except Exception as exc:
                logger.error(
                    "app=%s state=%s failed, skipping: %s", app_config.name, unit, exc
                )

    return ledger
