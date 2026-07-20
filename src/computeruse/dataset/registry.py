"""
App registry — the declarative spec `orchestrator.py` walks. Loads
data/gui_grounding/apps.yaml (schema derived from
docs/research/gui-grounding-dataset-design.md's app table and split design)
into typed objects. Only the orchestrator interprets what a DriveStep means;
this module just parses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DriveStep:
    """One input action needed to move an app from its current state toward
    a target state. `kind` selects which fields apply:
      - "key": `text` is a pyautogui-style key combo, e.g. "ctrl+f"
      - "type": `text` is literal text to type
      - "click_element": identifies a UIA element in the current foreground
        window by `automation_id` if given (more reliable — some native
        controls, e.g. Control Panel's search box, expose a placeholder
        `name` like "\r" rather than real text), else by `name` +
        `control_type`. Its center is clicked. Deliberately never a
        hardcoded pixel coordinate, so a registry entry survives a different
        screen resolution.
      - "wait": pause for `duration_seconds`
    """

    kind: str
    text: Optional[str] = None
    name: Optional[str] = None
    control_type: Optional[str] = None
    automation_id: Optional[str] = None
    duration_seconds: float = 0.5


@dataclass
class GeometryVariant:
    """One window size/position to capture a state at.

    The 2026-07-19 audit found the dataset had only 32 unique screenshots
    behind 542 rows -- screenshots, not rows, are the scarce resource, and
    every app was always captured at exactly one geometry. That teaches
    "the Close button is at (982, 24)" rather than "the Close button is the
    X at the top-right of *this window*". Re-capturing the same state at a
    few sizes multiplies screenshot count without authoring new states, and
    the ground-truth box moves with the window, so it directly penalizes
    memorizing absolute position.

    `name` is appended to the state name for the progress ledger, so
    variants resume independently. width/height are in real screen px;
    None means "leave the window as-is" (the app's natural size).
    """

    name: str
    width: Optional[int] = None
    height: Optional[int] = None
    left: int = 60
    top: int = 60


@dataclass
class AppState:
    name: str
    split: str  # "train" | "dev" | "test_same_app" | "test_held_out_app"
    steps: list[DriveStep] = field(default_factory=list)
    settle_seconds: float = 0.5
    # Every variant of one state shares that state's split, deliberately:
    # two sizes of the same screen are near-duplicates, so splitting them
    # across train/dev would be leakage dressed up as more data.
    geometry_variants: list[GeometryVariant] = field(default_factory=list)


@dataclass
class AppLaunch:
    method: str  # "exe" (subprocess, args on PATH) | "shell" (os.startfile — URIs, .msc files)
    target: str
    args: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    name: str
    pool: str  # "train" | "held_out"
    richness: str  # "rich" | "weak"
    launch: AppLaunch
    window_title_contains: str
    states: list[AppState]
    # some apps show a splash/loading window before their real one (e.g.
    # GIMP's "GIMP Startup" for ~20s) -- override per-app when the
    # orchestrator's default window-find timeout is too tight.
    launch_timeout_seconds: float = 15.0
    # for apps whose title is unreliable (Windows Terminal shows the active
    # tab's shell name -- "Windows PowerShell" -- never "Terminal"), match
    # on window class instead. A window is accepted if it matches EITHER
    # the title or the class, so this is additive, not a replacement.
    window_class_contains: Optional[str] = None
    # True for apps that persist across launches instead of opening fresh
    # (Windows 11 Notepad restores real tabs into whatever window is already
    # running) -- collecting against an already-running instance risks
    # capturing the user's own real content instead of a neutral document.
    # See docs/journal.md, 2026-07-16 incident.
    single_instance: bool = False


def load_registry(path: Path) -> list[AppConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    apps: list[AppConfig] = []
    for app_raw in raw["apps"]:
        launch_raw = app_raw["launch"]
        launch = AppLaunch(
            method=launch_raw["method"],
            target=launch_raw["target"],
            args=launch_raw.get("args", []),
        )
        states = [
            AppState(
                name=state_raw["name"],
                split=state_raw["split"],
                steps=[DriveStep(**step) for step in state_raw.get("steps", [])],
                settle_seconds=state_raw.get("settle_seconds", 0.5),
                geometry_variants=[
                    GeometryVariant(**variant)
                    for variant in state_raw.get(
                        "geometry_variants", app_raw.get("geometry_variants", [])
                    )
                ],
            )
            for state_raw in app_raw["states"]
        ]
        apps.append(
            AppConfig(
                name=app_raw["name"],
                pool=app_raw["pool"],
                richness=app_raw["richness"],
                launch=launch,
                window_title_contains=app_raw["window_title_contains"],
                states=states,
                launch_timeout_seconds=app_raw.get("launch_timeout_seconds", 15.0),
                window_class_contains=app_raw.get("window_class_contains"),
                single_instance=app_raw.get("single_instance", False),
            )
        )
    return apps
