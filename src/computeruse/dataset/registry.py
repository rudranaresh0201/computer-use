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
class AppState:
    name: str
    split: str  # "train" | "dev" | "test_same_app" | "test_held_out_app"
    steps: list[DriveStep] = field(default_factory=list)
    settle_seconds: float = 0.5


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
            )
        )
    return apps
