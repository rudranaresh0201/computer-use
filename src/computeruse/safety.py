"""
Everything that stands between "a Brain decided to do X" and "X actually
happened on the real machine": an allow-list, a rate limiter, and an action
log. The kill-switch itself is pyautogui's built-in FAILSAFE (slamming the
mouse into a screen corner raises pyautogui.FailSafeException) rather than
something reimplemented here — see action/controller.py for where that
exception is caught and turned into a logged, non-fatal abort.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .action.schema import Action, ActionResult, ActionType


class ActionNotAllowed(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


@dataclass
class SafetyConfig:
    allowed_actions: set[ActionType] = field(default_factory=lambda: set(ActionType))
    max_actions_per_minute: int = 60


class RateLimiter:
    def __init__(self, max_per_minute: int, clock=time.monotonic) -> None:
        self.max_per_minute = max_per_minute
        self._clock = clock
        self._timestamps: list[float] = []

    def check(self) -> None:
        now = self._clock()
        cutoff = now - 60
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_per_minute:
            raise RateLimitExceeded(
                f"Exceeded {self.max_per_minute} actions/minute"
            )
        self._timestamps.append(now)


class ActionLogger:
    """Appends one JSON object per line to a run log — cheap, greppable,
    and the foundation for the Phase 4 replay tool."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: Action, result: ActionResult) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action.to_dict(),
            "success": result.success,
            "error": result.error,
            "screenshot_path": result.screenshot_path,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class SafetyGuard:
    def __init__(self, config: SafetyConfig, logger: ActionLogger) -> None:
        self.config = config
        self.logger = logger
        self.rate_limiter = RateLimiter(config.max_actions_per_minute)

    def check(self, action: Action) -> None:
        """Raises ActionNotAllowed / RateLimitExceeded. Does not log —
        callers log the outcome (including the rejection) themselves so
        every check, not just every executed action, ends up in the trail."""
        if action.type not in self.config.allowed_actions:
            raise ActionNotAllowed(f"'{action.type.value}' is not in the allow-list")
        self.rate_limiter.check()

    def log(self, action: Action, result: ActionResult) -> None:
        self.logger.log(action, result)
