import json

import pytest

from computeruse.action.schema import Action, ActionResult, ActionType
from computeruse.safety import (
    ActionLogger,
    ActionNotAllowed,
    RateLimitExceeded,
    RateLimiter,
    SafetyConfig,
    SafetyGuard,
)


def test_rate_limiter_allows_up_to_the_limit():
    clock = _FakeClock()
    limiter = RateLimiter(max_per_minute=3, clock=clock)

    limiter.check()
    limiter.check()
    limiter.check()

    with pytest.raises(RateLimitExceeded):
        limiter.check()


def test_rate_limiter_resets_after_a_minute():
    clock = _FakeClock()
    limiter = RateLimiter(max_per_minute=1, clock=clock)

    limiter.check()
    with pytest.raises(RateLimitExceeded):
        limiter.check()

    clock.advance(61)
    limiter.check()  # should succeed again now that the window rolled over


def test_safety_guard_rejects_action_not_in_allow_list(tmp_path):
    config = SafetyConfig(allowed_actions={ActionType.SCREENSHOT})
    guard = SafetyGuard(config, ActionLogger(tmp_path / "log.jsonl"))

    action = Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1))
    with pytest.raises(ActionNotAllowed):
        guard.check(action)


def test_safety_guard_allows_action_in_allow_list(tmp_path):
    config = SafetyConfig(allowed_actions={ActionType.LEFT_CLICK})
    guard = SafetyGuard(config, ActionLogger(tmp_path / "log.jsonl"))

    guard.check(Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1)))  # no raise


def test_action_logger_writes_one_json_line_per_action(tmp_path):
    log_path = tmp_path / "run" / "actions.jsonl"
    logger = ActionLogger(log_path)

    logger.log(Action(type=ActionType.WAIT, duration=1.0), ActionResult(success=True))
    logger.log(
        Action(type=ActionType.LEFT_CLICK, coordinate=(5, 5)),
        ActionResult(success=False, error="boom"),
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["action"]["action"] == "wait"
    assert first["success"] is True

    second = json.loads(lines[1])
    assert second["action"]["action"] == "left_click"
    assert second["success"] is False
    assert second["error"] == "boom"


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds
