"""
The Action vocabulary the whole system speaks internally.

Deliberately mirrors Anthropic's own computer-use tool action set
(see docs/research/anthropic-computer-use.md) so that when a real Brain
(Phase 3) starts producing Anthropic tool_use blocks, converting them into
an Action is a field rename, not a redesign. See docs/decisions/0001-architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ActionType(str, Enum):
    SCREENSHOT = "screenshot"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK_DRAG = "left_click_drag"
    TYPE = "type"
    KEY = "key"
    HOLD_KEY = "hold_key"
    SCROLL = "scroll"
    WAIT = "wait"
    DONE = "done"


# Action types that require a `coordinate` field to be present.
_REQUIRES_COORDINATE = {
    ActionType.LEFT_CLICK,
    ActionType.RIGHT_CLICK,
    ActionType.MIDDLE_CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.TRIPLE_CLICK,
    ActionType.MOUSE_MOVE,
}


class ActionValidationError(ValueError):
    pass


@dataclass
class Action:
    type: ActionType
    coordinate: Optional[tuple[int, int]] = None
    start_coordinate: Optional[tuple[int, int]] = None  # left_click_drag origin
    text: Optional[str] = None  # TYPE payload, KEY combo (e.g. "ctrl+s"), or a click/scroll modifier
    scroll_direction: Optional[str] = None  # "up" | "down" | "left" | "right"
    scroll_amount: Optional[int] = None
    duration: Optional[float] = None  # WAIT seconds, HOLD_KEY seconds

    def validate(self) -> None:
        if self.type in _REQUIRES_COORDINATE and self.coordinate is None:
            raise ActionValidationError(f"{self.type.value} requires a coordinate")

        if self.type == ActionType.LEFT_CLICK_DRAG:
            if self.start_coordinate is None or self.coordinate is None:
                raise ActionValidationError(
                    "left_click_drag requires both start_coordinate and coordinate"
                )

        if self.type == ActionType.TYPE and not self.text:
            raise ActionValidationError("type requires non-empty text")

        if self.type == ActionType.KEY and not self.text:
            raise ActionValidationError("key requires text (e.g. 'ctrl+s')")

        if self.type == ActionType.HOLD_KEY:
            if not self.text:
                raise ActionValidationError("hold_key requires text (the key to hold)")
            if self.duration is None or self.duration <= 0:
                raise ActionValidationError("hold_key requires a positive duration")

        if self.type == ActionType.SCROLL:
            if self.scroll_direction not in {"up", "down", "left", "right"}:
                raise ActionValidationError(
                    "scroll requires scroll_direction in {up, down, left, right}"
                )
            if self.scroll_amount is not None and self.scroll_amount < 0:
                raise ActionValidationError("scroll_amount must be non-negative")

        for coord_field in (self.coordinate, self.start_coordinate):
            if coord_field is not None:
                x, y = coord_field
                if x < 0 or y < 0:
                    raise ActionValidationError(f"coordinate must be non-negative, got {coord_field}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action": self.type.value}
        if self.coordinate is not None:
            d["coordinate"] = list(self.coordinate)
        if self.start_coordinate is not None:
            d["start_coordinate"] = list(self.start_coordinate)
        if self.text is not None:
            d["text"] = self.text
        if self.scroll_direction is not None:
            d["scroll_direction"] = self.scroll_direction
        if self.scroll_amount is not None:
            d["scroll_amount"] = self.scroll_amount
        if self.duration is not None:
            d["duration"] = self.duration
        return d

    @classmethod
    def from_anthropic_tool_input(cls, tool_input: dict[str, Any]) -> "Action":
        """Build an Action from a raw Anthropic computer-use tool_use.input dict.

        Field names already match (see module docstring); this just renames
        `action` -> `type` and coerces coordinate lists to tuples.
        """
        raw = dict(tool_input)
        action_type = ActionType(raw.pop("action"))
        for key in ("coordinate", "start_coordinate"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        return cls(type=action_type, **raw)


@dataclass
class ActionResult:
    success: bool
    error: Optional[str] = None
    screenshot_path: Optional[str] = None
