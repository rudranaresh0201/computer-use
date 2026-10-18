"""
What a brain is allowed to say: one JSON object per turn, an *intent*, not a mouse event.

    {"thought": "The Brush tool has no name in the list.", "action": "click", "target": "the Brush tool"}
    {"thought": "...", "action": "click", "element": "u12"}
    {"thought": "...", "action": "type", "element": "e4", "text": "ada@example.com"}
    {"thought": "...", "action": "check", "step": 2, "evidence": {"kind": "text", "value": "Saved"}}
    {"thought": "...", "action": "done", "claim": "The note is saved."}

Which fields a brain may fill depends on the arm (brains.ARMS): in the paper arm it may name an
element or describe a target but never give x, y for a click; in the vision arm it gives x, y.
The router (router.py) turns the intent into an element or a point, and says which channel did it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

ACTIONS = ("plan", "click", "double_click", "right_click", "type", "key", "scroll", "drag", "wait", "check", "done")
POINTED = ("click", "double_click", "right_click")  # actions that need a place on the screen
EVIDENCE = ("file", "text", "changed", "url")


@dataclass
class Intent:
    action: str
    thought: str = ""
    element: Optional[str] = None  # a ref from the observation: u12 (desktop) / e7 (web)
    target: Optional[str] = None  # a description, for what the list does not name: "the Brush tool"
    x: Optional[int] = None  # screenshot pixels (vision arm, and drawing)
    y: Optional[int] = None
    to_x: Optional[int] = None
    to_y: Optional[int] = None
    to_target: Optional[str] = None  # drag end, as a description
    text: Optional[str] = None
    keys: Optional[str] = None
    direction: str = "down"
    amount: int = 3
    seconds: float = 1.0
    steps: list[str] = field(default_factory=list)  # plan
    step: Optional[int] = None  # check
    evidence: dict[str, Any] = field(default_factory=dict)  # check: {"kind": ..., "value": ...}
    claim: str = ""  # done

    def brief(self) -> str:
        """One line for the history the brain reads back."""
        if self.action == "plan":
            return f"plan {len(self.steps)} steps"
        if self.action == "check":
            return f"check step {self.step} by {self.evidence.get('kind')} {self.evidence.get('value', '')!r}"
        if self.action == "done":
            return f"done: {self.claim[:80]}"
        where = (f"element {self.element}" if self.element else f"target {self.target!r}" if self.target
                 else f"({self.x}, {self.y})" if self.x is not None else "")
        extra = (f" text {self.text[:40]!r}" if self.text else "") + (f" keys {self.keys}" if self.keys else "")
        return " ".join(p for p in (self.action, where, extra.strip()) if p)


class BadIntent(ValueError):
    """The brain's reply is not a usable intent; the message says why, for the brain to read."""


def _int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        raise BadIntent(f"{value!r} is not a number") from None


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_intent(reply: str) -> Intent:
    """The first JSON object in `reply` (models sometimes wrap it in prose or a code fence)."""
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        raise BadIntent("the reply had no JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise BadIntent(f"the JSON did not parse ({e.msg})") from None
    if not isinstance(data, dict):
        raise BadIntent("the JSON was not an object")
    action = str(data.get("action", "")).strip().lower()
    if action not in ACTIONS:
        raise BadIntent(f"action {action!r} is not one of {', '.join(ACTIONS)}")
    evidence = data.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    intent = Intent(
        action=action, thought=str(data.get("thought", "")).strip(),
        element=_str(data.get("element")), target=_str(data.get("target")),
        x=_int(data.get("x")), y=_int(data.get("y")), to_x=_int(data.get("to_x")), to_y=_int(data.get("to_y")),
        to_target=_str(data.get("to_target")), text=data.get("text") if data.get("text") is not None else None,
        keys=_str(data.get("keys")), direction=str(data.get("direction") or "down"),
        amount=_int(data.get("amount")) or 3, seconds=float(data.get("seconds") or 1.0),
        steps=[str(s) for s in data.get("steps") or [] if str(s).strip()], step=_int(data.get("step")),
        evidence={"kind": str(evidence.get("kind", "")), "value": str(evidence.get("value", ""))} if evidence else {},
        claim=str(data.get("claim", "")).strip(),
    )
    if action == "plan" and not intent.steps:
        raise BadIntent("plan needs steps: a list of 2-8 short checkpoints")
    if action == "check":
        if intent.step is None:
            raise BadIntent("check needs step (the plan step's number)")
        if intent.evidence.get("kind") not in EVIDENCE:
            raise BadIntent(f"check needs evidence.kind, one of {', '.join(EVIDENCE)}")
    if action == "key" and not intent.keys:
        raise BadIntent("key needs keys, e.g. 'ctrl+s'")
    if action == "type" and not intent.text:
        raise BadIntent("type needs text")
    return intent
