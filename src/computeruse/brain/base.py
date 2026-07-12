"""
The Brain interface: the one seam every decision-making backend implements.
See docs/decisions/0002-brain-abstraction-and-cost-strategy.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..action.schema import Action
from ..perception.screenshot import Screenshot
from ..perception.uia import UIElement


class Brain(ABC):
    @abstractmethod
    def decide(
        self,
        goal: str,
        screenshot: Screenshot,
        ui_elements: list[UIElement],
        action_history: list[Action],
    ) -> Action:
        """Look at the current state and return the single next Action to take.

        Returning an Action with type=ActionType.DONE signals the goal is
        complete and the orchestrator should stop looping.
        """
        raise NotImplementedError
