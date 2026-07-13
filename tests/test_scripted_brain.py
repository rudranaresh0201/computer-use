import pytest

from computeruse.action.schema import Action, ActionType
from computeruse.brain.scripted import ScriptExhaustedError, ScriptedBrain


def test_returns_actions_in_order():
    script = [
        Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1)),
        Action(type=ActionType.DONE),
    ]
    brain = ScriptedBrain(script)

    first = brain.decide(goal="x", screenshot=None, ui_elements=[], action_history=[])
    second = brain.decide(goal="x", screenshot=None, ui_elements=[], action_history=[])

    assert first is script[0]
    assert second is script[1]


def test_raises_when_script_exhausted():
    brain = ScriptedBrain([Action(type=ActionType.DONE)])
    brain.decide(goal="x", screenshot=None, ui_elements=[], action_history=[])

    with pytest.raises(ScriptExhaustedError):
        brain.decide(goal="x", screenshot=None, ui_elements=[], action_history=[])
