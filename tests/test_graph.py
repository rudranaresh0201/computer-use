from langgraph.graph import END

from computeruse.orchestrator.graph import (
    MAX_CONSECUTIVE_FAILURES,
    MAX_STEPS,
    _route_after_act,
)


def base_state(**overrides) -> dict:
    state = {
        "goal": "test",
        "screenshot": None,
        "ui_elements": [],
        "action_history": [],
        "done": False,
        "step_count": 0,
        "consecutive_failures": 0,
    }
    state.update(overrides)
    return state


def test_routes_to_end_when_brain_declares_done():
    assert _route_after_act(base_state(done=True)) == END


def test_routes_to_end_when_step_ceiling_hit():
    assert _route_after_act(base_state(step_count=MAX_STEPS)) == END


def test_routes_to_end_when_failure_ceiling_hit():
    assert _route_after_act(base_state(consecutive_failures=MAX_CONSECUTIVE_FAILURES)) == END


def test_routes_back_to_decide_otherwise():
    assert _route_after_act(base_state()) == "decide"
