import io
import json

from computeruse.action.schema import ActionType
from computeruse.brain.chat import ScriptedChat
from computeruse.brain.llm_desktop import LLMDesktopBrain
from computeruse.perception.uia import UIElement
from computeruse.planner import make_plan
from computeruse.sidecar import Sidecar

APPS = ["calculator", "notepad_pp", "paint"]
CALC = [UIElement("Seven", "Button", (10, 300, 100, 360))]


def test_a_command_becomes_an_app_a_precise_goal_and_checkpoints():
    chat = ScriptedChat([("plan", {"app": "calculator", "goal": "Compute 123 x 45",
                                   "steps": ["Enter 123", "Multiply by 45", "Read the result"]})])
    plan = make_plan(chat, "whats 123 times 45", APPS)
    assert (plan.app, plan.goal, plan.steps) == ("calculator", "Compute 123 x 45",
                                                 ["Enter 123", "Multiply by 45", "Read the result"])
    assert "notepad_pp" in chat.seen[0][1]["content"]  # the model saw the app menu


def test_no_app_fits_means_no_run():
    chat = ScriptedChat([("plan", {"app": "none", "goal": "None of these apps can send email", "steps": []})])
    assert make_plan(chat, "email my boss", APPS).app is None


def test_a_reply_that_is_not_a_plan_falls_back_instead_of_crashing():
    plan = make_plan(ScriptedChat([("click", {"id": "u1"})]), "draw a cat", APPS, app="paint")
    assert plan.app == "paint" and plan.steps == ["draw a cat"]


def test_too_many_steps_are_cut_and_an_app_the_person_chose_wins():
    chat = ScriptedChat([("plan", {"app": "paint", "goal": "g", "steps": [str(i) for i in range(10)]})])
    plan = make_plan(chat, "x", APPS, app="notepad_pp")
    assert plan.app == "notepad_pp" and len(plan.steps) == 6


def test_the_brain_ticks_the_plan_without_spending_an_action():
    ticks = []
    brain = LLMDesktopBrain(ScriptedChat([("check", {"step": 1}), ("click", {"id": "u0"})]), "calculator",
                            foreground_title=lambda: "Calculator", plan=["Enter 7", "Read it"], on_check=ticks.append)
    action = brain.decide("type 7", None, CALC, [])
    assert ticks == [1] and action.type == ActionType.LEFT_CLICK  # the check and the click, one turn
    assert "[x] 1. Enter 7" in brain.chat.seen[1][1]["content"]  # the model saw its own tick


def test_a_bad_or_repeated_check_ticks_nothing():
    ticks = []
    brain = LLMDesktopBrain(ScriptedChat([("check", {"step": 9}), ("check", {"step": "x"}), ("check", {"step": 1}),
                                          ("check", {"step": 1}), ("done", {"answer": "ok"})]),
                            "calculator", foreground_title=lambda: "Calculator", plan=["a"], on_check=ticks.append)
    action = brain.decide("x", None, CALC, [])
    assert ticks == [1] and action.type == ActionType.WAIT  # four checks in a row: told to act instead


def test_without_a_plan_the_brain_is_offered_no_check_tool():
    chat = ScriptedChat([("done", {"answer": "ok"})])
    LLMDesktopBrain(chat, "calculator", foreground_title=lambda: "Calculator").decide("x", None, CALC, [])
    assert "Plan:" not in chat.seen[0][1]["content"]


def test_the_sidecar_plans_first_and_streams_the_ticks():
    def planner(command, app):
        return {"app": "calculator", "goal": "Compute 2+2", "steps": ["Enter 2+2", "Read it"]}

    def runner(goal, app, on_step, should_stop, plan=None, on_check=None):
        assert (goal, app, plan) == ("Compute 2+2", "calculator", ["Enter 2+2", "Read it"])
        on_check(1)
        on_step("done", "4")
        on_check(2)
        return {"status": "done", "answer": "4"}

    out = io.StringIO()
    s = Sidecar(runner, APPS, out, planner=planner)
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"goal": "2+2?"}}))  # no app
    s.wait(5)
    methods = [m.get("method") for m in map(json.loads, out.getvalue().splitlines()) if "method" in m]
    assert methods == ["planning", "plan", "started", "check", "step", "check", "finished"]


def test_the_sidecar_ends_without_moving_when_no_app_fits():
    def runner(*a, **k):
        raise AssertionError("must not run")

    out = io.StringIO()
    s = Sidecar(runner, APPS, out, planner=lambda c, a: {"app": None, "goal": "No app can send email"})
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"goal": "email"}}))
    s.wait(5)
    last = json.loads(out.getvalue().splitlines()[-1])
    assert last["method"] == "finished" and last["params"]["status"] == "stopped"
    assert "email" in last["params"]["answer"]
