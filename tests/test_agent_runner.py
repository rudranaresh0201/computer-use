"""The shared runner behind `computeruse agent` and the window's sidecar: picking the target from the
goal, a desktop app run through our loop (fake desktop, scripted brain), and the sidecar's loop mode
sending exactly one ending with the checker's verdict."""

import io
import json
from types import SimpleNamespace

from computeruse.agent.brains import ScriptedBrain
from computeruse.agent.runner import guess_target, run_goal
from computeruse.sidecar import Sidecar, loop_runner

TARGETS = {"paint": "Windows app: Paint", "calculator": "Windows app: Calculator", "notepad_pp": "Windows app: Notepad++",
           "tinkercad": "Tinkercad: 3D design", "excalidraw": "whiteboard", "sheets": "Google Sheets"}


def test_the_target_named_in_the_goal_wins_and_circuit_words_mean_tinkercad():
    assert guess_target("Draw a house in Paint and save it", TARGETS) == "paint"
    assert guess_target("open notepad++ and write a poem", TARGETS) == "notepad_pp"
    assert guess_target("Light a red LED from a coin cell battery", TARGETS) == "tinkercad"
    assert guess_target("what is 12 times 9", TARGETS) is None  # left to the brain's one-word answer


def test_a_goal_across_two_apps_splits_by_target_and_keeps_same_app_steps_together():
    from computeruse.agent.runner import split_goal

    goal = ("Open Tinkercad and build a circuit where a coin cell lights a red LED, then run the simulation, "
            "then open Calculator and compute (3 - 2) / 0.02")
    parts = split_goal(goal, TARGETS)
    assert [t for _, t in parts] == ["tinkercad", "calculator"]
    assert "run the simulation" in parts[0][0] and parts[1][0].startswith("open Calculator")
    assert split_goal("Draw a house in Paint, then save it as house.png", TARGETS) == [
        ("Draw a house in Paint, then save it as house.png", None)]  # one app: not split


def fake_session_factory():
    from tests.test_agent_loop import desktop

    session, clicks, _ = desktop()
    session.target = None
    chrome = SimpleNamespace(page=lambda: None, close=lambda: None)
    return (lambda events, approve, eyes, extra=None: (_rebind(session, events, approve), chrome)), clicks


def _rebind(session, events, approve):
    session.events, session.approve = events, approve
    return session


def test_a_desktop_app_runs_through_our_loop_and_reports_channels():
    factory, clicks = fake_session_factory()
    seen = []
    brain = ScriptedBrain([json.dumps({"action": "click", "element": "u0"}),
                           json.dumps({"action": "click", "target": "the brush icon"}),
                           json.dumps({"action": "done", "claim": "picked the tools"})])
    ep, summary = run_goal("pick the rectangle and the brush in Paint", None, lambda m, p: seen.append(m),
                           brain_kind="claude", eyes=None, session_factory=factory, brain=brain)
    assert ep.target == "paint" and summary["status"] == "done"
    assert ep.channels() == {"tree": 1} and len(clicks) == 1  # no eyes running: the description is refused
    assert seen[:2] == ["planning", "started"] and "refused" in seen and "finished" in seen


def test_the_sidecar_loop_mode_sends_one_ending_with_the_verdict():
    out = io.StringIO()
    car = Sidecar(lambda *a, **k: {}, ["paint"], out, app_optional=True)

    def fake_run_goal(goal, app, emit, **kw):
        emit("thought", {"text": "plan it"})
        emit("finished", {"status": "done"})  # the loop's own ending: dropped, the sidecar sends one
        return None, {"status": "done", "answer": "PASSED: saved house.png", "actions": 3}

    summary = loop_runner("draw", "paint", car.notify, lambda: False, {"path": None}, run_goal=fake_run_goal)
    sent = [json.loads(line)["method"] for line in out.getvalue().splitlines()]
    assert sent == ["thought"] and summary["answer"].startswith("PASSED")
