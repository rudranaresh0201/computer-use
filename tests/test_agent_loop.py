"""The agent loop end to end on a fake desktop: the brain decides, the router grounds, the Session acts
and verifies. No screen, no model."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

from computeruse.agent.brains import ScriptedBrain
from computeruse.agent.loop import Agent
from computeruse.agent.router import Router
from computeruse.desktop_session import DesktopSurface
from computeruse.session import Session, Shot


@dataclass
class El:
    name: str
    control_type: str
    center: tuple


def desktop(screens=None):
    clicks, events = [], []
    tree = [El("Rectangle", "Button", (900, 120)), El("", "Button", (950, 120)), El("Send", "Button", (1000, 120))]
    screens = list(screens or [])

    def shoot():
        small = screens.pop(0) if screens else bytes([len(clicks) * 40 % 256]) * 16  # each action changes the screen
        return Shot(b"jpeg", (1280, 720), lambda x, y: (x * 2, y * 2), small)

    surface = DesktopSurface(apps={"paint": (["mspaint.exe"], "Paint")}, tree=lambda: tree, shoot=shoot,
                             execute=lambda a: clicks.append(a) or SimpleNamespace(success=True, error=None),
                             foreground=lambda: "Untitled - Paint", focus=lambda w: True, launch=lambda app: "Paint")
    s = Session(surfaces={"paint": surface}, events=lambda m, p: events.append((m, p)), sleep=lambda s: None,
                eyes=lambda jpeg, d: ((475, 60), 1.5))
    s.open("paint")
    return s, clicks, events


def j(**kw):
    return json.dumps(kw)


def run(session, replies, arm="paper", checker=None, eyes=True):
    eyes_fn = (lambda d: session.ground_with_eyes(d)) if eyes else None
    agent = Agent(session, ScriptedBrain(list(replies)), Router(arm, eyes_fn), "draw", checker=checker)
    return agent.run()


def test_plan_act_through_tree_and_eyes_check_and_done():
    s, clicks, events = desktop()
    ep = run(s, [
        j(action="plan", steps=["Pick the rectangle", "Draw"]),
        j(thought="named, so by id", action="click", element="u0"),
        j(action="check", step=1, evidence={"kind": "changed"}),
        j(thought="no name: describe it", action="click", target="the brush icon"),
        j(action="check", step=2, evidence={"kind": "changed"}),
        j(action="done", claim="drew it"),
    ])
    assert ep.status == "done" and ep.passed and not ep.false_success
    assert ep.channels() == {"tree": 1, "eyes": 1}
    assert clicks[0].coordinate == (900, 120)  # UIA centres are already real pixels
    assert clicks[1].coordinate == (950, 120)  # the eyes' (475, 60) mapped to the real screen
    steps = [p for m, p in events if m == "step"]
    assert [p["channel"] for p in steps] == ["tree", "eyes"]  # the window shows which channel did it
    assert any(m == "thought" for m, _ in events) and events[-1][0] == "finished"


def test_done_with_unverified_steps_is_refused_then_ends_unverified():
    s, _, _ = desktop()
    ep = run(s, [j(action="plan", steps=["a", "b"])] + [j(action="done", claim="all good")] * 3)
    assert ep.status == "unverified" and not ep.passed and ep.false_success
    assert "not verified" in ep.turns[1].result


def test_the_task_checker_not_the_agent_decides_pass():
    s, _, _ = desktop()
    ep = run(s, [j(action="done", claim="saved the file")], checker=lambda: (False, "no file on disk"))
    assert ep.status == "done" and not ep.passed and ep.false_success and ep.checker == (False, "no file on disk")


def test_a_refused_route_goes_back_to_the_brain_with_the_reason():
    s, clicks, _ = desktop()
    ep = run(s, [j(action="click", x=5, y=5), j(action="click", target="the brush"), j(action="done", claim="ok")])
    assert "say WHAT" in ep.turns[0].result and not ep.turns[0].ok
    assert len(clicks) == 1 and ep.status == "done"


def test_a_second_plan_is_refused_and_names_the_step_to_act_on():
    s, clicks, _ = desktop()
    plan = j(action="plan", steps=["Pick the rectangle", "Draw"])
    ep = run(s, [plan, plan, j(action="click", element="u0"), j(action="done", claim="x")])
    assert "Do not plan again: act on step 1 now (Pick the rectangle)" in ep.turns[1].result
    assert not ep.turns[1].ok and len(clicks) == 1


def test_unusable_replies_and_repeated_failures_stop_the_run():
    s, clicks, _ = desktop()
    ep = run(s, ["not json"] * 5)
    assert ep.status == "gave_up" and clicks == []


def test_the_approval_gate_still_holds_for_a_click_found_by_the_eyes():
    s, clicks, _ = desktop()
    s.eyes = lambda jpeg, d: ((500, 60), 1.0)  # lands on "Send" (centre 1000,120 real = 500,60 shown)
    ep = run(s, [j(action="click", target="the send button"), j(action="done", claim="sent")])
    assert "did not approve" in ep.turns[0].result and clicks == []


def test_text_only_brain_never_receives_the_screenshot():
    s, _, _ = desktop()
    brain = ScriptedBrain([j(action="done", claim="x")])
    Agent(s, brain, Router("paper_blind", None), "g").run()
    from computeruse.agent.brains import claude_message, prompt_text

    view = brain.seen[0]
    assert "accessibility tree" in prompt_text(view)
    assert '"image"' not in claude_message(view)


def test_vision_arm_brain_gets_the_image_not_the_tree():
    s, clicks, _ = desktop()
    brain = ScriptedBrain([j(action="click", x=100, y=50), j(action="done", claim="x")])
    ep = Agent(s, brain, Router("vision"), "g").run()
    from computeruse.agent.brains import claude_message, prompt_text

    assert "accessibility tree" not in prompt_text(brain.seen[0]) and '"image"' in claude_message(brain.seen[0])
    assert ep.channels() == {"vision": 1} and clicks[0].coordinate == (200, 100)
