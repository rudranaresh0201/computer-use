"""DesktopSession: the rules behind the MCP tools, with a fake screen, tree and mouse."""

import os
from dataclasses import dataclass
from types import SimpleNamespace

from computeruse.action.schema import ActionType
from computeruse.desktop_session import DesktopSurface
from computeruse.session import Session, Shot


@dataclass
class El:
    name: str
    control_type: str
    center: tuple


def make(foreground="Untitled - Paint", focus_ok=True, screens=None, home=None, approve=None):
    done, events = [], []
    screens = list(screens or [])
    tree = [El("Rectangle", "Button", (900, 120)), El("", "Button", (950, 120)), El("File name", "Edit", (700, 900)),
            El("Delete", "Button", (1000, 120))]

    def shoot():
        small = screens.pop(0) if screens else bytes([0]) * 16
        # the model sees a 1280x720 image of a 2560x1440 screen
        return Shot(b"jpeg", (1280, 720), lambda x, y: (x * 2, y * 2), small)

    desktop = DesktopSurface(
        apps={"paint": (["mspaint.exe"], "Paint")},
        tree=lambda: tree, shoot=shoot,
        execute=lambda a: done.append(a) or SimpleNamespace(success=True, error=None),
        foreground=lambda: foreground, focus=lambda w: focus_ok, launch=lambda app: "Paint",
    )
    s = Session(surfaces={"paint": desktop}, events=lambda m, p: events.append((m, p)), sleep=lambda s: None,
                **({"home": home} if home else {}), **({"approve": approve} if approve else {}))
    return s, done, events


def changed_screens(n):
    return [bytes([0]) * 16 if i % 2 == 0 else bytes([200]) * 16 for i in range(n)]


def test_nothing_happens_before_an_app_is_open():
    s, done, _ = make()
    assert "open(target) first" in s.act("click", x=10, y=10) and done == []


def test_observe_lists_elements_by_id_sends_the_screen_and_returns_the_image():
    s, _, events = make()
    s.open("paint")
    text, image = s.observe()
    assert '[u0] Button "Rectangle"' in text and "1280x720" in text and image == b"jpeg"
    assert any(m == "screen" and p["src"].startswith("data:image/jpeg;base64,") for m, p in events)


def test_a_named_element_is_clicked_at_its_centre_through_the_tree_channel():
    s, done, events = make(screens=changed_screens(4))
    s.open("paint")
    s.observe()
    reply = s.act("click", element="u0")
    assert done[-1].type == ActionType.LEFT_CLICK and done[-1].coordinate == (900, 120)
    step = [p for m, p in events if m == "step"][-1]
    assert step["channel"] == "tree" and "Rectangle" in step["label"] and "changed" in reply
    assert 0 <= step["x"] <= 1 and 0 <= step["y"] <= 1


def test_a_nameless_element_is_refused_and_nothing_moves():
    s, done, _ = make()
    s.open("paint")
    s.observe()
    reply = s.act("click", element="u1")
    assert "has no name" in reply and "x, y" in reply and done == []


def test_screenshot_pixels_are_mapped_to_the_real_screen_on_the_vision_channel():
    s, done, events = make(screens=changed_screens(4))
    s.open("paint")
    s.observe()
    s.act("click", x=640, y=360)
    assert done[-1].coordinate == (1280, 720)
    step = [p for m, p in events if m == "step"][-1]
    assert step["channel"] == "vision" and (step["x"], step["y"]) == (0.5, 0.5)
    assert "outside" in s.act("click", x=2000, y=10)


def test_drag_draws_between_two_mapped_points_and_keys_go_through_the_api_channel():
    s, done, events = make(screens=changed_screens(8))
    s.open("paint")
    s.observe()
    s.act("drag", x=100, y=100, to_x=300, to_y=200)
    assert done[-1].type == ActionType.LEFT_CLICK_DRAG
    assert (done[-1].start_coordinate, done[-1].coordinate) == ((200, 200), (600, 400))
    s.act("key", keys="ctrl+s")
    assert done[-1].type == ActionType.KEY and done[-1].text == "ctrl+s"
    assert [p["channel"] for m, p in events if m == "step"] == ["vision", "api"]


def test_typing_into_a_named_field_clicks_it_first():
    s, done, _ = make(screens=changed_screens(4))
    s.open("paint")
    s.observe()
    s.act("type", element="u2", text="house.png")
    assert [a.type for a in done[-2:]] == [ActionType.LEFT_CLICK, ActionType.TYPE] and done[-1].text == "house.png"


def test_an_action_that_changes_nothing_says_so():
    s, _, _ = make(screens=[bytes([5]) * 16] * 4)
    s.open("paint")
    s.observe()
    assert "did NOT change" in s.act("click", element="u0")


def test_it_refuses_to_act_when_another_window_took_the_front_and_cannot_be_moved():
    s, done, _ = make(foreground="Inbox - Gmail", focus_ok=False)
    s.open("paint")
    s.surface.shot = Shot(b"", (1280, 720), lambda x, y: (x, y))
    assert "not in front" in s.act("click", x=1, y=1) and done == []


def test_a_step_is_ticked_only_by_a_verified_check(tmp_path):
    (tmp_path / "Pictures").mkdir()
    s, _, events = make(home=tmp_path)
    s.open("paint")
    s.set_plan(["Draw the house", "Save as house.png"])
    assert "NOT verified" in s.check(2, "file", "house.png") and 2 not in s.done
    (tmp_path / "Pictures" / "house.png").write_bytes(b"x" * 2048)
    reply = s.check(2, "file", "house.png")
    assert "verified" in reply and 2 in s.done
    receipts = [p for m, p in events if m == "receipt"]
    assert [r["verdict"] for r in receipts] == ["contradicted", "verified"]
    assert [p for m, p in events if m == "check"] == [{"step": 2}]


def test_an_old_file_does_not_count_as_this_task_saving_it(tmp_path):
    (tmp_path / "Pictures").mkdir()
    old = tmp_path / "Pictures" / "house.png"
    old.write_bytes(b"x" * 10)
    os.utime(old, (0, 0))
    s, _, _ = make(home=tmp_path)
    s.set_plan(["Save"])
    assert "before this task" in s.check(1, "file", "house.png")


def test_text_check_reads_the_live_tree():
    s, _, _ = make()
    s.open("paint")
    s.set_plan(["Pick the rectangle"])
    assert "verified" in s.check(1, "text", "rectangle")
    assert "NOT verified" in s.check(1, "text", "ellipse")


def test_changed_is_judged_against_the_screen_when_the_step_started_not_the_last_look():
    # live bug 2026-10-16: the agent drew, observed, then checked -- "changed since the last
    # observation" said no, six times, though the house was on the canvas
    screens = [bytes([0]) * 16] * 3 + [bytes([200]) * 16] * 10
    s, _, _ = make(screens=screens)
    s.open("paint")
    s.set_plan(["Draw the house"])  # baseline: blank
    s.observe()
    s.observe()                     # the model looks again after drawing...
    assert "verified" in s.check(1, "changed")  # ...and the step still counts


def test_an_irreversible_click_waits_for_the_person_and_a_no_means_nothing_happens():
    asked = []
    s, done, _ = make(screens=changed_screens(6), approve=lambda reason: asked.append(reason) or False)
    s.open("paint")
    s.observe()
    reply = s.act("click", element="u3")
    assert asked == ["clicking 'Delete'"] and "did not approve" in reply and done == []
    s.approve = lambda reason: True
    s.act("click", element="u3")
    assert done[-1].coordinate == (1000, 120)


def test_ordinary_clicks_never_ask():
    asked = []
    s, done, _ = make(screens=changed_screens(4), approve=lambda reason: asked.append(reason) or True)
    s.open("paint")
    s.observe()
    s.act("click", element="u0")
    assert asked == [] and len(done) == 1


def test_the_file_approver_waits_for_the_answer_file_and_treats_silence_as_no(tmp_path):
    from computeruse.session import file_approver

    events, now = [], [0.0]

    def sleep(d):
        now[0] += d
        if now[0] >= 1.0 and events:  # the person clicks Allow a second in
            aid = events[0][1]["id"]
            (tmp_path / "approvals").mkdir(exist_ok=True)
            (tmp_path / "approvals" / f"{aid}.json").write_text('{"ok": true}')

    ask = file_approver(str(tmp_path / "events.jsonl"), lambda m, p: events.append((m, p)),
                        timeout=5, clock=lambda: now[0], sleep=sleep)
    assert ask("sending this email") is True
    assert events[0] == ("approval", {"id": events[0][1]["id"], "action": "sending this email"})
    assert events[-1][0] == "approval-done" and events[-1][1]["ok"] is True

    silent = file_approver(str(tmp_path / "e2.jsonl"), lambda m, p: None, timeout=1,
                           clock=lambda: now[0], sleep=lambda d: now.__setitem__(0, now[0] + d))
    now[0] = 0.0
    assert silent("deleting a file") is False
    assert file_approver(None, lambda m, p: None)("anything") is False  # no app to ask: no
