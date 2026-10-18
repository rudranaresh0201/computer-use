"""The demo's pieces without a model: the Excalidraw checker on scenes shaped like the app's own
autosave, and the panel's event stream (a late panel replays what it missed)."""

import json
import urllib.request

import pytest

from computeruse.agent.checks import excalidraw_verdict, parse_connect
from computeruse.agent.panel import Panel


def box(i, label=None):
    els = [{"id": f"b{i}", "type": "rectangle"}]
    if label:
        els.append({"id": f"t{i}", "type": "text", "text": label, "containerId": f"b{i}"})
    return els


def arrow(a, b, bound=True):
    return {"id": f"a{a}{b}", "type": "arrow", "startBinding": {"elementId": f"b{a}"} if bound else None,
            "endBinding": {"elementId": f"b{b}"} if bound else None}


def test_labels_and_bound_arrows_pass():
    scene = box(1, "Brain") + box(2, "Router") + box(3, "Eyes") + [arrow(1, 2), arrow(2, 3)]
    ok, detail = excalidraw_verdict(scene, ["Brain", "Router", "Eyes"], [("Brain", "Router"), ("Router", "Eyes")])
    assert ok, detail
    assert "3 shapes, 2 arrows (2 bound at both ends)" in detail


def test_an_arrow_that_only_ends_near_a_box_does_not_connect_it():
    scene = box(1, "Brain") + box(2, "Router") + [arrow(1, 2, bound=False)]
    ok, detail = excalidraw_verdict(scene, ["Brain", "Router"], [("Brain", "Router")])
    assert not ok and "not connected ['Brain>Router']" in detail


def test_a_missing_label_and_deleted_shapes_fail():
    scene = box(1, "Brain") + [{"id": "b9", "type": "rectangle", "isDeleted": True},
                               {"id": "t9", "type": "text", "text": "Eyes", "containerId": "b9", "isDeleted": True}]
    ok, detail = excalidraw_verdict(scene, ["Brain", "Eyes"], [])
    assert not ok and "missing labels ['Eyes']" in detail


def test_the_arrow_direction_and_case_do_not_matter():
    scene = box(1, "the BRAIN") + box(2, "Router") + [arrow(2, 1)]
    assert excalidraw_verdict(scene, ["brain"], [("Brain", "Router")])[0]


def test_parse_connect():
    assert parse_connect("Brain>Router, Router > Eyes") == [("Brain", "Router"), ("Router", "Eyes")]
    with pytest.raises(ValueError):
        parse_connect("Brain-Router")


def test_the_goal_typed_into_the_panel_reaches_the_agent_once():
    import threading
    import urllib.error
    import urllib.request

    panel = Panel(port=0).start()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("goal", panel.ask("e.g. light an LED", timeout=10)))
    t.start()

    def post(goal):
        req = urllib.request.Request(panel.url + "/run", data=json.dumps({"goal": goal}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=5).status

    try:
        with pytest.raises(urllib.error.HTTPError):
            post("   ")  # an empty goal is refused and the box stays up
        assert post("Light a red LED from a coin cell") == 204
        t.join(5)
        assert got["goal"] == "Light a red LED from a coin cell"
        with pytest.raises(urllib.error.HTTPError) as second:
            post("another goal")
        assert second.value.code == 409  # one run per panel
        assert [n for n, _ in panel.history] == ["ask", "asked"]
    finally:
        panel.stop()


def test_the_panel_serves_its_page_and_replays_events_to_a_late_viewer():
    panel = Panel(port=0).start()
    try:
        panel("run", {"goal": "draw", "brain": "claude", "arm": "paper"})
        panel("screen", {"src": "data:old"})
        panel("screen", {"src": "data:new"})
        panel("route", {"intent": "click e5", "channel": "tree", "why": "named"})
        assert b"EventSource" in urllib.request.urlopen(panel.url, timeout=5).read()
        with urllib.request.urlopen(panel.url + "/events", timeout=5) as r:
            got = []
            while len(got) < 3:
                line = r.readline().decode().strip()
                if line.startswith("event:"):
                    name = line.split(": ", 1)[1]
                    data = json.loads(r.readline().decode().split(": ", 1)[1])
                    got.append((name, data))
        assert [g[0] for g in got] == ["run", "route", "screen"]
        assert got[2][1]["src"] == "data:new"  # only the latest screenshot is replayed
    finally:
        panel.stop()
