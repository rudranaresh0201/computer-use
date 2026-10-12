"""Tests C/D (realtest/): classifying a person's box against the full UIA tree, and the label page's storage."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from computeruse.realtest import classify, label


def capture(elements):
    return {"id": "vlc_1", "app": "vlc", "scale_x": 2.0, "scale_y": 2.0, "scaled_size": [1000, 500],
            "real_size": [2000, 1000], "elements": elements}


def el(i, name, rect, kind="Button", depth=1):
    return {"i": i, "parent": 0, "depth": depth, "name": name, "control_type": kind, "automation_id": None, "rect": rect}


def test_a_box_is_named_unnamed_or_absent_and_a_window_never_counts():
    elements = [el(0, "VLC", [0, 0, 2000, 1000], "Window", 0), el(1, "Play", [100, 100, 160, 140]),
                el(2, "", [300, 100, 340, 140])]
    assert classify.status_of((102, 98, 158, 142), elements)[0] == "named"
    assert classify.status_of((300, 100, 340, 140), elements)[0] == "unnamed"
    assert classify.status_of((800, 600, 840, 640), elements)[0] == "absent"  # only the whole window covers it


def test_classify_reads_captures_and_labels_into_items_and_coverage(tmp_path):
    elements = [el(0, "Play", [100, 100, 160, 140]), el(1, "", [300, 100, 340, 140])]
    (tmp_path / "vlc_1.uia.json").write_text(json.dumps(capture(elements)))
    (tmp_path / "vlc_1.png").write_bytes(b"png")
    labels = [{"box": [50, 50, 80, 70], "instruction": "play it"}, {"box": [150, 50, 170, 70], "instruction": "the gear"},
              {"box": [400, 300, 420, 320], "instruction": "seek forward"}]
    (tmp_path / "vlc_1.labels.json").write_text(json.dumps(labels))
    items, coverage = classify.classify(tmp_path)
    assert [i["condition"] for i in items] == ["named", "unnamed", "absent"]
    assert items[0]["gt_box_norm"] == [50, 100, 80, 140] and items[0]["uia_name"] == "Play"
    assert coverage["vlc"]["exposed"] == 2 / 3 and coverage["vlc"]["absent"] == 1


def test_the_tree_is_given_to_the_tree_methods_in_screenshot_units():
    (play,) = classify.tree_elements(capture([el(0, "Play", [100, 100, 160, 140])]))
    assert (play.ref, play.role, play.name) == ("u0", "button", "Play")
    assert play.box == (50.0, 100.0, 30.0, 40.0)
    assert '- button "Play" [ref=u0]' in classify.tree_text(capture([el(0, "Play", [100, 100, 160, 140])]))


def test_the_label_page_saves_and_serves_labels_and_refuses_unknown_ids(tmp_path):
    (tmp_path / "vlc_1.uia.json").write_text(json.dumps(capture([])))
    (tmp_path / "vlc_1.png").write_bytes(b"\x89PNG")
    server = ThreadingHTTPServer(("127.0.0.1", 0), label.make_handler(tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        body = json.dumps([{"box": [1, 2, 30, 40], "instruction": "pause"}, {"box": [1, 2], "instruction": "bad"}]).encode()
        urllib.request.urlopen(urllib.request.Request(f"{base}/api/labels/vlc_1", data=body, method="POST"))
        assert json.loads(urllib.request.urlopen(f"{base}/api/labels/vlc_1").read()) == [{"box": [1, 2, 30, 40], "instruction": "pause"}]
        assert json.loads(urllib.request.urlopen(f"{base}/api/captures").read()) == [{"id": "vlc_1", "labels": 1}]
        try:
            urllib.request.urlopen(f"{base}/img/..%2F..%2Fsecret.png")
            raise AssertionError("path traversal was served")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
