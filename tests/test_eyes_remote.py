"""The remote eyes: a real HTTP server on localhost with a fake model behind it. Checks the wire
(image in, point out), the token, and that a dead server is "no answer", not a crashed episode."""

import threading

import pytest

from computeruse.eyes_remote import RemoteEyes, serve


@pytest.fixture
def server():
    seen = []

    def locate(image, instruction):
        seen.append((image, instruction))
        return ((412, 96), "(412,96)", 0.5) if "save" in instruction else (None, "File", 0.5)

    srv = serve(locate, port=0, token="s3cret", info={"device": "fake"}, host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_a_point_comes_back_for_the_image_that_was_sent(server):
    url, seen = server
    eyes = RemoteEyes(url, "s3cret")
    assert eyes.health() == {"ok": True, "device": "fake"}
    point, seconds = eyes(b"\xff\xd8jpeg-bytes", "the floppy-disk save icon")
    assert point == (412, 96) and seconds >= 0
    assert seen == [(b"\xff\xd8jpeg-bytes", "the floppy-disk save icon")]  # bytes survive the base64 trip


def test_no_readable_answer_is_none_with_the_raw_text_kept(server):
    eyes = RemoteEyes(server[0], "s3cret")
    assert eyes(b"x", "the trash icon")[0] is None and eyes.last_raw == "File"


def test_a_wrong_token_is_refused_and_the_model_never_runs(server):
    url, seen = server
    with pytest.raises(RuntimeError, match="401"):
        RemoteEyes(url, "guess").health()
    assert RemoteEyes(url, "guess")(b"x", "the save icon")[0] is None and seen == []


def test_an_unreachable_server_is_no_answer_not_a_crash():
    eyes = RemoteEyes("http://127.0.0.1:9", "t", timeout=2)
    point, _ = eyes(b"x", "the save icon")
    assert point is None and "unreachable" in eyes.last_raw
