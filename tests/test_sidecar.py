import io
import json
import threading

from computeruse.sidecar import Sidecar, step_kind


def fake_runner(goal, app, on_step, should_stop):
    on_step('click u3 Button "Seven"', "")
    on_step("click 'u99'", "refused: that id is not in the current list")
    on_step("done", "7")
    return {"status": "done", "answer": "7", "actions": 3}


def messages(out):
    return [json.loads(line) for line in out.getvalue().splitlines()]


def sidecar(runner=fake_runner):
    out = io.StringIO()
    return Sidecar(runner, ["calculator", "paint"], out), out


def test_every_line_is_one_jsonrpc_message_and_a_run_streams_steps_then_finished():
    s, out = sidecar()
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"goal": "type 7", "app": "calculator"}}))
    s.wait(5)
    msgs = messages(out)
    assert all(m["jsonrpc"] == "2.0" for m in msgs)
    assert msgs[0]["id"] == 1 and msgs[0]["result"] == {"started": True}  # the reply comes first
    methods = [m.get("method") for m in msgs if "method" in m]
    assert methods == ["started", "step", "step", "step", "finished"]
    steps = [m["params"] for m in msgs if m.get("method") == "step"]
    assert [p["kind"] for p in steps] == ["click", "noop", "done"] and [p["n"] for p in steps] == [1, 2, 3]
    assert msgs[-1]["params"]["answer"] == "7"


def test_bad_input_gets_jsonrpc_errors_not_a_crash():
    s, out = sidecar()
    s.handle("not json")
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "fly"}))
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "run", "params": {"goal": "x", "app": "steam"}}))
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "run", "params": {"goal": "  ", "app": "paint"}}))
    codes = [m["error"]["code"] for m in messages(out)]
    assert codes == [-32700, -32601, -32602, -32602]


def test_a_notification_gets_no_reply():
    s, out = sidecar()
    s.handle(json.dumps({"jsonrpc": "2.0", "method": "ping"}))
    assert out.getvalue() == ""


def test_stop_reaches_a_running_agent_and_a_second_run_is_refused_while_busy():
    started, seen_stop = threading.Event(), threading.Event()

    def slow_runner(goal, app, on_step, should_stop):
        started.set()
        while not should_stop():
            pass
        seen_stop.set()
        return {"status": "stopped", "answer": "STOPPED: stopped by the person"}

    s, out = sidecar(slow_runner)
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"goal": "a", "app": "paint"}}))
    assert started.wait(5)
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "run", "params": {"goal": "b", "app": "paint"}}))
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "stop"}))
    s.wait(5)
    by_id = {m["id"]: m for m in messages(out) if "id" in m}
    assert by_id[2]["error"]["code"] == -32000 and by_id[3]["result"] == {"stopping": True}
    assert seen_stop.is_set() and messages(out)[-1]["params"]["status"] == "stopped"


def test_a_crash_inside_the_agent_still_ends_with_finished():
    def broken(goal, app, on_step, should_stop):
        raise RuntimeError("calc.exe did not open")

    s, out = sidecar(broken)
    s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"goal": "a", "app": "calculator"}}))
    s.wait(5)
    last = messages(out)[-1]
    assert last["method"] == "finished" and last["params"]["status"] == "error"
    assert "did not open" in last["params"]["answer"]


def test_closing_stdin_stops_the_run():
    started = threading.Event()

    def runner(goal, app, on_step, should_stop):
        started.set()
        while not should_stop():
            pass
        return {"status": "stopped", "answer": "x"}

    s, out = sidecar(runner)
    line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"goal": "a", "app": "paint"}})
    s.serve(io.StringIO(line + "\n"))  # returns only once the run has ended
    assert messages(out)[-1]["method"] == "finished"


def test_step_kind():
    assert step_kind("type '7'", "") == "type" and step_kind("key ctrl+s", "") == "key"
    assert step_kind("(no action)", "the reply was not a valid tool call") == "noop"
