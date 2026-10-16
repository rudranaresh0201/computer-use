"""The claude brain: parsing Claude Code's stream-json, forwarding the MCP server's events, stopping."""

import json
import sys
import time
from pathlib import Path

from computeruse.claude_brain import mcp_config, result_from, run_claude, thoughts_from

FIXTURE = Path(__file__).parent / "fixtures" / "claude_stream.jsonl"  # recorded from a real claude -p run


def test_sentences_become_thoughts_and_the_result_becomes_the_ending():
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    thoughts = [t for ln in lines for t in thoughts_from(ln)]
    assert thoughts[0].startswith("I'll list") and len(thoughts) == 2
    result = [r for ln in lines if (r := result_from(ln))][0]
    assert result["status"] == "done" and "paint" in result["answer"]
    assert thoughts_from("not json") == [] and result_from('{"type":"assistant"}') is None


def test_the_server_gets_the_events_file_through_its_environment(tmp_path):
    cfg = mcp_config(tmp_path / "e.jsonl")["mcpServers"]["computeruse"]
    assert cfg["env"]["CU_EVENTS"].endswith("e.jsonl") and "mcp" in cfg["args"]


def fake_claude(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def test_a_run_forwards_thoughts_and_server_events_and_ends_with_the_result(tmp_path):
    stream = FIXTURE.read_text(encoding="utf-8")
    script = (
        "import os, json, sys, time\n"
        "with open(os.environ['CU_EVENTS'], 'a') as f:\n"
        "    f.write(json.dumps({'method': 'step', 'params': {'n': 1, 'kind': 'click', 'channel': 'tree', 'label': 'click u0'}}) + '\\n')\n"
        "time.sleep(0.4)\n"
        f"sys.stdout.write({stream!r})\n"
    )
    seen = []
    result = run_claude("draw", None, lambda m, p: seen.append((m, p)), lambda: False,
                        command=fake_claude(script), workdir=tmp_path)
    methods = [m for m, _ in seen]
    assert methods.count("thought") == 2 and "step" in methods
    assert result["status"] == "done" and result["actions"] == 1 and result["brain"] == "claude"


def test_stop_kills_the_process_tree_and_says_stopped(tmp_path):
    t0 = time.time()
    result = run_claude("draw", None, lambda m, p: None, lambda: time.time() - t0 > 0.5,
                        command=fake_claude("import time; time.sleep(60)"), workdir=tmp_path)
    assert result["status"] == "stopped" and time.time() - t0 < 15


def test_a_claude_that_dies_without_a_result_is_an_error_not_a_hang(tmp_path):
    result = run_claude("draw", None, lambda m, p: None, lambda: False,
                        command=fake_claude("import sys; sys.stderr.write('auth failed\\n'); sys.exit(3)"), workdir=tmp_path)
    assert result["status"] == "error" and "auth failed" in result["answer"]
