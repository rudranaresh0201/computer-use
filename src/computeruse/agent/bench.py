"""
`computeruse bench`: the A/B/C/D comparison on the sandbox suite (suite.py), in a headless Chrome.

    computeruse bench --brain groq --arm paper_blind --eyes remote --seeds 3    # arm D, free brain
    computeruse bench --brain claude --arm vision --seeds 3                     # arm A
    computeruse bench --brain claude --arm tree --seeds 3                       # arm B
    computeruse bench --brain claude --arm paper --eyes remote --seeds 3        # arm C

--eyes remote: our grounder on a Kaggle T4 (eyes_remote.py); --eyes real: on this CPU (int8, G11).

Each episode is scored by the page's own checker, never by the agent's claim. Output:
runs/agent/<stamp>-<brain>-<arm>.jsonl (one episode per line, full trace) and a summary with the
numbers the paper and the blog need: pass rate [95% CI], pass-all-seeds per task (the reliability
gap), false-success rate, actions, seconds, tokens, and how often each channel was used.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .loop import Agent, Episode
from .router import Router
from .suite import TASKS, VIEWPORT, build

OUT = Path("runs/agent")


def grounder_eyes(long_edge: Optional[int] = None):
    """Our grounder as the Session's eyes: (jpeg, description) -> (point, seconds). Loads once."""
    from ..eyes import Grounder

    grounder = Grounder(**({"long_edge": long_edge} if long_edge else {})).load()

    def eyes(jpeg: bytes, description: str):
        found = grounder.locate(jpeg, description)
        return found.point, found.seconds

    return eyes


def run_episode(page, task_name: str, seed: int, brain, arm: str, eyes=None, workdir: Path = OUT / "pages",
                events: Callable[[str, dict], None] = lambda m, p: None, max_turns: int = 30) -> Episode:
    """One task, one seed, one fresh page."""
    from ..session import Session
    from ..web_surface import WebSurface

    task, url = build(task_name, seed, workdir)
    surface = WebSurface({task_name: (url, f"a small web app: {task_name}")}, page=lambda: page)
    session = Session(surfaces={task_name: surface}, events=events, sleep=lambda s: time.sleep(min(s, 0.3)),
                      eyes=eyes)
    session.open(task_name)

    def checker() -> tuple[bool, str]:
        ok, detail = page.evaluate("window.__verdict ? window.__verdict() : [false, 'no checker on the page']")
        return bool(ok), str(detail)

    router = Router(arm, (lambda d: session.ground_with_eyes(d)) if eyes else None)
    agent = Agent(session, brain, router, task.goal, checker=checker, max_turns=max_turns)
    episode = agent.run()
    episode.target = f"{task_name}#{seed}"
    return episode


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    from ..eval.report import bootstrap_ci

    passed = [bool(e["passed"]) for e in episodes]
    by_task: dict[str, list[bool]] = {}
    for e in episodes:
        by_task.setdefault(e["task"], []).append(bool(e["passed"]))
    claimed = [e for e in episodes if e["claim"]]
    channels: dict[str, int] = {}
    for e in episodes:
        for c, n in e["channels"].items():
            channels[c] = channels.get(c, 0) + n
    tokens = [sum(e["tokens"].values()) for e in episodes]
    return {
        "episodes": len(episodes),
        "pass_rate": sum(passed) / len(passed) if passed else 0.0,
        "pass_ci95": list(bootstrap_ci(passed)) if passed else None,
        "per_task": {t: {"passed": sum(v), "runs": len(v), "all_passed": all(v)} for t, v in sorted(by_task.items())},
        "tasks_all_seeds_passed": sum(all(v) for v in by_task.values()),
        "false_success": sum(bool(e["false_success"]) for e in episodes),
        "claimed_done": len(claimed),
        "median_actions": statistics.median(e["actions"] for e in episodes) if episodes else 0,
        "median_seconds": statistics.median(e["seconds"] for e in episodes) if episodes else 0,
        "tokens_total": sum(tokens),
        "channels": channels,
        "statuses": {s: sum(e["status"] == s for e in episodes) for s in {e["status"] for e in episodes}},
    }


def run_bench(brain_kind: str, arm: str, tasks: list[str], seeds: list[int], eyes_kind: str = "none",
              model: Optional[str] = None, headed: bool = False, brain=None, eyes=None,
              out: Path = OUT, log: Callable[[str], None] = print) -> tuple[Path, dict[str, Any]]:
    from playwright.sync_api import sync_playwright

    from .brains import make_brain

    brain = brain or make_brain(brain_kind, arm, model)
    if eyes is None and eyes_kind == "real":
        log("loading our grounder (CPU) ...")
        eyes = grounder_eyes()
    if eyes is None and eyes_kind == "remote":  # our grounder on a GPU elsewhere (eyes_remote.py)
        from ..eyes_remote import RemoteEyes

        eyes = RemoteEyes.from_env()
        log(f"eyes: {eyes.url} -> {eyes.health()}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{brain.name.replace('/', '_')}-{arm}.jsonl"
    rows: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not headed)  # the installed Chrome, as web/browser.py
        for seed in seeds:
            for name in tasks:
                page = browser.new_page(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]})
                try:
                    ep = run_episode(page, name, seed, brain, arm, eyes, out / "pages")
                    row = {"task": name, "seed": seed, **ep.to_json()}
                except Exception as e:  # noqa: BLE001 -- one broken episode is a recorded failure, not a lost bench
                    row = {"task": name, "seed": seed, "status": "error", "passed": False, "false_success": False,
                           "claim": "", "channels": {}, "tokens": {}, "actions": 0, "seconds": 0.0,
                           "error": f"{e.__class__.__name__}: {str(e)[:300]}", "arm": arm, "brain": brain.name}
                finally:
                    page.close()
                rows.append(row)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                verdict = "PASS" if row["passed"] else "fail"
                log(f"  {name:<8} seed {seed}: {verdict}  status={row['status']}  actions={row['actions']}  "
                    f"channels={row['channels']}  {row.get('checker') or row.get('error', '')}")
        browser.close()
    summary = summarize(rows)
    path.with_suffix(".summary.json").write_text(json.dumps({"brain": brain.name, "arm": arm, "eyes": eyes_kind,
                                                             "tasks": tasks, "seeds": seeds, **summary}, indent=2),
                                                 encoding="utf-8")
    return path, summary
