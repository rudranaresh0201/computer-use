"""
`computeruse run "<goal>"` — wires a ScriptedBrain (the only Brain that
exists until Phase 3/4) into the real StateGraph and runs it to completion,
printing the resulting trace. Proves the harness itself works end-to-end;
not a general "do this goal" command yet, since no real decision-making
Brain exists until later phases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .action.controller import ActionController
from .action.schema import Action, ActionType
from .brain.scripted import ScriptedBrain
from .dataset import orchestrator as dataset_orchestrator
from .orchestrator.graph import build_graph
from .safety import ActionLogger, SafetyConfig, SafetyGuard


def _initial_state(goal: str) -> dict:
    return {
        "goal": goal,
        "screenshot": None,
        "ui_elements": [],
        "action_history": [],
        "done": False,
        "step_count": 0,
        "consecutive_failures": 0,
    }


def run(goal: str) -> None:
    brain = ScriptedBrain([Action(type=ActionType.DONE)])
    guard = SafetyGuard(SafetyConfig(), ActionLogger(Path("logs/cli_run.jsonl")))
    controller = ActionController(guard)
    graph = build_graph(brain, controller)

    final_state = graph.invoke(_initial_state(goal))

    print(f"goal: {goal!r}")
    print(f"steps: {final_state['step_count']}")
    print(f"done: {final_state['done']}")
    print("actions taken:")
    for action in final_state["action_history"]:
        print(f"  - {action.type.value}")


def collect_dataset(
    registry: Path,
    dataset_root: Path,
    apps: list[str] | None,
    resume: bool,
) -> None:
    import logging
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    guard = SafetyGuard(SafetyConfig(), ActionLogger(Path("logs/dataset_collection.jsonl")))
    controller = ActionController(guard)
    session_id = time.strftime("%Y-%m-%dT%H-%M-%S")

    ledger = dataset_orchestrator.run(
        registry_path=registry,
        dataset_root=dataset_root,
        controller=controller,
        session_id=session_id,
        apps_filter=apps,
        resume=resume,
    )
    print(f"session: {session_id}")
    print(f"states collected so far (all sessions): {len(ledger.completed)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="computeruse")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the agent toward a goal")
    run_parser.add_argument("goal", help="Natural-language goal for the agent")

    collect_parser = subparsers.add_parser(
        "collect-dataset", help="Walk the app registry and collect GUI-grounding labels"
    )
    collect_parser.add_argument(
        "--registry", type=Path, default=Path("data/gui_grounding/apps.yaml")
    )
    collect_parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/gui_grounding")
    )
    collect_parser.add_argument(
        "--apps", type=str, default=None, help="Comma-separated app names to restrict to"
    )
    collect_parser.add_argument(
        "--no-resume", action="store_true", help="Ignore progress.json and start fresh"
    )

    web_parser = subparsers.add_parser(
        "web", help="Drive Chrome by hand through the agent's own eyes and hands"
    )
    web_parser.add_argument("url", nargs="?", default=None, help="Page to open first")
    web_parser.add_argument("--headless", action="store_true", help="Run Chrome without a window")

    web_run_parser = subparsers.add_parser(
        "web-run", help="The agent does a web task on its own in Chrome (brain: Groq, free tier)"
    )
    web_run_parser.add_argument("goal", help='e.g. "When was Alan Turing born? Use Wikipedia."')
    web_run_parser.add_argument("--start", default=None, help="Page to open first, e.g. en.wikipedia.org")
    web_run_parser.add_argument("--model", default=None, help="Groq model id (default: openai/gpt-oss-120b)")
    web_run_parser.add_argument("--max-steps", type=int, default=15)
    web_run_parser.add_argument("--headless", action="store_true", help="No Chrome window")
    web_run_parser.add_argument("--no-video", action="store_true", help="Don't record runs/videos/*.webm")

    capture_parser = subparsers.add_parser(
        "capture", help="Save a screenshot + the full UIA tree of the app you click on (Tests C/D)"
    )
    capture_parser.add_argument("app", help="a short name for the app, e.g. vlc, audacity, spotify")
    capture_parser.add_argument("--delay", type=int, default=5, help="seconds to click on the app's window")
    label_parser = subparsers.add_parser("label", help="Open the page for boxing what a person can click (Tests C/D)")
    label_parser.add_argument("--port", type=int, default=8765)
    subparsers.add_parser("coverage", help="Test D: how much of what a person can click does UIA expose, per app")

    args = parser.parse_args()

    if args.command == "capture":
        from .realtest.capture import capture

        capture(args.app, delay=args.delay)
    elif args.command == "label":
        from .realtest.label import serve

        serve(port=args.port)
    elif args.command == "coverage":
        from .realtest.capture import OUT_DIR
        from .realtest.classify import classify

        items, coverage = classify(OUT_DIR)
        print(f"{len(items)} labels across {len(coverage)} apps\n")
        print(f"{'app':<18}{'labels':>8}{'named':>8}{'unnamed':>9}{'absent':>8}{'UIA exposes':>14}")
        for app, c in coverage.items():
            print(f"{app:<18}{c['labels']:>8}{c['named']:>8}{c['unnamed']:>9}{c['absent']:>8}{c['exposed']:>13.0%}")
    elif args.command == "web-run":
        from .web.run import DEFAULT_MODEL
        from .web.run import main as web_run_main

        sys.exit(web_run_main(args.goal, args.start, args.model or DEFAULT_MODEL, args.max_steps,
                              args.headless, record=not args.no_video))
    elif args.command == "web":
        from .web.shell import main as web_main

        web_main(args.url, headless=args.headless)
    elif args.command == "run":
        run(args.goal)
    elif args.command == "collect-dataset":
        apps_filter = args.apps.split(",") if args.apps else None
        collect_dataset(args.registry, args.dataset_root, apps_filter, resume=not args.no_resume)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
