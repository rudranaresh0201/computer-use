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

    args = parser.parse_args()

    if args.command == "run":
        run(args.goal)
    elif args.command == "collect-dataset":
        apps_filter = args.apps.split(",") if args.apps else None
        collect_dataset(args.registry, args.dataset_root, apps_filter, resume=not args.no_resume)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
