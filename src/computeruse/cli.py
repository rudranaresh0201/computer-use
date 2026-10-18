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

    desktop_parser = subparsers.add_parser(
        "desktop-run", help="The LangGraph agent does a task in a Windows app (brain: Groq, eyes: UIA)"
    )
    desktop_parser.add_argument("goal", help='e.g. "What is 123 times 45?"')
    desktop_parser.add_argument("--app", default="calculator", help="calculator | notepad_pp | paint")
    desktop_parser.add_argument("--model", default="openai/gpt-oss-20b", help="Groq model id")
    desktop_parser.add_argument("--max-steps", type=int, default=20)

    sidecar_parser = subparsers.add_parser(
        "sidecar", help="The desktop agent as a child process: JSON-RPC over stdin/stdout, for the app window"
    )
    sidecar_parser.add_argument("--model", default="openai/gpt-oss-20b", help="Groq model id")
    sidecar_parser.add_argument("--max-steps", type=int, default=20)
    sidecar_parser.add_argument("--demo", action="store_true", help="Scripted steps, no mouse or keyboard")
    sidecar_parser.add_argument("--brain", choices=["claude", "groq"], default="groq",
                                help="claude: Claude Code drives the MCP server (your subscription); groq: the free fallback")

    subparsers.add_parser("mcp", help="The desktop as an MCP server over stdio, for Claude Code or any MCP host")
    subparsers.add_parser("chrome", help="Open the agent's own Chrome profile to sign in (Google, Tinkercad) once")

    capture_parser = subparsers.add_parser(
        "capture", help="Save a screenshot + the full UIA tree of the app you click on (Tests C/D)"
    )
    capture_parser.add_argument("app", help="a short name for the app, e.g. vlc, audacity, spotify")
    capture_parser.add_argument("--delay", type=int, default=5, help="seconds to click on the app's window")
    label_parser = subparsers.add_parser("label", help="Open the page for boxing what a person can click (Tests C/D)")
    label_parser.add_argument("--port", type=int, default=8765)
    subparsers.add_parser("coverage", help="Test D: how much of what a person can click does UIA expose, per app")

    bench_parser = subparsers.add_parser("bench", help="The A/B/C/D comparison on the sandbox suite (agent/bench.py)")
    bench_parser.add_argument("--brain", choices=["claude", "groq"], default="groq")
    bench_parser.add_argument("--arm", choices=["paper", "paper_blind", "vision", "tree", "both"], default="paper_blind")
    bench_parser.add_argument("--eyes", choices=["none", "remote", "real"], default="none",
                              help="remote: our grounder on a GPU (CU_EYES_URL/CU_EYES_TOKEN); real: on this CPU")
    bench_parser.add_argument("--tasks", default="signup,note,mixer,archive,connect")
    bench_parser.add_argument("--seeds", type=int, default=3, help="seeds 1..N")
    bench_parser.add_argument("--model", default=None)
    bench_parser.add_argument("--headed", action="store_true", help="show the browser")

    serve_parser = subparsers.add_parser("eyes-serve", help="Serve our grounder over HTTP from a GPU (eyes_remote.py)")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--device", default="cuda")
    serve_parser.add_argument("--long-edge", type=int, default=1568, help="1568 = the size the eval scored")

    args = parser.parse_args()

    if args.command == "bench":
        from .agent.bench import run_bench

        path, summary = run_bench(args.brain, args.arm, args.tasks.split(","), list(range(1, args.seeds + 1)),
                                  eyes_kind=args.eyes, model=args.model, headed=args.headed)
        print(f"\n{path}\npass {summary['pass_rate']:.0%} {summary['pass_ci95']}  false success "
              f"{summary['false_success']}/{summary['claimed_done']}  channels {summary['channels']}")
    elif args.command == "eyes-serve":
        from .eyes import Grounder
        from .eyes_remote import grounder_locate, serve

        grounder = Grounder(long_edge=args.long_edge, device=args.device).load()
        server = serve(grounder_locate(grounder), port=args.port,
                       info={"device": args.device, "long_edge": args.long_edge})
        print(f"eyes loaded in {grounder.load_seconds:.0f}s; serving on :{args.port}")
        print(f"token: {server.token}")
        server.serve_forever()
    elif args.command == "capture":
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
    elif args.command == "desktop-run":
        from .desktop_run import main as desktop_main

        sys.exit(desktop_main(args.goal, args.app, args.model, args.max_steps))
    elif args.command == "chrome":
        from .web.cdp import PORT, PROFILE, endpoint, launch
        from .web_surface import SIGN_IN

        if endpoint(PORT):
            print("The agent's Chrome is already running. Sign in there, then leave it open or close it.")
        else:
            launch(urls=SIGN_IN)
            print(f"Opened the agent's Chrome (profile {PROFILE}).")
            print("Sign in to Google and Tinkercad in it, by hand. The agent uses this profile only;")
            print("your everyday Chrome is untouched. You can close the window afterwards.")
    elif args.command == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
    elif args.command == "sidecar":
        from .sidecar import main as sidecar_main

        sidecar_main(args.model, args.max_steps, demo=args.demo, brain=args.brain)
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
