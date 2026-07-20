"""
Runs the Phase 3 dataset orchestrator against the real dataset
(data/gui_grounding/), walking every app in the registry. Resumable: apps and
states already recorded in progress.json are skipped, so this can be re-run
after fixing one app (e.g. Windows Terminal, 2026-07-14) without recollecting
everything else. A failure launching or driving one app is logged and
skipped, not fatal to the run -- see orchestrator.run()'s docstring.

DO NOT USE THE MACHINE WHILE THIS RUNS. It drives real windows, moves focus
and sends keystrokes. Touching the keyboard or mouse steals focus from the
target app -- which is exactly how 196 of the previous dataset's 542 rows
ended up being the code editor's UI labeled as "file_explorer" and
"audacity" (docs/journal.md, 2026-07-19). The collector now detects this and
raises instead of writing bad rows, so the cost is a skipped state rather
than a poisoned dataset -- but a skipped state is still a state you have to
re-collect.

Run:
    .venv/Scripts/python.exe scripts/run_dataset_collection.py
    .venv/Scripts/python.exe scripts/run_dataset_collection.py --apps file_explorer,audacity

task_manager and device_manager additionally require an ELEVATED session
(Start -> right-click Terminal -> "Terminal (Admin)"). Verify elevation
directly before trusting it -- a UAC prompt appearing upstream does not mean
the shell you are in is elevated (docs/journal.md, 2026-07-15).
"""

import argparse
import time
from pathlib import Path

from computeruse.action.controller import ActionController
from computeruse.action.schema import ActionType
from computeruse.dataset.orchestrator import run
from computeruse.safety import ActionLogger, SafetyConfig, SafetyGuard

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "gui_grounding"
REGISTRY_PATH = DATASET_ROOT / "apps.yaml"
RUN_DIR = REPO_ROOT / "runs" / "dataset_collection"

# The default 60/min exists for autonomous agent runs, where a runaway loop
# is the thing being guarded against. This script is a fixed, finite,
# fully-scripted walk of the registry: at 212 screenshots with 2-3 drive
# steps each it will exceed 60/min within the first app, and SafetyGuard
# *raises* on the limit -- which the orchestrator catches per state and
# skips, silently losing states. Raised deliberately, with the run still
# bounded by the registry rather than by this number.
MAX_ACTIONS_PER_MINUTE = 600

COUNTDOWN_SECONDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apps",
        default=None,
        help="comma-separated app names to collect (default: every app in the registry)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore progress.json and re-collect every state",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the hands-off countdown"
    )
    args = parser.parse_args()

    apps_filter = [a.strip() for a in args.apps.split(",")] if args.apps else None

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not args.yes:
        print("This run will take over the mouse and keyboard.")
        print("DO NOT touch the machine until it prints 'Collection run complete'.")
        print(f"Collecting: {', '.join(apps_filter) if apps_filter else 'all apps'}")
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"  starting in {remaining}...", end="\r", flush=True)
            time.sleep(1)
        print(" " * 40, end="\r")

    config = SafetyConfig(
        allowed_actions=set(ActionType), max_actions_per_minute=MAX_ACTIONS_PER_MINUTE
    )
    logger = ActionLogger(RUN_DIR / "actions.jsonl")
    guard = SafetyGuard(config, logger)
    controller = ActionController(guard)

    ledger = run(
        registry_path=REGISTRY_PATH,
        dataset_root=DATASET_ROOT,
        controller=controller,
        session_id="collection-run",
        apps_filter=apps_filter,
        resume=not args.no_resume,
    )

    by_app: dict[str, list[str]] = {}
    for entry in sorted(ledger.completed):
        app, state = entry.split(":", 1)
        by_app.setdefault(app, []).append(state)

    print("\n== Collection run complete ==")
    for app, states in by_app.items():
        print(f"  {app}: {len(states)} states done")
    print(f"\nTotal apps with at least one collected state: {len(by_app)}")
    print(f"Action log: {RUN_DIR / 'actions.jsonl'}")
    print()
    print("NEXT: python scripts/validate_dataset.py   <- must pass before training")


if __name__ == "__main__":
    main()
