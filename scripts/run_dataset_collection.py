"""
Runs the Phase 3 dataset orchestrator against the real dataset
(data/gui_grounding/), walking every app in the registry. Resumable: apps and
states already recorded in progress.json are skipped, so this can be re-run
after fixing one app (e.g. Windows Terminal, 2026-07-14) without recollecting
everything else. A failure launching or driving one app is logged and
skipped, not fatal to the run -- see orchestrator.run()'s docstring.

Run: .venv/Scripts/python.exe scripts/run_dataset_collection.py
"""

from pathlib import Path

from computeruse.action.controller import ActionController
from computeruse.action.schema import ActionType
from computeruse.dataset.orchestrator import run
from computeruse.safety import ActionLogger, SafetyConfig, SafetyGuard

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "gui_grounding"
REGISTRY_PATH = DATASET_ROOT / "apps.yaml"
RUN_DIR = REPO_ROOT / "runs" / "dataset_collection"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    config = SafetyConfig(allowed_actions=set(ActionType), max_actions_per_minute=60)
    logger = ActionLogger(RUN_DIR / "actions.jsonl")
    guard = SafetyGuard(config, logger)
    controller = ActionController(guard)

    ledger = run(
        registry_path=REGISTRY_PATH,
        dataset_root=DATASET_ROOT,
        controller=controller,
        session_id="collection-run",
        apps_filter=None,
        resume=True,
    )

    by_app: dict[str, list[str]] = {}
    for entry in sorted(ledger.completed):
        app, state = entry.split(":", 1)
        by_app.setdefault(app, []).append(state)

    print("\n== Collection run complete ==")
    for app, states in by_app.items():
        print(f"  {app}: {len(states)} states done ({', '.join(states)})")
    print(f"\nTotal apps with at least one collected state: {len(by_app)}")
    print(f"Action log: {RUN_DIR / 'actions.jsonl'}")


if __name__ == "__main__":
    main()
