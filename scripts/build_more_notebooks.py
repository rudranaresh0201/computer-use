"""
Build the notebooks for the paper's last training runs from the two that already
ran on Kaggle, so the proven cells are reused, not rewritten:

    notebooks/phase3_more_train_kaggle.ipynb   <- phase3_train_kaggle.ipynb
    notebooks/phase3_more_eval_kaggle.ipynb    <- phase3_eval_kaggle.ipynb

One setting, RUN, picks the experiment:
    "seed2"                   the full v3 run again with training seed 7 (robustness)
    "n100" "n200" "n500" "n1000"   a nested subset of train, 3 epochs (data scaling)
Each run gets its own Hub repo, computer-use-grounder-<RUN>.

Usage: python scripts/build_more_notebooks.py <CODE_REF>
Every edit asserts that its anchor text was found exactly once, so a change to
the source notebooks fails here instead of producing a silently wrong notebook.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

RUNS_DOC = """\
| `RUN` | train examples | seed | steps (3 epochs) | snapshot every | ~T4 hours |
|---|---|---|---|---|---|
| `"seed2"` | 1572 (all) | 7 | 1179 | 100 | ~8 |
| `"n100"` | 100 | 42 | 75 | 15 | ~0.6 |
| `"n200"` | 200 | 42 | 150 | 25 | ~1.1 |
| `"n500"` | 500 | 42 | 375 | 50 | ~2.6 |
| `"n1000"` | 1000 | 42 | 750 | 100 | ~5.2 |

Subsets are nested (the 100 are inside the 200, inside the 500 ...), fixed by `train_subset_indices`.
The original run (seed 42, all 1572) is the reference point both studies are compared to."""

SETTINGS_RUNS = '''\
RUN = "seed2"                      # "seed2", "n100", "n200", "n500", "n1000" -- see the table in the first cell
RUNS = {  # name: (train_limit or None for all 1572, training seed, snapshot/eval every N steps)
    "seed2": (None, 7, 100),
    "n100": (100, 42, 15),
    "n200": (200, 42, 25),
    "n500": (500, 42, 50),
    "n1000": (1000, 42, 100),
}
assert RUN in RUNS, RUN
TRAIN_LIMIT, TRAIN_SEED, EVERY = RUNS[RUN]
HF_REPO_NAME = f"computer-use-grounder-{RUN}"   # one repo per experiment, never the original run's'''


def load(name: str) -> dict:
    return json.loads((NB / name).read_text(encoding="utf-8"))


def src(cell: dict) -> str:
    return "".join(cell["source"])


def set_src(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def edit(cell: dict, old: str, new: str) -> None:
    text = src(cell)
    assert text.count(old) == 1, f"anchor found {text.count(old)} times: {old[:80]!r}"
    set_src(cell, text.replace(old, new))


def clean(nb: dict) -> dict:
    nb = copy.deepcopy(nb)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return nb


def build_train(code_ref: str) -> dict:
    nb = clean(load("phase3_train_kaggle.ipynb"))
    c = nb["cells"]
    set_src(c[0], f"""\
# Phase 3, last runs: a second seed and data scaling (Kaggle, sessions that survive)

Same notebook as `phase3_train_kaggle.ipynb` (the run that produced the paper's numbers), with one setting,
`RUN`, that picks the experiment. Everything else -- the Hub checkpoints, resume, wall-clock stop -- is unchanged.

{RUNS_DOC}

**Why these two:** a reviewer's first question about one training run is "was it a lucky seed?" (`seed2`),
and the second is "how much data does this need?" (`n100` ... `n1000`).

## Setup (same as before)
Secret `HF_TOKEN` (write) attached, **GPU T4 x2**, Internet **On**, the v3 dataset attached (and nothing else with a `labels.jsonl`).

## How to run (Save Version -> "Save & Run All (Commit)")
| # | `RUN_MODE` | Ends with |
|---|---|---|
| 1 | `"smoke"` | `SMOKE PART 1 OK` (stops at step 3) |
| 2 | `"smoke"` | `resuming from step 3` ... `SMOKE PASSED` |
| 3 | `"train"` | `TRAINING FINISHED` (or "commit again" if the clock ran out) |

Smoke once per code version is enough; after that, set `RUN`, `RUN_MODE = "train"` and commit.
When a run says `TRAINING FINISHED`, evaluate it with `phase3_more_eval_kaggle.ipynb` (same `RUN`).
""")
    edit(c[1], 'HF_REPO_NAME = "computer-use-grounder"\n', SETTINGS_RUNS + "\n")
    edit(c[1], 'CODE_REF = "d1e135af1bbc31818fd869636fe31b6d980c2aa3"',
         f'CODE_REF = "{code_ref}"')
    edit(c[1], 'print(f"RUN_MODE={RUN_MODE}  CODE_REF={CODE_REF}  budget={BUDGET_HOURS} h")',
         'print(f"RUN={RUN} (train_limit={TRAIN_LIMIT}, seed={TRAIN_SEED})  RUN_MODE={RUN_MODE}  '
         'CODE_REF={CODE_REF}  budget={BUDGET_HOURS} h")')
    edit(c[7], "    TOTAL_STEPS, SAVE_STEPS, EVAL_STEPS, EVAL_SUBSET = 1179, 100, 100, 120  # 393 steps/epoch x 3",
         "    # Trainer: steps/epoch = examples // (batch 1 x accumulation 4); 3 epochs. Every RUN size is a\n"
         "    # multiple of 4, so this is exact (1572 -> 1179, 1000 -> 750, 500 -> 375, 200 -> 150, 100 -> 75).\n"
         "    n_examples = TRAIN_LIMIT or n_train\n"
         "    assert n_examples % 4 == 0, n_examples\n"
         "    TOTAL_STEPS, SAVE_STEPS, EVAL_STEPS, EVAL_SUBSET = 3 * (n_examples // 4), EVERY, EVERY, 120")
    edit(c[8], "hub_cb = HubCheckpointCallback(sync, snapshot_every=100)",
         "hub_cb = HubCheckpointCallback(sync, snapshot_every=SAVE_STEPS)")
    edit(c[8], "        callbacks=[clock_cb, nan_cb, hub_cb],",
         "        seed=TRAIN_SEED,\n"
         "        train_limit=TRAIN_LIMIT,\n"
         "        callbacks=[clock_cb, nan_cb, hub_cb],")
    edit(c[8], '    print(f"GPU memory after model load',
         '    assert len(trainer.train_dataset) == (TRAIN_LIMIT or n_train), len(trainer.train_dataset)\n'
         '    print(f"GPU memory after model load')
    edit(c[10], 'print("TRAINING FINISHED. Adapter at final/ and adapters/step-* on the Hub. Next: evaluation.")',
         'print(f"TRAINING FINISHED ({RUN}). Next: phase3_more_eval_kaggle.ipynb with RUN = {RUN!r}.")')
    return nb


def build_eval(code_ref: str) -> dict:
    nb = clean(load("phase3_eval_kaggle.ipynb"))
    c = nb["cells"]
    set_src(c[0], f"""\
# Phase 3, last runs: evaluate a second seed or a data-scaling run (Kaggle, resumable)

The same protocol as `phase3_eval_kaggle.ipynb`, on the run `RUN` names:
1. **Choose on dev only**: every adapter snapshot is scored on the same seeded 200-example dev subset;
   the best click accuracy wins, and the choice is pushed to the Hub before any test example is scored.
2. **Test once**: the chosen adapter on the test splits.

{RUNS_DOC}

- `seed2` runs the full protocol (zero-shot + chosen adapter on dev and both test splits, H1/H3 report), so it can
  be compared with the original run line by line.
- Scaling runs skip zero-shot (the base model does not change between runs; its numbers come from the original run)
  and score the chosen adapter on the two **test** splits only: ~1.5 h instead of ~3.7 h each.

Setup and smoke are the same as before: `RUN_MODE = "smoke"` once, then `"eval"`. If a session ends early, commit again.
""")
    edit(c[1], 'HF_REPO_NAME = "computer-use-grounder"   # the training run\'s repo: adapters come from it, results go back to it\n',
         SETTINGS_RUNS + "\nSCALING = RUN != \"seed2\"           # scaling runs: no zero-shot, test splits only\n")
    edit(c[1], 'CODE_REF = "a27a0814d6674148f589226d1b2933c4ea15a319"', f'CODE_REF = "{code_ref}"')
    edit(c[1], 'print(f"RUN_MODE={RUN_MODE}  CODE_REF={CODE_REF}")',
         'print(f"RUN={RUN} (scaling={SCALING})  RUN_MODE={RUN_MODE}  CODE_REF={CODE_REF}")')
    edit(c[8], "for arm in [ZERO_SHOT] + ARMS:", "for arm in ([] if SCALING else [ZERO_SHOT]) + ARMS:")
    edit(c[8], 'print("\\nzero-shot answer formats on this subset:", format_counts(dev_results[ZERO_SHOT]))\n'
               'for r in dev_results[ZERO_SHOT][:5]:\n',
         'for r in ([] if SCALING else dev_results[ZERO_SHOT][:5]):\n')
    edit(c[9], "if RUN_MODE == \"eval\":\n    FINAL_EXAMPLES = [e for s in EVAL_SPLITS for e in SPLIT_EXAMPLES[s]]",
         "TEST_SPLITS = [s for s in EVAL_SPLITS if s != \"dev\"]\n"
         "if RUN_MODE == \"eval\" and SCALING:\n"
         "    FINAL_EXAMPLES = [e for s in TEST_SPLITS for e in SPLIT_EXAMPLES[s]]\n"
         "elif RUN_MODE == \"eval\":\n"
         "    FINAL_EXAMPLES = [e for s in EVAL_SPLITS for e in SPLIT_EXAMPLES[s]]")
    edit(c[9], "for arm in (ZERO_SHOT, CHOSEN):", "for arm in ((CHOSEN,) if SCALING else (ZERO_SHOT, CHOSEN)):")
    edit(c[9], 'print(f"stage 2 done: {len(FINAL_EXAMPLES)} examples x 2 arms")',
         'print(f"stage 2 done: {len(FINAL_EXAMPLES)} examples x {len(final_results)} arm(s)")')
    report = src(c[10])
    set_src(c[10], '''\
# Cell 10 -- the report, and what it says.
if SCALING:
    from computeruse.eval.report import bootstrap_ci

    by_split = {}
    for split in (TEST_SPLITS if RUN_MODE == "eval" else ["dev"]):
        hits = [r.hit for r in final_results[CHOSEN] if r.split == split]
        by_split[split] = {"n": len(hits), "accuracy": sum(hits) / max(len(hits), 1), "ci95": list(bootstrap_ci(hits))}
    REPORT = {"run": RUN, "train_limit": TRAIN_LIMIT, "seed": TRAIN_SEED, "chosen": CHOSEN,
              "curve": CURVE, "selection": SELECTION, "by_split": by_split}
    Path("/kaggle/working/eval_report.json").write_text(json.dumps(REPORT, indent=2))
    store.push_json("report.json", REPORT)
    print(f"\\n{RUN}: {TRAIN_LIMIT} train examples, chosen adapter {CHOSEN} (on dev)")
    for split, d in by_split.items():
        print(f"  {split:<20} {d['accuracy']:6.1%}  [{d['ci95'][0]:.1%}, {d['ci95'][1]:.1%}]  n={d['n']}")
    hours = (time.time() - SESSION_START) / 3600
    print("\\n" + "=" * 70)
    print(f"{'SMOKE PASSED' if RUN_MODE == 'smoke' else 'EVAL FINISHED'} in {hours:.2f} h. "
          f"Report: {HF_REPO}/{PREFIX}/report.json")
    print("=" * 70)
else:
''' + "".join("    " + ln if ln.strip() else ln for ln in report.splitlines(keepends=True)[1:]))
    return nb


def main() -> None:
    code_ref = sys.argv[1]
    assert len(code_ref) == 40, "pass the full 40-character commit the notebooks should check out"
    for name, nb in (("phase3_more_train_kaggle.ipynb", build_train(code_ref)),
                     ("phase3_more_eval_kaggle.ipynb", build_eval(code_ref))):
        (NB / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote notebooks/{name} (CODE_REF {code_ref[:7]})")


if __name__ == "__main__":
    main()
