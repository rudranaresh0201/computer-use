"""
After the last GPU run: everything that needs no GPU, in one go.

    1. guard design  -- every candidate in eval/guards.py on the DEV pages; pick by the fixed rule
    2. guard test    -- the chosen rule, once, on the 250-item test subset (Test A pages p1000-p1049)
    3. Test C / D    -- tree methods vs the grounder on boxes a person drew in real apps (if labelled)

    python scripts/after_final_run.py     (after: hf download of final/, runs/guard_dev/tree_llm.jsonl complete)
"""

from __future__ import annotations

import json
from pathlib import Path

from computeruse.eval import compare, guards
from computeruse.eval.report import bootstrap_ci, clustered_paired_difference
from computeruse.eval.vlm_grounder import is_hit_normalized
from computeruse.web.browser import parse_snapshot

HUB = Path("runs/hub")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_guards(root: Path, vlm_path: Path, llm_path: Path) -> list[dict]:
    items = _read(root / "items.jsonl")
    viewport = tuple(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["viewport"])
    vlm = {r["example_id"]: r["predicted_point"] for r in _read(vlm_path)}
    llm = {r["id"]: r for r in _read(llm_path)}
    trees, rows = {}, []
    for item in items:
        if item["id"] not in llm or item["id"] not in vlm:
            continue
        if item["tree"] not in trees:
            trees[item["tree"]] = parse_snapshot((root / item["tree"]).read_text(encoding="utf-8"))
        point = tuple(vlm[item["id"]]) if vlm[item["id"]] else None
        truth = tuple(item["gt_box_norm"])
        hits = {"vlm": is_hit_normalized(point, truth), "tree_llm": llm[item["id"]]["hit"]}
        for rule in guards.GUARDS:
            hits[rule] = is_hit_normalized(guards.decide(rule, llm[item["id"]], point, trees[item["tree"]], viewport), truth)
        rows.append({"id": item["id"], "page": item["page_id"], "condition": item["condition"], "hits": hits})
    return rows


def table(rows: list[dict], arms, title: str) -> dict[str, float]:
    print(f"\n{title} (n={len(rows)})")
    print(f"{'arm':12s}" + "".join(f"{c:>15s}" for c in compare.CONDITIONS) + f"{'mean':>8s}")
    means = {}
    for arm in arms:
        cells, accs = "", []
        for c in compare.CONDITIONS:
            hits = [r["hits"][arm] for r in rows if r["condition"] == c]
            if not hits:
                cells += f"{'-':>15s}"
                continue
            lo, hi = bootstrap_ci(hits)
            accs.append(sum(hits) / len(hits))
            cells += f"{100 * accs[-1]:6.1f} [{100 * lo:3.0f},{100 * hi:3.0f}]"
        means[arm] = sum(accs) / len(accs)
        print(f"{arm:12s}{cells}{100 * means[arm]:8.1f}")
    return means


def diff(rows, a, b) -> str:
    r = clustered_paired_difference({s["id"]: s["hits"][a] for s in rows}, {s["id"]: s["hits"][b] for s in rows},
                                    {s["id"]: s["page"] for s in rows})
    return f"{a} - {b}: {100 * r['diff']:+.1f} [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}] (pages resampled)"


def main() -> None:
    dev = score_guards(Path("runs/guard_dev"), HUB / "final/guard_dev/results/step-01179.jsonl", Path("runs/guard_dev/tree_llm.jsonl"))
    means = table(dev, ("vlm", "tree_llm", *guards.GUARDS), "1. DEV pages (seeds 100-159): guard design")
    chosen = guards.choose({rule: means[rule] for rule in guards.GUARDS})
    print(f"\nchosen on dev by the fixed rule: {chosen}")

    test = score_guards(Path("runs/compare_v1"), HUB / "compare-v1/results/step-01179.jsonl", Path("runs/tree_llm_test.jsonl"))
    table(test, ("vlm", "tree_llm", "llm_guarded", chosen), f"2. TEST subset, scored once: {chosen}")
    for other in ("vlm", "llm_guarded"):
        if other != chosen:
            print("  " + diff(test, chosen, other))

    if (HUB / "final/realtest/results/step-01179.jsonl").exists():
        print("\n3. Test C: see scripts/score_realtest.py (needs the LLM-over-UIA run on the labelled captures)")


if __name__ == "__main__":
    main()
