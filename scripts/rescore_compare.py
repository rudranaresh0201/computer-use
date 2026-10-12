"""
Re-score Test A (sandbox, 1,000 items) on CPU from the grounder's saved
answers, adding arms that were frozen after the Kaggle run started:

    guarded      tree only for one exact name, else the grounder (compare.py, frozen d665c19)
    tree_llm     an LLM reads the tree and picks a ref (tree_llm.py, frozen 938f46a; 250-item subset)
    llm_guarded  the LLM's pick, else -- when it says NONE -- the grounder (pre-registered 1ad9836)

The arms the notebook already scored must reproduce exactly; that check runs first.

    python scripts/rescore_compare.py runs/compare_v1 runs/hub/compare-v1 runs/tree_llm_test.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from computeruse.eval import compare
from computeruse.eval.report import bootstrap_ci, clustered_paired_difference
from computeruse.eval.vlm_grounder import is_hit_normalized
from computeruse.web.browser import parse_snapshot


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(x: float) -> float:
    return round(100 * x, 1)


def main(root: Path, hub: Path, llm_path: Path) -> None:
    items = _read(root / "items.jsonl")
    viewport = tuple(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["viewport"])
    vlm = {r["example_id"]: r["predicted_point"] for r in _read(hub / "results" / "step-01179.jsonl")}
    zero = {r["example_id"]: r["predicted_point"] for r in _read(hub / "results" / "zero_shot.jsonl")}
    old = {s["id"]: s for s in json.loads((hub / "scored.json").read_text(encoding="utf-8"))}
    llm = {r["id"]: r for r in _read(llm_path)}

    trees, scored = {}, []
    for item in items:
        if item["tree"] not in trees:
            trees[item["tree"]] = parse_snapshot((root / item["tree"]).read_text(encoding="utf-8"))
        p, z = vlm[item["id"]], zero[item["id"]]
        s = compare.score_item(item, trees[item["tree"]], tuple(p) if p else None, tuple(z) if z else None, viewport)
        for arm, hit in old[item["id"]]["hits"].items():
            assert s["hits"][arm] == hit, f"{item['id']} {arm} does not reproduce the Kaggle run"
        if item["id"] in llm:
            answer = llm[item["id"]]
            pick = tuple(answer["point"]) if answer["point"] else None
            point = pick if pick is not None else (tuple(p) if p else None)
            s["hits"]["tree_llm"] = answer["hit"]
            s["hits"]["llm_guarded"] = is_hit_normalized(point, tuple(item["gt_box_norm"]))
        s["page"] = item["page_id"]
        scored.append(s)
    print(f"reproduced the Kaggle run's {len(old)} items x {len(next(iter(old.values()))['hits'])} arms exactly\n")

    def table(rows, arms, title):
        print(title)
        print(f"{'arm':12s}" + "".join(f"{c:>16s}" for c in compare.CONDITIONS) + f"{'mean':>8s}")
        for arm in arms:
            cells, accs = "", []
            for c in compare.CONDITIONS:
                hits = [r["hits"][arm] for r in rows if r["condition"] == c]
                lo, hi = bootstrap_ci(hits)
                accs.append(sum(hits) / len(hits))
                cells += f"{_pct(accs[-1]):6.1f} [{_pct(lo):3.0f},{_pct(hi):3.0f}]"
            print(f"{arm:12s}{cells}{_pct(sum(accs) / len(accs)):8.1f}")
        print()

    table(scored, compare.ARMS, "Test A, all 1,000 items (200 per condition)")
    subset = [s for s in scored if "llm_guarded" in s["hits"]]
    table(subset, ("tree", "vlm", "guarded", "tree_llm", "llm_guarded"), "LLM subset, 250 items (50 per condition)")

    def diff(rows, a, b, label):
        r = clustered_paired_difference({s["id"]: s["hits"][a] for s in rows}, {s["id"]: s["hits"][b] for s in rows},
                                        {s["id"]: s["page"] for s in rows})
        print(f"  {label:44s} {_pct(r['diff']):+6.1f}  page-clustered CI [{_pct(r['ci95'][0])}, {_pct(r['ci95'][1])}]  "
              f"n={r['n']}, pages={r['clusters']}")

    print("Differences (all conditions pooled; pages resampled whole)")
    for a, b in (("vlm", "tree"), ("snap", "tree"), ("snap", "vlm"), ("guarded", "vlm"), ("fallback", "vlm"),
                 ("vlm", "vlm_zero_shot")):
        diff(scored, a, b, f"{a} - {b} (1,000)")
    for a, b in (("llm_guarded", "vlm"), ("llm_guarded", "tree_llm"), ("llm_guarded", "guarded")):
        diff(subset, a, b, f"{a} - {b} (250)")
    for c in compare.CONDITIONS:
        rows = [s for s in subset if s["condition"] == c]
        diff(rows, "llm_guarded", "vlm", f"llm_guarded - vlm, {c} (50)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
