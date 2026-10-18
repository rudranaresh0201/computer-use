"""
Every number and figure in docs/paper/arxiv/ from the raw per-example files -- nothing typed by hand
except the dev learning curve, which only exists in the eval log (journal 2026-10-12).

    .venv\\Scripts\\python scripts\\paper_numbers.py

Reads   runs/hub/{eval,compare-v1,final/screenspot_v2}/  runs/tree_llm_test.jsonl
        runs/compare_v1/items.jsonl  data/gui_grounding/labels.jsonl  runs/realtest/*.uia.json
Writes  docs/paper/arxiv/numbers.json  docs/paper/arxiv/figures/*.pdf (+ .png previews)

New analyses beyond the journal:
  decomposition   fallback - vision = (help - harm) / n exactly, where help = the tree answered and was
                  right while vision was wrong, harm = the tree answered and was wrong while vision was
                  right. The tree's coverage (how often it answers) and its precision when it answers
                  explain each condition's loss.
  gate audit      how the LLM-over-tree answers split into NONE / named pick / nameless pick, and how
                  precise each kind is: why requiring a name is the right abstention signal.
  grounder        hit rate by control type, by target-size quartile, miss distance from the box edge.
  exposure        how much of 8 real apps' windows UIA covers with named elements (a proxy, not Test D).
"""

from __future__ import annotations

import glob
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs" / "paper" / "arxiv"
FIG = OUT / "figures"

# dev click accuracy (200-example seeded subset) by training step, from the eval's selection log
LEARNING_CURVE = [(0, 4.0), (100, 42.5), (200, 45.5), (300, 53.0), (400, 55.0), (500, 61.0), (600, 63.0),
                  (700, 64.5), (800, 65.5), (900, 69.5), (1000, 68.0), (1100, 72.5), (1179, 73.0)]


def jl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def boot(xs, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [round(100 * means[int(0.025 * reps)], 1), round(100 * means[int(0.975 * reps)], 1)]


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def decompose(rows, tree_point, tree_hit, vis_hit):
    """rows -> coverage, precision, help, harm, fallback - vision (all in % of rows)."""
    n = len(rows)
    answered = [r for r in rows if tree_point(r) is not None]
    help_ = sum(1 for r in answered if tree_hit(r) and not vis_hit(r))
    harm = sum(1 for r in answered if not tree_hit(r) and vis_hit(r))
    return {"n": n, "coverage": pct(len(answered), n),
            "precision_when_answering": pct(sum(tree_hit(r) for r in answered), len(answered)),
            "vision_on_answered": pct(sum(vis_hit(r) for r in answered), len(answered)),
            "help": pct(help_, n), "harm": pct(harm, n), "fallback_minus_vision": pct(help_ - harm, n)}


def test_a():
    scored = json.load(open(ROOT / "runs/hub/compare-v1/scored.json"))
    out = {}
    for c in ("C1", "C2", "C3", "C4", "C5"):
        rows = [r for r in scored if r["condition"] == c]
        out[c] = decompose(rows, lambda r: r["points"]["tree"], lambda r: r["hits"]["tree"], lambda r: r["hits"]["vlm"])
        out[c]["accuracy"] = {arm: pct(sum(r["hits"][arm] for r in rows), len(rows)) for arm in rows[0]["hits"]}
    out["all"] = decompose(scored, lambda r: r["points"]["tree"], lambda r: r["hits"]["tree"], lambda r: r["hits"]["vlm"])
    return out


def test_b():
    from computeruse.eval import real_tree

    labels = jl(ROOT / "data/gui_grounding/labels.jsonl")
    scored = real_tree.run(labels, jl(ROOT / "runs/hub/eval/results/step-01179.jsonl"))
    out = {}
    for split in ("test_same_app", "test_held_out_app"):
        for cond in real_tree.CONDITIONS:
            rows = [s for s in scored if s["split"] == split and s["condition"] == cond]
            d = decompose(rows, lambda r: r["points"]["tree"], lambda r: r["hits"]["tree"], lambda r: r["hits"]["vlm"])
            d["accuracy"] = {arm: pct(sum(r["hits"][arm] for r in rows), len(rows)) for arm in real_tree.ARMS}
            out[f"{split}/{cond}"] = d
    return out


def gate_audit():
    """The 250 LLM-over-tree answers (first 50 test pages) against the grounder's answers on them."""
    llm = {r["id"]: r for r in jl(ROOT / "runs/tree_llm_test.jsonl")}
    vis = {r["example_id"]: r for r in jl(ROOT / "runs/hub/compare-v1/results/step-01179.jsonl")}
    out = {}
    for c in ("C1", "C2", "C3", "C4", "C5", "all"):
        ids = [i for i, r in llm.items() if c == "all" or r["condition"] == c]
        kinds = defaultdict(list)
        for i in ids:
            r = llm[i]
            kind = "none" if r["ref"] is None else ("named" if (r["name"] or "").strip() else "nameless")
            kinds[kind].append((r["hit"], vis[i]["hit"]))
        out[c] = {k: {"share": pct(len(v), len(ids)), "llm_precision": pct(sum(a for a, _ in v), len(v)),
                      "vision_on_these": pct(sum(b for _, b in v), len(v))} for k, v in kinds.items()}
        out[c]["n"] = len(ids)
        out[c]["tokens_per_item"] = round(statistics.mean(llm[i]["usage"]["prompt_tokens"] + llm[i]["usage"]["completion_tokens"] for i in ids))
    return out


def grounder():
    labels = {r["id"]: r for r in jl(ROOT / "data/gui_grounding/labels.jsonl")}
    ft = jl(ROOT / "runs/hub/eval/results/step-01179.jsonl")
    zs = {r["example_id"]: r for r in jl(ROOT / "runs/hub/eval/results/zero_shot.jsonl")}
    out = {}
    for split in ("test_same_app", "test_held_out_app"):
        rows = [r for r in ft if r["split"] == split]
        by_type = defaultdict(list)
        for r in rows:
            by_type[labels[r["example_id"]]["element"]["control_type"]].append((r["hit"], zs[r["example_id"]]["hit"]))
        types = {t: {"n": len(v), "fine_tuned": pct(sum(a for a, _ in v), len(v)), "zero_shot": pct(sum(b for _, b in v), len(v))}
                 for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])) if len(v) >= 15}
        area = lambda r: (r["ground_truth_bbox_norm"][2] - r["ground_truth_bbox_norm"][0]) * (r["ground_truth_bbox_norm"][3] - r["ground_truth_bbox_norm"][1])  # noqa: E731
        ordered = sorted(rows, key=area)
        q = len(ordered) // 4
        quart = []
        for i in range(4):
            part = ordered[i * q:(i + 1) * q if i < 3 else len(ordered)]
            quart.append({"n": len(part), "median_area_units2": statistics.median(map(area, part)),
                          "fine_tuned": pct(sum(r["hit"] for r in part), len(part)),
                          "zero_shot": pct(sum(zs[r["example_id"]]["hit"] for r in part), len(part))})

        def edge(r):
            x, y = r["predicted_point"]
            x1, y1, x2, y2 = r["ground_truth_bbox_norm"]
            return ((max(x1 - x, 0, x - x2)) ** 2 + (max(y1 - y, 0, y - y2)) ** 2) ** 0.5

        misses = [edge(r) for r in rows if not r["hit"] and r["predicted_point"]]
        out[split] = {"n": len(rows), "by_control_type": types, "by_size_quartile": quart,
                      "misses": len(misses), "miss_near_le10": pct(sum(d <= 10 for d in misses), len(misses)),
                      "miss_mid_10_50": pct(sum(10 < d <= 50 for d in misses), len(misses)),
                      "miss_far_gt50": pct(sum(d > 50 for d in misses), len(misses)),
                      "median_target_area_pct_of_screen": round(statistics.median(map(area, rows)) / 1e4, 3)}
    fmt = Counter()
    for r in zs.values():
        t = r["predicted_text"]
        fmt["box" if len(re.findall(r"-?\d+", t)) >= 4 else ("point" if r["predicted_point"] else "none")] += 1
    out["zero_shot_answer_formats"] = dict(fmt)
    return out


def screenspot():
    rep = json.load(open(ROOT / "runs/hub/final/screenspot_v2/report.json"))
    cells = ["mobile/text", "mobile/icon", "desktop/text", "desktop/icon", "web/text", "web/icon"]
    return {arm: {c: round(100 * rep["arms"][arm][c]["accuracy"], 1) for c in cells} for arm in ("zero_shot", "fine_tuned")}


def exposure():
    """Named interactive UIA elements on 8 real apps, and the share of the window they cover."""
    interactive = {"Button", "MenuItem", "TabItem", "CheckBox", "RadioButton", "ComboBox", "Edit", "ListItem",
                   "TreeItem", "Hyperlink", "SplitButton", "Slider", "Spinner", "DataItem"}
    out = {}
    for f in sorted(glob.glob(str(ROOT / "runs/realtest/*.uia.json"))):
        u = json.load(open(f, encoding="utf-8"))
        app = Path(f).name.split("_2026")[0]
        wx1, wy1, wx2, wy2 = u["window"]["rect"]
        W, H = u["real_size"]
        wx1, wy1, wx2, wy2 = max(wx1, 0), max(wy1, 0), min(wx2, W), min(wy2, H)
        els = u["elements"]
        named = [e for e in els if e["control_type"] in interactive and (e.get("name") or "").strip()]
        nameless = [e for e in els if e["control_type"] in interactive and not (e.get("name") or "").strip()]
        grid = set()  # union of named boxes on a 10-px grid
        for e in named:
            x1, y1, x2, y2 = e["rect"]
            for gx in range(max(x1, wx1) // 10, min(x2, wx2) // 10):
                for gy in range(max(y1, wy1) // 10, min(y2, wy2) // 10):
                    grid.add((gx, gy))
        win = max(1, ((wx2 - wx1) // 10) * ((wy2 - wy1) // 10))
        out[app] = {"elements": len(els), "interactive_named": len(named), "interactive_nameless": len(nameless),
                    "window_area_covered_by_named_pct": round(100 * len(grid) / win, 1)}
    return out


def figures(n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    FIG.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.pdf")
        fig.savefig(FIG / f"{name}.png", dpi=170)
        plt.close(fig)

    # 1. learning curve
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    xs, ys = zip(*LEARNING_CURVE)
    ax.plot(xs, ys, marker="o", ms=3, color="#2b5fab")
    ax.set_xlabel("training step")
    ax.set_ylabel("dev click accuracy (%)")
    ax.set_ylim(0, 80)
    ax.axvline(393, ls=":", c="gray", lw=0.8)
    ax.axvline(786, ls=":", c="gray", lw=0.8)
    ax.text(196, 5, "ep 1", ha="center", color="gray", fontsize=7)
    ax.text(590, 5, "ep 2", ha="center", color="gray", fontsize=7)
    ax.text(983, 5, "ep 3", ha="center", color="gray", fontsize=7)
    save(fig, "learning_curve")

    # 2. Test A heatmap
    arms = ["tree", "vlm_zero_shot", "vlm", "fallback", "snap"]
    names = {"tree": "tree (name match)", "vlm_zero_shot": "vision, zero-shot", "vlm": "vision, fine-tuned",
             "fallback": "fallback (tree, else vision)", "snap": "snap"}
    conds = ["C1", "C2", "C3", "C4", "C5"]
    M = [[n["test_a"][c]["accuracy"][a] for c in conds] for a in arms]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(5), ["C1\nnamed", "C2\nunnamed", "C3\nreworded", "C4\ncanvas", "C5\nambig."], fontsize=7)
    ax.set_yticks(range(len(arms)), [names[a] for a in arms], fontsize=7)
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 60 else "black")
    ax.spines[:].set_visible(False)
    save(fig, "test_a_heatmap")

    # 3. help vs harm per condition (Test A) -- the decomposition
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    h = [n["test_a"][c]["help"] for c in conds]
    k = [-n["test_a"][c]["harm"] for c in conds]
    ax.bar(conds, h, color="#3c9f6b", label="help: tree right, vision wrong")
    ax.bar(conds, k, color="#c8473f", label="harm: tree wrong, vision right")
    for i, c in enumerate(conds):
        ax.text(i, min(k[i], 0) - 4, f"{n['test_a'][c]['fallback_minus_vision']:+.0f}", ha="center", fontsize=7)
    ax.axhline(0, c="black", lw=0.6)
    ax.set_ylabel("% of items")
    ax.set_ylim(-35, 15)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "test_a_help_harm")

    # 4. Test B (held-out apps)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    states = ["intact", "nameless", "absent"]
    barms = ["tree", "vlm", "fallback", "snap", "guarded"]
    colors = ["#7a8597", "#2b5fab", "#c8473f", "#e0a43a", "#3c9f6b"]
    w = 0.16
    for j, (a, col) in enumerate(zip(barms, colors)):
        vals = [n["test_b"][f"test_held_out_app/{s}"]["accuracy"][a] for s in states]
        ax.bar([i + (j - 2) * w for i in range(3)], vals, w, label=a, color=col)
    ax.set_xticks(range(3), ["tree intact", "target nameless", "target absent"], fontsize=7)
    ax.set_ylabel("click accuracy (%)")
    ax.axhline(n["test_b"]["test_held_out_app/intact"]["accuracy"]["vlm"], ls=":", c="#2b5fab", lw=0.8)
    ax.legend(fontsize=6, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "test_b")

    # 5. size quartiles
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    qs = n["grounder"]["test_held_out_app"]["by_size_quartile"]
    ax.bar([i - 0.18 for i in range(4)], [q["zero_shot"] for q in qs], 0.36, color="#9aa6bb", label="zero-shot")
    ax.bar([i + 0.18 for i in range(4)], [q["fine_tuned"] for q in qs], 0.36, color="#2b5fab", label="fine-tuned")
    ax.set_xticks(range(4), ["Q1\nsmallest", "Q2", "Q3", "Q4\nlargest"], fontsize=7)
    ax.set_ylabel("hit rate (%), unseen apps")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "size_quartiles")

    # 6. ScreenSpot-v2 cells
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    cells = list(n["screenspot_v2"]["zero_shot"])
    ax.bar([i - 0.18 for i in range(6)], [n["screenspot_v2"]["zero_shot"][c] for c in cells], 0.36, color="#9aa6bb", label="zero-shot")
    ax.bar([i + 0.18 for i in range(6)], [n["screenspot_v2"]["fine_tuned"][c] for c in cells], 0.36, color="#2b5fab", label="fine-tuned")
    ax.set_xticks(range(6), [c.replace("/", "\n") for c in cells], fontsize=7)
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "screenspot_v2")


def main():
    n = {"test_a": test_a(), "test_b": test_b(), "gate_audit": gate_audit(), "grounder": grounder(),
         "screenspot_v2": screenspot(), "uia_exposure": exposure(), "learning_curve": LEARNING_CURVE}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1), encoding="utf-8")
    figures(n)
    for c in ("C1", "C2", "C3", "C4", "C5", "all"):
        print("A", c, {k: v for k, v in n["test_a"][c].items() if k != "accuracy"})
    for k, v in n["test_b"].items():
        print("B", k, {kk: vv for kk, vv in v.items() if kk != "accuracy"})
    print(json.dumps(n["gate_audit"], indent=0)[:1600])
    print(json.dumps(n["grounder"]["test_held_out_app"], indent=0)[:1800])
    print(n["grounder"]["zero_shot_answer_formats"])
    print(json.dumps(n["uia_exposure"], indent=0))


if __name__ == "__main__":
    main()
