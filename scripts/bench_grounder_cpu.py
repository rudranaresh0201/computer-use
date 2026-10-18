"""
Can the eyes run live on this laptop's CPU? Latency AND accuracy of the grounder per image size,
on held-out-app test examples (apps the adapter never trained on -- the case the eyes are for).

    .venv\\Scripts\\python scripts\\bench_grounder_cpu.py --n 12 --edges 1568,1024,768
    .venv\\Scripts\\python scripts\\bench_grounder_cpu.py --remote --n 40      # the Kaggle eyes (P1b)

--remote asks the eyes server (CU_EYES_URL / CU_EYES_TOKEN) instead of loading the model here: the
same examples, so a remote hit rate near the paper's 46.5 says the live eyes are the scored model.

Shrinking the screenshot cuts visual tokens (one per 28x28 px), so it is faster; whether it
still hits small buttons is what the hit column answers. Writes runs/bench/grounder_cpu.json.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from PIL import Image

from computeruse.eval.vlm_grounder import is_hit_normalized, normalize_bbox
from computeruse.eyes import Grounder, fit
from computeruse.training.dataset import resolve_path
from computeruse.training.prepare_dataset import COORD_NORM_RANGE

ROOT = Path("data/gui_grounding")


class RemoteGrounder:
    """Grounder.locate() over the wire, so the loop below scores both the same way."""

    long_edge = 0

    def __init__(self, eyes) -> None:
        self.eyes = eyes

    def locate(self, image, instruction):
        import io
        from types import SimpleNamespace

        buf = io.BytesIO()
        image.save(buf, "PNG")  # lossless: the server sees the pixels the eval scored
        start = time.perf_counter()
        answer = self.eyes.locate(buf.getvalue(), instruction)
        point = tuple(answer["point"]) if answer.get("point") else None
        return SimpleNamespace(point=point, raw=answer.get("raw", ""), seconds=time.perf_counter() - start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--edges", default="1568,1024,768")
    ap.add_argument("--split", default="test_held_out_app")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--bf16", action="store_true", help="skip 8-bit quantization (needs ~5.4 GB free)")
    ap.add_argument("--remote", action="store_true", help="ask the eyes server instead (the server picks the size)")
    args = ap.parse_args()
    if args.remote:
        args.edges = "0"

    labels = {r["id"]: r for r in map(json.loads, (ROOT / "labels.jsonl").open(encoding="utf-8"))}
    rows = [json.loads(l) for l in (ROOT / "splits" / f"{args.split}.jsonl").open(encoding="utf-8")]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]

    if args.remote:
        from computeruse.eyes_remote import RemoteEyes

        grounder = RemoteGrounder(RemoteEyes.from_env())
        print(f"remote eyes: {grounder.eyes.health()}", flush=True)
        report = {"split": args.split, "n": len(rows), "weights": "remote", "edges": {}}
    else:
        import psutil

        grounder = Grounder(threads=args.threads, int8=not args.bf16).load()
        rss = psutil.Process().memory_info().rss / 2**30
        print(f"loaded in {grounder.load_seconds:.0f}s, {'int8' if grounder.int8 else 'bf16'}, {rss:.1f} GB resident", flush=True)
        report = {"split": args.split, "n": len(rows), "load_seconds": grounder.load_seconds,
                  "weights": "int8" if grounder.int8 else "bf16", "resident_gb": round(rss, 2), "edges": {}}
    for edge in map(int, args.edges.split(",")):
        grounder.long_edge = edge
        times, hits = [], []
        for row in rows:
            image = Image.open(resolve_path(ROOT, row["image_path"])).convert("RGB")
            found = grounder.locate(image, row["prompt"])
            label = labels[row["id"]]
            box = normalize_bbox(tuple(label["element"]["bbox_real"]), tuple(label["real_size"]),
                                 label["scale_x"], label["scale_y"])
            norm = None
            if found.point:
                norm = (round(found.point[0] / image.width * COORD_NORM_RANGE),
                        round(found.point[1] / image.height * COORD_NORM_RANGE))
            times.append(found.seconds)
            hits.append(is_hit_normalized(norm, box))
            print(f"  {edge:>5} {found.seconds:5.1f}s {'hit ' if hits[-1] else 'miss'} {row['prompt'][:50]!r} -> {found.raw!r}", flush=True)
        size = fit(image.size, edge) if edge else image.size  # 0: remote, the server sizes it
        report["edges"][edge] = {"size": size, "median_s": statistics.median(times), "max_s": max(times),
                                 "hit_rate": sum(hits) / len(hits)}
        print(f"edge {edge} {size}: median {statistics.median(times):.1f}s, hits {sum(hits)}/{len(hits)}", flush=True)
    out = Path(f"runs/bench/grounder_cpu_{report['weights']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report["when"] = time.strftime("%Y-%m-%d %H:%M")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["edges"], indent=2))


if __name__ == "__main__":
    main()
