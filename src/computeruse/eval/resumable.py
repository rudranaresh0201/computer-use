"""
Eval results that survive the session they were scored in.

The training run's lesson (training/hub_checkpoint.py), applied to the
ablation: a Kaggle commit keeps /kaggle/working only if the whole notebook
finishes, and scoring every arm over all three splits is hours of GPU. So
each scored example is appended to its arm's JSONL file, the files go to
the Hub every few minutes, and the next session pulls them back and skips
what is already scored. Decoding is greedy (vlm_grounder.greedy_config),
so an example scored twice gets the same answer -- a resumed arm is the
same arm, not a re-roll.

Hub layout, inside the training repo:

    <prefix>/results/<arm>.jsonl    one vlm_grounder.ArmResult per line
    <prefix>/report.json            written once, at the end

Arms are named by what produced them: "zero_shot", "step-00700", "final".
The dev-selection stage and the final stage share these files, so dev
examples scored while choosing an adapter are not scored again when that
adapter is evaluated on the full splits.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from computeruse.eval.vlm_grounder import ArmResult, result_from_json, result_to_json
from computeruse.training.prepare_dataset import TrainingExample


class ResultStore:
    """Per-arm scored examples: in memory, appended to local JSONL, and
    pushed to the Hub at most every `flush_every_seconds` (and at the end
    of every arm). A failed push is logged, not raised: the local files
    still hold everything, the next push retries, and a notebook that
    finishes keeps /kaggle/working anyway -- an eval should not die over a
    network blip the way a half-uploaded result never could hurt it."""

    def __init__(
        self,
        sync: Any,
        prefix: str,
        local_dir: Path,
        flush_every_seconds: float = 300,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = print,
    ) -> None:
        self.sync = sync
        self.prefix = prefix.strip("/")
        self.local_dir = Path(local_dir)
        self.flush_every_seconds = flush_every_seconds
        self._clock = clock
        self._log = log
        self._results: dict[str, dict[str, ArmResult]] = {}
        self._dirty = False
        self._last_flush = clock()

    @property
    def results_dir(self) -> Path:
        return self.local_dir / self.prefix / "results"

    def pull(self) -> int:
        """Load whatever earlier sessions pushed. Returns how many results."""
        self.sync.fetch([f"{self.prefix}/results/*.jsonl"], self.local_dir)
        self._results = {}
        for path in sorted(self.results_dir.glob("*.jsonl")):
            rows = self._results.setdefault(path.stem, {})
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    result = result_from_json(json.loads(line))
                    rows[result.example_id] = result
        return self.total()

    def total(self) -> int:
        return sum(len(rows) for rows in self._results.values())

    def done(self, arm: str) -> dict[str, ArmResult]:
        return dict(self._results.get(arm, {}))

    def add(self, arm: str, result: ArmResult) -> None:
        self._results.setdefault(arm, {})[result.example_id] = result
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with (self.results_dir / f"{arm}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result_to_json(result)) + "\n")
        self._dirty = True
        if self._clock() - self._last_flush >= self.flush_every_seconds:
            self.flush()

    def flush(self) -> bool:
        """Push every arm's file in one commit. True if the Hub is current."""
        if not self._dirty:
            return True
        files = {
            f"{self.prefix}/results/{p.name}": str(p) for p in sorted(self.results_dir.glob("*.jsonl"))
        }
        try:
            self.sync.push_files(files, f"eval: {self.total()} results")
        except Exception as e:  # see the class docstring
            self._log(f"[eval] push to the Hub failed, results are safe locally, will retry: {e!r}")
            return False
        self._dirty = False
        self._last_flush = self._clock()
        return True

    def push_json(self, name: str, payload: Any) -> None:
        self.sync.push_files(
            {f"{self.prefix}/{name}": json.dumps(payload, indent=2).encode()}, f"eval: {name}"
        )


def score_arm(
    store: ResultStore,
    arm: str,
    examples: list[TrainingExample],
    score_fn: Callable[[TrainingExample], ArmResult],
    log_every: int = 25,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.time,
) -> list[ArmResult]:
    """Score `examples` under `arm`, skipping the ones already in `store`.
    Returns a result for every example, in the order given."""
    done = store.done(arm)
    todo = [e for e in examples if e.id not in done]
    log(f"[eval] {arm}: {len(examples) - len(todo)}/{len(examples)} already scored, {len(todo)} to go")
    start = clock()
    try:
        for i, example in enumerate(todo, 1):
            result = score_fn(example)
            if result.example_id != example.id:
                raise ValueError(f"scored {result.example_id!r} for example {example.id!r}")
            store.add(arm, result)
            if i % log_every == 0 or i == len(todo):
                per = (clock() - start) / i
                log(
                    f"[eval] {arm}: {i}/{len(todo)} at {per:.1f} s/example, "
                    f"~{(len(todo) - i) * per / 60:.0f} min left in this arm"
                )
    except Exception:
        store.flush()  # an error inside Python keeps what it scored; only a hard kill loses the tail
        raise
    store.flush()
    done = store.done(arm)
    return [done[e.id] for e in examples]
