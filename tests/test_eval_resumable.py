"""
Tests for eval/resumable.py and eval/ablation.py.

The property that matters, checked rather than assumed: an eval session
that dies part-way loses nothing already pushed, and the next session
scores only what is left -- and ends with exactly the results an
uninterrupted session would have. Uses the same FakeHub as the training
checkpoint tests (a local directory with HfApi's method shapes).
"""

from __future__ import annotations

import pytest
from test_hub_checkpoint import FakeHub, make_sync

from computeruse.eval.ablation import (
    FINAL,
    ZERO_SHOT,
    UiaRow,
    adapter_arms,
    adapter_path,
    assemble_report,
    choose_adapter,
    dev_selection_subset,
    learning_curve,
)
from computeruse.eval.resumable import ResultStore, score_arm
from computeruse.eval.vlm_grounder import ArmResult
from computeruse.training.prepare_dataset import TrainingExample


def example(i: int, split: str = "dev", app: str = "notepad") -> TrainingExample:
    return TrainingExample(f"{app}_{i:04d}", f"images/{i}.png", "click", "(1,1)", app, split)


def fake_score(example: TrainingExample, hit_every: int = 2) -> ArmResult:
    n = int(example.id.rsplit("_", 1)[1])
    hit = n % hit_every == 0
    return ArmResult(
        example.id, example.app, example.split, "(5,5)", (5, 5), (0, 0, 10, 10), hit, 0.0 if hit else 50.0
    )


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def store_for(hub: FakeHub, tmp_path, name: str, clock=None) -> ResultStore:
    return ResultStore(
        make_sync(hub), "eval", tmp_path / name, flush_every_seconds=60, clock=clock or Clock(), log=lambda m: None
    )


# -- ResultStore / score_arm ----------------------------------------------


def test_a_killed_session_loses_only_its_unpushed_tail_and_the_next_one_scores_the_rest(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    examples = [example(i) for i in range(10)]

    # Session 1 scores 5 at 20 s each; the 60 s flush fires after the 3rd. Then
    # Kaggle kills the container: no Python cleanup runs, the store is just gone.
    clock = Clock()
    first = store_for(hub, tmp_path, "s1", clock)
    first.pull()
    for e in examples[:5]:
        clock.t += 20
        first.add("final", fake_score(e))

    # Session 2 starts empty, pulls, and scores only what the Hub does not have.
    second = store_for(hub, tmp_path, "s2")
    pulled = second.pull()
    assert pulled == 3  # the pushed 3 survived; the 2 scored after the last push did not
    scored = []
    results = score_arm(second, "final", examples, lambda e: scored.append(e.id) or fake_score(e), log=lambda m: None)

    assert len(scored) == 10 - pulled
    assert results == [fake_score(e) for e in examples]  # same as one uninterrupted session
    third = store_for(hub, tmp_path, "s3")
    assert third.pull() == 10  # and all of it is on the Hub now


def test_an_error_inside_python_still_pushes_everything_scored_before_it(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    store = store_for(hub, tmp_path, "s")
    store.pull()

    def fails_on_the_sixth(e):
        if e.id.endswith("_0005"):
            raise RuntimeError("CUDA error")
        return fake_score(e)

    with pytest.raises(RuntimeError):
        score_arm(store, "final", [example(i) for i in range(10)], fails_on_the_sixth, log=lambda m: None)
    assert store_for(hub, tmp_path, "s2").pull() == 5


def test_arms_are_kept_apart_and_a_dev_subset_is_reused_by_the_full_split(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    store = store_for(hub, tmp_path, "s")
    store.pull()
    dev = [example(i) for i in range(8)]
    score_arm(store, ZERO_SHOT, dev[:3], fake_score, log=lambda m: None)

    scored = []
    score_arm(store, ZERO_SHOT, dev, lambda e: scored.append(e.id) or fake_score(e), log=lambda m: None)
    assert scored == [e.id for e in dev[3:]]
    assert store.done("final") == {}


def test_a_failed_push_keeps_the_results_and_retries_later(tmp_path):
    hub = FakeHub(tmp_path / "hub", fail_commits=10)
    store = store_for(hub, tmp_path, "s")
    store.pull()
    store.add("final", fake_score(example(0)))
    assert store.flush() is False  # logged, not raised
    hub.fail_commits = 0
    assert store.flush() is True
    assert store_for(hub, tmp_path, "s2").pull() == 1


def test_score_fn_answering_for_the_wrong_example_is_caught(tmp_path):
    store = store_for(FakeHub(tmp_path / "hub"), tmp_path, "s")
    store.pull()
    with pytest.raises(ValueError, match="scored"):
        score_arm(store, "final", [example(1)], lambda e: fake_score(example(2)), log=lambda m: None)


# -- ablation decisions -----------------------------------------------------


def test_adapter_arms_lists_complete_snapshots_oldest_first_then_final():
    files = [
        "adapters/step-00200/adapter_model.safetensors",
        "adapters/step-00200/adapter_config.json",
        "adapters/step-00100/adapter_model.safetensors",
        "adapters/step-00100/adapter_config.json",
        "adapters/step-00300/adapter_model.safetensors",  # config missing: incomplete, skipped
        "final/adapter_model.safetensors",
        "final/adapter_config.json",
        "last-checkpoint/adapter_model.safetensors",
    ]
    assert adapter_arms(files) == ["step-00100", "step-00200", FINAL]
    assert adapter_path("step-00100", "/d").as_posix() == "/d/adapters/step-00100"
    assert adapter_path(FINAL, "/d").as_posix() == "/d/final"


def test_dev_subset_is_fixed_by_seed_and_independent_of_file_order():
    dev = [example(i) for i in range(50)]
    a = dev_selection_subset(dev, 10, seed=0)
    assert a == dev_selection_subset(list(reversed(dev)), 10, seed=0)
    assert a != dev_selection_subset(dev, 10, seed=1)
    assert len(a) == 10 and dev_selection_subset(dev, 999) == dev


def _results(arm_hits: list[bool], distance: float = 10.0) -> list[ArmResult]:
    return [
        ArmResult(f"e{i}", "notepad", "dev", "(1,1)", (1, 1), (0, 0, 2, 2), hit, distance)
        for i, hit in enumerate(arm_hits)
    ]


def test_choose_adapter_takes_best_dev_accuracy_and_never_zero_shot():
    curve = learning_curve(
        {
            ZERO_SHOT: _results([True] * 10),  # even if the baseline scored best
            "step-00100": _results([True] * 3 + [False] * 7),
            "step-00200": _results([True] * 6 + [False] * 4),
            FINAL: _results([True] * 5 + [False] * 5),
        },
        final_step=300,
    )
    assert [row["step"] for row in curve] == [0, 100, 200, 300]
    assert choose_adapter(curve) == "step-00200"


def test_choose_adapter_breaks_ties_on_distance_then_later_step():
    tie = {"step-00100": _results([True, False], 30.0), "step-00200": _results([True, False], 20.0)}
    assert choose_adapter(learning_curve(tie, 300)) == "step-00200"
    tie = {"step-00100": _results([True, False], 20.0), "step-00200": _results([True, False], 30.0)}
    assert choose_adapter(learning_curve(tie, 300)) == "step-00100"
    same = {"step-00100": _results([True, False]), "step-00200": _results([True, False])}
    assert choose_adapter(learning_curve(same, 300)) == "step-00200"


def _split_results(split: str, hits: list[bool], app: str) -> list[ArmResult]:
    return [
        ArmResult(f"{split}_{i}", app, split, "(1,1)", (1, 1), (0, 0, 2, 2), hit, 1.0)
        for i, hit in enumerate(hits)
    ]


def test_report_has_every_arm_h1_h3_and_intervals():
    zero = _split_results("test_same_app", [False] * 40, "notepad") + _split_results(
        "test_held_out_app", [False] * 40, "audacity"
    )
    tuned = _split_results("test_same_app", [True] * 30 + [False] * 10, "notepad") + _split_results(
        "test_held_out_app", [True] * 10 + [False] * 30, "audacity"
    )
    uia = [UiaRow(r.example_id, r.app, "rich", r.split, available=i % 2 == 0, hit=True) for i, r in enumerate(zero)]

    report = assemble_report(
        zero_shot=zero,
        fine_tuned=tuned,
        chosen_arm="step-00200",
        richness_by_app={"notepad": "rich", "audacity": "weak"},
        uia_rows=uia + [UiaRow("unscored", "notepad", "rich", "dev", True, True)],
        n_boot=200,
    )
    assert set(report["by_arm"]) == {"zero_shot", "fine_tuned", "uia_only", "hybrid"}
    assert report["h1_test_same_app"]["diff"] == 0.75 and report["h1_test_same_app"]["ci95"][0] > 0
    assert report["h3_test_held_out_app"]["diff"] == 0.25
    assert report["h3_fine_tuned_same_vs_held_out"] == {"test_same_app": 0.75, "test_held_out_app": 0.25}
    assert report["uia_resolved"] == {"resolved": 40, "of": 80}
    lo, hi = report["ci95_by_arm_and_split"]["fine_tuned"]["test_same_app"]
    assert lo <= 0.75 <= hi
    assert report["diagnostics"]["zero_shot"]["test_same_app"]["formats"] == {"point": 40, "box": 0, "none": 0}


def test_report_refuses_arms_scored_on_different_examples():
    with pytest.raises(ValueError, match="different examples"):
        assemble_report(
            zero_shot=_split_results("dev", [True], "notepad"),
            fine_tuned=_split_results("dev", [True, True], "notepad"),
            chosen_arm=FINAL,
            richness_by_app={"notepad": "rich"},
        )


def test_report_refuses_a_scored_example_with_no_uia_result():
    results = _split_results("dev", [True, False], "notepad")
    with pytest.raises(ValueError, match="no UIA result"):
        assemble_report(
            zero_shot=results,
            fine_tuned=results,
            chosen_arm=FINAL,
            richness_by_app={"notepad": "rich"},
            uia_rows=[UiaRow("dev_0", "notepad", "rich", "dev", True, True)],
        )
