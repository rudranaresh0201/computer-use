"""
Tests for training/hub_checkpoint.py.

The Hub is replaced by FakeHub, a local directory with the same method
shapes as huggingface_hub.HfApi (whoami, create_repo, repo_info,
list_repo_files, create_commit) plus a snapshot_download stand-in. The
operations passed to it are the real CommitOperationAdd/Delete objects.

The last test is the one that matters: a tiny real model trained by the
real Trainer, once straight through and once stopped mid-run, pushed,
and resumed by a fresh Trainer from what was pulled back. Both must end
with identical weights -- "a session can die and we lose nothing",
checked rather than assumed.
"""

from __future__ import annotations

import fnmatch
import time
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from transformers import TrainerControl

from computeruse.training.hub_checkpoint import (
    ADAPTERS_DIR,
    LAST_CHECKPOINT,
    STATE_FILE,
    HubCheckpointCallback,
    HubSync,
    NanGuardCallback,
    WallClockCallback,
)


class FakeHub:
    def __init__(self, root: Path, private: bool = True, fail_commits: int = 0) -> None:
        self.root = root
        self.private = private
        self.fail_commits = fail_commits
        self.commits: list[str] = []

    def whoami(self, token=None, **_):
        return {"name": "tester"}

    def create_repo(self, repo_id, **_):
        (self.root / repo_id).mkdir(parents=True, exist_ok=True)

    def repo_info(self, repo_id, **_):
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, repo_id, **_):
        base = self.root / repo_id
        if not base.exists():
            return []
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())

    def create_commit(self, repo_id, operations, commit_message, **_):
        if self.fail_commits > 0:
            self.fail_commits -= 1
            raise ConnectionError("simulated network failure")
        base = self.root / repo_id
        for op in operations:
            target = base / op.path_in_repo
            if isinstance(op, CommitOperationDelete):
                target.unlink()
            elif isinstance(op, CommitOperationAdd):
                target.parent.mkdir(parents=True, exist_ok=True)
                content = op.path_or_fileobj
                if isinstance(content, (bytes, bytearray)):
                    target.write_bytes(content)
                else:
                    shutil.copyfile(content, target)
        self.commits.append(commit_message)

    def download(self, repo_id, allow_patterns, local_dir, **_):
        base = self.root / repo_id
        for rel in self.list_repo_files(repo_id):
            if any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                out = Path(local_dir) / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base / rel, out)
        return local_dir


def make_sync(hub: FakeHub, **kw) -> HubSync:
    return HubSync("tester/grounder", token="t", api=hub, downloader=hub.download, sleep=lambda s: None, log=lambda m: None, **kw)


def fake_checkpoint(dirpath: Path, step: int, extra: str | None = None) -> Path:
    dirpath.mkdir(parents=True)
    (dirpath / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (dirpath / "adapter_model.safetensors").write_bytes(f"weights@{step}".encode())
    (dirpath / "adapter_config.json").write_text("{}")
    (dirpath / "optimizer.pt").write_bytes(b"opt")
    if extra:
        (dirpath / extra).write_bytes(b"x")
    return dirpath


# -- HubSync ---------------------------------------------------------------


def test_preflight_writes_a_real_commit(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    assert make_sync(hub).preflight() == "tester"
    assert hub.commits == ["preflight: write check"]


def test_preflight_refuses_a_public_repo(tmp_path):
    with pytest.raises(RuntimeError, match="PUBLIC"):
        make_sync(FakeHub(tmp_path / "hub", private=False)).preflight()


def test_fresh_repo_has_no_checkpoint_to_resume(tmp_path):
    sync = make_sync(FakeHub(tmp_path / "hub"))
    assert sync.fetch_last_checkpoint(tmp_path / "dl") is None
    assert sync.read_state() is None


def test_push_then_fetch_round_trips_the_checkpoint(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    sync = make_sync(hub)
    sync.push_checkpoint(fake_checkpoint(tmp_path / "checkpoint-50", 50), 50, 1179, snapshot=False)

    got = sync.fetch_last_checkpoint(tmp_path / "dl")
    assert (got / "adapter_model.safetensors").read_bytes() == b"weights@50"
    assert json.loads((got / "trainer_state.json").read_text())["global_step"] == 50
    files = hub.list_repo_files("tester/grounder")
    assert not any(f.startswith(ADAPTERS_DIR) for f in files)  # snapshot=False


def test_a_newer_push_removes_files_the_older_checkpoint_had(tmp_path):
    # Stale files left in last-checkpoint/ would be resumed from as if
    # they belonged to the newer checkpoint.
    hub = FakeHub(tmp_path / "hub")
    sync = make_sync(hub)
    sync.push_checkpoint(fake_checkpoint(tmp_path / "c1", 50, extra="scaler.pt"), 50, 1179, snapshot=False)
    sync.push_checkpoint(fake_checkpoint(tmp_path / "c2", 100), 100, 1179, snapshot=True)

    files = hub.list_repo_files("tester/grounder")
    assert f"{LAST_CHECKPOINT}/scaler.pt" not in files
    assert f"{ADAPTERS_DIR}/step-00100/adapter_model.safetensors" in files
    assert make_sync(hub).read_state()["global_step"] == 100


def test_push_refuses_an_incomplete_checkpoint(tmp_path):
    (tmp_path / "half").mkdir()
    with pytest.raises(FileNotFoundError):
        make_sync(FakeHub(tmp_path / "hub")).push_checkpoint(tmp_path / "half", 5, 10, snapshot=False)


def test_a_transient_failure_is_retried(tmp_path):
    hub = FakeHub(tmp_path / "hub", fail_commits=2)
    make_sync(hub).push_checkpoint(fake_checkpoint(tmp_path / "c", 7), 7, 10, snapshot=False)
    assert len(hub.commits) == 1


def test_push_final_marks_the_run_finished(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    sync = make_sync(hub)
    sync.push_final(fake_checkpoint(tmp_path / "final", 1179), 1179, 1179)
    assert sync.read_state()["finished"] is True


# -- callbacks -----------------------------------------------------------


def state(step, max_steps=1179):
    return SimpleNamespace(global_step=step, max_steps=max_steps, is_world_process_zero=True)


def test_checkpoint_callback_snapshots_only_on_multiples(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    cb = HubCheckpointCallback(make_sync(hub), snapshot_every=100, log=lambda m: None)
    args = SimpleNamespace(output_dir=str(tmp_path))
    for step in (37, 100):
        fake_checkpoint(tmp_path / f"checkpoint-{step}", step)
        cb.on_save(args, state(step), TrainerControl())
    snaps = {f.split("/")[1] for f in hub.list_repo_files("tester/grounder") if f.startswith(ADAPTERS_DIR)}
    assert snaps == {"step-00100"}
    assert cb.last_pushed_step == 100


def test_checkpoint_callback_stops_training_after_repeated_failures(tmp_path):
    hub = FakeHub(tmp_path / "hub", fail_commits=10**6)
    cb = HubCheckpointCallback(make_sync(hub, retries=1), max_consecutive_failures=3, log=lambda m: None)
    args = SimpleNamespace(output_dir=str(tmp_path))
    control = TrainerControl()
    for step in (10, 20, 30):
        fake_checkpoint(tmp_path / f"checkpoint-{step}", step)
        control = cb.on_save(args, state(step), control)
        assert control.should_training_stop is (step == 30)


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_wall_clock_saves_on_interval_and_stops_at_budget():
    clock = Clock()
    cb = WallClockCallback(session_start=0, budget_seconds=3600, save_every_seconds=600, clock=clock, log=lambda m: None)
    cb.on_train_begin(None, state(0), TrainerControl())

    clock.t = 300
    c = cb.on_step_end(None, state(1), TrainerControl())
    assert not c.should_save and not c.should_training_stop

    clock.t = 650
    c = cb.on_step_end(None, state(2), TrainerControl())
    assert c.should_save and not c.should_training_stop
    cb.on_save(None, state(2), c)

    clock.t = 900  # only 250s since the save
    assert not cb.on_step_end(None, state(3), TrainerControl()).should_save

    clock.t = 3601
    c = cb.on_step_end(None, state(4), TrainerControl())
    assert c.should_save and c.should_training_stop
    assert "budget" in cb.stop_reason


def test_setup_time_counts_against_the_budget():
    # session_start is the notebook's first cell: 50 min of installs and
    # model download already happened before training began.
    clock = Clock(3000)
    cb = WallClockCallback(session_start=0, budget_seconds=3600, save_every_seconds=10**6, clock=clock, log=lambda m: None)
    cb.on_train_begin(None, state(0), TrainerControl())
    clock.t = 3700
    assert cb.on_step_end(None, state(1), TrainerControl()).should_training_stop


def test_nan_guard_stops_without_saving_after_patience():
    cb = NanGuardCallback(patience=3, log=lambda m: None)
    control = TrainerControl()
    for loss in (1.2, float("nan"), float("inf")):
        control = cb.on_log(None, state(1), control, logs={"loss": loss})
    assert not control.should_training_stop
    control.should_save = True
    control = cb.on_log(None, state(1), control, logs={"loss": float("nan")})
    assert control.should_training_stop and not control.should_save and cb.tripped


def test_nan_guard_resets_on_a_finite_loss():
    cb = NanGuardCallback(patience=2, log=lambda m: None)
    control = TrainerControl()
    for loss in (float("nan"), 0.9, float("nan")):
        control = cb.on_log(None, state(1), control, logs={"loss": loss})
    assert not control.should_training_stop


# -- the real thing: interrupted + resumed == uninterrupted ------------------


def _tiny_trainer(output_dir: Path, callbacks: list):
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    class Tokens(Dataset):
        def __init__(self):
            g = torch.Generator().manual_seed(123)
            self.x = torch.randint(0, 64, (32, 16), generator=g)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return {"input_ids": self.x[i], "labels": self.x[i]}

    torch.manual_seed(0)  # identical base weights and LoRA init every time
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    )
    model = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,  # 32 rows / 2 / 2 = 8 optimizer steps
        num_train_epochs=1,
        learning_rate=5e-2,
        warmup_ratio=0.25,
        save_strategy="steps",
        save_steps=1000,  # only the callbacks trigger saves
        logging_steps=1,
        seed=42,
        data_seed=42,
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
    )
    return Trainer(model=model, args=args, train_dataset=Tokens(), callbacks=callbacks)


def _lora_weights(trainer) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in trainer.model.named_parameters() if "lora_" in n}


def test_interrupted_and_resumed_run_matches_an_uninterrupted_one(tmp_path):
    quiet = dict(log=lambda m: None)

    straight = _tiny_trainer(tmp_path / "a", [])
    straight.train()
    expected = _lora_weights(straight)
    assert straight.state.global_step == 8

    hub = FakeHub(tmp_path / "hub")
    sync = make_sync(hub)

    # Session 1: stops itself at step 3 and pushes on the way out.
    clock_cb = WallClockCallback(session_start=time.time(), budget_seconds=10**9, save_every_seconds=10**9, stop_after_step=3, **quiet)
    first = _tiny_trainer(tmp_path / "b1", [clock_cb, HubCheckpointCallback(sync, snapshot_every=100, **quiet)])
    first.train()
    assert first.state.global_step == 3
    pushed = sync.read_state()
    assert (pushed["global_step"], pushed["total_steps"], pushed["finished"]) == (3, 8, False)

    # Session 2: a brand-new process would only have what the Hub has.
    resumed_from = sync.fetch_last_checkpoint(tmp_path / "pulled")
    second = _tiny_trainer(tmp_path / "b2", [HubCheckpointCallback(sync, **quiet)])
    second.train(resume_from_checkpoint=str(resumed_from))
    assert second.state.global_step == 8

    got = _lora_weights(second)
    assert got.keys() == expected.keys()
    for name in expected:
        assert not torch.equal(expected[name], torch.zeros_like(expected[name])) or "lora_B" not in name
        torch.testing.assert_close(got[name], expected[name], rtol=0, atol=1e-6, msg=name)

    # And the schedule was the full one, not a per-session one: the
    # learning rate logged at step 4 (first step after resume) matches.
    def lr_at(trainer, step):
        return next(h["learning_rate"] for h in trainer.state.log_history if h.get("step") == step and "learning_rate" in h)

    assert lr_at(second, 4) == pytest.approx(lr_at(straight, 4))


def test_wall_clock_refuses_a_nonsense_session_start():
    with pytest.raises(ValueError, match="session_start"):
        WallClockCallback(session_start=0, budget_seconds=3600, save_every_seconds=600)


def test_a_second_train_call_does_not_inherit_the_previous_stop():
    # 2026-10-11, live on Kaggle: cell 9 was run twice in one session. The
    # reused WallClockCallback still held "smoke stop at step 3" from the
    # first run and ended the second one right after step 1.
    clock = Clock(100)
    cb = WallClockCallback(session_start=0, budget_seconds=10**6, save_every_seconds=10**6,
                           stop_after_step=3, clock=clock, log=lambda m: None)
    cb.on_train_begin(None, state(0), TrainerControl())
    assert cb.on_step_end(None, state(3), TrainerControl()).should_training_stop

    cb.stop_after_step = None
    cb.on_train_begin(None, state(0), TrainerControl())
    assert not cb.on_step_end(None, state(1), TrainerControl()).should_training_stop


def test_nan_guard_and_hub_callback_start_each_run_clean(tmp_path):
    nan = NanGuardCallback(patience=1, log=lambda m: None)
    nan.on_log(None, state(1), TrainerControl(), logs={"loss": float("nan")})
    assert nan.tripped
    nan.on_train_begin(None, state(0), TrainerControl())
    assert not nan.tripped and nan.bad_logs == 0

    hub = HubCheckpointCallback(make_sync(FakeHub(tmp_path / "hub")), log=lambda m: None)
    hub.consecutive_failures = 2
    hub.on_train_begin(None, state(0), TrainerControl())
    assert hub.consecutive_failures == 0
