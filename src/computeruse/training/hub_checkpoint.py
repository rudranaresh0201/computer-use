"""
Checkpoints that survive the session they were made in.

Five training runs were lost with healthy loss curves (docs/journal.md,
2026-07-23 to 2026-08-30). The common cause: every checkpoint lived on the
session's own disk, and a Kaggle commit only keeps /kaggle/working if the
whole notebook finishes. A session that dies -- or a later cell that
raises -- takes every checkpoint with it.

The fix is to move each checkpoint off the machine the moment it is
written. `HubCheckpointCallback` pushes every Trainer save to a private
Hugging Face repo as one commit, and `HubSync.fetch_last_checkpoint`
pulls the newest one back at the start of the next session. A session
dying now costs at most the time since the last save, never the run.

Repo layout:

    last-checkpoint/        full resume state (adapter, optimizer, scheduler,
                            rng, trainer_state) -- overwritten on every save
    adapters/step-00100/    adapter-only snapshot every `snapshot_every`
                            steps, kept so the best step can be chosen later
                            by dev click accuracy rather than dev loss
    final/                  the adapter at the end of the full schedule
    run_state.json          {global_step, total_steps, finished}

Each push is a single `create_commit`: the Hub applies a commit as a whole
or not at all, so a session killed mid-upload leaves the previous
checkpoint intact instead of a half-written one.

`WallClockCallback` is the other half. The learning-rate schedule is built
from the run's total step count, so each session must keep the full
3-epoch schedule and stop on *time*, not on a per-session max_steps -- a
per-session max_steps decays the learning rate to zero at the end of every
session (the chunked notebook before this one did exactly that).
"""

from __future__ import annotations

import fnmatch
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from transformers import TrainerCallback

LAST_CHECKPOINT = "last-checkpoint"
ADAPTERS_DIR = "adapters"
FINAL_DIR = "final"
STATE_FILE = "run_state.json"
HEARTBEAT_FILE = "heartbeat.txt"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


class HubSync:
    """Everything that talks to the Hub. `api` and `downloader` default to
    the real huggingface_hub objects; tests pass a local-directory fake
    with the same method shapes."""

    def __init__(
        self,
        repo_id: str,
        token: Optional[str],
        api: Any = None,
        downloader: Optional[Callable[..., str]] = None,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        if api is None or downloader is None:
            from huggingface_hub import HfApi, snapshot_download

            api = api or HfApi(token=token)
            downloader = downloader or snapshot_download
        self.repo_id = repo_id
        self.token = token
        self.api = api
        self._download = downloader
        self._retries = retries
        self._sleep = sleep
        self._log = log

    # -- startup -----------------------------------------------------------

    def preflight(self) -> str:
        """Prove, before any GPU time is spent, that this session can write
        to the repo: the token resolves to a user, the repo exists and is
        private, and a real commit goes through. Raises on any failure."""
        user = self.api.whoami(token=self.token)["name"]
        self.api.create_repo(self.repo_id, private=True, exist_ok=True, token=self.token)
        if not self.api.repo_info(self.repo_id, token=self.token).private:
            raise RuntimeError(
                f"{self.repo_id} exists and is PUBLIC. Make it private on huggingface.co "
                "(Settings -> Change visibility) or pick another repo name."
            )
        stamp = datetime.now(timezone.utc).isoformat()
        self._commit([self._add(HEARTBEAT_FILE, f"preflight write check {stamp}\n".encode())], "preflight: write check")
        return user

    # -- push --------------------------------------------------------------

    def push_checkpoint(self, checkpoint_dir: Path, global_step: int, total_steps: int, snapshot: bool) -> None:
        """Replace last-checkpoint/ with `checkpoint_dir` (and optionally keep
        an adapter-only snapshot of this step) in one commit."""
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "trainer_state.json").exists():
            raise FileNotFoundError(f"{checkpoint_dir} has no trainer_state.json -- not a complete checkpoint")

        new_paths: set[str] = set()
        ops = []
        for f in sorted(p for p in checkpoint_dir.rglob("*") if p.is_file()):
            path_in_repo = f"{LAST_CHECKPOINT}/{f.relative_to(checkpoint_dir).as_posix()}"
            new_paths.add(path_in_repo)
            ops.append(self._add(path_in_repo, str(f)))
        # Files from an older checkpoint that this one doesn't have would
        # otherwise linger and be resumed from as if they belonged together.
        for existing in self._list_files():
            if existing.startswith(f"{LAST_CHECKPOINT}/") and existing not in new_paths:
                ops.append(self._delete(existing))
        if snapshot:
            for name in ADAPTER_FILES:
                ops.append(self._add(f"{ADAPTERS_DIR}/step-{global_step:05d}/{name}", str(checkpoint_dir / name)))
        ops.append(self._add(STATE_FILE, self._state_bytes(global_step, total_steps, finished=False)))
        self._commit(ops, f"checkpoint step {global_step}/{total_steps}")

    def push_final(self, adapter_dir: Path, global_step: int, total_steps: int) -> None:
        adapter_dir = Path(adapter_dir)
        ops = [self._add(f"{FINAL_DIR}/{name}", str(adapter_dir / name)) for name in ADAPTER_FILES]
        ops.append(self._add(STATE_FILE, self._state_bytes(global_step, total_steps, finished=True)))
        self._commit(ops, f"final adapter at step {global_step}")

    def push_files(self, files: dict[str, Any], message: str) -> None:
        """Any set of files in one commit: {path_in_repo: local path or bytes}."""
        self._commit([self._add(path, content) for path, content in files.items()], message)

    # -- pull --------------------------------------------------------------

    def read_state(self) -> Optional[dict]:
        if STATE_FILE not in self._list_files():
            return None
        import tempfile

        local = self._download(
            repo_id=self.repo_id, allow_patterns=[STATE_FILE], local_dir=tempfile.mkdtemp(prefix="hub_state_"), token=self.token
        )
        return json.loads((Path(local) / STATE_FILE).read_text())

    def fetch_last_checkpoint(self, dest: Path) -> Optional[Path]:
        """Download last-checkpoint/ into dest and return its path, or None
        on a genuinely fresh run (nothing pushed yet)."""
        if f"{LAST_CHECKPOINT}/trainer_state.json" not in self._list_files():
            return None
        local = self._download(
            repo_id=self.repo_id, allow_patterns=[f"{LAST_CHECKPOINT}/*"], local_dir=str(dest), token=self.token
        )
        path = Path(local) / LAST_CHECKPOINT
        if not (path / "trainer_state.json").exists():
            raise RuntimeError(f"downloaded {LAST_CHECKPOINT} but {path}/trainer_state.json is missing")
        return path

    def fetch(self, patterns: list[str], dest: Path) -> list[str]:
        """Download every repo file matching any of `patterns` into dest,
        keeping repo paths. Returns the repo paths fetched (none is fine)."""
        wanted = [f for f in self._list_files() if any(fnmatch.fnmatch(f, p) for p in patterns)]
        if wanted:
            self._download(repo_id=self.repo_id, allow_patterns=wanted, local_dir=str(dest), token=self.token)
        return wanted

    # -- internals ---------------------------------------------------------

    def _list_files(self) -> list[str]:
        return self.api.list_repo_files(self.repo_id, token=self.token)

    def _commit(self, ops: list, message: str) -> None:
        delay = 10.0
        for attempt in range(1, self._retries + 1):
            try:
                self.api.create_commit(self.repo_id, operations=ops, commit_message=message, token=self.token)
                return
            except Exception as e:  # network blips are the expected failure here
                if attempt == self._retries:
                    raise
                self._log(f"[hub] commit failed (attempt {attempt}/{self._retries}): {e!r} -- retrying in {delay:.0f}s")
                self._sleep(delay)
                delay *= 3

    @staticmethod
    def _add(path_in_repo: str, content):
        from huggingface_hub import CommitOperationAdd

        return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=content)

    @staticmethod
    def _delete(path_in_repo: str):
        from huggingface_hub import CommitOperationDelete

        return CommitOperationDelete(path_in_repo=path_in_repo)

    @staticmethod
    def _state_bytes(global_step: int, total_steps: int, finished: bool) -> bytes:
        return json.dumps(
            {
                "global_step": global_step,
                "total_steps": total_steps,
                "finished": finished,
                "pushed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ).encode()


class HubCheckpointCallback(TrainerCallback):
    """Push every checkpoint the Trainer writes. A failed push is retried
    inside HubSync; if pushes keep failing, training stops rather than
    burning hours of GPU quota with nothing safe off the machine (the local
    checkpoints still reach the Kaggle output when the notebook completes)."""

    def __init__(
        self,
        sync: HubSync,
        snapshot_every: int = 100,
        max_consecutive_failures: int = 3,
        log: Callable[[str], None] = print,
    ) -> None:
        self.sync = sync
        self.snapshot_every = snapshot_every
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.last_pushed_step: Optional[int] = None
        self._log = log

    def on_train_begin(self, args, state, control, **kwargs):
        self.consecutive_failures = 0
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        step = state.global_step
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        snapshot = step % self.snapshot_every == 0 or step >= state.max_steps
        t0 = time.time()
        try:
            self.sync.push_checkpoint(checkpoint_dir, step, state.max_steps, snapshot=snapshot)
        except Exception as e:
            self.consecutive_failures += 1
            self._log(
                f"[hub] !! push of step {step} FAILED ({self.consecutive_failures}/"
                f"{self.max_consecutive_failures} in a row): {e!r}"
            )
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._log("[hub] !! stopping training: nothing new is reaching the Hub")
                control.should_training_stop = True
            return control
        self.consecutive_failures = 0
        self.last_pushed_step = step
        self._log(
            f"[hub] pushed step {step}/{state.max_steps} in {time.time() - t0:.0f}s"
            + (" (+ adapter snapshot)" if snapshot else "")
        )
        return control


class WallClockCallback(TrainerCallback):
    """Save on a time interval (step speed is unknown until the run starts)
    and end the session cleanly before Kaggle's session limit, saving on
    the way out. `session_start` is when the notebook started, so setup
    time counts against the budget too.

    `stop_after_step` exists only for the smoke test: it ends the session
    at a fixed step so the next session's resume can be checked."""

    def __init__(
        self,
        session_start: float,
        budget_seconds: float,
        save_every_seconds: float,
        stop_after_step: Optional[int] = None,
        progress_every_steps: int = 10,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = print,
        gpu_peak_gb: Optional[Callable[[], float]] = None,
    ) -> None:
        elapsed = clock() - session_start
        if not 0 <= elapsed < 24 * 3600:
            # e.g. session_start=0 (1970) would read as "budget long gone"
            # and end a real run after its first step.
            raise ValueError(
                f"session_start is {elapsed / 3600:.1f} h from now -- pass time.time() "
                "recorded in the notebook's first cell"
            )
        self.session_start = session_start
        self.budget_seconds = budget_seconds
        self.save_every_seconds = save_every_seconds
        self.stop_after_step = stop_after_step
        self.progress_every_steps = progress_every_steps
        self._clock = clock
        self._log = log
        self._gpu_peak_gb = gpu_peak_gb
        self.stop_reason: Optional[str] = None
        self._last_save = clock()
        self._train_start = clock()
        self._first_step: Optional[int] = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Per-run state starts clean: a callback object can outlive one
        # train() call (a notebook cell run twice), and a stop reason left
        # over from the last run would end this one after its first step.
        self.stop_reason = None
        now = self._clock()
        self._last_save = now
        self._train_start = now
        self._first_step = state.global_step
        self._log(
            f"[clock] training starts at step {state.global_step}/{state.max_steps}, "
            f"{(now - self.session_start) / 60:.0f} min into the session, "
            f"budget {self.budget_seconds / 3600:.1f} h"
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        now = self._clock()
        step = state.global_step
        if self.stop_after_step is not None and step >= self.stop_after_step:
            self.stop_reason = f"smoke stop at step {step}"
        elif now - self.session_start >= self.budget_seconds:
            self.stop_reason = f"session budget reached at step {step}"
        if self.stop_reason:
            self._log(f"[clock] {self.stop_reason} -- saving and ending this session")
            control.should_save = True
            control.should_training_stop = True
        elif now - self._last_save >= self.save_every_seconds:
            control.should_save = True

        done_here = step - (self._first_step or 0)
        if done_here in (1, 5) or (done_here > 0 and step % self.progress_every_steps == 0):
            per_step = (now - self._train_start) / done_here
            left = state.max_steps - step
            mem = f", GPU peak {self._gpu_peak_gb():.1f} GB" if self._gpu_peak_gb else ""
            self._log(
                f"[clock] step {step}/{state.max_steps}: {per_step:.1f} s/step, "
                f"{left * per_step / 3600:.1f} h of training left in total, "
                f"session at {(now - self.session_start) / 3600:.2f} h{mem}"
            )
        return control

    def on_save(self, args, state, control, **kwargs):
        self._last_save = self._clock()
        return control


class NanGuardCallback(TrainerCallback):
    """fp16 on a T4 can overflow to inf/NaN. The GradScaler skips the odd
    bad step on its own; a loss that stays non-finite means the run is
    broken. Stop without saving so the last good checkpoint on the Hub is
    not replaced by a broken one."""

    def __init__(self, patience: int = 3, log: Callable[[str], None] = print) -> None:
        self.patience = patience
        self.bad_logs = 0
        self.tripped = False
        self._log = log

    def on_train_begin(self, args, state, control, **kwargs):
        self.bad_logs = 0
        self.tripped = False
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is None:
            return control
        if math.isfinite(loss):
            self.bad_logs = 0
            return control
        self.bad_logs += 1
        self._log(f"[nan-guard] non-finite loss {loss} at step {state.global_step} ({self.bad_logs}/{self.patience})")
        if self.bad_logs >= self.patience:
            self.tripped = True
            control.should_save = False
            control.should_training_stop = True
            self._log("[nan-guard] !! stopping WITHOUT saving -- the Hub keeps the last good checkpoint")
        return control
