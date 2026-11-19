"""Weights & Biases tracking for train.py and measure.py, kept beside the local CSV and JSON records.

One W&B run per training run: its id is the run's ``subdir``. ``train.py`` opens it, and ``measure.py``
resumes it by that id, so the training curve and the measured numbers share a run summary.

Keys, with ``<label>`` present only when ``wandb.label`` is set:

  fit/<term>, validate/<term>
                          loss terms of ``lewm_forward`` (loss, pred_loss, sigreg_loss, pixel_loss, MotionJEPA's
                          diff_* terms; copy_loss on validation), as ``_step`` and ``_epoch`` series
  fit/<term>_grad_share, fit/<term>_grad_share_min
                          gradient-norm share of each prediction term of the hybrid objectives
  gap/<term>_train, _val, _val_over_train
                          per-epoch means and validation-to-training ratios (``GapCallback``)
  pair/<rep>/<pairing>    pair cosines during training (``PairCallback``)
  trainer/global_step     the x axis of all of the above
  checkpoint/file, checkpoint/sha256
                          last saved weights file of the training run, in the summary
  checkpoint/artifact     with ``wandb.upload_weights``, the artifact holding that file
  measure[/<label>]/<metric>/...
                          every scalar of measure.py's metrics, in the summary; lists of numbers stay lists
  measure[/<label>]/provenance/*
                          what produced them: git commit and dirty flag, policy file, its sha256, seeds
  headline[/<label>]/*    r, rollout_rmse, erank, pred_copy_ratio and every probe ``eps``, in the summary

The run config holds the resolved train config, ``provenance`` (git commit, dirty flag, seeds) and, per
measurement, the resolved measure config under ``measure[_<label>]``.
"""

import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

REPO = Path(__file__).resolve().parent
ARTIFACT_TMP = "wandb_artifacts"
RESUME_WAIT = 120  # seconds

HEADLINE = {
    "rollout_probe/r": "r",
    "rollout_probe/mean_rmse": "rollout_rmse",
    "rank_spectrum/erank": "erank",
    "pred_copy/ratio": "pred_copy_ratio",
}


def git_state():
    """``git_commit`` of this checkout and ``git_dirty``, whether tracked files differ from it (None without git)."""
    try:
        git = ["git", "-C", str(REPO)]
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            git + ["status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(cfg):
    """Fail before any work if a W&B run could not be opened: missing entity or project, a mode that would
    log nothing, or a failed online login.

    Offline runs need ``WANDB_MODE=offline`` in the environment.
    """
    if not cfg.entity or not cfg.project:
        raise ValueError("wandb.enabled needs WANDB_ENTITY and WANDB_PROJECT (or wandb.entity and wandb.project)")
    mode = os.environ.get("WANDB_MODE", "online")
    if mode in ("disabled", "dryrun") or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        raise RuntimeError(f"wandb.enabled with WANDB_MODE={mode}: nothing would be logged")
    if mode == "online" and not wandb.login(verify=True):
        raise RuntimeError("wandb login failed")


def start(cfg, *, name, run_id, run_dir, job_type=None, config=None):
    """Open the W&B run ``run_id`` after ``check`` passed, or fail.

    A run silently offline through a saved W&B setting is an error. ``cfg.resume`` is ``allow`` to create
    the run if it is new and ``must`` to continue an existing one. ``job_type=None`` keeps a resumed run's.
    """
    mode = os.environ.get("WANDB_MODE", "online")
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.upload_weights:
        # artifact staging and cache live under the run directory, so removing them touches nothing else
        os.environ["WANDB_CACHE_DIR"] = str(run_dir / ARTIFACT_TMP / "cache")
        os.environ["WANDB_DATA_DIR"] = str(run_dir / ARTIFACT_TMP / "data")
    kwargs = {
        "entity": cfg.entity, "project": cfg.project, "name": name, "id": run_id, "group": cfg.group,
        "job_type": job_type, "tags": list(cfg.tags), "dir": str(run_dir), "resume": cfg.resume,
    }
    # A run finished seconds ago can still be unknown to the server, which "must" reports as missing;
    # measure queued right after its training hits this, so retry for up to RESUME_WAIT seconds.
    for waited in range(0, RESUME_WAIT + 1, 15):
        try:
            run = wandb.init(**kwargs)
            break
        except wandb.errors.UsageError:
            if cfg.resume != "must" or mode != "online" or waited >= RESUME_WAIT:
                raise
            time.sleep(15)
    if run.settings.mode != mode:
        run.finish(exit_code=1)
        raise RuntimeError(f"wandb is in mode {run.settings.mode}, not the requested {mode}")
    # In "must" mode a missing run already raised; the id check guards against a different run being served.
    assert run.id == run_id, (run.id, run_id)
    if config:
        run.config.update(config, allow_val_change=True)
    return run


class RunLogger(Logger):
    """Lightning logger over an open W&B run.

    stable-pretraining's Manager special-cases ``WandbLogger``: on an offline run without an earlier
    offline run it reads a config file that does not exist, and it can swap the run id from a sidecar.
    This logger takes the run that ``start`` opened, so neither applies. The run config is complete at
    ``start``, hence ``log_hyperparams`` has nothing to add.
    """

    def __init__(self, run):
        super().__init__()
        self.run = run
        run.define_metric("trainer/global_step")
        run.define_metric("*", step_metric="trainer/global_step", step_sync=True)

    @property
    def name(self):
        return self.run.project

    @property
    def version(self):
        return self.run.id

    @property
    def experiment(self):
        return self.run

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        metrics = dict(metrics)
        if step is not None:
            metrics["trainer/global_step"] = step
        self.run.log(metrics)


class CheckpointHashCallback(Callback):
    """Records the file name and sha256 of the last ``weights_epoch_N.pt`` in the run summary at the end of
    training, and with ``upload`` uploads that file as an artifact.

    Once an online upload is confirmed, the run's local artifact staging and cache are deleted. An offline
    run keeps them for ``wandb sync``.
    """

    def __init__(self, run, run_dir, upload=False):
        super().__init__()
        self.run = run
        self.run_dir = Path(run_dir)
        self.upload = upload

    def on_train_end(self, trainer, pl_module):
        files = list(self.run_dir.glob("weights_epoch_*.pt"))
        if not trainer.is_global_zero or not files:
            return
        last = max(files, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        digest = sha256(last)
        self.run.summary["checkpoint/file"] = last.name
        self.run.summary["checkpoint/sha256"] = digest
        if self.upload:
            self._upload(last, digest)

    def _upload(self, path, digest):
        artifact = wandb.Artifact(f"{self.run.id}-weights", type="model", metadata={"file": path.name, "sha256": digest})
        artifact.add_file(str(path), name=path.name)
        logged = self.run.log_artifact(artifact, aliases=["latest"])
        self.run.summary["checkpoint/artifact"] = artifact.name
        if self.run.settings.mode == "online":
            logged.wait()
            if logged.state != "COMMITTED":
                raise RuntimeError(f"weights artifact {artifact.name} is {logged.state}, not committed")
            shutil.rmtree(self.run_dir / ARTIFACT_TMP, ignore_errors=True)


def flatten(tree, prefix=""):
    """Numbers and lists of numbers of a nested dict, under ``/``-joined keys. Strings and None are dropped."""
    out = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[path] = value
        elif isinstance(value, (list, tuple)) and all(v is None or isinstance(v, (int, float)) for v in value):
            out[path] = list(value)
    return out


def headline(flat):
    """The summary numbers worth reading first, from ``flatten`` of measure.py's metrics."""
    out = {name: flat[key] for key, name in HEADLINE.items() if key in flat}
    out.update({f"eps/{key.removesuffix('/eps')}": v for key, v in flat.items() if key.endswith("/eps")})
    return out


def log_measure(run, metrics, provenance, config, label=None):
    """Write one measurement into the summary of ``run`` and its config under ``measure[_<label>]``."""
    ns = f"/{label}" if label else ""
    flat = flatten(metrics)
    summary = {f"measure{ns}/{key}": value for key, value in flat.items()}
    summary.update({f"measure{ns}/provenance/{key}": value for key, value in provenance.items()})
    summary.update({f"headline{ns}/{key}": value for key, value in headline(flat).items()})
    run.summary.update(summary)
    run.config.update({f"measure_{label}" if label else "measure": config}, allow_val_change=True)
