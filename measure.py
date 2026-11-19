import json
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import tracking
from metrics import data, pairs, predictor, suite
from metrics.latents import Latents, extract
from tag import PixelTag, resolve_seed


def measure_pairs(cfg, model):
    """Pair cosines of the ``pairs.suite``: the tag's colours on dataset frames, or t_position."""
    if cfg.pairs.suite == "t_position":
        tag = PixelTag("episode", cfg.data.tag.size, cfg.data.tag.seed)
        return pairs.measure_t_position(model, cfg.pairs.n, tag, cfg.data.img_size)
    frames = pairs.load_frames(cfg.data.name, cfg.pairs.n, cfg.data.seed, cfg.paths.data)
    tag = PixelTag("episode", cfg.data.tag.size, cfg.data.tag.seed)
    return pairs.measure(model, frames, tag, cfg.data.img_size, cfg.data.seed)


def measure_colour(cfg, model):
    """Colour probe on dataset frames, each stamped with its own colour from the tag's hash."""
    frames = pairs.load_frames(cfg.data.name, cfg.colour.n, cfg.data.seed, cfg.paths.data)
    tag = PixelTag("episode", cfg.data.tag.size, cfg.data.tag.seed)
    return pairs.measure_colour(
        model, frames, tag, cfg.data.img_size, cfg.colour.kinds, cfg.colour.train_fraction, cfg.data.seed
    )


def get_latents(cfg, model, latents_path):
    if cfg.get("latents_path"):
        return Latents.load(cfg.latents_path)

    w = cfg.windows
    loader, target, episode, train = data.build(
        cfg.data, cfg.targets, w.frames, w.min_frames, cfg.paths.data
    )
    names, angle, group = data.target_meta(cfg.targets)
    out = extract(model, loader, w.precision, rollout=w.get("rollout", False))
    lat = Latents(
        **out,
        target=torch.from_numpy(target),
        episode=torch.from_numpy(episode),
        train=torch.from_numpy(train),
        target_names=names,
        target_angle=angle,
        target_group=group,
    )
    if cfg.cache_latents:
        lat.save(latents_path)
    return lat


def run_dirs(policy, runs_root):
    """Results and checkpoint directories of ``policy``, a weights file or a run directory under the checkpoints root."""
    run_path = Path(swm.data.utils.get_cache_dir(runs_root, sub_folder="checkpoints"), policy)
    results_path = Path(swm.data.utils.get_cache_dir(runs_root), policy)
    if run_path.is_dir():
        return results_path, run_path
    return results_path.parent, run_path.parent


def checkpoint_file(policy, runs_root):
    """The weights file ``policy`` loads, or None for a Hugging Face repo id."""
    path = Path(swm.data.utils.get_cache_dir(runs_root, sub_folder="checkpoints"), policy)
    files = [path] if path.suffix == ".pt" else sorted(path.glob("*.pt")) if path.is_dir() else []
    return files[0] if len(files) == 1 else None


def start_wandb(cfg, run_path, weights):
    """Open the W&B run of the checkpoint's training run: the name and id saved in its config.yaml, else the
    run directory's name. Probe and test seeds are the resolved ones."""
    saved = OmegaConf.load(run_path / "config.yaml") if (run_path / "config.yaml").exists() else {}
    name = saved.get("output_model_name") or run_path.name
    wb = tracking.start(
        cfg.wandb, name=name, run_id=saved.get("subdir") or name, job_type=None,
        run_dir=run_path, config={},
    )
    commit, dirty = tracking.git_state()
    provenance = {
        "git_commit": commit,
        "git_dirty": dirty,
        "policy": cfg.policy,
        "checkpoint_sha256": tracking.sha256(weights) if weights else None,
        "data_seed": cfg.data.seed,
        "tag_seed": cfg.data.tag.seed,
        "noise_seed": cfg.data.noise.get("seed"),
    }
    return wb, provenance


def resolve_generated(cfg, run_path):
    """Fill cfg.data's generator, run seed, noise settings and action normalizer from the checkpoint's config.

    Probe and test sequences are regenerated from the run seed's own streams, never the training stream, so
    nothing is stored and no sequence or noise pattern is shared with training.
    """
    saved = OmegaConf.load(run_path / "config.yaml")
    assert saved.history_size == 1, (
        "the dot rollout protocol requires a model trained with history_size=1"
    )
    assert saved.num_preds == 1, "the dot rollout protocol requires one-step prediction"
    with open_dict(cfg):
        cfg.data.generator = saved.data.generator
        cfg.data.action_normalizer = saved.data.action_normalizer
        cfg.data.seed = saved.seed
        if cfg.data.noise.mode is None:
            cfg.data.noise.mode = saved.noise.mode
        if cfg.data.noise.alpha is None:
            cfg.data.noise.alpha = saved.noise.alpha


@hydra.main(version_base=None, config_path="./config/measure", config_name="pusht")
def run(cfg: DictConfig):
    """Compute the metric suite for one checkpoint."""
    off = {"tag": {"mode": "off", "size": 5, "seed": None}, "noise": {"mode": "off", "alpha": 0.0, "seed": None}}
    cfg = OmegaConf.merge({"data": off}, cfg)
    if cfg.wandb.enabled:
        tracking.check(cfg.wandb)
    model = swm.wm.utils.load_pretrained(cfg.policy, cache_dir=cfg.paths.runs)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)

    results_path, run_path = run_dirs(cfg.policy, cfg.paths.runs)
    resolve_seed(cfg.data.tag, run_path)
    if "generator" in cfg.data:
        resolve_generated(cfg, run_path)
    else:
        resolve_seed(cfg.data.noise, run_path, key="noise")
    results_path.mkdir(parents=True, exist_ok=True)
    wb, provenance = (
        start_wandb(cfg, run_path, checkpoint_file(cfg.policy, cfg.paths.runs)) if cfg.wandb.enabled else (None, None)
    )
    stem = Path(cfg.output.filename).stem
    latents_path = results_path / f"{Path(cfg.policy).stem}_{stem}_latents.pt"

    with tracking.finishing(wb):
        start_time = time.time()
        metrics = {}
        if cfg.metrics or cfg.get("pred_copy"):
            lat = get_latents(cfg, model, latents_path)
        if cfg.metrics:
            metrics = suite.run(lat, OmegaConf.to_container(cfg.metrics, resolve=True))
        if cfg.get("pairs"):
            metrics["pairs"] = measure_pairs(cfg, model)
        if cfg.get("pred_copy"):
            metrics["pred_copy"] = predictor.measure(model, lat, **cfg.pred_copy)
        if cfg.get("colour"):
            metrics["colour_probe"] = measure_colour(cfg, model)
        end_time = time.time()

        print(json.dumps(metrics, indent=1))

        with (results_path / cfg.output.filename).open("a") as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {json.dumps(metrics)}\n")
            f.write(f"latents: {cfg.get('latents_path') or (latents_path if cfg.cache_latents else 'not saved')}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")

        record = {
            "policy": cfg.policy,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": metrics,
        }
        if wb:
            record["wandb"] = {"entity": wb.entity, "project": wb.project, "id": wb.id}
        with (results_path / f"{stem}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

        if wb:
            tracking.log_measure(wb, metrics, provenance, record["config"], cfg.wandb.label)


if __name__ == "__main__":
    run()
