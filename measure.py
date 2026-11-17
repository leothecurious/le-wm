import json
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

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
    lat.save(latents_path)
    return lat


def run_dirs(policy, runs_root):
    """Results and checkpoint directories of ``policy``, a weights file or a run directory under the checkpoints root."""
    run_path = Path(swm.data.utils.get_cache_dir(runs_root, sub_folder="checkpoints"), policy)
    results_path = Path(swm.data.utils.get_cache_dir(runs_root), policy)
    if run_path.is_dir():
        return results_path, run_path
    return results_path.parent, run_path.parent


@hydra.main(version_base=None, config_path="./config/measure", config_name="pusht")
def run(cfg: DictConfig):
    """Compute the metric suite for one checkpoint."""
    off = {"tag": {"mode": "off", "size": 5, "seed": None}, "noise": {"mode": "off", "alpha": 0.0, "seed": None}}
    cfg = OmegaConf.merge({"data": off}, cfg)
    model = swm.wm.utils.load_pretrained(cfg.policy, cache_dir=cfg.paths.runs)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)

    results_path, run_path = run_dirs(cfg.policy, cfg.paths.runs)
    resolve_seed(cfg.data.tag, run_path)
    if cfg.data.get("probe") is not None:
        saved = OmegaConf.load(run_path / "config.yaml")
        assert saved.history_size == 1, (
            "the dot rollout protocol requires a model trained with history_size=1"
        )
        assert saved.num_preds == 1, "the dot rollout protocol requires one-step prediction"
        training_noise_seed = int(saved.noise.seed)
        if cfg.data.noise.mode is None:
            cfg.data.noise.mode = saved.noise.mode
        if cfg.data.noise.alpha is None:
            cfg.data.noise.alpha = saved.noise.alpha
        if cfg.data.noise.seed is None:
            cfg.data.noise.seed = training_noise_seed + 2
        if cfg.data.probe.noise_seed is None:
            cfg.data.probe.noise_seed = training_noise_seed + 1
        if cfg.data.noise.mode != "off":
            assert training_noise_seed not in (cfg.data.noise.seed, cfg.data.probe.noise_seed), (
                "dot probe fitting and test must use noise seeds distinct from encoder training"
            )
        datasets_dir = swm.data.utils.get_cache_dir(cfg.paths.data, sub_folder="datasets")
        training_path = Path(datasets_dir, saved.data.dataset.name)
        for name in (cfg.data.name, cfg.data.probe.name):
            assert Path(datasets_dir, name).resolve() != Path(training_path).resolve(), (
                "dot probe fitting and test require datasets separate from encoder training"
            )
        with open_dict(cfg):
            cfg.data.action_normalizer = saved.data.action_normalizer
    else:
        resolve_seed(cfg.data.noise, run_path, key="noise")
    results_path.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.output.filename).stem
    latents_path = results_path / f"{Path(cfg.policy).stem}_{stem}_latents.pt"

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
        f.write(f"latents: {cfg.get('latents_path') or latents_path}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    record = {
        "policy": cfg.policy,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
    }
    with (results_path / f"{stem}.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    run()
