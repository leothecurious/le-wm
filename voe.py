"""Live-frame violation-of-expectation probe for TwoRoom."""

import json
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler

from eval import img_transform


def score_episode(
    model, initial_state, actions, normalized_actions, midpoint, perturbation, rng, cfg, device
):
    """Measure teacher-forced one-step prediction error on rendered frames."""
    env = gym.make(cfg.world.env_name, render_mode="rgb_array")
    try:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        raw = env.unwrapped
        raw._set_state(initial_state)
        frames = [raw.render()]
        for i, action_block in enumerate(actions):
            for action in action_block:
                raw.step(action)
            frame_index = i + 1
            if frame_index == midpoint and perturbation == "color":
                raw.variation_space["agent"]["color"].set_value(
                    np.array([0, 0, 255], dtype=np.uint8)
                )
            elif frame_index == midpoint and perturbation == "teleport":
                x = rng.uniform(20, 95) if rng.integers(2) == 0 else rng.uniform(129, 204)
                raw._set_state(np.array([x, rng.uniform(20, 204)], dtype=np.float32))
            frames.append(raw.render())

        transform = img_transform(cfg)
        pixels = torch.stack([transform(frame) for frame in frames]).to(device)
        with torch.inference_mode():
            z = model.encode({"pixels": pixels[None]})["emb"].float()
            action_t = torch.as_tensor(
                normalized_actions, dtype=torch.float32, device=device
            )[None]
            action_emb = model.action_encoder(action_t)
            errors = []
            last = min(midpoint + cfg.steps_after_midpoint - 1, len(frames) - 1)
            for j in range(midpoint, last + 1):
                if j == 0:
                    continue
                pred = model.predict(
                    z[:, j - cfg.history_size:j],
                    action_emb[:, j - cfg.history_size:j],
                )
                errors.append((pred[:, -1] - z[:, j]).pow(2).mean().item())
        return float(np.mean(errors))
    finally:
        env.close()


@hydra.main(version_base=None, config_path="./config/voe", config_name="tworoom")
def run(cfg: DictConfig):
    """Compare surprise after appearance and state perturbations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swm.wm.utils.load_pretrained(cfg.policy).to(device).eval()
    model.requires_grad_(False)

    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cfg.cache_dir,
    )
    raw_actions = dataset.get_col_data("action")
    scaler = StandardScaler().fit(raw_actions[~np.isnan(raw_actions).any(axis=1)])
    states = dataset.get_col_data("proprio")
    lengths = np.asarray(dataset.lengths)
    model_steps = lengths // cfg.frameskip
    eligible = np.flatnonzero(model_steps >= cfg.min_model_steps)
    rng = np.random.default_rng(cfg.seed)
    episodes = rng.choice(eligible, size=min(cfg.n, len(eligible)), replace=False)

    result = {"unperturbed": [], "color": [], "teleport": []}
    for ep in episodes:
        offset, length = int(dataset.offsets[ep]), int(dataset.lengths[ep])
        n_steps = length // cfg.frameskip
        raw_ep_actions = np.asarray(
            raw_actions[offset:offset + n_steps * cfg.frameskip], dtype=np.float32
        ).reshape(n_steps, cfg.frameskip, -1)
        normalized = scaler.transform(
            raw_ep_actions.reshape(-1, raw_ep_actions.shape[-1])
        ).reshape(n_steps, -1).astype(np.float32)
        initial_state = np.asarray(states[offset], dtype=np.float32)
        midpoint = n_steps // 2
        for condition in result:
            score = score_episode(
                model, initial_state, raw_ep_actions, normalized, midpoint,
                condition if condition != "unperturbed" else None, rng, cfg, device,
            )
            result[condition].append(score)
        if len(result["unperturbed"]) % 5 == 0:
            print(f"episodes {len(result['unperturbed'])}/{len(episodes)}", flush=True)

    baseline = np.asarray(result["unperturbed"])
    out = {
        "n": len(baseline),
        "steps_after_midpoint": cfg.steps_after_midpoint,
        "unperturbed_mean_surprise": float(baseline.mean()),
        "conditions": {},
    }
    for condition in ("color", "teleport"):
        perturbed = np.asarray(result[condition])
        stat = ttest_rel(perturbed, baseline)
        out["conditions"][condition] = {
            "mean_surprise": float(perturbed.mean()),
            "mean_ratio": float(perturbed.mean() / baseline.mean()),
            "t": float(stat.statistic),
            "p": float(stat.pvalue),
            "n": len(baseline),
        }
    print(json.dumps(out, indent=2))
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"config": OmegaConf.to_container(cfg, resolve=True), "results": out}
    output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    run()
