"""Hierarchical planner for ``hjepa.HJEPA`` on Push-T, planned top-down.

Level 2 plans latent actions toward the goal's level-2 latent. Its first predicted
latent and latent action then become the target of level 1, which plans the real
actions and is scored on its predictions encoded up to level 2. Once level 2's horizon
has decayed to 1 it is skipped and level 1 plans toward the goal's level-1 latent.

Ported from kevinghst/H-JEPA@97054ec (MIT), ``h_jepa/hierarchical_solver.py`` and
``stable_worldmodel/solver/gd.py``, config ``pusht_l2``, for stable-worldmodel's
``WorldModelPolicy``. Only the two-level Push-T path is kept; the vendored policy
tracks the step count itself, here ``HierarchicalSolver`` does.
"""

import math
from dataclasses import dataclass

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PlanConfig:
    """``swm.PlanConfig`` that is not frozen: ``HierarchicalSolver`` sets ``receding_horizon`` for the plan it returns."""

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = False

    @property
    def plan_len(self):
        return self.horizon * self.action_block


def decreasing_horizon_schedule(num_calls, configured_horizon):
    """Horizon of each planning call: ``configured_horizon`` down to 1, every value for as many calls."""
    if configured_horizon == 1:
        return [1] * num_calls
    if num_calls == 1:
        return [configured_horizon]
    if num_calls < configured_horizon:
        span = configured_horizon - 1
        return [max(1, math.ceil(1 + span * (num_calls - 1 - i) / (num_calls - 1))) for i in range(num_calls)]
    values = list(range(configured_horizon, 0, -1))
    counts = [num_calls // len(values)] * len(values)
    for i in range(len(values) - num_calls % len(values), len(values)):
        counts[i] += 1
    return [h for h, count in zip(values, counts) for _ in range(count)]


class LevelCostModel(nn.Module):
    """Rolls one level out in its own latent space and scores it against a goal.

    With ``info["project"]`` the goal is a level-2 latent: the level-1 rollout is encoded
    up with ``upper`` before the comparison, and its actions are pooled into latent actions
    that ``action_cost_weight`` ties to the level-2 plan.
    """

    def __init__(self, level, upper=None):
        super().__init__()
        self.level = level
        self.upper = upper

    def latent_action_queue(self):
        queue = getattr(self.level, "latent_action_queue", None)
        return queue if queue is not None and queue.numel() else None

    def rollout(self, emb, actions):
        """emb (B, S, 1, D), actions (B, S, T, A): latents (B, S, T + 1, D), one step of context."""
        B, S, T = actions.shape[:3]
        emb = emb.flatten(0, 1)
        act_emb = self.level.action_encoder(actions.flatten(0, 1))
        for t in range(T):
            pred = self.level.predict(emb[:, -1:], act_emb[:, t : t + 1])
            emb = torch.cat([emb, pred], dim=1)
        return emb.unflatten(0, (B, S))

    def get_cost(self, info, actions):
        pred = self.rollout(info["embed_0"], actions)
        action_cost = 0.0
        weight = float(info["action_cost_weight"])
        if info["project"]:
            B, S = pred.shape[:2]
            T = pred.shape[2]
            if weight:
                act_emb = self.level.action_encoder(actions.flatten(0, 1))
                act_emb = F.pad(act_emb, (0, 0, 0, T - act_emb.shape[1]))
                # the last pooled action points past the last encoded state
                pooled = self.upper.pool(act_emb).unflatten(0, (B, S))[..., :-1, :]
                action_cost = weight * self._cost(pooled, info["goal_action"], info["cost_last_n"], first=0)
            pred = self.upper.embed(pred.flatten(0, 1)).unflatten(0, (B, S))
        return self._cost(pred, info["goal_embed_0"], info["cost_last_n"], first=1) + action_cost

    @staticmethod
    def _cost(pred, goal, last_n, first):
        """Squared error of the last ``last_n`` predictions after the first ``first``."""
        pred = pred[:, :, first:]
        pred = pred[:, :, -min(last_n, pred.shape[2]) :]
        return F.mse_loss(pred, goal.expand_as(pred).detach(), reduction="none").sum(dim=(2, 3))


class GradientSolver:
    """Adam on the action sequences of ``num_samples`` candidates per environment, backpropagating through the model.

    Candidate 0 starts at the action prior's mean, the others add noise scaled by its std. The
    prior is the queue of latent actions seen in training, if the level has one, else zero mean
    and unit std. With a queue the actions are clipped to its 2% to 98% range after each step.
    """

    def __init__(
        self, model, n_steps, action_dim, min_its=30, batch_size=None, var_scale=1.0, num_samples=1,
        action_noise=0.0, early_stop_patience=5, early_stop_rel_delta=1e-2, device="cpu", seed=1234,
        optimizer_cls=torch.optim.AdamW, optimizer_kwargs=None,
    ):
        self.model = model
        self.n_steps = n_steps
        self.action_dim = action_dim
        self.min_its = min_its
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.action_noise = action_noise
        self.early_stop_patience = early_stop_patience
        self.early_stop_rel_delta = early_stop_rel_delta
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1.0}

    def _init_actions(self, n_envs, horizon):
        queue = self.model.latent_action_queue()
        if queue is None:
            mean, std, bounds = torch.zeros(self.action_dim, device=self.device), torch.ones(self.action_dim, device=self.device), None
        else:
            queue = queue.detach().to(self.device)
            mean, std = queue.mean(0), queue.std(0, unbiased=False).clamp_min(1e-4)
            bounds = (torch.quantile(queue, 0.02, dim=0), torch.quantile(queue, 0.98, dim=0))
        actions = mean.view(1, 1, 1, -1).expand(n_envs, self.num_samples, horizon, -1).clone()
        noise = torch.randn(actions[:, 1:].shape, generator=self.torch_gen, device=self.device)
        actions[:, 1:] += noise * (std * self.var_scale).view(1, 1, 1, -1)
        return actions, bounds

    @staticmethod
    def _improved(score, best, rel_delta):
        return not np.isfinite(best) or best - score > rel_delta * max(abs(best), 1e-12)

    def solve(self, info, horizon):
        """Best actions (E, horizon, A) and the latents (E, horizon + 1, D) they predict, on CPU."""
        info = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in info.items()}
        n_envs = len(info["embed_0"])
        init, bounds = self._init_actions(n_envs, horizon)
        batch_size = self.batch_size or n_envs
        top = []
        for start in range(0, n_envs, batch_size):
            end = min(start + batch_size, n_envs)
            batch = {
                k: v[start:end].unsqueeze(1).expand(end - start, self.num_samples, *v.shape[1:]) if torch.is_tensor(v) else v
                for k, v in info.items()
            }
            actions = init[start:end].clone().requires_grad_(True)
            optim = self.optimizer_cls([actions], **self.optimizer_kwargs)
            best, stalled = float("inf"), 0
            for step in range(self.n_steps):
                costs = self.model.get_cost(batch, actions)
                costs.sum().backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                if self.action_noise > 0:
                    noise = torch.randn(actions.shape, generator=self.torch_gen, device=self.device)
                    actions.data += noise * self.action_noise
                if bounds is not None:
                    actions.data.clamp_(min=bounds[0].view(1, 1, 1, -1), max=bounds[1].view(1, 1, 1, -1))
                if self.early_stop_patience > 0:
                    score = float(costs.min(dim=1).values.detach().mean())
                    if self._improved(score, best, self.early_stop_rel_delta):
                        best, stalled = score, 0
                    else:
                        stalled += 1
                        if step + 1 >= self.min_its and stalled >= self.early_stop_patience:
                            break
            with torch.no_grad():
                final = self.model.get_cost(batch, actions)
                best_idx = final.argsort(dim=1)[:, 0]
                top.append(actions[torch.arange(end - start), best_idx].detach())
        top = torch.cat(top)
        with torch.no_grad():
            pred = self.model.rollout(info["embed_0"][:, None], top[:, None])[:, 0]
        return top.cpu(), pred.cpu()


class HierarchicalSolver:
    """One ``GradientSolver`` per level of an ``HJEPA``, for ``WorldModelPolicy``.

    ``plan`` holds ``receding_horizon`` (level-1 steps executed per call), ``level1`` with
    ``horizon``, ``pre_decay_horizon``, ``action_block``, ``cost_last_n``, ``action_cost_weight``,
    and ``level2`` with ``horizon``, ``action_dim``, ``cost_last_n``. Horizons shrink over the
    ``eval_budget`` as in H-JEPA: level 2 from ``horizon`` to 1, level 1 holds
    ``pre_decay_horizon`` until level 2 reaches 1 and then shrinks too.
    """

    def __init__(self, model, solvers, plan, eval_budget):
        self.model = model
        self.plan = plan
        self.eval_budget = int(eval_budget)
        models = [LevelCostModel(model, model.upper), LevelCostModel(model.upper)]
        self.solvers = [
            hydra.utils.instantiate(solvers[f"level{i}"], model=cm, action_dim=dim)
            for i, (cm, dim) in enumerate(zip(models, [plan.level1.action_block * 2, plan.level2.action_dim]), 1)
        ]

    def configure(self, *, action_space, n_envs, config):
        assert int(np.prod(action_space.shape[1:])) == 2 and config.action_block == self.plan.level1.action_block
        self._n_envs = n_envs
        self._config = config
        self.steps_taken = 0

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self.plan.level1.action_block * 2

    @property
    def horizon(self):
        return self.plan.level1.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _horizons(self):
        """Level-2 and level-1 horizons of every planning call."""
        interval = self.plan.receding_horizon * self.plan.level1.action_block
        calls = math.ceil(self.eval_budget / interval)
        upper = decreasing_horizon_schedule(calls, self.plan.level2.horizon)
        pre = self.plan.level1.pre_decay_horizon
        if 1 not in upper:
            return upper, [pre] * calls
        tail = upper.index(1)
        return upper, [pre] * tail + decreasing_horizon_schedule(calls - tail, self.plan.level1.horizon)

    @torch.no_grad()
    def _encode(self, pixels):
        """Latents of the last frame at both levels, each (E, 1, D)."""
        emb = self.model.encode({"pixels": pixels.to(next(self.model.parameters()).device)})["emb"][:, -1:]
        return [emb, self.model.upper.embed(emb)]

    def solve(self, info_dict, init_action=None):
        plan = self.plan
        interval = plan.receding_horizon * plan.level1.action_block
        upper_horizons, lower_horizons = self._horizons()
        call = min(self.steps_taken // interval, len(lower_horizons) - 1)
        obs, goal = self._encode(info_dict["pixels"]), self._encode(info_dict["goal"])

        subgoal = subgoal_action = None
        with torch.enable_grad():
            if upper_horizons[call] > 1:
                info = {
                    "embed_0": obs[1], "goal_embed_0": goal[1], "project": False,
                    "cost_last_n": plan.level2.cost_last_n, "action_cost_weight": 0.0,
                }
                subgoal_action, subgoal = self.solvers[1].solve(info, upper_horizons[call])
            info = {"embed_0": obs[0], "cost_last_n": plan.level1.cost_last_n}
            if subgoal is None:
                info.update(goal_embed_0=goal[0], project=False, action_cost_weight=0.0)
            else:
                info.update(
                    goal_embed_0=subgoal[:, 1:2], goal_action=subgoal_action[:, :1], project=True,
                    action_cost_weight=plan.level1.action_cost_weight,
                )
            actions, _ = self.solvers[0].solve(info, lower_horizons[call])

        keep = min(plan.receding_horizon, actions.shape[1])
        self._config.receding_horizon = keep
        self.steps_taken += keep * plan.level1.action_block
        return {"actions": actions}
