"""Group Relative Policy Optimization (Shao et al., 2024 — DeepSeekMath).

The policy-optimization algorithm behind modern LLM RLHF, adapted to classic
control. GRPO drops the value network entirely: instead of a learned critic it
samples a **group** of ``group_size`` episodes from the current policy and uses
the **group-normalized return** as the advantage for every step of an episode::

    A_i = (G_i - mean(group returns)) / (std(group returns) + eps)

This is *outcome supervision* — the whole trajectory shares one advantage — and it
removes critic bias and the cost of fitting a value function. Updates use the PPO
clipped surrogate plus a KL penalty to a reference policy (the pre-update policy
by default), estimated with Schulman's low-variance ``k3`` estimator.

Pairs naturally with :mod:`decisionrl.rlhf`: train a :class:`~decisionrl.rlhf.RewardModel`
from preferences, wrap the env, then optimize with GRPO — the same pipeline used to
align language models.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.policies import CategoricalActor, GaussianActor
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["GRPO"]


class GRPO(BaseAgent):
    def __init__(
        self,
        env: Env,
        learning_rate: float = 3e-4,
        group_size: int = 8,
        groups_per_update: int = 4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 256,
        ent_coef: float = 0.0,
        kl_coef: float = 0.04,
        ref_update_interval: int = 1,
        max_grad_norm: float = 0.5,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.gamma = float(gamma)
        self.group_size = int(group_size)
        self.groups_per_update = int(groups_per_update)
        self.clip_range = float(clip_range)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.ent_coef = float(ent_coef)
        self.kl_coef = float(kl_coef)
        self.ref_update_interval = int(ref_update_interval)
        self.max_grad_norm = float(max_grad_norm)
        self.hidden_sizes = tuple(hidden_sizes)

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.discrete = is_discrete(self.action_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        if self.discrete:
            self.n_actions = int(self.action_space.n)
            self.actor = CategoricalActor(self.obs_dim, self.n_actions, self.hidden_sizes, act_fn)
        else:
            self.act_dim = int(self.action_space.shape[0])
            self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.action_space.high, dtype=np.float32)
            self.actor = GaussianActor(self.obs_dim, self.act_dim, self.hidden_sizes, act_fn)
        self.actor.to(self.device)
        # Frozen reference policy for the KL penalty.
        self.ref_actor = copy.deepcopy(self.actor)
        for p in self.ref_actor.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    # -- action helpers ----------------------------------------------------
    def _log_prob(self, actor, obs_t, actions_t):
        dist = actor(obs_t)
        if self.discrete:
            return dist.log_prob(actions_t), dist.entropy()
        return dist.log_prob(actions_t).sum(dim=-1), dist.entropy().sum(dim=-1)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        if self.discrete:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item())
        action = dist.mean if deterministic else dist.sample()
        return np.clip(action.cpu().numpy()[0], self.action_low, self.action_high)

    @torch.no_grad()
    def _sample_action(self, obs):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action) if self.discrete else dist.log_prob(action).sum(dim=-1)
        a = action.cpu().numpy()[0] if not self.discrete else int(action.item())
        return a, float(log_prob.item())

    # -- rollout: groups of episodes with group-relative advantages ---------
    def _rollout_episode(self):
        obs, _ = self.env.reset()
        obs_l, act_l, logp_l = [], [], []
        ep_return = 0.0
        done = False
        while not done:
            action, log_prob = self._sample_action(obs)
            env_action = action
            if not self.discrete:
                env_action = np.clip(action, self.action_low, self.action_high)
            next_obs, reward, terminated, truncated, _ = self.env.step(env_action)
            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(action)
            logp_l.append(log_prob)
            ep_return += float(reward)
            obs = next_obs
            self.num_timesteps += 1
            done = terminated or truncated
        return obs_l, act_l, logp_l, ep_return

    def _collect(self):
        obs_all, act_all, logp_all, adv_all = [], [], [], []
        group_returns = []
        for _ in range(self.groups_per_update):
            group = [self._rollout_episode() for _ in range(self.group_size)]
            returns = np.array([g[3] for g in group], dtype=np.float32)
            group_returns.extend(returns.tolist())
            # Group-relative (whitened) advantage, shared across an episode's steps.
            baseline = returns.mean()
            std = returns.std()
            norm = (returns - baseline) / (std + 1e-8)
            for (obs_l, act_l, logp_l, _), adv in zip(group, norm):
                obs_all.extend(obs_l)
                act_all.extend(act_l)
                logp_all.extend(logp_l)
                adv_all.extend([float(adv)] * len(obs_l))
        return obs_all, act_all, logp_all, adv_all, float(np.mean(group_returns))

    def _update(self, obs_all, act_all, logp_all, adv_all) -> dict:
        obs_t = to_tensor(np.asarray(obs_all, dtype=np.float32), self.device)
        if self.discrete:
            act_t = to_tensor(np.asarray(act_all), self.device, dtype=torch.long)
        else:
            act_t = to_tensor(np.asarray(act_all, dtype=np.float32), self.device)
        old_logp_t = to_tensor(np.asarray(logp_all, dtype=np.float32), self.device)
        adv_t = to_tensor(np.asarray(adv_all, dtype=np.float32), self.device)

        with torch.no_grad():
            ref_logp_t, _ = self._log_prob(self.ref_actor, obs_t, act_t)

        n = obs_t.shape[0]
        pg_losses, kls, entropies = [], [], []
        for _epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                log_prob, entropy = self._log_prob(self.actor, obs_t[idx], act_t[idx])
                ratio = torch.exp(log_prob - old_logp_t[idx])
                adv = adv_t[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL(current || reference) via Schulman's k3 estimator (>= 0).
                log_ratio_ref = ref_logp_t[idx] - log_prob
                kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()

                loss = policy_loss + self.kl_coef * kl - self.ent_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(float(policy_loss.item()))
                kls.append(float(kl.item()))
                entropies.append(float(entropy.mean().item()))
        return {
            "policy_loss": float(np.mean(pg_losses)),
            "kl": float(np.mean(kls)),
            "entropy": float(np.mean(entropies)),
        }

    def learn(self, total_steps: int, callback=None, log_interval: int = 1) -> "GRPO":
        self._total_timesteps = self.num_timesteps + total_steps
        # Seed the env once here, as off_policy, tabular and sac_discrete do at
        # the top of their own `learn`. `_rollout_episode` resets per episode
        # without a seed, and an env owns an `np.random.default_rng()` built
        # without one, so until this line GRPO drew every episode's starting
        # state from OS entropy: `GRPO(seed=0)` returned 91.2 and then 94.7.
        # A reset keeps the generator it was handed, so the unseeded resets
        # that follow draw from a seeded stream.
        if self.seed is not None:
            self.env.reset(seed=self.seed)
        if callback is not None:
            callback.on_training_start(self)
        returns_window: deque = deque(maxlen=100)
        iteration = 0
        while self.num_timesteps < total_steps:
            if iteration % self.ref_update_interval == 0:
                self.ref_actor.load_state_dict(self.actor.state_dict())
            obs_all, act_all, logp_all, adv_all, mean_return = self._collect()
            metrics = self._update(obs_all, act_all, logp_all, adv_all)
            returns_window.append(mean_return)
            iteration += 1
            if callback is not None and not callback.on_step():
                break
            if iteration % log_interval == 0:
                self.logger.record("rollout/ep_return_mean", float(np.mean(returns_window)))
                for k, v in metrics.items():
                    self.logger.record(f"train/{k}", v)
                self.logger.dump(self.num_timesteps)
        if callback is not None:
            callback.on_training_end()
        return self

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "config": dict(
                    gamma=self.gamma, group_size=self.group_size,
                    groups_per_update=self.groups_per_update, clip_range=self.clip_range,
                    n_epochs=self.n_epochs, batch_size=self.batch_size, ent_coef=self.ent_coef,
                    kl_coef=self.kl_coef, ref_update_interval=self.ref_update_interval,
                    max_grad_norm=self.max_grad_norm, hidden_sizes=self.hidden_sizes,
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "GRPO":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.ref_actor.load_state_dict(checkpoint["actor"])
        return agent
