"""Imitation learning: learn a policy from demonstrations, not rewards.

* :class:`BC` - Behavioral Cloning: supervised action prediction from a fixed
  demonstration dataset.
* :class:`DAgger` - Dataset Aggregation (Ross et al., 2011): iteratively roll out
  the current policy, relabel the visited states with an expert, aggregate and
  retrain — fixes BC's compounding-error problem.
* :class:`GAIL` - Generative Adversarial Imitation Learning (Ho & Ermon, 2016): a
  discriminator learns to tell apart expert and policy transitions; the policy is
  trained with PPO to *fool* it (reward = ``-log(1 - D(s, a))``), matching the
  expert's occupancy measure without ever seeing a reward.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algorithms import PPO
from .core.agent import BaseAgent
from .core.env import Env, Wrapper
from .core.spaces import is_discrete
from .data import TransitionDataset, collect_dataset
from .networks.mlp import build_mlp
from .networks.policies import CategoricalActor, GaussianActor
from .utils.torch_utils import get_device, to_tensor

__all__ = ["BC", "DAgger", "GAIL", "GAILDiscriminator"]


class BC(BaseAgent):
    """Behavioral Cloning: supervised imitation of demonstrated actions."""

    def __init__(self, env: Env, hidden_sizes: Sequence[int] = (64, 64), learning_rate: float = 1e-3,
                 activation: str = "tanh", device: str = "auto", seed: Optional[int] = None, **kwargs):
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.hidden_sizes = tuple(hidden_sizes)
        self.discrete = is_discrete(self.action_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        if self.discrete:
            self.actor = CategoricalActor(self.obs_dim, int(self.action_space.n), self.hidden_sizes, act_fn)
        else:
            self.act_dim = int(self.action_space.shape[0])
            self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.action_space.high, dtype=np.float32)
            self.actor = GaussianActor(self.obs_dim, self.act_dim, self.hidden_sizes, act_fn)
        self.actor.to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    def train(self, dataset: TransitionDataset, n_iters: int = 2000, batch_size: int = 64,
              log_interval: int = 0) -> dict:
        losses: deque = deque(maxlen=100)
        for it in range(n_iters):
            # .to(self.device), as CQL, IQL, TD3BC and DiffusionPolicy all do.
            # `device="auto"` puts the actor on a GPU when there is one, while
            # TransitionDataset defaults to "cpu" and collect_expert_dataset
            # never passes anything else -- so without this the documented way
            # of using BC raises on the first batch of any machine with a GPU.
            batch = dataset.sample(batch_size).to(self.device)
            dist = self.actor(batch.obs)
            if self.discrete:
                loss = F.cross_entropy(dist.logits, batch.actions.long())
            else:
                loss = F.mse_loss(dist.mean, batch.actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
            if log_interval and (it + 1) % log_interval == 0:
                self.logger.record("bc/loss", float(np.mean(losses)))
                self.logger.dump(it + 1)
        return {"loss": float(np.mean(losses))}

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        if self.discrete:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item())
        action = dist.mean if deterministic else dist.sample()
        return np.clip(action.cpu().numpy()[0], self.action_low, self.action_high)

    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError("BC is supervised; use train(dataset).")

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "config": dict(hidden_sizes=self.hidden_sizes)}, path)

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "BC":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        return agent


class DAgger(BC):
    """DAgger: aggregate expert-labelled data from the policy's own rollouts."""

    def learn_dagger(self, env: Env, expert: Callable[[np.ndarray], object], iterations: int = 10,
                     steps_per_iter: int = 1000, train_iters: int = 500, batch_size: int = 64) -> "DAgger":
        obs_buf: list = []
        act_buf: list = []
        for it in range(iterations):
            # roll out the current policy (BC after the first iter), label with the expert
            obs, _ = env.reset(seed=self.seed if it == 0 else None)
            for _ in range(steps_per_iter):
                obs_buf.append(np.asarray(obs, dtype=np.float32))
                act_buf.append(np.asarray(expert(obs), dtype=np.float32))
                action = self.predict(obs, deterministic=True) if it > 0 else expert(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
            obs_arr = np.asarray(obs_buf, dtype=np.float32)
            act_arr = np.asarray(act_buf)
            dataset = TransitionDataset(
                obs_arr, act_arr, np.zeros(len(obs_arr)), obs_arr, np.zeros(len(obs_arr)),
                device=str(self.device), seed=self.seed,
            )
            self.train(dataset, n_iters=train_iters, batch_size=batch_size)
        return self


class GAILDiscriminator(nn.Module):
    """Classifier of (state, action) as expert (1) vs policy (0)."""

    def __init__(self, obs_dim: int, action_space, hidden_sizes=(64, 64), device="cpu") -> None:
        super().__init__()
        self.device = get_device(device)
        self.discrete = is_discrete(action_space)
        self.action_dim = int(action_space.n) if self.discrete else int(action_space.shape[0])
        self.net = build_mlp(obs_dim + self.action_dim, 1, hidden_sizes).to(self.device)

    def _encode(self, obs_t, act_t):
        if self.discrete:
            act = F.one_hot(act_t.long().reshape(-1), self.action_dim).float()
        else:
            act = act_t.reshape(act_t.shape[0], -1).float()
        return torch.cat([obs_t.reshape(obs_t.shape[0], -1), act], dim=-1)

    def logits(self, obs_t, act_t):
        return self.net(self._encode(obs_t, act_t)).squeeze(-1)

    @torch.no_grad()
    def reward(self, obs, action) -> float:
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        act_t = to_tensor(np.asarray(action).reshape(1, -1), self.device)
        d = torch.sigmoid(self.logits(obs_t, act_t))
        return float(-torch.log(1 - d + 1e-8).item())  # high when it looks expert-like


class _GAILRewardWrapper(Wrapper):
    def __init__(self, env: Env, discriminator: GAILDiscriminator) -> None:
        super().__init__(env)
        self.discriminator = discriminator
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["true_reward"] = float(reward)
        gail_reward = self.discriminator.reward(self._last_obs, action)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, gail_reward, terminated, truncated, info


class GAIL:
    """Generative Adversarial Imitation Learning (PPO policy + a discriminator)."""

    def __init__(self, env: Env, expert_dataset: TransitionDataset, learning_rate: float = 3e-4,
                 disc_lr: float = 3e-4, hidden_sizes=(64, 64), n_steps: int = 1024, device: str = "auto",
                 seed: Optional[int] = None, **ppo_kwargs):
        self.env = env
        self.expert = expert_dataset
        self.device = get_device(device)
        self.discriminator = GAILDiscriminator(
            int(np.prod(env.observation_space.shape)), env.action_space, hidden_sizes, device
        )
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.wrapped = _GAILRewardWrapper(env, self.discriminator)
        self.policy = PPO(self.wrapped, learning_rate=learning_rate, n_steps=n_steps,
                          hidden_sizes=hidden_sizes, device=device, seed=seed, **ppo_kwargs)
        self.rng = np.random.default_rng(seed)

    def _collect_policy_transitions(self, n: int):
        obs_l, act_l = [], []
        obs, _ = self.env.reset()
        for _ in range(n):
            action = self.policy.predict(obs, deterministic=False)
            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(np.asarray(action, dtype=np.float32))
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset()
        return np.asarray(obs_l, dtype=np.float32), np.asarray(act_l)

    def _update_discriminator(self, pol_obs, pol_act, epochs, batch_size):
        pol = TransitionDataset(pol_obs, pol_act, np.zeros(len(pol_obs)), pol_obs,
                                np.zeros(len(pol_obs)), device=str(self.device))
        losses = []
        for _ in range(epochs):
            # The expert dataset comes from the caller and is on whatever
            # device they built it on, usually the CPU; `pol` is built on
            # self.device just above. Both are moved, so the discriminator
            # never has to care which of its two inputs came from where.
            e = self.expert.sample(batch_size).to(self.device)
            p = pol.sample(batch_size).to(self.device)
            e_logits = self.discriminator.logits(e.obs, e.actions)
            p_logits = self.discriminator.logits(p.obs, p.actions)
            loss = F.binary_cross_entropy_with_logits(e_logits, torch.ones_like(e_logits)) + \
                F.binary_cross_entropy_with_logits(p_logits, torch.zeros_like(p_logits))
            self.disc_opt.zero_grad()
            loss.backward()
            self.disc_opt.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def learn(self, iterations: int = 20, steps_per_iter: int = 2048, disc_epochs: int = 5,
              disc_batch: int = 64) -> "GAIL":
        for _ in range(iterations):
            pol_obs, pol_act = self._collect_policy_transitions(steps_per_iter)
            self._update_discriminator(pol_obs, pol_act, disc_epochs, disc_batch)
            self.policy.learn(self.policy.num_timesteps + steps_per_iter)
        return self

    def predict(self, obs, deterministic: bool = True):
        return self.policy.predict(obs, deterministic=deterministic)

    def reset_states(self) -> None:
        self.policy.reset_states()


def collect_expert_dataset(env: Env, expert: Callable[[np.ndarray], object], n_transitions: int,
                           seed: Optional[int] = None, device: str = "cpu") -> TransitionDataset:
    """Convenience wrapper around :func:`decisionrl.data.collect_dataset`."""
    return collect_dataset(env, expert, n_transitions, seed=seed, device=device)
