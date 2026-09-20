"""Shared machinery for on-policy actor-critic algorithms (A2C, PPO).

Handles environment vectorization, action sampling, and rollout collection with
GAE and correct time-limit bootstrapping. Subclasses only implement
:meth:`_update`, which consumes the filled :class:`RolloutBuffer`.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..buffers.rollout import RolloutBuffer
from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..networks.policies import CategoricalActor, GaussianActor
from ..networks.value import VNetwork
from ..utils.torch_utils import get_device, to_tensor
from ..wrappers.vector import SyncVectorEnv

__all__ = ["OnPolicyAgent"]


class OnPolicyAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        anneal_lr: bool = False,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Wrap a single env transparently so everything is vectorized internally.
        if hasattr(env, "num_envs"):
            self.venv = env
            self.num_envs = int(env.num_envs)
            obs_space = env.single_observation_space
            act_space = env.single_action_space
        else:
            self.venv = SyncVectorEnv([lambda e=env: e])  # type: ignore[misc]
            self.num_envs = 1
            obs_space = env.observation_space
            act_space = env.action_space

        # BaseAgent stores single-env spaces for predict().
        self._obs_space = obs_space
        self._act_space = act_space
        super().__init__(_SpaceHolder(obs_space, act_space), seed=seed, **kwargs)
        self.env = env

        self.device = get_device(device)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.hidden_sizes = tuple(hidden_sizes)
        self.discrete = is_discrete(act_space)

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.obs_dim = int(np.prod(obs_space.shape))
        if self.discrete:
            self.n_actions = int(act_space.n)
            self.actor = CategoricalActor(self.obs_dim, self.n_actions, self.hidden_sizes, act_fn)
        else:
            self.act_dim = int(act_space.shape[0])
            self.action_low = np.asarray(act_space.low, dtype=np.float32)
            self.action_high = np.asarray(act_space.high, dtype=np.float32)
            self.actor = GaussianActor(self.obs_dim, self.act_dim, self.hidden_sizes, act_fn)
        self.critic = VNetwork(self.obs_dim, self.hidden_sizes, act_fn)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate
        )
        self.initial_lr = float(learning_rate)
        self.anneal_lr = bool(anneal_lr)
        self.buffer = RolloutBuffer(
            self.n_steps, self.num_envs, obs_space, act_space,
            gamma=gamma, gae_lambda=gae_lambda, device=str(self.device), seed=self.seed,
        )
        self._last_obs: Optional[np.ndarray] = None
        self._last_episode_starts = np.ones(self.num_envs, dtype=np.float32)
        self.ep_return_buffer: deque = deque(maxlen=100)

    # -- action helpers ----------------------------------------------------
    def _distribution(self, obs_t: torch.Tensor):
        return self.actor(obs_t)

    def _evaluate_actions(self, obs_t, actions_t) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs_t)
        values = self.critic(obs_t)
        if self.discrete:
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy()
        else:
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values

    @torch.no_grad()
    def _select_actions(self, obs_t):
        dist = self._distribution(obs_t)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) if self.discrete else dist.log_prob(actions).sum(dim=-1)
        values = self.critic(obs_t)
        return actions, log_probs, values

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self._distribution(obs_t)
        if self.discrete:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item())
        action = dist.mean if deterministic else dist.sample()
        action = action.cpu().numpy()[0]
        return np.clip(action, self.action_low, self.action_high)

    # -- rollout collection ------------------------------------------------
    def collect_rollouts(self, callback=None) -> bool:
        self.buffer.reset()
        self._ep_returns_running = getattr(self, "_ep_returns_running", np.zeros(self.num_envs, np.float32))

        for _ in range(self.n_steps):
            obs_t = to_tensor(np.asarray(self._last_obs, dtype=np.float32), self.device)
            actions, log_probs, values = self._select_actions(obs_t)
            actions_np = actions.cpu().numpy()

            if self.discrete:
                env_actions = actions_np
            else:
                env_actions = np.clip(actions_np, self.action_low, self.action_high)

            next_obs, rewards, terminateds, truncateds, infos = self.venv.step(env_actions)
            rewards = np.asarray(rewards, dtype=np.float32).copy()
            self._ep_returns_running += rewards

            # Bootstrap value for time-limit truncations (not true terminations).
            if "final_observation" in infos:
                for i in range(self.num_envs):
                    if truncateds[i] and not terminateds[i]:
                        fo = np.asarray(infos["final_observation"][i], dtype=np.float32).reshape(1, -1)
                        with torch.no_grad():
                            v = float(self.critic(to_tensor(fo, self.device)).item())
                        rewards[i] += self.gamma * v

            self.buffer.add(
                np.asarray(self._last_obs, dtype=np.float32),
                actions_np,
                log_probs.cpu().numpy(),
                rewards,
                values.cpu().numpy(),
                self._last_episode_starts,
            )

            dones = np.logical_or(terminateds, truncateds)
            for i in range(self.num_envs):
                if dones[i]:
                    self.ep_return_buffer.append(float(self._ep_returns_running[i]))
                    self._ep_returns_running[i] = 0.0

            self._last_obs = next_obs
            self._last_episode_starts = dones.astype(np.float32)
            self.num_timesteps += self.num_envs

            if callback is not None and not callback.on_step():
                return False

        with torch.no_grad():
            last_value = self.critic(
                to_tensor(np.asarray(self._last_obs, dtype=np.float32), self.device)
            ).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_value, self._last_episode_starts)
        return True

    # -- training loop -----------------------------------------------------
    def _update(self) -> dict:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def learn(self, total_steps: int, callback=None, log_interval: int = 1) -> "OnPolicyAgent":
        self._total_timesteps = total_steps  # on-policy loop runs until num_timesteps >= total_steps
        if callback is not None:
            callback.on_training_start(self)
        if self._last_obs is None:
            self._last_obs, _ = self.venv.reset(seed=self.seed)
            self._last_episode_starts = np.ones(self.num_envs, dtype=np.float32)

        iteration = 0
        while self.num_timesteps < total_steps:
            if self.anneal_lr:
                frac = max(0.0, 1.0 - self.num_timesteps / total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = frac * self.initial_lr
            if not self.collect_rollouts(callback):
                break
            metrics = self._update()
            iteration += 1
            if iteration % log_interval == 0:
                if self.ep_return_buffer:
                    self.logger.record("rollout/ep_return_mean", float(np.mean(self.ep_return_buffer)))
                for k, v in metrics.items():
                    self.logger.record(f"train/{k}", v)
                self.logger.dump(self.num_timesteps)

        if callback is not None:
            callback.on_training_end()
        return self

    # -- persistence -------------------------------------------------------
    def _config(self) -> dict:
        cfg = dict(
            n_steps=self.n_steps, gamma=self.gamma, gae_lambda=self.gae_lambda,
            hidden_sizes=self.hidden_sizes,
        )
        cfg.update(self._extra_config())
        return cfg

    def _extra_config(self) -> dict:
        return {}

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self._config(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "OnPolicyAgent":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        return agent


class _SpaceHolder:
    """Minimal object exposing observation/action spaces to BaseAgent."""

    def __init__(self, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
