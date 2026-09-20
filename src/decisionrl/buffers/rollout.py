"""On-policy rollout buffer with Generalized Advantage Estimation (GAE).

Handles time-limit truncation correctly: the caller augments the reward at a
truncated step with ``gamma * V(final_obs)`` while marking the episode boundary,
exactly as in Schulman et al. (2016) / Stable-Baselines3. This keeps advantage
estimates unbiased when episodes are cut short by a time limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from ..core.spaces import Space, is_discrete

__all__ = ["RolloutBuffer", "RolloutBatch"]


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        observation_space: Space,
        action_space: Space,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.n_steps = int(n_steps)
        self.num_envs = int(num_envs)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = torch.device(device)
        # Own seeded RNG for minibatch shuffling, so two agents in one process with
        # different seeds do not share (and perturb) a single global stream.
        self.rng = np.random.default_rng(seed)

        self.discrete_actions = is_discrete(action_space)
        obs_shape = observation_space.shape if observation_space.shape is not None else ()
        act_shape = () if self.discrete_actions else (action_space.shape or ())

        shape = (self.n_steps, self.num_envs)
        self.obs = np.zeros((*shape, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(
            (*shape, *act_shape), dtype=np.int64 if self.discrete_actions else np.float32
        )
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.episode_starts = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def add(self, obs, action, log_prob, reward, value, episode_start) -> None:
        assert self.ptr < self.n_steps, "rollout buffer is full; call reset()"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.episode_starts[self.ptr] = episode_start
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value: np.ndarray, last_done: np.ndarray) -> None:
        """GAE backward pass. ``last_value`` = V(obs) after the final step."""
        last_value = np.asarray(last_value, dtype=np.float32).reshape(self.num_envs)
        last_done = np.asarray(last_done, dtype=np.float32).reshape(self.num_envs)

        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def _flat(self, arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        flat = arr.reshape((self.n_steps * self.num_envs, *arr.shape[2:]))
        return torch.as_tensor(flat, device=self.device, dtype=dtype)

    def get(self, batch_size: Optional[int] = None) -> Iterator[RolloutBatch]:
        """Yield shuffled minibatches over the whole rollout."""
        total = self.n_steps * self.num_envs
        batch_size = total if batch_size is None else batch_size

        obs = self._flat(self.obs, torch.float32)
        actions = self._flat(self.actions, torch.long if self.discrete_actions else torch.float32)
        log_probs = self._flat(self.log_probs, torch.float32)
        advantages = self._flat(self.advantages, torch.float32)
        returns = self._flat(self.returns, torch.float32)
        values = self._flat(self.values, torch.float32)

        indices = self.rng.permutation(total)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                obs=obs[idx],
                actions=actions[idx],
                old_log_probs=log_probs[idx],
                advantages=advantages[idx],
                returns=returns[idx],
                old_values=values[idx],
            )
