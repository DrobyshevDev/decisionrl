"""Running normalization of observations and rewards.

These are among the most impactful "best practices" for on-policy algorithms
(PPO/A2C) and continuous control. Statistics are updated online with
:class:`~decisionrl.utils.running_mean_std.RunningMeanStd`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.env import Env, Wrapper
from ..core.spaces import Box
from ..utils.running_mean_std import RunningMeanStd

__all__ = ["NormalizeObservation", "NormalizeReward"]


class NormalizeObservation(Wrapper):
    """Standardize observations to zero mean / unit variance online."""

    def __init__(self, env: Env, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__(env)
        assert isinstance(self.observation_space, Box), "NormalizeObservation needs a Box space"
        self.rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training = True
        low = np.full(self.observation_space.shape, -clip, dtype=np.float32)  # type: ignore[type-var]
        high = np.full(self.observation_space.shape, clip, dtype=np.float32)  # type: ignore[type-var]
        self.observation_space = Box(low, high, dtype=np.float32)

    def set_training(self, training: bool = True) -> None:
        """Freeze (``False``) or resume (``True``) updates to the running statistics.

        Call ``set_training(False)`` before evaluation so the policy sees observations
        normalized with the statistics learned during training, rather than statistics
        that keep drifting toward the evaluation distribution.
        """
        self.training = bool(training)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.training:
            self.rms.update(obs[None])
        out = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize(np.asarray(obs, dtype=np.float32)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(np.asarray(obs, dtype=np.float32)), reward, terminated, truncated, info


class NormalizeReward(Wrapper):
    """Scale rewards by the std of the discounted return estimate.

    Follows the widely-used implementation from the PPO/OpenAI-baselines lineage:
    rewards are divided by (but not centered on) the running standard deviation
    of the discounted returns.
    """

    def __init__(self, env: Env, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__(env)
        self.rms = RunningMeanStd(shape=())
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training = True
        self._ret = 0.0

    def set_training(self, training: bool = True) -> None:
        """Freeze (``False``) or resume (``True``) reward normalization.

        Reward scaling is a training aid, so with ``training=False`` the wrapper passes
        the environment's original rewards through unchanged and stops updating its
        statistics. Evaluate with it frozen (or on the unwrapped environment) to report
        true returns rather than scaled ones.
        """
        self.training = bool(training)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self._ret = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.training:
            return obs, reward, terminated, truncated, info
        self._ret = self._ret * self.gamma + reward
        self.rms.update(np.array([self._ret]))
        norm_reward = reward / np.sqrt(self.rms.var + self.epsilon)
        norm_reward = float(np.clip(norm_reward, -self.clip, self.clip))
        if terminated or truncated:
            self._ret = 0.0
        return obs, norm_reward, terminated, truncated, info
