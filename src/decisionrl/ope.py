"""Off-policy evaluation (OPE) for contextual bandits.

Estimate the value of a *target* policy from data logged under a different *behaviour*
policy, without deploying the target. This is how you decide whether a new pricing or
recommendation rule is worth shipping: you already have a log of what the old rule did and
what happened, and you want the counterfactual "what would the new rule have earned?".

Given a log of ``(context, action, propensity, reward)`` and the target policy's action
probabilities on the same contexts, four estimators with different bias/variance
trade-offs:

* :func:`inverse_propensity_score` (IPS): unbiased, but high variance when the target and
  behaviour policies disagree.
* :func:`self_normalized_ips` (SNIPS): IPS divided by the mean importance weight; slightly
  biased, consistent, and far steadier than raw IPS.
* :func:`direct_method` (DM): fit a reward model and average its predictions under the
  target; low variance, but biased when the model is wrong.
* :func:`doubly_robust` (DR): DM plus an IPS-style correction. Unbiased if *either* the
  propensities or the reward model is right, and usually the lowest error of the four.

Everything here is pure NumPy; there is no training loop and no PyTorch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

__all__ = [
    "BanditLog",
    "collect_bandit_log",
    "uniform_behavior",
    "epsilon_greedy_behavior",
    "greedy_target_probs",
    "inverse_propensity_score",
    "self_normalized_ips",
    "direct_method",
    "doubly_robust",
]


@dataclass
class BanditLog:
    """A logged contextual-bandit dataset for off-policy evaluation.

    ``propensities[i]`` is the probability the behaviour policy assigned to the action it
    actually took in context ``i``; it must be positive for the log to be usable.
    """

    contexts: np.ndarray   # (n, d)
    actions: np.ndarray    # (n,) int in [0, n_arms)
    propensities: np.ndarray  # (n,) behaviour prob of the taken action, in (0, 1]
    rewards: np.ndarray    # (n,)
    n_arms: int

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def collect_bandit_log(env, behavior: Callable[[np.ndarray], Tuple[int, float]],
                       n_rounds: int, seed: int = 0) -> BanditLog:
    """Run a stochastic behaviour policy on a contextual-bandit env and log the result.

    ``behavior(context) -> (action, propensity)`` returns the sampled action and the
    probability the behaviour policy gave it. The behaviour must explore (assign positive
    probability to the actions the target might take), or the log cannot support evaluating
    that target.
    """
    n_arms = int(env.action_space.n)
    obs, _ = env.reset(seed=seed)
    contexts, actions, propensities, rewards = [], [], [], []
    for _ in range(n_rounds):
        action, propensity = behavior(np.asarray(obs, dtype=np.float64))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        contexts.append(np.asarray(obs, dtype=np.float64))
        actions.append(int(action))
        propensities.append(float(propensity))
        rewards.append(float(reward))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return BanditLog(np.asarray(contexts), np.asarray(actions, dtype=int),
                     np.asarray(propensities), np.asarray(rewards), n_arms)


def uniform_behavior(n_arms: int, seed: int = 0) -> Callable[[np.ndarray], Tuple[int, float]]:
    """A behaviour policy that picks arms uniformly at random (full coverage)."""
    rng = np.random.default_rng(seed)

    def behave(context: np.ndarray) -> Tuple[int, float]:
        return int(rng.integers(n_arms)), 1.0 / n_arms

    return behave


def epsilon_greedy_behavior(scorer: Callable[[np.ndarray], np.ndarray], n_arms: int,
                            epsilon: float = 0.2, seed: int = 0):
    """An epsilon-greedy behaviour policy over ``scorer(context) -> per-arm scores``."""
    rng = np.random.default_rng(seed)

    def behave(context: np.ndarray) -> Tuple[int, float]:
        greedy = int(np.argmax(scorer(context)))
        arm = int(rng.integers(n_arms)) if rng.random() < epsilon else greedy
        propensity = epsilon / n_arms + (1.0 - epsilon) * (1.0 if arm == greedy else 0.0)
        return arm, propensity

    return behave


def greedy_target_probs(scorer: Callable[[np.ndarray], np.ndarray], contexts: np.ndarray,
                        n_arms: int) -> np.ndarray:
    """Action probabilities of the deterministic policy that plays ``argmax scorer``."""
    probs = np.zeros((len(contexts), n_arms))
    for i, context in enumerate(contexts):
        probs[i, int(np.argmax(scorer(context)))] = 1.0
    return probs


def _importance_weights(log: BanditLog, target_probs: np.ndarray) -> np.ndarray:
    taken = target_probs[np.arange(len(log)), log.actions]
    return taken / log.propensities


def inverse_propensity_score(log: BanditLog, target_probs: np.ndarray) -> float:
    """IPS estimate of the target policy's value. Unbiased, high variance."""
    return float(np.mean(_importance_weights(log, target_probs) * log.rewards))


def self_normalized_ips(log: BanditLog, target_probs: np.ndarray) -> float:
    """Self-normalised IPS (SNIPS): steadier than IPS, consistent, mildly biased."""
    w = _importance_weights(log, target_probs)
    total = np.sum(w)
    return float(np.sum(w * log.rewards) / total) if total > 0 else float("nan")


def _fit_reward_model(log: BanditLog, ridge: float) -> np.ndarray:
    """Per-arm ridge regression of reward on context; predict for every arm and row."""
    n, d = log.contexts.shape
    rhat = np.zeros((n, log.n_arms))
    for arm in range(log.n_arms):
        mask = log.actions == arm
        if not mask.any():
            continue
        x_arm, r_arm = log.contexts[mask], log.rewards[mask]
        gram = x_arm.T @ x_arm + ridge * np.eye(d)
        theta = np.linalg.solve(gram, x_arm.T @ r_arm)
        rhat[:, arm] = log.contexts @ theta
    return rhat


def direct_method(log: BanditLog, target_probs: np.ndarray, ridge: float = 1.0) -> float:
    """Direct method: average a fitted reward model under the target policy."""
    rhat = _fit_reward_model(log, ridge)
    return float(np.mean(np.sum(target_probs * rhat, axis=1)))


def doubly_robust(log: BanditLog, target_probs: np.ndarray, ridge: float = 1.0) -> float:
    """Doubly robust: the direct method plus an IPS correction on its residuals.

    Unbiased when either the propensities or the reward model is correct, and typically the
    most accurate of the four estimators here.
    """
    n = len(log)
    rhat = _fit_reward_model(log, ridge)
    baseline = np.sum(target_probs * rhat, axis=1)
    weights = _importance_weights(log, target_probs)
    correction = weights * (log.rewards - rhat[np.arange(n), log.actions])
    return float(np.mean(baseline + correction))
