"""Tests for off-policy evaluation of contextual bandits."""

import numpy as np

from decisionrl.envs import ContextualBandit
from decisionrl.ope import (
    collect_bandit_log,
    direct_method,
    doubly_robust,
    greedy_target_probs,
    inverse_propensity_score,
    self_normalized_ips,
    uniform_behavior,
)


def _log_and_target(n_arms=5, n_features=6, n_rounds=6000):
    env = ContextualBandit(n_arms=n_arms, n_features=n_features, horizon=10**9, noise=0.1, seed=1)
    log = collect_bandit_log(env, uniform_behavior(n_arms, seed=0), n_rounds, seed=0)
    scorer = lambda x: env.theta @ x  # noqa: E731 - true expected reward per arm
    target = greedy_target_probs(scorer, log.contexts, n_arms)
    truth = float(np.mean(np.sum(target * (log.contexts @ env.theta.T), axis=1)))
    return log, target, truth


def test_all_estimators_recover_target_value():
    log, target, truth = _log_and_target()
    for name, est in [
        ("IPS", inverse_propensity_score(log, target)),
        ("SNIPS", self_normalized_ips(log, target)),
        ("DM", direct_method(log, target)),
        ("DR", doubly_robust(log, target)),
    ]:
        assert abs(est - truth) < 0.05, f"{name} {est:.3f} vs truth {truth:.3f}"


def test_ips_of_behavior_policy_equals_mean_reward():
    # Evaluating the (uniform) behaviour policy itself should return its own mean reward.
    log, _, _ = _log_and_target()
    uniform_target = np.full((len(log), log.n_arms), 1.0 / log.n_arms)
    assert abs(inverse_propensity_score(log, uniform_target) - log.rewards.mean()) < 1e-9


def test_doubly_robust_survives_a_broken_reward_model():
    # With a hopeless reward model (huge ridge -> predictions collapse to zero) the direct
    # method is badly biased, but doubly robust still recovers the value via the IPS term.
    log, target, truth = _log_and_target()
    assert abs(direct_method(log, target, ridge=1e9)) < 0.05          # DM collapses toward 0
    assert abs(doubly_robust(log, target, ridge=1e9) - truth) < 0.05  # DR stays accurate


def test_collect_bandit_log_shapes_and_propensities():
    env = ContextualBandit(n_arms=4, n_features=3, horizon=50, seed=2)
    log = collect_bandit_log(env, uniform_behavior(4, seed=0), 200, seed=0)
    assert len(log) == 200
    assert log.contexts.shape == (200, 3)
    assert log.n_arms == 4
    assert np.all((log.propensities > 0) & (log.propensities <= 1))
    assert np.all((log.actions >= 0) & (log.actions < 4))
