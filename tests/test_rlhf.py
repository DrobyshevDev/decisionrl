"""Tests for the RLHF pipeline: reward model, preferences and env wrapper."""

import numpy as np

from decisionrl.envs import PointMass
from decisionrl.rlhf import (
    DPO,
    RewardModel,
    RewardModelWrapper,
    collect_segments,
    synthetic_preferences,
    train_reward_model,
)


def _random_policy(env, seed: int = 0):
    """A random policy whose actions are the same on every run.

    `set_seed` in conftest seeds `random`, `numpy`'s legacy global state and
    torch. It cannot seed a space: `Space.__init__` builds its own
    `np.random.default_rng()` with no seed, so `action_space.sample()` draws
    from OS entropy and every run collects different segments. Passing
    `seed=` to `collect_segments` does not help either -- that reaches
    `env.reset(seed=...)`, which is a different generator.

    The effect is not subtle. Three runs of `test_dpo_learns_preferences` at
    one thread gave accuracies of 0.9138, 0.9188 and 0.9206 before this line
    and 0.914375 three times after it. The assertion threshold is 0.8, so the
    spread usually clears it and occasionally does not -- which is the flake
    reported in #15.

    `test_scenarios.py::test_env_seeding_is_deterministic` already does this;
    it is a test about determinism, so it had to.
    """
    env.action_space.seed(seed)
    return lambda o: env.action_space.sample()


def test_reward_model_learns_preferences_and_recovers_true_reward():
    env = PointMass()
    segments = collect_segments(env, _random_policy(env), n_segments=120, seg_len=25, seed=0)
    prefs = synthetic_preferences(segments, n_pairs=800, rational=True, seed=1)

    # PointMass reward is state-only (-distance), so a state reward model suffices.
    model = RewardModel(obs_dim=2, action_space=env.action_space, use_action=False)
    metrics = train_reward_model(model, prefs, n_iters=500, batch_size=32)
    assert metrics["accuracy"] > 0.85

    # The learned reward should correlate strongly with the true reward r(s)=-||s||.
    grid = np.random.default_rng(2).uniform(-1, 1, size=(400, 2)).astype(np.float32)
    true_r = -np.linalg.norm(grid, axis=1)
    learned = model.predict_rewards(grid)
    corr = float(np.corrcoef(true_r, learned)[0, 1])
    assert corr > 0.8


def test_reward_model_wrapper_reports_true_reward():
    env = PointMass()
    model = RewardModel(obs_dim=2, action_space=env.action_space, use_action=False)
    wrapped = RewardModelWrapper(PointMass(), model)

    wrapped.reset(seed=0)
    obs, learned_reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
    assert "true_reward" in info
    assert isinstance(learned_reward, float)
    # the true reward is the real environment reward (negative distance)
    assert info["true_reward"] <= 0.0


def test_dpo_learns_preferences(quiet_logger):
    env = PointMass()
    segments = collect_segments(env, _random_policy(env), n_segments=150, seg_len=20, seed=0)
    train_prefs = synthetic_preferences(segments[:120], n_pairs=700, seed=1)
    test_prefs = synthetic_preferences(segments[120:], n_pairs=200, seed=2)

    dpo = DPO(PointMass(), beta=0.5, seed=0, logger=quiet_logger)
    metrics = dpo.train(train_prefs, n_iters=600, batch_size=32)
    assert metrics["accuracy"] > 0.8

    # The learned implicit reward ranks held-out preferences correctly. This is
    # the core DPO property and is robust; the downstream true-return improvement
    # from random-policy preferences is noisy, so it is not asserted here.
    import torch

    oa, aa, ob, ab, lab = test_prefs.sample(200, dpo.device)
    with torch.no_grad():
        margin = (dpo._segment_logprob(dpo.actor, oa, aa) - dpo._segment_logprob(dpo.reference, oa, aa)) - (
            dpo._segment_logprob(dpo.actor, ob, ab) - dpo._segment_logprob(dpo.reference, ob, ab)
        )
        held_out_acc = float((((2 * lab - 1) * margin) > 0).float().mean())
    assert held_out_acc > 0.7


def test_dpo_predict_and_save_load(tmp_path, quiet_logger):
    env = PointMass()
    segments = collect_segments(env, _random_policy(env), n_segments=40, seg_len=20, seed=0)
    prefs = synthetic_preferences(segments, n_pairs=200, seed=1)
    dpo = DPO(PointMass(), seed=0, logger=quiet_logger)
    dpo.train(prefs, n_iters=50, batch_size=16)

    obs, _ = PointMass().reset(seed=3)
    action = np.asarray(dpo.predict(obs, deterministic=True))
    assert action.shape == (2,)
    path = str(tmp_path / "dpo.pt")
    dpo.save(path)
    loaded = DPO.load(path, env=PointMass())
    np.testing.assert_allclose(
        np.asarray(dpo.predict(obs, deterministic=True)),
        np.asarray(loaded.predict(obs, deterministic=True)),
        atol=1e-5,
    )


def test_synthetic_preferences_label_higher_return():
    env = PointMass()
    segments = collect_segments(env, _random_policy(env), n_segments=40, seg_len=20, seed=3)
    prefs = synthetic_preferences(segments, n_pairs=200, rational=True, seed=4)
    for a, b, label in prefs.pairs:
        if label == 1.0:
            assert a.true_return >= b.true_return
        else:
            assert b.true_return > a.true_return
