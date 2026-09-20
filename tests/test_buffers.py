import numpy as np
import pytest
import torch

from decisionrl.buffers import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer, SumTree
from decisionrl.core.spaces import Box, Discrete


def make_replay(cls=ReplayBuffer, capacity=100, **kw):
    obs_space = Box(-1.0, 1.0, shape=(3,))
    act_space = Discrete(4)
    return cls(capacity, obs_space, act_space, seed=0, **kw)


def test_replay_batch_to_moves_tensors_and_keeps_indices():
    buf = make_replay(capacity=20)
    for i in range(20):
        buf.add(np.full(3, i, np.float32), i % 4, float(i), np.full(3, i + 1, np.float32), False)
    batch = buf.sample(8)
    moved = batch.to("cpu")
    # Every tensor field lands on the requested device; a fresh object is returned.
    assert moved is not batch
    assert moved.obs.device.type == "cpu"
    assert moved.actions.device.type == "cpu"
    assert moved.rewards.device.type == "cpu"
    assert torch.equal(moved.obs, batch.obs)
    # Idempotent for a batch already on the target device.
    assert torch.equal(moved.to("cpu").next_obs, batch.next_obs)


def test_replay_add_and_len_ring():
    buf = make_replay(capacity=5)
    for i in range(7):
        buf.add(np.full(3, i, np.float32), i % 4, float(i), np.full(3, i + 1, np.float32), i == 6)
    assert len(buf) == 5  # capped at capacity
    assert buf.pos == 2  # wrapped around


def test_replay_sample_shapes_and_dtypes():
    buf = make_replay(capacity=50)
    for i in range(50):
        buf.add(np.random.randn(3).astype(np.float32), i % 4, 1.0, np.random.randn(3).astype(np.float32), False)
    batch = buf.sample(16)
    assert batch.obs.shape == (16, 3)
    assert batch.next_obs.shape == (16, 3)
    assert batch.actions.shape == (16,)
    assert batch.actions.dtype.is_floating_point is False  # long for discrete
    assert batch.rewards.shape == (16,)
    assert batch.dones.shape == (16,)


def test_replay_continuous_actions():
    obs_space = Box(-1, 1, shape=(2,))
    act_space = Box(-1, 1, shape=(2,))
    buf = ReplayBuffer(20, obs_space, act_space, seed=0)
    for _ in range(20):
        buf.add(np.zeros(2, np.float32), np.array([0.1, -0.2], np.float32), 0.0, np.zeros(2, np.float32), False)
    batch = buf.sample(8)
    assert batch.actions.shape == (8, 2)
    assert batch.actions.dtype.is_floating_point


def test_replay_has_discounts_field():
    buf = make_replay(capacity=10)
    for i in range(10):
        buf.add(np.zeros(3, np.float32), i % 4, 1.0, np.zeros(3, np.float32), False)
    batch = buf.sample(4)
    assert batch.discounts.shape == (4,)
    assert torch.allclose(batch.discounts, torch.full((4,), 0.99), atol=1e-6)


def test_nstep_aggregation():
    obs_space = Box(-10, 10, shape=(3,))
    buf = ReplayBuffer(20, obs_space, Discrete(2), seed=0, n_step=3, gamma=0.9)
    for i in range(5):
        o = np.full(3, i, np.float32)
        no = np.full(3, i + 1, np.float32)
        buf.add(o, 0, 1.0, no, False, episode_end=(i == 4))
    assert len(buf) == 5
    # r = 1 + 0.9 + 0.81 for full 3-step windows; then 1.9 and 1.0 for the flush
    np.testing.assert_allclose(buf.rewards[:5], [2.71, 2.71, 2.71, 1.9, 1.0], rtol=1e-5)
    np.testing.assert_allclose(buf.discounts[:5], [0.729, 0.729, 0.729, 0.81, 0.9], rtol=1e-5)
    # first stored transition starts at obs 0 and bootstraps from the 3rd next_obs (=3)
    np.testing.assert_array_equal(buf.obs[0], [0, 0, 0])
    np.testing.assert_array_equal(buf.next_obs[0], [3, 3, 3])


def test_nstep_early_termination():
    buf = ReplayBuffer(20, Box(-1, 1, shape=(2,)), Discrete(2), seed=0, n_step=3, gamma=0.9)
    buf.add(np.zeros(2, np.float32), 0, 1.0, np.zeros(2, np.float32), False)
    buf.add(np.zeros(2, np.float32), 0, 2.0, np.zeros(2, np.float32), True)  # terminal
    assert len(buf) == 2
    # window [t0,t1]: 1 + 0.9*2 = 2.8 (stops at terminal); then [t1]: 2.0
    np.testing.assert_allclose(buf.rewards[:2], [2.8, 2.0], rtol=1e-5)
    np.testing.assert_allclose(buf.dones[:2], [1.0, 1.0])


def test_sumtree_total_and_get():
    tree = SumTree(4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    assert tree.total == pytest.approx(10.0)
    # cumulative ranges: [0,1)->0, [1,3)->1, [3,6)->2, [6,10)->3
    assert tree.get(0.5) == 0
    assert tree.get(1.5) == 1
    assert tree.get(4.0) == 2
    assert tree.get(9.0) == 3


def test_prioritized_sampling_returns_weights_and_indices():
    buf = make_replay(PrioritizedReplayBuffer, capacity=50)
    for i in range(50):
        buf.add(np.random.randn(3).astype(np.float32), i % 4, 1.0, np.random.randn(3).astype(np.float32), False)
    batch = buf.sample(16, beta=0.4)
    assert batch.weights is not None and batch.weights.shape == (16,)
    assert batch.indices is not None and batch.indices.shape == (16,)
    assert float(batch.weights.max()) == pytest.approx(1.0)  # normalized by max


def test_prioritized_high_priority_dominates_sampling():
    buf = make_replay(PrioritizedReplayBuffer, capacity=20)
    for _ in range(20):
        buf.add(np.zeros(3, np.float32), 0, 0.0, np.zeros(3, np.float32), False)
    # give index 5 a huge TD-error
    buf.update_priorities(np.array([5]), np.array([1000.0]))
    counts = np.zeros(20)
    for _ in range(30):
        b = buf.sample(8)
        for idx in b.indices:
            counts[idx] += 1
    assert counts[5] == counts.max() and counts[5] > 0


def _reference_gae(rewards, values, episode_starts, last_value, last_done, gamma, lam):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_nt = 1.0 - last_done
            next_v = last_value
        else:
            next_nt = 1.0 - episode_starts[t + 1]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * next_nt - values[t]
        gae = delta + gamma * lam * next_nt * gae
        adv[t] = gae
    return adv, adv + np.asarray(values)


def test_rollout_gae_matches_reference():
    obs_space = Box(-1, 1, shape=(2,))
    act_space = Discrete(2)
    gamma, lam = 0.99, 0.95
    buf = RolloutBuffer(3, 1, obs_space, act_space, gamma=gamma, gae_lambda=lam)

    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    episode_starts = [1.0, 0.0, 0.0]
    for t in range(3):
        buf.add(
            np.zeros((1, 2), np.float32), np.array([0]), np.array([0.0]),
            np.array([rewards[t]]), np.array([values[t]]), np.array([episode_starts[t]]),
        )
    last_value = np.array([0.8])
    last_done = np.array([0.0])
    buf.compute_returns_and_advantages(last_value, last_done)

    exp_adv, exp_ret = _reference_gae(rewards, values, episode_starts, 0.8, 0.0, gamma, lam)
    np.testing.assert_allclose(buf.advantages[:, 0], exp_adv, rtol=1e-5)
    np.testing.assert_allclose(buf.returns[:, 0], exp_ret, rtol=1e-5)


def test_rollout_gae_resets_on_episode_boundary():
    obs_space = Box(-1, 1, shape=(1,))
    act_space = Discrete(2)
    buf = RolloutBuffer(2, 1, obs_space, act_space, gamma=0.99, gae_lambda=0.95)
    # episode ends at step 0 -> episode_starts[1] = 1
    buf.add(np.zeros((1, 1), np.float32), np.array([0]), np.array([0.0]), np.array([1.0]), np.array([0.5]), np.array([1.0]))
    buf.add(np.zeros((1, 1), np.float32), np.array([0]), np.array([0.0]), np.array([1.0]), np.array([0.5]), np.array([1.0]))
    buf.compute_returns_and_advantages(np.array([0.5]), np.array([0.0]))
    # step 0 should not bootstrap across the boundary: adv0 = r0 - v0 = 0.5
    assert buf.advantages[0, 0] == pytest.approx(0.5, abs=1e-5)


def test_rollout_get_minibatches_cover_all():
    obs_space = Box(-1, 1, shape=(2,))
    act_space = Discrete(2)
    buf = RolloutBuffer(4, 2, obs_space, act_space)
    for _ in range(4):
        buf.add(np.zeros((2, 2), np.float32), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    buf.compute_returns_and_advantages(np.zeros(2), np.zeros(2))
    seen = sum(len(b.obs) for b in buf.get(batch_size=3))
    assert seen == 4 * 2  # every transition yielded exactly once


def _rollout_shuffle_order(seed):
    obs_space, act_space = Box(-1, 1, shape=(3,)), Discrete(2)
    buf = RolloutBuffer(4, 2, obs_space, act_space, seed=seed)
    tag = 0.0
    for _ in range(4):
        obs = np.zeros((2, 3), np.float32)
        obs[:, 0] = [tag, tag + 1]  # a unique marker per (step, env)
        tag += 2
        buf.add(obs, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    buf.compute_returns_and_advantages(np.zeros(2), np.zeros(2))
    return [float(b.obs[0, 0]) for b in buf.get(batch_size=1)]


def test_rollout_shuffle_is_seeded_and_isolated_from_global_rng():
    # Same seed reproduces the minibatch order; a different global np.random state must
    # not perturb it (the buffer owns its RNG); a different seed reorders it.
    np.random.seed(0)
    a = _rollout_shuffle_order(123)
    np.random.seed(999)
    b = _rollout_shuffle_order(123)
    c = _rollout_shuffle_order(456)
    assert a == b
    assert a != c


def test_sumtree_batch_update_matches_sequential():
    from decisionrl.buffers.prioritized import SumTree

    rng = np.random.default_rng(0)
    for cap in (8, 100, 1000):  # power-of-two and not
        seq, batch = SumTree(cap), SumTree(cap)
        init = rng.random(cap)
        for i in range(cap):
            seq.update(i, float(init[i]))
        batch.update_batch(np.arange(cap), init)

        idx = rng.integers(0, cap, size=64)  # includes duplicates
        pri = rng.random(64)
        for i, p in zip(idx, pri):
            seq.update(int(i), float(p))
        batch.update_batch(idx, pri)

        np.testing.assert_allclose(seq.tree, batch.tree)
