import numpy as np

from decisionrl.envs import CartPole, GridWorld, PointMass
from decisionrl.wrappers import (
    NormalizeObservation,
    NormalizeReward,
    SyncVectorEnv,
    TimeLimit,
)


def test_time_limit_truncates():
    env = TimeLimit(GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4), max_steps=1000), max_episode_steps=3)
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_normalize_observation_space_and_output():
    env = NormalizeObservation(CartPole())
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (4,)
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(np.abs(obs) <= 10.0 + 1e-5)
        if term or trunc:
            break


def test_normalize_reward_runs():
    env = NormalizeReward(CartPole(), gamma=0.99)
    env.reset(seed=0)
    rewards = []
    for _ in range(30):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        rewards.append(r)
        if term or trunc:
            env.reset()
    assert all(np.isfinite(rewards))


def test_normalize_observation_freezes_stats_in_eval():
    env = NormalizeObservation(CartPole())
    env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample())
    env.set_training(False)
    frozen_mean, frozen_var = env.rms.mean.copy(), env.rms.var.copy()
    env.reset(seed=1)
    for _ in range(20):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break
    assert np.array_equal(env.rms.mean, frozen_mean)  # statistics no longer move
    assert np.array_equal(env.rms.var, frozen_var)


def test_normalize_reward_passes_raw_reward_in_eval():
    env = NormalizeReward(CartPole(), gamma=0.99)
    env.set_training(False)
    env.reset(seed=0)
    for _ in range(20):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert r == 1.0  # CartPole pays +1 per step; frozen wrapper returns it untouched
        if term or trunc:
            break


def test_sync_vector_env_step_shapes():
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(4)])
    assert venv.num_envs == 4
    obs, _ = venv.reset(seed=0)
    assert obs.shape == (4, 4)
    actions = [venv.single_action_space.sample() for _ in range(4)]
    obs, rewards, terminateds, truncateds, infos = venv.step(actions)
    assert obs.shape == (4, 4)
    assert rewards.shape == (4,)
    assert terminateds.shape == (4,) and truncateds.shape == (4,)


def test_sync_vector_env_autoreset_final_observation():
    # PointMass truncates at max_steps -> forces episode boundaries quickly.
    venv = SyncVectorEnv([lambda: PointMass(max_steps=2) for _ in range(2)])
    venv.reset(seed=0)
    saw_final = False
    for _ in range(6):
        actions = [np.zeros(2, np.float32) for _ in range(2)]
        _, _, terminateds, truncateds, infos = venv.step(actions)
        if "final_observation" in infos:
            saw_final = True
            for i in range(2):
                if terminateds[i] or truncateds[i]:
                    assert infos["final_observation"][i] is not None
    assert saw_final
