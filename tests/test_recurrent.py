import pytest

from decisionrl.algorithms import RecurrentPPO
from decisionrl.envs import CartPole
from decisionrl.training import evaluate_policy
from decisionrl.wrappers import SyncVectorEnv


def test_recurrent_ppo_predict_and_reset(quiet_logger):
    agent = RecurrentPPO(CartPole(), n_steps=16, n_epochs=1, seed=0, logger=quiet_logger)
    obs, _ = CartPole().reset(seed=0)
    agent.reset_states()
    assert agent.predict(obs, deterministic=True) in (0, 1)


def test_recurrent_ppo_vector_smoke(quiet_logger):
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(3)])
    agent = RecurrentPPO(venv, n_steps=16, n_epochs=1, n_minibatches=1, seed=0, logger=quiet_logger)
    agent.learn(200)
    assert agent.num_timesteps >= 200


def test_recurrent_ppo_save_load(tmp_path, quiet_logger):
    agent = RecurrentPPO(CartPole(), n_steps=16, n_epochs=1, seed=0, logger=quiet_logger)
    agent.learn(64)
    path = str(tmp_path / "rppo.pt")
    agent.save(path)
    loaded = RecurrentPPO.load(path, env=CartPole())
    obs, _ = CartPole().reset(seed=1)
    agent.reset_states()
    loaded.reset_states()
    assert agent.predict(obs) == loaded.predict(obs)


@pytest.mark.slow
def test_recurrent_ppo_learns_cartpole(quiet_logger):
    venv = SyncVectorEnv([lambda: CartPole() for _ in range(4)])
    agent = RecurrentPPO(venv, n_steps=128, n_epochs=6, n_minibatches=4, ent_coef=0.0,
                         learning_rate=1e-3, seed=0, logger=quiet_logger)
    agent.learn(25_000)
    # `seed=` on the evaluation, not just on the agent. CartPole builds its own
    # `np.random.default_rng()` with no seed, and `reset(seed=None)` leaves it
    # alone, so an unseeded evaluation starts all twenty episodes from states
    # drawn out of OS entropy. Training here is reproducible -- three full runs
    # on one machine gave a bit-identical parameter checksum -- and the measurement
    # taken afterwards was not: 121.30, 122.50 and 119.05 for the same weights,
    # against 120.25 three times with this seed.
    #
    # That is a few points of noise, not the collapse to random-policy level
    # reported in #15, so this does not close that. It removes the one source
    # of variance that is ours.
    mean = evaluate_policy(agent, CartPole(), n_episodes=20, seed=12345)[0]
    assert mean > 60, f"RecurrentPPO failed to learn CartPole (mean_return={mean:.1f})"
