"""Tests for imitation learning: BC, DAgger and GAIL."""

import numpy as np
import pytest
import torch

from decisionrl.envs import CartPole
from decisionrl.imitation import BC, GAIL, DAgger, GAILDiscriminator, collect_expert_dataset
from decisionrl.training import evaluate_policy
from decisionrl.utils import set_seed


def _expert(o):
    return 1 if (o[2] + 0.5 * o[3]) > 0 else 0  # a heuristic that balances CartPole


def test_bc_predicts_and_round_trips(tmp_path, quiet_logger):
    data = collect_expert_dataset(CartPole(), _expert, 800, seed=0)
    bc = BC(CartPole(), seed=0, logger=quiet_logger)
    bc.train(data, n_iters=100, batch_size=64)
    obs, _ = CartPole().reset(seed=0)
    assert CartPole().action_space.contains(int(bc.predict(obs)))

    path = str(tmp_path / "bc.pt")
    bc.save(path)
    loaded = BC.load(path, env=CartPole())
    for s in range(10):
        o, _ = CartPole().reset(seed=s)
        assert bc.predict(o) == loaded.predict(o)


def test_gail_discriminator_reward_is_finite():
    disc = GAILDiscriminator(4, CartPole().action_space)
    r = disc.reward(np.zeros(4, dtype=np.float32), 1)
    assert np.isfinite(r)


@pytest.mark.slow
def test_bc_imitates_expert(quiet_logger):
    data = collect_expert_dataset(CartPole(), _expert, 4000, seed=0)
    bc = BC(CartPole(), seed=0, logger=quiet_logger)
    bc.train(data, n_iters=1500, batch_size=64)
    mean_return, _ = evaluate_policy(bc, CartPole(), n_episodes=10, seed=100)
    assert mean_return > 200.0


@pytest.mark.slow
def test_dagger_imitates_expert(quiet_logger):
    dagger = DAgger(CartPole(), seed=0, logger=quiet_logger)
    dagger.learn_dagger(CartPole(), _expert, iterations=4, steps_per_iter=800, train_iters=500)
    mean_return, _ = evaluate_policy(dagger, CartPole(), n_episodes=10, seed=100)
    assert mean_return > 200.0


@pytest.mark.slow
def test_gail_imitates_expert(quiet_logger):
    data = collect_expert_dataset(CartPole(), _expert, 4000, seed=0)
    gail = GAIL(CartPole(), data, n_steps=1024, batch_size=64, n_epochs=4, seed=0, logger=quiet_logger)
    gail.learn(iterations=10, steps_per_iter=2048, disc_epochs=5)
    after, _ = evaluate_policy(gail, CartPole(), n_episodes=10, seed=100)
    # GAIL matches the expert from demonstrations alone (no env reward); random ~= 22.
    assert after > 200.0


def _spy_on_batch_moves(dataset, recorder):
    """Wrap ``dataset.sample`` so every ``batch.to(device)`` is recorded."""
    original_sample = dataset.sample

    def sample(batch_size):
        batch = original_sample(batch_size)
        original_to = batch.to

        def to(device):
            recorder.append(torch.device(device))
            return original_to(device)

        batch.to = to
        return batch

    dataset.sample = sample
    return dataset


def test_bc_moves_every_batch_to_the_actors_device(quiet_logger):
    """BC.train must hand the actor tensors that are where the actor is.

    `device="auto"` puts the actor on a GPU when the machine has one, while
    TransitionDataset defaults to "cpu" and collect_expert_dataset never passes
    anything else. Without the move, the documented way of using BC raises
    "Expected all tensors to be on the same device" on its very first batch --
    on a GPU machine, which CI is not, which is why this went unnoticed.

    Checked here as the step rather than as the crash, so it runs everywhere:
    on CI the move is a no-op, but its absence is still a failure.
    """
    moved: list = []
    data = _spy_on_batch_moves(collect_expert_dataset(CartPole(), _expert, 200, seed=0), moved)
    bc = BC(CartPole(), seed=0, logger=quiet_logger)
    bc.train(data, n_iters=3, batch_size=16)
    assert moved == [bc.device] * 3


def test_gail_moves_the_expert_batch_to_the_discriminators_device(quiet_logger):
    """The expert dataset is the caller's, and is usually on the CPU.

    GAIL builds its policy dataset on `self.device` already, so the expert side
    is the one that can arrive from somewhere else — and does, every time
    collect_expert_dataset is used the way the README uses it.
    """
    moved: list = []
    expert = _spy_on_batch_moves(collect_expert_dataset(CartPole(), _expert, 200, seed=0), moved)
    gail = GAIL(CartPole(), expert, n_steps=64, batch_size=16, n_epochs=1, seed=0,
                logger=quiet_logger)
    gail.learn(iterations=1, steps_per_iter=64, disc_epochs=2, disc_batch=16)
    assert moved == [gail.device] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU; CI runners have none")
def test_bc_trains_when_the_dataset_is_on_another_device(quiet_logger):
    """The crash itself, for anyone running the suite on a machine with a GPU.

    CI cannot run this, and saying so is the point: three tests in this file
    failed on every GPU machine and passed on every runner, for as long as that
    difference went unstated.
    """
    data = collect_expert_dataset(CartPole(), _expert, 200, seed=0)   # cpu
    bc = BC(CartPole(), seed=0, logger=quiet_logger)                  # cuda
    assert data.device != bc.device
    bc.train(data, n_iters=3, batch_size=16)


def test_gail_is_reproducible_from_its_seed(quiet_logger):
    """Two GAIL runs at one seed have to give one answer.

    They did not. `seed=` named nothing, for two reasons that `set_seed` cannot
    reach and its own docstring warns about:

    * the policy rollouts start from `self.env`, and an env owns an
      `np.random.default_rng()` built without a seed, so every iteration of
      `learn` drew its starting states from OS entropy;
    * the policy dataset the discriminator trains against was rebuilt each
      iteration as `TransitionDataset(...)` with no seed, so its minibatch
      indices came from OS entropy too, on every discriminator epoch.

    Three fresh processes at seed 0 returned 421.70, 496.20 and 385.10 before
    this; afterwards they return one number, three times.

    `set_seed` is called per run, which is what conftest does per test, because
    that is the contract as it stands: an agent's networks are initialised from
    global torch state rather than from its own `seed`, here and in every other
    algorithm in this package. So `seed=` currently means "reproducible given
    the same global state", not "reproducible". Making it mean the second is a
    change to every agent, not to this one, and is not what this test is for --
    but without saying so, the `set_seed` below looks like ceremony.

    Short on purpose -- this is about determinism, not about learning, and the
    full-size run is `test_gail_imitates_expert` above.
    """
    def once():
        set_seed(0)
        data = collect_expert_dataset(CartPole(), _expert, 2000, seed=0)
        gail = GAIL(CartPole(), data, n_steps=512, batch_size=64, n_epochs=2, seed=0,
                    logger=quiet_logger)
        gail.learn(iterations=2, steps_per_iter=1024, disc_epochs=3, disc_batch=64)
        return evaluate_policy(gail, CartPole(), n_episodes=10, seed=100)[0]

    first, second = once(), once()
    assert first == second, f"GAIL(seed=0) gave {first} and then {second}"
