"""Multi-agent PPO: independent learners or shared-policy self-play (discrete).

* ``shared_policy=True`` (self-play): one policy controls every agent and learns
  from all of their experience at once - the agents are treated as parallel
  columns of a single rollout buffer.
* ``shared_policy=False`` (independent PPO / IPPO): each agent has its own policy,
  value function and buffer, updated independently.

Both consume a :class:`~decisionrl.multiagent.env.MultiAgentEnv` with discrete,
homogeneous action spaces.
"""

from __future__ import annotations

from collections import deque
from itertools import chain
from typing import Optional, Sequence

import numpy as np
import torch

from ..buffers.rollout import RolloutBuffer
from ..networks.policies import CategoricalActor
from ..networks.value import VNetwork
from ..utils.logger import Logger
from ..utils.torch_utils import get_device, to_tensor
from .env import MultiAgentEnv

__all__ = ["MultiAgentPPO"]


def _ppo_update(actor, critic, optimizer, buffer, clip_range, ent_coef, vf_coef,
                n_epochs, batch_size, max_grad_norm) -> dict:
    params = list(chain(actor.parameters(), critic.parameters()))
    pg_losses, entropies = [], []
    for _ in range(n_epochs):
        for batch in buffer.get(batch_size):
            dist = actor(batch.obs)
            log_probs = dist.log_prob(batch.actions)
            entropy = dist.entropy()
            values = critic(batch.obs)

            adv = batch.advantages
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(log_probs - batch.old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
            pg_loss = -torch.min(surr1, surr2).mean()
            vf_loss = ((values - batch.returns) ** 2).mean()
            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            pg_losses.append(pg_loss.item())
            entropies.append(entropy.mean().item())
    return {"policy_loss": float(np.mean(pg_losses)), "entropy": float(np.mean(entropies))}


class MultiAgentPPO:
    def __init__(
        self,
        env: MultiAgentEnv,
        shared_policy: bool = True,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        n_epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: Sequence[int] = (64, 64),
        device: str = "auto",
        seed: Optional[int] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        self.env = env
        self.agents = list(env.agents)
        self.shared_policy = bool(shared_policy)
        self.device = get_device(device)
        self.n_steps = int(n_steps)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.rng = np.random.default_rng(seed)
        self.logger = logger if logger is not None else Logger()
        self.num_timesteps = 0

        a0 = self.agents[0]
        obs_space = env.observation_spaces[a0]
        act_space = env.action_spaces[a0]
        self.obs_dim = int(np.prod(obs_space.shape))
        self.n_actions = int(act_space.n)

        def build():
            actor = CategoricalActor(self.obs_dim, self.n_actions, hidden_sizes).to(self.device)
            critic = VNetwork(self.obs_dim, hidden_sizes).to(self.device)
            opt = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=learning_rate)
            return actor, critic, opt

        if self.shared_policy:
            self.actor, self.critic, self.optimizer = build()
            self.buffer = RolloutBuffer(self.n_steps, len(self.agents), obs_space, act_space,
                                        gamma=gamma, gae_lambda=gae_lambda, device=str(self.device),
                                        seed=int(self.rng.integers(1 << 31)))
        else:
            self.actors, self.critics, self.optimizers, self.buffers = {}, {}, {}, {}
            for a in self.agents:
                self.actors[a], self.critics[a], self.optimizers[a] = build()
                self.buffers[a] = RolloutBuffer(self.n_steps, 1, obs_space, act_space,
                                                gamma=gamma, gae_lambda=gae_lambda,
                                                device=str(self.device),
                                                seed=int(self.rng.integers(1 << 31)))
        self.ep_return_buffer: dict = {a: deque(maxlen=100) for a in self.agents}

    def _nets(self, agent):
        if self.shared_policy:
            return self.actor, self.critic
        return self.actors[agent], self.critics[agent]

    @torch.no_grad()
    def policy_probs(self, agent: str, obs) -> np.ndarray:
        actor, _ = self._nets(agent)
        obs_t = to_tensor(np.asarray(obs, np.float32).reshape(1, -1), self.device)
        return actor(obs_t).probs.cpu().numpy()[0]

    @torch.no_grad()
    def predict(self, obs, agent: Optional[str] = None, deterministic: bool = True) -> int:
        agent = agent or self.agents[0]
        actor, _ = self._nets(agent)
        obs_t = to_tensor(np.asarray(obs, np.float32).reshape(1, -1), self.device)
        dist = actor(obs_t)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return int(action.item())

    def learn(self, total_steps: int, log_interval: int = 10) -> "MultiAgentPPO":
        obs, _ = self.env.reset(seed=self.rng.integers(1 << 31).item())
        ep_start = dict.fromkeys(self.agents, 1.0)
        running = dict.fromkeys(self.agents, 0.0)
        iteration = 0

        while self.num_timesteps < total_steps:
            if self.shared_policy:
                self.buffer.reset()
            else:
                for a in self.agents:
                    self.buffers[a].reset()

            for _ in range(self.n_steps):
                cache = {}
                for a in self.agents:
                    actor, critic = self._nets(a)
                    obs_t = to_tensor(np.asarray(obs[a], np.float32).reshape(1, -1), self.device)
                    dist = actor(obs_t)
                    action = dist.sample()
                    cache[a] = (np.asarray(obs[a], np.float32), int(action.item()),
                                float(dist.log_prob(action).item()), float(critic(obs_t).item()))
                actions = {a: cache[a][1] for a in self.agents}
                next_obs, rewards, terms, truncs, _ = self.env.step(actions)

                if self.shared_policy:
                    self.buffer.add(
                        np.stack([cache[a][0] for a in self.agents]),
                        np.array([cache[a][1] for a in self.agents]),
                        np.array([cache[a][2] for a in self.agents], np.float32),
                        np.array([rewards[a] for a in self.agents], np.float32),
                        np.array([cache[a][3] for a in self.agents], np.float32),
                        np.array([ep_start[a] for a in self.agents], np.float32),
                    )
                else:
                    for a in self.agents:
                        self.buffers[a].add(cache[a][0][None], np.array([cache[a][1]]),
                                            np.array([cache[a][2]], np.float32),
                                            np.array([rewards[a]], np.float32),
                                            np.array([cache[a][3]], np.float32),
                                            np.array([ep_start[a]], np.float32))

                for a in self.agents:
                    running[a] += rewards[a]
                    ep_start[a] = float(terms[a] or truncs[a])
                    if terms[a] or truncs[a]:
                        self.ep_return_buffer[a].append(running[a])
                        running[a] = 0.0
                obs = next_obs
                self.num_timesteps += 1

            metrics = self._update(obs, ep_start)
            iteration += 1
            if log_interval and iteration % log_interval == 0:
                for a in self.agents:
                    if self.ep_return_buffer[a]:
                        self.logger.record(f"{a}/ep_return", float(np.mean(self.ep_return_buffer[a])))
                self.logger.record("train/entropy", metrics.get("entropy", 0.0))
                self.logger.dump(self.num_timesteps)
        return self

    def _update(self, obs, ep_start) -> dict:
        if self.shared_policy:
            with torch.no_grad():
                last_val = np.array([
                    self.critic(to_tensor(np.asarray(obs[a], np.float32).reshape(1, -1), self.device)).item()
                    for a in self.agents
                ], dtype=np.float32)
            last_done = np.array([ep_start[a] for a in self.agents], dtype=np.float32)
            self.buffer.compute_returns_and_advantages(last_val, last_done)
            return _ppo_update(self.actor, self.critic, self.optimizer, self.buffer,
                               self.clip_range, self.ent_coef, self.vf_coef,
                               self.n_epochs, self.batch_size, self.max_grad_norm)
        metrics = {}
        for a in self.agents:
            with torch.no_grad():
                last_val = np.array([
                    self.critics[a](to_tensor(np.asarray(obs[a], np.float32).reshape(1, -1), self.device)).item()
                ], dtype=np.float32)
            self.buffers[a].compute_returns_and_advantages(last_val, np.array([ep_start[a]], np.float32))
            metrics = _ppo_update(self.actors[a], self.critics[a], self.optimizers[a], self.buffers[a],
                                  self.clip_range, self.ent_coef, self.vf_coef,
                                  self.n_epochs, self.batch_size, self.max_grad_norm)
        return metrics
