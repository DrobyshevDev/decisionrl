# Contextual bandits

Many operational decisions are one-shot: set a price, pick which offer to show, choose a
treatment, and observe the outcome immediately with no long-term state to plan through.
These are contextual bandit problems, and they do not need the full reinforcement-learning
machinery. `decisionrl.bandits` provides three closed-form linear methods with no gradient
training, so they are fast, deterministic under a seed, and easy to reason about.

All three assume a linear reward model, `E[r | x, a] = x . theta_a`, and keep the same
per-arm ridge-regression statistics. They differ only in how they act on those estimates:

| Algorithm | Exploration strategy |
|---|---|
| `LinUCB` | add an optimism bonus (upper confidence bound) to the estimate |
| `LinearThompsonSampling` | sample a parameter from the posterior, act greedily |
| `EpsilonGreedyBandit` | act greedily, explore uniformly with probability epsilon |

## Environment

[`ContextualBandit`](https://github.com/DrobyshevDev/decisionrl/blob/main/src/decisionrl/envs/contextual_bandit.py)
is the canonical linear testbed: each round draws a context vector, every arm has an
unknown linear value, and the best arm depends on the context. The step `info` reports the
exact regret (the gap to the optimal arm), so cumulative regret can be measured directly.

## Usage

```python
from decisionrl.envs import ContextualBandit
from decisionrl.bandits import LinUCB, run_bandit

env = ContextualBandit(n_arms=6, n_features=8, horizon=3000)
agent = LinUCB(n_arms=6, n_features=8, alpha=1.0)
result = run_bandit(agent, env, seed=0)
print(result["cumulative_regret"])
```

To use one in your own loop, call `select(context)` and `update(context, arm, reward)`
directly:

```python
arm = agent.select(context)
# ... obtain reward for the chosen arm ...
agent.update(context, arm, reward)
```

## Results

Cumulative regret over 3000 rounds on a 6-arm, 8-feature problem, averaged over 5 seeds
(lower is better). Reproduce with `python examples/contextual_bandits.py`:

| Algorithm | Cumulative regret |
|---|---:|
| LinUCB | 23.9 |
| Thompson sampling | 49.1 |
| epsilon-greedy (0.1) | 158.0 |
| random | 1243.3 |

The confidence-based methods explore where they are uncertain rather than uniformly, so
their regret grows sublinearly and stays a small fraction of both epsilon-greedy and a
random policy.

To estimate what a new bandit policy would earn from a log of past decisions, before
deploying it, see [off-policy evaluation](ope.md).
