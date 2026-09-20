# Environments

All environments follow the Gymnasium API
(`reset(seed=...) -> (obs, info)`, `step(action) -> (obs, reward, terminated,
truncated, info)`) and need no external dependencies.

## Classic control

| Env | Class | Obs | Action |
|---|---|---|---|
| Grid navigation | `GridWorld` | Discrete or one-hot Box | Discrete(4) |
| Multi-armed bandit | `MultiArmedBandit` | Box(1) | Discrete(k) |
| CartPole | `CartPole` | Box(4) | Discrete(2) |
| Pendulum swing-up | `Pendulum` | Box(3) | Box(1) |
| Point-mass reach | `PointMass` | Box(n) | Box(n) |
| Mountain Car | `MountainCar` | Box(2) | Discrete(3) |
| Mountain Car (continuous) | `MountainCarContinuous` | Box(2) | Box(1) |
| Acrobot | `Acrobot` | Box(6) | Discrete(3) |
| Bit-flipping (goal, sparse) | `BitFlipping` | Box(2n) | Discrete(n) |

## Complex scenarios

Higher-dimensional, harder tasks spanning distinct domains — richer observations,
non-linear dynamics and real credit-assignment / exploration challenges.

| Env | Class | Obs | Action | Domain |
|---|---|---|---|---|
| Two-link reaching arm | `ReacherArm` | Box(10) | Box(2) | robotic manipulation |
| 2-D maze navigation (lidar) | `Navigation2D` | Box(14) | Box(2) | navigation / hard exploration |
| Lunar lander | `LunarLander` | Box(8) | Discrete(4) | rocket soft-landing control |
| Portfolio allocation | `PortfolioAllocation` | Box(4·n) | Box(n) | finance / sequential allocation |

- **ReacherArm** — 2-DoF torque-controlled arm; dense negative-distance reward with
  a control-effort penalty. Non-linear kinematics; solve with SAC / TD3.
- **Navigation2D** — point robot with momentum and lidar range sensors must reach a
  goal past walls with gaps; progress reward, collision penalty, terminal bonus.
  Pairs well with the `decisionrl.exploration` curiosity bonuses.
- **LunarLander** — self-contained 2-D rigid-body lander; potential-based shaping
  (distance, speed, tilt, legs, fuel) plus a large land/crash terminal reward.
- **PortfolioAllocation** — allocate across correlated assets with AR(1) *momentum*
  returns and transaction costs; recent returns are predictive, so the optimal
  policy is not static and should beat equal-weight.

## Applied (operational decisions)

The flagship set: environments modelling the decisions businesses actually make.
Each pairs with a classic operations-research baseline so a learned policy can be
*proved* better, not just asserted (see `examples/applied_rl_demo.py`).

| Env | Class | Obs | Action | Problem | Baseline to beat |
|---|---|---|---|---|---|
| Inventory management | `InventoryManagement` | Box(1) | Discrete | order under stochastic demand | base-stock ("order up to S") |
| Non-stationary inventory | `NonstationaryInventory` | Box(2) | Discrete | order as the demand rate drifts between regimes | best fixed base-stock (RL beats it) |
| Thermostat / HVAC | `Thermostat` | Box(2) | Box(1) | hold a setpoint at minimal energy | bang-bang |
| Dynamic pricing | `DynamicPricing` | Box(2) | Discrete | price a finite stock over a deadline | best fixed price |
| Queue admission control | `QueueAdmissionControl` | Box(2) | Discrete(2) | admit/reject jobs at a busy server | admit-all |
| Energy microgrid | `EnergyMicrogrid` | Box(6) | Box(1) | charge/discharge a battery vs price & solar | no battery |
| Supply chain (2-echelon) | `SupplyChain` | Box(5) | Box(2) | coordinate orders across retailer + warehouse | per-echelon base-stock |
| Joint pricing + inventory | `JointPricingInventory` | Box(2) | Box(2) | set price *and* order together (coupled) | best static (price, base-stock) |

- **NonstationaryInventory** — the case where the classic formula *breaks*: the demand
  rate switches between a low and a high regime, so no single base-stock level is right.
  The agent sees inventory plus an EWMA of recent demand (a read on the current regime)
  and learns an adaptive order-up-to level that beats the best fixed base-stock — the
  clearest "why RL, not a solver" example.
- **DynamicPricing** — revenue management: sell limited inventory before a deadline
  under price-elastic, stochastic demand; the optimal price rises as stock gets
  scarce relative to time (airline/hotel pricing).
- **QueueAdmissionControl** — admit a high-value job or shed it to protect a
  finite buffer from congestion; the optimal policy is a value threshold that
  tightens as the queue fills (load shedding / call admission).
- **EnergyMicrogrid** — store cheap/surplus solar energy and discharge it into the
  evening price peak (battery arbitrage + self-consumption).
- **SupplyChain** — the "beer game": order across a serial two-echelon chain with
  shipment lead times, balancing holding vs stockout cost while avoiding bullwhip.
- **JointPricingInventory** — set price *and* reorder quantity together: price drives
  demand, demand drives ordering, and overstock is best cleared by a markdown. The
  decisions are coupled with no closed-form joint optimum; a state-dependent learned
  policy edges the best *static* (price, base-stock) rule.

See the [applied solutions](https://github.com/DrobyshevDev/decisionrl#why-decisionrl)
in the README for trained results.

## Gymnasium interop (optional)

```python
from decisionrl.envs import make_gym
env = make_gym("CartPole-v1")     # requires: pip install "decisionrl[gym]"
```

Gymnasium environments already match this library's API, so agents consume them
directly; `make_gym` simply wraps one with decisionrl's `Box`/`Discrete` spaces.

For vectorized Gymnasium training use `make_gym_vec`:

```python
from decisionrl.envs import make_gym_vec
from decisionrl.algorithms import PPO

venv = make_gym_vec("CartPole-v1", num_envs=8, asynchronous=True)
PPO(venv, n_steps=256, seed=0).learn(200_000)
```

It vectorizes Gymnasium *single* envs with decisionrl's own vector envs, which use
correct immediate-autoreset and `final_observation` bootstrapping — stable across
Gymnasium autoreset-API changes.

For Atari, `make_atari` applies the standard DQN preprocessing (grayscale, resize
to 84×84, frame-skip, 4-frame stack) — ready for the built-in CNN:

```python
from decisionrl.envs import make_atari
from decisionrl.algorithms import DQN

agent = DQN(make_atari("ALE/Pong-v5"), seed=0)   # needs: pip install "gymnasium[atari]" ale-py
```

`make_minigrid("MiniGrid-Empty-5x5-v0")` (needs `pip install minigrid`) wraps
MiniGrid navigation envs, and `decisionrl.multiagent.make_pettingzoo(...)` (needs
`pip install pettingzoo`) adapts PettingZoo parallel envs to `MultiAgentEnv`.

## Wrappers

`TimeLimit`, `NormalizeObservation`, `NormalizeReward`, `FrameStack`,
`FlattenObservation`, `OneHotObservation`, `SyncVectorEnv`, `AsyncVectorEnv`.

`NormalizeObservation` and `NormalizeReward` update their running statistics online while
training. Call `set_training(False)` before evaluation to freeze them: observations are
then normalized with the statistics learned during training, and reward normalization
(a training aid) is switched off so evaluation reports the environment's true returns.
