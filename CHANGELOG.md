# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `NormalizeObservation` and `NormalizeReward` gained `set_training(bool)`. Called with
  `False` before evaluation, the observation wrapper freezes its running statistics (the
  policy sees observations normalized the way it was trained, not statistics that drift
  toward the evaluation distribution) and the reward wrapper stops scaling and passes the
  environment's original rewards through, so evaluation reports true returns. This matches
  the freeze semantics of Stable-Baselines3's `VecNormalize`.

### Fixed
- The on-policy rollout buffer shuffled its minibatches with the global NumPy RNG. Two
  on-policy agents seeded differently in the same process therefore drew from one shared
  shuffle stream and perturbed each other, and the buffer never held the per-instance RNG
  the README credits every buffer with. It now owns a `numpy.random.Generator` seeded from
  the agent's seed (threaded through the on-policy base and IPPO), so PPO and the other
  on-policy agents are reproducible per seed and independent of global RNG state.

### Changed
- The top-level package now resolves its public names lazily (PEP 562 `__getattr__`),
  so `import decisionrl` imports nothing on its own. `decisionrl.envs`,
  `decisionrl.baselines`, `decisionrl.core`, `decisionrl.solvers` and
  `decisionrl.wrappers` import with PyTorch absent entirely; torch arrives on first
  use of anything that trains (`decisionrl.algorithms`, `PPO`, `decisionrl.networks`).
  `import decisionrl.envs` drops from ~1.9 s to ~0.2 s, the remainder being NumPy.
  The public API is unchanged — `from decisionrl import PPO` resolves as before.
- `decisionrl.utils` defers its `torch_utils` re-exports (`get_device`, `to_tensor`,
  `soft_update`, …) for the same reason: `decisionrl.core` imports it for `Logger`.
- **Breaking (packaging): PyTorch is now an optional dependency.** `pip install
  decisionrl` installs NumPy only and gives you the environments, the classical
  baselines, the solvers and the core API. Install `decisionrl[torch]` for the
  algorithms — everything that trains. Touching a torch-backed name without it raises
  a `ModuleNotFoundError` that names the attribute and the command that fixes it,
  rather than a bare "No module named 'torch'". `decisionrl[dev]` includes torch, so
  contributor setup is unchanged.
- `test_neuroevolution_cem_solves_cartpole` is now
  `test_neuroevolution_cem_beats_random_on_cartpole`: it takes the median of three
  seeds and measures it against the random-policy return rather than asserting a single
  seed clears 300 of a possible 500. At this budget CEM returns roughly 283 / 105 / 241
  / 97 across seeds 0–3, so the old threshold was a threshold on luck — it is what made
  CI fail on unrelated pull requests.

### Added
- Python 3.13 to the CI matrix and to the packaging classifiers.
- `decisionrl.envs.APPLIED_ENVIRONMENTS`: the applied subset named in code instead of
  counted by hand, since its size is quoted in the README, the packaging description
  and `CITATION.cff`.
- `tests/test_documented_counts.py`: the advertised algorithm and environment counts are
  now checked against the package, and `CITATION.cff`'s version against
  `decisionrl.__version__`.

### Fixed
- `NeuroevolutionAgent` never seeded its environment, so `seed=` reached only the
  optimizer's search while the rollout start states — the fitness signal itself — came
  from OS entropy. Every run was irreproducible regardless of the seed. It now seeds the
  environment once at the top of `learn`, as every other agent already did.
- The advertised counts disagreed with the package and with each other: `CITATION.cff`
  claimed twenty-two environments where twenty-four ship, and the algorithm count was
  31 in the README, the packaging description and the citation file where 32 agents are
  exported. All three now read 32 algorithms and 24 environments, 9 of them applied.
- `CITATION.cff` had no `version`, `date-released` or `type`, which left the citation
  incomplete.

## [0.4.0] - 2026-07-18

### Added
- `decisionrl.solvers`: exact dynamic-programming optima via value iteration (pure
  NumPy). The learned inventory policy matches the DP optimum (196.1), and the module
  marks the boundary where a small stationary MDP stops being solvable exactly.
- `JointPricingInventory`: an environment with coupled price and order decisions and no
  closed-form joint optimum. 20 environments now, 8 of them applied.
- Gymnasium registration: `register_envs()` and `to_gymnasium()` expose every built-in
  environment through `gymnasium.make("decisionrl/<Env>-v0")`.
- Applied RL cookbook (`docs/cookbook.md`) and a self-contained in-browser demo
  (`docs/demo/inventory.html`) that runs a trained policy against the base-stock rule.
- Nightly workflow (`.github/workflows/verify.yml`) that re-runs the multi-seed applied
  verification and fails if any reported result regresses below its baseline.

### Changed
- README rewritten in a concise, professional style without decorative content.
- Benchmarks: added a CleanRL reference column with citation; test coverage (86 percent)
  is now measured in CI with a badge.

## [0.3.0] - 2026-07-18

### Added
- **`decisionrl.baselines`** — reusable classical baselines (base-stock + the exact
  newsvendor `analytic_base_stock_level`, best fixed price, best value threshold,
  greedy price-threshold battery, per-echelon base-stock) with `rollout_return` /
  `best_of` helpers.
- **`NonstationaryInventory`** environment — the "classical methods break" case: the
  demand rate switches between regimes, so no fixed base-stock is right. An adaptive
  DQN policy that reads an EWMA of recent demand beats the best fixed base-stock
  (**278.5 ± 2.4 vs 240.7, +16%, over 3 seeds**) — the clearest "why RL, not a solver"
  example. Now 19 environments (7 applied).
- **`decisionrl.config`** — declarative experiments from YAML/JSON/dict (`build`,
  `run`, CLI `decisionrl run <config>`); new `config` extra.
- **`decisionrl.tracking`** — reproducibility manifests (git SHA, library versions,
  seed, config, metrics → JSON).
- **`render_rgb`** on every applied environment; `bars_frame` / `line_frame` render
  helpers; correctness tests for the applied env dynamics.
- **Applied RL notebook** (`examples/applied_rl.ipynb`, Colab) and a multi-seed
  verification script (`examples/verify_applied_claims.py`).

### Changed
- **Honest, multi-seed proof.** README numbers are now mean ± std over 3 seeds against
  the *strong* classical baseline (queue vs best value-threshold; energy vs greedy
  price-threshold; inventory vs the exact base-stock optimum), not single-seed figures
  or straw-man defaults. RL wins five and matches the optimum on two.
- **Repositioned end-to-end**: the broad RL library (RLHF, AlphaZero, char-GPT, swarm
  optimizers, meta-RL, serving, …) now sits under a clearly-secondary "Beyond
  operations" section; the operational core leads. Added a "Why RL, and not a solver?"
  section.
- **Real Stable-Baselines3 benchmark** in `docs/benchmarks.md` (PPO CartPole parity
  500/500; DQN CartPole 327 ± 122 vs 96 ± 57), replacing placeholders.
- **CI**: the slow learning job runs in parallel via `pytest-xdist`.
- Repaired rename artifacts: `REINFORCE_MODEL` env var → `DECISIONRL_MODEL`, and a
  broken phrase in the RLHF section.

## [0.2.0] - 2026-07-14

### Changed
- **Renamed to `decisionrl`** (was `reinforce` / distributed as `reinforce-rl`). One
  name everywhere: `pip install decisionrl`, `import decisionrl`, repo `decisionrl`.
  The old three-name split is gone. **Breaking**: update imports `reinforce.*` →
  `decisionrl.*`.
- **Repositioned around applied RL.** The project now leads with reinforcement
  learning for *operational decisions* — pricing, inventory, energy, queues, supply
  chains — with the classic baseline shipped alongside each environment so a learned
  policy can be proved better, not just asserted. The full 31-algorithm library is
  unchanged underneath.

### Added
- **Four applied environments** (NumPy-only, registered in `make_env`, each with a
  textbook baseline): `DynamicPricing` (revenue management vs best fixed price),
  `QueueAdmissionControl` (admission control vs admit-all), `EnergyMicrogrid`
  (battery arbitrage vs no-battery), `SupplyChain` (2-echelon "beer game" vs
  base-stock). `examples/applied_rl_demo.py` trains all six applied tasks and prints
  the learned-vs-baseline proof table.
- **Baseline comparison harness** (`examples/benchmark_vs_baselines.py`): trains
  `decisionrl` against Stable-Baselines3 on the same Gymnasium env, seeds and step
  budget, and reports mean ± std return and wall-clock side by side (JSON output).
  The SB3 side is optional; CleanRL comparison procedure documented in
  `docs/benchmarks.md`.
- **Meta-RL / RL²** (`decisionrl.meta.RL2Env`, `make_meta_bandit`): meta-learning by
  training a recurrent policy across a task distribution so its hidden state performs
  online adaptation with no test-time gradients. `RL2Env` turns any discrete task
  distribution into a single-trial environment (previous action/reward/done fed
  alongside each observation; the same task is kept alive across inner episodes for a
  whole trial). Trained with `RecurrentPPO`, the policy discovers an explore-then-
  exploit bandit algorithm — on held-out 5-arm Bernoulli bandits it pulls the best
  arm ~2× more often than chance. Adds `decisionrl.envs.BernoulliBandit`.
- **TRPO** (`decisionrl.algorithms.TRPO`): Trust Region Policy Optimization — the
  natural-gradient step is found by conjugate gradient on the Fisher-vector product
  (KL Hessian) and scaled by a backtracking line search that enforces the KL trust
  region and a surrogate improvement, with the value function fit by regression.
  Built on the shared on-policy rollout machinery (GAE, correct time-limit
  bootstrapping); solves CartPole. Discrete and continuous.
- **Migration guide** (`docs/migration.md`): mapping Stable-Baselines3 and CleanRL
  patterns to `decisionrl`.
- **`decisionrl play`**: CLI command to watch a trained agent run episodes (or save
  one as a GIF with `--gif`).
- **Hindsight Experience Replay** (`decisionrl.algorithms.HERDQN` + `HERReplayBuffer`,
  `decisionrl.envs.BitFlipping`): goal relabeling ("future" strategy) that makes
  sparse-reward goal-conditioned tasks learnable — solves BitFlipping (100% success)
  where vanilla DQN cannot.
- **RSSM world model** (`decisionrl.algorithms.RSSM`, `DreamerRSSM`): a proper
  Recurrent State-Space Model — a GRU deterministic state plus a stochastic latent
  with learned prior/observation-posterior, trained on sequences via an ELBO
  (reconstruction + reward + KL with free nats), and actor-critic learned in
  imagination. The world model demonstrably learns the dynamics (experimental for
  control; use MBPO for a robust model-based agent).
- **Ecosystem adapters**: `make_minigrid` (MiniGrid navigation) and
  `make_pettingzoo` (PettingZoo parallel envs -> `MultiAgentEnv`), alongside the
  existing `make_gym` / `make_atari`.
- **In-browser demo + model zoo**:

### Changed
- **`DistributedActorLearner` robustness**: the learner now polls each actor with a
  configurable `recv_timeout` (default 60s) instead of blocking on `recv()` forever —
  a crashed or hung actor raises a clear `TimeoutError`/`RuntimeError` instead of
  deadlocking the run.
- **Docstrings**: `BaseAgent.predict` now documents the observation/action contract
  (single un-batched `np.ndarray` in; `int` for discrete or `np.ndarray` for
  continuous out); training examples state their expected results.
- **Prioritized replay**: sum-tree priority updates are now vectorized
  (`SumTree.update_batch`) — a batch costs `O(depth)` NumPy ops instead of a Python
  loop per leaf, much faster for large buffers (identical results). `export_json` dumps the policy weights as JSON;
  `examples/make_browser_demo.py` builds a self-contained `docs/demo/cartpole.html`
  that runs a trained PPO policy in pure JavaScript on a canvas (no server, no
  onnxruntime, no CDN). `decisionrl.zoo` (`save_to_zoo`/`list_pretrained`/
  `load_pretrained`) stores and loads pretrained ONNX policies.
- **Live training dashboard** (`decisionrl dashboard run.csv`): a lightweight
  Flask + Plotly web dashboard that live-reads a `Logger` metrics CSV and
  auto-refreshes one chart per metric (reward/loss/...). New `dashboard` extra.
- **LLM alignment / RLHF on text** (`decisionrl.text`): a char-level GPT
  (`CharGPT` + `CharTokenizer`), supervised pre-training (`sft_train`), and
  `rlhf_finetune` — the industry RLHF loop (group-normalized advantages, GRPO-style,
  with a KL penalty to the reference/SFT model). Steers generation toward a reward
  (e.g. `char_frequency_reward`, `lexicon_reward`) without drifting into gibberish.
- **Diffusion Policy** (`DiffusionPolicy`): a conditional denoising-diffusion
  policy over continuous actions (robotics-style), trained by behavior cloning;
  clones a PointMass expert to within noise of optimal.
- **Atari convenience** (`make_atari`): standard DQN preprocessing (grayscale,
  84×84 resize, frame-skip, 4-frame stack) in one call, ready for the CNN.
- **Vectorized (batched) fitness** for evolution: `minimize(..., batched=True)`
  evaluates the whole population in one call; the benchmark functions are now
  batch-friendly (reduce over the last axis).
- **Imitation learning** (`decisionrl.imitation`): `BC` (behavioral cloning),
  `DAgger` (dataset aggregation) and `GAIL` (adversarial imitation via a
  discriminator + PPO), plus `collect_expert_dataset`. On CartPole, BC/DAgger
  clone a heuristic expert to return 500; GAIL reaches ~400 from demonstrations
  alone with no environment reward.
- **AlphaZero** (`decisionrl.alphazero`): MCTS + self-play for two-player,
  perfect-information games. Includes `TicTacToe` and `Connect4` (canonical-form
  `Game` interface), a residual policy+value net (`AlphaZeroNet`), PUCT tree
  search (`MCTS`) with Dirichlet root noise, and the `AlphaZero` self-play trainer
  with an MCTS-backed `predict`. Learns Tic-Tac-Toe to near-perfect play from
  self-play alone; `pit` / `random_player` helpers for evaluation.
- **DPO** (`decisionrl.rlhf.DPO`): Direct Preference Optimization — optimize a
  policy directly from preference pairs against a frozen reference (no reward
  model, no RL loop). Discrete + continuous; learns the implicit reward directly
  (~0.9 held-out preference accuracy).
- **Evolutionary & swarm optimization** (`decisionrl.evolution`): a unified
  ask/tell family of gradient-free optimizers — evolution strategies (`CEM`,
  `CMAES`, `DifferentialEvolution`, `GeneticAlgorithm`, `OpenAIES`, `ARS`,
  `SimulatedAnnealing`) and swarm intelligence (`PSO`, `FireflyAlgorithm`,
  `ArtificialBeeColony`, `GreyWolfOptimizer`, `BatAlgorithm`, `AntColonyTSP`) —
  benchmark functions, and a `NeuroevolutionAgent` that trains RL policies with
  any optimizer (no gradients; solves CartPole). Figures via
  `examples/evolution_demo.py`.
- **Policy serving** (`decisionrl.serving`): `export_onnx` / `export_torchscript`
  freeze the deterministic policy (+ JSON metadata); `OnnxPolicy` runs inference
  with onnxruntime alone; `create_app` exposes a FastAPI `/predict` service, with
  a `deploy/Dockerfile`. New `serve` optional-dependency extra.
- **Renders for the complex scenarios**: `render_rgb()` on `ReacherArm`,
  `Navigation2D` (with lidar) and `LunarLander` for GIFs/figures.
- **Complex scenario environments** (self-contained, NumPy-only): `ReacherArm`
  (2-link torque-controlled reaching), `Navigation2D` (continuous maze with lidar
  sensors and hard exploration), `LunarLander` (2-D rigid-body rocket landing with
  shaped + terminal reward), and `PortfolioAllocation` (allocate across correlated
  momentum assets with transaction costs). All registered in `make_env`; trained
  end-to-end in `examples/complex_scenarios.py`.
- **RLHF pipeline** (`decisionrl.rlhf`): learn a reward from *preferences* the way
  language models are aligned — `collect_segments`, `synthetic_preferences`
  (Bradley-Terry teacher), a `RewardModel` trained on the preference likelihood,
  and `RewardModelWrapper` to optimize any agent against the learned reward while
  keeping the true reward in `info` for evaluation.
- **GRPO** (`GRPO`): Group Relative Policy Optimization (DeepSeekMath) — the
  critic-free, group-normalized-advantage policy-optimization method behind modern
  LLM RLHF, with a PPO clipped objective and a KL penalty to a reference policy.
  Pairs directly with `decisionrl.rlhf`.
- **Decision Transformer** (`DecisionTransformer`): offline RL as return-conditioned
  sequence modeling (causal GPT over `(return-to-go, state, action)` tokens), with
  a `TrajectoryDataset` / `collect_trajectories` data module and a faithful
  return-conditioned `evaluate`.
- **Curiosity / intrinsic motivation** (`decisionrl.exploration`): `RND` (Random
  Network Distillation) and `ICM` (Intrinsic Curiosity Module), plus a
  `CuriosityWrapper` that adds a normalized novelty bonus to any environment so
  every agent gets exploration for sparse-reward tasks for free.
- **Distributed actors** (`DistributedActorLearner`): true multi-process
  IMPALA-style training — actor processes run env + local inference and stream
  trajectories to a central V-trace learner that broadcasts fresh weights.
- **Dreamer** (`Dreamer`, experimental): compact latent world model with an
  actor-critic trained in imagination via analytic gradients through the learned
  dynamics. The world model learns the dynamics; the policy-learning part is not
  tuned to be competitive (use `MBPO` for a robust model-based agent).
- **Property-based tests** (Hypothesis) for spaces, replay buffer storage/n-step,
  schedules and running statistics, and for the **algorithms themselves**:
  predictions stay inside the action space for arbitrary observations, and
  save/load round-trips are a no-op on the deterministic policy.
- **PyPI packaging**: distributed as `decisionrl` (imported as `decisionrl`);
  release process documented in [`RELEASING.md`](RELEASING.md) via trusted
  publishing. README gains a hero banner (`docs/assets/banner.svg`).
- **MBPO** (`MBPO`): model-based policy optimization — an `EnsembleDynamics`
  model generates short synthetic rollouts to augment SAC training on a mix of
  real and model data.
- **IMPALA** (`IMPALA`): V-trace off-policy actor-critic for parallel-actor
  training (sync/async vector envs); corrects behaviour/target policy lag across
  update epochs. Discrete and continuous action spaces.
- **Interactive Plotly dashboards**: `decisionrl.utils.plot_dashboard` renders a
  self-contained HTML dashboard (one panel per metric) from a `HistoryLogger`,
  a metrics dict, or a Logger CSV.
- **Prioritized Experience Replay + n-step for continuous control**: DDPG, TD3 and
  SAC now accept `prioritized=True` (with `per_alpha`/`per_beta_start`) and `n_step`,
  with importance-weighted critic losses and TD-error priority updates.
- **Rainbow DQN** (`Rainbow`): combines Double, Dueling, Prioritized Replay,
  n-step returns, distributional (C51) and Noisy Nets; new `NoisyLinear` /
  `RainbowNetwork` building blocks (C51's projection refactored for reuse).
- **Gymnasium vectorized training**: `make_gym_vec(id, num_envs, asynchronous)`
  vectorizes Gymnasium single envs with decisionrl's own (correct-autoreset)
  vector envs for on-policy training.
- **Model-based RL**: `DynaQ` (tabular Dyna-Q — learned model + planning steps).
- **Episode GIF recording**: `render_rgb()` on CartPole/Pendulum/GridWorld/
  MountainCar and `decisionrl.utils.record_gif`; animated agent GIFs in the README
  (`examples/record_gifs.py`).
- **Multi-agent RL** (`decisionrl.multiagent`): `MultiAgentEnv` interface,
  `RockPaperScissors`, `CoordinationGame` and the multi-step cooperative
  `MultiAgentGridWorld`, and `MultiAgentPPO` supporting both shared-policy
  self-play and independent PPO (IPPO).
- **Recurrent PPO** (`RecurrentPPO`): LSTM actor-critic with proper hidden-state
  reset masking and truncated BPTT (minibatched over environments) for partially
  observable tasks; `BaseAgent.reset_states()` hook, called by `evaluate_policy`.
- **Environments**: `MountainCar`, `MountainCarContinuous`, `Acrobot`.
- **Dict observation spaces**: `core.spaces.Dict` + `flatten`/`flatdim` and the
  `FlattenDictObservation` wrapper (multi-modal observations).
- **Offline RL**: `IQL` (implicit Q-learning) and `CQL` (conservative Q-learning)
  in addition to TD3+BC.
- **Optuna** hyperparameter search (`decisionrl.tuning.optuna_search`).
- **Weights & Biases** logging sink in `Logger`.
- Static type checking with **mypy** (config, CI step, badge); **Colab/Jupyter
  quickstart** notebook.
- **Developer experience**: `ProgressBarCallback` (live tqdm bar), `CheckpointCallback`
  (periodic saves), `EvalCallback` now saves the best model (`best_model_save_path`),
  `make_vec_env(...)` one-line vectorization, `anneal_lr` linear LR decay for PPO/A2C,
  and CLI `--version` / `--progress` / `--n-envs` / `--async`.
- **n-step returns** in `ReplayBuffer`/`PrioritizedReplayBuffer` (per-sample
  discount, correct across termination and truncation); `n_step` option on DQN
  and the continuous off-policy agents.
- **C51** and **QR-DQN** distributional value-based agents (`CategoricalQNetwork`,
  `QuantileQNetwork`).
- **Discrete SAC** (`SACDiscrete`): max-entropy off-policy for discrete actions.
- **Offline RL**: `TD3BC` (TD3 + behavior cloning) and `IQL` (implicit Q-learning),
  plus a `TransitionDataset` / `collect_dataset` data module.
- **Benchmark suite** (`examples/benchmark_scores.py`) reproducing scores for
  every algorithm; results table in `docs/benchmarks.md`.
- **CLI**: `decisionrl train|eval|list` console script and `python -m decisionrl`.
- **Registry & configs**: `make_agent` / `make_env`, per-(algo, env) tuned
  hyperparameters.
- **AsyncVectorEnv**: subprocess-based vectorized env (spawn) with the same API
  as `SyncVectorEnv`, for parallel data collection; `maybe_compile` torch.compile
  helper.
- **Pixel observations**: `CNNFeatureExtractor` + `ImageQNetwork`; DQN auto-uses a
  CNN for image `(C, H, W)` observations. Observation wrappers `FrameStack`,
  `FlattenObservation`, `OneHotObservation`.
- `py.typed` marker, pre-commit config, PyPI publish workflow (trusted publishing).
- **Documentation site** (MkDocs Material) with a GitHub Pages deploy workflow.

## [0.1.0] - 2026-07-05

Initial release.

### Added
- **Core**: Gymnasium-compatible `Env`/`Wrapper`, dependency-free `Box`/`Discrete`
  spaces, `BaseAgent` with a unified `predict`/`learn`/`save`/`load` API.
- **Environments** (no external deps): classic control `GridWorld`,
  `MultiArmedBandit`, `CartPole`, `Pendulum`, `PointMass`; **applied**
  `InventoryManagement` (operations) and `Thermostat` (HVAC/energy); plus
  optional Gymnasium interop (`make_gym`, `GymAdapter`).
- **Algorithms**:
  - Tabular: `QLearning`, `SARSA`, `ExpectedSARSA`.
  - Value-based: `DQN` with Double, Dueling and Prioritized Replay options.
  - Policy gradient: `REINFORCE` (with baseline), `A2C`, `PPO`.
  - Continuous control: `DDPG`, `TD3`, `SAC` (automatic entropy tuning).
- **Buffers**: `ReplayBuffer`, `PrioritizedReplayBuffer` (sum-tree),
  `RolloutBuffer` with GAE and correct time-limit bootstrapping.
- **Networks**: `build_mlp` with orthogonal init, Q-networks (incl. dueling),
  categorical/Gaussian/squashed-Gaussian/deterministic policies, value & Q critics.
- **Exploration**: linear/exponential/constant schedules, Gaussian and
  Ornstein-Uhlenbeck action noise.
- **Wrappers**: `TimeLimit`, `NormalizeObservation`, `NormalizeReward`,
  `SyncVectorEnv`.
- **Utils**: `set_seed`, `Logger` (stdout/CSV/optional TensorBoard),
  `HistoryLogger` (in-memory learning curves), `RunningMeanStd`, polyak/hard
  updates, explained variance.
- **Training**: `evaluate_policy`, callbacks (`EvalCallback`,
  `StopOnRewardThreshold`).
- 92 tests (unit + learning) and GitHub Actions CI across Python 3.9–3.12.
