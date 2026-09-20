# Off-policy evaluation

Before you ship a new pricing or recommendation rule, you want to know what it would earn,
without exposing customers to it first. You usually cannot A/B test every candidate, but
you do have a log of what the current rule did and what happened. Off-policy evaluation
estimates the value of a target policy from that log alone.

`decisionrl.ope` provides four estimators for the contextual-bandit case. They take a log
of `(context, action, propensity, reward)` and the target policy's action probabilities on
the logged contexts, and differ in how they trade bias against variance:

| Estimator | Idea | Trade-off |
|---|---|---|
| `inverse_propensity_score` | reweight logged rewards by target/behaviour probability | unbiased, high variance |
| `self_normalized_ips` | IPS divided by the mean weight | consistent, much steadier |
| `direct_method` | fit a reward model, average it under the target | low variance, biased if the model is wrong |
| `doubly_robust` | direct method plus an IPS correction on its residuals | unbiased if *either* piece is right |

The one requirement is coverage: the behaviour policy must explore, giving positive
probability to the actions the target might take. A purely greedy log cannot tell you about
actions it never tried.

## Usage

```python
from decisionrl.envs import ContextualBandit
from decisionrl.ope import (collect_bandit_log, uniform_behavior,
                            greedy_target_probs, doubly_robust)
import numpy as np

env = ContextualBandit(n_arms=5, n_features=6)
log = collect_bandit_log(env, uniform_behavior(5), n_rounds=6000)

# Evaluate the greedy policy under a learned reward scorer, from the log alone.
target = greedy_target_probs(scorer, log.contexts, n_arms=5)
value = doubly_robust(log, target)
```

## Does it work?

On a synthetic bandit where the true value is known, all four estimators recover it from a
log collected under a uniform behaviour policy (6000 rounds, true optimal value 0.443):

| Estimator | Estimate |
|---|---:|
| Inverse propensity score | 0.447 |
| Self-normalized IPS | 0.447 |
| Direct method | 0.444 |
| Doubly robust | 0.447 |

The "doubly" in doubly robust is worth seeing: give it a hopeless reward model (predictions
collapse to zero) and the direct method falls to about zero, but doubly robust still returns
the correct value, because the importance-weighted correction carries it. A test covers this
exact case.
