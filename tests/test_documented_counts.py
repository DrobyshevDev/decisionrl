"""The counts we advertise have to match the package we ship.

CITATION.cff and the packaging description both state how many algorithms and
environments decisionrl has. Both numbers were maintained by hand and drifted:
the citation file claimed twenty-two environments while the package exported
twenty-four and the registry knew twenty. Nothing checked them, so nothing caught
it. These tests make the prose answerable to the code.
"""

import inspect
import re
from pathlib import Path

import pytest

import decisionrl
from decisionrl.core.agent import BaseAgent
from decisionrl.core.env import Env
from decisionrl.envs import APPLIED_ENVIRONMENTS
from decisionrl.envs.gym import GymAdapter
import decisionrl.evolution

_ROOT = Path(__file__).resolve().parents[1]


def _exported(module, base, exclude=()):
    return [
        name
        for name in module.__all__
        if inspect.isclass(getattr(module, name))
        and issubclass(getattr(module, name), base)
        and getattr(module, name) not in exclude
    ]


def _read(name):
    path = _ROOT / name
    if not path.exists():  # running against an installed wheel, not the repo
        pytest.skip(f"{name} is not present next to the tests")
    return path.read_text(encoding="utf-8")


# "Algorithm" here means: a concrete agent a user can construct and train, reachable from
# the top-level namespace. Counting the algorithms subpackage instead gets this wrong twice
# over - it includes the abstract OnPolicyAgent / OffPolicyContinuousAgent bases, which are
# not algorithms, and it misses BC, DAgger, DPO and NeuroevolutionAgent, which are agents
# that happen to live elsewhere. Both mistakes are how a hand-maintained total drifts.
ALGORITHMS = len(_exported(decisionrl, BaseAgent))
# GymAdapter is interop, not a built-in environment: it has nothing to run without
# Gymnasium installed and an environment id to wrap.
ENVIRONMENTS = len(_exported(decisionrl.envs, Env, exclude=(GymAdapter,)))
APPLIED = len(APPLIED_ENVIRONMENTS)
OPTIMIZERS = len(decisionrl.evolution.OPTIMIZERS)


def test_counts_are_what_we_think_they_are():
    """Pin the totals, so a change that moves them has to say so here first."""
    assert (ALGORITHMS, ENVIRONMENTS, APPLIED) == (32, 24, 9)


def test_applied_environments_all_exist_and_are_environments():
    for name in APPLIED_ENVIRONMENTS:
        env_cls = getattr(decisionrl.envs, name)
        assert inspect.isclass(env_cls) and issubclass(env_cls, Env), name


def test_citation_file_quotes_the_real_counts():
    citation = _read("CITATION.cff")
    match = re.search(r"(\d+) algorithms and (\d+) environments, (\d+) of them applied", citation)
    assert match, "CITATION.cff no longer states the counts in the expected form"
    assert tuple(int(g) for g in match.groups()) == (ALGORITHMS, ENVIRONMENTS, APPLIED)


def test_citation_version_matches_the_package():
    citation = _read("CITATION.cff")
    match = re.search(r"^version: *(\S+)$", citation, re.MULTILINE)
    assert match, "CITATION.cff has no version field"
    assert match.group(1).strip("'\"") == decisionrl.__version__


def test_packaging_description_quotes_the_real_algorithm_count():
    pyproject = _read("pyproject.toml")
    match = re.search(r"(\d+) algorithms", pyproject)
    assert match, "pyproject.toml no longer states an algorithm count"
    assert int(match.group(1)) == ALGORITHMS


def test_readme_quotes_the_real_algorithm_count():
    """Every algorithm count in the README, not just the first one."""
    readme = _read("README.md")
    quoted = [int(n) for n in re.findall(r"(\d+) algorithms", readme)]
    assert quoted, "README no longer states an algorithm count"
    assert set(quoted) == {ALGORITHMS}, quoted


def _badge(readme, label):
    """The number a shields.io badge displays, by its label."""
    match = re.search(rf"img\.shields\.io/badge/{label}-(\d+)", readme)
    assert match, f"README has no {label} badge in the expected form"
    return int(match.group(1))


def test_readme_badges_quote_the_real_counts():
    """The badges drifted exactly where the prose did not.

    `test_readme_quotes_the_real_algorithm_count` matches "32 algorithms" and
    stayed green for weeks while the badge three lines above it read 31 and the
    environment badge read 22 — because a badge spells the figure
    `algorithms-32`, which that regex never sees. A reader sees the badge first.
    Both forms answer to the same source now.
    """
    readme = _read("README.md")
    assert _badge(readme, "algorithms") == ALGORITHMS
    assert _badge(readme, "gradient--free%20optimizers") == OPTIMIZERS

    match = re.search(r"img\.shields\.io/badge/environments-(\d+)%20\((\d+)%20applied\)", readme)
    assert match, "README has no environments badge in the expected form"
    assert tuple(int(g) for g in match.groups()) == (ENVIRONMENTS, APPLIED)


def test_zenodo_metadata_quotes_the_real_counts():
    """The archived record outlives the README, so it has to be right too."""
    zenodo = _read(".zenodo.json")
    match = re.search(r"(\d+) algorithms and (\d+) environments, (\d+) of them applied", zenodo)
    assert match, ".zenodo.json no longer states the counts in the expected form"
    assert tuple(int(g) for g in match.groups()) == (ALGORITHMS, ENVIRONMENTS, APPLIED)
