"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_seed"]


def set_seed(seed: Optional[int], deterministic: bool = False) -> Optional[int]:
    """Seed Python, NumPy and PyTorch RNGs.

    Does **not** seed a :class:`~decisionrl.core.spaces.Space`. Each space owns
    an ``np.random.default_rng()`` built without a seed, so ``space.sample()``
    draws from OS entropy until ``space.seed()`` is called on that instance.
    A test whose policy is ``env.action_space.sample()`` is not reproducible
    because this function ran -- see the seeding in ``tests/test_rlhf.py``.

    Parameters
    ----------
    seed:
        The seed. If ``None`` nothing is done and ``None`` is returned.
    deterministic:
        If ``True``, also request deterministic CuDNN/torch algorithms. This can
        slow training down but makes runs bit-for-bit reproducible on the same
        hardware.
    """
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    # Only reaches processes started after this call. Hash randomisation is
    # fixed when an interpreter starts, so assigning it here does nothing to the
    # current one -- `PYTHONHASHSEED=0 python ...` is the only way to fix hashes
    # in this process. Kept because worker processes spawned later do inherit
    # it, and because removing it would quietly change that.
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # pragma: no cover - older torch
                torch.use_deterministic_algorithms(True)
    except ImportError:  # pragma: no cover - torch always present in practice
        pass

    return seed
