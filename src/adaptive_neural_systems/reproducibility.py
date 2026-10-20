"""Seed control so that every experiment can be rerun and produce the same numbers.

Neural network training uses randomness in three places: weight initialisation,
data shuffling and stochastic layers such as dropout. If those random streams are
not seeded, two runs of the same experiment give different results and we cannot
tell whether a change helped or we were just lucky. Seeding every generator that
an experiment touches is therefore a precondition for any reported result.

NumPy and PyTorch are optional: they are seeded only if they are installed.
"""

from __future__ import annotations

import os
import random
from importlib.util import find_spec


def set_seed(seed: int) -> list[str]:
    """Seed every available random number generator.

    Args:
        seed: Non-negative integer used for all generators.

    Returns:
        Names of the generators that were seeded, for recording in experiment logs.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    seeded = ["python"]
    random.seed(seed)
    # Affects hash randomisation only in subprocesses started after this point.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if find_spec("numpy") is not None:
        import numpy as np

        # Seeds the legacy global state for third-party code that still uses it.
        # Our own code passes np.random.default_rng(seed) explicitly instead.
        np.random.seed(seed % 2**32)  # noqa: NPY002
        seeded.append("numpy")

    if find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        seeded.append("torch")

    return seeded
