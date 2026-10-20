"""Dataset loading and fixed train / validation / test splits.

The split is controlled by its own seed, separate from the model seed. That way
we can train many models with different initialisations on *exactly the same*
held-out data, and differences between runs come from the model, not the split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Pixel intensities in the digits dataset are integers in [0, 16].
DIGITS_MAX_PIXEL = 16.0


@dataclass(frozen=True)
class DataSplits:
    """Feature matrices X: (n_samples, n_features) and integer labels y: (n_samples,)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_digits_splits(
    split_seed: int = 0, val_fraction: float = 0.2, test_fraction: float = 0.2
) -> DataSplits:
    """Load scikit-learn's 8x8 digits and split them into train / validation / test.

    Splits are stratified, so each keeps the ~10% share of every class. Features
    are scaled to [0, 1] by dividing by the known maximum pixel value. Using a
    fixed constant means no statistics are computed from the data, so no
    information can leak from the validation or test sets into training.

    Returns:
        DataSplits with X_*: (n, 64) float64 in [0, 1] and y_*: (n,) int64 in 0..9.
    """
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("fractions must be in (0, 1) and sum to less than 1")

    digits = load_digits()
    X = digits.data / DIGITS_MAX_PIXEL
    y = digits.target.astype(np.int64)

    # First carve off the test set, then split the remainder into train and validation.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=split_seed
    )
    val_share_of_rest = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share_of_rest, stratify=y_rest, random_state=split_seed
    )
    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)
