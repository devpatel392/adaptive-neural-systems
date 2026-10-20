import numpy as np
import pytest

from adaptive_neural_systems.data import load_digits_splits


@pytest.fixture(scope="module")
def splits():
    return load_digits_splits(split_seed=0)


def test_shapes_and_sizes(splits):
    sizes = [len(splits.y_train), len(splits.y_val), len(splits.y_test)]
    assert sum(sizes) == 1797
    for X, y in [
        (splits.X_train, splits.y_train),
        (splits.X_val, splits.y_val),
        (splits.X_test, splits.y_test),
    ]:
        assert X.shape == (len(y), 64)
    # 60 / 20 / 20, allowing for rounding.
    assert abs(sizes[1] - 0.2 * 1797) <= 2 and abs(sizes[2] - 0.2 * 1797) <= 2


def test_features_scaled_to_unit_interval(splits):
    for X in (splits.X_train, splits.X_val, splits.X_test):
        assert X.min() >= 0.0 and X.max() <= 1.0


def test_splits_are_disjoint(splits):
    # Compare rows as bytes; no sample may appear in two splits.
    def rows(X):
        return {r.tobytes() for r in X}

    train, val, test = rows(splits.X_train), rows(splits.X_val), rows(splits.X_test)
    assert not (train & test) and not (val & test) and not (train & val)


def test_every_split_contains_every_class_in_similar_proportion(splits):
    for y in (splits.y_train, splits.y_val, splits.y_test):
        shares = np.bincount(y, minlength=10) / len(y)
        assert np.all(np.abs(shares - 0.1) < 0.02)


def test_same_seed_same_split_different_seed_different_split():
    a, b = load_digits_splits(split_seed=3), load_digits_splits(split_seed=3)
    c = load_digits_splits(split_seed=4)
    assert np.array_equal(a.X_test, b.X_test)
    assert not np.array_equal(a.X_test, c.X_test)


@pytest.mark.parametrize("val, test", [(0.0, 0.2), (0.5, 0.5), (0.2, 1.0)])
def test_rejects_invalid_fractions(val, test):
    with pytest.raises(ValueError):
        load_digits_splits(val_fraction=val, test_fraction=test)
