import numpy as np
import pytest

from adaptive_neural_systems import training
from adaptive_neural_systems.data import load_digits_splits
from adaptive_neural_systems.training import (
    NonFiniteError,
    TrainConfig,
    accuracy,
    minibatch_indices,
    sgd_step,
    train,
)


@pytest.fixture(scope="module")
def splits():
    return load_digits_splits(split_seed=0)


# --- building blocks -----------------------------------------------------------


def test_minibatches_cover_every_example_exactly_once():
    batches = list(minibatch_indices(10, 4, np.random.default_rng(0)))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))


def test_minibatch_order_is_reproducible_and_seed_dependent():
    def order(seed):
        return np.concatenate(list(minibatch_indices(50, 8, np.random.default_rng(seed))))

    assert np.array_equal(order(1), order(1))
    assert not np.array_equal(order(1), order(2))


def test_sgd_step_moves_against_the_gradient():
    params = {"w": np.array([1.0, -2.0]), "b": np.array([0.5])}
    grads = {"w": np.array([0.5, -1.0]), "b": np.array([2.0])}
    sgd_step(params, grads, learning_rate=0.1)
    np.testing.assert_allclose(params["w"], [0.95, -1.9])
    np.testing.assert_allclose(params["b"], [0.3])


def test_accuracy():
    # Zero weights, b2 favouring class 3: every prediction is 3.
    params = {
        "W1": np.zeros((64, 32)),
        "b1": np.zeros(32),
        "W2": np.zeros((32, 10)),
        "b2": np.eye(10)[3],
    }
    assert accuracy(params, np.zeros((4, 64)), np.array([3, 3, 1, 0])) == 0.5


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0.0}, {"epochs": 0}, {"batch_size": 0}])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


# --- training ------------------------------------------------------------------


def test_history_has_one_entry_per_epoch_plus_initial(splits):
    _, h = train(
        splits.X_train[:64],
        splits.y_train[:64],
        splits.X_val,
        splits.y_val,
        TrainConfig(epochs=3, batch_size=16),
    )
    assert all(len(v) == 4 for v in h.to_dict().values())


def test_training_is_reproducible(splits):
    cfg = TrainConfig(learning_rate=0.1, epochs=5, batch_size=16, seed=7)
    args = (splits.X_train[:128], splits.y_train[:128], splits.X_val, splits.y_val)
    p1, h1 = train(*args, cfg)
    p2, h2 = train(*args, cfg)
    assert h1 == h2
    assert all(np.array_equal(p1[k], p2[k]) for k in p1)


def test_validation_data_never_influences_the_weights(splits):
    # Swapping the validation set must not change anything learned from training data.
    cfg = TrainConfig(epochs=5, batch_size=16, seed=3)
    X, y = splits.X_train[:128], splits.y_train[:128]
    p1, h1 = train(X, y, splits.X_val, splits.y_val, cfg)
    p2, h2 = train(X, y, splits.X_test, splits.y_test, cfg)
    assert all(np.array_equal(p1[k], p2[k]) for k in p1)
    assert h1.train_loss == h2.train_loss


def test_network_can_overfit_a_small_batch(splits):
    # A model that cannot memorise 32 examples has a bug in its gradients or update.
    X, y = splits.X_train[:32], splits.y_train[:32]
    _, h = train(X, y, X, y, TrainConfig(learning_rate=0.5, epochs=300, batch_size=32))
    assert h.train_loss[-1] < 0.05
    assert h.train_acc[-1] == 1.0


def test_non_finite_inputs_are_detected(splits):
    X = splits.X_train[:16].copy()
    X[0, 0] = np.nan
    with pytest.raises(NonFiniteError, match="loss"):
        train(X, splits.y_train[:16], X, splits.y_train[:16], TrainConfig(epochs=1))


def test_non_finite_gradients_are_detected(splits, monkeypatch):
    def nan_backward(params, cache, y):
        return {k: np.full_like(v, np.nan) for k, v in params.items()}

    monkeypatch.setattr(training, "backward", nan_backward)
    X, y = splits.X_train[:16], splits.y_train[:16]
    with pytest.raises(NonFiniteError, match="gradient"):
        train(X, y, X, y, TrainConfig(epochs=1))
