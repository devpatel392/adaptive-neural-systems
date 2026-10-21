"""Mini-batch gradient descent for the NumPy MLP.

One training step:

    1. forward pass on a mini-batch         -> cache
    2. backward pass                         -> gradients dL/dtheta
    3. update every parameter                theta <- theta - lr * dL/dtheta

One epoch is one pass over the whole training set in a freshly shuffled order.
Shuffling uses a generator seeded from the run's seed, so a run is exactly
reproducible: same seed, same initial weights, same batch order, same result.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

import numpy as np

from adaptive_neural_systems.mlp import Params, backward, cross_entropy, forward, init_params


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters of one training run."""

    learning_rate: float = 0.1
    epochs: int = 100
    batch_size: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError(f"invalid config: {self}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class History:
    """Metrics measured on the full train and validation sets.

    Index 0 is the untrained network (before any update); index e is after epoch e.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class NonFiniteError(FloatingPointError):
    """Raised when a loss or gradient becomes NaN or infinite."""


def predict(params: Params, X: np.ndarray) -> np.ndarray:
    """Most probable class for each row of X: (N, 64) -> (N,)."""
    P, _ = forward(params, X)
    return P.argmax(axis=1)


def accuracy(params: Params, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of examples whose predicted class equals the label."""
    return float((predict(params, X) == y).mean())


def evaluate(params: Params, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(mean cross-entropy, accuracy) on a full dataset."""
    P, cache = forward(params, X)
    return cross_entropy(cache["Z2"], y), float((P.argmax(axis=1) == y).mean())


def minibatch_indices(
    n_examples: int, batch_size: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield index arrays that cover 0..n_examples-1 exactly once, in random order.

    The last batch is smaller when batch_size does not divide n_examples.
    """
    order = rng.permutation(n_examples)
    for start in range(0, n_examples, batch_size):
        yield order[start : start + batch_size]


def sgd_step(params: Params, grads: Params, learning_rate: float) -> None:
    """In-place gradient descent update: theta <- theta - learning_rate * grad."""
    for name in params:
        params[name] -= learning_rate * grads[name]


def _check_finite(what: str, epoch: int, *values: float | np.ndarray) -> None:
    for value in values:
        if not np.all(np.isfinite(value)):
            raise NonFiniteError(f"non-finite {what} at epoch {epoch}")


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> tuple[Params, History]:
    """Train a freshly initialised MLP with mini-batch gradient descent.

    The seed drives two independent generators, one for the initial weights and one
    for the batch order, so changing one never shifts the other's random stream.

    Args:
        X_train: (N_train, 64), y_train: (N_train,)
        X_val: (N_val, 64), y_val: (N_val,). Used only for monitoring, never for updates.
        config: Hyperparameters.

    Returns:
        The trained parameters and per-epoch metrics.

    Raises:
        NonFiniteError: if any loss or gradient becomes NaN or infinite.
    """
    init_rng, shuffle_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(2)
    )
    params = init_params(init_rng)
    history = History()

    def record(epoch: int) -> None:
        train_loss, train_acc = evaluate(params, X_train, y_train)
        val_loss, val_acc = evaluate(params, X_val, y_val)
        _check_finite("loss", epoch, train_loss, val_loss)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

    record(0)
    for epoch in range(1, config.epochs + 1):
        for idx in minibatch_indices(len(y_train), config.batch_size, shuffle_rng):
            _, cache = forward(params, X_train[idx])
            grads = backward(params, cache, y_train[idx])
            _check_finite("gradient", epoch, *grads.values())
            sgd_step(params, grads, config.learning_rate)
        record(epoch)
    return params, history
