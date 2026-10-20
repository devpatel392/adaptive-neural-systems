"""A two-layer neural network (multilayer perceptron) written with plain NumPy.

Architecture for experiment E1.1:

    X (N, 64) -> [W1, b1] -> Z1 (N, 32) -> ReLU -> H (N, 32)
              -> [W2, b2] -> Z2 (N, 10) -> softmax -> P (N, 10)

N is the number of examples in the batch. Every operation works on the whole
batch at once: each row of X is one image, each row of P is that image's
probability distribution over the 10 digit classes.
"""

from __future__ import annotations

import numpy as np

Params = dict[str, np.ndarray]
Cache = dict[str, np.ndarray]


def init_params(
    rng: np.random.Generator, n_inputs: int = 64, n_hidden: int = 32, n_outputs: int = 10
) -> Params:
    """Create weights with He initialisation and zero biases.

    He initialisation draws W ~ Normal(0, 2 / fan_in), where fan_in is the number
    of inputs to the layer. Each pre-activation is a sum of fan_in terms, so its
    variance grows with fan_in; dividing by fan_in cancels that growth. The factor
    2 compensates for ReLU zeroing roughly half of its inputs, which halves the
    variance passed to the next layer. The result is that signals keep a similar
    scale from layer to layer instead of exploding or vanishing.

    Random (not zero) weights are essential: if all weights were equal, every
    hidden unit would compute the same function and receive the same gradient,
    so they could never become different ("symmetry").

    Args:
        rng: Explicit NumPy generator, e.g. np.random.default_rng(seed).

    Returns:
        {"W1": (n_inputs, n_hidden), "b1": (n_hidden,),
         "W2": (n_hidden, n_outputs), "b2": (n_outputs,)}
    """
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_inputs), size=(n_inputs, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_outputs)),
        "b2": np.zeros(n_outputs),
    }


def relu(Z: np.ndarray) -> np.ndarray:
    """Element-wise max(0, z). Same shape as Z."""
    return np.maximum(0.0, Z)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax: P[i, k] = exp(Z[i, k]) / sum_j exp(Z[i, j]).

    Subtracting each row's maximum first does not change the result, because the
    common factor exp(-max) cancels between numerator and denominator. It does
    prevent overflow: the largest exponent becomes exp(0) = 1, and every other
    term is in (0, 1].

    Args:
        Z: Logits, (N, n_classes).

    Returns:
        Probabilities, (N, n_classes); every row is non-negative and sums to 1.
    """
    shifted = Z - Z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def forward(params: Params, X: np.ndarray) -> tuple[np.ndarray, Cache]:
    """Compute class probabilities for a batch.

    Args:
        params: Output of init_params.
        X: Inputs, (N, n_inputs).

    Returns:
        P: Probabilities, (N, n_outputs).
        cache: The intermediate values backpropagation needs:
            X (N, n_inputs), Z1 (N, n_hidden), H (N, n_hidden), Z2 (N, n_outputs).
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    if X.ndim != 2 or X.shape[1] != W1.shape[0]:
        raise ValueError(f"X must have shape (N, {W1.shape[0]}), got {X.shape}")

    # (N, 64) @ (64, 32) -> (N, 32). Z1[i, j] = sum_k X[i, k] * W1[k, j] + b1[j]:
    # hidden unit j's weighted vote over all 64 pixels of image i.
    Z1 = X @ W1 + b1
    H = relu(Z1)
    # (N, 32) @ (32, 10) -> (N, 10). Z2[i, c] = score for class c, built from the
    # 32 hidden features of image i.
    Z2 = H @ W2 + b2
    P = softmax(Z2)
    return P, {"X": X, "Z1": Z1, "H": H, "Z2": Z2}
