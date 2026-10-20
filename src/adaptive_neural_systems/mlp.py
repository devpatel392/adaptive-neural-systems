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


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise log of softmax, computed without ever forming tiny probabilities.

    log p_k = z_k - log sum_j exp(z_j)
            = (z_k - m) - log sum_j exp(z_j - m),   with m = max_j z_j.

    Computing log(softmax(Z)) directly fails when a probability underflows to 0.0
    (log 0 = -inf). This form only takes the log of a sum that is at least 1.

    Args:
        Z: Logits, (N, n_classes).

    Returns:
        Log-probabilities, (N, n_classes); every entry is <= 0.
    """
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(Z2: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy loss L = -(1/N) sum_i log p[i, y_i], from logits.

    Args:
        Z2: Logits, (N, n_classes).
        y: Integer class labels, (N,).

    Returns:
        Scalar loss. 0 means the correct class always gets probability 1;
        ln(n_classes) is the loss of a uniform guess.
    """
    if y.shape != (Z2.shape[0],):
        raise ValueError(f"y must have shape ({Z2.shape[0]},), got {y.shape}")
    n = Z2.shape[0]
    # Pick log p[i, y_i] for each row i with fancy indexing: rows 0..N-1, columns y.
    return float(-log_softmax(Z2)[np.arange(n), y].mean())


def loss(params: Params, X: np.ndarray, y: np.ndarray) -> float:
    """Forward pass followed by cross-entropy. X: (N, n_inputs), y: (N,)."""
    _, cache = forward(params, X)
    return cross_entropy(cache["Z2"], y)


def backward(params: Params, cache: Cache, y: np.ndarray) -> Params:
    """Gradients of the mean cross-entropy with respect to every parameter.

    Applies the chain rule from the loss back to the inputs, one layer at a time.
    Each gradient has the same shape as the parameter it belongs to.

    Args:
        params: The parameters used in the forward pass.
        cache: The cache returned by forward(params, X).
        y: Integer class labels, (N,).

    Returns:
        {"W1": (64, 32), "b1": (32,), "W2": (32, 10), "b2": (10,)} gradients.
    """
    X, Z1, H, Z2 = cache["X"], cache["Z1"], cache["H"], cache["Z2"]
    n, n_classes = Z2.shape

    # Output layer. For softmax + cross-entropy, dL/dZ2 = (P - Y) / N, where Y is
    # the one-hot encoding of y. The 1/N comes from taking the mean over the batch.
    Y = np.eye(n_classes)[y]  # (N, 10)
    dZ2 = (softmax(Z2) - Y) / n  # (N, 10)

    # Z2 = H @ W2 + b2.
    dW2 = H.T @ dZ2  # (32, N) @ (N, 10) -> (32, 10): sums each example's contribution
    db2 = dZ2.sum(axis=0)  # (10,): b2 is added to every row, so its gradient sums the rows
    dH = dZ2 @ params["W2"].T  # (N, 10) @ (10, 32) -> (N, 32): error sent back to H

    # H = ReLU(Z1). The derivative is 1 where Z1 > 0 and 0 where Z1 < 0. At Z1 == 0
    # it is undefined; we use 0 (a valid subgradient), which is also what Z1 > 0 gives.
    dZ1 = dH * (Z1 > 0)  # (N, 32), element-wise

    # Z1 = X @ W1 + b1.
    dW1 = X.T @ dZ1  # (64, N) @ (N, 32) -> (64, 32)
    db1 = dZ1.sum(axis=0)  # (32,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
