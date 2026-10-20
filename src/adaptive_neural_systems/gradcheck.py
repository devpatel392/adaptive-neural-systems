"""Numerical gradient checking with central differences.

For a scalar function L(theta), the derivative with respect to one parameter can
be approximated by nudging that parameter up and down by a small eps:

    dL/dtheta  ~=  (L(theta + eps) - L(theta - eps)) / (2 * eps)

The Taylor expansion shows why the *central* difference is used. The even-order
terms cancel, so the error is O(eps^2) instead of O(eps) for a one-sided
difference. With eps = 1e-5 in float64 the truncation error is ~1e-10, small
compared with the gradients we check.

This is slow (two loss evaluations per parameter), so it is only for testing an
analytical gradient implementation, never for training.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Params = dict[str, np.ndarray]


def numerical_gradient(
    loss_fn: Callable[[Params], float], params: Params, eps: float = 1e-5
) -> Params:
    """Central-difference estimate of d loss_fn / d params, one entry at a time.

    Each parameter array is modified in place and restored before moving on, so
    params is unchanged when the function returns.

    Returns:
        Arrays with the same keys and shapes as params.
    """
    grads = {}
    for name, value in params.items():
        grad = np.zeros_like(value)
        # np.ndindex yields every index tuple of an array of any shape.
        for idx in np.ndindex(value.shape):
            original = value[idx]
            value[idx] = original + eps
            loss_plus = loss_fn(params)
            value[idx] = original - eps
            loss_minus = loss_fn(params)
            value[idx] = original
            grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        grads[name] = grad
    return grads


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Scale-free distance ||a - n|| / (||a|| + ||n||), in [0, 1].

    An absolute difference of 1e-7 means very different things for a gradient of
    size 1 and one of size 1e-8, so we divide by the gradients' size. Returns 0 if
    both gradients are exactly zero.
    """
    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / denominator)


def relu_kink_margin(Z1: np.ndarray) -> float:
    """Smallest |pre-activation| in the batch: distance to ReLU's kink at 0.

    ReLU is not differentiable at 0. If a perturbation of size eps moves some
    Z1[i, j] across 0, the finite difference averages two different slopes and no
    longer approximates the analytical (sub)gradient. A gradient check is only
    trustworthy when this margin exceeds the largest change one perturbation can
    cause in Z1 (see max_z1_change).
    """
    return float(np.abs(Z1).min())


def max_z1_change(X: np.ndarray, eps: float) -> float:
    """Largest change in any Z1 entry caused by nudging one W1 or b1 entry by eps.

    Z1[i, j] = sum_k X[i, k] W1[k, j] + b1[j]. Nudging W1[k, j] changes Z1[i, j]
    by eps * X[i, k]; nudging b1[j] changes it by eps. Nudging W2 or b2 does not
    touch Z1 at all.
    """
    return eps * max(1.0, float(np.abs(X).max()))
