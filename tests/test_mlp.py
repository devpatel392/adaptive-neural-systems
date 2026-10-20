import numpy as np
import pytest

from adaptive_neural_systems.data import load_digits_splits
from adaptive_neural_systems.gradcheck import (
    max_z1_change,
    numerical_gradient,
    relative_error,
    relu_kink_margin,
)
from adaptive_neural_systems.mlp import (
    backward,
    cross_entropy,
    forward,
    init_params,
    loss,
    relu,
    softmax,
)


@pytest.fixture
def params():
    return init_params(np.random.default_rng(0))


# --- init_params ---------------------------------------------------------------


def test_param_shapes(params):
    assert params["W1"].shape == (64, 32)
    assert params["b1"].shape == (32,)
    assert params["W2"].shape == (32, 10)
    assert params["b2"].shape == (10,)
    assert sum(p.size for p in params.values()) == 2410


def test_biases_start_at_zero(params):
    assert not params["b1"].any() and not params["b2"].any()


def test_he_initialisation_scale():
    # With many samples the empirical std should be close to sqrt(2 / fan_in).
    p = init_params(np.random.default_rng(1), n_inputs=400, n_hidden=500, n_outputs=300)
    assert p["W1"].std() == pytest.approx(np.sqrt(2 / 400), rel=0.02)
    assert p["W2"].std() == pytest.approx(np.sqrt(2 / 500), rel=0.02)
    assert abs(p["W1"].mean()) < 0.002


def test_init_is_deterministic_per_seed():
    a, b = init_params(np.random.default_rng(5)), init_params(np.random.default_rng(5))
    c = init_params(np.random.default_rng(6))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["W1"], c["W1"])


# --- relu and softmax ----------------------------------------------------------


def test_relu():
    np.testing.assert_array_equal(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])


def test_softmax_known_value():
    # exp(0) : exp(ln 3) = 1 : 3  ->  [0.25, 0.75]
    np.testing.assert_allclose(softmax(np.array([[0.0, np.log(3.0)]])), [[0.25, 0.75]])


def test_softmax_is_invariant_to_adding_a_constant_per_row():
    Z = np.random.default_rng(2).normal(size=(4, 10))
    np.testing.assert_allclose(softmax(Z), softmax(Z + 123.0))


def test_softmax_is_stable_for_huge_logits():
    P = softmax(np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]]))
    assert np.isfinite(P).all()
    np.testing.assert_allclose(P[0], softmax(np.array([[0.0, 1.0, 2.0]]))[0])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


# --- forward -------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7])
def test_forward_shapes(params, n):
    X = np.random.default_rng(3).random((n, 64))
    P, cache = forward(params, X)
    assert P.shape == (n, 10)
    assert cache["X"].shape == (n, 64)
    assert cache["Z1"].shape == (n, 32)
    assert cache["H"].shape == (n, 32)
    assert cache["Z2"].shape == (n, 10)


def test_forward_outputs_are_valid_probabilities_on_real_data(params):
    X = load_digits_splits(split_seed=0).X_train
    P, _ = forward(params, X)
    assert np.isfinite(P).all()
    assert (P >= 0).all() and (P <= 1).all()
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_cache_is_consistent_with_the_equations(params):
    X = np.random.default_rng(4).random((5, 64))
    P, c = forward(params, X)
    np.testing.assert_allclose(c["Z1"], X @ params["W1"] + params["b1"])
    np.testing.assert_array_equal(c["H"], np.maximum(0.0, c["Z1"]))
    np.testing.assert_allclose(c["Z2"], c["H"] @ params["W2"] + params["b2"])
    np.testing.assert_allclose(P, softmax(c["Z2"]))


def test_forward_rejects_wrong_feature_count(params):
    with pytest.raises(ValueError):
        forward(params, np.zeros((3, 63)))


# --- cross-entropy -------------------------------------------------------------


def test_cross_entropy_matches_naive_formula_on_moderate_logits():
    Z = np.random.default_rng(5).normal(size=(6, 10))
    y = np.array([0, 3, 9, 1, 1, 7])
    naive = -np.log(softmax(Z)[np.arange(6), y]).mean()
    assert cross_entropy(Z, y) == pytest.approx(naive)


def test_cross_entropy_of_uniform_logits_is_ln_num_classes():
    # Equal logits give p = 1/10 exactly, so this value is exact, not an assumption
    # about random initialisation.
    assert cross_entropy(np.zeros((4, 10)), np.array([0, 1, 2, 3])) == pytest.approx(np.log(10))


def test_cross_entropy_is_finite_when_naive_log_would_be_infinite():
    # The correct class is 1000 below the others: p underflows to 0, log p = -inf.
    Z = np.array([[0.0, 1000.0, 1000.0]])
    y = np.array([0])
    with np.errstate(divide="ignore"):
        assert np.isinf(np.log(softmax(Z)[0, 0]))
    assert cross_entropy(Z, y) == pytest.approx(1000.0 + np.log(2.0))


def test_cross_entropy_near_zero_for_confident_correct_predictions():
    Z = np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 50.0]])
    assert cross_entropy(Z, np.array([0, 2])) < 1e-20


def test_cross_entropy_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        cross_entropy(np.zeros((3, 10)), np.array([0, 1]))


# --- backward ------------------------------------------------------------------


@pytest.fixture(scope="module")
def batch():
    s = load_digits_splits(split_seed=0)
    return s.X_train[:32], s.y_train[:32]


def test_gradient_shapes_and_finiteness(params, batch):
    X, y = batch
    _, cache = forward(params, X)
    grads = backward(params, cache, y)
    assert grads.keys() == params.keys()
    for name in params:
        assert grads[name].shape == params[name].shape
        assert np.isfinite(grads[name]).all()


def test_output_bias_gradient_sums_to_zero(params, batch):
    # Each row of P - Y sums to 1 - 1 = 0, so the column sums (db2) also sum to 0.
    X, y = batch
    _, cache = forward(params, X)
    assert backward(params, cache, y)["b2"].sum() == pytest.approx(0.0, abs=1e-12)


def test_gradients_are_zero_for_always_blank_pixels(params, batch):
    # dW1 = X^T dZ1: a pixel that is 0 in every image contributes nothing.
    X, y = batch
    _, cache = forward(params, X)
    dW1 = backward(params, cache, y)["W1"]
    blank = ~X.any(axis=0)
    assert blank.any()
    assert not dW1[blank].any()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_matches_numerical_gradient(seed, batch):
    X, y = batch
    params = init_params(np.random.default_rng(seed))
    eps = 1e-5
    _, cache = forward(params, X)
    # The check is only valid if no single nudge can push a Z1 entry across 0.
    assert relu_kink_margin(cache["Z1"]) > max_z1_change(X, eps)

    analytic = backward(params, cache, y)
    numeric = numerical_gradient(lambda p: loss(p, X, y), params, eps)
    for name in params:
        assert relative_error(analytic[name], numeric[name]) < 1e-6, name


def test_gradient_check_breaks_when_a_unit_sits_on_the_relu_kink(params, batch):
    # Place hidden unit 0 exactly at Z1 = 0 for the first example by choosing its
    # bias. The finite difference then averages the slopes on both sides of the
    # kink, so it disagrees with the analytical subgradient. The margin test
    # detects this case in advance.
    X, y = batch[0][:1], batch[1][:1]
    params = {k: v.copy() for k, v in params.items()}
    params["b1"][0] = -(X[0] @ params["W1"][:, 0])
    eps = 1e-5
    _, cache = forward(params, X)
    assert relu_kink_margin(cache["Z1"]) <= max_z1_change(X, eps)

    analytic = backward(params, cache, y)["b1"][0]
    numeric = numerical_gradient(lambda p: loss(p, X, y), params, eps)["b1"][0]
    # Gradient if the unit were active: dL/db1[0] = (P - Y) @ W2[0, :].
    slope_if_active = ((forward(params, X)[0] - np.eye(10)[y]) @ params["W2"][0])[0]
    # Rounding leaves Z1 at exactly 0 or a hair either side, so the analytical
    # gradient takes one side's slope: 0 (inactive) or slope_if_active (active).
    assert analytic == 0.0 or analytic == pytest.approx(slope_if_active)
    # The central difference straddles the kink and averages the two slopes.
    assert numeric == pytest.approx(0.5 * slope_if_active, rel=1e-4)
    assert abs(numeric - analytic) > 0.25 * abs(slope_if_active)
