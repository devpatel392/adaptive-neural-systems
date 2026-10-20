import numpy as np
import pytest

from adaptive_neural_systems.data import load_digits_splits
from adaptive_neural_systems.mlp import forward, init_params, relu, softmax


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
