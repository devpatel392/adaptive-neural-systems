import random

import pytest

import adaptive_neural_systems
from adaptive_neural_systems.reproducibility import set_seed


def test_package_exposes_version():
    assert adaptive_neural_systems.__version__ == "0.0.1"


def test_same_seed_gives_same_python_sequence():
    set_seed(42)
    first = [random.random() for _ in range(5)]
    set_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_different_seeds_give_different_sequences():
    set_seed(1)
    first = [random.random() for _ in range(5)]
    set_seed(2)
    second = [random.random() for _ in range(5)]
    assert first != second


def test_reports_seeded_generators():
    assert set_seed(0)[0] == "python"


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "42", True, None])
def test_rejects_invalid_seed(bad_seed):
    with pytest.raises(ValueError):
        set_seed(bad_seed)


def test_seeds_numpy_when_installed():
    np = pytest.importorskip("numpy")
    set_seed(7)
    first = np.random.rand(3)
    set_seed(7)
    second = np.random.rand(3)
    assert (first == second).all()
