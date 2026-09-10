import sys
from pathlib import Path

import numpy as np
import unittest
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/geometry_map"))
from screen_nonlinear_interfaces import eligible, fit_readouts, predict, rbf_jacobian, score


def test_splits_exclude_confirmation():
    assert eligible({"seed": 1, "episode": 49}, "discovery")
    assert not eligible({"seed": 1, "episode": 50}, "discovery")
    assert eligible({"seed": 3, "episode": 74}, "validation")
    assert not eligible({"seed": 3, "episode": 75}, "validation")
    with unittest.TestCase().assertRaises(ValueError):
        eligible({"seed": 1, "episode": 90}, "confirmation")


def test_rbf_gradient_finite_difference():
    rng = np.random.default_rng(4)
    z, train, dual = rng.normal(size=(3, 5)), rng.normal(size=(17, 5)), rng.normal(size=(17, 2))
    actual = rbf_jacobian(z, train, dual, .2)
    for column in range(5):
        delta = np.zeros_like(z)
        delta[:, column] = 1e-5
        expected = ((rbf_kernel(z + delta, train, gamma=.2) -
                     rbf_kernel(z - delta, train, gamma=.2)) @ dual) / 2e-5
        np.testing.assert_allclose(actual[:, column], expected, atol=1e-8)


def test_prediction_does_not_refit_preprocessing():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(60, 8)), rng.normal(size=(60, 4))
    bundle = fit_readouts(x, y, components=5)
    mean = bundle["mean"].copy()
    before = predict(bundle, x[:2])
    predict(bundle, x[:2] + 100)
    np.testing.assert_array_equal(bundle["mean"], mean)
    for name, value in predict(bundle, x[:2]).items():
        np.testing.assert_array_equal(value, before[name])


def test_circular_metric_wraps():
    angles = np.deg2rad([179, -179, 90, -90])
    y = np.column_stack((np.arange(4), np.arange(4), np.cos(angles), np.sin(angles)))
    pred = y.copy()
    pred[:2, 3] *= -1
    result = score(y, pred, np.ones(4, dtype=bool))
    np.testing.assert_allclose(result["direction_angular_mae_degrees"], 1)


def test_nonfinite_test_values_fail_instead_of_refit():
    rng = np.random.default_rng(5)
    bundle = fit_readouts(rng.normal(size=(50, 8)), rng.normal(size=(50, 4)), components=4)
    with unittest.TestCase().assertRaises(ValueError):
        predict(bundle, np.full((1, 8), np.nan))


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in list(globals().items())
                               if name.startswith("test_") and callable(fn))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
