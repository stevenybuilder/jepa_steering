from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import public_panel_sonar_math as sm  # noqa: E402


def test_mixture_energy_gradient_matches_finite_difference() -> None:
    weights = np.array([0.4, 0.6])
    means = np.array([[-1.0, 0.2], [1.5, -0.3]])
    covariances = np.array([[[0.7, 0.1], [0.1, 1.2]], [[1.1, -0.2], [-0.2, 0.6]]])
    x = np.array([[0.3, -0.8]])
    energy, gradient, responsibilities = sm.mixture_energy_and_gradient(x, weights, means, covariances)
    numeric = np.zeros(2)
    epsilon = 1e-6
    for axis in range(2):
        offset = np.zeros_like(x)
        offset[0, axis] = epsilon
        plus = sm.mixture_energy_and_gradient(x + offset, weights, means, covariances)[0]
        minus = sm.mixture_energy_and_gradient(x - offset, weights, means, covariances)[0]
        numeric[axis] = float(((plus - minus) / (2 * epsilon))[0])
    assert np.allclose(gradient[0], numeric, atol=1e-5)
    assert np.allclose(responsibilities.sum(1), 1.0)
    assert np.isfinite(energy).all()


def test_trust_region_step_hits_metric_boundary_and_descends() -> None:
    gradient = np.array([[1.0, -2.0]])
    metric = np.array([[3.0, 0.4], [0.4, 1.5]])
    radius = 0.25
    delta = sm.trust_region_step(gradient, metric, radius)[0]
    assert np.isclose(delta @ metric @ delta, radius**2)
    assert gradient[0] @ delta < 0


def test_capped_metric_step_preserves_small_gradient_magnitude() -> None:
    metric = np.diag([4.0, 1.0])
    small = sm.capped_metric_step([[0.1, 0.0]], metric, step_size=1.0, radius=1.0)
    large = sm.capped_metric_step([[10.0, 0.0]], metric, step_size=1.0, radius=1.0)
    np.testing.assert_allclose(small, [[-0.025, 0.0]])
    assert np.isclose(float(np.sqrt(large[0] @ metric @ large[0])), 1.0)


def test_angular_radial_decomposition_is_exact_and_orthogonal() -> None:
    x = np.array([[3.0, 4.0]])
    delta = np.array([[2.0, -1.0]])
    parts = sm.angular_radial_parts(x, delta)
    assert np.allclose(parts["radial"] + parts["angular"], delta)
    assert abs(float((parts["angular"] @ x[0])[0])) < 1e-12


def test_gaussian_ot_maps_mean_and_covariance() -> None:
    source_mean = np.array([1.0, -2.0])
    target_mean = np.array([-1.0, 0.5])
    source_covariance = np.array([[2.0, 0.4], [0.4, 0.7]])
    target_covariance = np.array([[0.8, -0.2], [-0.2, 1.4]])
    affine, offset = sm.gaussian_ot_affine(source_mean, source_covariance, target_mean, target_covariance)
    assert np.allclose(affine @ source_mean + offset, target_mean)
    assert np.allclose(affine @ source_covariance @ affine.T, target_covariance, atol=1e-8)


def test_soft_conceptor_and_matched_spectrum_control() -> None:
    covariance = np.diag([9.0, 1.0, 0.0])
    operator = sm.conceptor(covariance, aperture=1.0)
    assert np.allclose(np.linalg.eigvalsh(operator), [0.0, 0.5, 0.9])
    sham = sm.matched_spectrum(operator, seed=7)
    assert np.allclose(np.linalg.eigvalsh(sham), np.linalg.eigvalsh(operator))
    assert not np.allclose(sham, operator)


def test_singular_conceptor_and_not_restricts_to_range_intersection() -> None:
    success = np.diag([0.8, 0.0])
    failure = np.diag([0.2, 0.1])
    got = sm.conceptor_and_not(success, failure)
    expected = np.diag([1.0 / (1.0 / 0.8 + 1.0 / 0.8 - 1.0), 0.0])
    np.testing.assert_allclose(got, expected)
    values = np.linalg.eigvalsh(got)
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_conceptor_and_not_handles_hard_projector_boundaries() -> None:
    first_axis = np.diag([1.0, 0.0, 0.0])
    second_axis = np.diag([0.0, 1.0, 0.0])
    zero = np.zeros((3, 3))
    np.testing.assert_allclose(sm.conceptor_and_not(first_axis, zero), first_axis)
    np.testing.assert_allclose(sm.conceptor_and_not(first_axis, first_axis), zero)
    np.testing.assert_allclose(sm.conceptor_and_not(first_axis, second_axis), first_axis)


def test_full_rank_conceptor_and_not_matches_direct_formula() -> None:
    rng = np.random.default_rng(19)
    success_basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    failure_basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    success = (success_basis * np.array([0.8, 0.7, 0.5, 0.3, 0.1])) @ success_basis.T
    failure = (failure_basis * np.array([0.75, 0.6, 0.4, 0.2, 0.05])) @ failure_basis.T
    expected = np.linalg.inv(
        np.linalg.inv(success) + np.linalg.inv(np.eye(5) - failure) - np.eye(5)
    )
    np.testing.assert_allclose(sm.conceptor_and_not(success, failure), expected, atol=1e-11)


def test_contrastive_conceptor_separate_centers_and_translation_invariance() -> None:
    rng = np.random.default_rng(81)
    success = rng.normal(size=(80, 4)) * [3.0, 0.2, 0.2, 0.2] + 100.0
    failure = rng.normal(size=(80, 4)) * [0.2, 3.0, 0.2, 0.2] - 100.0
    fit = sm.contrastive_conceptor(success, failure, aperture=1.0)
    np.testing.assert_allclose(fit["success_center"], success.mean(0))
    np.testing.assert_allclose(fit["failure_center"], failure.mean(0))
    values = np.linalg.eigvalsh(fit["contrastive"])
    assert values.min() >= -1e-6 and values.max() <= 1.0 + 1e-6
    shifted = sm.contrastive_conceptor(success + 7.0, failure - 13.0, aperture=1.0)
    np.testing.assert_allclose(fit["contrastive"], shifted["contrastive"], atol=1e-10)


def test_generalized_eigen_contrast_is_scale_normalized() -> None:
    success = np.diag([8.0, 2.0, 1.0])
    failure = np.diag([2.0, 2.0, 4.0])
    values, vectors = sm.generalized_eigen_contrast(success, failure)
    assert np.allclose(values, [4.0, 1.0, 0.25])
    assert np.argmax(np.abs(vectors[:, 0])) == 0


def test_weighted_shrinkage_reports_fractional_effective_n() -> None:
    rows = np.arange(20, dtype=float).reshape(10, 2)
    weights = np.array([0.9] + [0.1 / 9] * 9)
    _mean, covariance, effective_n = sm.fit_shrinkage_gaussian(rows, weights=weights)
    assert 1.0 < effective_n < 2.0
    assert np.linalg.eigvalsh(covariance).min() > 0
