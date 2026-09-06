#!/usr/bin/env python3
"""Numerical core for the performance-first JEPA-WM Sonar intervention.

The functions here implement only the operators that survived the paper review:

* shrinkage Gaussian outcome/regime energies;
* minimum-metric-norm trust-region descent on the proper mixture energy contrast;
* Gaussian optimal transport as a separately named comparator;
* angular/radial displacement ablations;
* generalized-eigen and soft-conceptor diagnostics with matched-spectrum controls.
* paper-style contrastive conceptors with separate success/failure centering.

All fitting is offline NumPy.  Runtime torch code consumes the serialized arrays; it
must not refit or select an operator on evaluation outcomes.
"""

from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return (matrix + matrix.T) / 2.0


def psd_eigh(matrix: np.ndarray, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    return np.maximum(values, floor), vectors


def psd_power(matrix: np.ndarray, power: float, floor: float = 1e-6) -> np.ndarray:
    values, vectors = psd_eigh(matrix, floor=floor)
    return (vectors * np.power(values, power)) @ vectors.T


def fit_shrinkage_gaussian(
    rows: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    shrinkage: float = 0.2,
    floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a weighted Gaussian and shrink covariance toward its isotropic scale."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("rows must have shape [N,D] with N>=2")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0,1]")
    w = np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(x),) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive mass")
    w = w / w.sum()
    mean = w @ x
    centered = x - mean
    covariance = (centered * w[:, None]).T @ centered
    # Kish effective sample size is logged because fractional regime assignments can
    # otherwise make an apparently large cell nearly unsupported.
    effective_n = float(1.0 / np.square(w).sum())
    scale = max(float(np.trace(covariance) / x.shape[1]), floor)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(x.shape[1])
    values, vectors = psd_eigh(covariance, floor=floor)
    covariance = (vectors * values) @ vectors.T
    return mean, covariance, effective_n


def log_gaussian(rows: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    mean = np.asarray(mean, dtype=np.float64)
    precision = np.linalg.inv(symmetrize(covariance))
    sign, logdet = np.linalg.slogdet(symmetrize(covariance))
    if sign <= 0:
        raise ValueError("covariance is not positive definite")
    centered = x - mean
    quadratic = np.einsum("ni,ij,nj->n", centered, precision, centered)
    return -0.5 * (x.shape[1] * np.log(2.0 * np.pi) + logdet + quadratic)


def mixture_responsibilities(
    rows: np.ndarray,
    mixture_weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(mixture_weights, dtype=np.float64)
    log_joint = np.stack(
        [np.log(weights[k] + 1e-300) + log_gaussian(rows, means[k], covariances[k]) for k in range(len(weights))],
        axis=1,
    )
    maximum = log_joint.max(axis=1, keepdims=True)
    normalizer = maximum + np.log(np.exp(log_joint - maximum).sum(axis=1, keepdims=True) + 1e-300)
    return np.exp(log_joint - normalizer)


def mixture_energy_and_gradient(
    rows: np.ndarray,
    mixture_weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``-log sum_z pi_z N(x;mu_z,Sigma_z)`` and its exact gradient."""
    x = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    responsibility = mixture_responsibilities(x, mixture_weights, means, covariances)
    component_log = np.stack(
        [
            np.log(np.asarray(mixture_weights)[k] + 1e-300)
            + log_gaussian(x, np.asarray(means)[k], np.asarray(covariances)[k])
            for k in range(len(mixture_weights))
        ],
        axis=1,
    )
    maximum = component_log.max(axis=1)
    energy = -(maximum + np.log(np.exp(component_log - maximum[:, None]).sum(axis=1) + 1e-300))
    gradients = np.zeros_like(x)
    for k in range(len(mixture_weights)):
        precision = np.linalg.inv(symmetrize(covariances[k]))
        gradients += responsibility[:, k, None] * ((x - means[k]) @ precision)
    return energy, gradients, responsibility


def outcome_energy_contrast_gradient(
    rows: np.ndarray,
    success: dict[str, np.ndarray],
    failure: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Gradient of ``E_success - E_failure``; descending favors success density."""
    es, gs, rs = mixture_energy_and_gradient(
        rows, success["weights"], success["means"], success["covariances"]
    )
    ef, gf, rf = mixture_energy_and_gradient(
        rows, failure["weights"], failure["means"], failure["covariances"]
    )
    return es - ef, gs - gf, {"success": rs, "failure": rf}


def trust_region_step(gradient: np.ndarray, metric: np.ndarray, radius: float) -> np.ndarray:
    """Solve min ``g^T delta`` subject to ``delta^T G delta <= radius^2``."""
    g = np.atleast_2d(np.asarray(gradient, dtype=np.float64))
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    inverse_times_gradient = np.linalg.solve(symmetrize(metric), g.T).T
    denominator = np.sqrt(np.maximum(np.einsum("ni,ni->n", g, inverse_times_gradient), 1e-30))
    return -float(radius) * inverse_times_gradient / denominator[:, None]


def capped_metric_step(
    gradient: np.ndarray,
    metric: np.ndarray,
    *,
    step_size: float,
    radius: float,
) -> np.ndarray:
    """Metric-quadratic descent with a cap that preserves gradient magnitude.

    This solves ``min g.T d + (2*step_size)^-1 d.T G d`` subject to
    ``d.T G d <= radius^2``. Unlike the boundary-only linear objective, small
    gradients remain strictly inside the trust region.
    """
    g = np.atleast_2d(np.asarray(gradient, dtype=np.float64))
    matrix = np.asarray(metric, dtype=np.float64)
    if step_size < 0 or radius < 0:
        raise ValueError("step size and radius must be nonnegative")
    if matrix.ndim == 2:
        symmetric = symmetrize(matrix)
        inverse_times_gradient = np.linalg.solve(symmetric, g.T).T
        metric_batch = np.broadcast_to(symmetric, (len(g),) + symmetric.shape)
    elif matrix.ndim == 3 and matrix.shape[0] == len(g):
        metric_batch = np.stack([symmetrize(value) for value in matrix])
        inverse_times_gradient = np.linalg.solve(metric_batch, g[..., None])[..., 0]
    else:
        raise ValueError("metric must have shape [D,D] or [N,D,D]")
    unconstrained = -float(step_size) * inverse_times_gradient
    norm = np.sqrt(
        np.maximum(np.einsum("ni,nij,nj->n", unconstrained, metric_batch, unconstrained), 0.0)
    )
    scale = np.minimum(1.0, float(radius) / np.maximum(norm, 1e-30))
    return unconstrained * scale[:, None]


def angular_radial_parts(rows: np.ndarray, displacement: np.ndarray, floor: float = 1e-8) -> dict[str, np.ndarray]:
    x = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    delta = np.atleast_2d(np.asarray(displacement, dtype=np.float64))
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), floor)
    radial = np.einsum("ni,ni->n", unit, delta)[:, None] * unit
    return {"radial": radial, "angular": delta - radial, "joint": delta}


def gaussian_ot_affine(
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    *,
    floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Bures/Wasserstein affine map ``target_mean + A(x-source_mean)``."""
    source_sqrt = psd_power(source_covariance, 0.5, floor=floor)
    source_inv_sqrt = psd_power(source_covariance, -0.5, floor=floor)
    middle = psd_power(source_sqrt @ target_covariance @ source_sqrt, 0.5, floor=floor)
    affine = source_inv_sqrt @ middle @ source_inv_sqrt
    offset = np.asarray(target_mean, dtype=np.float64) - affine @ np.asarray(source_mean, dtype=np.float64)
    return symmetrize(affine), offset


def apply_affine_transport(rows: np.ndarray, affine: np.ndarray, offset: np.ndarray, dose: float = 1.0) -> np.ndarray:
    x = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    transported = x @ np.asarray(affine, dtype=np.float64).T + np.asarray(offset, dtype=np.float64)
    return (1.0 - dose) * x + dose * transported


def generalized_eigen_contrast(
    success_covariance: np.ndarray,
    failure_covariance: np.ndarray,
    *,
    floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``Sigma_s v = lambda Sigma_f v`` via symmetric whitening."""
    failure_inv_sqrt = psd_power(failure_covariance, -0.5, floor=floor)
    whitened = symmetrize(failure_inv_sqrt @ success_covariance @ failure_inv_sqrt)
    values, vectors_white = np.linalg.eigh(whitened)
    order = np.argsort(values)[::-1]
    vectors = failure_inv_sqrt @ vectors_white[:, order]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=0, keepdims=True), 1e-12)
    return values[order], vectors


def conceptor(covariance: np.ndarray, aperture: float) -> np.ndarray:
    """Soft PCA/conceptor resolvent ``R(R+alpha^-2 I)^-1``."""
    if aperture <= 0:
        raise ValueError("aperture must be positive")
    values, vectors = psd_eigh(covariance, floor=0.0)
    weights = values / (values + aperture ** -2)
    return (vectors * weights) @ vectors.T


def weighted_covariance(
    rows: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return a population covariance and its separately fitted class center."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("rows must be a nonempty finite [N,D] array")
    if weights is None:
        w = np.full(len(x), 1.0 / len(x), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(x),) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("weights must be finite, nonnegative, and have positive mass")
        w = w / w.sum()
    center = w @ x
    centered = x - center
    covariance = (centered * w[:, None]).T @ centered
    return center, symmetrize(covariance)


def _symmetric_range_basis(matrix: np.ndarray, rcond: float) -> np.ndarray:
    """Return an orthonormal basis for a symmetric matrix's numerical range."""
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    return vectors[:, np.abs(values) > rcond * scale]


def _range_intersection_basis(
    left: np.ndarray, right: np.ndarray, rcond: float
) -> np.ndarray:
    """Return a basis for ``range(left) intersect range(right)``.

    Singular values equal to one in ``U_left.T @ U_right`` identify the exact
    common subspace.  The tolerance only resolves floating-point rank; it does
    not turn nearby but distinct subspaces into an intersection.
    """
    left_basis = _symmetric_range_basis(left, rcond)
    right_basis = _symmetric_range_basis(right, rcond)
    if left_basis.shape[1] == 0 or right_basis.shape[1] == 0:
        return np.zeros((left.shape[0], 0), dtype=np.float64)
    left_vectors, singular_values, _ = np.linalg.svd(
        left_basis.T @ right_basis, full_matrices=False
    )
    tolerance = max(rcond, np.finfo(np.float64).eps * max(left.shape) * 10.0)
    common = singular_values >= 1.0 - tolerance
    if not np.any(common):
        return np.zeros((left.shape[0], 0), dtype=np.float64)
    basis, _ = np.linalg.qr(left_basis @ left_vectors[:, common], mode="reduced")
    return basis


def conceptor_and_not(success: np.ndarray, failure: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Generalized Boolean ``success AND NOT failure`` for singular conceptors.

    Jaeger's generalized AND first restricts the inverse expression to the
    intersection of the operands' ranges.  Applying a Moore--Penrose inverse
    directly in the ambient space is not equivalent for singular conceptors
    and can create eigenvalues above one.

    No jitter or substantive spectrum clipping is hidden in this operation.
    Invalid spectra still fail the caller's fit validation.
    """
    cs, cf = symmetrize(success), symmetrize(failure)
    if cs.shape != cf.shape or cs.ndim != 2 or cs.shape[0] != cs.shape[1] or rcond <= 0:
        raise ValueError("conceptors must be same-width square matrices and rcond must be positive")
    identity = np.eye(cs.shape[0], dtype=np.float64)
    not_failure = symmetrize(identity - cf)
    basis = _range_intersection_basis(cs, not_failure, rcond)
    if basis.shape[1] == 0:
        return np.zeros_like(cs, dtype=np.float64)
    core = (
        np.linalg.pinv(cs, rcond=rcond, hermitian=True)
        + np.linalg.pinv(not_failure, rcond=rcond, hermitian=True)
        - identity
    )
    reduced = symmetrize(basis.T @ core @ basis)
    result = basis @ np.linalg.inv(reduced) @ basis.T
    return symmetrize(result)


def contrastive_conceptor(
    success_rows: np.ndarray,
    failure_rows: np.ndarray,
    aperture: float,
    *,
    success_weights: np.ndarray | None = None,
    failure_weights: np.ndarray | None = None,
    rcond: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Fit COAST's contrastive operator with separate class centering."""
    success_center, success_covariance = weighted_covariance(success_rows, success_weights)
    failure_center, failure_covariance = weighted_covariance(failure_rows, failure_weights)
    success = conceptor(success_covariance, aperture)
    failure = conceptor(failure_covariance, aperture)
    contrastive = conceptor_and_not(success, failure, rcond=rcond)
    values = np.linalg.eigvalsh(contrastive)
    if values[0] < -1e-6 or values[-1] > 1.0 + 1e-6:
        raise ValueError(
            f"contrastive conceptor spectrum outside [0,1]: [{values[0]}, {values[-1]}]"
        )
    return {
        "success": success,
        "failure": failure,
        "contrastive": contrastive,
        "success_center": success_center,
        "failure_center": failure_center,
    }


def matched_spectrum(matrix: np.ndarray, seed: int) -> np.ndarray:
    values = np.linalg.eigvalsh(symmetrize(matrix))
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=matrix.shape))
    return (q * values) @ q.T
