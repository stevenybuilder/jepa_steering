from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
from summarize_support_and_replication import bootstrap_mean, fit_support_metric, paired_distance_summary, physical_metrics


def test_frozen_coordinates_and_mahalanobis_units():
    reference = np.array([[0., 0.], [2., 0.], [0., 4.], [2., 4.]])
    mean, scale = np.array([1., 2.]), np.array([1., 2.])
    measure, shrinkage = fit_support_metric(reference, mean, scale)
    result = measure(np.array([[1., 2.], [2., 4.]]))
    np.testing.assert_allclose(result["nearest_reference_euclidean"], [np.sqrt(2), 0])
    np.testing.assert_allclose(result["shrinkage_mahalanobis"], [0, np.sqrt(2)])
    assert 0 <= shrinkage <= 1


def test_pairing_is_not_ratio_of_summary_medians():
    result = paired_distance_summary([1., 4.], [2., 4.])
    assert result["paired_after_over_before"]["mean"] == 1.5
    assert result["ratio_of_means"] == 1.2
    with pytest.raises(ValueError):
        paired_distance_summary([0.], [1.])


def test_physical_effect_sign_not_change_magnitude():
    result = physical_metrics([0, 0, 0], [0, 0, 1], [[.2, 0, .4]], [0, 0, .5], -1)
    assert result["signed_target_z_effect_m"] == pytest.approx(.1)
    assert result["goal_improvement_vs_unsteered_m"] < 0
    assert result["off_target_xy_effect_l2_m"] == pytest.approx(.2)
    assert result["observed_path_xyz_m"][0] == [0., 0., 0.]


def test_bootstrap_resamples_episodes_and_is_reproducible():
    result = bootstrap_mean([1, 2, 3, 4, 5])
    assert result == bootstrap_mean([1, 2, 3, 4, 5])
    assert result["N_episodes"] == 5
    assert result["mean"] == 3
    constant = bootstrap_mean([2] * 5)
    assert constant["descriptive_bootstrap_ci95"] == [2., 2.]
