import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"geometry_map"))
from action_path_geometry import central_estimates, interior_estimates, sampled_path_geometry


def test_cubic_uses_only_noncentral_samples_and_recovers_cubic():
    t = torch.tensor([-1., -.5, .5, 1.], dtype=torch.float64)
    x = torch.stack([2+3*t+4*t*t+5*t**3, -1+t*t], dim=1)
    result = central_estimates(x)
    torch.testing.assert_close(result["cubic_equal_data"], torch.tensor([2., -1.], dtype=torch.float64))
    assert not torch.allclose(result["linear_equal_data"], result["cubic_equal_data"])
    with pytest.raises(ValueError):
        central_estimates(torch.zeros(5, 2))


def test_constant_and_linear_interpolation():
    t = torch.tensor([-1., -.5, .5, 1.])
    x = torch.stack([1000+3*t, 400-2*t], dim=1)
    for estimate in central_estimates(x).values():
        torch.testing.assert_close(estimate, torch.tensor([1000., 400.]), rtol=0, atol=0)
    constant = torch.full((4, 2, 3), 12345.)
    for estimate in central_estimates(constant).values():
        assert torch.equal(estimate, constant[0])


def test_nonlinear_speed_along_straight_line_is_not_bending():
    t = torch.tensor([-1., -.5, 0., .5, 1.], dtype=torch.float64)
    x = torch.stack([t**3, 2*t**3], dim=1)
    result = sampled_path_geometry(x)
    assert result["max_orthogonal_fraction_of_chord"] < 1e-12
    assert result["path_chord_ratio"] == pytest.approx(1)
    assert max(result["normal_second_difference"]) < 1e-12
    assert result["backtracking_segments"] == 0


def test_parabola_bends_and_is_rigid_motion_invariant():
    t = torch.tensor([-1., -.5, 0., .5, 1.], dtype=torch.float64)
    x = torch.stack([t, t*t], dim=1)
    original = sampled_path_geometry(x)
    rotated = sampled_path_geometry(x@torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)+123.)
    assert original["path_chord_ratio"] > 1
    assert original["max_orthogonal_fraction_of_chord"] == pytest.approx(.5)
    assert original["finite_sample_path_curvature"][1] == pytest.approx(2)
    for key in ("path_length", "chord_length", "path_chord_ratio", "max_orthogonal_fraction_of_chord"):
        assert rotated[key] == pytest.approx(original[key])


def test_backtracking_and_degenerate_paths_not_false_curvature():
    x = torch.tensor([[0., 0.], [2., 0.], [1., 0.], [3., 0.], [4., 0.]])
    result = sampled_path_geometry(x)
    assert result["backtracking_segments"] == 1
    assert result["path_chord_ratio"] > 1
    assert result["max_orthogonal_fraction_of_chord"] == 0
    zero = sampled_path_geometry(torch.zeros(5, 3))
    assert zero["degenerate_endpoint_chord"]
    assert zero["path_chord_ratio"] is None
    with pytest.raises(ValueError):
        sampled_path_geometry(torch.full((5, 2), float("nan")))


def test_reparameterized_chord_separates_speed_from_off_line_geometry():
    t = torch.tensor([-1., -.5, .5, 1.], dtype=torch.float64)
    # Nonlinear parameterization of a straight line: cubic wins without bending.
    scalar = 1+t+.2*t*t
    line = torch.stack([scalar, 2*scalar], dim=1)
    result = central_estimates(line)
    torch.testing.assert_close(result["cubic_equal_data"], result["reparameterized_chord"])
    assert not torch.allclose(result["endpoint_chord"], result["cubic_equal_data"])
    parabola = torch.stack([t, t*t], dim=1)
    result = central_estimates(parabola)
    torch.testing.assert_close(result["cubic_equal_data"], torch.tensor([0., 0.], dtype=torch.float64))
    torch.testing.assert_close(result["reparameterized_chord"], torch.tensor([0., 1.], dtype=torch.float64))
    torch.testing.assert_close(result["reflected_curvature"], torch.tensor([0., 2.], dtype=torch.float64))


def test_unseen_interiors_use_frozen_cubic_and_equal_data_affine():
    donor_t = torch.tensor([-1., -.5, .5, 1.], dtype=torch.float64)
    donors = torch.stack([2+3*donor_t+4*donor_t**2+5*donor_t**3, -1+donor_t**2], dim=1)
    for t in (-.25, .25):
        result = interior_estimates(donors, t)
        expected = torch.tensor([2+3*t+4*t*t+5*t**3, -1+t*t], dtype=torch.float64)
        torch.testing.assert_close(result["cubic_equal_data"], expected)
        design = torch.stack([torch.ones_like(donor_t), donor_t], dim=1)
        fitted = torch.linalg.lstsq(design, donors).solution
        torch.testing.assert_close(result["linear_equal_data"], fitted[0]+t*fitted[1])
    for name, expected in central_estimates(donors).items():
        assert torch.equal(interior_estimates(donors, 0)[name], expected)
    with pytest.raises(ValueError):
        interior_estimates(donors, .75)
