import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from decompose_actual_context_paths import physical_spaces, compact_geometry


def test_physical_units_and_angle_wrap():
    states = torch.zeros(5, 7, 7, dtype=torch.float64)
    states[..., 0] = 100; states[..., 2] = 200
    states[1, :, 4] = 2*torch.pi
    spaces = physical_spaces(states)
    assert spaces['agent_xy_pixels'][0, 0, 0] == 100
    assert spaces['block_xy_pixels'][0, 0, 0] == 200
    assert torch.allclose(spaces['block_angle_unit_circle'][0], spaces['block_angle_unit_circle'][1], atol=1e-14)


def test_variable_speed_straight_path_is_not_bending():
    points = torch.tensor([[0., 0.], [.1, 0.], [.4, 0.], [.8, 0.], [1., 0.]])
    geometry = compact_geometry(points, 'pixels')
    assert geometry['max_orthogonal_fraction_of_chord'] == 0
    assert geometry['path_chord_ratio'] == 1
    assert compact_geometry(torch.ones(5, 2), 'pixels')['degenerate_endpoint_chord']
