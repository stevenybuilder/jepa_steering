import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from join_native_hybrid_references import horizon_mse


def test_spatial_mse_not_meanpooled_error():
    x = torch.zeros(6, 1, 16, 16, 384)
    y = x.clone(); y[:, :, :8] = 1; y[:, :, 8:] = -1
    assert torch.equal(horizon_mse(x, y), torch.ones(6))
    assert y.mean() == 0


def test_future_axis_is_preserved():
    x = torch.zeros(6, 1, 16, 16, 384); y = x.clone()
    y[5] = 2
    assert torch.equal(horizon_mse(x, y), torch.tensor([0., 0., 0., 0., 0., 4.]))
