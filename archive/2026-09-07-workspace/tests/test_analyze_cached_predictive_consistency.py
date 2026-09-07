import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from analyze_cached_predictive_consistency import average_ranks, rank_correlation, path_metrics


def test_tied_ranks_and_constant_control():
    assert torch.equal(average_ranks([2., 1., 1., 4.]), torch.tensor([2., .5, .5, 3.], dtype=torch.float64))
    assert rank_correlation([1, 1], [1, 2]) is None
    assert abs(rank_correlation([1, 2, 3], [3, 2, 1])+1) < 1e-14


def test_common_context_shift_is_not_action_effect_change():
    native = torch.arange(5.).reshape(5, 1, 1, 1, 1).expand(5, 1, 16, 16, 384)
    changed = native+2; goal = torch.zeros(1, 16, 16, 384)
    result = path_metrics(native, changed, native, goal)
    assert result['paired_same_action_visual_mse'] == 4
    assert result['paired_action_effect_visual_mse'] == 0
    assert result['latent_spread_mse']['native'] == result['latent_spread_mse']['actual_past'] == 2
