import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from capture_actual_context_curvature import allowed_history_indices,ActualPastInputs


def test_history_excludes_target_and_has_native_window():
    assert [allowed_history_indices(h) for h in range(1,7)]==[[0],[0,1],[1,2],[2,3],[3,4],[4,5]]
    for h in range(1,7):assert max(allowed_history_indices(h))<h
    with pytest.raises(ValueError):allowed_history_indices(7)


def test_actual_past_only_actions_preserved_and_h1_identity():
    class Predictor(torch.nn.Module):
        def forward(self,v,a,p):return v,a,p
    mod=Predictor();actual_v=torch.arange(7.)[None,:,None,None].expand(7,7,2,3).clone()
    actual_p=actual_v.clone();zero_v=torch.zeros(7,2,2,3);action=torch.randn(7,2,10)
    with ActualPastInputs(mod,actual_v,actual_p) as hook:
        v,a,p=mod(zero_v[:,:1],action[:,:1],zero_v[:,:1]);assert torch.equal(v,zero_v[:,:1])
        for h in range(2,7):
            v,a,p=mod(zero_v,action,zero_v)
            assert a is action
            assert torch.equal(v,actual_v[:,h-2:h])
            assert float(v.max())==h-1
    assert hook.calls==6 and not mod._forward_pre_hooks
