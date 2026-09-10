import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from run_action_path_curvature import fixed_action_path, replace_newest, ResidualReplacement, interpolation_diagnostics, tensor_mse_by_horizon


def test_antithetic_paths_and_fixed_center():
    g=torch.Generator().manual_seed(7)
    center=torch.randn(30,2,generator=g)
    pairs=[torch.randn(30,2,generator=g)*.01 for _ in range(4)]
    raw=torch.stack([center]+[v for d in pairs for v in (center+d,center-d)])
    path,meta=fixed_action_path(raw,2,4)
    assert torch.equal(path[2],center)
    torch.testing.assert_close(path[4]-center,4*(raw[5]-center))
    assert not meta['clipping_applied']
    with pytest.raises(ValueError):fixed_action_path(raw,4,1)


def test_full_spatial_replacement_keeps_old_context():
    original=torch.randn(5,512,400)
    replacement=torch.randn(5,256,400)
    result=replace_newest(original,replacement)
    assert torch.equal(result[:,:256],original[:,:256])
    assert torch.equal(result[:,-256:],replacement)
    assert torch.equal(replace_newest(original,original[:,-256:]),original)


def test_replacement_single_horizon_and_hook_removal():
    module=torch.nn.Identity(); x=torch.zeros(5,256,400); replacement=torch.ones_like(x)
    with ResidualReplacement(module,3,replacement) as patch:
        values=[module(x) for _ in range(6)]
    assert [float(v.sum()) for v in values]==[0,0,float(x.numel()),0,0,0]
    assert patch.fired==1 and not module._forward_hooks


def test_scores_keep_horizon_and_detect_overshoot():
    native=torch.zeros(6,5,1,2,2,3); changed=native.clone(); changed[3,2]=2
    mse=tensor_mse_by_horizon(changed,native)
    assert mse.shape==(6,5) and float(mse.sum())==4
    sample=torch.tensor([[0.,1.],[1.,2.],[2.,3.],[3.,4.]])
    d=interpolation_diagnostics(torch.tensor([4.,5.]),sample,torch.tensor([1.5,2.5]))
    assert d['coordinate_box_overshoot_fraction']==1
