import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_chart_behavior_pullback import raw_lift,control_regularizer,verified_load
from native_chart_control import LearnedChart
from run_behavior_pullback import spline_weights


def test_exact_zero_native_anchor_and_common_support():
    g=torch.Generator().manual_seed(12)
    recipient=torch.randn(9,4,20,generator=g)
    basis=torch.linalg.qr(torch.randn(80,64,generator=g,dtype=torch.float64)).Q.T
    zero=torch.zeros(9,64,dtype=torch.float64)
    assert torch.equal(raw_lift(recipient,zero,basis),recipient)
    delta=torch.randn(9,64,generator=g,dtype=torch.float64)*.02
    lifted=(raw_lift(recipient,delta,basis)-recipient).double().flatten(1)
    assert float((lifted-(lifted@basis.T)@basis).abs().max())<2e-7


def test_shared_controls_and_zero_spline_are_identical():
    weights=spline_weights(torch.linspace(0,1,10),torch.linspace(0,1,20))
    controls=torch.zeros(10,4,dtype=torch.float64,requires_grad=True)
    path=weights@controls
    assert torch.equal(path,torch.zeros_like(path))
    assert control_regularizer(path)==0
    control_regularizer(path).backward()
    assert torch.equal(controls.grad,torch.zeros_like(controls))


def test_regularizer_is_equal_for_both_chart_families():
    u=torch.full((20,4),100.,dtype=torch.float64)
    torch.testing.assert_close(control_regularizer(u),torch.tensor(.001,dtype=torch.float64))


def test_double_raw_lift_preserves_autograd():
    recipient=torch.zeros(9,4,20,dtype=torch.float64)
    basis=torch.eye(80,dtype=torch.float64)[:64]
    delta=torch.randn(9,64,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:raw_lift(recipient,x,basis),(delta,))


def test_loader_rejects_split_before_tensor_loading():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        (root/'DONE.json').write_text(json.dumps(dict(complete=True,episode=0,split='confirmation',outputs=[])))
        try:verified_load(root,'nonexistent.pt',0,'development_external')
        except ValueError as exc:assert 'split' in str(exc)
        else:raise AssertionError('Sealed split accepted')


def test_coordinate_drift_separates_retained_coordinates_from_complement():
    from analyze_chart_behavior_pullback import coordinate_drift
    basis=torch.eye(64,dtype=torch.float64)[:4]
    scale=torch.ones(4,dtype=torch.float64)
    control=torch.full((20,4),.2,dtype=torch.float64)
    requested=.5*control.tanh()
    delta=(requested@basis)[:,None].expand(20,9,64).clone()
    delta[:,:,4:]=123.
    assert coordinate_drift(delta,control,basis,scale)['max_coordinate_drift_norm']==0
    delta[:,:,0]+=.5
    torch.testing.assert_close(torch.tensor(coordinate_drift(delta,control,basis,scale)['mean_coordinate_drift_norm']),torch.tensor(.5))
