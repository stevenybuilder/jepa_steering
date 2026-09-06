import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_coordinate_preserving_chart import first_ray_match,decode_shift
from native_chart_control import LearnedChart


def test_linear_ray_matches_each_independent_target():
    target=torch.tensor([.25,.5,.75],dtype=torch.float64)
    delta,meta=first_ray_match(lambda a:torch.stack([2*a,0*a],1),target)
    torch.testing.assert_close(delta.norm(dim=-1),target,atol=1e-10,rtol=0)
    torch.testing.assert_close(meta['alpha'],target/2,atol=1e-10,rtol=0)


def test_nonmonotonic_ray_uses_first_observed_crossing():
    target=torch.tensor([.3],dtype=torch.float64)
    f=lambda a:(a+.3*torch.sin(4*torch.pi*a))[:,None]
    delta,meta=first_ray_match(f,target)
    assert meta['observed_grid_crossings'][0]>1
    assert meta['alpha'][0]<.2
    torch.testing.assert_close(delta.norm(dim=-1),target,atol=1e-10,rtol=0)


def test_unbracketed_target_fails_not_silently_clipped():
    try:first_ray_match(lambda a:a[:,None],torch.tensor([2.],dtype=torch.float64))
    except ValueError as exc:assert 'bracket' in str(exc)
    else:raise AssertionError('No feasible dose incorrectly accepted')


def test_zero_target_remains_exact_zero():
    delta,meta=first_ray_match(lambda a:torch.stack([a,a.square()],1),torch.zeros(2,dtype=torch.float64))
    assert torch.equal(delta,torch.zeros_like(delta))
    assert torch.equal(meta['alpha'],torch.zeros_like(meta['alpha']))


def test_coordinate_ray_keeps_graph_coordinates_after_dose_matching():
    generator=torch.Generator().manual_seed(8)
    c=torch.randn(40,4,generator=generator,dtype=torch.float64)
    data=torch.cat([c,c.square(),torch.zeros(40,56,dtype=torch.float64)],1)
    chart=LearnedChart.fit(data);x=data[:3];shift=torch.tensor([.2,-.1,.1,.15],dtype=torch.float64)
    full=decode_shift(chart,x,shift,torch.ones(3,dtype=torch.float64),True)
    matched,meta=first_ray_match(lambda a:decode_shift(chart,x,shift,a,True),.8*full.norm(dim=-1))
    torch.testing.assert_close(matched@chart.basis.T,meta['alpha'][:,None]*shift,atol=1e-12,rtol=1e-10)
    assert torch.all((meta['alpha']>=0)&(meta['alpha']<=1))
