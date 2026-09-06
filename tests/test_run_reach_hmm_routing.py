import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from run_reach_hmm_routing import fit_hmm,filter_beliefs,route,equal_energy


def test_hmm_is_normalized_and_does_not_read_future_at_route():
    torch.manual_seed(6);train=torch.randn(10,6,7)
    model=fit_hmm(train,iterations=3)
    torch.testing.assert_close(model['transition'].sum(1),torch.ones(2,dtype=torch.float64))
    a=torch.randn(5,6,7);b=a.clone();b[:,2:]+=100
    assert torch.equal(route(a,model)['hmm'],route(b,model)['hmm'])


def test_routing_preserves_aggregate_patch_energy():
    weight=torch.tensor([.5,.7,1.],dtype=torch.float64);norm=torch.tensor([2.,3.,4.],dtype=torch.float64)
    static=torch.full((3,),.8,dtype=torch.float64);actual=equal_energy(weight,norm,static)
    torch.testing.assert_close((norm*actual).square().sum(),(norm*static).square().sum())
