import copy
import json
from pathlib import Path

import pytest
import torch

from offline_study.action_counterfactual_pilot import (
    ARMS, CounterfactualHook, action_range, private_seed, random_directions,
    reconstruction, replacement, swap_h3_actions, validate_protocol,
)


def fixture_weight():
    weight=torch.zeros(400,20)
    weight[:20]=torch.eye(20)
    return weight


def test_exact35_registry_and_protocol():
    protocol=json.loads((Path(__file__).parents[1]/'paper/data/action_counterfactual_protocol.json').read_text())
    validate_protocol(protocol)
    assert len(ARMS)==len(set(ARMS))==35
    protocol['arms_per_bank'].pop()
    with pytest.raises(ValueError,match='registry'):validate_protocol(protocol)


def test_action_swap_only_h3_cyclic_next():
    actions=torch.arange(6*300*20).reshape(6,300,20).float()
    before=actions.clone();changed=swap_h3_actions(actions)
    assert torch.equal(actions,before)
    assert torch.equal(changed[2,:-1],actions[2,1:])
    assert torch.equal(changed[2,-1],actions[2,0])
    assert torch.equal(changed[[0,1,3,4,5]],actions[[0,1,3,4,5]])


def test_literal_weight_svd_rank_and_reconstruction():
    q,audit=action_range(fixture_weight())
    assert q.shape==(400,20) and q.dtype==torch.float64
    assert audit['retained_rank']==20
    assert audit['singular_values']==[1.]*20
    assert audit['svd_reconstruction_relative_error']==0
    bad=fixture_weight();bad[:,19]*=1e-8
    _,audit=action_range(bad)
    assert audit['retained_rank']==19
    with pytest.raises(ValueError):action_range(torch.zeros(400,20))


def test_shared_gaussian_projection_private_rng_and_delivered_norms():
    q,_=action_range(fixture_weight())
    state=torch.get_rng_state().clone()
    directions,sha=random_directions(q,private_seed('reach',0,'original',2),'cpu')
    repeat,repeat_sha=random_directions(q,private_seed('reach',0,'original',2),'cpu')
    assert torch.equal(state,torch.get_rng_state()) and sha==repeat_sha
    assert torch.allclose(directions['random_range']+directions['random_off_range'],directions['random_isotropic'])
    generator=torch.Generator().manual_seed(8)
    z=torch.randn(300,2,20,generator=generator)@fixture_weight().T
    audits={}
    for mode,direction in directions.items():
        changed,audit=replacement(z,1,mode,q,direction)
        audits[mode]=audit
        assert torch.equal(changed[:,0],z[:,0])
        assert audit['max_relative_norm_error']<=1e-5
        assert torch.equal(direction,repeat[mode])
    assert max(audits['random_range']['off_range_energy_fraction'])<1e-12
    assert max(audits['random_off_range']['in_range_energy_fraction'])<1e-12
    donor,audit=replacement(z,0,'donor',q)
    assert torch.equal(donor[:,0],torch.roll(z[:,0],-1,0))
    assert torch.equal(donor[:,1],z[:,1])


def test_zero_donor_targets_remain_zero_without_dropping():
    q,_=action_range(fixture_weight());z=torch.ones(300,2,400)
    directions,_=random_directions(q,1,'cpu')
    changed,audit=replacement(z,1,'random_range',q,directions['random_range'])
    assert torch.equal(changed,z) and audit['zero_target_count']==300
    assert audit['in_range_energy_fraction']==[None]*300


class ToyBlock(torch.nn.Module):
    def forward(self,x,z):
        # Older action conditioning affects newest output: omitting H4 is detectable.
        return x+z+.125*z.sum(1,keepdim=True)


class ToyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks=torch.nn.ModuleList([ToyBlock() for _ in range(6)])
        self.weight=fixture_weight()
        self.eval()

    def forward(self,x,actions):
        z=actions@self.weight.T
        for block in self.predictor_blocks:x=block(x,z)
        return x


def toy_unroll(predictor,actions):
    states=[torch.zeros(300,400)]
    for h in range(6):
        x=torch.stack(states[-2:],dim=1)
        a=actions[max(0,h-1):h+1].transpose(0,1)
        states.append(predictor(x,a)[:,-1])
    return torch.stack(states)


def test_persistent_allblock_matches_raw_action_and_h3_only_does_not():
    predictor=ToyPredictor();q,_=action_range(fixture_weight())
    generator=torch.Generator().manual_seed(6)
    actions=torch.randn(6,300,20,generator=generator)
    native=toy_unroll(predictor,actions)
    with CounterfactualHook(predictor,basis=q) as capture:zero=toy_unroll(predictor,actions)
    assert torch.equal(native,zero) and len(capture.captured)==12
    coherent=toy_unroll(predictor,swap_h3_actions(actions))
    with CounterfactualHook(predictor,basis=q,cache=capture.captured,layers=range(6),persistent=True,mode='donor') as persistent:
        patched=toy_unroll(predictor,actions)
    assert torch.equal(patched,coherent)
    assert {(a['horizon'],a['position']) for a in persistent.audits}=={(3,1),(4,0)}
    assert len(persistent.audits)==12
    with CounterfactualHook(predictor,basis=q,cache=capture.captured,layers=range(6),mode='donor'):
        transient=toy_unroll(predictor,actions)
    assert torch.equal(transient[:4],coherent[:4])
    assert not torch.equal(transient[4:],coherent[4:])
    assert torch.equal(patched[:3],native[:3])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in predictor.modules())


def test_hooks_cleanup_on_exception_and_incomplete_calls():
    predictor=ToyPredictor();q,_=action_range(fixture_weight())
    with pytest.raises(RuntimeError):
        with CounterfactualHook(predictor,basis=q):raise RuntimeError('fixture')
    assert all(not m._forward_pre_hooks for m in predictor.modules())
    with pytest.raises(ValueError,match='Incomplete'):
        with CounterfactualHook(predictor,basis=q):pass


def test_reconstruction_ratio_of_means_negative_and_undefined():
    native={m:torch.zeros(7,300,2) for m in ('visual','proprio')}
    coherent={m:torch.ones(7,300,2) for m in native}
    forecast={m:torch.full((7,300,2),3.) for m in native}
    coherent['visual'][:,0]=0;forecast['visual'][:,0]=0
    rows=reconstruction(native,coherent,forecast)
    assert len(rows)==9
    visual=next(r for r in rows if r['modality']=='visual' and r['horizon']==6)
    assert visual['reconstruction'][0] is None and visual['n_defined']==299
    assert visual['pooled_reconstruction']==-3
    official=next(r for r in rows if r['modality']=='official' and r['horizon']==6)
    assert official['denominator_mse'][0]==pytest.approx(.1)
    assert official['pooled_reconstruction']==pytest.approx(-3)
    assert all(r['pooled_reconstruction'] is None for r in reconstruction(native,native,native))
