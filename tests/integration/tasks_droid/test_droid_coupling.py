import types
import unittest

import torch

from offline_study.tasks.droid.droid_coupling import ARMS, METHOD, Capture, Hook, Intervention, arm_fields


class Block(torch.nn.Module):
    def forward(self,x,z,**kwargs):
        return x+z[:,:,None,None,None,:]


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.predictor_blocks=torch.nn.ModuleList([Block() for _ in range(12)])
    def forward(self,x,actions,proprio=None):
        z=actions[...,:1].expand(*actions.shape[:2],1024)
        for block in self.predictor_blocks:x=block(x,z,T=x.shape[1])
        return x


class Model:
    device='cpu'
    def __init__(self):self.model=types.SimpleNamespace(predictor=Predictor());self.calls=0
    def unroll(self,context,act_suffix=None):
        self.calls+=1;out=[context[:,0]]
        for i in range(len(act_suffix)):
            history=torch.stack(out[-2:],dim=1)
            actions=act_suffix[max(0,i-1):i+1].transpose(0,1)
            out.append(self.model.predictor(history,actions,None)[:,-1])
        return torch.stack(out)


class DroidCouplingTests(unittest.TestCase):
    def bank(self):
        v=torch.zeros(1,16,16,1024);v.flatten()[0]=1.
        a=torch.zeros(1024);a[0]=1.
        rv=torch.zeros_like(v);rv.flatten()[1]=1.
        ra=torch.zeros_like(a);ra[1]=1.
        return {'method':METHOD,'visual':v,'action':a,'permuted_visual':v.flip(1),
            'random_visual':rv,'random_action':ra,'visual_dose':.2,'action_dose':.3}
    def test_full_factorial_energy_and_random_controls(self):
        bank=self.bank()
        for arm in ARMS:
            v,a=arm_fields(bank,arm,'cpu')
            if arm=='native':self.assertIsNone(v);self.assertIsNone(a)
            if 'equal_standardized_energy' in arm:
                self.assertAlmostEqual(float(v.norm()**2/.2**2+a.norm()**2/.3**2),1.,places=5)
        v,a=arm_fields(bank,'matched_random','cpu')
        self.assertEqual(float((v*bank['visual']).sum()),0)
        self.assertEqual(float((a*bank['action']).sum()),0)
    def test_one_native_pass_identity_short_horizon_and_capture(self):
        model=Model();context=torch.zeros(2,1,1,16,16,1024);actions=torch.zeros(3,2,7)
        native=model.unroll(context,act_suffix=actions)
        with Capture(model.model.predictor) as capture:
            observed=model.unroll(context,act_suffix=actions)
        self.assertTrue(torch.equal(native,observed));self.assertEqual(capture.visual.shape,(2,1,16,16,1024))
        for arm in ('native','zero_dose'):
            before=model.calls;got=Intervention(model,self.bank(),arm)(context,act_suffix=actions)
            self.assertTrue(torch.equal(native,got));self.assertEqual(model.calls-before,1)
        edit=Intervention(model,self.bank(),'joint');before=model.calls;got=edit(context,act_suffix=actions)
        self.assertEqual(model.calls-before,1);self.assertTrue(torch.equal(native[:3],got[:3]))
        self.assertFalse(torch.equal(native[3],got[3]));self.assertEqual(set(edit.last_energy),{'visual','action'})
        self.assertTrue(torch.equal(edit(context,act_suffix=actions[:2]),model.unroll(context,act_suffix=actions[:2])))
        self.assertFalse(model.model.predictor._forward_pre_hooks)
        with Hook(model.model.predictor,torch.zeros(1,16,16,1024),torch.zeros(1024)):
            self.assertTrue(torch.equal(native,model.unroll(context,act_suffix=actions)))
    def test_no_wrong_architecture_or_hook_composition(self):
        model=Model();v,a=arm_fields(self.bank(),'joint','cpu')
        with Hook(model.model.predictor,v,a):
            with self.assertRaises(ValueError):Hook(model.model.predictor,v,a).__enter__()
            with self.assertRaises(ValueError):Capture(types.SimpleNamespace(predictor_blocks=[None]*6))
            model.unroll(torch.zeros(1,1,1,16,16,1024),act_suffix=torch.zeros(3,1,7))


if __name__=='__main__':unittest.main()
