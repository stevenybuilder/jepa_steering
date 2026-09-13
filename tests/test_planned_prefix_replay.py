import copy
import ast
import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from offline_study import planned_prefix_replay as m


def registry():
    manifest = {'scenarios':{t:list(range(4,32)) for t in m.TASKS},
        'model_arms':list(m.ARMS),'plan_sources':list(m.ARMS),
        'package_versions':dict(m.PACKAGES),
        'tail_qualification_cases':[{'task':'reach','episode':4}]}
    protocol = {'episode_ids':list(range(4,32)),'cases':56,
        'model_arms':list(m.ARMS),'plan_sources':list(m.ARMS)}
    return manifest,protocol


class FakeEnv:
    def __init__(self):self.calls=[];self.short=False;self.bad=False;self.mutate=False
    def step_multiple(self,actions):
        self.calls.append(actions.clone())
        n=14 if self.short else 15
        infos=[{'state':np.full(4,i,dtype=np.float32),'proprio':torch.full((1,4),float(i)),
                'success':i==14} for i in range(n)]
        if self.bad:infos[-1]['state'][0]=np.nan
        if self.mutate:actions.add_(1)
        return [torch.ones(1,3,2,2,dtype=torch.uint8)*i for i in range(n)],list(range(n)),[False]*n,infos


class PrefixContractTests(unittest.TestCase):
    def crop_fixture(self):
        path=Path(__file__).resolve().parents[1]/'vendor/jepa-wms/src/datasets/utils/video/transforms.py'
        tree=ast.parse(path.read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
            ('_get_param_spatial_crop','random_resized_crop')]
        namespace={'math':math,'random':random,'np':np,'torch':torch}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
        return SimpleNamespace(spatial_transform=namespace['random_resized_crop'],
            random_resize_scale=(1.,1.),random_resize_aspect_ratio=(1.,1.),
            random_horizontal_flip=False,auto_augment=False,motion_shift=False,reprob=0.,hwc=False)

    def test_original_degenerate_crop_consumes_global_rng_but_private_crop_does_not(self):
        transform=self.crop_fixture();image=torch.arange(3*224*224,dtype=torch.float32).reshape(3,1,224,224)
        original=transform.spatial_transform;before=m.rng_signature('cpu')
        expected=original(image,224,224,(1.,1.),(1.,1.))
        self.assertNotEqual(before,m.rng_signature('cpu'))
        before=m.rng_signature('cpu')
        with m.PrivateEvalCrop(transform) as crop:
            for _ in range(5):
                actual=transform.spatial_transform(image,224,224,(1.,1.),(1.,1.))
                m.assert_bytes(actual,expected,'private/original crop fixture')
        self.assertEqual(before,m.rng_signature('cpu'));self.assertEqual(crop.calls,5)
        self.assertIs(transform.spatial_transform,original)

    def test_private_crop_rejects_augmentation_and_nonsquare_input(self):
        transform=self.crop_fixture();transform.random_resize_scale=(.5,1.)
        with self.assertRaises(ValueError):m.PrivateEvalCrop(transform)
        transform=self.crop_fixture()
        with m.PrivateEvalCrop(transform):
            with self.assertRaises(ValueError):transform.spatial_transform(torch.zeros(3,1,224,223),224,224,(1.,1.),(1.,1.))

    def test_registry_rejects_initial_eight_or_missing_cases(self):
        manifest,protocol=registry();m.validate_registry(manifest,protocol)
        manifest['scenarios']['reach'].insert(0,0)
        with self.assertRaises(ValueError):m.validate_registry(manifest,protocol)

    def test_exact_package_and_qualification_gate(self):
        manifest,protocol=registry();manifest['package_versions']['mujoco']='3.4.0'
        with self.assertRaises(ValueError):m.validate_registry(manifest,protocol)
        manifest,protocol=registry();manifest['tail_qualification_cases']=[]
        with self.assertRaises(ValueError):m.validate_registry(manifest,protocol)

    def test_selected_prefix_is_sixty_coordinates_not_full_mean(self):
        value=torch.arange(60,dtype=torch.float32).reshape(3,20)
        self.assertTrue(torch.equal(m.selected_prefix(value.tolist()),value))
        for invalid in (torch.zeros(6,20),torch.full((3,20),float('nan'))):
            with self.assertRaises(ValueError):m.selected_prefix(invalid)

    def test_zero_and_nonzero_tail_keep_exact_prefix_and_rng(self):
        value=torch.arange(60,dtype=torch.float32).reshape(3,20)
        rng=torch.get_rng_state().clone()
        zero=m.padded_actions(value);other=m.padded_actions(value,True)
        self.assertEqual(zero.shape,(6,1,20))
        self.assertTrue(torch.equal(zero[:3,0],value))
        self.assertTrue(torch.equal(other[:3],zero[:3]))
        self.assertEqual(torch.count_nonzero(zero[3:]),0)
        self.assertEqual(torch.count_nonzero(other[3:]),60)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_denormalize_exactly_once_no_action_repetition(self):
        class Preprocessor:
            calls=0
            def denormalize_actions(self,x):self.calls+=1;return x*2+3
        pre=Preprocessor();value=torch.arange(60,dtype=torch.float32).reshape(3,20)
        actions=m.elementary_actions(value,pre)
        self.assertEqual(pre.calls,1);self.assertEqual(actions.shape,(15,4))
        self.assertTrue(torch.equal(actions,value.reshape(15,4)*2+3))

    def simulate(self,env,initial='initial',physics='physics'):
        record={'environment_seed':17,'initial_sha256':initial,'physics_sha256':physics}
        fake_initial={'marker':'initial'}
        def digest(obs):return 'initial' if 'marker' in obs else 'endpoint'
        with patch.object(m,'reset_initial',return_value=(fake_initial,{'state':np.zeros(4,dtype=np.float32)})), \
             patch.object(m,'physics',return_value=({'qpos':np.zeros(2)},'physics')), \
             patch.object(m,'observation_digest',side_effect=digest):
            return m.physical_prefix(env,record,torch.zeros(15,4))

    def test_exact_reset_gate_precedes_any_action(self):
        env=FakeEnv()
        with self.assertRaises(ValueError):self.simulate(env,initial='wrong')
        self.assertEqual(env.calls,[])
        with self.assertRaises(ValueError):self.simulate(env,physics='wrong')
        self.assertEqual(env.calls,[])

    def test_physical_capture_returns_all_states_and_actual_endpoint(self):
        env=FakeEnv();result=self.simulate(env)
        self.assertEqual(len(env.calls),1)
        self.assertEqual(result['states'].shape,(15,4))
        self.assertTrue(torch.equal(result['endpoint_observation']['visual'],torch.full((1,3,2,2),14,dtype=torch.uint8)))
        self.assertTrue(bool(result['successes'][-1]))
        self.assertEqual(len(result['rewards']),15)
        m.assert_native_repeat(result,self.simulate(FakeEnv()))

    def test_short_nonfinite_or_mutated_action_replay_fails(self):
        for flag in ('short','bad','mutate'):
            env=FakeEnv();setattr(env,flag,True)
            with self.assertRaises(ValueError):self.simulate(env)

    def test_native_repeat_checks_every_state_reward_done_and_success(self):
        original=self.simulate(FakeEnv())
        for key in ('states','rewards','dones','successes','executed_actions'):
            changed=copy.deepcopy(original)
            value=changed[key].reshape(-1)
            value[0]=not value[0] if value.dtype==torch.bool else value[0]+1
            with self.assertRaises(ValueError):m.assert_native_repeat(original,changed)
        changed=copy.deepcopy(original);changed['endpoint_physics_sha256']='different'
        with self.assertRaises(ValueError):m.assert_native_repeat(original,changed)

    def test_byte_digest_detects_dtype_shape_and_signed_zero(self):
        self.assertNotEqual(m.tree_hash(torch.zeros(2)),m.tree_hash(torch.zeros(2,dtype=torch.float64)))
        self.assertNotEqual(m.tree_hash(torch.zeros(2)),m.tree_hash(torch.zeros(1,2)))
        self.assertNotEqual(m.tree_hash(torch.tensor([0.])),m.tree_hash(torch.tensor([-0.])))

    def test_forecast_h3_not_h6_or_zero_based_h2(self):
        context={k:torch.zeros(1,1,2) for k in m.MODALITIES}
        forecast={k:torch.arange(7,dtype=torch.float32)[:,None,None].expand(7,1,2).clone() for k in m.MODALITIES}
        result=m.forecast_endpoint(forecast,context)
        self.assertTrue(torch.equal(result['visual'],torch.tensor([3.,3.])))
        forecast['visual'][0,0,0]=1
        with self.assertRaises(ValueError):m.forecast_endpoint(forecast,context)

    def test_encoding_and_distances_do_not_broadcast(self):
        encoded={k:torch.zeros(1,1,2) for k in m.MODALITIES}
        actual=m.encoded_endpoint(encoded)
        target={'visual':torch.ones(2),'proprio':torch.full((2,),2.)}
        self.assertEqual(m.distances(actual,target),{'visual':1.,'proprio':4.,'weighted':1.4})
        target['visual']=torch.ones(1,2)
        with self.assertRaises(ValueError):m.distances(actual,target)
        encoded['visual']=torch.zeros(1,2,2)
        with self.assertRaises(ValueError):m.encoded_endpoint(encoded)

    def test_archive_member_requires_exact_case_and_unique_identity(self):
        path='run/reach/episode-4/physical-prefix.pt'
        m.archive_member({path:'abc'},'reach',4,'physical-prefix.pt','abc')
        for members in ({path:'changed'}, {'run/reach/episode-5/physical-prefix.pt':'abc'},
                        {path:'abc','other/'+path:'abc'}):
            with self.assertRaises(ValueError):m.archive_member(members,'reach',4,'physical-prefix.pt','abc')


if __name__=='__main__':unittest.main()
