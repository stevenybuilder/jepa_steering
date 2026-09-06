from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from causal_response_transfer import fit_response,predict_response,select_rows,TRAIN_KEYS,restore_fullprecision_goal

class ResponseTransferTests(unittest.TestCase):
    def test_fixed_pca_and_dimensions(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(48,400));y=rng.normal(size=(48,2,384))
        m=fit_response(x,y)
        self.assertEqual(m['components'].shape,(16,400));self.assertEqual(predict_response(m,x[:2]).shape,(2,2,384))
        np.testing.assert_allclose(m['global_mean'],y.mean(0))
    def test_train_only_split_guard(self):
        d=dict(complete=True,outputs=[dict(path='a.pt',key=TRAIN_KEYS[0],split='fit'),dict(path='b.pt',key='train-1818',split='development_validation')])
        self.assertEqual(len(select_rows(d,'capture')),1)
        with self.assertRaises(ValueError):select_rows(d,'capture',['train-1818'])
        d['outputs'][0]['split']='development_validation'
        with self.assertRaises(ValueError):select_rows(d,'capture')
    def test_constant_jacobian_global_and_ridge(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(48,400));y=np.ones((48,2,384))*.7
        m=fit_response(x,y);np.testing.assert_allclose(predict_response(m,x[:3]+100),.7,atol=1e-12)
    def test_saved_fullprecision_goal_only(self):
        import torch
        from types import SimpleNamespace
        agent=SimpleNamespace(device='cpu',goal_state_enc={'visual':torch.ones(2),'proprio':torch.ones(2)},
            objective=SimpleNamespace(),planner=SimpleNamespace(set_objective=lambda value:None))
        saved={k:torch.ones(2)+1e-7 for k in ('visual','proprio')}
        errors=restore_fullprecision_goal(agent,saved)
        self.assertGreater(errors['visual'],0)
        self.assertTrue(torch.equal(agent.objective.target_enc['visual'],saved['visual']))
        with self.assertRaises(ValueError):restore_fullprecision_goal(agent,{k:v.half() for k,v in saved.items()})
    def test_near_panel_cannot_fit(self):
        d=dict(complete=True,outputs=[dict(path='n.pt',key='near-dev-000',split='development_external')])
        self.assertEqual(len(select_rows(d,'evaluate',near_plan=True)),1)
        with self.assertRaises(ValueError):select_rows(d,'capture',near_plan=True)

if __name__=='__main__':unittest.main()
