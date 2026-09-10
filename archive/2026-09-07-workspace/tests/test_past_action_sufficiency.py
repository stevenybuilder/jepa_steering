import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from analyze_past_action_sufficiency import ARMS,check_nulls,contrast,costs,rank_comparison,validate_sources,reconstructed_arguments
from run_action_history_clamp import clamp_arguments


class PastActionTests(unittest.TestCase):
    def test_nulls_reject_changed_central_or_h1(self):
        predictions={arm:{k:torch.zeros(7,5,2) for k in ('visual','proprio')} for arm in ARMS}
        self.assertTrue(check_nulls(predictions))
        predictions[ARMS[2]]['visual'][1,0,0]=1
        with self.assertRaises(ValueError):check_nulls(predictions)

    def test_reconstruction_changes_only_previous_action(self):
        actions=torch.arange(5*6*10,dtype=torch.float32).reshape(5,6,10)
        predictions={arm:dict(visual=torch.zeros(7,5,1,1,1,3),proprio=torch.zeros(7,5,2,2)) for arm in ARMS}
        refs=[];shapes=[]
        for h in range(1,7):
            w=min(h,2);ref=(torch.zeros(1,w,1,1,1,3),actions[2:3,h-w:h],torch.zeros(1,w,2,2));refs.append(ref)
            shapes.append([dict(shape=list(x.repeat(7,*([1]*(x.ndim-1))).shape),stride=list(x.repeat(7,*([1]*(x.ndim-1))).stride())) for x in ref])
        p=dict(predictions=predictions,normalized_actions=actions,central_predictor_inputs=refs,context_shapes={arm:shapes for arm in ARMS})
        self.assertFalse(reconstructed_arguments(p,1,clamp_arguments)['previous_action_changed'])
        self.assertTrue(reconstructed_arguments(p,3,clamp_arguments)['previous_action_changed'])

    def test_noncentral_effect_and_native_scale(self):
        base=torch.tensor([[-2.],[-1.],[0.],[1.],[2.]])
        changed=base*2
        result=contrast(base,changed,base)
        self.assertEqual(result['noncentral_mse'],2.5)
        self.assertEqual(result['effect_over_native_variation'],1.)
        self.assertTrue(result['central_exact'])

    def test_native_weighting_and_choice(self):
        enc=dict(visual=torch.ones(7,5,1,2),proprio=torch.ones(7,5,2)*2)
        goal=dict(visual=torch.zeros(1,1,2),proprio=torch.zeros(1,2))
        torch.testing.assert_close(costs(enc,goal),torch.full((7,5),1.4))
        result=rank_comparison([0,1,2,3,4],[1,0,2,3,4])
        self.assertEqual(result['pair_order_changes'],1)
        self.assertTrue(result['choice_changed'])

    def test_held_filter_before_tensor_load(self):
        done=dict(complete=True,paths=32,held_access=False,outputs=[])
        protocol=dict(rows=[dict(key=f'near-dev-{ep:03}',split='held') for ep in range(4)])
        with self.assertRaisesRegex(ValueError,'Held source'):validate_sources(done,protocol)


if __name__=='__main__':unittest.main()
