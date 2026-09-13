import hashlib
import unittest

import torch

from offline_study.steered_cem_pilot import compact_trace


class SteeredCemCompactTest(unittest.TestCase):
    def test_preserves_native_selected_prefix_and_full_proposal(self):
        actions=torch.arange(6*300*20,dtype=torch.float32).reshape(6,300,20)
        row={'iteration':0,'candidate_actions':actions,'proposal_mean':torch.zeros(6,20),
             'proposal_std':torch.ones(6,20),'proposal_entropy_nats':torch.tensor(170.27),
             'proposal_entropy_semantics':'diagonal_Gaussian_before_clipping_and_mean_sample_insertion',
             'objective_costs':torch.arange(300,dtype=torch.float32),
             'elite_indices':torch.arange(10),'best_runnerup_margin':1.,'elite_boundary_margin':1.}
        payload={'selected_plan':torch.zeros(3,20),'final_mean':torch.ones(6,20),'iterations':[row]}
        before=actions.clone()
        result=compact_trace(payload)
        self.assertEqual(len(result['selected_plan']),3)
        self.assertEqual(len(result['final_mean']),6)
        self.assertEqual(result['iterations'][0]['candidate_actions_shape'],[6,300,20])
        self.assertEqual(result['iterations'][0]['candidate_actions_dtype'],'float32')
        self.assertEqual(result['iterations'][0]['candidate_actions_sha256'],
                         hashlib.sha256(actions.numpy().tobytes()).hexdigest())
        self.assertEqual(len(result['iterations'][0]['objective_costs']),300)
        self.assertEqual(result['iterations'][0]['elite_indices'],list(range(10)))
        self.assertTrue(torch.equal(actions,before))


if __name__=='__main__':unittest.main()
