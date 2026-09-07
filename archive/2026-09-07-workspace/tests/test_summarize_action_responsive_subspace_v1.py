import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from summarize_action_responsive_subspace_v1 import aggregate


def example():
    reports=[]
    for e in range(4):
        score=[]
        for name in ['native']+[k+'_'+s for s in ('minus','plus') for k in ('targeted','random_basis','random_patches','random_both')]:
            score.append(dict(arm=name,choice=0,goal_coverage_delta=0.,selected_goal_coverage=.2,selected_xy_distance_px=2.,xy_delta_px=0.,
                equal_candidate_mse_by_horizon={'visual':[2.]*6,'proprio':[3.]*6},edit=dict(delivered_l2=[0.,1.],raw_relative_l2=[0.,.1],delivered_norm_mismatch=[0.,0.],off_subspace_l2=[0.,1e-6],rounding_l2=[0.,2e-6])))
        reports.append(dict(episode=e,complete=True,all_identity_guards_exact=True,score=score,
            support={**{a['arm']:[1.,1.] for a in score},'native_nearest_other':[1.,1.]},
            geometry=dict(retained_centered_variance=.5,projected_contrast_energy_fraction=.4,selected_projected_contrast_energy_fraction=.3,random_basis_overlap=.01),
            goal_coverage_headroom=.1,full_tensor_sha256='x',full_tensor_bytes=1))
    return reports


class Tests(unittest.TestCase):
    def test_zero_and_four_state_requirement(self):
        r=example();result=aggregate(r)
        self.assertFalse(result['exploratory_gate']['plus']['exploratory_lead'])
        self.assertEqual(result['arms']['targeted_plus']['positive_states'],0)
        with self.assertRaises(ValueError):aggregate(r[:3])

    def test_requires_beating_each_control(self):
        r=example()
        for state in r:
            next(x for x in state['score'] if x['arm']=='targeted_plus')['goal_coverage_delta']=.02
        self.assertTrue(aggregate(r)['exploratory_gate']['plus']['exploratory_lead'])
        next(x for x in r[0]['score'] if x['arm']=='random_both_plus')['goal_coverage_delta']=.08
        self.assertFalse(aggregate(r)['exploratory_gate']['plus']['exploratory_lead'])

    def test_equal_state_not_pooled_error_gain(self):
        r=example()
        for i,state in enumerate(r):
            base=next(x for x in state['score'] if x['arm']=='native')
            edit=next(x for x in state['score'] if x['arm']=='targeted_plus')
            base['equal_candidate_mse_by_horizon']['visual']=[10.**i]*6
            edit['equal_candidate_mse_by_horizon']['visual']=[10.**i*(1-.01*i)]*6
        self.assertAlmostEqual(aggregate(r)['arms']['targeted_plus']['equal_state_forecast_error_reduction_percent_by_horizon']['visual'][-1],1.5)


if __name__=='__main__':unittest.main()
