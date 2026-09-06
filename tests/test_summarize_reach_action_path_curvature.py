import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from summarize_reach_action_path_curvature import summarize

class SummaryTests(unittest.TestCase):
    def test_refuses_missing_or_substituted_states(self):
        with self.assertRaises(ValueError):summarize([])
        rows=[{'complete':True,'source':{'episode':e},'path':{'pair':p,'radius':r}}
              for e in (0,1,4,12) for p in range(4) for r in (1,4)]
        with self.assertRaises(ValueError):summarize(rows)
    def test_equal_state_weight_and_native_truth_distinction(self):
        rows=[]
        for e in (0,1,4,7):
            for p in range(4):
                for radius in (1,4):
                    sites=[]
                    for b in (1,3,5):
                        for h in (1,3,6):
                            methods=[]
                            for name in ('cubic_equal_data','linear_equal_data','near_chord','reparameterized_chord','reflected_curvature'):
                                methods.append(dict(method=name,midpoint_l2_error=1.,downstream_native_fullspatial_mse_after_patch=1.,
                                    actual_future_mse_delta_after_patch=-e if name=='cubic_equal_data' else 0.))
                            sites.append(dict(block=b,intervention_horizon=h,self_patch_exact=True,methods=methods,
                                geometry=dict(path_length=1.,chord_length=1.,path_chord_ratio=1.,max_orthogonal_fraction_of_chord=.1)))
                    rows.append(dict(complete=True,source=dict(episode=e,historical_single_vs_batch_visual_maxabs=0.),
                        path=dict(pair=p,radius=radius,normalization_raw_roundtrip_maxabs=0.,raw_xyz_outside_unit_box_counts=[0]*5),
                        rows=sites,baseline_actual_fullspatial_mse_by_horizon=[10.]*6))
        # Keep test mathematical; avoid repeated descriptive bootstrap sampling.
        with patch('summarize_reach_action_path_curvature.episode_interval',side_effect=lambda x:dict(initial_state_values=x,mean=sum(x)/len(x))):
            result=summarize(rows)
        values=result['views'][0]['methods']['cubic_equal_data']['actual_future_error_reduction_percent_vs_native']
        self.assertEqual(values['initial_state_values'],[0.,10.,40.,70.]);self.assertEqual(values['mean'],30.)
        self.assertEqual(result['independent_initial_states'],4)

if __name__=='__main__':unittest.main()
