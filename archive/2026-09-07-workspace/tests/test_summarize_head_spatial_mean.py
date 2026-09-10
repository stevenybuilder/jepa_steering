from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from summarize_head_spatial_mean import summary

def fixtures():
    rows=[]
    for i in range(4):
        conditions=[]
        for name,distance in (('unsteered',10.),('head00_H1_spatial_semantic',9.),('head00_H1_spatial_sham',10.)):
            c=dict(name=name,forecast_visual_mse=distance,selected_actual_block_distance_px=distance,selected_actual_wrapped_angle_rad=.1,
                ranking=dict(actual_encoded_regret=distance,selected_physical_xy_distance=distance,rank_correlation_actual_encoded=.5,selected_candidate=0 if name=='unsteered' else 1))
            if name!='unsteered':c.update(head=0,horizon=1,kind='spatial',sham=name.endswith('sham'),
                norm_diagnostics=dict(actual_rounded_residual_norm=[2.]*9,suppression_fraction=[.1]*9))
            conditions.append(c)
        rows.append(dict(key=f'near-dev-{i:03d}',conditions=conditions,seconds=1.,old_single_vs_fresh_batch={}))
    return rows

class HeadSummaryTests(unittest.TestCase):
    def test_state_paired_not_candidate_n(self):
        r=summary(fixtures());m=r['rows'][0]['metrics']['actual_4coordinate_distance_px']['semantic_vs_sham']
        self.assertEqual(m['n_initialstates'],4);self.assertEqual(m['mean'],-1.);self.assertEqual(m['interval95'],[-1.,-1.])
        self.assertEqual(r['rows'][0]['max_matched_actual_norm_error'],0.)
    def test_missing_start_rejected(self):
        with self.assertRaises(ValueError):summary(fixtures()[:3])

if __name__=='__main__':unittest.main()
