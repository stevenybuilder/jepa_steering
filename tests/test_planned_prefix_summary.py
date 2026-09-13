import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prefix_analysis',ROOT/'analysis/mechanism/planned_prefix_summary.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def triplet(value):return {'visual':float(value),'proprio':0.,'weighted':float(value)}


def fixture():
    prefix=np.arange(60,dtype=np.float32).reshape(3,20).tolist()
    parent={'traces':{arm:{'selected_plan':prefix} for arm in m.ARMS}}
    physical={}
    for arm,distance,cost in zip(m.ARMS,(10.,9.,11.),(2.,3.,1.)):
        physical[arm]={'terminal_ee_distance':distance,'initial_ee_distance':12.,'actual_goal_cost':triplet(cost),
            'prefix_success':False,'elementary_steps':15,'selected_prefix_sha256':m.prefix_sha(prefix),
            'executed_actions_sha256':'a'*64,'endpoint_observation_sha256':'b'*64,'endpoint_physics_sha256':'c'*64}
    forecast={arm:{plan:{'actual_endpoint_mse':triplet(1.),'predicted_goal_cost':triplet(2.)}
        for plan in m.ARMS} for arm in m.ARMS}
    forecast[m.EDITS[0]]['native']['actual_endpoint_mse']=triplet(10.)
    forecast[m.EDITS[0]][m.EDITS[0]]['actual_endpoint_mse']=triplet(2.)
    forecast[m.EDITS[0]][m.EDITS[0]]['predicted_goal_cost']=triplet(1.)
    preferences={m.EDITS[0]:{'predicted_H3_own_minus_native':-1.,'actual_H3_own_minus_native':1.,
            'preference_reversal':True,'predicted_tie':False,'actual_tie':False},
        m.EDITS[1]:{'predicted_H3_own_minus_native':0.,'actual_H3_own_minus_native':-1.,
            'preference_reversal':False,'predicted_tie':True,'actual_tie':False}}
    return {'forecast_horizon_scored':3,'CEM_optimized_horizon':6,'executed_elementary_steps_per_prefix':15,
        'metrics':{'physical':physical,'forecast':forecast,'preferences':preferences,'initial_goal_cost':triplet(4.)}},parent


def complete_frames():
    payload,parent=fixture();rows={name:[] for name in ('physical_cases','forecast_cases','preference_cases','primary_cases')}
    for task,episode in sorted(m.EXPECTED):
        for name,values in m.extract_case(payload,(task,episode),parent).items():rows[name].extend(values)
    return {name:pd.DataFrame(values) for name,values in rows.items()}


class PrefixAnalysisTests(unittest.TestCase):
    def test_prefix_hash_matches_actual_torch_source_convention(self):
        value=torch.arange(60,dtype=torch.float32).reshape(3,20)
        expected=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.numpy().tobytes()).hexdigest()
        self.assertEqual(m.prefix_sha(value.tolist()),expected)
        with self.assertRaises(ValueError):m.prefix_sha(np.zeros((6,20)))

    def test_all_nine_crosses_and_correct_same_native_prefix_primary(self):
        payload,parent=fixture();rows=m.extract_case(payload,('reach',4),parent)
        self.assertEqual(len(rows['forecast_cases']),9)
        self.assertEqual(len(rows['primary_cases']),6)
        primary={row['metric']:row['effect'] for row in rows['primary_cases'] if row['arm']==m.EDITS[0]}
        self.assertEqual(primary,{'terminal_ee_distance':-1.,'actual_weighted_goal_cost':1.,
            'same_native_prefix_forecast_weighted_mse':9.})

    def test_preference_reversal_and_tie_are_separate(self):
        payload,parent=fixture();rows=m.extract_case(payload,('reach-wall',4),parent)['preference_cases']
        self.assertTrue(rows[0]['preference_reversal']);self.assertFalse(rows[0]['either_tie'])
        self.assertFalse(rows[1]['preference_reversal']);self.assertTrue(rows[1]['predicted_tie'])
        self.assertEqual((rows[1]['predicted_preference_sign'],rows[1]['actual_preference_sign']),(0,-1))

    def test_preference_declaration_cannot_disagree_with_grid(self):
        payload,parent=fixture();payload['metrics']['preferences'][m.EDITS[0]]['preference_reversal']=False
        with self.assertRaises(ValueError):m.extract_case(payload,('reach',4),parent)

    def test_rejects_incomplete_cross_wrong_horizon_and_unregistered_case(self):
        payload,parent=fixture();del payload['metrics']['forecast'][m.EDITS[0]]['native']
        with self.assertRaises(ValueError):m.extract_case(payload,('reach',4),parent)
        payload,parent=fixture();payload['forecast_horizon_scored']=6
        with self.assertRaises(ValueError):m.extract_case(payload,('reach',4),parent)
        payload,parent=fixture()
        with self.assertRaises(ValueError):m.extract_case(payload,('reach',0),parent)

    def test_weighted_cost_and_actual_selected_actions_are_audited(self):
        payload,parent=fixture();payload['metrics']['physical']['native']['actual_goal_cost']['weighted']=12.
        with self.assertRaises(ValueError):m.extract_case(payload,('reach',4),parent)
        payload,parent=fixture();parent['traces']['native']['selected_plan']=np.zeros((3,20)).tolist()
        with self.assertRaises(ValueError):m.extract_case(payload,('reach',4),parent)

    def test_no_nonfinite_negative_or_boolean_error(self):
        for invalid in (float('nan'),float('inf'),-1.,True):
            with self.assertRaises(ValueError):m.scalar(invalid)

    def test_partial_duplicate_and_outside_scenario_coverage_rejected(self):
        frames=complete_frames();m.validate_coverage(frames)
        for mutation in ('missing','duplicate','outside'):
            changed={name:frame.copy() for name,frame in frames.items()}
            if mutation=='missing':changed['forecast_cases']=changed['forecast_cases'].iloc[:-1]
            elif mutation=='duplicate':changed['forecast_cases']=pd.concat([changed['forecast_cases'],changed['forecast_cases'].iloc[:1]])
            else:changed['forecast_cases'].loc[0,'episode']=0
            with self.assertRaises(ValueError):m.validate_coverage(changed)

    def test_bootstrap_is_scenario_paired_deterministic_and_task_specific(self):
        a=m.weights('reach',100);b=m.weights('reach',100);c=m.weights('reach-wall',100)
        np.testing.assert_array_equal(a,b);self.assertFalse(np.array_equal(a,c))
        np.testing.assert_allclose(a.sum(1),1.)
        with self.assertRaises(ValueError):m.statistics(np.arange(56),a)

    def test_family_twelve_quantiles_and_se_are_exact(self):
        values=np.arange(28,dtype=float);w=m.weights('reach',2000)
        result=m.statistics(values,w,True);expected=np.quantile(w@values,[.05/24,1-.05/24])
        self.assertEqual(result['bonferroni_family'],12)
        self.assertEqual(result['bonferroni_95_low'],expected[0]);self.assertEqual(result['bonferroni_95_high'],expected[1])
        self.assertAlmostEqual(result['scenario_se'],values.std(ddof=1)/np.sqrt(28))
        self.assertIsNone(m.statistics(values,w)['bonferroni_95_low'])

    def test_complete_summary_preserves_all12_primary_cells_and_tie_contingencies(self):
        result=m.summarize(complete_frames(),replicates=100)
        self.assertEqual(len(result['primary_summary']),12)
        self.assertEqual(set(result['primary_summary'].n),{28})
        self.assertEqual(set(result['primary_summary'].bonferroni_family),{12})
        self.assertEqual(len(result['forecast_cases']),504);self.assertEqual(len(result['physical_cases']),168)
        self.assertEqual(len(result['preference_contingency']),36)
        self.assertTrue((result['preference_contingency'].groupby(['task','arm'])['count'].sum()==28).all())
        self.assertEqual(set(result['forecast_summary'].inference_scope),{'descriptive_only'})

    def test_complete_path_gate_rejects_before_outcomes_read(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'before any physical outcome read'):m.complete_paths(temp)

    def test_member_requires_unique_case_scoped_full_archive(self):
        key=('reach',4);cloud={'verified_archive_members':{'run/reach/episode-4/physical-prefix.pt':'abc'}}
        m.member(cloud,key,'physical-prefix.pt','abc')
        with self.assertRaises(ValueError):m.member(cloud,('reach',5),'physical-prefix.pt','abc')
        cloud['verified_archive_members']['other/reach/episode-4/physical-prefix.pt']='abc'
        with self.assertRaises(ValueError):m.member(cloud,key,'physical-prefix.pt','abc')


if __name__=='__main__':unittest.main()
