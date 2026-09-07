import copy
import unittest
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from publish_geometry_followup import measured_rows, render, insert_panel, decision_evidence, render_decisions


class FollowupTests(unittest.TestCase):
    def fixture(self):
        items=[]
        for radius in (1,4):
            for block in (1,3,5):
                for horizon in (1,3,6):
                    measure=dict(mean=.2, n_initial_states=4, initial_state_values=[.1,.2,.2,.3])
                    items.append(dict(radius=radius,block=block,horizon=horizon,
                        modes={mode:dict(max_orthogonal_fraction_of_chord=copy.deepcopy(measure))
                               for mode in ('native','actual_past_context')}))
        return dict(complete=True,initial_states=4,geometry_views=items,
            forecast_views=[dict(radius=1,horizon=3,native_actual_mse=dict(mean=2),actual_past_actual_mse=dict(mean=1.5))])

    def test_correct_direction_and_independent_count(self):
        rows=measured_rows(self.fixture())
        self.assertEqual(rows[-1]['value'],25)
        self.assertEqual(rows[-1]['n_initial_states'],4)
        self.assertIsNone(rows[-1]['block'])
        self.assertEqual(len(rows),37)

    def test_incomplete_source_rejected(self):
        report=self.fixture();report['complete']=False
        with self.assertRaises(ValueError):measured_rows(report)

    def test_unique_measured_cells_required(self):
        rows=measured_rows(self.fixture())
        self.assertIn('offline diagnostic',render(rows))
        with self.assertRaises(ValueError):render(rows+[rows[0]])

    def test_existing_html_fragment_supported(self):
        self.assertEqual(insert_panel('<style>x</style><h1>Map</h1>', '<section>new</section>'),
                         '<style>x</style><section>new</section><h1>Map</h1>')
        with self.assertRaises(ValueError):insert_panel('invalid', 'new')

    def test_verified_decisions_keep_nulls_controls_and_state_units(self):
        root=Path(__file__).resolve().parents[2]/'artifacts/geometry_map'
        data=decision_evidence(root)
        self.assertEqual(len(data['rows']),58)
        self.assertTrue(all(r['n_initial_states']==4 and r['plans_per_state']==64 for r in data['rows']))
        self.assertTrue(all(r['heldout_performance']=='untested' for r in data['rows']))
        self.assertFalse(any(r['continuation_gate']=='pass_exploratory_only' for r in data['rows']))
        row=next(r for r in data['rows'] if r['arm']=='calibrated_semantic_plus')
        self.assertAlmostEqual(row['mean'],.46003854075)
        self.assertEqual(row['positive_states'],1)
        row=next(r for r in data['rows'] if r['arm']=='pointwise/joint_xy_squared_proxy/learned/span_only' and r['channel']=='predicted')
        self.assertAlmostEqual(row['mean'],row['matched_control_mean_pp'])
        self.assertEqual(row['positive_states'],2)
        self.assertEqual(len(data['proposed_decision_matrix']),4)
        ET.fromstring(render_decisions(data))
        json.dumps(data,allow_nan=False)

    def test_cem_kept_separate_from_fixed64_and_held_efficacy(self):
        root=Path(__file__).resolve().parents[2]/'artifacts/geometry_map'
        data=decision_evidence(root)
        self.assertEqual(data['cem_audit']['status'],'complete_compact_source_verified')
        self.assertFalse(data['cem_audit']['report']['closed_loop_full_episode'])
        self.assertAlmostEqual(data['cem_audit']['report']['stages']['30']['mean_minus_best_requested_coverage']['equal_state_mean'],-.002910190754)


if __name__=='__main__':unittest.main()
