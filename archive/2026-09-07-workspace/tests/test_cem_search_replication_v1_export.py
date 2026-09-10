import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from cem_search_replication_v1_export import aggregate
from cem_search_replication_v1 import SOURCES


class Tests(unittest.TestCase):
    def reports(self):
        return [dict(episode=e,complete=True,all_physics_replays_exact=True,
            metric_rows=[dict(stage=s,kind=k,requested_coverage_final=s/100.,predicted_cost_by_horizon=[1/s],goal_xy_distance_by_horizon=[1/s],native_goal_success_final=False,native_goal_ever_success=False) for s in (1,15,30) for k in ('mean','best')],
            stage_differences=[dict(stage=s,mean_minus_best_requested_coverage=0.,mean_minus_best_model_cost=0.,mean_minus_best_xy_px=0.) for s in (1,15,30)]) for e in SOURCES]
    def test_all_four_new_units_and_equal_weight(self):
        r=aggregate(self.reports());self.assertEqual(r['source_ids'],list(SOURCES));self.assertEqual(r['independent_new_development_initial_groups'],4)
        self.assertAlmostEqual(r['primary']['stage30_minus15_mean_requested_coverage']['equal_source_mean'],.15)
        self.assertEqual(r['selected_stage_plans'],24)
    def test_no_missing_replaced_or_old_states(self):
        rows=self.reports()
        with self.assertRaises(ValueError):aggregate(rows[:3])
        rows[0]['episode']=0
        with self.assertRaises(ValueError):aggregate(rows)


if __name__=='__main__':unittest.main()
