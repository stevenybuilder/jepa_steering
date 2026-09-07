import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from cem_search_audit_v1_export import aggregate


class Tests(unittest.TestCase):
    def test_four_state_mean_and_sign(self):
        rows=[dict(episode=e,complete=True,all_physics_replays_exact=True,stage_differences=[dict(stage=s,mean_minus_best_requested_coverage=(e-1)*.01,mean_minus_best_model_cost=2.,mean_minus_best_xy_px=-1.) for s in (1,15,30)]) for e in range(4)]
        r=aggregate(rows);self.assertAlmostEqual(r['stages']['30']['mean_minus_best_requested_coverage']['equal_state_mean'],.005)
        self.assertFalse(r['closed_loop_full_episode'])
        with self.assertRaises(ValueError):aggregate(rows[:3])

    def test_incomplete_not_scientific_negative(self):
        with self.assertRaises(ValueError):aggregate([])


if __name__=='__main__':unittest.main()
