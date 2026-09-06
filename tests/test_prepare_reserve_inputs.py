from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
from prepare_reserve_inputs import validate_reserve


class ReserveTests(unittest.TestCase):
    def fixture(self,task):
        ids=list(range(62,112)) if task=="reach_wall" else list(range(50,100))
        a,b=(2026090500,90500) if task=="reach_wall" else (2026090700,92600)
        return {"task":task,"episode_ids":ids,"starts":[{"episode":i,"environment_seed":a+i,"planner_seed":b+i} for i in ids],"existing_same_task_environment_seeds":list(range(a,a+min(ids))),"full_model_rollouts_authorized":False}
    def test_reach_original_seed_rule_and_disjoint_ids(self):self.assertEqual(len(validate_reserve(self.fixture("reach_wall"),"reach_wall")),50)
    def test_push_original_seed_rule_and_disjoint_ids(self):self.assertEqual(len(validate_reserve(self.fixture("pusht"),"pusht")),50)
    def test_overlap_rejected(self):
        m=self.fixture("pusht");m["existing_same_task_environment_seeds"].append(m["starts"][0]["environment_seed"])
        with self.assertRaises(RuntimeError):validate_reserve(m,"pusht")
    def test_no_full_rollout_authority(self):
        m=self.fixture("reach_wall");m["full_model_rollouts_authorized"]=True
        with self.assertRaises(RuntimeError):validate_reserve(m,"reach_wall")


if __name__=="__main__":unittest.main()
