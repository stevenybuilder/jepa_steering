from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
from collect_eight_addendum import reach_shard,push_inputs


class EightTests(unittest.TestCase):
    def test_reach_two_disjoint_shards(self):
        m={"addendum":"eight-full-baselines-v1","allowed_new_episode_ids":list(range(62,66)),"shards":[{"instance_id":1,"episode_ids":[62,63]},{"instance_id":2,"episode_ids":[64,65]}]}
        self.assertEqual(reach_shard(m,2)["episode_ids"],[64,65])
    def test_push_frozen_first_four_only(self):
        m={"panel":"pusht_scripted_push_goal_reserve_v1"};r={**m,"complete":True,"outputs":[{"episode":i} for i in range(50,100)]}
        self.assertEqual(len(push_inputs(m,r,[50,51])),2)
        with self.assertRaises(RuntimeError):push_inputs(m,r,[54,55])


if __name__=="__main__":unittest.main()
