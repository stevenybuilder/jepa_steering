import io
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
import collect_pusht_scripted_baselines as adapter


class AdapterTests(unittest.TestCase):
    def test_all_fifty_required(self):
        receipt={"complete":True,"panel":"pusht_scripted_push_goal_v2","outputs":[{"episode":i} for i in range(50)]}
        self.assertEqual(adapter.validate_inputs({"panel":receipt["panel"]},receipt,[0]),[{"episode":0}])
        receipt["outputs"].pop()
        with self.assertRaises(RuntimeError):adapter.validate_inputs({"panel":receipt["panel"]},receipt,[0])

    def test_old_panel_not_silently_reused(self):
        with self.assertRaises(RuntimeError):adapter.validate_inputs({"panel":"official21"},{},[0])

    def test_success_is_not_logged(self):
        target=io.StringIO()
        adapter.Progress(target).write(json.dumps({"event":"episode_saved","episode":0,"seconds":100,"success":True,"final_success":True})+"\n")
        row=json.loads(target.getvalue())
        self.assertNotIn("success",row)
        self.assertNotIn("final_success",row)
        self.assertEqual(row["seconds"],100)


if __name__=="__main__":unittest.main()
