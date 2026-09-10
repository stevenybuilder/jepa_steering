import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
from collect_steering_expansion import ProgressOnly, validate_request


class ExpansionTests(unittest.TestCase):
    def manifest(self):
        return {"allowed_new_episode_ids": list(range(24, 52)),
                "seed_rules": {"environment_seed_base": 2026090500, "planner_seed_base": 90500},
                "shards": [{"instance_id": i, "episode_ids": list(range(24+7*i, 31+7*i))} for i in range(4)]}

    def test_frozen_new_ids_and_original_seed_bases(self):
        self.assertEqual(validate_request(self.manifest(), 0)["episode_ids"], list(range(24, 31)))
        self.assertEqual(validate_request(self.manifest(), 3)["episode_ids"], list(range(45, 52)))

    def test_reject_original_episode_overlap(self):
        value = self.manifest(); value["allowed_new_episode_ids"][0] = 23
        with self.assertRaises(RuntimeError): validate_request(value, 0)

    def test_reject_shard_overlap(self):
        value = self.manifest(); value["shards"][1]["episode_ids"][0] = 24
        with self.assertRaises(RuntimeError): validate_request(value, 0)

    def test_reject_seed_changes(self):
        value = self.manifest(); value["seed_rules"]["planner_seed_base"] += 1
        with self.assertRaises(RuntimeError): validate_request(value, 0)

    def test_unknown_worker_rejected(self):
        with self.assertRaises(RuntimeError): validate_request(self.manifest(), 9)

    def test_separate_tail_preserves_previous_shards(self):
        value = self.manifest(); value["allowed_new_episode_ids"] = list(range(52,62))
        value["shards"] = [{"instance_id": i, "episode_ids": list(range(52+5*i,57+5*i))} for i in range(2)]
        self.assertEqual(validate_request(value, 0)["episode_ids"], list(range(52,57)))

    def test_logs_drop_outcome_fields_and_handle_split_writes(self):
        target = io.StringIO(); stream = ProgressOnly(target)
        message = json.dumps({"event": "episode", "episode": 24, "final_success": True,
                              "ever_success": False, "total_reward": 9, "sha256": "abc"})
        stream.write(message[:15]); stream.write(message[15:]+"\n")
        stream.write("arbitrary non-JSON model log\n")
        self.assertEqual(json.loads(target.getvalue()), {"event": "episode", "episode": 24, "sha256": "abc"})


if __name__ == "__main__": unittest.main()
