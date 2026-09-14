import copy
import json
from pathlib import Path
import tempfile
import unittest

from offline_study.evaluation.behavioral_development import assigned_rows, schedule
from offline_study.tasks.navigation.navigation_coupling_behavior import ARMS
from offline_study.planning.planning_native_smoke import SMOKE_SEED
from offline_study.core.protocol import sha256, write_json
from offline_study.tasks.pusht.pusht_coupling_behavior import INTERACTION, analyze, load_native, make_contract, paired_inputs, validate_engineering, validate_push_records


def record(index, arm="native"):
    row = schedule()[index]
    return {**row, "arm": arm, "seconds": 1.,
        "result": {"initial_sha256": str(index), "goal_sha256": str(index + 100),
            "native_success": False, "native_state_distance": 1., "native_reward": 0.,
            "observed_frames": 31, "elementary_steps": 30, "published_candidate_count": 300,
            "planning_calls": [{"iterations": 30, "returned_model_actions": 6}]},
        "unroll_calls": [[6, 300], [6, 1]] * 30,
        "planned_actions": [[[0.] * 10 for _ in range(6)]],
        "source_segment": {"source_index": index % 4, "trajectory_id": f"row{index % 4}",
            "lineage_group": f"family{index % 4}", "source_pool": "val", "segment_frames": 31,
            "sampled_states_sha256": "a" * 64, "sampled_actions_sha256": "b" * 64,
            "sampled_initial_state": [0., 1.]}}


def cohort():
    return {"evaluation": [{"index": i, "trajectory_id": f"row{i}", "lineage_group": f"family{i}"}
                            for i in range(21)]}


def fit_protocol():
    # Original finite protocol is an existing test input, not measured outcomes.
    path = Path(__file__).resolve().parents[3] / (
        "artifacts/offline_study/utah-durable-20260907/pusht-author-replication-20260907/"
        "fits-v1/bfloat16/pusht/vision_action_coupling/protocol.json")
    if path.is_file():
        return json.loads(path.read_text())
    # Clean-checkout structural fixture. No inference about real fit validity.
    pairs = [{"name": f"contrast{i}", "candidate": ARMS[1 + i % 8], "control": "native"} for i in range(15)]
    return {"category": "vision_action_coupling", "arms": [{"name": a, "edits": []}
            for a in (*ARMS, "zero_dose")], "primary_contrasts": pairs}


class PushTCouplingTests(unittest.TestCase):
    def test_registry_preserves_all_arms_and_96_total_without_offline_gate(self):
        fit = fit_protocol()
        original = copy.deepcopy(fit)
        protocol = make_contract({}, fit, {"example": "binding"})
        self.assertEqual(fit, original)
        self.assertEqual(protocol["arms"], list(ARMS))
        self.assertEqual(len(protocol["episodes"]), 96)
        self.assertEqual(protocol["analysis"]["interval_family_size"], 16)
        self.assertFalse(protocol["candidate_admission_requires_offline_significance"])
        self.assertFalse(protocol["fresh_confirmation"])
        self.assertEqual(protocol["source_arms"], fit["arms"])
        changed = copy.deepcopy(fit); changed["arms"].pop()
        with self.assertRaisesRegex(ValueError, "original full"):
            make_contract({}, changed, {})
        changed = copy.deepcopy(fit); changed["support_operator"] = {}
        with self.assertRaisesRegex(ValueError, "original full"):
            make_contract({}, changed, {})

    def test_pairing_rejects_family_segment_goal_and_stream_changes(self):
        base = record(0)
        candidate = copy.deepcopy(base); candidate["arm"] = "joint"
        paired_inputs(base, candidate)
        for key in ("lineage_group", "sampled_states_sha256", "sampled_actions_sha256", "trajectory_id"):
            changed = copy.deepcopy(candidate); changed["source_segment"][key] = "changed"
            with self.assertRaisesRegex(ValueError, "source segment/family"):
                paired_inputs(base, changed)
        changed = copy.deepcopy(candidate); changed["result"]["goal_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "initial/goal"):
            paired_inputs(base, changed)
        changed = copy.deepcopy(candidate); changed["local_seed"] += 1
        with self.assertRaisesRegex(ValueError, "stream/episode"):
            paired_inputs(base, changed)

    def test_actual_family_membership_full_actions_and_budget_are_required(self):
        row = record(0)
        validate_push_records([row], [schedule()[0]], cohort())
        mutations = [lambda r: r["source_segment"].update(lineage_group="unknown"),
                     lambda r: r["source_segment"].update(segment_frames=30),
                     lambda r: r["planned_actions"][0][0].pop(),
                     lambda r: r["unroll_calls"].pop()]
        for mutation in mutations:
            changed = copy.deepcopy(row); mutation(changed)
            with self.assertRaises((ValueError, RuntimeError)):
                validate_push_records([changed], [schedule()[0]], cohort())

    def test_analysis_clusters_original_families_not_96_crops(self):
        panels = {arm: [record(i, arm) for i in range(96)] for arm in ARMS}
        for row in panels["joint"]:
            row["result"]["native_success"] = row["source_segment"]["source_index"] == 0
        protocol = make_contract({}, fit_protocol(), {})
        result = analyze(panels, protocol)
        self.assertEqual(len(result["contrasts"]), 16)
        summary = result["arm_summaries"]["joint"]
        self.assertEqual(summary["source_families"], 4)
        self.assertEqual(summary["successes"], 24)
        self.assertEqual(summary["success_percent"], 25.)
        self.assertEqual(result["source_family_draw_counts"], {f"family{i}": 24 for i in range(4)})
        self.assertTrue(all(r["source_families"] == 4 for r in result["contrasts"]))
        self.assertFalse(result["fresh_confirmation"])
        self.assertEqual(result, analyze(panels, protocol))
        panels["joint"].pop()
        with self.assertRaisesRegex(ValueError, "complete nine-arm"):
            analyze(panels, protocol)

    def test_complete_native_reference_rejects_missing_or_tampered_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "freeze").mkdir()
            write_json(root / "freeze/protocol.json", {"test": True})
            native = {"source_sha256": "old_source"}
            for rank in range(8):
                shard = root / "native" / f"shard-{rank}"
                shard.mkdir(parents=True)
                expected = assigned_rows(schedule(), [rank])
                write_json(shard / "protocol.json", {"freeze_sha256": sha256(root / "freeze/protocol.json"),
                    "source_sha256": "old_source", "task": "pusht", "arm": "native", "logical_ranks": [rank],
                    "expected_episodes": expected, "fresh_confirmation": False})
                hashes = {}
                for row in expected:
                    name = f"episode-{row['episode']:03d}.json"
                    write_json(shard / name, record(row["episode"])); hashes[name] = sha256(shard / name)
                write_json(shard / "report.json", {"status": "native_pusht_planning_replication_shard_complete",
                    "task": "pusht", "arm": "native", "episodes": 12, "parameters_unchanged": True,
                    "fresh_confirmation": False, "protocol_sha256": sha256(shard / "protocol.json"),
                    "episode_files_sha256": hashes})
                write_json(shard / "DONE.json", {"report_sha256": sha256(shard / "report.json")})
            rows, bindings = load_native(root, native, cohort())
            self.assertEqual(len(rows), 96); self.assertEqual(len(bindings), 8)
            partial, _ = load_native(root, native, cohort(), [2])
            self.assertEqual([r["episode"] for r in partial], list(range(24, 36)))
            (root / "native/shard-7/episode-095.json").write_text("{}")
            # A complete assigned stream can run while a different native stream
            # is still unfinished. Full-panel analysis still rejects that input.
            self.assertEqual(len(load_native(root, native, cohort(), [2])[0]), 12)
            with self.assertRaisesRegex(ValueError, "episode changed"):
                load_native(root, native, cohort())

    def test_engineering_requires_all_eleven_and_exact_native_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = record(0)
            names = ("native", "native", "zero_dose") + ARMS[1:]
            launch = {"freeze_sha256": "freeze", "engineering_only": True, "device_uuid": "gpu-test"}
            write_json(root / "protocol.json", launch)
            def save(mutate=False):
                hashes = {}
                for i, arm in enumerate(names):
                    row = copy.deepcopy(smoke)
                    row.update(arm=arm, episode=0, logical_rank=0, local_seed=SMOKE_SEED,
                        environment_seed=SMOKE_SEED, source_parity_candidate_counts=[1, 300])
                    if mutate and arm == "zero_dose":
                        row["planned_actions"][0][0][0] = .1
                    name = f"engineering-{i:02d}.json"
                    write_json(root / name, row); hashes[name] = sha256(root / name)
                write_json(root / "report.json", {"status": "complete_pusht_coupling_engineering", "episodes": 11,
                    "source_sha256": "source", "freeze_sha256": "freeze", "device_uuid": "gpu-test",
                    "protocol_sha256": sha256(root / "protocol.json"), "parameters_unchanged": True,
                    "fresh_confirmation": False, "scientific_efficacy_measurement": False, "episode_files_sha256": hashes})
                write_json(root / "DONE.json", {"report_sha256": sha256(root / "report.json")})
            save()
            validate_engineering(root, "freeze", "source", smoke)
            save(mutate=True)
            with self.assertRaisesRegex(ValueError, "actions or outcomes differ"):
                validate_engineering(root, "freeze", "source", smoke)


if __name__ == "__main__":
    unittest.main()
