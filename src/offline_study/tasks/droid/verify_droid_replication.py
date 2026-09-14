"""Verify a complete released-checkpoint DROID reference, never a robot success rate."""
import argparse
import json
from pathlib import Path

import torch

from offline_study.tasks.droid.droid_contract import action_metrics, checkpoint_score
from offline_study.planning.planning_contract import seed_schedule
from offline_study.core.protocol import sha256, write_json


def verify(root):
    done = json.loads((root / "DONE.json").read_text())
    report = json.loads((root / "report.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if (sha256(root / "report.json") != done["report_sha256"] or
            sha256(root / "protocol.json") != report["protocol_sha256"] or
            report["status"] != "released_droid_native_replication_shard_complete" or
            report["episodes"] != 64 or report["robot_executions"] != 0 or
            protocol["assigned_episodes"] != seed_schedule(1, 64, 8, 3) or
            protocol["fresh_confirmation"] is not False or protocol["native_baseline_only"] is not True):
        raise ValueError("Incomplete, altered or incorrectly labelled DROID reference")
    expected_files = {f"episode-{row['episode']:03d}.json" for row in protocol["assigned_episodes"]}
    if set(report["episode_files_sha256"]) != expected_files:
        raise ValueError("Missing or extra DROID episodes")
    errors, recordings, stimuli = [], set(), set()
    for row in protocol["assigned_episodes"]:
        name = f"episode-{row['episode']:03d}.json"
        if sha256(root / name) != report["episode_files_sha256"][name]:
            raise ValueError("DROID episode checksum changed")
        record = json.loads((root / name).read_text())
        if any(record[k] != v for k, v in row.items()) or record["arm"] != "native":
            raise ValueError("Wrong DROID stream or condition")
        result = record["result"]
        if (result["unroll_calls"] != [[3, 300], [3, 1]] * 15 or
                not result["dummy_success_intentionally_omitted"]):
            raise ValueError("Incomplete CEM or invalid dummy-success claim")
        # Planned values originate as float32; source recorded deltas are float64.
        metrics = action_metrics(torch.tensor(result["planned_actions"], dtype=torch.float32),
                                 torch.tensor(result["recorded_actions"], dtype=torch.float64))
        for metric, value in metrics.items():
            if float(value) != result["metrics"][metric]:
                raise ValueError("Native action metric does not reconstruct exactly")
        sample = result["dataset_sample"]
        raw, segment, offset = sample["raw_frame_indices"], sample["goal_segment_raw_frame_indices"], sample["goal_segment_offset"]
        if len(raw) != 5 or offset not in (0, 1) or segment != raw[offset:offset + 4]:
            raise ValueError("Invalid five-frame/goal-segment trace")
        recordings.add(sample["path"])
        stimuli.add((sample["path"], tuple(segment)))
        errors.append(float(metrics["action_error_xyz"]))
    score = float(checkpoint_score(torch.tensor(errors, dtype=torch.float64)))
    if score != report["official_checkpoint_score_if_complete"]:
        raise ValueError("DROID mean-then-transform score differs")
    return {"status": "complete_native_droid_reference_locally_verified",
        "source_report_sha256": done["report_sha256"], "source_protocol_sha256": report["protocol_sha256"],
        "source_episode_files_sha256": report["episode_files_sha256"],
        "episodes": 64, "sampled_recording_families": len(recordings),
        "unique_recording_and_goal_segments": len(stimuli),
        "mean_xyz_action_error": sum(errors) / len(errors), "official_checkpoint_score": score,
        "score_is_not_a_percentage_success_rate": True, "maximum_possible_score": 80,
        "robot_executions": 0, "intervention_comparison_complete": False,
        "fresh_confirmation": False, "three_training_seed_history_complete": False,
        "paper_16_recording_population_exactly_matched": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", result)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    print(json.dumps({k: v for k, v in result.items() if k != "source_episode_files_sha256"}), flush=True)


if __name__ == "__main__":
    main()
