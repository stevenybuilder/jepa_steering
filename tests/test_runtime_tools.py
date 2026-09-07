import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from offline_study.compare import compare_runs
from offline_study.multigpu import aggregate_shards, benchmark_command, normalize_device
from offline_study.protocol import sha256, write_json


def report(execution, rate=2., memory=100, **values):
    return {
        "gpu_benchmark_valid": True,
        "execution": execution,
        "manifest_sha256": "manifest",
        "exposure_registry_sha256": "registry",
        "selection_sha256": "selection",
        "backend": {"checkpoint_sha256": "checkpoint"},
        "harness_source_sha256": "harness",
        "shard_index": 0,
        "num_shards": 1,
        "environment": {"gpu": "GPU", "gpu_capability": [8, 0], "torch": "2.7", "cuda": "12.8"},
        "windows_per_second": rate,
        "peak_reserved_gpu_bytes": memory,
        **values,
    }


class PrecisionComparisonTests(unittest.TestCase):
    def test_compares_identical_windows_and_reports_speedup(self):
        strict = report({
            "precision": "float32", "allow_tf32": False,
            "batch_size": 8, "prefetch_batches": 2,
        })
        candidate = report({
            "precision": "bfloat16", "allow_tf32": False,
            "batch_size": 8, "prefetch_batches": 2,
        }, rate=4., memory=60)
        reference_rows = [{
            "task": "reach", "trajectory_id": "a", "lineage_group": "group-a", "start": 0,
            "metrics": {"visual_mse_h1": 2., "proprio_mse_h1": 1.},
        }]
        candidate_rows = [{
            "task": "reach", "trajectory_id": "a", "lineage_group": "group-a", "start": 0,
            "metrics": {"visual_mse_h1": 2.02, "proprio_mse_h1": 1.},
        }]
        result = compare_runs(strict, reference_rows, candidate, candidate_rows)
        self.assertEqual(result["status"], "precision_candidate_measured_not_adjudicated")
        self.assertAlmostEqual(result["throughput_speedup"], 2.)
        self.assertAlmostEqual(result["peak_reserved_memory_ratio"], .6)
        self.assertAlmostEqual(result["metric_drift"]["visual_mse_h1"]["max_absolute"], .02)

    def test_rejects_non_strict_reference_and_changed_selection(self):
        row = [{"task": "reach", "trajectory_id": "a", "lineage_group": "group-a",
                "start": 0, "metrics": {"m": 1.}}]
        with self.assertRaisesRegex(ValueError, "strict float32"):
            compare_runs(
                report({
                    "precision": "float32", "allow_tf32": True,
                    "batch_size": 8, "prefetch_batches": 2,
                }), row,
                report({
                    "precision": "bfloat16", "allow_tf32": False,
                    "batch_size": 8, "prefetch_batches": 2,
                }), row)
        changed = report({
            "precision": "bfloat16", "allow_tf32": False,
            "batch_size": 8, "prefetch_batches": 2,
        })
        changed["selection_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "selection_sha256"):
            compare_runs(report({
                "precision": "float32", "allow_tf32": False,
                "batch_size": 8, "prefetch_batches": 2,
            }), row, changed, row)


class MultiGpuTests(unittest.TestCase):
    def test_device_and_command_are_explicit(self):
        self.assertEqual(normalize_device("1"), "cuda:1")
        self.assertEqual(normalize_device("cuda:7"), "cuda:7")
        with self.assertRaises(ValueError):
            normalize_device("cpu")
        args = SimpleNamespace(
            output=Path("out"), vendor=Path("vendor"), checkpoint=Path("checkpoint"),
            checkpoint_sha256="abc", manifest=Path("manifest"), exposure_registry=Path("registry"),
            data_root=Path("data"), tasks=["mw-reach"], split="development", max_trajectories=12,
            batch_size=8, warmup=3, precision="float32", prefetch_batches=2,
            cache_queue_depth=2, allow_tf32=False, write_cache=False,
        )
        command = benchmark_command(args, "cuda:1", 1, 2)
        self.assertEqual(command[0:3], [__import__("sys").executable, "-m", "offline_study.benchmark"])
        self.assertIn("cuda:1", command)
        self.assertEqual(command[-4:], ["--num-shards", "2", "--shard-index", "1"])

    def test_aggregate_rejects_overlap_and_writes_valid_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, trajectory in enumerate(("a", "b")):
                child = root / f"shard-{index:03d}"
                child.mkdir()
                child_report = report(
                    {"precision": "float32", "allow_tf32": False},
                    shard_index=index, num_shards=2, windows=1, rollout_trajectories=1,
                    independent_lineage_groups=1,
                    measured_pipeline_seconds=2 + index, peak_reserved_gpu_bytes=100,
                    environment={"device": f"cuda:{index}"},
                )
                config = {"device": f"cuda:{index}", "output": str(child), "shard_index": index,
                          "num_shards": 2, "batch_size": 8}
                selection = [{"trajectory_id": trajectory, "lineage_group": f"group-{trajectory}",
                              "task": "reach"}]
                windows = [{"trajectory_id": trajectory, "lineage_group": f"group-{trajectory}",
                            "task": "reach", "start": 0,
                            "metrics": {"visual_mse_h1": float(index)}}]
                write_json(child / "config.json", config)
                write_json(child / "selection.json", selection)
                write_json(child / "window_metrics.json", windows)
                child_report["selection_sha256"] = sha256(child / "selection.json")
                write_json(child / "report.json", child_report)
                write_json(child / "DONE.json", {
                    "report_sha256": sha256(child / "report.json"),
                    "window_metrics_sha256": sha256(child / "window_metrics.json"),
                    "selection_sha256": sha256(child / "selection.json"),
                })
            result = aggregate_shards(root, 2, wall_seconds=4.)
            self.assertEqual(result["windows"], 2)
            self.assertEqual(result["rollout_trajectories"], 2)
            self.assertEqual(result["independent_lineage_groups"], 2)
            self.assertEqual(result["task_measured_trajectory_counts"], {"reach": 2})
            self.assertAlmostEqual(result["end_to_end_windows_per_second"], .5)


if __name__ == "__main__":
    unittest.main()
