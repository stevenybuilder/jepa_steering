import tempfile
import unittest
from pathlib import Path

from offline_study.evaluation.checkpoint.author_summary import verified_analysis
from offline_study.core.protocol import sha256, write_json


class SummaryTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "window_metrics.json"
        write_json(data, [])
        report = {"status": "verified_full_primary_author_split_replication_complete", "task": "mw-reach",
                  "precision": "bfloat16", "category": "operator_rank", "rollout_trajectories": 33,
                  "coverage": {"complete_author_split": True}, "fresh_confirmation": False,
                  "input_sha256": {str(data): sha256(data)}}
        write_json(root / "report.json", report)
        write_json(root / "DONE.json", {"report_sha256": sha256(root / "report.json")})
        return report

    def test_verified_report_and_underlying_evidence_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            report, _ = verified_analysis(root, "reach", "bfloat16", "operator_rank")
            self.assertEqual(report["rollout_trajectories"], 33)
            write_json(root / "window_metrics.json", ["changed"])
            with self.assertRaisesRegex(ValueError, "measurement changed"):
                verified_analysis(root, "reach", "bfloat16", "operator_rank")

    def test_partial_pool_cannot_claim_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.fixture(root)
            report["rollout_trajectories"] = 29
            write_json(root / "report.json", report)
            write_json(root / "DONE.json", {"report_sha256": sha256(root / "report.json")})
            with self.assertRaisesRegex(ValueError, "complete authorized"):
                verified_analysis(root, "reach", "bfloat16", "operator_rank")
