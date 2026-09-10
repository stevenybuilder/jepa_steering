"""Destructive closeout is tested with mocked provider/Drive calls only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/vast"
sys.path.insert(0, str(SCRIPTS))
import finish_decision_diagnostic as closeout


class DecisionCloseoutTests(unittest.TestCase):
    def exercise(self, drive_failure):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            local = base / "run"
            local.mkdir()
            (base / "reports").mkdir()
            analysis = {"scenarios": 192, "protocol_sha256": "p",
                "contrasts": [{"task": "reach", "treatment": "fixed_rank4", "reference": "native",
                    "simultaneous_95_interval_m": [-.1, .1], "mean_distance_gain_m": 0., "changed_choices": 4} for _ in range(8)],
                "rankings": [{"task": "reach", "arm": "native", "mean_spearman": .9,
                              "mean_top10_overlap": .8, "changed_from_native": 4} for _ in range(10)]}
            (local / "analysis.json").write_text(json.dumps(analysis))
            def command(argv, timeout=180):
                if argv[0] == "ssh":
                    if "decision_preserve.py pack" in argv[-1]:
                        return json.dumps({"sha256": "h", "bytes": 100})
                    return "COMPLETE"
                if "destroy" in argv:
                    events.append("destroy")
                    self.assertEqual(argv[3], "50531754")
                    return json.dumps({"success": True})
                return json.dumps([{"id": 50125440}, {"id": 50205763}])
            def drive(*a, **kw):
                events.append("drive_full_readback")
                self.assertTrue(kw["stream"])
                if drive_failure:
                    raise ValueError("Drive bytes changed")
                return {"verified": True, "full_byte_readback": True}
            with patch.multiple(closeout, ROOT=base / "closeout", LOCAL=local, PROJECT=base), \
                 patch.object(closeout, "authority", return_value={"id": 50531754}), \
                 patch.object(closeout, "command", side_effect=command), \
                 patch.object(closeout, "mirror"), \
                 patch.object(closeout.subprocess, "run"), \
                 patch.object(closeout, "verify_members", return_value={"sha256": "h", "bytes": 100}), \
                 patch.object(closeout, "upload_direct", return_value={"id": "drive-test"}), \
                 patch.object(closeout, "verify_drive", side_effect=drive):
                if drive_failure:
                    with self.assertRaises(ValueError):
                        closeout.main()
                    self.assertNotIn("destroy", events)
                    self.assertFalse((base / "closeout/DONE.json").exists())
                else:
                    closeout.main()
                    self.assertEqual(events, ["drive_full_readback", "destroy"])
                    self.assertTrue((base / "closeout/DONE.json").exists())

    def test_failed_drive_verification_never_destroys_source(self):
        self.exercise(True)

    def test_only_new_worker_deleted_after_drive_readback(self):
        self.exercise(False)

    def test_cross_platform_float_comparison_does_not_change_integers(self):
        self.assertTrue(closeout.close_numeric({"x": .1}, {"x": .1 + 1e-14}))
        self.assertFalse(closeout.close_numeric({"n": 95}, {"n": 96}))
        self.assertFalse(closeout.close_numeric({"x": .1}, {"x": .10001}))


if __name__ == "__main__":
    unittest.main()
