import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).parents[1] / "scripts/geometry_map/capture_pusht_calibration.py"
SPEC = importlib.util.spec_from_file_location("capture_pusht_calibration", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class CalibrationTests(unittest.TestCase):
    def manifest(self):
        return {"dataset_split": "train", "rows": M.frozen_rows(), "checkpoint_sha256": M.CHECKPOINT_SHA}

    def test_split_is_source_disjoint(self):
        rows = M.frozen_rows()
        self.assertEqual([sum(r["split"] == s for r in rows) for s in ("fit", "validation", "confirmation")], [100, 25, 25])
        self.assertEqual([r["source_id"] for r in rows], list(range(150)))

    def test_rejects_held_or_replaced_sources(self):
        for key, value in (("dataset_split", "val"), ("checkpoint_sha256", "bad")):
            m = self.manifest(); m[key] = value
            with self.assertRaises(ValueError): M.validate_manifest(m)
        m = self.manifest(); m["rows"][0]["source_id"] = 150
        with self.assertRaises(ValueError): M.validate_manifest(m)

    def test_exact_alignment(self):
        M.validate_manifest(self.manifest())
        self.assertEqual(M.INDICES, (0, 5, 10, 15, 20, 25, 30))

    def test_q_uses_periodic_angle(self):
        try: import torch
        except ImportError: self.skipTest("Torch unavailable locally; run native CPU test remotely")
        x = torch.tensor([[10., 20., 30., 40., 0., 90., 91.]])
        self.assertTrue(torch.equal(M.physical_q(x), torch.tensor([[10., 20., 30., 40., 0., 1.]])))


if __name__ == "__main__": unittest.main()
