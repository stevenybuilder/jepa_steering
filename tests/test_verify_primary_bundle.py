from pathlib import Path
import unittest

from offline_study.verify_primary_bundle import ORIGINAL, MW, PUSHT, relocated


class BundlePathTests(unittest.TestCase):
    def test_exact_known_source_roots_are_remapped(self):
        root = Path("/safe/bundle")
        self.assertEqual(relocated(ORIGINAL / "fits-v1/x", root), root / "fits-v1/x")
        self.assertEqual(relocated(MW / "analysis/x", root), root / MW.name / "analysis/x")
        self.assertEqual(relocated(PUSHT / "analysis/x", root), root / PUSHT.name / "analysis/x")

    def test_external_and_traversal_paths_fail_closed(self):
        for path in ("/other/data.json", str(ORIGINAL / "../private.json"), "relative.json"):
            with self.assertRaises(ValueError):
                relocated(path, Path("/safe/bundle"))
