import sys
import tempfile
import unittest
from pathlib import Path

from offline_study import VENDOR_COMMIT
from offline_study.models.vendor import VENDOR_SNAPSHOT_SHA256, use_vendor


class VendorSnapshotTests(unittest.TestCase):
    def test_verified_snapshot_markers_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ".source-commit").write_text(VENDOR_COMMIT + "\n")
            (path / ".source-archive-sha256").write_text(VENDOR_SNAPSHOT_SHA256 + "\n")
            self.assertEqual(use_vendor(path), path.resolve())
            self.assertEqual(sys.path.pop(0), str(path.resolve()))

    def test_snapshot_markers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ".source-commit").write_text(VENDOR_COMMIT + "\n")
            (path / ".source-archive-sha256").write_text("0" * 64 + "\n")
            with self.assertRaisesRegex(RuntimeError, "archive SHA256 marker is invalid"):
                use_vendor(path)


if __name__ == "__main__":
    unittest.main()
