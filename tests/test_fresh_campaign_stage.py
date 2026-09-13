import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("stage", Path(__file__).resolve().parents[1] / "scripts/vast/fresh_campaign_stage.py")
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class StagingVerification(unittest.TestCase):
    def test_receiving_hash_and_size_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset").write_bytes(b"verified")
            metadata = {"files": {"asset": {"bytes": 8, "sha256": stage.sha(root / "asset")}}}
            (root / "STAGING.json").write_text(json.dumps(metadata))
            stage.verify(root)
            (root / "asset").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                stage.verify(root)

    def test_symlink_is_not_receiving_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").write_bytes(b"verified")
            (root / "asset").symlink_to(root / "real")
            (root / "STAGING.json").write_text(json.dumps({"files": {"asset": {
                "bytes": 8, "sha256": stage.sha(root / "real")}}}))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                stage.verify(root)


if __name__ == "__main__":
    unittest.main()
