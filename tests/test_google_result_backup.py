import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("google_backup", Path(__file__).resolve().parents[1] /
    "scripts/vast/backup_results_to_google.py")
BACKUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKUP)


class GoogleBackupTests(unittest.TestCase):
    def test_stream_readback_members_and_archive_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.tar.gz"
            data = b"fixed experiment result\n"
            with tarfile.open(path, "w:gz") as tar:
                member = tarfile.TarInfo("./task/report.json")
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            raw = path.read_bytes()
            command = [sys.executable, "-c", "import sys;sys.stdout.buffer.write(open(sys.argv[1],'rb').read())", str(path)]
            manifest = {"task/report.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
            BACKUP.verify_archive(command, hashlib.sha256(raw).hexdigest(), len(raw), manifest)
            BACKUP.verify_archive(command, hashlib.sha256(raw).hexdigest(), len(raw))
            with self.assertRaises(ValueError):
                BACKUP.verify_archive(command, "0" * 64, len(raw), manifest)
            with self.assertRaises(ValueError):
                BACKUP.verify_archive(command, hashlib.sha256(raw).hexdigest(), len(raw), {})


if __name__ == "__main__":
    unittest.main()
