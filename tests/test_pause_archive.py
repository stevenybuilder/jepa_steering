import hashlib
import io
from pathlib import Path
import sys
import tarfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
from upload_pause_archive import verify_stream


class PauseArchiveTests(unittest.TestCase):
    def fixture(self, duplicate=False):
        data = b'completed and partial records remain distinct\n'
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for _ in range(2 if duplicate else 1):
                info = tarfile.TarInfo('result.json'); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        raw = stream.getvalue()
        manifest = {'result.json': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}
        return raw, manifest

    def test_full_stream_and_members(self):
        raw, manifest = self.fixture()
        verify_stream(io.BytesIO(raw), hashlib.sha256(raw).hexdigest(), len(raw), manifest)

    def test_wrong_archive_hash(self):
        raw, manifest = self.fixture()
        with self.assertRaises(ValueError):
            verify_stream(io.BytesIO(raw), '0' * 64, len(raw), manifest)

    def test_wrong_member_hash(self):
        raw, manifest = self.fixture(); manifest['result.json']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            verify_stream(io.BytesIO(raw), hashlib.sha256(raw).hexdigest(), len(raw), manifest)

    def test_omitted_member(self):
        raw, manifest = self.fixture(); manifest['missing.json'] = manifest['result.json']
        with self.assertRaises(ValueError):
            verify_stream(io.BytesIO(raw), hashlib.sha256(raw).hexdigest(), len(raw), manifest)

    def test_duplicate_member(self):
        raw, manifest = self.fixture(duplicate=True)
        with self.assertRaises(ValueError):
            verify_stream(io.BytesIO(raw), hashlib.sha256(raw).hexdigest(), len(raw), manifest)


if __name__ == '__main__':
    unittest.main()
