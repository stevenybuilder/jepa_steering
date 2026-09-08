import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
from upload_pause_archive import connector_part, verify_stream


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

    def test_connector_scratch_only_removed_after_exact_provider_binding(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); archive=root/'source.tar.gz'; archive.write_bytes(b'fixture')
            proof={'archive_bytes':7,'archive_sha256':hashlib.sha256(b'fixture').hexdigest()}
            with patch('upload_pause_archive.shutil.disk_usage') as disk:
                disk.return_value.free=10 << 30
                connector_part(root,archive,proof,0)
            scratch=root/'connector-parts-v1/source.tar.gz.part-0000'
            metadata={'id':'part_1234567890','name':scratch.name,'size':'7',
                      'sha256Checksum':proof['archive_sha256'],'parents':['12r-UzKMTuyYm4wXKl9xPb3r5dcjfwsYr']}
            config=MagicMock(); config.__getitem__.return_value={'token':json.dumps({'access_token':'fixture'})}
            with patch('upload_pause_archive.configparser.ConfigParser',return_value=config), \
                    patch('upload_pause_archive.urllib.request.urlopen',return_value=io.BytesIO(json.dumps({**metadata,'sha256Checksum':'bad'}).encode())):
                with self.assertRaises(ValueError): connector_part(root,archive,proof,0,'part_1234567890')
            self.assertTrue(scratch.exists())
            with patch('upload_pause_archive.configparser.ConfigParser',return_value=config), \
                    patch('upload_pause_archive.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(metadata).encode())):
                connector_part(root,archive,proof,0,'part_1234567890')
            self.assertFalse(scratch.exists())
            self.assertEqual(archive.read_bytes(),b'fixture')
            self.assertTrue((root/'connector-parts-v1/part-0000-uploaded.json').is_file())


if __name__ == '__main__':
    unittest.main()
