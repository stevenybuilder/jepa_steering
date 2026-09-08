"""Local fixtures only: no Drive, credential, or GPU access."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
reader = importlib.import_module('verify_drive_archive_by_id')


class DriveReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / 'source.tar.gz'
        data = b'preserved scientific output'
        with tarfile.open(self.archive, 'w:gz') as output:
            member = tarfile.TarInfo('report.json'); member.size = len(data)
            output.addfile(member, io.BytesIO(data))
        self.manifest = self.root / 'FILES.json'
        self.manifest.write_text(json.dumps({'report.json': {
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}))
        self.body = self.archive.read_bytes()
        self.metadata = {'id': 'file_1234567890', 'name': self.archive.name,
            'parents': ['folder_1234567890'], 'size': str(len(self.body)),
            'sha256Checksum': hashlib.sha256(self.body).hexdigest()}
        self.config = MagicMock()
        self.config.__getitem__.return_value = {'token': json.dumps({'access_token': 'fixture-only'})}
        self.output = self.root / 'receiving'

    def run_readback(self, metadata=None, body=None):
        metadata = self.metadata if metadata is None else metadata
        body = self.body if body is None else body
        responses = [io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(body)]
        with patch.object(reader.configparser, 'ConfigParser', return_value=self.config), \
                patch.object(reader.urllib.request, 'urlopen', side_effect=responses) as network, \
                patch.object(reader.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 10 << 30})()):
            reader.verify('file_1234567890', 'folder_1234567890', self.archive, self.manifest, self.output)
        return network

    def test_all_bytes_and_members_verified_with_no_path_lookup(self):
        network = self.run_readback()
        report = json.loads((self.output / 'VERIFIED.json').read_text())
        self.assertEqual(report['files'], 1)
        self.assertEqual(report['archive_sha256'], self.metadata['sha256Checksum'])
        self.assertTrue(report['no_local_or_worker_deletions'])
        self.assertEqual(network.call_count, 2)
        self.assertTrue(network.call_args_list[1].args[0].full_url.endswith('?alt=media'))
        self.assertNotIn('fixture-only', (self.output / 'VERIFIED.json').read_text())

    def test_provider_parent_or_checksum_mismatch_fails_without_success(self):
        changed = {**self.metadata, 'parents': ['different_parent']}
        with self.assertRaises(RuntimeError): self.run_readback(metadata=changed)
        self.assertTrue((self.output / 'FAILED.json').is_file())
        self.assertFalse((self.output / 'VERIFIED.json').exists())
        self.assertFalse((self.output / self.archive.name).exists())

    def test_truncated_download_preserved_but_never_verified(self):
        with self.assertRaises(RuntimeError): self.run_readback(body=self.body[:-1])
        self.assertTrue((self.output / self.archive.name).is_file())
        self.assertFalse((self.output / 'VERIFIED.json').exists())

    def test_invalid_id_rejected_before_reading_credentials(self):
        with patch.object(reader.configparser, 'ConfigParser') as config:
            with self.assertRaises(ValueError):
                reader.verify('https://invalid', 'folder_1234567890', self.archive, self.manifest, self.output)
            config.assert_not_called()

    def test_never_overwrite_previous_readback(self):
        self.run_readback()
        before = (self.output / 'VERIFIED.json').read_bytes()
        with self.assertRaises(FileExistsError): self.run_readback()
        self.assertEqual((self.output / 'VERIFIED.json').read_bytes(), before)

    def split_spec(self):
        bodies = [self.body[:50], self.body[50:]]
        parts = [{'name': 'source.tar.gz.part-' + str(i), 'bytes': len(body),
                  'sha256': hashlib.sha256(body).hexdigest(), 'drive_file_id': 'partfile_00000' + str(i)}
                 for i, body in enumerate(bodies)]
        spec = {'archive_name': self.archive.name, 'archive_bytes': len(self.body),
                'archive_sha256': self.metadata['sha256Checksum'],
                'member_manifest_sha256': reader.digest(self.manifest), 'parts_in_join_order': parts}
        path = self.root / 'PARTS.json'; path.write_text(json.dumps(spec))
        return path, spec, bodies

    def test_split_readback_rejoins_all_bytes_and_verifies_members(self):
        path, spec, bodies = self.split_spec(); responses = []
        for part, body in zip(spec['parts_in_join_order'], bodies):
            metadata = {'id': part['drive_file_id'], 'name': part['name'],
                'size': str(part['bytes']), 'sha256Checksum': part['sha256'],
                'parents': ['folder_1234567890']}
            responses += [io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(body)]
        with patch.object(reader.configparser, 'ConfigParser', return_value=self.config), \
                patch.object(reader.urllib.request, 'urlopen', side_effect=responses), \
                patch.object(reader.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 10 << 30})()):
            reader.verify_parts(path, 'folder_1234567890', self.archive, self.manifest, self.output)
        report = json.loads((self.output / 'VERIFIED.json').read_text())
        self.assertEqual(len(report['parts']), 2)
        self.assertEqual((self.output / self.archive.name).read_bytes(), self.body)

    def test_split_duplicate_identity_rejected_before_credentials(self):
        path, spec, _ = self.split_spec()
        spec['parts_in_join_order'][1]['drive_file_id'] = spec['parts_in_join_order'][0]['drive_file_id']
        path.write_text(json.dumps(spec))
        with patch.object(reader.configparser, 'ConfigParser') as config:
            with self.assertRaises(ValueError):
                reader.verify_parts(path, 'folder_1234567890', self.archive, self.manifest, self.output)
            config.assert_not_called()

    def test_split_stream_verifies_without_download_spool(self):
        path, spec, bodies = self.split_spec(); responses = []
        for part, body in zip(spec['parts_in_join_order'], bodies):
            metadata = {'id':part['drive_file_id'],'name':part['name'],'size':str(part['bytes']),
                        'sha256Checksum':part['sha256'],'parents':['folder_1234567890']}
            responses += [io.BytesIO(json.dumps(metadata).encode()),io.BytesIO(body)]
        with patch.object(reader.configparser,'ConfigParser',return_value=self.config), \
                patch.object(reader.urllib.request,'urlopen',side_effect=responses), \
                patch.object(reader.shutil,'disk_usage',return_value=type('Disk',(),{'free':2 << 20})()):
            reader.verify_parts(path,'folder_1234567890',self.archive,self.manifest,self.output,stream=True)
        report=json.loads((self.output/'VERIFIED.json').read_text())
        self.assertTrue(report['bounded_stream_no_download_spool'])
        self.assertEqual(len(report['parts']),2)
        self.assertFalse((self.output/self.archive.name).exists())

    def test_stream_corruption_fails(self):
        _,spec,bodies=self.split_spec(); responses=[]
        for part,body in zip(spec['parts_in_join_order'],bodies):
            metadata={'id':part['drive_file_id'],'name':part['name'],'size':str(part['bytes']),
                      'sha256Checksum':part['sha256'],'parents':['folder_1234567890']}
            responses += [io.BytesIO(json.dumps(metadata).encode()),io.BytesIO(body[:-1])]
        def request(*args): return responses.pop(0)
        source=reader.PartsReader(spec['parts_in_join_order'],'folder_1234567890',request)
        with self.assertRaises(ValueError): source.read(4 << 20)
        source.close()
        self.assertFalse(source.verified)


if __name__ == '__main__':
    unittest.main()
