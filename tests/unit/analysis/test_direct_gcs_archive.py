import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/vast'))
from direct_gcs_archive import build_archive, create_session, upload_file, validate_session

BUCKET = 'rgt-jepa-archive-2026'
SESSION = 'https://storage.googleapis.com/upload/storage/v1/b/' + BUCKET + '/o?uploadType=resumable&upload_id=test-secret'


class DirectArchiveTests(unittest.TestCase):
    def test_session_destination_restriction(self):
        validate_session(SESSION, BUCKET)
        for value in (SESSION.replace('https:', 'http:'), SESSION.replace('googleapis.com', 'evil.example'),
                      SESSION.replace(BUCKET, 'another-bucket'), SESSION + '#fragment', SESSION.split('?')[0]):
            with self.assertRaises(ValueError): validate_session(value, BUCKET)

    def test_invalid_cloud_destination_before_credentials(self):
        with patch('direct_gcs_archive.subprocess.check_output') as credentials:
            for uri in ('gs://other/a', 'gs://' + BUCKET + '/other/a', 'https://evil.example/a'):
                with self.assertRaises(ValueError): create_session(uri, 100)
            credentials.assert_not_called()

    def test_archive_no_overwrite_and_exact_members(self):
        import tarfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'result.json').write_bytes(b'{}')
            output = root / 'core-completion-preservation-test'
            with patch('shutil.disk_usage') as disk:
                disk.return_value.free = 20 * 1024**3
                result = build_archive(root, output, ['result.json'], 2)
                with self.assertRaises(FileExistsError): build_archive(root, output, ['result.json'], 2)
            with tarfile.open(result['path']) as archive:
                self.assertEqual(archive.getnames(), ['result.json'])
                self.assertEqual(archive.extractfile('result.json').read(), b'{}')
            self.assertEqual(result['sha256'], hashlib.sha256(Path(result['path']).read_bytes()).hexdigest())

    def test_bad_archive_scope_and_disk_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'result.json').write_bytes(b'{}')
            output = root / 'core-completion-preservation-test'
            for names in (['../result.json'], ['/result.json'], ['result.json', 'result.json']):
                with self.assertRaises(ValueError): build_archive(root, output, names, 2)
            with patch('shutil.disk_usage') as disk:
                disk.return_value.free = 1
                with self.assertRaises(ValueError): build_archive(root, output, ['result.json'], 2)
            self.assertFalse(output.exists())

    def test_upload_content_binding_and_redacted_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'archive'; path.write_bytes(b'{}')
            payload = {'path':str(path), 'session':SESSION, 'bucket':BUCKET, 'name':'test',
                       'size':2, 'sha256':hashlib.sha256(b'{}').hexdigest()}
            response = io.BytesIO(json.dumps({'bucket':BUCKET, 'name':'test', 'size':'2', 'generation':'1'}).encode())
            response.status = 200
            with patch('urllib.request.build_opener') as factory:
                factory.return_value.open.return_value = response
                result = upload_file(payload)
                request = factory.return_value.open.call_args.args[0]
                self.assertEqual(request.get_header('Content-length'), '2')
                self.assertEqual(result['generation'], '1')
            with patch('urllib.request.build_opener') as factory:
                factory.return_value.open.side_effect = RuntimeError(SESSION)
                with self.assertRaises(RuntimeError) as raised: upload_file(payload)
                self.assertNotIn('test-secret', str(raised.exception))
            with self.assertRaises(ValueError): upload_file({**payload, 'sha256':'0' * 64})


if __name__ == '__main__':
    unittest.main()
