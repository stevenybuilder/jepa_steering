"""Local preservation-gate regression checks; never contact Vast or Drive."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
import final_preservation_batches as batches
import final_release_receipt as release
import final_storage_release as storage


class FinalPreservationTests(unittest.TestCase):
    def fixture(self, project):
        base = project / 'artifacts/offline_study/final-storage-release-20260910-v1'
        root = base / '50159352'
        source = {'path': 'jepa-runtime/checkpoint-custom.pth', 'bytes': 200 << 20,
                  'mode': '-rw-r--r--', 'mtime': '2026/09/10 01:00:00'}
        old = {'bytes': source['bytes'], 'sha256': hashlib.sha256(b'old').hexdigest(),
               'proof': 'old/VERIFIED.json'}
        plan = {'candidates_needing_backup_or_exact_hash_reconciliation': [],
                'historical_same_path_size_matches_require_final_review': [
                    {**source, 'historical_archive': old}]}
        storage.write(root / 'PRESERVATION_CANDIDATES.json', plan)
        storage.write(root / '20260910T010000Z-inventory.json',
                      {'returncode': 0, 'entries': [source]})
        storage.write(base / 'PRIOR_DRIVE_ARCHIVES_REVERIFIED.json', [old])
        storage.write(root / 'ALL_BATCHES_VERIFIED.json', {})
        storage.write(root / 'SYMLINK_TARGETS.json', [])
        return base, root, source, old

    def test_historical_same_size_backup_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            base, root, _, _ = self.fixture(project)
            with patch.object(release, 'BASE', base), patch.object(release, 'PROJECT', project), \
                    patch.object(release, 'upload_direct') as upload, \
                    patch.object(release, 'verify') as verify:
                with self.assertRaisesRegex(ValueError, 'uncovered source files'):
                    release.prepare(50159352)
                upload.assert_not_called()
                verify.assert_not_called()
            self.assertFalse((root / 'FINAL_RELEASE_COVERAGE.json').exists())
            self.assertFalse((root / 'RELEASE_READY.json').exists())

    def test_large_historical_match_requires_fresh_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            base, _, source, _ = self.fixture(project)
            with patch.object(batches, 'BASE', base):
                plan = batches.prepare(50159352)
            self.assertEqual(plan['files'], 1)
            self.assertEqual(plan['batches'][0]['kind'], 'file')
            self.assertEqual(plan['batches'][0]['files'][0]['path'], source['path'])

    def test_changed_existing_coverage_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            base, root, source, old = self.fixture(project)
            storage.write(root / 'batch-000/DRIVE_VERIFIED.json', {
                'verified': True, 'full_byte_readback': True,
                'manifest': {source['path']: {'bytes': source['bytes'],
                                             'sha256': hashlib.sha256(b'new').hexdigest()}}})
            stale = {'historical_sha256': old['sha256']}
            storage.write(root / 'FINAL_RELEASE_COVERAGE.json', stale)
            with patch.object(release, 'BASE', base), patch.object(release, 'PROJECT', project), \
                    patch.object(release, 'upload_direct') as upload:
                with self.assertRaisesRegex(ValueError, 'Existing coverage differs'):
                    release.prepare(50159352)
                upload.assert_not_called()
            self.assertEqual(json.loads((root / 'FINAL_RELEASE_COVERAGE.json').read_text()), stale)

    def test_unsupported_source_rejected_before_sdk_or_credentials(self):
        for instance, path in ((0, '/workspace/results'),
                               (50159352, '/workspace/../secrets'),
                               (50159352, '/etc/passwd')):
            with self.subTest(instance=instance, path=path), patch.object(storage, 'client') as api:
                with self.assertRaises(ValueError):
                    storage.command(instance, path)
                api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
