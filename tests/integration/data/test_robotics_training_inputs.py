"""Raw-byte/provenance guards only; fixtures are not native reader/GPU evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from offline_study.data import robotics_training_inputs as inputs


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RoboticsInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'data'
        self.root.mkdir()
        self.manifest = self.base / 'manifest.json'

    def mw_fixture(self):
        files = []
        for name in sorted(inputs.required_files('metaworld')):
            data = name.encode()
            (self.root / name).write_bytes(data)
            path = 'metaworld/data/' + name
            files.append({'path': path, 'bytes': len(data), 'sha256': sha(data),
                'url': 'https://huggingface.co/datasets/facebook/jepa-wms/resolve/' + inputs.DATA_REVISION + '/' + path})
        self.manifest.write_text(json.dumps({'data_revision': inputs.DATA_REVISION, 'files': files}))
        self.addCleanup(patch.stopall)
        patch.dict(inputs.MANIFEST_HASHES, metaworld=inputs.sha256(self.manifest)).start()
        return files

    def pt_fixture(self):
        self.addCleanup(patch.stopall)
        patch.dict(inputs.COUNTS, pusht=(2, 1)).start()
        manifest = {}
        archive = self.base / 'pusht.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for name in sorted(inputs.required_files('pusht')):
                data = name.encode()
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                full = 'pusht_noise/' + name
                manifest[full] = {'bytes': len(data), 'sha256': sha(data)}
                zipped.writestr(full, data)
        self.manifest.write_text(json.dumps(manifest))
        patch.dict(inputs.MANIFEST_HASHES, pusht=inputs.sha256(self.manifest)).start()
        patch.object(inputs, 'PUSHT_ARCHIVE_HASH', inputs.sha256(archive)).start()
        return archive

    def test_production_exact_required_populations(self):
        self.assertEqual(len(inputs.required_files('metaworld')), 126)
        pt = inputs.required_files('pusht')
        self.assertEqual(len(pt), 18718)
        self.assertEqual(sum(n.startswith('train/obses/') for n in pt), 18685)
        self.assertEqual(sum(n.startswith('val/obses/') for n in pt), 21)
        self.assertIn('train/tokens.pth', pt)
        self.assertNotIn('train/shapes.pkl', pt)
        with self.assertRaises(ValueError):
            inputs.required_files('robocasa')

    def test_complete_mw_bytes_do_not_grant_model_or_protected_access(self):
        self.mw_fixture()
        output = self.base / 'receipt'
        report = inputs.verify_inputs('metaworld', self.root, provenance=self.manifest, output=output)
        self.assertEqual(report['files'], 126)
        self.assertEqual(report['status'], 'complete_robotics_raw_inputs_verified')
        for field in ('training_or_validation_authorized', 'native_reader_parity_established',
                      'model_or_data_loader_initialized'):
            self.assertIs(report[field], False)
        self.assertEqual(json.loads((output / 'DONE.json').read_text())['report_sha256'],
                         inputs.sha256(output / 'report.json'))
        with self.assertRaises(FileExistsError):
            inputs.verify_inputs('metaworld', self.root, provenance=self.manifest, output=output)

    def test_no_unbound_or_changed_provenance(self):
        self.mw_fixture()
        self.manifest.write_text(self.manifest.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            inputs.verify_inputs('metaworld', self.root, provenance=self.manifest)

    def test_missing_or_extra_hf_parquet_rejected(self):
        self.mw_fixture()
        path = self.root / 'train-00000-of-00126.parquet'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'parquet population'):
            inputs.verify_inputs('metaworld', self.root, provenance=self.manifest)
        path.write_bytes(b'changed')
        (self.root / 'extra.parquet').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'parquet population'):
            inputs.verify_inputs('metaworld', self.root, provenance=self.manifest)

    def test_changed_raw_file_preserves_failed_receipt(self):
        self.mw_fixture()
        (self.root / 'train-00000-of-00126.parquet').write_bytes(b'changed')
        output = self.base / 'failed'
        with self.assertRaisesRegex(ValueError, 'raw input changed'):
            inputs.verify_inputs('metaworld', self.root, provenance=self.manifest, output=output)
        self.assertTrue((output / 'FAILED.json').is_file())
        self.assertFalse((output / 'DONE.json').exists())

    def test_symlink_and_unsafe_path_rejected(self):
        target = self.base / 'target'
        target.write_bytes(b'value')
        (self.root / 'link').symlink_to(target)
        for name in ('link', '../target', '/absolute', 'a\\b', './target'):
            with self.assertRaises(ValueError):
                inputs.safe_member(self.root, name)

    def test_complete_pt_requires_zip_and_every_raw_video(self):
        archive = self.pt_fixture()
        with self.assertRaisesRegex(ValueError, 'Full pinned ZIP'):
            inputs.verify_inputs('pusht', self.root, provenance=self.manifest)
        report = inputs.verify_inputs('pusht', self.root, provenance=self.manifest, archive=archive)
        self.assertEqual(report['files'], 15)
        self.assertEqual((report['training_rows'], report['validation_rows']), (2, 1))
        (self.root / 'train/obses/episode_001.mp4').unlink()
        with self.assertRaisesRegex(ValueError, 'video population'):
            inputs.verify_inputs('pusht', self.root, provenance=self.manifest, archive=archive)

    def test_pt_pinned_zip_hash_is_required(self):
        archive = self.pt_fixture()
        archive.write_bytes(archive.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'ZIP differs'):
            inputs.verify_inputs('pusht', self.root, provenance=self.manifest, archive=archive)

    def test_optional_unpublished_shapes_cannot_change_native_defaults(self):
        archive = self.pt_fixture()
        (self.root / 'train/shapes.pkl').write_bytes(b'not part of pinned release')
        with self.assertRaisesRegex(ValueError, 'optional shapes must remain absent'):
            inputs.verify_inputs('pusht', self.root, provenance=self.manifest, archive=archive)

    def test_pt_extract_must_match_pinned_zip(self):
        archive = self.pt_fixture()
        (self.root / 'train/states.pth').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'raw input changed'):
            inputs.verify_inputs('pusht', self.root, provenance=self.manifest, archive=archive)

    def test_newly_self_hashed_wrong_zip_member_is_not_accepted(self):
        archive = self.pt_fixture()
        manifest = inputs.bound_manifest('pusht', self.manifest)
        manifest['val/states.pth']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'manifest differs from ZIP'):
            inputs.verify_archive_members(archive, manifest)


if __name__ == '__main__':
    unittest.main()
