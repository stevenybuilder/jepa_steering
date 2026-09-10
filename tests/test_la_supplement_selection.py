"""Local inventory fixtures; never contact a rental or Google."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
from preserve_la_supplement import INVENTORY


class SupplementSelectionTests(unittest.TestCase):
    def test_results_logs_vendor_and_old_code_included_cache_inventoried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['jepa-runtime/fixed-behavior-queue.log', 'jepa-runtime/hmm-old/code.py',
                     'jepa-runtime/core/result.json', 'jepa-runtime/core-priority-archive-20260908-v1/copy.gz',
                     'jepa-runtime/fixed-response-assets-20260908-v1/metaworld/data/train.parquet',
                     'jepa_steering/vendor/jepa-wms/app/custom.py',
                     'jepa_steering/vendor/jepa-wms/.git/config', 'jepa-runtime/__pycache__/a.pyc']
            for name in names:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fixture')
            code = INVENTORY.replace("pathlib.Path('/workspace')", f'pathlib.Path({directory!r})')
            result = json.loads(subprocess.check_output([sys.executable, '-c', code],
                input=json.dumps(['core/result.json']), text=True))
            self.assertEqual(set(result['files']), {names[0], names[1], names[5]})
            self.assertEqual(set(result['released_dataset_cache_retained_on_source_volume']), {names[4]})
            self.assertFalse(result['skipped_symlinks'])

    def test_credential_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'jepa-runtime').mkdir()
            (root / 'jepa-runtime/.env').write_text('fixture')
            code = INVENTORY.replace("pathlib.Path('/workspace')", f'pathlib.Path({directory!r})')
            result = subprocess.run([sys.executable, '-c', code], input='[]', text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('fixture', result.stdout)


if __name__ == '__main__':
    unittest.main()
