"""Input-only safeguards; never evaluate a protected model outcome."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/prepare_protected_inputs.py'
SPEC = importlib.util.spec_from_file_location('protected_preparation', SCRIPT)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class PreparationSafetyTests(unittest.TestCase):
    def test_family_hashes_cover_historical_shapes_and_precision(self):
        state = [101., 202., 303., 204., 1.5, 0., 0.]
        expected = {
            'pusht:initial-state:' + hashlib.sha256(
                np.asarray(state[:length], dtype=dtype).tobytes()).hexdigest()
            for length in (5, 7) for dtype in ('float32', 'float64')
        }
        self.assertEqual(prep.family_hashes(state), expected)

    def test_array_hash_binds_shape_and_dtype(self):
        values = np.arange(6, dtype='float32')
        self.assertNotEqual(prep.array_hash(values), prep.array_hash(values.reshape(2, 3)))
        self.assertNotEqual(prep.array_hash(values), prep.array_hash(values.astype('float64')))

    def test_changed_protocol_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep.write_json(root / 'protocol.json', {'changed': True})
            prep.write_json(root / 'FROZEN.json', {'protocol_sha256': 'not-the-current-hash'})
            with patch.object(prep, 'OUT', root):
                with self.assertRaisesRegex(ValueError, 'Input protocol changed'):
                    prep.check()

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                prep.fresh_dir(Path(directory))


if __name__ == '__main__':
    unittest.main()
