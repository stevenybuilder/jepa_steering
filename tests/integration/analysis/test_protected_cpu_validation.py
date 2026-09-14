"""Small CPU tests of conversion and immutable validation safeguards."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/validate_protected_inputs_cpu.py'
SPEC = importlib.util.spec_from_file_location('cpu_validation', SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ValidationTests(unittest.TestCase):
    def test_native_translation_and_gripper_conversion(self):
        states = np.zeros((5, 7), dtype='float64')
        states[:, 0] = np.arange(5) * 0.02
        states[:, 6] = np.arange(5) * 0.1
        actions = mod.upstream_pose_converter()(states)
        np.testing.assert_allclose(actions[:, 0], 0.02)
        np.testing.assert_allclose(actions[:, 1:6], 0)
        np.testing.assert_allclose(actions[:, 6], 0.1)

    def test_hash_includes_shape(self):
        a = np.arange(4, dtype='float32')
        self.assertNotEqual(mod.array_hash(a), mod.array_hash(a.reshape(2, 2)))

    def test_changed_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mod.write_json(root / 'protocol.json', {})
            mod.write_json(root / 'FROZEN.json', {'protocol_sha256': 'different'})
            with patch.object(mod, 'OUT', root):
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    mod.check()


if __name__ == '__main__':
    unittest.main()
