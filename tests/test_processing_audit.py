import importlib.util
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / 'scripts/vast/profile_native_forecast.py'
spec = importlib.util.spec_from_file_location('processing_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProcessingAuditTests(unittest.TestCase):
    def test_precision_difference_not_hidden(self):
        native = {'visual': torch.tensor([1., 1.23456])}
        self.assertTrue(audit.compare(native, native)['visual']['bitwise_equal'])
        other = {'visual': native['visual'].to(torch.bfloat16)}
        result = audit.compare(native, other)['visual']
        self.assertFalse(result['bitwise_equal'])
        self.assertGreater(result['max_absolute_error'], 0.)

    def test_invalid_forecast_rejected(self):
        native = {'visual': torch.ones(2)}
        for other in ({'other': torch.ones(2)}, {'visual': torch.ones(3)},
                      {'visual': torch.tensor([float('nan'), 0.])}):
            with self.assertRaises(ValueError):
                audit.compare(native, other)

    def test_hash_includes_dtype_shape_and_values(self):
        value = torch.tensor([[1., 2.]])
        self.assertEqual(audit.tensor_hash(value), audit.tensor_hash(value.clone()))
        self.assertNotEqual(audit.tensor_hash(value), audit.tensor_hash(value.flatten()))
        self.assertNotEqual(audit.tensor_hash(value), audit.tensor_hash(value.to(torch.bfloat16)))


if __name__ == '__main__':
    unittest.main()
