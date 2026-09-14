import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'analysis/mechanism'))
import common


class EnergySchemaTests(unittest.TestCase):
    def frame(self, energy):
        record = {'calls': [{'horizon': 6, 'candidates': 300, 'energy': energy}]}
        with patch.object(common, 'iter_records', return_value=iter([(0, 'coupling_only', record)])):
            return common.energy_frame(Path('.'), tasks=('reach',), arms=('coupling_only',)).iloc[0]

    def test_real_mean_schema(self):
        row = self.frame({'candidates': 300, 'requested_squared_l2_mean': 2.,
                          'realized_squared_l2_mean': 2.001})
        self.assertEqual(row.requested_sq_l2_sum, 600.)
        self.assertAlmostEqual(row.realized_sq_l2_sum, 600.3)
        self.assertTrue(np.isnan(row.edited_candidates))

    def test_legacy_sum_schema(self):
        row = self.frame({'edited_candidates': 300, 'requested_squared_l2_sum': 600.,
                          'realized_squared_l2_sum': 601.})
        self.assertEqual(row.realized_sq_l2_sum, 601.)
        self.assertEqual(row.edited_candidates, 300)

    def test_mismatched_population_rejected(self):
        with self.assertRaises(ValueError):
            self.frame({'candidates': 20, 'requested_squared_l2_mean': 2.})


if __name__ == '__main__':
    unittest.main()
