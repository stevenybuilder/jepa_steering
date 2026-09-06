import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'geometry_map'))
from summarize_intervention_rank_margins import rank_margins


class RankMarginTests(unittest.TestCase):
    def test_identity(self):
        r = rank_margins([1., 2., 3.], [1., 2., 3.])
        self.assertTrue(r['strict_choice_preservation_bound'])
        self.assertEqual(r['maximum_positive_gap_fraction_closed'], 0)

    def test_shared_shift_cannot_change_choice(self):
        r = rank_margins([1., 2., 3.], [-9., -8., -7.])
        self.assertFalse(r['choice_changed'])
        self.assertEqual(r['action_contrast_shift_rms'], 0)
        self.assertFalse(r['strict_choice_preservation_bound'])  # Sufficient, not necessary.

    def test_crossing(self):
        r = rank_margins([1., 2., 3.], [1., .5, 3.])
        self.assertEqual(r['edited_choice'], 1)
        self.assertEqual(r['maximum_positive_gap_fraction_closed'], 1.5)
        self.assertLess(r['minimum_remaining_gap'], 0)

    def test_tie(self):
        r = rank_margins([1., 1., 3.], [1., 1., 3.])
        self.assertEqual(r['edited_choice'], 0)
        self.assertIsNone(r['competitors'][0]['fraction_of_positive_gap_closed'])
        self.assertFalse(r['strict_choice_preservation_bound'])

    def test_invalid_costs(self):
        with self.assertRaises(ValueError):
            rank_margins([0., float('nan')], [0., 1.])


if __name__ == '__main__':
    unittest.main()
