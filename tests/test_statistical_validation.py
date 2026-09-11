"""Small exact identities and published planning fixture; no GPU or new data."""
import copy
import json
from pathlib import Path
import unittest

from offline_study.statistical_validation import (
    audit_core, binomial_cdf, exact_binomial_interval, holm,
    paired_exact_p, paired_sample_size,
)


class StatisticalValidationTests(unittest.TestCase):
    def test_cdf_known_values(self):
        self.assertAlmostEqual(binomial_cdf(1, 2, .5), .75)
        self.assertEqual(binomial_cdf(-1, 2, .5), 0)
        self.assertEqual(binomial_cdf(2, 2, .5), 1)

    def test_zero_success_interval_is_not_degenerate(self):
        low, high = exact_binomial_interval(0, 96)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1 - .025**(1 / 96), places=12)
        self.assertGreater(exact_binomial_interval(0, 96, .05 / 8)[1], high)

    def test_all_success_and_symmetry(self):
        low, high = exact_binomial_interval(96, 96)
        self.assertAlmostEqual(low, .025**(1 / 96), places=12)
        self.assertEqual(high, 1)
        lo, hi = exact_binomial_interval(5, 10)
        self.assertAlmostEqual(lo, 1 - hi, places=12)
        self.assertAlmostEqual(lo, .18708602844739852, places=10)

    def test_binomial_inversion(self):
        for k in (1, 3, 40):
            lo, hi = exact_binomial_interval(k, 96)
            self.assertAlmostEqual(binomial_cdf(k - 1, 96, lo), .975, places=11)
            self.assertAlmostEqual(binomial_cdf(k, 96, hi), .025, places=11)

    def test_invalid_intervals_rejected(self):
        for args in ((-1, 96), (97, 96), (True, 96), (1, 0), (1, 2, 0)):
            with self.assertRaises(ValueError):
                exact_binomial_interval(*args)

    def test_paired_exact_and_holm(self):
        self.assertEqual(paired_exact_p(0, 0), 1)
        self.assertEqual(paired_exact_p(17, 16), 1)
        self.assertAlmostEqual(paired_exact_p(23, 17), .42959050784338615)
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])

    def test_planning_matches_published_medcalc_example(self):
        # MedCalc manual: 20% vs 10% discordance, alpha .05, power .8 -> 234 pairs.
        self.assertEqual(paired_sample_size(.3, .1, family=1), 234)
        self.assertGreater(paired_sample_size(.3, .05), paired_sample_size(.3, .1))
        self.assertGreater(paired_sample_size(.3, .05, family=8), paired_sample_size(.3, .05, family=1))

    def test_planning_invalid_assumptions(self):
        for args in ((.02, .05), (.5, 0), (.5, .05, .4), (.5, .05, .8, 0)):
            with self.assertRaises(ValueError):
                paired_sample_size(*args)

    def test_frozen_core_counts_and_no_mutation(self):
        path = Path(__file__).resolve().parents[1] / 'artifacts/offline_study/core-completion-preservation-20260908-v1/ANALYSIS_REPORT.json'
        if not path.exists():
            self.skipTest('Private retained core artifact not installed')
        report = json.loads(path.read_text())
        original = copy.deepcopy(report)
        checked = audit_core(report)
        self.assertEqual(len(checked), 8)
        self.assertEqual(report, original)
        report['contrasts'][0]['discordant_win_clusters'] += 1
        with self.assertRaises(ValueError):
            audit_core(report)


if __name__ == '__main__':
    unittest.main()
