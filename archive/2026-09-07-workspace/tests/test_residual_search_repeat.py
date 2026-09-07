import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("repeat", Path(__file__).parents[1]/"scripts/geometry_map/run_residual_search_repeat_v1.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class RepeatTests(unittest.TestCase):
    def test_only_seen_development(self):
        M.validate_repeat([8, 9, 10, 11], [M.FROZEN])
        for episodes in ([], [0], [7], [12], [8, 8]):
            with self.assertRaises(ValueError): M.validate_repeat(episodes, [M.FROZEN])

    def test_exact_operator_not_only_name(self):
        for key, value in (("dose", .5), ("pulse", 1), ("head", 4), ("sign", 1), ("id", "broad0-126")):
            c = {**M.FROZEN, key: value}
            with self.assertRaises(ValueError): M.validate_repeat([8], [c])

    def test_reporting_retains_original_fields(self):
        original = {"phase": "adaptive_development_full99", "source_sha256": {"base": "abc"}, "episodes": [8], "native_cem": {"population": 300}}
        before = copy.deepcopy(original); result = M.augment_protocol(original)
        self.assertEqual(original, before)
        self.assertEqual(result["native_cem"], before["native_cem"])
        self.assertEqual(result["source_sha256"]["base"], "abc")
        self.assertFalse(result["held_data_opened"])
        self.assertIn("not untouched confirmation", result["interpretation"])


if __name__ == "__main__": unittest.main()
