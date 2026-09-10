import unittest

from offline_study.droid_catalogue import PREFIX, validate_object


class DroidCatalogueTests(unittest.TestCase):
    def test_native_existence_rule_retains_failure_and_success(self):
        for folder in ("success", "failure"):
            row = {"name": PREFIX + f"lab/{folder}/day/episode/trajectory.h5",
                   "generation": "123", "size": "50", "md5Hash": "checksum"}
            self.assertEqual(validate_object(row), row)

    def test_unexpected_paths_or_missing_generation_rejected(self):
        row = {"name": PREFIX + "lab/success/episode/trajectory.h5",
               "generation": "123", "size": "50", "md5Hash": "checksum"}
        for key, bad in (("name", "/private/trajectory.h5"), ("generation", ""), ("size", "0")):
            with self.assertRaises(ValueError):
                validate_object({**row, key: bad})
