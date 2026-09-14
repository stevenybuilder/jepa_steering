import unittest

import torch

from offline_study.tasks.navigation.navigation_env_check import observation


class NavigationObservationTests(unittest.TestCase):
    def test_native_pixels_and_info_are_separate(self):
        pixels = torch.arange(3 * 224 * 224).remainder(256).byte().reshape(1, 3, 224, 224)
        value = observation(pixels, {"proprio": torch.zeros(1, 4)})
        self.assertEqual(value["visual_shape"], [1, 3, 224, 224])
        self.assertEqual(len(value["visual_sha256"]), 64)
        with self.assertRaises(ValueError):
            observation(torch.zeros_like(pixels), {"proprio": torch.zeros(1, 4)})


if __name__ == "__main__":
    unittest.main()
