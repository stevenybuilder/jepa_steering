import tempfile
import unittest
import importlib.util
from pathlib import Path

import torch

from offline_study.droid_native import assert_same
from offline_study.navigation_input_check import SelectedFrameSlicer, rng_state, restore_rng
from offline_study.vendor import use_vendor


class NavigationInputTests(unittest.TestCase):
    def test_selected_frames_preserve_subset_mapping_actions_pixels_and_rng(self):
        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("Native image preprocessing test runs in the GPU-worker environment")
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        if not vendor.exists():
            self.skipTest("Pinned upstream unavailable")
        use_vendor(vendor)
        from app.plan_common.datasets.wall_dset import WallDataset
        from app.plan_common.datasets.traj_dset import TrajSubset, TrajSlicerDataset
        from app.plan_common.datasets.transforms import make_transforms
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "obses").mkdir()
            generator = torch.Generator().manual_seed(20)
            for name in ("states", "actions", "door_locations", "wall_locations"):
                torch.save(torch.randn(3, 30, 2, generator=generator), root / f"{name}.pth")
            for i in range(3):
                torch.save(torch.randint(0, 256, (30, 3, 32, 48), generator=generator,
                                         dtype=torch.uint8), root / f"obses/episode_{i:03d}.pth")
            transform = make_transforms(img_size=32, random_horizontal_flip=False,
                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1.777, 1.777))
            dataset = WallDataset(str(root), transform=transform, normalize_action=True)
            nested = TrajSubset(TrajSubset(dataset, [2, 0, 1]), [1, 0])
            native = TrajSlicerDataset(nested, 4, frameskip=5, generator=generator)
            optimized = SelectedFrameSlicer(native)
            for i in (0, 4, len(native) - 1):
                before = rng_state()
                expected = native[i]
                after = rng_state()
                restore_rng(before)
                actual = optimized[i]
                assert_same(expected, actual)
                assert_same(after, rng_state())
            native.num_frames = 5
            with self.assertRaises(ValueError):
                SelectedFrameSlicer(native)


if __name__ == "__main__":
    unittest.main()
