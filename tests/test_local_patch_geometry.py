import importlib.util
from pathlib import Path
import unittest
import torch

SPEC = importlib.util.spec_from_file_location("local_geo", Path(__file__).parents[1]/"scripts/geometry_map/local_patch_geometry.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class LocalGeometryTests(unittest.TestCase):
    def test_pca_reconstruction_and_rotation(self):
        torch.manual_seed(5)
        x = torch.randn(40, 3)@torch.randn(3, 12)
        mean, basis, variance = M.gram_pca(x, 3)
        torch.testing.assert_close(basis@basis.T, torch.eye(3), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close((x-mean)@basis.T@basis+mean, x, atol=2e-5, rtol=2e-5)
        self.assertGreater(float(variance.min()), 0.)

    def test_neighbors_respect_episode_cap(self):
        groups = [i//4 for i in range(24)]
        indices = M.balanced_neighbors(torch.arange(24), groups, 12, 2)
        self.assertEqual(len(indices), 12)
        self.assertTrue(all(sum(groups[i] == g for i in indices) == 2 for g in set(groups)))

    def test_principal_angles_not_coordinate_signs(self):
        a = torch.eye(5)[:2]
        b = a.flip(0)*-1
        self.assertAlmostEqual(M.subspace_overlap(a, b)["mean_squared_principal_cosine"], 1.)

    def test_local_chart_recovers_curved_circle_better_than_line(self):
        theta = torch.arange(64)*2*torch.pi/64
        train = torch.stack([theta.cos(), theta.sin()], 1)
        angle = theta[:16]+.037
        test = torch.stack([angle.cos(), angle.sin()], 1)
        result = M.analyze_chart(train, test, [str(i//4) for i in range(64)], ["held"]*16,
                                 rank=1, ambient_rank=2, neighbors=12, max_per_group=4)
        means = result["equal_episode_means"]
        self.assertLess(means["local_reconstruction_fraction"], means["global_reconstruction_fraction"]*.2)
        self.assertLess(means["local_reconstruction_fraction"], means["random_neighborhood_reconstruction_fraction"])
        self.assertEqual(result["independent_development_episodes"], 1)

    def test_overlap_and_rank_fail_loudly(self):
        x = torch.randn(20, 4)
        with self.assertRaises(ValueError): M.analyze_chart(x, x, ["same"]*20, ["same"]*20)
        with self.assertRaises(ValueError): M.gram_pca(torch.ones(10, 4), 2)


if __name__ == "__main__": unittest.main()
