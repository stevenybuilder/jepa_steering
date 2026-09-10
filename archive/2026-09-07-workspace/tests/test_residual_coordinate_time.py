import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/geometry_map"))
from run_residual_coordinate_time import projected_difference, cap_delta, match_norm, random_basis, DiscoveryDonors, ResidualPulse, native_unroll_actions, hand_goal_metrics, aggregate_fixed_rows


class ResidualTests(unittest.TestCase):
    def test_aggregate_groups_whole_episodes(self):
        row = {"requested_rank": 8, "intervention_step": 1, "sign": 1, "sham": False, "measurement_horizon": 6,
               "episode": 0, "decoded_xyz_shift": [0, 0, .01], "latent_error_change": .2, "physical_readout_error_m": .03}
        second = {**row, "episode": 1, "decoded_xyz_shift": [0, 0, -.02], "latent_error_change": -.1}
        value = aggregate_fixed_rows([row, second])[0]
        self.assertEqual(value["episode_count"], 2)
        self.assertAlmostEqual(value["mean_decoded_z_shift_m"], -.005)
        self.assertEqual(value["latent_error_improved_count"], 1)
        with self.assertRaises(RuntimeError):
            aggregate_fixed_rows([row, row])

    def test_progress_is_physical_hand_goal_not_native_success(self):
        states = np.zeros((15, 6)); states[:, 0] = np.linspace(.9, .2, 15)
        result = hand_goal_metrics(np.array([1., 0, 0]), states)
        self.assertAlmostEqual(result["hand_goal_progress_m"], .8)
        self.assertAlmostEqual(result["minimum_hand_goal_distance_m"], .2)
        self.assertNotIn("success", result)

    def test_native_positional_and_keyword_actions(self):
        actions = torch.zeros(6, 300, 20)
        self.assertIs(native_unroll_actions((None, actions), {}), actions)
        self.assertIs(native_unroll_actions((None,), {"act_suffix": actions}), actions)

    def artifact(self):
        basis = np.eye(400)[:, :2]
        donors = np.zeros((3, 400)); donors[:, 0] = [-1, 1, 2]; donors[2, 5] = 5
        coef = np.zeros((3, 400)); coef[2, 0] = 1
        return {"basis_raw_rank2": basis, "donor_p3": donors, "coordinate_coef_raw": coef,
                "coordinate_intercept_raw": np.zeros(3), "cap_radius": .25}

    def test_raw_projection_preserves_complement(self):
        basis = random_basis(10, 2)
        current, donor = torch.randn(3, 10).double(), torch.randn(3, 10).double()
        delta = projected_difference(current, donor, basis)
        self.assertTrue(torch.allclose(delta-(delta@basis)@basis.T, torch.zeros_like(delta), atol=1e-12))

    def test_cap_then_sham_norm(self):
        delta = cap_delta(torch.tensor([[3., 4.]]), .5)
        sham = match_norm(torch.tensor([[0., 2.]]), delta)
        self.assertTrue(torch.allclose(delta.norm(dim=-1), sham.norm(dim=-1)))
        self.assertAlmostEqual(float(delta.norm()), .5)

    def test_zero_sham_fails_nonzero_reference(self):
        with self.assertRaises(RuntimeError):
            match_norm(torch.zeros(1, 4), torch.ones(1, 4))

    def test_signed_discovery_donor_and_complement_match(self):
        pool = DiscoveryDonors(self.artifact(), 2)
        delta, row = pool.delta(torch.zeros(1, 400), 1)
        self.assertEqual(row["donor_index"].item(), 1)
        self.assertGreater(float(delta[0, 0]), 0)
        delta, row = pool.delta(torch.zeros(1, 400), -1)
        self.assertEqual(row["donor_index"].item(), 0)
        self.assertLess(float(delta[0, 0]), 0)

    def test_missing_donor_identity(self):
        pool = DiscoveryDonors(self.artifact(), 2)
        value = torch.zeros(1, 400); value[0, 0] = 20
        delta, row = pool.delta(value, 1)
        self.assertFalse(row["donor_available"].item())
        self.assertEqual(row["donor_index"].item(), -1)
        self.assertEqual(delta.count_nonzero(), 0)

    def test_pulse_only_chosen_time_and_newest_tokens(self):
        block = torch.nn.Identity()
        pool = DiscoveryDonors(self.artifact(), 2)
        def unroll(value):
            return [block(value.clone()) for _ in range(6)]
        value = torch.zeros(1, 512, 400)
        with ResidualPulse(block, unroll, pool, 3) as patch:
            result = patch.unroll(value)
        self.assertEqual(sum(int(x.count_nonzero()>0) for x in result), 1)
        self.assertEqual(result[2][:, :256].count_nonzero(), 0)
        self.assertFalse(block._forward_hooks)
        with ResidualPulse(block, unroll, pool, 1, identity=True) as patch:
            result = patch.unroll(value)
        self.assertTrue(all(torch.equal(x, value) for x in result))


if __name__ == "__main__":
    unittest.main()
