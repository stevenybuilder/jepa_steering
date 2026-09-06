import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/geometry_map"))
from residual_search_v1 import token_mask, masked_equal_energy, signed_permutation_sham, pilot_candidates, broad_candidates, interaction_delta, screen_actions, physical_arm_specs


class SearchTests(unittest.TestCase):
    def test_projected_physical_factorial_is_explicit_five_arms(self):
        c = broad_candidates()[0]
        self.assertEqual(physical_arm_specs([c], True), [(False,None,False),(True,None,False),(True,c,False),(True,c,True),(False,c,False)])
        self.assertEqual(len(physical_arm_specs([c], False)), 3)
        with self.assertRaises(ValueError): physical_arm_specs([c,c], True)

    def test_disabled_input_projection_is_exact_bypass(self):
        actions = torch.randn(6, 2, 20)
        projected, stats = screen_actions(actions, object(), False)
        self.assertIs(projected, actions)
        self.assertFalse(stats['enabled'])

    def test_interaction_zero_for_additive_and_known_bilinear_term(self):
        base = torch.randn(6, 2, 3)
        a, b = torch.randn_like(base), torch.randn_like(base)
        torch.testing.assert_close(interaction_delta(base, base+a, base+b, base+a+b), torch.zeros_like(base).double(), atol=1e-6, rtol=0)
        actual = interaction_delta(base, base+a, base+b, base+a+b+a*b)
        torch.testing.assert_close(actual, (a*b).double(), atol=1e-6, rtol=1e-6)

    def test_broad_population_covers_all96_native_heads_and18_sites(self):
        rows = broad_candidates()
        self.assertEqual(len(rows), 256)
        self.assertEqual({(r['block'], r['head']) for r in rows[:96]}, {(b, h) for b in range(6) for h in range(16)})
        for family in ['action_gain', 'pca_gain']:
            self.assertEqual({(r['block'], r['site'], r['rank']) for r in rows if r['family'] == family},
                             {(b,s,r) for b in range(6) for s in ['residual','attention_preproj','mlp_output'] for r in [1,4,8,16]})
        self.assertEqual(rows, broad_candidates())

    def test_mask_preserves_total_energy_and_support(self):
        delta = torch.randn(2, 256, 400)
        mask = token_mask("top")
        value = masked_equal_energy(delta, mask)
        self.assertEqual(value[:, ~mask].count_nonzero(), 0)
        self.assertTrue(torch.allclose(value.flatten(1).norm(dim=-1), delta.flatten(1).norm(dim=-1), rtol=1e-6))

    def test_zero_masked_direction_fails(self):
        value = torch.zeros(1, 256, 2); value[:, 128:] = 1
        with self.assertRaises(RuntimeError):
            masked_equal_energy(value, token_mask("top"))

    def test_sham_preserves_each_token_norm(self):
        value = torch.randn(2, 256, 400)
        sham = signed_permutation_sham(value, 1)
        self.assertTrue(torch.allclose(value.norm(dim=-1), sham.norm(dim=-1), rtol=1e-6))
        self.assertTrue(torch.equal(sham, signed_permutation_sham(value, 1)))

    def test_pilot24_unique_no_invalid_coordinate_transfer(self):
        rows = pilot_candidates()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['id'] for r in rows}), 24)
        self.assertTrue(all(r['block'] == 3 and r['site'] == 'residual' for r in rows))


if __name__ == "__main__":
    unittest.main()
