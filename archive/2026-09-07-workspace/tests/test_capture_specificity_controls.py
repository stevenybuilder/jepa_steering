"""Specificity controls must preserve raw dose, timing and older context."""
import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("specificity", ROOT / "scripts/geometry_map/capture_specificity_controls.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


class SpecificityTests(unittest.TestCase):
    def test_per_candidate_raw_norm_matching_and_direction(self):
        delta = torch.tensor([[3., 4.], [1., -1.], [5., 7.]], dtype=torch.float64)
        reference = torch.tensor([[0., 10.], [2., 0.], [0., 0.]], dtype=torch.float64)
        matched = controls.match_raw_norm(delta, reference)
        torch.testing.assert_close(matched.norm(dim=-1), reference.norm(dim=-1), atol=1e-12, rtol=0)
        torch.testing.assert_close(matched[0], 2 * delta[0])
        self.assertEqual(torch.count_nonzero(matched[2]), 0)

    def test_zero_direction_cannot_be_replaced_by_arbitrary_edit(self):
        with self.assertRaises(ValueError):
            controls.match_raw_norm(torch.zeros(2, 3), torch.ones(2, 3))

    def test_shifted_edit_is_exact_same_vector_on_latest_tokens_only(self):
        delta = torch.tensor([[2., -3., 4.], [-1., 2., 0.]])
        patch = controls.RecordedEditAtStep(torch.nn.Identity(), lambda: None, delta, 2)
        first = torch.zeros(2, 256, 3)
        second = torch.ones(2, 512, 3) * 100
        self.assertIs(patch.hook(None, None, first), first)
        result = patch.hook(None, None, second)
        torch.testing.assert_close(result[:, :256], second[:, :256], atol=0, rtol=0)
        torch.testing.assert_close(result[:, -256:], second[:, -256:] + delta[:, None], atol=0, rtol=0)
        torch.testing.assert_close(patch.record["requested_delta"], delta, atol=0, rtol=0)
        self.assertEqual(patch.edits, 1)
        self.assertIs(patch.hook(None, None, second), second)

    def test_identity_keeps_exact_tensor_and_dtype(self):
        patch = controls.RecordedEditAtStep(torch.nn.Identity(), lambda: None, torch.zeros(2, 3), 1)
        value = torch.randn(2, 256, 3).half()
        self.assertIs(patch.hook(None, None, value), value)
        self.assertEqual(patch.record["realized_delta"].count_nonzero(), 0)

    def test_hook_counter_resets_for_each_complete_unroll(self):
        block = torch.nn.Identity()
        def unroll():
            return [block(torch.zeros(2, 256 if i == 0 else 512, 3)) for i in range(6)]
        patch = controls.RecordedEditAtStep(block, unroll, torch.ones(2, 3), 2)
        with patch:
            for _ in range(2):
                output = patch.unroll()
                self.assertEqual(patch.calls, 6)
                self.assertEqual(patch.edits, 1)
                self.assertEqual(sum(bool(value.count_nonzero()) for value in output), 1)
                self.assertTrue(output[1][:, -256:].eq(1).all())
        self.assertEqual(len(block._forward_hooks), 0)

    def test_held_receipt_filter_rejects_negative_and_future_ids(self):
        receipt = {"complete": True, "outputs": [{"episode": e} for e in [-1, 0, 49, 50, 75, 99]]}
        self.assertEqual([row["episode"] for row in controls.discovery_rows(receipt)], [0, 49])


if __name__ == "__main__":
    unittest.main()
