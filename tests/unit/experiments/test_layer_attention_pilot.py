import unittest

import torch
from torch import nn
import torch.nn.functional as F

from offline_study.experiments.layer_attention_pilot import attention_summary, PredictorObserver


class Attention(nn.Module):
    def forward(self, x, T=1, H=1, W=2, action_tokens=0):
        return F.scaled_dot_product_attention(x, x, x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()

    def forward(self, x):
        return self.attn(x, T=1, H=1, W=2)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for block in self.predictor_blocks:
            x = block(x)
        return x


class LayerPilotTests(unittest.TestCase):
    def test_uniform_distance(self):
        q = torch.zeros(1, 1, 2, 4)
        s = attention_summary(q, q, T=1, H=1, W=2)
        self.assertEqual(s['spatial_distance_patches'], [[.5]])
        self.assertEqual(s['visual_mass'], [[1.]])

    def test_diagonal_mask(self):
        q = torch.zeros(1, 1, 2, 4)
        for mask in (torch.eye(2, dtype=torch.bool),
                     torch.tensor([[0., -float('inf')], [-float('inf'), 0.]])):
            s = attention_summary(q, q, T=1, H=1, W=2, attn_mask=mask)
            self.assertEqual(s['spatial_distance_patches'], [[0.]])

    def test_conditioning_not_spatial(self):
        q = torch.zeros(1, 1, 3, 4)
        s = attention_summary(q, q, T=1, H=1, W=2, action_tokens=1)
        self.assertAlmostEqual(s['spatial_distance_patches'][0][0], .5)
        self.assertAlmostEqual(s['conditioning_mass'][0][0], 1/3)

    def test_causal_temporal(self):
        q = torch.zeros(1, 1, 2, 4)
        s = attention_summary(q, q, T=2, H=1, W=1, is_causal=True)
        self.assertEqual(s['temporal_distance_frames'], [[.25]])

    def test_chunk_invariance(self):
        torch.manual_seed(4)
        q, k = torch.randn(2, 3, 8, 4), torch.randn(2, 3, 8, 4)
        a = attention_summary(q, k, T=2, H=2, W=2, chunk=1)
        b = attention_summary(q, k, T=2, H=2, W=2, chunk=64)
        for key in ('spatial_distance_patches', 'temporal_distance_frames', 'visual_mass'):
            torch.testing.assert_close(torch.tensor(a[key]), torch.tensor(b[key]))

    def test_bad_layout_and_fully_masked_fail(self):
        q = torch.zeros(1, 1, 2, 4)
        with self.assertRaises(ValueError):
            attention_summary(q, q, T=1, H=2, W=2)
        with self.assertRaises(ValueError):
            attention_summary(q, q, T=1, H=1, W=2, attn_mask=torch.zeros(2, 2, dtype=torch.bool))

    def test_observer_preserves_kernel_outputs_and_rng(self):
        p = Predictor().eval()
        x = torch.randn(2, 3, 2, 4)
        ref = p(x)
        original = F.scaled_dot_product_attention
        rng = torch.get_rng_state().clone()
        with PredictorObserver(p) as observer:
            result = p(x)
        self.assertTrue(torch.equal(ref, result))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(original, F.scaled_dot_product_attention)
        self.assertEqual([r['layer'] for r in observer.rows], [0, 1])
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in p.modules()))

    def test_restores_after_exception(self):
        p = Predictor().eval()
        original = F.scaled_dot_product_attention
        with self.assertRaises(RuntimeError):
            with PredictorObserver(p):
                raise RuntimeError('test')
        self.assertIs(original, F.scaled_dot_product_attention)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in p.modules()))


if __name__ == '__main__':
    unittest.main()
