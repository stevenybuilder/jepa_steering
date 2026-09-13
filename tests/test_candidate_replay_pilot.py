import unittest

import torch
from torch import nn

from offline_study.candidate_replay_pilot import ReplayField, components


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])

    def forward(self, value):
        for block in self.predictor_blocks:
            value = block(value)
        return value


class ReplayTests(unittest.TestCase):
    def test_complete_bank_decomposition_and_no_renormalization(self):
        values = torch.arange(1800,dtype=torch.float32).reshape(300,2,3)
        mean,centered = components(values)
        self.assertTrue(torch.equal(mean+centered,values))
        self.assertTrue(torch.equal(centered.mean(0),torch.zeros(2,3)))
        total = values.double().square().sum()
        parts = centered.double().square().sum()+300*mean.double().square().sum()
        self.assertEqual(float(total),float(parts))
        with self.assertRaises(ValueError):
            components(values[:299])

    def test_replay_requires_native_field_and_cleans_hooks(self):
        predictor = Predictor()
        native = torch.ones(3,256,4)
        delta = torch.full_like(native,2.)
        with ReplayField(predictor,native,delta):
            results = [predictor(native) for _ in range(6)]
        self.assertTrue(torch.equal(results[2],native+delta))
        self.assertTrue(all(torch.equal(v,native) for i,v in enumerate(results) if i!=2))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in predictor.modules()))
        with self.assertRaises(ValueError):
            with ReplayField(predictor,native+1,delta):
                for _ in range(6):
                    predictor(native)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in predictor.modules()))

    def test_zero_replay_preserves_signed_zero_bytes(self):
        predictor = Predictor()
        native = torch.zeros(3,256,4)
        native[0,0,0] = -0.
        with ReplayField(predictor,native,None):
            result = [predictor(native) for _ in range(6)]
        self.assertTrue(all(torch.equal(x.view(torch.uint8),native.view(torch.uint8)) for x in result))


if __name__ == '__main__':
    unittest.main()
