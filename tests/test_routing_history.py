import unittest

import torch

from offline_study.routing_history import NativeHistoryCapture, history_tensor


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

    def forward(self, x):
        for block in self.predictor_blocks:
            x = block(x)
        return x


class RoutingHistoryTests(unittest.TestCase):
    def test_actual_hidden_width_includes_proprio_features(self):
        values = {h: torch.full((2, 400), float(h)) for h in range(1, 7)}
        self.assertEqual(history_tensor(values, 2).shape, (2, 6, 400))
        with self.assertRaises(ValueError):
            history_tensor({h: v[:, :384] for h, v in values.items()}, 2)
        values[2][0, 399] = float("nan")
        with self.assertRaises(ValueError):
            history_tensor(values, 2)

    def test_passive_capture_keeps_outputs_and_horizon_identity(self):
        model = Predictor()
        with NativeHistoryCapture(model) as capture:
            for h in range(1, 7):
                x = torch.full((2, 512, 4), float(h))
                x[:, :256] = 1000
                self.assertIs(model(x), x)
        for h in range(1, 7):
            self.assertTrue(torch.equal(capture.values[h], torch.full((2, 4), float(h))))
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.predictor_blocks[3]._forward_hooks)

    def test_incomplete_capture_fails_and_removes_hooks(self):
        model = Predictor()
        with self.assertRaises(ValueError):
            with NativeHistoryCapture(model):
                model(torch.zeros(1, 256, 4))
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.predictor_blocks[3]._forward_hooks)

    def test_seventh_step_is_rejected(self):
        model = Predictor()
        with self.assertRaises(ValueError):
            with NativeHistoryCapture(model):
                for _ in range(7):
                    model(torch.zeros(1, 256, 4))
        self.assertFalse(model._forward_pre_hooks)
