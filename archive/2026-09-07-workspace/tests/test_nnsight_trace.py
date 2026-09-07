import importlib.util
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/geometry_map"))
from nnsight_trace import NNsightPredictorTrace, apply_raw_delta


class ToyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(6)])
        self.predictor_norm = torch.nn.LayerNorm(4)
        self.predictor_proj = torch.nn.Linear(4, 3)

    def forward(self, value):
        for block in self.predictor_blocks:
            value = block(value)
        value = self.predictor_norm(value)
        return self.predictor_proj(value[:, None])


class ToyWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.predictor = ToyPredictor()

    def unroll(self, z, act_suffix):
        return self.model.predictor(z)  # One first-step call is enough for tracer API tests.


class TraceTests(unittest.TestCase):
    def test_zero_edit_preserves_tensor_identity(self):
        value = torch.randn(2, 256, 4)
        self.assertIs(apply_raw_delta(value, torch.zeros(2, 4)), value)
        self.assertIs(apply_raw_delta(value, None), value)

    def test_newest_tokens_only(self):
        value = torch.zeros(2, 512, 4)
        result = apply_raw_delta(value, torch.ones(2, 4))
        self.assertEqual(result[:, :256].count_nonzero(), 0)
        self.assertTrue(result[:, -256:].eq(1).all())

    def test_bad_delta_rejected(self):
        with self.assertRaises(ValueError):
            apply_raw_delta(torch.zeros(2, 256, 4), torch.ones(3, 4))

    @unittest.skipUnless(importlib.util.find_spec("nnsight"), "optional NNsight not installed")
    def test_nnsight_identity_and_nonzero_edit_match_native_hook(self):
        torch.manual_seed(905)
        model = ToyWM().eval().requires_grad_(False)
        z, actions = torch.randn(2, 256, 4), torch.zeros(1, 2, 1)
        baseline = model.unroll(z, actions)
        tracer = NNsightPredictorTrace(model, (0, 1))
        zero = tracer.trace_unroll(z, actions, torch.zeros(2, 4))
        self.assertTrue(torch.equal(zero["prediction"], baseline))
        delta = torch.ones(2, 4) * .125
        handle = model.model.predictor.predictor_blocks[3].register_forward_hook(lambda _m, _a, output: apply_raw_delta(output, delta))
        try:
            native = model.unroll(z, actions)
        finally:
            handle.remove()
        traced = tracer.trace_unroll(z, actions, delta)
        self.assertTrue(torch.equal(traced["prediction"], native))
        self.assertEqual(traced["stages"]["predictor_proj_input"].shape, (2, 1, 256, 4))
        tracer.close()
        tracer.close()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertTrue(all(not hasattr(m, "__nnsight_forward__") for m in model.modules()))
        self.assertTrue(torch.equal(model.unroll(z, actions), baseline))
        with self.assertRaises(RuntimeError):
            tracer.trace_unroll(z, actions)

    @unittest.skipUnless(importlib.util.find_spec("nnsight"), "optional NNsight not installed")
    def test_cleanup_preserves_existing_hooks_and_forward_on_exception(self):
        model = ToyWM().eval().requires_grad_(False)
        block = model.model.predictor.predictor_blocks[3]
        handle = block.register_forward_hook(lambda _m, _a, output: output)
        original_forward = block.forward
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with NNsightPredictorTrace(model, (0,)):
                raise RuntimeError("test failure")
        self.assertEqual(set(block._forward_hooks), {handle.id})
        self.assertEqual(block.forward, original_forward)
        handle.remove()


if __name__ == "__main__":
    unittest.main()
