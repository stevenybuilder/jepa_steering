import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DistributedSampler

from offline_study.training_pilot import VirtualRankBatchSampler, FixedRate, native_step, objective


class TinyModel(torch.nn.Module):
    """Exercise the entire upstream train=True control flow, not GPU numerics."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.9))
        self.predictor = torch.nn.Identity()
        self.heads = {}

    def encode(self, obs, action):
        return obs["visual"], obs["proprio"], action

    def forward_pred(self, visual, actions, proprio):
        return visual * self.weight, None, proprio * self.weight

    def compute_loss(self, pv, pp, visual, proprio, shift):
        return {"loss": (pv - visual).square().mean() + (pp - proprio).square().mean()}

    def rollout(self, **kwargs):
        loss = kwargs["pred_video_features"][:, kwargs["t"]].square().mean()
        return {"loss": loss[None]}, loss, None, None

    def backward(self, loss):
        loss.backward()

    def optimization_step(self):
        return None, None


class TrainingSamplerTests(unittest.TestCase):
    def test_extracted_native_training_control_flow_and_objective(self):
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        if not (vendor / "app/vjepa_wm/train.py").is_file():
            self.skipTest("Pinned upstream source not installed")
        model = TinyModel()
        obs = {"visual": torch.arange(24.).reshape(2, 4, 3), "proprio": torch.ones(2, 4, 2)}
        actions = torch.zeros(2, 4, 2)
        run = native_step(vendor, model, FixedRate(.001), FixedRate(.000001))
        torch.manual_seed(71)
        rng = torch.get_rng_state()
        with patch("torch.amp.autocast", side_effect=lambda *a, **kw: nullcontext()):
            result = run(obs, actions, torch.zeros(2, 4, 2), torch.zeros(2, 4), train=True)
            grad, after = model.weight.grad.clone(), torch.get_rng_state()
            model.zero_grad()
            torch.set_rng_state(rng)
            actual = objective(model, obs, actions)
            actual.backward()
        self.assertEqual(result[0], float(actual.detach()))
        self.assertTrue(torch.equal(grad, model.weight.grad))
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_virtual_batches_match_each_native_distributed_stream(self):
        for length in (53568, 145800, 129, 127):
            sampler = VirtualRankBatchSampler(length, epoch=3)
            actual = list(sampler)
            self.assertEqual(len(actual), len(sampler))
            for rank in range(16):
                native = DistributedSampler(range(length), num_replicas=16, rank=rank, shuffle=True)
                native.set_epoch(3)
                stream = list(native)
                used = [index for batch in actual for index in batch[rank * 8:(rank + 1) * 8]]
                self.assertEqual(used, stream[:len(stream) // 8 * 8])


if __name__ == "__main__":
    unittest.main()
