import importlib.util
from pathlib import Path
import unittest
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location("hooks", Path(__file__).parents[1]/"scripts/geometry_map/residual_search_hooks.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.num_heads = 4; self.proj = nn.Linear(8, 8)
    def forward(self, x): return self.proj(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = Attention(); self.norm2 = nn.Identity()
        self.mlp = nn.Linear(8, 8); self.adaLN_modulation = nn.Linear(8, 48)
    def forward(self, x, z):
        _, _, ga, _, _, gm = self.adaLN_modulation(z).repeat_interleave(x.shape[1]//z.shape[1], 1).chunk(6, -1)
        x = x + self.attn(x)*ga
        return x + self.mlp(self.norm2(x))*gm


class HookTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10); self.block = Block().eval()
        self.x, self.z = torch.randn(2, 512, 8), torch.randn(2, 2, 8)
    def spec(self, site):
        return {"site": site, "head": 1 if site == "attention_preproj" else None,
                "mask": "top", "pulse": 1, "sign": 1, "dose": .25}
    def delta(self, x, step): return x*.3 + .01

    def test_identity_and_cleanup_all_sites(self):
        baseline = self.block(self.x, self.z)
        weights = {k:v.clone() for k,v in self.block.state_dict().items()}
        for site in ("residual", "attention_preproj", "mlp_output"):
            with M.ComponentPatch(self.block, self.spec(site), self.delta, .1, identity=True):
                self.assertTrue(torch.equal(baseline, self.block(self.x, self.z)))
            self.assertTrue(torch.equal(baseline, self.block(self.x, self.z)))
        self.assertTrue(all(torch.equal(v, self.block.state_dict()[k]) for k,v in weights.items()))

    def test_post_gate_cap_and_sham_match(self):
        for site in ("residual", "attention_preproj", "mlp_output"):
            norms = []
            for sham in (False, True):
                with M.ComponentPatch(self.block, self.spec(site), self.delta, .1, sham=sham) as hook:
                    self.block(self.x, self.z)
                row = hook.records[0]
                self.assertLessEqual(float(row["requested_residual_norm"].max()), .100001)
                self.assertLess(row["normmatch_max_abs"], 1e-6)
                norms.append(row["requested_residual_norm"])
            torch.testing.assert_close(norms[0], norms[1], atol=1e-6, rtol=1e-5)

    def test_token_prefix_and_mask_preserved_at_residual(self):
        baseline = self.block(self.x, self.z)
        with M.ComponentPatch(self.block, self.spec("residual"), self.delta, .1):
            edited = self.block(self.x, self.z)
        self.assertTrue(torch.equal(baseline[:, :256], edited[:, :256]))
        self.assertTrue(torch.equal(baseline[:, 384:], edited[:, 384:]))
        self.assertFalse(torch.equal(baseline, edited))

    def test_head_restriction_before_projection(self):
        captured = []
        def capture(module, args): captured.append(args[0].clone())
        spec = self.spec("attention_preproj")
        with M.ComponentPatch(self.block, spec, self.delta, .1):
            h = self.block.attn.proj.register_forward_pre_hook(capture)
            self.block(self.x, self.z); h.remove()
        delta = captured[0] - self.x
        self.assertEqual(float(delta[..., :2].abs().max()), 0.)
        self.assertEqual(float(delta[..., 4:].abs().max()), 0.)
        self.assertGreater(float(delta[..., 2:4].abs().max()), 0.)

    def test_timing_and_reset(self):
        spec = dict(self.spec("residual"), pulse=3)
        baseline = self.block(self.x, self.z)
        with M.ComponentPatch(self.block, spec, self.delta, .1) as hook:
            values = [self.block(self.x, self.z) for _ in range(6)]
            self.assertEqual([r["step"] for r in hook.records], [3])
            self.assertTrue(all(torch.equal(values[i], baseline) for i in (0, 1, 3, 4, 5)))
            with self.assertRaises(RuntimeError): self.block(self.x, self.z)
            hook.reset(); self.assertTrue(torch.equal(self.block(self.x, self.z), baseline))

    def test_exception_cleanup_and_random_stream_unchanged(self):
        baseline = self.block(self.x, self.z)
        rng = torch.random.get_rng_state().clone()
        try:
            with M.ComponentPatch(self.block, self.spec("attention_preproj"), self.delta, .1, sham=True):
                self.block(self.x, self.z); raise ValueError("expected")
        except ValueError: pass
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(baseline, self.block(self.x, self.z)))

    def test_dose_after_cap_stays_distinct_when_saturated(self):
        for site in ("residual", "attention_preproj", "mlp_output"):
            values = []
            for dose in (.125, .25, .5, 1.):
                spec = dict(self.spec(site), dose=dose)
                with M.ComponentPatch(self.block, spec, lambda x, step: x*1000, .1) as hook:
                    self.block(self.x, self.z)
                self.assertTrue(bool(hook.records[0]["cap_saturated"].all()))
                values.append(hook.records[0]["requested_residual_norm"])
            for dose, value in zip((.125, .25, .5, 1.), values):
                torch.testing.assert_close(value, values[-1]*dose, atol=1e-6, rtol=1e-5)

    def test_replayed_semantic_norm_controls(self):
        spec = dict(self.spec("attention_preproj"), dose=1.)
        with M.ComponentPatch(self.block, spec, self.delta, .1, sham=True) as hook:
            hook.norm_targets = {1: torch.tensor([.01, .02])}
            self.block(self.x, self.z)
        torch.testing.assert_close(hook.records[0]["requested_residual_norm"], torch.tensor([.01, .02]), atol=1e-6, rtol=1e-5)
        self.assertEqual(hook.records[0]["norm_target_source"], "recorded_semantic_trajectory")


if __name__ == "__main__": unittest.main()
