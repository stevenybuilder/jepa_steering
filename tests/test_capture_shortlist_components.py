"""Fail-closed checks for component identity, data split and spatial alignment."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("components", ROOT / "scripts/geometry_map/capture_shortlist_components.py")
components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components)


class ComponentCaptureTests(unittest.TestCase):
    def test_released_adaln_contract_uses_actual_attributes(self):
        model = SimpleNamespace(pred_type="AdaLN", predictor=SimpleNamespace(proprio_encoding="feature"))
        components.verify_predictor_contract(model)
        model.pred_type = "vjepa2_ac"
        with self.assertRaises(ValueError):
            components.verify_predictor_contract(model)

    def test_manifest_filters_before_payload_loading(self):
        rows = [{"parquet": "selected.parquet", "task": "mw-reach-wall", "seed": 2, "episode": ep,
                 "split": "discovery" if ep < 50 else "validation" if ep < 75 else "confirmation"} for ep in range(100)]
        selected = components.selected_manifest(rows, "selected.parquet")
        self.assertEqual([row["episode"] for row in selected], list(range(50)))
        with self.assertRaises(ValueError):
            components.selected_manifest(rows[1:], "selected.parquet")

    def test_encoder_pool_excludes_cls_and_preserves_grid(self):
        value = torch.ones(20, 257, 384)
        value[:, 0] = 1000
        pooled, spatial = components.pool_component(value, "encoder")
        self.assertTrue(torch.equal(pooled, torch.ones(20, 384, dtype=torch.float16)))
        self.assertEqual(tuple(spatial.shape), (20, 256, 384))

    def test_predictor_token_contract_rejects_extra_action_tokens(self):
        pooled, _ = components.pool_component(torch.ones(19, 256, 400), "predictor")
        self.assertEqual(tuple(pooled.shape), (19, 400))
        with self.assertRaises(ValueError):
            components.pool_component(torch.ones(19, 258, 400), "predictor")

    def test_finite_check_rejects_nan(self):
        with self.assertRaises(ValueError):
            components.assert_finite_tensors({"bad": torch.tensor([float("nan")])})

    def test_missing_named_component_fails_closed(self):
        blocks = {"encoder": [torch.nn.Identity() for _ in range(12)],
                  "predictor": [torch.nn.Identity() for _ in range(6)]}
        with self.assertRaises(ValueError):
            components.component_modules(blocks)


if __name__ == "__main__":
    unittest.main()
