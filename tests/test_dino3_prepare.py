import unittest

import torch

from offline_study.dino3_prepare import hf_key, native_state


class Dino3PreparationTests(unittest.TestCase):
    def test_documented_names(self):
        for original, converted in (("cls_token", "embeddings.cls_token"),
            ("blocks.23.ls2.gamma", "layer.23.layer_scale2.lambda1"),
            ("blocks.3.mlp.fc2.weight", "layer.3.mlp.down_proj.weight"),
            ("blocks.5.attn.proj.weight", "layer.5.attention.o_proj.weight")):
            self.assertEqual(hf_key(original), converted)

    def test_qkv_and_mask_reshape_with_complete_accounting(self):
        hf = {"embeddings.mask_token": torch.ones(1, 1, 2),
              "layer.0.attention.q_proj.weight": torch.ones(2, 2),
              "layer.0.attention.k_proj.weight": torch.full((2, 2), 2.),
              "layer.0.attention.v_proj.weight": torch.full((2, 2), 3.)}
        template = {"mask_token": torch.zeros(1, 2), "blocks.0.attn.qkv.weight": torch.zeros(6, 2)}
        result, _ = native_state(hf, template)
        self.assertEqual(result["blocks.0.attn.qkv.weight"][:, 0].tolist(), [1., 1., 2., 2., 3., 3.])
        self.assertEqual(result["mask_token"].shape, (1, 2))
        with self.assertRaisesRegex(ValueError, "Unaccounted"):
            native_state({**hf, "unexpected": torch.zeros(2)}, template)
        with self.assertRaisesRegex(ValueError, "Missing"):
            native_state({}, template)


if __name__ == "__main__":
    unittest.main()
