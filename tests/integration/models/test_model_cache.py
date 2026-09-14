import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_study.models import model_loader
from offline_study.core.protocol import sha256


class ModelCacheTests(unittest.TestCase):
    def test_environment_mapping_is_explicit_and_unknown_fails_before_loading(self):
        from offline_study.models.backends import JepaBackend
        for kind in ("metaworld", "pusht", "pointmaze", "wall"):
            self.assertEqual(model_loader.model_name_for_dataset(kind), "jepa_wm_" + kind)
            self.assertEqual(model_loader.MODEL_DATASETS["jepa_wm_" + kind], kind)
        for kind in ("maze", "Wall", "droid", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                JepaBackend(Path("missing"), Path("missing"), "bad", kind, "cpu")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            model_loader.load_headless(Path("missing"), model_name="jepa_wm_unknown")

    def test_missing_cache_fails_before_network_access(self):
        with tempfile.TemporaryDirectory() as temporary, patch("torch.hub.get_dir", return_value=temporary), patch("torch.hub.load") as load:
            with self.assertRaisesRegex(ValueError, "audited runtime"):
                with model_loader.verified_local_dino_cache():
                    pass
            load.assert_not_called()

    def test_only_verified_encoder_is_redirected_and_original_restored(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "facebookresearch_dinov2_main"
            source.mkdir()
            for i in range(157):
                (source / f"file{i:03}.py").write_text("# fixture\n")
            digest = hashlib.sha256()
            for path in sorted(source.rglob("*.py")):
                digest.update(str(path.relative_to(source)).encode() + b"\0" + path.read_bytes())
            weights = root / "checkpoints/dinov2_vits14_pretrain.pth"
            weights.parent.mkdir()
            weights.write_bytes(b"fixture")
            with patch("torch.hub.get_dir", return_value=temporary), patch("torch.hub.load", return_value="model") as load, \
                 patch.object(model_loader, "DINO_SOURCE_SHA256", digest.hexdigest()), \
                 patch.object(model_loader, "DINO_WEIGHT_SHA256", sha256(weights)):
                with model_loader.verified_local_dino_cache():
                    self.assertEqual(torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"), "model")
                    load.assert_called_once_with(str(source), "dinov2_vits14", source="local")
                self.assertIs(torch.hub.load, load)
