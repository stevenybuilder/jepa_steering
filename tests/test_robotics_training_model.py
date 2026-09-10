"""Constructor AST/guard tests only, not native GPU initialization evidence."""
import copy
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from offline_study import robotics_training_model as m
from offline_study.droid_native import assert_same
from offline_study.robotics_training_pilot import native_contract

VENDOR = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"


class Native32ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = {task: native_contract(VENDOR, task) for task in ("metaworld", "pusht")}

    def test_actual_native_dimensions_and_world_model_defaults(self):
        for task, action_dim in (("metaworld", 20), ("pusht", 10)):
            proof = m.constructor_contract(self.contracts[task])
            model, world = proof["calls"]["model"], proof["calls"]["world_model"]
            self.assertEqual((model["action_dim"], world["action_dim"]), (action_dim, action_dim))
            self.assertEqual((model["proprio_dim"], world["proprio_dim"]), (4, 4))
            # Preserve actual source's top-level default, distinct from the
            # custom frame stride used to calculate the concatenated actions.
            self.assertEqual(world["frameskip"], 1)
            self.assertEqual(world["action_skip"], 1)
            self.assertEqual(world["img_size"], 224)
            self.assertEqual(world["heads"], {})
            self.assertTrue(world["mixed_precision"])

    def test_native_optimizer_cadence_no_navigation_16rank_default(self):
        for task, updates in (("metaworld", 3543), ("pusht", 7741)):
            opt = m.constructor_contract(self.contracts[task])["calls"]["optimizer"]
            self.assertEqual(opt["iterations_per_epoch"], updates)
            self.assertEqual(opt["num_epochs"], 50)
            self.assertTrue(opt["freeze_encoder"])
            self.assertEqual(opt["clip_grad"], 1)

    def test_metadata_audit_does_not_construct_or_authorize(self):
        with patch.object(m.torch.cuda, "is_available", side_effect=AssertionError("No CUDA query")):
            proof = m.constructor_contract(self.contracts["pusht"])
        self.assertTrue(proof["metadata_only"])
        self.assertFalse(proof["model_initialized"])
        self.assertEqual(proof["gpu_calls"], 0)
        self.assertFalse(proof["history_launch_authorized"])

    def test_contract_and_original_config_unchanged(self):
        contract = self.contracts["pusht"]
        before = copy.deepcopy(contract)
        m.constructor_contract(contract)
        self.assertEqual(contract, before)
        self.assertIsNone(contract["config"]["optimization"]["transition_model"]["iterations_per_epoch"])

    def test_all_model_seeds_keep_same_constructor_hyperparameters(self):
        base = m.constructor_contract(self.contracts["pusht"])
        for seed in (235, 236):
            self.assertEqual(base, m.constructor_contract(native_contract(VENDOR, "pusht", seed)))

    def test_mutated_contract_rejected_before_constructor(self):
        contract = copy.deepcopy(self.contracts["pusht"])
        contract["config"]["model"]["predictor"]["pred_depth"] = 1
        with self.assertRaises(ValueError):
            m.constructor_contract(contract)

    def test_public_constructor_requires_input_proof_before_cuda(self):
        with patch.object(m, "_input_gate", side_effect=ValueError("Missing real input")), \
                patch.object(m.torch.cuda, "is_available", side_effect=AssertionError("No CUDA query")):
            with self.assertRaisesRegex(ValueError, "Missing real input"):
                m.build_receiving_model(self.contracts["pusht"], None, input_proof="missing")

    def test_no_cpu_fallback_after_input_gate(self):
        with patch.object(m, "_input_gate", return_value="fixture_only"), \
                patch.object(m.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "receiving CUDA"):
                m.build_receiving_model(self.contracts["pusht"], None, input_proof="fixture")

    def test_imported_constructor_identity_not_just_callable_name(self):
        def init_opt():
            pass
        with self.assertRaisesRegex(ValueError, "constructor differs"):
            m._check_callable(init_opt, VENDOR, "app/vjepa_wm/utils.py", "init_opt")

    def test_multi_gpu_visibility_rejected_before_context_or_import(self):
        with patch.object(m, "_input_gate", return_value="fixture_only"), \
                patch.object(m.torch.cuda, "is_available", return_value=True), \
                patch.object(m.torch.cuda, "device_count", return_value=2), \
                patch.object(m.torch.random, "fork_rng", side_effect=AssertionError("No context")):
            with self.assertRaisesRegex(ValueError, "exactly its one assigned GPU"):
                m.build_receiving_model(self.contracts["pusht"], None, input_proof="fixture")

    def test_import_failure_restores_caller_rng_cpu_fixture(self):
        # CPU-only failure-path test: this never builds a model or CUDA proof.
        before = (torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()), random.getstate())
        original_fork = torch.random.fork_rng
        flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32

        def failed_import(*args):
            torch.rand(1)
            np.random.rand()
            random.random()
            raise ValueError("Fixture import failure")

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with patch.object(m, "_input_gate", return_value="fixture_only"), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "device_count", return_value=1), \
                    patch.object(torch.random, "fork_rng", side_effect=lambda devices: original_fork(devices=[])), \
                    patch("offline_study.vendor.use_vendor", side_effect=failed_import):
                with self.assertRaisesRegex(ValueError, "Fixture import failure"):
                    m.build_receiving_model(self.contracts["pusht"], None, input_proof="fixture")
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        assert_same(before, (torch.get_rng_state(), np.random.get_state(), random.getstate()))


if __name__ == "__main__":
    unittest.main()
