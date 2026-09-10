import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from offline_study.training_history import (checkpoint_payload, restore_checkpoint, training_config, validation_step)
from offline_study.training_pilot import FixedRate
from offline_study.navigation_training_inputs import required_wall_files


class EmptyScaler:
    def state_dict(self):
        return {}
    def load_state_dict(self, state):
        if state != {}:
            raise ValueError("Unexpected scaler state")


class MonitorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = torch.nn.Linear(2,2)
        self.action_encoder = torch.nn.Linear(2,2)
        self.proprio_encoder = torch.nn.Linear(2,2)
        self.optimizer = torch.optim.AdamW(self.parameters())
        self.scaler = EmptyScaler()
        self.heads = {}
        self.calls = []

    def encode(self, obs, action):
        return obs["visual"], obs["proprio"], action

    def rollout(self, **kwargs):
        self.calls.append({key:kwargs[key] for key in ("rollout_steps","ctxt_window","action_noise","t")})
        value = torch.arange(kwargs["rollout_steps"],dtype=torch.float32) + kwargs["action_noise"]
        return {"visual_mse":value}, value.sum(), None, None


class TrainingHistoryTests(unittest.TestCase):
    def test_source_validation_executes_both_noise_settings_all_h6_prefixes(self):
        vendor = Path(__file__).resolve().parents[1]/"vendor/jepa-wms"
        cfg = training_config(vendor)
        model = MonitorModel()
        run = validation_step(vendor, model, cfg, FixedRate(0), FixedRate(0))
        obs = {"visual":torch.ones(2,8,2), "proprio":torch.ones(2,8,2)}
        with patch("torch.amp.autocast", side_effect=lambda *a,**kw:nullcontext()):
            result = run(obs, torch.ones(2,8,2), torch.ones(2,8,2), torch.zeros(2,8), train=False)
        self.assertEqual([row["t"] for row in model.calls], [0,1,0,1])
        self.assertEqual([row["action_noise"] for row in model.calls], [0,0,.05,.05])
        self.assertTrue(all(row["rollout_steps"]==6 and row["ctxt_window"]==3 for row in model.calls))
        self.assertIn("data_traj/val_rollout/visual_mse/6",result[3])
        self.assertIn("data_traj/noisy_val_rollout/visual_mse/6",result[3])
        self.assertTrue(model.training)

    def test_checkpoint_roundtrip_and_resume_scope(self):
        model=MonitorModel()
        cfg={"model": {"action_encoder":{},"proprio_encoder":{}}}
        scheduler=SimpleNamespace(_step=418)
        wd=SimpleNamespace(_step=418)
        cpu=[torch.Generator().manual_seed(i).get_state() for i in range(16)]
        loader=torch.Generator().manual_seed(5)
        payload=copy.deepcopy(checkpoint_payload(model,cfg,1,cpu,cpu,scheduler,wd,loader,{"seed":234},2))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        restored,_,events=restore_checkpoint(payload,model,cfg,scheduler,wd,loader,{"seed":234})
        self.assertEqual(events,2)
        self.assertTrue(torch.equal(restored[0],cpu[0]))
        self.assertTrue(torch.equal(model.predictor.weight,payload["predictor"]["weight"]))
        for key,value in (("scheduler_step",417),("validation_events",1),("epoch_boundary_only",False)):
            bad=copy.deepcopy(payload)
            bad["study_resume"][key]=value
            with self.assertRaises(ValueError):
                restore_checkpoint(bad,model,cfg,scheduler,wd,loader,{"seed":234})
        with self.assertRaises(ValueError):
            restore_checkpoint(payload,model,cfg,scheduler,wd,loader,{"seed":235})

    def test_full_population_is_not_fit_subset(self):
        files=required_wall_files()
        self.assertEqual(len(files),1924)
        self.assertEqual(len(set(files)),1924)
        self.assertIn("obses/episode_1919.pth",files)
