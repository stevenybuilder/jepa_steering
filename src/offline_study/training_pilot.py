"""Fit-only navigation training engineering against the actual upstream step.

Preserve 16 logical batches of eight (global128), source losses and optimizer.
No checkpoint history, validation outcomes, fitted intervention or seed selection.
"""
import argparse
import ast
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .droid_native import assert_same, verified_report
from .model_loader import verified_local_dino_cache
from .navigation_input_check import CONFIGS, SelectedFrameSlicer
from .protocol import sha256, write_json
from .vendor import use_vendor


class VirtualRankBatchSampler:
    """Exact upstream DistributedSampler + drop_last batches, without padding evals."""
    def __init__(self, length, ranks=16, microbatch=8, epoch=0):
        self.length, self.ranks, self.microbatch, self.epoch = length, ranks, microbatch, epoch

    def __len__(self):
        return ((self.length + self.ranks - 1) // self.ranks) // self.microbatch

    def __iter__(self):
        from torch.utils.data import DistributedSampler
        streams = []
        for rank in range(self.ranks):
            sampler = DistributedSampler(range(self.length), num_replicas=self.ranks, rank=rank, shuffle=True)
            sampler.set_epoch(self.epoch)
            streams.append(list(sampler))
        for step in range(len(self)):
            yield [index for stream in streams for index in
                   stream[step * self.microbatch:(step + 1) * self.microbatch]]


def build_model(cfg, dataset, ipe):
    from app.vjepa_wm.utils import init_video_model, init_opt
    from app.vjepa_wm.video_wm import VideoWM
    m = cfg["model"]
    excluded = {"rollout_cfg", "heads_cfg", "pretrained_path", "visual_encoder", "action_encoder",
                "proprio_encoder", "predictor", "wm_encoding", "attn"}
    kwargs = {k: v for k, v in m.items() if k not in excluded}
    for key in ("visual_encoder", "action_encoder", "proprio_encoder", "predictor"):
        kwargs.update(m[key])
    action_dim, proprio_dim = dataset.action_dim * 5, dataset.proprio_dim
    kwargs.update(device=torch.device("cuda:0"), img_size=224, action_dim=action_dim,
                  proprio_dim=proprio_dim, cfgs_attn_pattern=m["attn"], use_proprio=True, use_action=True)
    with verified_local_dino_cache():
        predictor, encoder, action_encoder, proprio_encoder = init_video_model(**kwargs)
    opt_cfg = dict(cfg["optimization"]["transition_model"], iterations_per_epoch=ipe)
    optimizer, scaler, scheduler, wd = init_opt(predictor, action_encoder, proprio_encoder,
                                               encoder, freeze_encoder=True, **opt_cfg)
    model = VideoWM(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        proprio_encoder=proprio_encoder, device=torch.device("cuda:0"), action_dim=action_dim,
        proprio_dim=proprio_dim, use_proprio=True, use_action=True,
        action_tokens=m["action_encoder"]["action_tokens"], proprio_tokens=m["proprio_encoder"]["proprio_tokens"],
        grid_size=m["grid_size"], tubelet_size_enc=m["tubelet_size_enc"],
        action_conditioning=m["action_conditioning"], proprio_encoding=m["proprio_encoding"],
        enc_type=m["visual_encoder"]["enc_type"], pred_type=m["predictor"]["pred_type"],
        action_encoder_inpred=m["action_encoder"]["action_encoder_inpred"],
        proprio_encoder_inpred=m["proprio_encoder"]["proprio_encoder_inpred"], **m["wm_encoding"],
        action_skip=cfg["data"].get("action_skip", 1), frameskip=cfg["data"].get("frameskip", 1),
        img_size=224, heads={}, scaler=scaler, optimizer=optimizer, clip_grad=1,
        mixed_precision=True, use_radamw=False, cfgs_loss=cfg["loss"])
    return model, scheduler, wd


def native_step(vendor, model, scheduler, wd):
    """Compile the unchanged nested upstream step; no rewritten reference loss."""
    path = vendor / "app/vjepa_wm/train.py"
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "step_model"]
    if len(matches) != 1:
        raise ValueError("Native training step is missing or ambiguous")
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict, world_model=model,
        scheduler=scheduler, wd_scheduler=wd, predictor=model.predictor,
        train_predictor=True, train_heads=False, train_heads_on_predictor=False,
        dtype=torch.bfloat16, mixed_precision=True,
        rollout_steps=2, do_sequential_rollout=True, do_parallel_rollout=False,
        train_rollout_prefixes="random", rollout_stop_gradient=True, ctxt_window_train_rollout=3)
    exec(compile(ast.Module(body=[matches[0]], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["step_model"]


def objective(model, obs, action):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        visual, proprio, actions = model.encode(obs, action)
        pred_visual, _, pred_proprio = model.forward_pred(visual, actions, proprio)
        teacher = model.compute_loss(pred_visual, pred_proprio, visual, proprio, shift=1)["loss"] / 3
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prefix = torch.randint(visual.shape[1] - 2, size=(1,))[0]
        _, rollout, _, _ = model.rollout(video_features=visual, pred_video_features=pred_visual,
            proprio_features=proprio, pred_proprio_features=pred_proprio, action_features=actions,
            action_noise=0., loss_weight=1., rollout_steps=1, rollout_stop_gradient=True,
            ctxt_window=3, mode="sequential", t=prefix)
    return teacher + rollout


def microbatches(batch):
    obs, action, state, reward = batch
    if len(action) != 128:
        raise ValueError("Training update must contain all 16x8 logical examples")
    for rank in range(16):
        section = slice(rank * 8, (rank + 1) * 8)
        yield ({key: val[section].to("cuda:0", dtype=torch.bfloat16, non_blocking=True) for key, val in obs.items()},
               action[section].to("cuda:0", dtype=torch.bfloat16, non_blocking=True),
               state[section].to("cuda:0", dtype=torch.bfloat16, non_blocking=True),
               reward[section].to("cuda:0", dtype=torch.bfloat16, non_blocking=True))


def accumulated_update(model, scheduler, wd, batch, rng_states):
    rates = scheduler.step(), wd.step()
    losses = []
    for rank, (obs, action, _, _) in enumerate(microbatches(batch)):
        torch.set_rng_state(rng_states[rank])
        loss = objective(model, obs, action)
        rng_states[rank] = torch.get_rng_state()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss, no skipped observations")
        model.backward(loss / 16)
        losses.append(float(loss.detach()))
    model.optimization_step()
    return {"loss": sum(losses) / 16, "lr": rates[0], "weight_decay": rates[1]}


class FixedRate:
    def __init__(self, value):
        self.value = value

    def step(self):
        return self.value


def verify_accumulation(vendor, model, scheduler, wd, batch, rng_states):
    """Actual upstream 16 local backwards / mean versus scaled accumulation."""
    reference, rs, rw = copy.deepcopy((model, scheduler, wd))
    original_step = reference.optimization_step
    lr, decay = rs.step(), rw.step()
    reference.optimization_step = lambda: (None, None)
    run = native_step(vendor, reference, FixedRate(lr), FixedRate(decay))
    reference_rngs = [state.clone() for state in rng_states]
    cuda_rng = torch.cuda.get_rng_state(0)
    losses = []
    for rank, (obs, action, state, reward) in enumerate(microbatches(batch)):
        torch.set_rng_state(reference_rngs[rank])
        result = run(obs, action, state, reward, train=True)
        losses.append(result[0])
        reference_rngs[rank] = torch.get_rng_state()
    for parameter in reference.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(16)
    reference.optimization_step = original_step
    original_step()
    if not torch.equal(cuda_rng, torch.cuda.get_rng_state(0)):
        raise ValueError("Native training consumed CUDA RNG; independent logical GPU streams required")
    actual = accumulated_update(model, scheduler, wd, batch, rng_states)
    if not torch.equal(cuda_rng, torch.cuda.get_rng_state(0)):
        raise ValueError("Accumulated training consumed CUDA RNG unexpectedly")
    assert_same(reference.state_dict(), model.state_dict())
    assert_same(reference.optimizer.state_dict(), model.optimizer.state_dict())
    assert_same(reference.scaler.state_dict(), model.scaler.state_dict())
    assert_same(reference_rngs, rng_states)
    if actual["loss"] != sum(losses) / 16:
        raise ValueError("Native training loss differs")
    return {"native_source_step_used": True, "all_parameters_optimizer_moments_scaler_and_rng_bitwise": True,
            "logical_ranks": 16, "microbatch": 8, "global_batch": 128,
            "cuda_rng_unchanged_in_both_paths": True,
            "source_collective_reduction_not_reproduced_bitwise": True,
            "reference": "sum native per-rank scaled gradients then divide16; hardware all-reduce order not claimed",
            "loss": actual["loss"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "assets", "input-check", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--task", choices=("wall", "pointmaze"), required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    cfg = yaml.safe_load((args.vendor / CONFIGS[args.task]).read_text())
    assets, assets_hash = verified_report(args.assets)
    inputs, input_hash = verified_report(args.input_check)
    if (assets["status"] != "official_navigation_assets_staged_and_verified" or
            inputs["status"] != "navigation_metadata_and_selected_frame_parity_passed"):
        raise ValueError("Require completed input provenance and exact native loader parity")
    input_protocol = json.loads((args.input_check / "protocol.json").read_text())
    if (inputs["protocol_sha256"] != sha256(args.input_check / "protocol.json") or
            input_protocol["assets_report_sha256"] != assets_hash or
            input_protocol["source_sha256"] != sha256(Path(__file__).with_name("navigation_input_check.py")) or
            input_protocol["native_configs_sha256"][args.task] != sha256(args.vendor / CONFIGS[args.task])):
        raise ValueError("Input audit must bind these exact assets, source and task config")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", {"role": "native_training_accumulation_engineering_not_history",
            "task": args.task, "source_sha256": sha256(Path(__file__)),
            "native_training_source_sha256": sha256(args.vendor / "app/vjepa_wm/train.py"),
            "native_config_sha256": sha256(args.vendor / CONFIGS[args.task]),
            "assets_report_sha256": assets_hash, "input_check_report_sha256": input_hash,
            "initialization_seed": 234, "updates": 5, "global_batch": 128,
            "logical_rank_microbatches": "16x8, exact source sampler and drop-last; one optimizer/scheduler update",
            "rng": "separate retained logical CPU streams initialized to same post-construction RNG; no diagnostic outcomes feed training",
            "no_author_exact_training_rng_or_collective_reduction_claim": True,
            "numerical_gate": "bitwise source local-objective gradient mean versus accumulation, weights/moments/scaler/RNG",
            "primary_precision": "bfloat16", "validation_or_confirmation_access": False,
            "checkpoint_history_complete": False, "no_checkpoint_or_seed_selection": True})
        from app.plan_common.datasets.transforms import make_transforms
        from app.plan_common.datasets.wall_dset import WallDataset
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.traj_dset import get_train_val_sliced
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(234)
        np.random.seed(234)
        torch.manual_seed(234)
        raw = args.assets / "extracted" / args.task / ("wall_single" if args.task == "wall" else "point_maze")
        cls = WallDataset if args.task == "wall" else PointMazeDataset
        dataset = cls(data_path=str(raw), n_rollout=None, normalize_action=True,
                      transform=make_transforms(img_size=224, **cfg["data_aug"]))
        train, validation, train_clips, _ = get_train_val_sliced(dataset, train_fraction=.9,
            random_seed=234, num_frames=4, num_frames_val=cfg["data"]["validation"]["num_frames_val"],
            frameskip=5, action_skip=1)
        sampler = VirtualRankBatchSampler(len(train_clips))
        loader = torch.utils.data.DataLoader(SelectedFrameSlicer(train_clips), batch_sampler=sampler,
            num_workers=4, pin_memory=True, persistent_workers=False,
            generator=torch.Generator().manual_seed(2026090725))
        model, scheduler, wd = build_model(cfg, dataset, len(sampler))
        encoder_versions = [(name, p._version) for name, p in model.encoder.named_parameters()]
        rng_states = [torch.get_rng_state().clone() for _ in range(16)]
        iterator = iter(loader)
        batch = next(iterator)
        parity = verify_accumulation(args.vendor, model, scheduler, wd, batch, rng_states)
        write_json(args.output / "PARITY.json", parity)
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for update in range(1, 5):
            before_input = time.monotonic()
            batch = next(iterator)
            loaded = time.monotonic()
            torch.cuda.synchronize()
            before_gpu = time.monotonic()
            values = accumulated_update(model, scheduler, wd, batch, rng_states)
            torch.cuda.synchronize()
            row = {"update": update + 1, **values, "input_wait_seconds": loaded - before_input,
                   "update_seconds": time.monotonic() - before_gpu}
            timings.append(row)
            write_json(args.output / "progress.json", {"updates_complete": update + 1, "timings": timings})
            print(json.dumps(row), flush=True)
        if encoder_versions != [(name, p._version) for name, p in model.encoder.named_parameters()]:
            raise ValueError("Frozen encoder was modified")
        write_json(args.output / "report.json", {"status": "native_training_accumulation_pilot_passed",
            "protocol_sha256": sha256(args.output / "protocol.json"), "parity_sha256": sha256(args.output / "PARITY.json"),
            "task": args.task, "training_rows": len(train), "unopened_validation_rows": len(validation),
            "training_clips": len(train_clips), "updates_per_epoch": len(sampler),
            "updates_complete": 5, "timings": timings, "seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "encoder_unchanged": True,
            "checkpoint_history_complete": False, "validation_outcomes_accessed": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "checkpoint_history_complete": False})
        raise


if __name__ == "__main__":
    main()
