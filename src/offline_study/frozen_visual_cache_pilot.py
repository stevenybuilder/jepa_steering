"""Bounded PointMaze cache engineering, never a training history or activation.

Launch on one explicitly reserved GPU with a hard outer process timeout, e.g.
``timeout --signal=TERM --kill-after=5s 595s python -m
offline_study.frozen_visual_cache_pilot ...``. This process also has a 580-second
GPU-stage alarm; no failure may turn into a truncated passing receipt.

The first two complete native 128-clip updates are fixed before model access.
One full native validation event is inserted between them ONLY in excluded
engineering clones, to test RNG propagation without requiring 200 training
updates. This is not the authors' validation cadence or a checkpoint history.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, nullcontext
import copy
from dataclasses import asdict
import itertools
import inspect
import json
import logging
from pathlib import Path
import random
import signal
import time
from typing import Callable
import warnings

import numpy as np
import torch

from . import frozen_visual_cache as cache
from .behavioral_development import verified_report
from .model_loader import DINO_SOURCE_SHA256, DINO_WEIGHT_SHA256
from .navigation_input_check import CONFIGS, SelectedFrameSlicer
from .pointmaze_training_history import (SAMPLER_POLICY, UPDATES, ValidationFrames,
    monitor, source_bindings, training_config, validation_batches)
from .pointmaze_training_inputs import verify_inputs
from .protocol import sha256, write_json
from .training_history import validation_step, verify_validation_reader
from .training_pilot import (VirtualRankBatchSampler, accumulated_update, build_model,
    microbatches, verify_accumulation)
from .vendor import use_vendor


GPU_STAGE_LIMIT_SECONDS = 580
REQUIRED_OUTER_TIMEOUT_SECONDS = 600
SCRATCH_LIMIT_BYTES = 1 << 30
CACHE_MEMORY_BYTES = 64 << 20
ENGINEERING_SEED = 234
LOADER_SEED = 2026090725
UPDATE_INDICES = (0, 1)
PROTOCOL_VERSION = "pointmaze_visual_cache_native_modes_engineering_v2"
V1_FAILURE = {"root": "visual-cache-engineering-20260908-v1",
    "failed_receipt_sha256": "a16050f3c235290d492258fc444395325995c11cff0ba28d79c41b974b07dd77",
    "protocol_sha256": "74c65872b45f4dab0fd2111304b926f62cb9d3058b12a65f7e6ffd7b348f81d0",
    "reason": "Native validation ends world_model.train(); default eval-only cache guard refused update1",
    "numerical_parity_failure_observed": False, "complete_update_parity_established": False}
DINO_MODE_FILES = {
    "dinov2/models/vision_transformer.py": "7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea",
    "dinov2/layers/block.py": "60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636",
    "dinov2/layers/attention.py": "79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d",
    "dinov2/layers/mlp.py": "255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa",
    "dinov2/layers/patch_embed.py": "40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142",
    "dinov2/layers/layer_scale.py": "dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135",
    "dinov2/layers/drop_path.py": "b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a",
    "dinov2/hub/backbones.py": "871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e"}


def native_dino_mode_policy(encoder, binding, vendor):
    """Reviewed Sep8 pinned source; a whitelist, not generic train-mode caching.

    DinoEncoder calls forward_features(Tensor), not DINO.forward(is_training=...).
    Tensor blocks take identical branches at sample_drop_ratio=0; attention SDPA
    dropout_p is zero in both modes. MLP/projection Dropout(p=0) is identity.
    No BatchNorm, registered buffers, nested-list attention-bias cache, mask input,
    trainable visual parameters or nonzero DropPath is allowed by this contract.
    These source facts do NOT substitute for the receiving bitwise/RNG proof.
    """
    from .model_loader import verified_local_dino_cache
    with verified_local_dino_cache():  # Hashes only; never calls torch.hub.load here.
        pass
    root = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    for relative, expected in DINO_MODE_FILES.items():
        if sha256(root / relative) != expected:
            raise ValueError("Pinned mode-sensitive DINO source changed")
    wrapper = vendor / "app/plan_common/models/dino.py"
    if (sha256(wrapper) != "6994115d43796ceda054509eb6bac9b9ea62ae1115b4d0f9dae25122f7ab0e9f" or
            sha256(vendor / "app/vjepa_wm/train.py") != "c1fc4c57b99cab18df14405236adf463363fa4eef23705635a2a48e5d6e98285"):
        raise ValueError("Pinned native encoder/validation mode lifecycle changed")
    classes = {
        "app.plan_common.models.dino.DinoEncoder": sha256(wrapper),
        "dinov2.models.vision_transformer.DinoVisionTransformer": DINO_MODE_FILES["dinov2/models/vision_transformer.py"],
        "dinov2.layers.block.NestedTensorBlock": DINO_MODE_FILES["dinov2/layers/block.py"],
        "dinov2.layers.attention.MemEffAttention": DINO_MODE_FILES["dinov2/layers/attention.py"],
        "dinov2.layers.mlp.Mlp": DINO_MODE_FILES["dinov2/layers/mlp.py"],
        "dinov2.layers.patch_embed.PatchEmbed": DINO_MODE_FILES["dinov2/layers/patch_embed.py"],
        "dinov2.layers.layer_scale.LayerScale": DINO_MODE_FILES["dinov2/layers/layer_scale.py"]}
    if (type(encoder).__module__ + "." + type(encoder).__qualname__ != "app.plan_common.models.dino.DinoEncoder" or
            encoder.name != "dinov2_vits14" or encoder.feature_key != "x_norm_patchtokens" or
            encoder.latent_ndim != 2 or encoder.base_model.chunked_blocks or
            encoder.base_model.bag_of_channels or encoder.base_model.num_register_tokens != 0 or
            encoder.base_model.n_blocks != 12 or encoder.base_model.embed_dim != 384):
        raise ValueError("Actual receiving encoder differs from the reviewed DINO-S/14 tensor path")
    backend = {}
    for name in ("dinov2.layers.block", "dinov2.layers.attention"):
        import sys
        module = sys.modules[name]
        backend[name] = {"XFORMERS_AVAILABLE": module.XFORMERS_AVAILABLE,
            "XFORMERS_ENABLED": module.XFORMERS_ENABLED}
        # Pinned receiving runtime uses PyTorch SDPA. Do not silently bless a
        # different external kernel or the stateful nested-list cache path.
        if module.XFORMERS_AVAILABLE:
            raise ValueError("Unaudited xFormers backend; native SDPA mode proof required")
    for module in encoder.modules():
        cls = type(module)
        if cls.__module__.startswith("dinov2.") and not Path(inspect.getfile(cls)).resolve().is_relative_to(root.resolve()):
            raise ValueError("Loaded DINO class is outside its verified source root")
    return cache.EngineeringModePolicy(encoder, binding, audited_classes=classes,
        source_receipt={"source_sha256": binding.source_sha256, "dino_source_sha256": DINO_SOURCE_SHA256,
            "dino_weight_sha256": DINO_WEIGHT_SHA256, "mode_sensitive_files": DINO_MODE_FILES,
            "backend": backend, "native_validation_train_restore_line": 1307,
            "observed_modes": "initial_eval; native_validation_eval_then_train",
            "review_basis": "zero stochastic rates; frozen buffer-free Tensor-only SDPA DINO-S/14 path"})


def actual_native_schedule(vendor, cfg, train_clips, val_clips, train_rows, val_rows):
    """Execute the original caller and init_data; replace only repeat dataset IO.

    Real, already-verified native datasets are reused across the 16 sampler
    constructions. No DataLoader is iterated here, so native16-worker kwargs do
    not create256 workers. The caller's kwargs and native sampler flags execute
    unchanged rather than being re-specified by this pilot.
    """
    source = vendor / "app/plan_common/datasets/utils.py"
    function = next(node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "init_data")
    training_source = vendor / "app/vjepa_wm/train.py"
    main = next(node for node in ast.parse(training_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main")
    first = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == "excluded_keys" for target in node.targets))
    last = next(i for i, node in enumerate(main.body) if i > first and isinstance(node, ast.Assign) and
        isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "init_data")
    caller = compile(ast.Module(body=main.body[first:last + 1], type_ignores=[]), str(training_source), "exec")
    data = cfg["data"]
    def dataset_io(transform, **kwargs):
        expected = {"n_rollout": None, "normalize_action": True, "split_ratio": .9,
            "num_hist": 3, "num_pred": 1, "num_frames_val": 8, "frameskip": 5,
            "action_skip": 1, "traj_subset": True, "random_seed": 234,
            "process_actions": "concat", "dset_fraction": 1}
        if any(kwargs.get(key) != value for key, value in expected.items()):
            raise ValueError("Actual upstream dataset construction differs from verified input setup")
        return {"train": train_clips, "valid": val_clips}, {"train": train_rows, "valid": val_rows}
    logger = logging.getLogger("frozen-visual-cache-native-sampler")
    logger.setLevel(logging.CRITICAL + 1)
    namespace = {"torch": torch, "Callable": Callable, "logger": logger,
        "load_point_maze_slice_train_val": dataset_io}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    updates, validation = [[], []], []
    for rank in range(16):
        arguments = {"cfgs_data": data, "cfgs_validation": data["validation"],
            "cfgs_loader": data["loader"], "cfgs_custom": data["custom"], "cfgs_droid": data["droid"],
            "dataset_paths": ["/verified/point_maze"], "val_dataset_paths": [], "transform": None,
            "world_size": cfg["nodes"] * cfg["tasks_per_node"], "rank": rank,
            "filter_first_episodes": data["custom"]["filter_first_episodes"],
            "num_workers": data["loader"]["num_workers"], "init_data": namespace["init_data"]}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This DataLoader will create .*worker processes.*")
            exec(caller, arguments)
        train_loader, val_loader = arguments["unsupervised_loader"], arguments["val_unsupervised_loader"]
        if (train_loader.sampler.shuffle or val_loader.sampler.shuffle or len(train_loader) != UPDATES or
                len(val_loader) != 191 or train_loader.batch_size != 8 or val_loader.batch_size != 4):
            raise ValueError("Corrected PointMaze native sampler contract changed")
        train_loader.sampler.set_epoch(0)
        native = list(itertools.islice(train_loader.batch_sampler, 2))
        for update in UPDATE_INDICES:
            updates[update].extend(native[update])
        validation.append(next(iter(val_loader.batch_sampler)))
    adapted = list(itertools.islice(VirtualRankBatchSampler(len(train_clips), shuffle=False), 2))
    if updates != adapted or validation != validation_batches(len(val_clips))[0]:
        raise ValueError("Current adapter differs from the actual upstream caller/sampler")
    return updates, validation


def ordered_frames(slicer, indices, files):
    """Resolve native slice/subset identities without reading or transforming images."""
    from app.plan_common.datasets.traj_dset import TrajSubset
    result = []
    for index in indices:
        row, start, end = slicer.slices[index]
        base = slicer.dataset
        while type(base) is TrajSubset:
            row, base = int(base.indices[row]), base.dataset
        name = f"obses/episode_{row:03d}.pth"
        if end - start != 20 or slicer.num_frames != 4 or slicer.frameskip != 5:
            raise ValueError("Original four-frame clip contract changed")
        result.extend(cache.Frame(name, files[name]["sha256"], frame) for frame in range(start, end, 5))
    if len(result) != 512:
        raise ValueError("A complete 128-clip update must contain512 ordered frames")
    return result


def two_batches(training, indices, loader_rng):
    """Original selected-frame transforms run in both native and cached paths."""
    loader = torch.utils.data.DataLoader(training, batch_sampler=indices, num_workers=4,
        pin_memory=True, persistent_workers=False, generator=loader_rng)
    iterator = iter(loader)
    try:
        result = list(iterator)
    finally:
        # This pilot created these CPU workers; never acts on an existing job.
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    if len(result) != 2 or any(len(batch[1]) != 128 for batch in result):
        raise ValueError("Partial engineering updates cannot pass")
    return result


def tensor_evidence(value):
    """Compact full-state evidence: exact bytes/layout, no huge checkpoint artifact."""
    if isinstance(value, torch.Tensor):
        return {"sha256": cache.tensor_digest(value), "dtype": str(value.dtype),
            "shape": list(value.shape), "stride": list(value.stride()), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(key): tensor_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_evidence(item) for item in value]
    if value is None or type(value) in (bool, float, int, str):
        return value
    raise ValueError("Unsupported native state in complete parity evidence")


class EncoderTiming:
    """CUDA-event encoder-boundary timing, separate from synchronized update wall time."""
    def __init__(self):
        self.events, self.wall = [], []

    def call(self, function, images):
        events = None
        if images.device.type == "cuda":
            events = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            events[0].record()
        started = time.monotonic()
        result = function(images)
        self.wall.append(time.monotonic() - started)
        if events is not None:
            events[1].record()
            self.events.append(events)
        return result

    def report(self):
        return {"encoder_calls": len(self.wall), "encoder_boundary_host_seconds": sum(self.wall),
            "encoder_boundary_stream_seconds": sum(start.elapsed_time(end) / 1000 for start, end in self.events),
            "stream_timing_includes_cpu_stalls_inside_cached_boundary": True}


@contextmanager
def routed_encoder(encoder, frames, store=None, mode_policy=None):
    """One scope per unchanged accumulated_update; validation stays uncached."""
    if len(frames) != 512:
        raise ValueError("Require all16 native32-image encoder calls")
    session = cache.EncoderCacheSession(encoder, store, mode="engineering_replay",
        mode_policy=mode_policy) if store else None
    scope = session if session else nullcontext()
    timing, calls = EncoderTiming(), 0
    with scope:
        original, had_forward = encoder.forward, "forward" in encoder.__dict__
        def forward(images):
            nonlocal calls
            if calls >= 16 or images.shape[0] != 32:
                raise ValueError("Native encoder batch count/composition changed")
            selected = frames[calls * 32:(calls + 1) * 32]
            calls += 1
            with session.frames(selected) if session else nullcontext():
                return timing.call(original, images)
        encoder.forward = forward
        try:
            yield timing
            if calls != 16:
                raise ValueError("Incomplete native encoder-call sequence")
        finally:
            if had_forward:
                encoder.forward = original
            else:
                del encoder.forward


@contextmanager
def gradient_evidence(model, enabled):
    """Observe exact gradients before native unscale/clip; never rewrite updates."""
    rows = []
    original, had_method = model.optimization_step, "optimization_step" in model.__dict__
    def step():
        rows.append(tensor_evidence({name: parameter.grad for name, parameter in model.named_parameters()}))
        return original()
    if enabled:
        model.optimization_step = step
    try:
        yield rows
    finally:
        if enabled:
            if had_method:
                model.optimization_step = original
            else:
                del model.optimization_step


def run_updates(initial, training, indices, frame_rows, input_evidence, initial_cpu, initial_cuda,
                vendor, cfg, validation, val_schedule, store, *, capture_gradients, mode_policy=None):
    model, scheduler, wd = copy.deepcopy(initial)
    cpu, cuda = [state.clone() for state in initial_cpu], [state.clone() for state in initial_cuda]
    loader_rng = torch.Generator().manual_seed(LOADER_SEED)
    load_started = time.monotonic()
    batches = two_batches(training, indices, loader_rng)
    load_seconds = time.monotonic() - load_started
    if tensor_evidence(batches) != input_evidence:
        raise ValueError("Reloaded original transformed inputs changed")
    losses, timings, validation_evidence, mode_events = [], [], None, []
    callback_started = time.monotonic()
    with gradient_evidence(model, capture_gradients) as gradients:
        for update, batch in enumerate(batches):
            # Context bookkeeping/weight hashing is outside step-only timing and
            # explicitly included in callback wall time below.
            mode_events.append({"event": "before_update" + str(update),
                "encoder_modes": {name: module.training for name, module in model.encoder.named_modules()}})
            if set(mode_events[-1]["encoder_modes"].values()) != {bool(update)}:
                raise ValueError("Native initial-eval / post-validation-train lifecycle changed")
            with routed_encoder(model.encoder, frame_rows[update], store, mode_policy) as encoder_timing:
                torch.cuda.synchronize()
                started = time.monotonic()
                values = accumulated_update(model, scheduler, wd, batch, cpu)
                torch.cuda.synchronize()
                seconds = time.monotonic() - started
                timings.append({"update_index": UPDATE_INDICES[update], "seconds": seconds,
                    **encoder_timing.report(), "gradient_observation_included": capture_gradients})
            losses.append(values)
            if update == 0:
                # Use the entire first native64-clip,16-rank,8-frame validation
                # event, including BOTH noisy and recorded-action H6 rollouts.
                native = validation_step(vendor, model, cfg, scheduler, wd)
                validation_evidence = monitor(native, validation, val_schedule, 0, cpu, cuda)
                if validation_evidence["global_clips"] != 64:
                    raise ValueError("Native validation event was truncated")
    torch.cuda.synchronize()
    result = {"losses": losses, "parameters": tensor_evidence(model.state_dict()),
        "gradients": gradients, "optimizer": tensor_evidence(model.optimizer.state_dict()),
        "scaler": tensor_evidence(model.scaler.state_dict()),
        "scheduler": {"learning_rate": scheduler._step, "weight_decay": wd._step},
        "logical_rngs": tensor_evidence({"cpu": cpu, "cuda": cuda}),
        "loader_rng": tensor_evidence(loader_rng.get_state()), "validation_events": 1,
        "validation": validation_evidence, "updates": 2, "native_mode_events": mode_events}
    return result, {"updates": timings, "input_load_seconds": load_seconds,
        "callback_seconds": time.monotonic() - callback_started,
        "callback_excludes_initial_model_clone_and_input_loading": True}


def composition_orders():
    """All1024 predetermined frames, new neighbors and positions; no RNG draws."""
    return {"frame_major_cross_batch": [batch * 32 + position for position in range(32) for batch in range(32)],
            "reverse": list(reversed(range(1024)))}


def composition_checks(encoder, store, batches, frame_rows):
    frames = list(itertools.chain.from_iterable(frame_rows))
    cpu_frames = [batch[0]["visual"].flatten(0, 1) for batch in batches]
    reports = []
    for name, order in composition_orders().items():
        if sorted(order) != list(range(1024)):
            raise ValueError("Composition check lost or duplicated an input frame")
        with cache.EncoderCacheSession(encoder, store, mode="record") as session:
            for start in range(0, 1024, 32):
                chosen = order[start:start + 32]
                images = torch.stack([cpu_frames[index // 512][index % 512] for index in chosen]).to(
                    "cuda:0", dtype=torch.bfloat16)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16), session.frames([frames[i] for i in chosen]):
                    encoder(images)
            reports.append({"composition": name, "order_sha256": cache.digest(order),
                "full_encoder_batches": session.calls, "all_existing_frame_bits_equal": True})
    return reports


def mode_composition_checks(encoder, store, batches, frame_rows, policy):
    frames = list(itertools.chain.from_iterable(frame_rows))
    cpu_frames = [batch[0]["visual"].flatten(0, 1) for batch in batches]
    orders = {"original": list(range(1024)), **composition_orders()}
    def supplied(order):
        for start in range(0, 1024, 32):
            chosen = order[start:start + 32]
            images = torch.stack([cpu_frames[index // 512][index % 512] for index in chosen]).to(
                "cuda:0", dtype=torch.bfloat16)
            yield [frames[index] for index in chosen], images
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return policy.prove(encoder, store, frames, orders, supplied)


def check_budget(binding, frames):
    if binding.output_dtype != "torch.float32" or binding.output_shape != (32, 256, 384):
        raise ValueError("Receiving DINO2 output differs from audited FP32 patch-token contract")
    count = len(set(frames))
    conservative = count * (256 * 384 * 4 + 8192) + (16 << 20)
    if conservative > SCRATCH_LIMIT_BYTES:
        raise ValueError("Full predetermined engineering input exceeds1GiB; no truncation permitted")
    return {"unique_frames": count, "conservative_scratch_bytes": conservative,
            "limit_bytes": SCRATCH_LIMIT_BYTES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "data-root", "input-receipt", "input-check", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise TimeoutError("580-second GPU-stage limit; incomplete cache engineering cannot pass")
    try:
        use_vendor(args.vendor)
        cfg = training_config(args.vendor)
        inputs, input_hash = verify_inputs(args.data_root, args.input_receipt)
        checked, check_hash = verified_report(args.input_check)
        ip = json.loads((args.input_receipt / "protocol.json").read_text())
        audited = json.loads((args.input_check / "pointmaze.json").read_text())
        if (checked["status"] != "navigation_metadata_and_selected_frame_parity_passed" or
                ip["input_check_report_sha256"] != check_hash or
                checked["task_reports_sha256"]["pointmaze"] != sha256(args.input_check / "pointmaze.json")):
            raise ValueError("Native input/source audit changed")
        from app.plan_common.datasets.transforms import make_transforms
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.traj_dset import get_train_val_sliced
        random.seed(ENGINEERING_SEED); np.random.seed(ENGINEERING_SEED); torch.manual_seed(ENGINEERING_SEED)
        dataset = PointMazeDataset(str(args.data_root), n_rollout=None, normalize_action=True,
            transform=make_transforms(img_size=224, **cfg["data_aug"]))
        train, val, train_clips, val_clips = get_train_val_sliced(dataset, train_fraction=.9, random_seed=234,
            num_frames=4, num_frames_val=8, frameskip=5, action_skip=1)
        if ((len(dataset), len(train), len(val), len(train_clips), len(val_clips)) != (2000,1800,200,145800,12200) or
                list(train.indices) != audited["native_train_indices"] or list(val.indices) != audited["native_validation_indices"]):
            raise ValueError("Native complete PointMaze input/split population changed")
        indices, native_val = actual_native_schedule(args.vendor, cfg, train_clips, val_clips, train, val)
        files = json.loads((args.input_receipt / "files.json").read_text())
        frame_rows = [ordered_frames(train_clips, update, files) for update in indices]
        training, validation = SelectedFrameSlicer(train_clips), ValidationFrames(val_clips)
        validation_reader = verify_validation_reader(val_clips, validation)
        batches = two_batches(training, indices, torch.Generator().manual_seed(LOADER_SEED))
        input_evidence = tensor_evidence(batches)
        sources = {**source_bindings(args.vendor), "frozen_visual_cache.py": sha256(Path(cache.__file__)),
            "frozen_visual_cache_pilot.py": sha256(Path(__file__)),
            "native_video_wm.py": sha256(args.vendor / "app/vjepa_wm/video_wm.py"),
            "native_init_utils.py": sha256(args.vendor / "app/vjepa_wm/utils.py"),
            "dino_source_sha256": DINO_SOURCE_SHA256, "dino_weight_sha256": DINO_WEIGHT_SHA256}
        protocol = {"schema": PROTOCOL_VERSION, "prior_engineering_failure": V1_FAILURE,
            "role": "excluded_pointmaze_frozen_visual_cache_parity_and_timing", "task": "pointmaze",
            "source_sha256": sources, "native_config_sha256": sha256(args.vendor / CONFIGS["pointmaze"]),
            "input_report_sha256": input_hash, "input_check_report_sha256": check_hash,
            "seed": ENGINEERING_SEED, "sampler_policy": SAMPLER_POLICY, "update_indices": list(UPDATE_INDICES),
            "updates": indices, "validation_clip_indices_by_rank": native_val,
            "sequence": ["native_update0", "native_validation_event0", "native_update1"],
            "author200_update_validation_cadence_reproduced": False,
            "native_training_objective": "existing accumulated_update; additional actual native_step parity",
            "native_logical_batch": "16x8; each encoder call32images",
            "validation": "full64clips,16ranks,8frames; noisy and recorded-action H6; always uncached",
            "gradient_parity_pairs": 1, "timing_pairs": 1, "timing_order": ["cached", "native"],
            "cache_population_original_batches": 32, "alternate_compositions": list(composition_orders()),
            "mode_parity_batches": 192, "mode_parity_gate": "eval/train x original/two alternate compositions",
            "mode_changes_for_proof": "isolated encoder clone only; preserve native update/validation modes",
            "gpu_stage_limit_seconds": GPU_STAGE_LIMIT_SECONDS,
            "required_outer_process_timeout_seconds": REQUIRED_OUTER_TIMEOUT_SECONDS,
            "scratch_limit_bytes": SCRATCH_LIMIT_BYTES, "memory_cache_bytes": CACHE_MEMORY_BYTES,
            "scientific_activation": False, "training_history_complete": False, "fresh_confirmation": False}
        write_json(args.output / "protocol.json", protocol)
        write_json(args.output / "INPUTS.json", {"frames_by_update": [[asdict(f) for f in row] for row in frame_rows],
            "actual_transformed_batches": input_evidence, "validation_reader": validation_reader})
        preflight_seconds = time.monotonic() - started
        signal.signal(signal.SIGALRM, timeout); signal.alarm(GPU_STAGE_LIMIT_SECONDS)
        gpu_started = time.monotonic()
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        initial = build_model(cfg, dataset, UPDATES)
        model = initial[0]
        cpu = [torch.get_rng_state().clone() for _ in range(16)]
        cuda = [torch.cuda.get_rng_state(0).clone() for _ in range(16)]
        before = cache.rng_snapshot()
        sample = batches[0][0]["visual"][:8].to("cuda:0", dtype=torch.bfloat16).flatten(0, 1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.encoder(sample)
            precision = cache.precision_identity("cuda")
        cache.assert_bitwise(before, cache.rng_snapshot())
        frames = sorted(set(itertools.chain.from_iterable(frame_rows)), key=lambda f: (f.relative_path, f.index))
        binding = cache.Binding(task="pointmaze", source_sha256=cache.digest(sources),
            encoder_sha256=cache.encoder_digest(model.encoder),
            transform_sha256=cache.digest({"config": cfg["data_aug"],
                "transforms": sha256(args.vendor / "app/plan_common/datasets/transforms.py"),
                "spatial": sha256(args.vendor / "src/datasets/utils/video/transforms.py")}),
            input_manifest_sha256=inputs["files_sha256"], frame_inventory_sha256=cache.digest(cache.frame_inventory(frames)),
            runtime_sha256=cache.digest(cache.runtime_identity("cuda:0")), precision_sha256=cache.digest(precision),
            input_shape=tuple(sample.shape), input_stride=sample.stride(), input_dtype=str(sample.dtype),
            output_shape=tuple(output.shape), output_stride=output.stride(), output_dtype=str(output.dtype))
        capacity = check_budget(binding, frames)
        mode_policy = native_dino_mode_policy(model.encoder, binding, args.vendor)
        write_json(args.output / "MODE_POLICY.json", mode_policy.receipt())
        write_json(args.output / "BINDING.json", {"binding": asdict(binding), "capacity": capacity})
        with cache.CacheStore.create(args.output / "cache", binding, frames,
                max_disk_bytes=SCRATCH_LIMIT_BYTES - (16 << 20), min_free_bytes=2 << 30,
                max_memory_bytes=CACHE_MEMORY_BYTES) as store:
            with cache.EncoderCacheSession(model.encoder, store, mode="record") as session:
                for update, batch in enumerate(batches):
                    for rank, (obs, _, _, _) in enumerate(microbatches(batch)):
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16), session.frames(frame_rows[update][rank*32:(rank+1)*32]):
                            model.encoder(obs["visual"].flatten(0, 1))
                if session.calls != 32:
                    raise ValueError("Incomplete original-batch cache population")
            compositions = mode_composition_checks(model.encoder, store, batches, frame_rows, mode_policy)
            write_json(args.output / "COMPOSITIONS.json", compositions)
            parity_model, parity_lr, parity_wd = copy.deepcopy(initial)
            native_parity = verify_accumulation(args.vendor, parity_model, parity_lr, parity_wd,
                batches[0], [state.clone() for state in cpu])
            del parity_model, parity_lr, parity_wd
            cache.restore_rng(before)
            captured = {}
            def run(label, gradients):
                evidence, timings = run_updates(initial, training, indices, frame_rows, input_evidence,
                    cpu, cuda, args.vendor, cfg, validation, validation_batches(),
                    store if label == "cached" else None, capture_gradients=gradients, mode_policy=mode_policy)
                captured[label] = {"evidence": evidence, "timings": timings}
                return evidence
            parity = cache.paired_update_check(lambda: run("native", True), lambda: run("cached", True))
            write_json(args.output / "UPDATE_PARITY.json", {"gate": parity, "actual_native_step_gate": native_parity,
                "native": captured["native"], "cached": captured["cached"],
                "all_tensor_state_comparisons_use_sha256_of_exact_bytes": True})
            # One fixed reverse-order pair, separate from expensive gradient
            # observation. The same two excluded updates are repeated on clones.
            timing_parity = cache.paired_update_check(lambda: run("cached", False), lambda: run("native", False))
            timed = {label: captured[label]["timings"] for label in ("native", "cached")}
            write_json(args.output / "TIMING.json", {"timing_parity": timing_parity, "paths": timed,
                "working_set": "bounded warm pilot; not full68GB cache/storage throughput proof",
                "order": ["cached", "native"], "repetitions_per_path": 1})
            native_seconds = sum(row["seconds"] for row in timed["native"]["updates"])
            cached_seconds = sum(row["seconds"] for row in timed["cached"]["updates"])
            encoder_seconds = sum(row["encoder_boundary_stream_seconds"] for row in timed["native"]["updates"])
            actual_cache_bytes = sum(path.stat().st_size for path in (args.output / "cache").rglob("*") if path.is_file())
        if cache.encoder_digest(model.encoder) != binding.encoder_sha256:
            raise ValueError("Original frozen encoder changed")
        complete = {"status": "bounded_pointmaze_visual_cache_engineering_passed",
            "protocol_sha256": sha256(args.output / "protocol.json"), "binding_sha256": binding.sha256,
            "receipts_sha256": {name: sha256(args.output / name) for name in
                ("INPUTS.json", "BINDING.json", "MODE_POLICY.json", "COMPOSITIONS.json", "UPDATE_PARITY.json", "TIMING.json")},
            "preflight_seconds": preflight_seconds, "gpu_stage_seconds": time.monotonic() - gpu_started,
            "full_update_bitwise_parity": True, "independent_batch_composition_bitwise_parity": True,
            "cache_bytes": actual_cache_bytes, "native_two_update_seconds": native_seconds,
            "cached_two_update_seconds": cached_seconds, "measured_step_speed_ratio": native_seconds / cached_seconds,
            "native_encoder_stream_seconds": encoder_seconds,
            "native_encoder_fraction_of_synchronized_update_wall": encoder_seconds / native_seconds,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "performance_measurement_scope": "single fixed warm two-update pair; gradient observation excluded",
            "full_cache_generation_or_scientific_activation": False, "complete_study": False}
        if complete["gpu_stage_seconds"] > GPU_STAGE_LIMIT_SECONDS or actual_cache_bytes > SCRATCH_LIMIT_BYTES:
            raise ValueError("Bounded pilot exceeded its registered resource limit")
        write_json(args.output / "report.json", complete)
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "seconds": time.monotonic() - started,
            "complete_study": False, "scientific_activation": False, "no_truncated_pass": True})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm)


if __name__ == "__main__":
    main()
