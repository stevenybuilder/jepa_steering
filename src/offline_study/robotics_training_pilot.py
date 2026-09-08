"""Receiving-only native MetaWorld/Push-T 32-rank update core; no launch CLI.

No raw-data reader, exposure authorization, history loop or completed GPU proof
is supplied here. Public update entrypoints require a bound input-parity receipt.
The numerical reference is the unchanged pinned upstream nested step_model.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import yaml

from .droid_native import assert_same, verified_report
from .protocol import sha256

CONFIGS = {
    "metaworld": "configs/vjepa_wm/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_AdaLN_ftprop_depth6_repro_2roll_save.yaml",
    "pusht": "configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml",
}
CONFIG_HASHES = {
    "metaworld": "db0c9cab6c12cb6541c222026358e7ccb482aa2276ed12b5a597ff4b547d0bba",
    "pusht": "7a16e8edbf40b260aab303949e90c1879dd5f778d51f3b6d6cb4bea1d99969e2",
}
SOURCE_HASHES = {
    "app/vjepa_wm/train.py": "c1fc4c57b99cab18df14405236adf463363fa4eef23705635a2a48e5d6e98285",
    "app/vjepa_wm/video_wm.py": "657bd3e9eb2ec35442cc06e208b4632a53503a35a83a96abbbf82608d2fe9154",
    "app/vjepa_wm/utils.py": "a81aaa29eaefce406bcd2f7f160b04cf233cd7c1fc21815be234da9f814ed7ef",
    "app/plan_common/datasets/utils.py": "e264c808ebdaa6bcd653538d7a2a53932a6cd985071e204efb5b966c6dff130a",
    "app/plan_common/datasets/traj_dset.py": "865f31178caf99c800b272148f876ab85d2f82ee97e2d7f18946459a80e8c893",
    "app/plan_common/datasets/metaworld_hf_dset.py": "83e5975c90870f316140a3d6db22887b9bb46a673ffbc0fd5cb3c33bcefb19c6",
    "app/plan_common/datasets/pusht_dset.py": "8a95851fbc2bc8a9ccc1bff2e7043362e36cf4e7a792c7e17fb14509c01fe167",
    "app/plan_common/datasets/transforms.py": "807446c6f944c2c771fdf8269dfac0ecab4a7fc7764e39d80b739eb0bdb13dc5",
    "src/utils/schedulers.py": "7d006141e10fd260850bd0e6d67deb1f4dfb23f6fe4d253f07df2f8971fdee48",
}
RANKS = 32


def _config(vendor, task):
    vendor = Path(vendor)
    if task not in CONFIGS:
        raise ValueError("Only MetaWorld and Push-T have a registered 32-rank core")
    for relative, expected in {**SOURCE_HASHES, CONFIGS[task]: CONFIG_HASHES[task]}.items():
        if sha256(vendor / relative) != expected:
            raise ValueError("Pinned native source/config changed: " + relative)
    return yaml.safe_load((vendor / CONFIGS[task]).read_text())


def native_loader_probe(vendor, task, train_length=257, validation_length=129, rank=0):
    """Actual caller + loader AST; metadata stand-ins only, no loader iteration.

    The source dataset path is replaced by a task-identifying metadata alias;
    native dataset constructors are the only I/O stubs. All sampler, DataLoader,
    config flattening and defaults are the original source, including workers.
    Constructing these loaders starts no workers and reads no protected rows.
    """
    cfg = _config(vendor, task)
    if any(type(n) is not int or n < 1 for n in (train_length, validation_length)):
        raise ValueError("Require positive metadata lengths")
    if type(rank) is not int or rank not in range(RANKS):
        raise ValueError("Invalid logical rank")
    calls = []

    def dataset_stub(transform, **kwargs):
        calls.append(kwargs)
        values = {"train": range(train_length), "valid": range(validation_length)}
        return values, values

    path = Path(vendor) / "app/plan_common/datasets/utils.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_data")
    namespace = {"torch": torch, "Callable": Callable,
        "logger": SimpleNamespace(info=lambda *a, **k: None),
        "load_metaworld_hf_slice_train_val": dataset_stub,
        "load_pusht_slice_train_val": dataset_stub}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    path = Path(vendor) / "app/vjepa_wm/train.py"
    main = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "excluded_keys" for t in n.targets))
    stop = next(i for i, n in enumerate(main.body[start:], start) if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                and n.value.func.id == "init_data")
    data = cfg["data"]
    namespace.update(cfgs_data=data, cfgs_validation=data["validation"],
        cfgs_loader=data["loader"], cfgs_custom=data["custom"], cfgs_droid=data["droid"],
        dataset_paths=[task], val_dataset_paths=[], transform=None,
        world_size=cfg["nodes"] * cfg["tasks_per_node"], rank=rank,
        filter_first_episodes=data["custom"].get("filter_first_episodes"),
        num_workers=data["loader"]["num_workers"])
    exec(compile(ast.Module(body=main.body[start:stop + 1], type_ignores=[]),
                 str(path), "exec"), namespace)
    if len(calls) != 1:
        raise ValueError("Native metadata branch changed")
    return namespace["unsupervised_loader"], namespace["val_unsupervised_loader"], calls[0]


def native_contract(vendor, task, model_seed=234):
    """CPU-only source/config-bound policy; not proof of pixel or GPU parity."""
    cfg = _config(vendor, task)
    if type(model_seed) is not int or model_seed not in (234, 235, 236):
        raise ValueError("Only the existing three registered reproduction seeds")
    train, val, call = native_loader_probe(vendor, task)
    policy = {"logical_ranks": train.sampler.num_replicas, "microbatch": train.batch_size,
        "global_batch": train.sampler.num_replicas * train.batch_size,
        "train_shuffle": train.sampler.shuffle, "validation_shuffle": val.sampler.shuffle,
        "sampler_seed": train.sampler.seed, "validation_sampler_seed": val.sampler.seed,
        "train_drop_last": train.drop_last, "validation_drop_last": val.drop_last,
        "sampler_drop_last": train.sampler.drop_last,
        "validation_microbatch": val.batch_size, "validation_sampler_epoch": val.sampler.epoch,
        "data_seed": call["random_seed"], "train_frames": call["num_hist"] + call["num_pred"],
        "validation_frames": call["num_frames_val"], "frameskip": call["frameskip"],
        "process_actions": call["process_actions"]}
    if (policy["logical_ranks"], policy["microbatch"], policy["global_batch"]) != (32, 8, 256):
        raise ValueError("Native 32x8 contract changed")
    return {"role": "native32_receiving_core_not_history", "task": task,
        "model_seed": model_seed, "authors_exact_seed_triplet_known": False,
        "vendor": str(Path(vendor).resolve()), "config_path": CONFIGS[task],
        "config_sha256": CONFIG_HASHES[task], "native_source_sha256": dict(SOURCE_HASHES),
        "sampler": policy, "config": cfg, "metadata_only": True,
        "complete_input_verification": False, "gpu_parity_established": False}


def _validate_contract(contract):
    if contract != native_contract(contract["vendor"], contract["task"], contract["model_seed"]):
        raise ValueError("Receiving contract differs from actual pinned caller")


class Native32BatchSampler:
    """Yield 32 separate index lists, retaining native partial validation tails."""
    def __init__(self, contract, length, epoch=0, validation=False):
        _validate_contract(contract)
        if type(epoch) is not int or epoch < 0 or (validation and epoch != 0):
            raise ValueError("Validation cursor wraps at native sampler epoch zero")
        if type(length) is not int or length <= 0:
            raise ValueError("Require positive clip count")
        self.contract, self.length, self.epoch, self.validation = contract, length, epoch, validation
        p = contract["sampler"]
        self.batch_size = p["validation_microbatch"] if validation else p["microbatch"]
        self.drop_last = p["validation_drop_last"] if validation else p["train_drop_last"]

    def _rank(self, rank):
        p = self.contract["sampler"]
        sampler = torch.utils.data.DistributedSampler(range(self.length), num_replicas=RANKS,
            rank=rank, shuffle=p["validation_shuffle"] if self.validation else p["train_shuffle"],
            seed=p["validation_sampler_seed"] if self.validation else p["sampler_seed"],
            drop_last=p["sampler_drop_last"])
        sampler.set_epoch(self.epoch)
        return torch.utils.data.BatchSampler(sampler, self.batch_size, self.drop_last)

    def __len__(self):
        return len(self._rank(0))

    def __iter__(self):
        yield from zip(*(iter(self._rank(rank)) for rank in range(RANKS)))


class _CudaState:
    def __init__(self, device):
        self.device = device

    def get(self):
        return torch.cuda.get_rng_state(self.device).clone()

    def set(self, state):
        torch.cuda.set_rng_state(state, self.device)


class LogicalRNGStreams:
    """32 explicit model RNG streams; ambient caller RNG is always restored.

    Initialization/loader consumption must come from the separately verified
    native initialization/reader contract, not an assumed seed reset per batch.
    _backend is only a private CPU-test seam, never accepted by public GPU gates.
    """
    def __init__(self, states, device=0, *, _backend=None):
        self.backend = _CudaState(device) if _backend is None else _backend
        self.device, self.states, self.poisoned = device, copy.deepcopy(states), False
        if len(self.states) != RANKS:
            raise ValueError("Exactly 32 explicit logical RNG states required")
        for state in self.states:
            if set(state) != {"cpu", "cuda", "python", "numpy"}:
                raise ValueError("Missing logical RNG component")
            for key in ("cpu", "cuda"):
                value = state[key]
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
                    raise ValueError("Invalid logical CPU/CUDA RNG byte state")

    def _capture(self):
        return {"cpu": torch.get_rng_state().clone(), "cuda": self.backend.get(),
                "python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state())}

    def _restore(self, state):
        torch.set_rng_state(state["cpu"])
        self.backend.set(state["cuda"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])

    @contextmanager
    def rank(self, rank):
        if self.poisoned or type(rank) is not int or rank not in range(RANKS):
            raise ValueError("Invalid or failed logical RNG stream; never auto-retry")
        ambient = self._capture()
        try:
            self._restore(self.states[rank])
            yield
            self.states[rank] = self._capture()
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._restore(ambient)

    def state_dict(self):
        return {"logical_ranks": RANKS, "device": self.device,
                "states": copy.deepcopy(self.states), "poisoned": self.poisoned}

    def clone(self):
        if self.poisoned:
            raise ValueError("Failed streams cannot become clean resume states")
        return LogicalRNGStreams(self.states, self.device, _backend=self.backend)


def native_reference_step(contract, model, scheduler, wd):
    """Compile the actual upstream step, with values from its pinned config.

    Exposes train=True only: validation needs its separate native input/cursor
    and protected-access contract, which this receiving core does not implement.
    """
    _validate_contract(contract)
    path = Path(contract["vendor"]) / "app/vjepa_wm/train.py"
    nodes = [n for n in ast.walk(ast.parse(path.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == "step_model"]
    if len(nodes) != 1:
        raise ValueError("Native step missing or ambiguous")
    cfg = contract["config"]
    r = cfg["model"]["rollout_cfg"]
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict, world_model=model,
        scheduler=scheduler, wd_scheduler=wd, predictor=model.predictor,
        train_predictor=cfg["optimization"]["main_optimizer"] == "transition_model",
        train_heads=cfg["optimization"]["train_heads"], train_heads_on_predictor=False,
        dtype=torch.bfloat16, mixed_precision=cfg["optimization"]["transition_model"]["mixed_precision"],
        **{k: r[k] for k in ("rollout_steps", "do_sequential_rollout", "do_parallel_rollout",
                            "train_rollout_prefixes", "rollout_stop_gradient", "ctxt_window_train_rollout")})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return lambda obs, action, state, reward: namespace["step_model"](obs, action, state, reward, train=True)


class _FixedRate:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def step(self):
        self.calls += 1
        return self.value


def _gradients(model):
    return {name: None if p.grad is None else p.grad.detach().clone()
            for name, p in model.named_parameters()}


def _drive_update(contract, model, scheduler, wd, rank_batches, rng, *, mean_after=False):
    """Internal CPU-testable mechanics, never a receiving proof by themselves."""
    _validate_contract(contract)
    if len(rank_batches) != RANKS:
        raise ValueError("Require all 32 native eight-clip batches")
    if rng.poisoned or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Require clean gradients and unfailed logical streams")
    modes = {name: module.training for name, module in model.named_modules()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    ambient = rng._capture()
    initial_steps = scheduler._step, wd._step
    rates = scheduler.step(), wd.step()
    lr, decay = _FixedRate(rates[0]), _FixedRate(rates[1])
    run = native_reference_step(contract, model, lr, decay)
    backward, optimize = model.backward, model.optimization_step
    had_backward, had_optimize = "backward" in model.__dict__, "optimization_step" in model.__dict__
    losses = []
    try:
        def scaled_backward(loss):
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite loss: fail, never skip an observation")
            backward(loss if mean_after else loss / RANKS)
        model.backward = scaled_backward
        model.optimization_step = lambda: (None, None)
        for rank, batch in enumerate(rank_batches):
            if len(batch[1]) != 8:
                raise ValueError("Every logical batch must retain its eight native clips")
            with rng.rank(rank):
                losses.append(run(*batch)[0])
        if mean_after:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(RANKS)
        gradients = _gradients(model)
        if any(g is not None and not torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError("Nonfinite gradients: no skipped optimizer update")
        optimize()
        if (scheduler._step, wd._step) != (initial_steps[0] + 1, initial_steps[1] + 1) or lr.calls != RANKS or decay.calls != RANKS:
            raise ValueError("One real scheduler step per native global update required")
        assert_same(buffers, dict(model.named_buffers()))
        assert_same(modes, {name: module.training for name, module in model.named_modules()})
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Native optimizer did not clear gradients")
        assert_same(ambient, rng._capture())
    except BaseException:
        rng.poisoned = True
        raise
    finally:
        rng._restore(ambient)
        if had_backward: model.backward = backward
        else: del model.backward
        if had_optimize: model.optimization_step = optimize
        else: del model.optimization_step
    return {"losses": losses, "loss": sum(losses) / RANKS,
            "lr": rates[0], "weight_decay": rates[1], "scaled_gradients": gradients}


def _batch_digest(batch):
    digest = hashlib.sha256()
    obs, action, state, reward = batch
    for name, value in [("obs/" + k, obs[k]) for k in sorted(obs)] + [("action", action), ("state", state), ("reward", reward)]:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad:
            raise ValueError("Input proof binds detached CPU tensors before native BF16 transfer")
        digest.update(name.encode() + b"\0" + str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _input_gate(contract, batch, root):
    """Check native batch and supporting evidence; no raw-data decode/access grant."""
    root = Path(root)
    if (root / "FAILED.json").exists():
        raise ValueError("Failed input proof cannot authorize engineering")
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    from .robotics_training_inputs import DATA_REVISION, MANIFEST_HASHES, COUNTS, bound_manifest
    raw_root = Path(protocol["raw_inputs_receipt"])
    if (raw_root / "FAILED.json").exists():
        raise ValueError("Raw input verification failed")
    raw, raw_hash = verified_report(raw_root)
    expected_raw = {"status": "complete_robotics_raw_inputs_verified", "task": contract["task"],
        "data_revision": DATA_REVISION, "raw_manifest_sha256": MANIFEST_HASHES[contract["task"]],
        "training_rows": COUNTS[contract["task"]][0], "validation_rows": COUNTS[contract["task"]][1],
        "training_or_validation_authorized": False, "native_reader_parity_established": False,
        "model_or_data_loader_initialized": False}
    if (any(raw.get(k) != v for k, v in expected_raw.items()) or
            protocol.get("raw_input_report_sha256") != raw_hash or
            protocol.get("data_root") != raw.get("data_root")):
        raise ValueError("Native batch proof must bind exact completed raw-byte verification")
    required = {"status": "native32_training_batch_input_parity_verified", "task": contract["task"],
        "native_config_sha256": contract["config_sha256"], "native_source_sha256": SOURCE_HASHES,
        "native_pixels_actions_proprio_states_rewards_rng_equal": True,
        "all_selected_inputs_content_verified": True, "permitted_training_rows_only": True,
        "validation_or_confirmation_access": False, "engineering_only": True,
        "global_batch": 256, "sampler_policy": contract["sampler"]}
    if any(report.get(k) != v for k, v in required.items()):
        raise ValueError("Require exact native training-only input parity and provenance")
    indices = protocol.get("training_clip_indices")
    if not isinstance(indices, list) or len(indices) != 256 or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Missing exact rank-ordered 256-clip input identity")
    # Only Push-T currently has a registered real-input producer. MetaWorld
    # needs its separate protected-row-aware reader receipt, not generic flags.
    if contract["task"] != "pusht":
        raise ValueError("Native MetaWorld input-parity producer is not yet registered")
    expected_indices = [rank + 32 * offset for rank in range(32) for offset in range(8)]
    if (indices != expected_indices or protocol.get("epoch") != 0 or
            protocol.get("global_update_index") != 0 or protocol.get("task") != "pusht" or
            protocol.get("role") != "training_only_first_update_input_parity" or
            protocol.get("version") != 1):
        raise ValueError("Input proof is not the registered first native 32x8 update")
    for filename, expected in (("comparisons.json", report.get("comparisons_sha256")),
                               ("selected_inputs.json", protocol.get("selected_inputs_sha256"))):
        path = root / filename
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError("Missing or changed native batch supporting evidence: " + filename)
    source = Path(__file__).with_name("robotics_training_batch.py")
    if protocol.get("runtime", {}).get("producer_sha256") != sha256(source):
        raise ValueError("Native input proof producer source changed")
    manifest = bound_manifest("pusht", Path(protocol["provenance"]))
    selected = json.loads((root / "selected_inputs.json").read_text())
    if not isinstance(selected, dict) or any(not name.startswith("train/") or
            name not in manifest or record != manifest[name] for name, record in selected.items()):
        raise ValueError("Selected inputs must be exact original-manifest training bytes")
    comparisons = json.loads((root / "comparisons.json").read_text())
    records = comparisons.get("clips")
    if not isinstance(records, list) or len(records) != 256:
        raise ValueError("Require exactly 256 native comparison records")
    needed = {"train/" + name for name in
              ("states.pth", "rel_actions.pth", "velocities.pth", "seq_lengths.pkl")}
    previous_rng = comparisons.get("initial_rng_sha256")
    for position, record in enumerate(records):
        identity = record.get("identity")
        if (not isinstance(identity, list) or len(identity) != 3 or
                any(type(i) is not int for i in identity) or
                not 0 <= identity[0] < COUNTS["pusht"][0] or identity[1] < 0 or
                identity[2] != identity[1] + 20 or record.get("rank") != position // 8 or
                record.get("rank_offset") != position % 8 or
                record.get("native_clip_index") != indices[position]):
            raise ValueError("Native rank/clip/row/frame comparison identity changed")
        for key in ("rng_before_sha256", "rng_after_sha256", "sample_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing exact comparison byte/RNG digest")
        if record["rng_before_sha256"] != previous_rng:
            raise ValueError("Native comparison RNG chain changed")
        previous_rng = record["rng_after_sha256"]
        needed.add(f"train/obses/episode_{identity[0]:03d}.mp4")
    if previous_rng != comparisons.get("final_rng_sha256") or set(selected) != needed:
        raise ValueError("Incomplete native RNG chain or selected-input population")
    obs, action, state, reward = batch
    shapes = {"visual": (256, 4, 3, 224, 224), "proprio": (256, 4, 4)}
    if (set(obs) != set(shapes) or any(tuple(obs[k].shape) != v for k, v in shapes.items()) or
            tuple(action.shape) != (256, 4, 10) or tuple(state.shape) != (256, 4, 7) or
            tuple(reward.shape) != (256, 4) or
            any(t.dtype != torch.float32 for t in [*obs.values(), action, state, reward])):
        raise ValueError("Native first-update CPU batch shape/dtype changed")
    if protocol.get("batch_sha256") != _batch_digest(batch):
        raise ValueError("Engineering batch differs from native input-parity evidence")
    for position, record in enumerate(records):
        sample = ({k: v[position] for k, v in obs.items()}, action[position], state[position], reward[position])
        if record["sample_sha256"] != _batch_digest(sample):
            raise ValueError("Comparison sample differs from actual supplied batch tensors")
    return digest


def _runtime(model, rng, contract=None):
    if type(rng.backend) is not _CudaState or rng.poisoned:
        raise ValueError("Public numerical evidence requires actual CUDA RNG streams")
    params = list(model.parameters())
    if not params or any(p.device != torch.device("cuda", rng.device) for p in params):
        raise ValueError("Require one receiving CUDA device, no CPU/synthetic proof")
    source = inspect.getsourcefile(type(model))
    if (type(model).__name__ != "VideoWM" or source is None or
            sha256(Path(source)) != SOURCE_HASHES["app/vjepa_wm/video_wm.py"] or
            rng.backend.device != rng.device):
        raise ValueError("Require actual pinned native VideoWM and matching CUDA state device")
    if not hasattr(model, "encoder") or any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError("Native DINO encoder must remain frozen")
    if (model.heads or model.mixed_precision is not True or model.use_radamw is not False or
            model.clip_grad != 1 or type(model.optimizer) is not torch.optim.AdamW or
            not model.scaler.is_enabled()):
        raise ValueError("Require native no-head BF16/scaler/AdamW update")
    if any(tuple(g["betas"]) != (.9, .999) or g["eps"] != 1e-8 for g in model.optimizer.param_groups):
        raise ValueError("Native AdamW betas/epsilon changed")
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):
        raise ValueError("No external activation edit or tracing hook during native training proof")
    if contract is not None:
        cfg = contract["config"]
        native = {"cfgs_loss": cfg["loss"], "enc_type": cfg["model"]["visual_encoder"]["enc_type"],
            "pred_type": cfg["model"]["predictor"]["pred_type"], "grid_size": cfg["model"]["grid_size"],
            "img_size": cfg["data"]["img_size"], "use_proprio": True, "use_action": True,
            "action_dim": 20 if contract["task"] == "metaworld" else 10, "proprio_dim": 4,
            "action_conditioning": cfg["model"]["action_conditioning"],
            "proprio_encoding": cfg["model"]["proprio_encoding"],
            "frameskip": cfg["data"].get("frameskip", 1),
            "action_skip": cfg["data"].get("action_skip", 1), **cfg["model"]["wm_encoding"]}
        if any(getattr(model, k, None) != v for k, v in native.items()):
            raise ValueError("Actual native model loss/input/encoding configuration differs")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("No unregistered TF32 precision change")
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
        "device_uuid": str(torch.cuda.get_device_properties(rng.device).uuid),
        "module_modes": {name: m.training for name, m in model.named_modules()},
        "model_structure": {name: [list(p.shape), str(p.dtype), p.requires_grad]
                            for name, p in model.named_parameters()}}


def _receiving_batches(batch, device):
    obs, action, state, reward = batch
    if set(obs) != {"visual", "proprio"} or obs["visual"].shape != (256, 4, 3, 224, 224):
        raise ValueError("Require actual native 256 four-frame RGB clips")
    if any(v.shape[:2] != (256, 4) for v in [*obs.values(), action, state, reward]):
        raise ValueError("All native modalities must share clip/time alignment")
    # Materialize CPU views only; transfer lazily inside each local forward.
    def one(rank):
        section = slice(rank * 8, (rank + 1) * 8)
        move = lambda v: v[section].to(device, dtype=torch.bfloat16, non_blocking=True)
        return {k: move(v) for k, v in obs.items()}, move(action), move(state), move(reward)
    class Batches:
        def __len__(self): return RANKS
        def __iter__(self):
            for rank in range(RANKS): yield one(rank)
    return Batches()


def _scheduler_gate(model, scheduler, wd):
    for value, name in ((scheduler, "WarmupCosineSchedule"), (wd, "CosineWDSchedule")):
        path = inspect.getsourcefile(type(value))
        if (type(value).__name__ != name or path is None or
                sha256(Path(path)) != SOURCE_HASHES["src/utils/schedulers.py"] or
                value.optimizer is not model.optimizer):
            raise ValueError("Require actual pinned schedulers bound to this model optimizer")
    if scheduler._step != wd._step or scheduler._step < 0 or scheduler._step != int(scheduler._step):
        raise ValueError("Native scheduler counters are not an aligned complete update boundary")


def verify_native_update(contract, model, scheduler, wd, batch, rng, *, input_proof):
    """Disposable native-vs-accumulated full update; never changes live model/RNG.

    Tests local gradient arithmetic, not the unpublished physical DDP all-reduce
    order. One update cannot establish history, loader reset or resume equivalence.
    Caller must separately authorize a receiving pilot and preserve this evidence.
    """
    _validate_contract(contract)
    inputs = _input_gate(contract, batch, input_proof)
    runtime = _runtime(model, rng, contract)
    _scheduler_gate(model, scheduler, wd)
    reference, rs, rw = copy.deepcopy((model, scheduler, wd))
    actual, actual_s, actual_w = copy.deepcopy((model, scheduler, wd))
    rr, ar = rng.clone(), rng.clone()
    batches = _receiving_batches(batch, torch.device("cuda", rng.device))
    left = _drive_update(contract, reference, rs, rw, batches, rr, mean_after=True)
    right = _drive_update(contract, actual, actual_s, actual_w, batches, ar)
    assert_same(left, right)
    assert_same(reference.state_dict(), actual.state_dict())
    assert_same(reference.optimizer.state_dict(), actual.optimizer.state_dict())
    assert_same(reference.scaler.state_dict(), actual.scaler.state_dict())
    assert_same(rr.state_dict(), ar.state_dict())
    assert_same({k: v for k, v in vars(rs).items() if k != "optimizer"},
                {k: v for k, v in vars(actual_s).items() if k != "optimizer"})
    assert_same({k: v for k, v in vars(rw).items() if k != "optimizer"},
                {k: v for k, v in vars(actual_w).items() if k != "optimizer"})
    return {"status": "native32_receiving_update_bitwise_passed", "task": contract["task"],
        "model_seed": contract["model_seed"],
        "config_sha256": contract["config_sha256"], "native_source_sha256": SOURCE_HASHES,
        "adapter_sha256": sha256(Path(__file__)), "input_report_sha256": inputs,
        "runtime": runtime, "logical_ranks": RANKS, "microbatch": 8, "global_batch": 256,
        "loss_gradient_parameters_optimizer_scaler_scheduler_rng_bitwise": True,
        "actual_upstream_step_used": True, "reference_gradient_mean": "native backward sum then divide32",
        "physical_ddp_allreduce_order_reproduced": False, "live_model_unchanged": True,
        "engineering_only": True, "complete_history_or_resume_proof": False}


def accumulated_update(contract, model, scheduler, wd, batch, rng, *, input_proof, numerical_proof):
    """One receiving engineering update; no implicit full-history permission."""
    _validate_contract(contract)
    inputs = _input_gate(contract, batch, input_proof)
    runtime = _runtime(model, rng, contract)
    _scheduler_gate(model, scheduler, wd)
    required = {"status": "native32_receiving_update_bitwise_passed", "task": contract["task"],
        "model_seed": contract["model_seed"],
        "config_sha256": contract["config_sha256"], "native_source_sha256": SOURCE_HASHES,
        "adapter_sha256": sha256(Path(__file__)), "input_report_sha256": inputs, "runtime": runtime,
        "loss_gradient_parameters_optimizer_scaler_scheduler_rng_bitwise": True,
        "actual_upstream_step_used": True, "engineering_only": True}
    if not isinstance(numerical_proof, dict) or any(numerical_proof.get(k) != v for k, v in required.items()):
        raise ValueError("Require actual receiving numerical proof for this source/input/mode/device")
    result = _drive_update(contract, model, scheduler, wd,
        _receiving_batches(batch, torch.device("cuda", rng.device)), rng)
    return {k: v for k, v in result.items() if k != "scaled_gradients"}
