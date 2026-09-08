"""Native32 model construction, isolated from active runs and history launch.

Use the pinned upstream constructor-call AST, not a navigation-specific copy.
This does NOT establish post-loader/LPIPS RNG parity or authorize a history.
The public CUDA constructor requires a verified real training-input receipt.
"""
from __future__ import annotations

import ast
import copy
import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .model_loader import verified_local_dino_cache
from .protocol import sha256
from .robotics_training_pilot import (
    SOURCE_HASHES, _input_gate, _validate_contract,
)

UPDATES = {"metaworld": 3543, "pusht": 7741}
DIMENSIONS = {"metaworld": (4, 4), "pusht": (2, 4)}


def _assignment(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _constructor_nodes(contract):
    """Exact original dimensions/kwargs/calls; deliberately omit DDP and I/O."""
    _validate_contract(contract)
    path = Path(contract["vendor"]) / "app/vjepa_wm/train.py"
    main = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = main.body
    model_call = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Name)
                      and node.value.func.id == "init_video_model")
    # The immediately preceding dimension branches are distinguished from
    # subsequent DDP wrapping by their actual computed output names.
    start = next(i for i, node in enumerate(body[:model_call])
                 if isinstance(node, ast.If) and any(
                     isinstance(child, ast.Name) and child.id == "actions_per_vid_feat"
                     for child in ast.walk(node)))
    optimizer_branch = next(node for node in body[model_call + 1:]
                            if isinstance(node, ast.If) and any(
                                isinstance(child, ast.Call)
                                and isinstance(child.func, ast.Name)
                                and child.func.id == "init_opt"
                                for child in ast.walk(node)))
    optimizer_nodes = optimizer_branch.body[:3]
    if (len(optimizer_nodes) != 3 or not _assignment(optimizer_nodes[1], "clip_grad")
            or not _assignment(optimizer_nodes[2], "use_radamw")):
        raise ValueError("Pinned optimizer constructor structure changed")
    wm_start = next(i for i, node in enumerate(body) if _assignment(node, "wm_kwargs"))
    if not _assignment(body[wm_start + 1], "world_model"):
        raise ValueError("Pinned VideoWM constructor structure changed")
    return path, body[start:model_call + 1] + optimizer_nodes + body[wm_start:wm_start + 2]


def _construct(contract, device, init_video_model, init_opt, video_wm):
    """Private constructor seam; metadata tests cannot produce GPU evidence."""
    path, nodes = _constructor_nodes(contract)
    cfg = copy.deepcopy(contract["config"])
    m, d, opt = cfg["model"], cfg["data"], cfg["optimization"]
    if (opt["main_optimizer"] != "transition_model" or opt["train_heads"]
            or cfg["meta"].get("freeze_encoder", True) is not True
            or m.get("heads_cfg", {}).get("pretrain_dec_path") is not None
            or m["visual_encoder"]["enc_type"] != "dino"):
        raise ValueError("Require the registered fresh predictor/frozen DINO/no-head recipe")
    opt["transition_model"]["iterations_per_epoch"] = UPDATES[contract["task"]]
    action_dim, proprio_dim = DIMENSIONS[contract["task"]]
    ns = dict(cfgs_model=m, cfgs_data=d, cfgs_opt=opt, cfgs_loss=cfg["loss"],
        cfgs_wm_encoding=m["wm_encoding"], device=device, img_size=d["img_size"],
        tubelet_size_enc=m["tubelet_size_enc"], frameskip=d["custom"]["frameskip"],
        action_skip=d["custom"]["action_skip"], state_skip=d["custom"]["state_skip"],
        action_tokens=m["action_encoder"]["action_tokens"],
        proprio_tokens=m["proprio_encoder"]["proprio_tokens"],
        use_action=True, use_proprio=True, freeze_encoder=True, mixed_precision=True,
        traj_dataset=SimpleNamespace(action_dim=action_dim, proprio_dim=proprio_dim),
        heads={}, init_video_model=init_video_model, init_opt=init_opt, VideoWM=video_wm)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def constructor_contract(contract):
    """CPU metadata-only actual caller audit; imports no native model or data."""
    calls = {}

    def model(**kwargs):
        calls["model"] = kwargs
        return "predictor", "encoder", "action_encoder", "proprio_encoder"

    def optimizer(**kwargs):
        calls["optimizer"] = kwargs
        return "optimizer", "scaler", "scheduler", "wd_scheduler"

    def wm(**kwargs):
        calls["world_model"] = kwargs
        return "world_model"

    _construct(contract, "cuda:0", model, optimizer, wm)
    return {"task": contract["task"], "config_sha256": contract["config_sha256"],
        "native_source_sha256": SOURCE_HASHES, "calls": calls,
        "optimizer_updates_per_epoch": UPDATES[contract["task"]],
        "metadata_only": True, "model_initialized": False, "gpu_calls": 0,
        "history_launch_authorized": False}


def _check_callable(value, vendor, relative, name):
    source = inspect.getsourcefile(value)
    expected = Path(vendor) / relative
    if (value.__name__ != name or source is None or Path(source).resolve() != expected.resolve()
            or sha256(expected) != SOURCE_HASHES[relative]):
        raise ValueError("Actual imported native constructor differs: " + name)


def build_receiving_model(contract, batch, *, input_proof, device=0):
    """Build a disposable native CUDA model; return explicit post-construction RNG.

    No validation access, pretrained predictor, optimizer update, loader iterator,
    LPIPS initialization, or automatic logical-rank RNG initialization occurs.
    Caller RNG is restored even on failure. The returned RNG is NOT claimed to
    be the native first-update RNG: full initialization/loader bookkeeping must
    be independently implemented before history use.
    """
    _validate_contract(contract)
    inputs_hash = _input_gate(contract, batch, input_proof)
    if type(device) is not int or device < 0 or not torch.cuda.is_available():
        raise ValueError("Require an explicitly assigned receiving CUDA device")
    if device != 0 or torch.cuda.device_count() != 1:
        raise ValueError("Receiving launcher must expose exactly its one assigned GPU")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Unregistered TF32 precision change")
    numpy_state, python_state = copy.deepcopy(np.random.get_state()), random.getstate()
    # manual_seed affects every visible CUDA device; the guard above restricts
    # this receiving process to its single assigned device before context setup.
    try:
        with torch.random.fork_rng(devices=[0]):
            from .vendor import use_vendor
            use_vendor(Path(contract["vendor"]))
            from app.vjepa_wm.utils import init_video_model, init_opt
            from app.vjepa_wm.video_wm import VideoWM
            _check_callable(init_video_model, contract["vendor"], "app/vjepa_wm/utils.py", "init_video_model")
            _check_callable(init_opt, contract["vendor"], "app/vjepa_wm/utils.py", "init_opt")
            _check_callable(VideoWM, contract["vendor"], "app/vjepa_wm/video_wm.py", "VideoWM")
            with torch.cuda.device(device), verified_local_dino_cache() as cache:
                np.random.seed(contract["model_seed"])
                torch.manual_seed(contract["model_seed"])
                ns = _construct(contract, torch.device("cuda", device), init_video_model, init_opt, VideoWM)
                rng = {"cpu": torch.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state(device).clone(),
                    "python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state())}
                model = ns["world_model"]
                if any(p.requires_grad for p in model.encoder.parameters()) or model.encoder.training:
                    raise ValueError("Constructor failed frozen/eval encoder invariant")
                if ns["scheduler"]._step != 0 or ns["wd_scheduler"]._step != 0:
                    raise ValueError("Fresh construction must not consume an optimizer update")
                report = {"status": "native32_receiving_model_constructed_not_history_ready",
                    "task": contract["task"], "model_seed": contract["model_seed"],
                    "native_config_sha256": contract["config_sha256"],
                    "native_source_sha256": SOURCE_HASHES, "constructor_sha256": sha256(Path(__file__)),
                    "input_report_sha256": inputs_hash, **cache,
                    "updates_per_epoch": UPDATES[contract["task"]], "optimizer_updates": 0,
                    "encoder_frozen": True, "validation_or_confirmation_access": False,
                    "physical_ddp_initialized": False, "training_history_initialized": False,
                    "native_post_loader_and_lpips_rng_parity": False,
                    "gpu_update_parity_established": False, "engineering_only": True,
                    "module_modes": {name: module.training for name, module in model.named_modules()}}
                return model, ns["scheduler"], ns["wd_scheduler"], rng, report
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
