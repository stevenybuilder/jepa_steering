"""Load released JEPA-WM checkpoints without the optional 3.39 GB RGB decoder.

The geometry map uses encoder and predictor activations only.  This calls the
official constructors and checkpoint loader with the published configuration; the
sole change is clearing ``heads_cfg`` so an unused visualization head is not built.
"""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def load_headless(
    repo: Path, device: str = "cuda:0", model_name: str = "jepa_wm_metaworld",
    checkpoint_override: Path | None = None,
) -> tuple[torch.nn.Module, Any, dict[str, str]]:
    repo = repo.resolve()
    sys.path.insert(0, str(repo))

    import hubconf
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    from src.utils.yaml_utils import expand_env_vars

    if model_name not in ("jepa_wm_metaworld", "jepa_wm_pusht"):
        raise ValueError(model_name)
    config_rel, weight_key = hubconf._MODEL_CONFIGS[model_name]
    config_path = repo / config_rel
    args_eval = expand_env_vars(yaml.safe_load(config_path.read_text()))
    model_kwargs = args_eval["model_kwargs"]
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = copy.deepcopy(model_kwargs.get("pretrain_kwargs", {}))
    pretrain_kwargs["heads_cfg"] = {}

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable; refusing silent CPU fallback")
    torch_device = torch.device(device)
    data_stats = get_data_stats("pusht" if model_name == "jepa_wm_pusht" else "metaworld")
    img_size = cfgs_data.get("img_size", 224)
    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug.get("normalize", [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]),
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)
    preprocessor = Preprocessor(
        action_mean=torch.tensor(data_stats["action_mean"]),
        action_std=torch.tensor(data_stats["action_std"]),
        state_mean=torch.tensor(data_stats["state_mean"]),
        state_std=torch.tensor(data_stats["state_std"]),
        proprio_mean=torch.tensor(data_stats["proprio_mean"]),
        proprio_std=torch.tensor(data_stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )
    checkpoint = str(checkpoint_override) if checkpoint_override else hubconf._get_checkpoint_path(weight_key, use_hf=True)
    # Import the configured model constructor directly. The upstream evaluation
    # entrypoint wraps this same call but also imports simulators and planners that
    # are deliberately outside this recorded-action benchmark.
    init_module = importlib.import_module(model_kwargs["module_name"]).init_module
    model = init_module(
        folder=args_eval.get("folder"),
        checkpoint=checkpoint,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=torch_device,
        action_dim=data_stats["action_dim"],
        proprio_dim=data_stats["proprio_dim"],
        preprocessor=preprocessor,
    )
    model.eval()
    provenance = {
        "model_name": model_name,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "loader": "official constructor/checkpoint; heads_cfg cleared",
    }
    return model, preprocessor, provenance


def load_headless_metaworld(repo: Path, device: str = "cuda:0") -> tuple[torch.nn.Module, Any, dict[str, str]]:
    return load_headless(repo, device=device)
