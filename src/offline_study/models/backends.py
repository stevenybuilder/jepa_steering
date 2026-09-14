"""Real JEPA adapter and an explicitly non-scientific CPU smoke fixture."""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from torch import nn

from offline_study.core.protocol import sha256
from offline_study.models.vendor import use_vendor


class JepaBackend:
    synthetic = False

    def __init__(
        self,
        vendor: Path,
        checkpoint: Path,
        checkpoint_sha256: str,
        dataset: str,
        device: str,
        precision: str = "float32",
        allow_tf32: bool = False,
    ):
        from offline_study.models.model_loader import load_headless, model_name_for_dataset
        model_name = model_name_for_dataset(dataset)
        use_vendor(vendor)
        if sha256(checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint checksum mismatch")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
        if precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"Unknown precision: {precision}")
        if precision != "float32" and not device.startswith("cuda"):
            raise ValueError("Reduced-precision JEPA execution requires CUDA")
        if allow_tf32 and (precision != "float32" or not device.startswith("cuda")):
            raise ValueError("TF32 is an opt-in CUDA float32 execution mode")
        if precision == "bfloat16":
            with torch.cuda.device(torch.device(device)):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This CUDA device does not support bfloat16")
        self.precision = precision
        self.autocast_dtype = {
            "float32": None,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[precision]
        self.allow_tf32 = allow_tf32
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        self.model, self.preprocessor, self.provenance = load_headless(
            vendor, device=device, model_name=model_name,
            checkpoint_override=checkpoint,
        )
        self.model.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.provenance.update(
            checkpoint_sha256=checkpoint_sha256,
            precision=precision,
            autocast=precision != "float32",
            allow_tf32=allow_tf32,
        )

    def autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def encode(self, visual, proprio):
        # Raw 0..255 images and unnormalized proprioception. Official encode does normalization.
        with self.autocast():
            return self.model.encode({"visual": visual, "proprio": proprio})

    def normalize_actions(self, actions):
        # [batch, horizon, five elementary actions, action dimensions]
        normalized = self.preprocessor.normalize_actions(actions)
        return normalized.flatten(2).transpose(0, 1).contiguous().to(self.device)

    def context(self, encoded):
        from tensordict import TensorDict
        return TensorDict({k: encoded[k][:, :1] for k in ("visual", "proprio")}, batch_size=[])

    def expand_context(self, context, repeats):
        from tensordict import TensorDict
        return TensorDict({
            key: context[key].repeat_interleave(repeats, dim=0)
            for key in ("visual", "proprio")
        }, batch_size=[])

    @property
    def predictor(self):
        return self.model.model.predictor

    def predict(self, context, actions, instrument=False):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.autocast())
            if instrument:
                for block in self.model.model.predictor.predictor_blocks:
                    handle = block.register_forward_hook(lambda module, args, output: output)
                    stack.callback(handle.remove)
            return self.model.unroll(context.clone(), act_suffix=actions)


class ToyBackend:
    """Tiny deterministic fixture. Its timings/errors cannot estimate JEPA-WM performance."""
    synthetic = True

    def __init__(self, device="cpu", precision="float32", allow_tf32=False):
        if precision != "float32" or allow_tf32:
            raise ValueError("The CPU toy fixture supports only strict float32")
        self.device = torch.device(device)
        torch.manual_seed(17)
        self.encoder = nn.Linear(3, 16).to(device).eval().requires_grad_(False)
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(6)]).to(device).eval().requires_grad_(False)
        self.action = nn.Linear(20, 16).to(device).eval().requires_grad_(False)
        self.provenance = {
            "model": "toy_fixture_NOT_JEPA_WM",
            "precision": "float32",
            "autocast": False,
            "allow_tf32": False,
        }

    def encode(self, visual, proprio):
        patches = visual.to(self.device).float().mean((-1, -2)) / 255
        return {"visual": self.encoder(patches), "proprio": proprio.to(self.device).float()}

    def normalize_actions(self, actions):
        return actions.flatten(2).transpose(0, 1).contiguous().to(self.device)

    def context(self, encoded):
        return {k: v[:, :1].clone() for k, v in encoded.items()}

    def expand_context(self, context, repeats):
        return {key: value.repeat_interleave(repeats, dim=0) for key, value in context.items()}

    def predict(self, context, actions, instrument=False):
        visual, proprio = context["visual"][:, 0], context["proprio"][:, 0]
        vs, ps = [visual], [proprio]
        with contextlib.ExitStack() as stack:
            if instrument:
                for block in self.blocks:
                    handle = block.register_forward_hook(lambda module, args, output: output)
                    stack.callback(handle.remove)
            for action in actions:
                visual = visual + self.action(action)
                for block in self.blocks:
                    visual = visual + 0.05 * block(visual).tanh()
                proprio = proprio + action[:, :proprio.shape[-1]]
                vs.append(visual)
                ps.append(proprio)
        return {"visual": torch.stack(vs), "proprio": torch.stack(ps)}
