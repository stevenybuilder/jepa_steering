"""Optional NNsight interface tracing for frozen JEPA-WM unrolls.

NNsight is imported only when constructing the tracer; importing this module
does not add a dependency to existing native-hook workflows. All interventions
use NNsight's execution-ordered input/output access and output replacement.
"""
from __future__ import annotations
import importlib.metadata
import torch


MODULE_PATHS = {
    "p3_edited_residual": "world_model.model.predictor.predictor_blocks.3.output",
    "predictor_norm_input": "world_model.model.predictor.predictor_norm.input",
    "predictor_norm_output": "world_model.model.predictor.predictor_norm.output",
    "predictor_proj_input": "world_model.model.predictor.predictor_proj.input",
    "predictor_proj_output": "world_model.model.predictor.predictor_proj.output",
}


def apply_raw_delta(output, delta):
    """Preserve every older token and return exact identity for zero/None."""
    if delta is None:
        return output
    if (output.ndim != 3 or output.shape[1] < 256 or output.shape[1] % 256
            or delta.shape != (output.shape[0], output.shape[2]) or not torch.isfinite(delta).all()):
        raise ValueError("Expected candidate-aligned raw channel edit and256-token frames")
    if not torch.count_nonzero(delta):
        return output
    edited = output.clone()
    edited[:, -256:] = output[:, -256:] + delta.to(output).unsqueeze(1)
    return edited


class UnrollForward(torch.nn.Module):
    """Expose the existing native unroll as a regular traceable forward."""
    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model

    @torch.no_grad()
    def forward(self, z, actions):
        return self.world_model.unroll(z, act_suffix=actions)


class NNsightPredictorTrace:
    """Reusable first-imagined-step P3/norm/projection tracer.

The raw vector must be frozen by the caller, not recomputed inside each dose.
All native candidates remain in the forward pass; only capture rows are sliced.
"""
    def __init__(self, world_model, indices=tuple(range(8))):
        try:
            from nnsight import NNsight
        except ImportError as exc:
            raise RuntimeError("Optional NNsight dependency unavailable; install nnsight in an isolated compatible environment") from exc
        self.indices = tuple(indices)
        if not self.indices or min(self.indices) < 0 or len(set(self.indices)) != len(self.indices):
            raise ValueError("Unique nonnegative capture indices required")
        self.forward_module = UnrollForward(world_model)
        self.nnsight_version = importlib.metadata.version("nnsight")
        if self.nnsight_version != "0.7.0":
            raise RuntimeError("Tracer lifecycle is verified for NNsight0.7.0; use the pinned optional environment")
        modules = tuple(self.forward_module.modules())
        if any(hasattr(module, "__nnsight_forward__") for module in modules):
            raise RuntimeError("Model already belongs to an NNsight tracer; close it before wrapping again")
        self._snapshots = []
        self._closed = False
        for module in modules:
            self._snapshots.append({"module": module,
                                    "attributes": {key: (key in module.__dict__, module.__dict__.get(key))
                                                   for key in ("forward", "__nnsight_forward__", "__path__")},
                                    "original_hooks": set(module._forward_hooks), "owned_hooks": set()})
        try:
            self.traced = NNsight(self.forward_module)
        except Exception:
            for snapshot in self._snapshots:
                snapshot["owned_hooks"] = set(snapshot["module"]._forward_hooks) - snapshot["original_hooks"]
            self.close()
            raise
        finally:
            # NNsight0.7 deliberately installs one persistent sentinel hook and
            # instance forward wrapper per module. Save only our own hook IDs.
            for snapshot in self._snapshots:
                snapshot["owned_hooks"] = set(snapshot["module"]._forward_hooks) - snapshot["original_hooks"]

    def close(self):
        """Remove only this tracer's persistent wrappers; preserve user hooks."""
        if self._closed:
            return
        for snapshot in self._snapshots:
            module = snapshot["module"]
            for hook_id in snapshot["owned_hooks"]:
                for key in ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
                    getattr(module, key, {}).pop(hook_id, None)
            for key, (existed, value) in snapshot["attributes"].items():
                if existed:
                    setattr(module, key, value)
                elif key in module.__dict__:
                    delattr(module, key)
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Tracer is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def trace_unroll(self, z, actions, delta=None):
        if self._closed:
            raise RuntimeError("Tracer is closed")
        from nnsight import save
        if actions.ndim != 3 or max(self.indices) >= actions.shape[1]:
            raise ValueError("Capture indices outside native candidate batch")
        indices = list(self.indices)
        predictor = self.traced.world_model.model.predictor
        with self.traced.trace(z, actions):
            residual = predictor.predictor_blocks[3].output
            edited = apply_raw_delta(residual, delta)
            if edited is not residual:
                predictor.predictor_blocks[3].output = edited
            before = save(residual[indices].detach().float().cpu().clone())
            p3 = save(edited[indices].detach().float().cpu().clone())
            norm_input = save(predictor.predictor_norm.input[indices].detach().float().cpu().clone())
            norm_output = save(predictor.predictor_norm.output[indices].detach().float().cpu().clone())
            projection_input = save(predictor.predictor_proj.input[indices].detach().float().cpu().clone())
            projection_output = save(predictor.predictor_proj.output[indices].detach().float().cpu().clone())
            prediction = save(self.traced.output)
        return {"prediction": prediction,
                "stages": {"p3_edited_residual": p3, "predictor_norm_input": norm_input,
                           "predictor_norm_output": norm_output, "predictor_proj_input": projection_input,
                           "predictor_proj_output": projection_output},
                "before_residual": before, "module_paths": dict(MODULE_PATHS),
                "capture_indices": self.indices, "nnsight_version": self.nnsight_version}
