"""Fixed development-only action geometry; no fit bank, planner, or physical targets.

FP32 encoding is shared. 'BF16' means autocast of the entire recursive predictor
unroll, not BF16 weights, encoder preprocessing, or an isolated arithmetic kernel.
Only small scalar/Gram receipts leave GPU scratch; no activation tensors are saved.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

OFFSETS = (-.1, -.05, 0., .05, .1)
ANCHORS = (0, 1, 3, 4)
WEIGHTS = {"linear": (.25, .25, .25, .25),
           "cubic": (-1 / 6, 2 / 3, 2 / 3, -1 / 6)}
COHORT = {task: list(range(32)) for task in ("reach", "reach-wall")}
INPUT_SHA = "7eaaec460a9057078a36cb348e859222cdf426842620107bb0d2cdabe7df70ef"
CHECKPOINT_SHA = "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
REQUIRED_SOURCE = {"controlled_geometry_pilot.py", "backends.py", "model_loader.py",
                   "vendor.py", "planning_contract.py", "protocol.py", "__init__.py"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def tree_hash(root):
    digest = hashlib.sha256()
    files = sorted(p for p in Path(root).rglob("*") if p.is_file()
                   and p.suffix in (".py", ".yaml", ".yml", ".json"))
    if not files:
        raise ValueError("Empty vendor source/config tree")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def tensor_hash(value):
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def rng_hash(device):
    states = [random.getstate(), np.random.get_state(), tensor_hash(torch.get_rng_state())]
    if torch.device(device).type == "cuda":
        states.append(tensor_hash(torch.cuda.get_rng_state(device)))
    return hashlib.sha256(pickle.dumps(states, protocol=4)).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def action_batch(actions):
    if tuple(actions.shape) != (6, 300, 20) or actions.dtype != torch.float32:
        raise ValueError("Expected original FP32 [6,300,20] action bank")
    if not torch.isfinite(actions).all() or torch.count_nonzero(actions[:, :1]).item():
        raise ValueError("Archived candidate0 must be exactly the zero-action plan")
    batch = actions[:, :1].repeat(1, 5, 1).contiguous()
    batch[2, :, 0] = torch.tensor(OFFSETS, dtype=batch.dtype, device=batch.device)
    return batch


def summarize_fields(fields):
    """Five [offset,...] fields; all reductions use float64, never an epsilon log."""
    if fields.ndim < 2 or fields.shape[0] != 5 or not torch.isfinite(fields).all():
        raise ValueError("Expected five finite fields")
    values = fields.detach().reshape(5, -1).double()
    center = values[2]
    anchors = values[list(ANCHORS)]
    centered = anchors - center
    gram = centered @ centered.T / center.numel()
    result = {"field_dtype": str(fields.dtype), "field_shape": list(fields.shape[1:]),
              "field_numel": center.numel(), "center_sha256": tensor_hash(fields[2]),
              "center_rms": center.square().mean().sqrt().item(),
              "centered_anchor_gram_mean_inner_product": gram.cpu().tolist(),
              "donor_response_rms": gram.diag().sqrt().cpu().tolist(),
              "exact_unchanged_fraction": (anchors == center).double().mean(1).cpu().tolist()}
    for name, coefficients in WEIGHTS.items():
        weights = torch.tensor(coefficients, dtype=torch.float64, device=values.device)
        # Direct absolute-field reconstruction is independent of the centered Gram.
        residual = (weights[:, None] * anchors).sum(0) - center
        direct = residual.square().mean().item()
        reconstructed = (weights @ gram @ weights).item()
        eps = torch.finfo(torch.float64).eps
        roundoff = 32 * eps * values.abs().max().item()
        tolerance = (2 * math.sqrt(max(direct, abs(reconstructed))) * roundoff + roundoff**2
                     + 64 * eps * gram.abs().max().item())
        if abs(direct - reconstructed) > tolerance:
            raise ValueError(f"{name} direct/Gram reconstruction mismatch")
        result[name + "_center_mse"] = direct
        result[name + "_gram_mse"] = reconstructed
        result[name + "_gram_absolute_error"] = abs(direct - reconstructed)
        result[name + "_gram_tolerance"] = tolerance
    result["finite_differences"] = []
    for radius, minus, plus in ((.05, 1, 3), (.1, 0, 4)):
        first = (values[plus] - values[minus]) / 2
        second = values[plus] - 2 * center + values[minus]
        result["finite_differences"].append({"radius": radius,
            "first_difference_mean": first.mean().item(),
            "first_difference_rms": first.square().mean().sqrt().item(),
            "second_difference_mean": second.mean().item(),
            "second_difference_rms": second.square().mean().sqrt().item(),
            "central_first_derivative_rms": (first / radius).square().mean().sqrt().item(),
            "central_second_derivative_rms": (second / radius**2).square().mean().sqrt().item()})
    return result


class CaptureH3:
    """Read-only hooks; newest 256 visual patches, all six zero-indexed blocks."""
    def __init__(self, predictor):
        self.predictor, self.handles, self.fields = predictor, [], {}
        self.horizon = 0
        self.calls = [0] * 6

    def __enter__(self):
        blocks = self.predictor.predictor_blocks
        if len(blocks) != 6 or any(m._forward_hooks or m._forward_pre_hooks
                                  for m in self.predictor.modules()):
            raise ValueError("Expected six hook-free predictor blocks")
        self.handles.append(self.predictor.register_forward_pre_hook(self.advance))
        for index, block in enumerate(blocks):
            self.handles.append(block.register_forward_hook(self.capture(index)))
        return self

    def advance(self, module, args):
        self.horizon += 1

    def capture(self, index):
        def hook(module, args, output):
            self.calls[index] += 1
            if self.horizon == 3:
                if index in self.fields or output.ndim != 3 or output.shape[0] != 5 or output.shape[1] < 256:
                    raise ValueError("Unexpected block output shape or repeated H3")
                self.fields[index] = output[:, -256:].detach().clone()
            return output
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def validate(self):
        if self.horizon != 6 or self.calls != [6] * 6 or set(self.fields) != set(range(6)):
            raise ValueError("Incomplete six-horizon/six-block capture")


@contextlib.contextmanager
def predictor_precision(backend, precision):
    original = backend.precision, backend.autocast_dtype
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    backend.precision = precision
    backend.autocast_dtype = None if precision == "float32" else torch.bfloat16
    try:
        yield
    finally:
        backend.precision, backend.autocast_dtype = original


def validate_manifest(manifest, protocol, source_root, vendor):
    if manifest.get("scenarios") != COHORT or protocol.get("scenarios") != COHORT:
        raise ValueError("Exact fixed 64-development cohort required")
    for obj in (manifest, protocol):
        if obj.get("input_manifest_sha256") != INPUT_SHA or obj.get("checkpoint_sha256") != CHECKPOINT_SHA:
            raise ValueError("Original input/checkpoint binding changed")
    if protocol.get("offsets") != list(OFFSETS) or protocol.get("off_center_evaluations") != []:
        raise ValueError("Frozen anchor/evaluation registry changed")
    if protocol.get("context_precision") != "float32_shared_encoded_once":
        raise ValueError("FP32 shared-context precision contract required")
    sources = manifest.get("source_sha256", {})
    if not REQUIRED_SOURCE <= sources.keys():
        raise ValueError("Missing required source hashes")
    for name, expected in sources.items():
        if Path(name).name != name or file_hash(Path(source_root) / name) != expected:
            raise ValueError("Source binding mismatch: " + name)
    if tree_hash(vendor) != manifest.get("vendor_source_sha256"):
        raise ValueError("Vendor source binding mismatch")


def run_case(backend, cfg, data, binding, output, provenance):
    from tensordict import TensorDict
    from evals.utils import prepare_obs
    if set(data) != {"initial", "goal", "actions"}:
        raise ValueError("Unexpected archived inputs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "STARTED.json", {**provenance, "binding": binding,
               "status": "controlled_geometry_development_started"})
    started = time.perf_counter()
    raw_actions = data["actions"].to(backend.device)
    actions = action_batch(raw_actions)
    initial = TensorDict(data["initial"], batch_size=[])
    context = backend.model.encode(prepare_obs(cfg["task_specification"]["obs"], initial)
                                  .to(backend.device).unsqueeze(0), act=True)
    context = backend.expand_context(context, 5)
    signatures = {key: tensor_hash(context[key]) for key in ("visual", "proprio")}
    action_sha = tensor_hash(actions)
    # Detect accidental writes without transferring/checksumming all model weights.
    def model_versions():
        return {"parameters": [(name, p._version) for name, p in backend.model.named_parameters()],
                "buffers": [(name, p._version) for name, p in backend.model.named_buffers()],
                "training": [(name, m.training) for name, m in backend.model.named_modules()]}
    initial_versions = model_versions()
    rows, timings = [], {}
    for precision in ("float32", "bfloat16"):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        rng_before = rng_hash(backend.device)
        with predictor_precision(backend, precision):
            native = backend.predict(context, actions)
            with CaptureH3(backend.predictor) as capture:
                observed = backend.predict(context, actions)
            capture.validate()
        for key in ("visual", "proprio"):
            if (native[key].dtype != observed[key].dtype or
                native[key].shape != observed[key].shape or
                tensor_hash(native[key]) != tensor_hash(observed[key])):
                raise ValueError("Native/capture-hook byte parity failed: " + key)
        if rng_hash(backend.device) != rng_before:
            raise ValueError("Native/capture global RNG parity failed")
        for layer, field in sorted(capture.fields.items()):
            rows.append({"condition": precision, "layer": layer, **summarize_fields(field)})
            if precision == "float32":
                rounded = field.to(torch.bfloat16).float()
                rows.append({"condition": "float32_field_bfloat16_roundtrip", "layer": layer,
                             **summarize_fields(rounded)})
        del native, observed, capture
        torch.cuda.synchronize()
        timings[precision + "_parity_and_reductions_seconds"] = time.perf_counter() - tick
    if signatures != {key: tensor_hash(context[key]) for key in signatures} or action_sha != tensor_hash(actions):
        raise ValueError("Context/action mutation")
    if initial_versions != model_versions():
        raise ValueError("Model parameter/buffer/mode mutation")
    report = {**provenance, "binding": binding, "status": "controlled_geometry_development_complete",
              "context_sha256": signatures, "perturbed_batch_sha256": action_sha,
              "native_hook_byte_parity": True, "rng_parity": True, "rows": rows,
              "timings": timings, "case_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "physical_outcomes_measured": False, "fresh_confirmation": False}
    write_json(output / "report.json", report)
    write_json(output / "DONE.json", {"status": report["status"],
               "report_sha256": file_hash(output / "report.json"),
               "started_sha256": file_hash(output / "STARTED.json"), "rows": len(rows)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "inputs", "manifest", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("manifest-sha256", "protocol-sha256", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--task", choices=tuple(COHORT), required=True)
    parser.add_argument("--episode", type=int, choices=range(32), required=True)
    args = parser.parse_args()
    if file_hash(args.manifest) != args.manifest_sha256 or file_hash(args.protocol) != args.protocol_sha256:
        raise ValueError("Execution manifest/protocol bytes changed")
    manifest, protocol = [json.loads(path.read_text()) for path in (args.manifest, args.protocol)]
    validate_manifest(manifest, protocol, Path(__file__).parent, args.vendor)
    if manifest.get("protocol_sha256") != args.protocol_sha256:
        raise ValueError("Execution manifest does not bind this protocol")
    if file_hash(args.checkpoint) != CHECKPOINT_SHA or file_hash(args.inputs / "INPUT_MANIFEST.json") != INPUT_SHA:
        raise ValueError("Checkpoint/input-manifest bytes changed")
    records = json.loads((args.inputs / "INPUT_MANIFEST.json").read_text())["records"]
    if len(records) != 64 or {(r["task"], r["episode"]) for r in records} != {(t, e) for t in COHORT for e in COHORT[t]}:
        raise ValueError("Input manifest does not contain exact unique 64 contexts")
    binding = next(r for r in records if (r["task"], r["episode"]) == (args.task, args.episode))
    path = args.inputs / args.task / f"episode-{args.episode:03d}" / "inputs.pt"
    if file_hash(path) != binding["inputs_sha256"]:
        raise ValueError("Archived case input bytes changed")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Require one visible BF16-capable CUDA GPU")
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable"))
    if uuid.removeprefix("GPU-").lower() != args.gpu_uuid.removeprefix("GPU-").lower():
        raise ValueError("Physical GPU UUID differs from explicit receiver binding")
    if os.environ.get("JEPA_VERIFIED_LOCAL_DINO") != "1":
        raise ValueError("Require JEPA_VERIFIED_LOCAL_DINO=1 for exact external encoder weights/source")
    from .backends import JepaBackend
    from .planning_contract import prepare
    torch.set_num_threads(1)
    backend = JepaBackend(args.vendor, args.checkpoint, CHECKPOINT_SHA, "metaworld", "cuda:0",
                          precision="float32", allow_tf32=False)
    cfg = prepare(args.vendor, args.task)["config"]
    provenance = {"execution_manifest_sha256": args.manifest_sha256,
        "protocol_sha256": args.protocol_sha256, "checkpoint_sha256": CHECKPOINT_SHA,
        "input_manifest_sha256": INPUT_SHA, "source_sha256": manifest["source_sha256"],
        "vendor_source_sha256": manifest["vendor_source_sha256"],
        "gpu": str(torch.cuda.get_device_properties(0)),
        "gpu_uuid": uuid, "backend_provenance": backend.provenance,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "precision_boundary": "FP32 shared encoding; predictor/unroll autocast only; FP32 weights; TF32 disabled"}
    with torch.no_grad():
        report = run_case(backend, cfg, torch.load(path, map_location="cpu", weights_only=True),
                          binding, args.output, provenance)
    print(json.dumps({"status": report["status"], "case_seconds": report["case_seconds"]}))


if __name__ == "__main__":
    main()
