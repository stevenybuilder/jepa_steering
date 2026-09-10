"""Push-T's first native 32x8 training batch: CPU-only, real-input evidence.

The reference is the pinned whole-video reader/transform/slicer. The independent
path decodes only the four selected frames and computes modalities from separately
loaded raw sidecars. Neither path reads validation/confirmation. This proves one
engineering batch, not a replacement DataLoader, worker RNG history, or training
resume. No synthetic-data or injected-reader option exists on the public producer.
Run in a dedicated CPU process: the native reader selects Decord's torch bridge.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import pickle
import random
import shutil
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch

from .droid_native import assert_same, verified_report
from .protocol import sha256, write_json
from .robotics_training_inputs import COUNTS, DATA_REVISION, MANIFEST_HASHES, bound_manifest, safe_member
from .robotics_training_pilot import SOURCE_HASHES, _batch_digest, _input_gate, native_contract, native_loader_probe
from .vendor import use_vendor

EXTRA_SOURCES = {
    "src/datasets/utils/video/transforms.py": "61fd92a11e788e2b34e2d121ee45376044802544a076ecdc7c571e834cccdf3c",
    "src/datasets/utils/video/functional.py": "16e6fa06e2d27001ce7d2aab861a6014e62b33ab027ed1598a759d44d36a5e31",
}
NATIVE_SIDECARS = ("states.pth", "rel_actions.pth", "velocities.pth", "seq_lengths.pkl")
TRAIN_CLIPS = 1981721
ENGINEERING_SEED = 234
MAX_OUTPUT_BYTES = 1 << 30


def _rng():
    return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())


def _restore(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _rng_hash(state):
    digest = hashlib.sha256(json.dumps(state[0]).encode())
    array = state[1][1]
    digest.update(json.dumps([state[1][0], *state[1][2:]]).encode())
    digest.update(str((array.shape, array.dtype)).encode() + array.tobytes())
    digest.update(state[2].numpy().tobytes())
    return digest.hexdigest()


def _raw_binding(root, receipt, provenance):
    root, receipt, provenance = Path(root), Path(receipt), Path(provenance)
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir() or
            root == Path(root.anchor) or ".." in root.parts):
        raise ValueError("Explicit existing nonsymlink raw-data root required")
    if (receipt / "FAILED.json").exists():
        raise ValueError("Failed raw-input receipt")
    raw, digest = verified_report(receipt)
    required = {"status": "complete_robotics_raw_inputs_verified", "task": "pusht",
        "data_root": str(root), "data_revision": DATA_REVISION,
        "raw_manifest_sha256": MANIFEST_HASHES["pusht"],
        "training_rows": COUNTS["pusht"][0], "validation_rows": COUNTS["pusht"][1],
        "training_or_validation_authorized": False, "native_reader_parity_established": False,
        "model_or_data_loader_initialized": False}
    if any(raw.get(k) != v for k, v in required.items()):
        raise ValueError("Require complete pinned Push-T raw-byte verification, not a fixture")
    # Parses manifest metadata only; does not open its validation members.
    return bound_manifest("pusht", provenance), digest


def _verify_training_members(root, manifest, names):
    if (root / "train/shapes.pkl").exists() or (root / "train/shapes.pkl").is_symlink():
        raise ValueError("Unregistered optional shapes would change the native reader")
    checked = {}
    for name in sorted(set(names)):
        if not name.startswith("train/") or name not in manifest:
            raise ValueError("Only manifest-bound training inputs may be opened")
        path = safe_member(root, name)
        record = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if record != manifest[name]:
            raise ValueError("Selected native training bytes changed: " + name)
        checked[name] = record
    return checked


def _native_training_branch(path, module_namespace, call, transform, root):
    """Execute only unchanged train assignments, never native val constructors.

    Private AST seam is tested with fixtures; the public caller supplies verified
    imported upstream classes and the real native caller's parameters only.
    """
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "load_pusht_slice_train_val")
    signature = inspect.signature(module_namespace[fn.name])
    bound = signature.bind_partial(transform, **call)
    bound.apply_defaults()
    namespace = dict(module_namespace, **bound.arguments)
    namespace["data_path"] = str(root)
    assignments = []
    for name in ("train_dset", "num_frames", "train_slices"):
        nodes = [n for n in fn.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        if len(nodes) != 1:
            raise ValueError("Native training-only branch changed")
        assignments.extend(nodes)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["train_slices"]


def _native_transform(vendor, cfg, make_transforms):
    path = vendor / "app/vjepa_wm/train.py"
    main = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    nodes = [n for n in main.body if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "transform"]
    if len(nodes) != 1:
        raise ValueError("Native transform caller changed")
    namespace = {"make_transforms": make_transforms, "img_size": cfg["data"]["img_size"],
                 "cfgs_data_aug": cfg["data_aug"]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["transform"]


def _first_update_indices(contract, length):
    indices = []
    for rank in range(32):
        # Metadata-only native loader; iterating its batch_sampler creates no
        # workers, opens no data, and preserves the actual rank-major order.
        loader, _, _ = native_loader_probe(contract["vendor"], "pusht", length, rank=rank)
        indices.extend(next(iter(loader.batch_sampler)))
    if len(indices) != 256 or len(set(indices)) != 256:
        raise ValueError("Require one complete nonduplicated native global update")
    return indices


def _independent_slices(lengths, indices, frames, stride, seed):
    """Prefix sums independently verify native flattened enumeration/permutation."""
    stops = np.cumsum([max(0, int(n) - frames * stride + 1) for n in lengths]).tolist()
    if not stops or stops[-1] <= max(indices):
        raise ValueError("Training slice population too small")
    order = torch.randperm(stops[-1], generator=torch.Generator().manual_seed(seed))
    selected = []
    for index in indices:
        offset = int(order[index])
        row = bisect_right(stops, offset)
        start = offset - (stops[row - 1] if row else 0)
        selected.append((row, start, start + frames * stride))
    return selected, stops[-1]


def _selected_sample(raw, constants, identity, image, transform):
    """Independent raw modalities, not views of native normalized dataset arrays."""
    row, start, end = identity
    frames = list(range(start, end, 5))
    state = torch.cat((raw["states"][row, frames].float(),
                       raw["velocities"][row, frames].float()), dim=-1)
    proprio = torch.cat((raw["states"][row, frames, :2].float(),
                         raw["velocities"][row, frames].float()), dim=-1)
    proprio = (proprio - constants["PROPRIO_MEAN"]) / constants["PROPRIO_STD"]
    action = raw["actions"][row, start:end].float() / 100.0
    action = (action - constants["ACTION_MEAN"]) / constants["ACTION_STD"]
    visual = transform((image / 255.0).permute(0, 3, 1, 2))
    return ({"visual": visual, "proprio": proprio}, action.reshape(4, 10),
            state, torch.zeros(4, dtype=torch.float32))


def _compare_clip(reference, selected, identity):
    before = _rng()
    try:
        expected = reference()
        after = _rng()
        _restore(before)
        actual = selected()
        assert_same(after, _rng())
        assert_same(expected, actual)
        # torch.equal alone would not distinguish signed zero; hash exact bytes.
        expected_hash, actual_hash = _batch_digest(expected), _batch_digest(actual)
        if expected_hash != actual_hash:
            raise ValueError("Native selected-frame bytes changed")
        return actual, {"identity": list(identity), "sample_sha256": actual_hash,
                        "rng_before_sha256": _rng_hash(before), "rng_after_sha256": _rng_hash(after)}
    except BaseException:
        _restore(before)
        raise


def _append_sample(batch, sample, index, count=256):
    obs, action, state, reward = sample
    values = {"visual": obs["visual"], "proprio": obs["proprio"],
              "action": action, "state": state, "reward": reward}
    expected = {"visual": (4, 3, 224, 224), "proprio": (4, 4),
                "action": (4, 10), "state": (4, 7), "reward": (4,)}
    if set(obs) != {"visual", "proprio"}:
        raise ValueError("Native observation keys changed")
    for name, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
                value.dtype != torch.float32 or tuple(value.shape) != expected[name] or
                value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Native CPU training modality changed: " + name)
        if index == 0:
            batch[name] = torch.empty((count, *value.shape), dtype=value.dtype)
        batch[name][index].copy_(value)


def _batch_tuple(batch):
    return ({"visual": batch["visual"], "proprio": batch["proprio"]},
            batch["action"], batch["state"], batch["reward"])


def _runtime_binding(vendor, module, transforms):
    expected = [(module, "app/plan_common/datasets/pusht_dset.py"),
                (transforms, "app/plan_common/datasets/transforms.py")]
    for imported, relative in expected:
        if Path(imported.__file__).resolve() != vendor / relative:
            raise ValueError("Previously imported native module belongs to another source")
    for relative, digest in EXTRA_SOURCES.items():
        if sha256(vendor / relative) != digest:
            raise ValueError("Native crop implementation changed")
    for name, relative in (("app.plan_common.datasets.traj_dset", "app/plan_common/datasets/traj_dset.py"),
                           ("src.datasets.utils.video.transforms", "src/datasets/utils/video/transforms.py"),
                           ("src.datasets.utils.video.functional", "src/datasets/utils/video/functional.py")):
        if Path(sys.modules[name].__file__).resolve() != vendor / relative:
            raise ValueError("Native dependency resolved outside pinned source")
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "torchvision", "decord", "einops", "Pillow")}
    return {"versions": versions, "python": sys.version,
            "extra_source_sha256": EXTRA_SOURCES,
            "producer_sha256": sha256(Path(__file__)),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}


def produce_pusht_batch(vendor, data_root, *, raw_inputs_receipt, provenance, output, model_seed=234):
    """Create a hash-checked first-update batch and _input_gate-compatible receipt.

    No permission is inferred from raw verification. This entrypoint implements
    the already authorized, training-only engineering check; it cannot launch a
    model or history. Its deterministic engineering RNG is *not* an assertion
    about native persistent-worker initialization or continuation states.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Dedicated CPU process with CUDA_VISIBLE_DEVICES empty required")
    vendor, data_root = Path(vendor).resolve(), Path(data_root)
    receipt, provenance, output = Path(raw_inputs_receipt), Path(provenance), Path(output)
    contract = native_contract(vendor, "pusht", model_seed)
    manifest, raw_hash = _raw_binding(data_root, receipt, provenance)
    if not output.is_absolute() or output.is_symlink() or data_root.resolve() in output.resolve().parents:
        raise ValueError("Fresh explicit evidence output outside raw data required")
    output.mkdir(parents=True, exist_ok=False)
    ambient, started = _rng(), time.monotonic()
    base_protocol = {"role": "training_only_first_update_input_parity", "version": 1,
        "task": "pusht", "data_root": str(data_root), "raw_inputs_receipt": str(receipt.resolve()),
        "raw_input_report_sha256": raw_hash, "provenance": str(provenance.resolve()),
        "epoch": 0, "global_update_index": 0, "model_seed": model_seed,
        "engineering_rng_seed": ENGINEERING_SEED, "data_seed": contract["sampler"]["data_seed"],
        "validation_or_confirmation_access": False, "engineering_only": True,
        "worker_rng_initialization_or_resume_verified": False, "gpu_parity_established": False}
    write_json(output / "protocol.json", base_protocol)
    try:
        sides = ["train/" + name for name in NATIVE_SIDECARS]
        checked = _verify_training_members(data_root, manifest, sides)
        use_vendor(vendor)
        module = importlib.import_module("app.plan_common.datasets.pusht_dset")
        transforms = importlib.import_module("app.plan_common.datasets.transforms")
        runtime = _runtime_binding(vendor, module, transforms)
        transform = _native_transform(vendor, contract["config"], transforms.make_transforms)
        _, _, call = native_loader_probe(vendor, "pusht")
        original = _native_training_branch(vendor / "app/plan_common/datasets/pusht_dset.py",
                                           vars(module), call, transform, data_root)
        if len(original.dataset) != COUNTS["pusht"][0] or len(original) != TRAIN_CLIPS:
            raise ValueError("Pinned native training population changed")
        indices = _first_update_indices(contract, len(original))
        # Independent sidecar loads: selected path never uses normalized native
        # arrays or native get_frames/getitem/slices to produce its modalities.
        raw = {key: torch.load(data_root / "train" / filename, map_location="cpu", weights_only=True)
               for key, filename in (("states", "states.pth"), ("actions", "rel_actions.pth"),
                                     ("velocities", "velocities.pth"))}
        with (data_root / "train/seq_lengths.pkl").open("rb") as source:
            lengths = pickle.load(source)
        identities, count = _independent_slices(lengths, indices, 4, 5, call["random_seed"])
        if count != len(original):
            raise ValueError("Independent clip count differs")
        assert_same([tuple(original.slices[i]) for i in indices], identities)
        if any(type(row) is not int or not 0 <= row < COUNTS["pusht"][0] for row, _, _ in identities):
            raise ValueError("Selection escaped released training rows")
        videos = [f"train/obses/episode_{row:03d}.mp4" for row, _, _ in identities]
        checked.update(_verify_training_members(data_root, manifest, videos))
        write_json(output / "selected_inputs.json", checked)
        constants = {name: getattr(module, name) for name in
                     ("ACTION_MEAN", "ACTION_STD", "PROPRIO_MEAN", "PROPRIO_STD")}
        random.seed(ENGINEERING_SEED)
        np.random.seed(ENGINEERING_SEED)
        torch.set_rng_state(torch.Generator().manual_seed(ENGINEERING_SEED).get_state())
        first_rng, batch, comparisons = _rng(), {}, []
        for position, (clip, identity) in enumerate(zip(indices, identities)):
            row, start, end = identity

            def selected():
                module.decord.bridge.set_bridge("torch")
                reader = module.VideoReader(str(data_root / f"train/obses/episode_{row:03d}.mp4"),
                                            num_threads=1)
                image = reader.get_batch(list(range(start, end, 5)))
                return _selected_sample(raw, constants, identity, image, transform)

            sample, comparison = _compare_clip(lambda: original[clip], selected, identity)
            _append_sample(batch, sample, position)
            comparisons.append({"rank": position // 8, "rank_offset": position % 8,
                                "native_clip_index": clip, **comparison})
            write_json(output / "progress.json", {"completed_clips": position + 1,
                "total_clips": 256, "validation_or_confirmation_access": False})
        final_rng = _rng()
        # Recheck bytes/source/provenance after decode; a stale raw audit alone
        # must not attest files that changed between verification and use.
        assert_same(checked, _verify_training_members(data_root, manifest, list(checked)))
        _, final_raw_hash = _raw_binding(data_root, receipt, provenance)
        assert_same(raw_hash, final_raw_hash)
        assert_same(contract, native_contract(vendor, "pusht", model_seed))
        assert_same(runtime, _runtime_binding(vendor, module, transforms))
        result = _batch_tuple(batch)
        size = sum(v.numel() * v.element_size() for v in batch.values())
        if size > MAX_OUTPUT_BYTES - (16 << 20) or shutil.disk_usage(output).free < size + (16 << 20):
            raise ValueError("Insufficient reserved CPU evidence disk; do not truncate the native batch")
        torch.save(result, output / "batch.pt.partial")
        if (output / "batch.pt.partial").stat().st_size > MAX_OUTPUT_BYTES:
            raise ValueError("Native batch evidence exceeds one GiB bound")
        os.replace(output / "batch.pt.partial", output / "batch.pt")
        batch_hash = _batch_digest(result)
        protocol = {**base_protocol, "training_clip_indices": indices, "batch_sha256": batch_hash,
            "batch_file_sha256": sha256(output / "batch.pt"), "runtime": runtime,
            "selected_inputs_sha256": sha256(output / "selected_inputs.json")}
        write_json(output / "protocol.json", protocol)
        write_json(output / "comparisons.json", {"clips": comparisons,
            "initial_rng_sha256": _rng_hash(first_rng), "final_rng_sha256": _rng_hash(final_rng)})
        report = {"status": "native32_training_batch_input_parity_verified", "task": "pusht",
            "protocol_sha256": sha256(output / "protocol.json"),
            "native_config_sha256": contract["config_sha256"], "native_source_sha256": SOURCE_HASHES,
            "native_pixels_actions_proprio_states_rewards_rng_equal": True,
            "all_selected_inputs_content_verified": True, "permitted_training_rows_only": True,
            "validation_or_confirmation_access": False, "engineering_only": True,
            "global_batch": 256, "sampler_policy": contract["sampler"],
            "comparisons_sha256": sha256(output / "comparisons.json"),
            "batch_bytes": size, "selected_training_rows": len({r for r, _, _ in identities}),
            "worker_rng_initialization_or_resume_verified": False,
            "gpu_parity_established": False, "history_execution_authorized": False,
            "proof_scope": "first native rank-ordered batch; whole-video versus selected-frame access",
            "seconds": time.monotonic() - started, "gpu_calls": 0}
        write_json(output / "report.json", report)
        write_json(output / "DONE.json", {"report_sha256": sha256(output / "report.json")})
        _input_gate(contract, result, output)
        return result, report
    except BaseException as error:
        write_json(output / "FAILED.json", {"error": str(error), "error_type": type(error).__name__,
            "validation_or_confirmation_access": False, "gpu_calls": 0,
            "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        _restore(ambient)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "data-root", "raw-inputs-receipt", "provenance", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--model-seed", type=int, choices=(234, 235, 236), default=234)
    args = parser.parse_args()
    _, report = produce_pusht_batch(args.vendor, args.data_root,
        raw_inputs_receipt=args.raw_inputs_receipt, provenance=args.provenance,
        output=args.output, model_seed=args.model_seed)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
