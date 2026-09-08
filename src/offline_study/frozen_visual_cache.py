"""Engineering-only, lossless memoization at a frozen image encoder boundary.

This module does not activate caching in a scientific training runner. The caller
must still execute the original dataset transforms, provide the original ordered
frame identities, and prove independent-batch plus complete-update/RNG parity.
Record mode always runs the *whole original batch*, never a reduced miss batch.
Replay mode is deliberately named engineering_replay and fails on every miss.
No downloads, cache population, model construction or GPU work occur on import.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import inspect
import json
import math
import marshal
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import struct
import tempfile

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def tensor_digest(tensor):
    """Hash actual bytes, preserving BF16, signed zero and every mantissa bit."""
    if tensor.layout != torch.strided:
        raise ValueError("Only dense encoder tensors are supported")
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(canonical([str(value.dtype), list(value.shape)]) +
        value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def encoder_digest(encoder):
    return digest({name: tensor_digest(value) for name, value in encoder.state_dict().items()})


def runtime_identity(device):
    device = torch.device(device)
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda, "device_type": device.type}
    if device.type == "cuda":
        # Only an explicitly invoked engineering session reaches this branch.
        props = torch.cuda.get_device_properties(device)
        result.update(device_uuid=str(props.uuid), device_name=props.name,
            capability=[props.major, props.minor], cudnn=torch.backends.cudnn.version())
    elif device.type != "cpu":
        raise ValueError("Only receiving CUDA or CPU-test runtimes are supported")
    return result


def precision_identity(device_type):
    if hasattr(torch, "get_autocast_dtype"):
        enabled = torch.is_autocast_enabled(device_type)
        dtype = torch.get_autocast_dtype(device_type)
    else:  # Local CPU fixtures may use an older torch than receiving workers.
        enabled = torch.is_autocast_cpu_enabled() if device_type == "cpu" else torch.is_autocast_enabled()
        dtype = torch.get_autocast_cpu_dtype() if device_type == "cpu" else torch.get_autocast_gpu_dtype()
    return {"autocast_enabled": enabled,
        "autocast_dtype": str(dtype),
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@dataclass(frozen=True)
class Frame:
    """Identity from a verified official-input manifest, not a sampled outcome."""
    relative_path: str
    file_sha256: str
    index: int

    def validate(self):
        path = PurePosixPath(self.relative_path)
        if (not self.relative_path or path.is_absolute() or ".." in path.parts or
                "\\" in self.relative_path or "\x00" in self.relative_path or len(self.relative_path.encode()) > 4096 or
                str(path) != self.relative_path or not valid_sha(self.file_sha256) or
                type(self.index) is not int or self.index < 0):
            raise ValueError("Invalid official frame identity")


def frame_inventory(frames):
    records = []
    for frame in frames:
        frame.validate()
        records.append(asdict(frame))
    records.sort(key=canonical)
    if not records or len({canonical(row) for row in records}) != len(records):
        raise ValueError("Frame inventory must be nonempty and unique")
    return records


@dataclass(frozen=True)
class Binding:
    task: str
    source_sha256: str
    encoder_sha256: str
    transform_sha256: str
    input_manifest_sha256: str
    frame_inventory_sha256: str
    runtime_sha256: str
    precision_sha256: str
    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    input_dtype: str
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    output_dtype: str

    def validate(self):
        if self.task not in ("wall", "pointmaze"):
            raise ValueError("Cache is scoped to the two audited navigation tasks")
        for name, value in asdict(self).items():
            if name.endswith("_sha256") and not valid_sha(value):
                raise ValueError("Missing/invalid binding: " + name)
        for side in ("input", "output"):
            shape, strides = getattr(self, side + "_shape"), getattr(self, side + "_stride")
            if (len(shape) != (4 if side == "input" else 3) or len(shape) != len(strides) or
                    any(type(n) is not int or n <= 0 for n in shape + strides) or
                    getattr(self, side + "_dtype") not in ("torch.float32", "torch.bfloat16", "torch.float16", "torch.float64")):
                raise ValueError("Invalid exact tensor shape/dtype/stride contract")
            # Restrict to nonoverlapping positive-stride layouts, including DINO's
            # class-token gap between successive patch-token images.
            span = 1
            for stride, size in sorted(zip(strides, shape)):
                if size > 1 and stride < span:
                    raise ValueError("Overlapping tensor layout is unsupported")
                span += (size - 1) * stride
            if span > math.prod(shape) * 2:
                raise ValueError("Unbounded padded tensor layout")
        if self.input_shape[0] != self.output_shape[0]:
            raise ValueError("Image encoder batch dimension changed")

    @property
    def sha256(self):
        self.validate()
        return digest(asdict(self))


def _publish(path, writer):
    """Publish a complete local file atomically, without replacing old evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        temporary.unlink(missing_ok=True)  # Only the tempfile created above.


class CacheStore:
    """Immutable checksummed frame entries; one writer and bounded host LRU.

    Opening for writing takes a process lock for the object's entire lifetime.
    Independent readers may share a worker-local cache. No network filesystem or
    Drive hot-path access is configured here. The explicit byte quota and free-
    disk reserve are required; callers must reserve storage before construction.
    """
    @classmethod
    def create(cls, root, binding, frames, *, max_disk_bytes, min_free_bytes, max_memory_bytes=0):
        binding.validate()
        records = frame_inventory(frames)
        if digest(records) != binding.frame_inventory_sha256:
            raise ValueError("Frame inventory differs from frozen binding")
        if any(type(n) is not int or n < 0 for n in (max_disk_bytes, min_free_bytes, max_memory_bytes)) or not max_disk_bytes:
            raise ValueError("Explicit nonnegative memory/disk limits required")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        _publish(root / "frames.json", lambda stream: stream.write(canonical(records)))
        metadata = {"schema": "frozen_visual_cache_engineering_v1", "binding": asdict(binding),
            "binding_sha256": binding.sha256, "max_disk_bytes": max_disk_bytes,
            "min_free_bytes": min_free_bytes, "scientific_activation": False}
        _publish(root / "manifest.json", lambda stream: stream.write(canonical(metadata)))
        return cls(root, binding, writable=True, max_memory_bytes=max_memory_bytes)

    def __init__(self, root, binding, *, writable=False, max_memory_bytes=0):
        binding.validate()
        if type(max_memory_bytes) is not int or max_memory_bytes < 0:
            raise ValueError("Invalid bounded memory allowance")
        self.root, self.binding = Path(root), binding
        if self.root.is_symlink():
            raise ValueError("Cache root cannot be a symbolic link")
        self.metadata = json.loads((self.root / "manifest.json").read_text())
        if (self.metadata.get("schema") != "frozen_visual_cache_engineering_v1" or
                self.metadata.get("scientific_activation") is not False or
                self.metadata.get("binding_sha256") != binding.sha256 or
                canonical(self.metadata.get("binding")) != canonical(asdict(binding))):
            raise ValueError("Cache source/task/runtime/precision binding changed")
        records = json.loads((self.root / "frames.json").read_text())
        if digest(records) != binding.frame_inventory_sha256:
            raise ValueError("Cache frame inventory changed")
        self.allowed = {canonical(asdict(Frame(**row))) for row in records}
        self.max_memory_bytes, self.memory_bytes = max_memory_bytes, 0
        self.memory, self.closed, self.lock = OrderedDict(), False, None
        self.writable = writable
        if writable:
            self.lock = (self.root / ".writer.lock").open("a+b")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                self.lock.close()
                raise
            # Recovery counts committed entries, not an interruptible counter.
            self.disk_bytes = sum(path.stat().st_size for path in self.root.glob("entries/*/*.pt"))
            if self.disk_bytes > self.metadata["max_disk_bytes"]:
                self.close()
                raise ValueError("Existing cache exceeds its reserved disk quota")

    def path(self, frame):
        frame.validate()
        if canonical(asdict(frame)) not in self.allowed:
            raise ValueError("Frame is absent from the bound official inventory")
        key = digest({"binding": self.binding.sha256, "frame": asdict(frame)})
        path = self.root / "entries" / key[:2] / (key + ".pt")
        if any(part.is_symlink() for part in (path, path.parent, path.parent.parent)):
            raise ValueError("Cache entry paths cannot contain symbolic links")
        return path

    def _tensor(self, value):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad or
                tuple(value.shape) != self.binding.output_shape[1:] or str(value.dtype) != self.binding.output_dtype or
                not value.is_contiguous() or not torch.isfinite(value).all()):
            raise ValueError("Cached feature dtype/shape/finite/detached contract failed")

    def get(self, frame, pixels_sha256):
        if self.closed or not valid_sha(pixels_sha256):
            raise ValueError("Closed cache or invalid transformed-pixel checksum")
        path = self.path(frame)
        if not path.is_file():
            raise KeyError("Unpopulated frozen visual frame: " + str(frame))
        if path.is_symlink():
            raise ValueError("Cache entries cannot be symbolic links")
        signature = (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
        prior = self.memory.pop(path, None)
        if prior is not None:
            payload, before = prior
            self.memory_bytes -= payload["tensor"].numel() * payload["tensor"].element_size()
            if before != signature:
                raise ValueError("Published cache entry changed while resident")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self._tensor(payload["tensor"])
        if (payload.get("binding_sha256") != self.binding.sha256 or payload.get("frame") != asdict(frame) or
                payload.get("pixels_sha256") != pixels_sha256 or
                payload.get("feature_sha256") != tensor_digest(payload["tensor"])):
            raise ValueError("Cache entry input/content/binding checksum failed")
        size = payload["tensor"].numel() * payload["tensor"].element_size()
        if size <= self.max_memory_bytes:
            while self.memory and self.memory_bytes + size > self.max_memory_bytes:
                _, (old, _) = self.memory.popitem(last=False)
                self.memory_bytes -= old["tensor"].numel() * old["tensor"].element_size()
            self.memory[path] = payload, signature
            self.memory_bytes += size
        return payload["tensor"].clone()  # Downstream writes never mutate cached storage.

    def put(self, frame, pixels_sha256, feature):
        if self.closed or not self.writable or not valid_sha(pixels_sha256):
            raise ValueError("An open exclusive writer and valid pixel checksum are required")
        if feature.requires_grad:
            raise ValueError("Never detach a trainable visual feature to make it cacheable")
        value = feature.detach().cpu().contiguous().clone()
        self._tensor(value)
        path = self.path(frame)
        if path.exists():
            if tensor_digest(self.get(frame, pixels_sha256)) != tensor_digest(value):
                raise ValueError("Independent batch composition changed this frame's encoder bits")
            return False
        payload = {"binding_sha256": self.binding.sha256, "frame": asdict(frame),
            "pixels_sha256": pixels_sha256, "feature_sha256": tensor_digest(value), "tensor": value}
        # Conservative metadata/serialization allowance, checked before writing.
        reserve = value.numel() * value.element_size() + 8192
        if (self.disk_bytes + reserve > self.metadata["max_disk_bytes"] or
                shutil.disk_usage(self.root).free < self.metadata["min_free_bytes"] + reserve):
            raise OSError("Reserved local cache capacity/free-disk guard reached")
        if not _publish(path, lambda stream: torch.save(payload, stream)):
            raise ValueError("Unexpected concurrent cache publisher")
        self.disk_bytes += path.stat().st_size
        return True

    def close(self):
        self.memory.clear()
        self.memory_bytes, self.closed = 0, True
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def rng_snapshot():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else []}


def restore_rng(snapshot):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if snapshot["torch_cuda"]:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def assert_bitwise(left, right):
    if type(left) is not type(right):
        raise ValueError("Parity value type changed")
    if isinstance(left, torch.Tensor):
        if (left.shape != right.shape or left.dtype != right.dtype or left.stride() != right.stride() or
                left.device != right.device or tensor_digest(left) != tensor_digest(right)):
            raise ValueError("Tensor bits/layout/device changed")
    elif isinstance(left, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise ValueError("Array/RNG bits changed")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Parity mapping changed")
        for key in left:
            assert_bitwise(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError("Parity sequence changed")
        for a, b in zip(left, right):
            assert_bitwise(a, b)
    elif isinstance(left, float):
        if struct.pack("!d", left) != struct.pack("!d", right):
            raise ValueError("Scalar bits changed")
    elif left != right:
        raise ValueError("Parity value changed")


def evidence_snapshot(value):
    """Prevent the second callback from changing aliased first-run evidence."""
    if isinstance(value, torch.Tensor):
        result = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
        result.copy_(value.detach())
        return result
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return type(value)((key, evidence_snapshot(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return type(value)(evidence_snapshot(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("Unsupported mutable full-update parity evidence")


def class_identity(cls):
    """Bind actual loaded Python class code, not only a matching class name."""
    source = Path(inspect.getfile(cls))
    methods = {base.__module__ + "." + base.__qualname__ + "." + name:
        hashlib.sha256(marshal.dumps(method.__code__)).hexdigest()
        for base in cls.__mro__ if base not in (torch.nn.Module, object)
        for name, method in vars(base).items() if inspect.isfunction(method)}
    return {"class": cls.__module__ + "." + cls.__qualname__,
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "methods": methods}


class EngineeringModePolicy:
    """Explicit source-bound exception to default eval-only engineering.

    This does not authorize scientific caching. Unknown classes, registered
    buffers (including nonpersistent state), nonzero stochastic rates, mutable
    Python state and mixed module modes are rejected. Replay requires a complete
    live proof on an isolated clone: eval/train x original/two alternate batches.
    The caller's actual encoder modes are never changed by the proof or sessions.
    """
    TORCH_TYPES = frozenset((torch.nn.ModuleList, torch.nn.Identity, torch.nn.Linear,
        torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.GELU, torch.nn.Dropout))
    ZERO_RATES = frozenset(("p", "drop_prob", "drop_path_rate", "sample_drop_ratio",
        "attn_drop", "proj_drop", "dropout", "dropout_p", "dropout_prob"))

    def __init__(self, encoder, binding, *, audited_classes, source_receipt):
        binding.validate()
        if source_receipt.get("source_sha256") != binding.source_sha256 or not audited_classes:
            raise ValueError("Explicit matching source/encoder mode policy required")
        if any(not valid_sha(value) for value in audited_classes.values()):
            raise ValueError("Every audited custom class needs its reviewed file checksum")
        self.binding_sha256 = binding.sha256
        self.audited_classes = dict(audited_classes)
        self.source_receipt = json.loads(canonical(source_receipt))
        self._proof = None
        if encoder_digest(encoder) != binding.encoder_sha256:
            raise ValueError("Mode policy encoder weights differ from binding")
        self.inventory = self._inventory(encoder)
        self.contract = digest(self.receipt())

    def receipt(self):
        return {"schema": "source_bound_encoder_mode_engineering_v2",
            "binding_sha256": self.binding_sha256, "source_receipt": self.source_receipt,
            "audited_classes": self.audited_classes, "module_inventory": self.inventory,
            "allowed_modes": ["all_eval", "all_train"], "registered_buffers_allowed": False,
            "native_modes_changed": False, "scientific_activation": False}

    def _inventory(self, encoder, *, routed_root=False):
        result, identities = [], {}
        base_attributes = set(vars(torch.nn.Module()))
        def immutable(value):
            if value is None or type(value) in (bool, int, str):
                return value
            if type(value) is float and math.isfinite(value):
                return value
            if type(value) in (tuple, list):
                return {type(value).__name__: [immutable(item) for item in value]}
            raise ValueError("Unaudited mutable/stateful encoder attribute")
        for name, module in encoder.named_modules():
            cls = type(module)
            qualified = cls.__module__ + "." + cls.__qualname__
            if cls not in self.TORCH_TYPES and qualified not in self.audited_classes:
                raise ValueError("Unaudited encoder component: " + qualified)
            if list(module.buffers(recurse=False)):
                raise ValueError("Stateful/buffer-bearing encoder components are unsupported")
            if cls not in identities:
                identities[cls] = class_identity(cls)
            identity = identities[cls]
            if cls not in self.TORCH_TYPES and identity["file_sha256"] != self.audited_classes[qualified]:
                raise ValueError("Reviewed encoder class source changed")
            attributes = {}
            for key, value in vars(module).items():
                if key in base_attributes or (not name and key == "forward" and routed_root):
                    continue
                if key in self.ZERO_RATES and (type(value) not in (int, float) or value != 0):
                    raise ValueError("Nonzero stochastic encoder rate: " + name + "." + key)
                attributes[key] = immutable(value)
            result.append({"name": name, "type": identity, "attributes": attributes})
        if len({module.training for module in encoder.modules()}) != 1:
            raise ValueError("Mixed encoder modes are outside the audited native lifecycle")
        return result

    def guard(self, encoder, binding, *, replay=False, routed_root=False):
        if binding.sha256 != self.binding_sha256 or digest(self.receipt()) != self.contract:
            raise ValueError("Bound encoder mode policy changed")
        if self._inventory(encoder, routed_root=routed_root) != self.inventory:
            raise ValueError("Audited encoder code/configuration/Python state changed")
        if replay and self._proof is None:
            raise ValueError("Train/eval and alternate-composition bitwise proof is required before replay")

    def prove(self, encoder, store, frames, orders, batches):
        """`batches(order)` yields original-size (frame IDs, transformed images).

        All keys must already exist; failures never add new cache entries or
        authorize replay. No transforms are skipped in the training comparison.
        Only the isolated clone changes modes, preserving original model/RNG.
        """
        self._proof = None
        self.guard(encoder, store.binding)
        size = store.binding.input_shape[0]
        if len(frames) % size or len(orders) != 3 or next(iter(orders)) != "original":
            raise ValueError("Require original and exactly two complete alternate compositions")
        full = list(range(len(frames)))
        if (orders["original"] != full or any(sorted(order) != full for order in orders.values()) or
                len({tuple(order) for order in orders.values()}) != 3):
            raise ValueError("Mode proof compositions must be distinct full permutations")
        if {canonical(asdict(frame)) for frame in frames} != store.allowed:
            raise ValueError("Mode proof must cover every bound frame")
        for frame in frames:
            if not store.path(frame).is_file():
                raise ValueError("Mode proof requires prepopulated original eval features")
        original_modes = [(name, module.training) for name, module in encoder.named_modules()]
        original_rng, original_state = rng_snapshot(), encoder_digest(encoder)
        rows = []
        try:
            clone = copy.deepcopy(encoder)
            for training in (False, True):
                clone.train(training)  # Excluded engineering clone ONLY.
                for label, order in orders.items():
                    count = 0
                    with EncoderCacheSession(clone, store, mode="record", mode_policy=self) as session:
                        for actual_frames, images in batches(order):
                            expected = tuple(frames[i] for i in order[count:count + size])
                            if tuple(actual_frames) != expected or len(expected) != size:
                                raise ValueError("Mode proof input order/complete batches changed")
                            with session.frames(actual_frames):
                                clone(images)  # put compares to prior eval bits, not to a new target.
                            count += size
                        if count != len(frames):
                            raise ValueError("Incomplete mode proof frame coverage")
                    rows.append({"training": training, "composition": label,
                        "order_sha256": digest(order), "full_encoder_batches": session.calls,
                        "all_existing_frame_bits_equal": True, "rng_and_state_unchanged": True})
            assert_bitwise(original_rng, rng_snapshot())
            self._proof = {"policy_sha256": self.contract, "binding_sha256": store.binding.sha256,
                "frame_inventory_sha256": store.binding.frame_inventory_sha256, "checks": rows,
                "native_modes_changed": False, "scientific_activation": False}
            return evidence_snapshot(self._proof)
        finally:
            restore_rng(original_rng)
            if ([(name, module.training) for name, module in encoder.named_modules()] != original_modes or
                    encoder_digest(encoder) != original_state):
                self._proof = None
                raise ValueError("Mode proof changed the original encoder")


class EncoderCacheSession:
    """Explicit engineering hook; never installs itself in a training runner.

    Frames must be bound AFTER the unchanged native transform, before exactly one
    encoder call. Context/activation-level hooks and trainable/stochastic encoders
    are rejected. Input hashes cover the actual transformed, dtype-cast pixels.
    Output reconstruction preserves DINO's original noncontiguous batch stride.
    """
    def __init__(self, encoder, store, *, mode, mode_policy=None):
        if mode not in ("record", "engineering_replay"):
            raise ValueError("Scientific activation is not implemented; independent batch/update/RNG gates required")
        self.encoder, self.store, self.mode = encoder, store, mode
        self.mode_policy = mode_policy
        self.pending, self.calls, self.active = None, 0, False

    def _guard(self):
        tensors = list(self.encoder.named_parameters()) + list(self.encoder.named_buffers())
        global_hooks = torch.nn.modules.module
        if (any(value.requires_grad for _, value in tensors) or
                (self.mode_policy is None and any(module.training for module in self.encoder.modules())) or
                any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                    for module in self.encoder.modules()) or
                any(getattr(global_hooks, name, {}) for name in
                    ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks"))):
            raise ValueError("Require frozen eval-mode encoder without external activation hooks")
        modes = tuple(module.training for module in self.encoder.modules())
        if hasattr(self, "native_modes") and modes != self.native_modes:
            raise ValueError("Native encoder mode changed inside one cache session")
        if self.mode_policy is not None:
            self.mode_policy.guard(self.encoder, self.store.binding,
                replay=self.mode == "engineering_replay", routed_root=self.active)
        versions = [(name, id(value), value.data_ptr(), value._version) for name, value in tensors]
        if hasattr(self, "versions") and versions != self.versions:
            raise ValueError("Frozen encoder state was changed or replaced")
        return versions

    def __enter__(self):
        if self.active:
            raise ValueError("Nested encoder cache session")
        self.versions = self._guard()
        self.native_modes = tuple(module.training for module in self.encoder.modules())
        if encoder_digest(self.encoder) != self.store.binding.encoder_sha256:
            raise ValueError("Actual encoder weights/buffers differ from cache binding")
        self.original, self.had_forward = self.encoder.forward, "forward" in self.encoder.__dict__
        self.encoder.forward, self.active = self.forward, True
        return self

    @contextmanager
    def frames(self, frames):
        if not self.active or self.pending is not None:
            raise ValueError("Inactive/nested frame binding")
        frames = tuple(frames)
        if len(frames) != self.store.binding.input_shape[0]:
            raise ValueError("Original full encoder batch must be preserved")
        for frame in frames:
            self.store.path(frame)
        self.pending, self.used = frames, False
        try:
            yield
            if not self.used:
                raise ValueError("Frame binding did not execute exactly one encoder call")
        finally:
            self.pending = None

    def forward(self, images):
        self._guard()
        binding = self.store.binding
        if self.pending is None or self.used:
            raise ValueError("Every encoder call requires exactly one ordered frame binding")
        self.used = True
        if (images.requires_grad or tuple(images.shape) != binding.input_shape or
                images.stride() != binding.input_stride or str(images.dtype) != binding.input_dtype or
                digest(runtime_identity(images.device)) != binding.runtime_sha256 or
                digest(precision_identity(images.device.type)) != binding.precision_sha256):
            raise ValueError("Native input/runtime/precision/gradient contract changed")
        before = rng_snapshot()
        pixels = [tensor_digest(image) for image in images]
        if self.mode == "record":
            output = self.original(images)  # No miss-only regrouping or smaller batch.
            # Never publish features from a stochastic or input-mutating encoder.
            assert_bitwise(before, rng_snapshot())
            if pixels != [tensor_digest(image) for image in images]:
                raise ValueError("Native encoder mutated its transformed input")
            self._guard()
            if (output.requires_grad or tuple(output.shape) != binding.output_shape or
                    output.stride() != binding.output_stride or str(output.dtype) != binding.output_dtype):
                raise ValueError("Native encoder output contract changed")
            for frame, pixel, feature in zip(self.pending, pixels, output):
                self.store.put(frame, pixel, feature)
        else:
            host = torch.stack([self.store.get(frame, pixel) for frame, pixel in zip(self.pending, pixels)])
            output = torch.empty_strided(binding.output_shape, binding.output_stride,
                dtype=host.dtype, device=images.device)
            output.copy_(host)
        assert_bitwise(before, rng_snapshot())
        self._guard()
        self.calls += 1
        return output

    def __exit__(self, kind, exc, traceback):
        try:
            if self.had_forward:
                self.encoder.forward = self.original
            else:
                del self.encoder.forward
        finally:
            self.active, self.pending = False, None
        if kind is None:
            self._guard()
            if encoder_digest(self.encoder) != self.store.binding.encoder_sha256:
                raise ValueError("Encoder content changed during cache engineering")


def paired_update_check(native_run, cached_run):
    """Run independent caller-owned complete-update fixtures from identical RNG.

    Callbacks must own separate, identically initialized model/optimizer states
    and return all mandatory evidence below. This helper never resets models or
    authorizes science; a receiving pilot must cover original data/compositions,
    multiple complete updates and a native validation boundary.
    """
    initial = rng_snapshot()
    required = {"losses", "parameters", "gradients", "optimizer", "scaler", "scheduler",
                "logical_rngs", "loader_rng", "validation_events", "updates"}
    try:
        native = evidence_snapshot(native_run())
        native_rng = rng_snapshot()
        if (not isinstance(native, dict) or not required <= native.keys() or
                type(native["updates"]) is not int or native["updates"] < 2 or
                type(native["validation_events"]) is not int or native["validation_events"] < 1):
            raise ValueError("Incomplete independent full-update parity evidence")
        restore_rng(initial)
        cached = cached_run()
        assert_bitwise(native, cached)
        assert_bitwise(native_rng, rng_snapshot())
        return {"scope": "engineering_only_not_scientific_activation", "complete_updates": native["updates"],
            "native_and_cached_state_and_rng_bitwise": True,
            "independent_batch_composition_gate_still_required": True}
    finally:
        restore_rng(initial)
