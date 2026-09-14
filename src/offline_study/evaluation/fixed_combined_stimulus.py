"""CPU-only preparation of one original Reach fitting clip for combined checks.

No model, dataset constructor, new normalization fit, or scientific evaluation.
Use published planning statistics and the original fitting image transform.
The native selected-row reader is an independent parity reference, not a claim
of parity with the full dataset constructor or its recomputed statistics.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.validation.fixed_combined_check import NORMALIZATION_SCOPE, RAW_PROOF_SHA256, STATS_SOURCE_SHA256, assert_bytes, deadline, exclusive_json, load_stimulus, verify_stimulus
from offline_study.data.inventory import _metaworld_trajectory_group
from offline_study.core.protocol import sha256
from offline_study.data.robotics_training_inputs import MANIFEST_HASHES, bound_manifest, safe_member
from offline_study.models.vendor import use_vendor

STATS_SOURCE = "app/plan_common/datasets/__init__.py"
READER_SOURCE = "app/plan_common/datasets/metaworld_hf_dset.py"
READER_SHA256 = "83e5975c90870f316140a3d6db22887b9bb46a673ffbc0fd5cb3c33bcefb19c6"
NORMALIZATION_KEYS = tuple(f"{name}_{stat}" for name in ("action", "state", "proprio")
                           for stat in ("mean", "std"))


def published_stats(path):
    if sha256(path) != STATS_SOURCE_SHA256:
        raise ValueError("Published normalization source changed")
    tree = ast.parse(path.read_text())
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "DATA_STATS" for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError("Ambiguous published statistics")
    raw = ast.literal_eval(assignments[0].value)["metaworld"]
    stats = {key: torch.tensor(raw[key], dtype=torch.float32) for key in NORMALIZATION_KEYS}
    for key, value in stats.items():
        if value.shape != ((39,) if key.startswith("state_") else (4,)) or not torch.isfinite(value).all():
            raise ValueError("Invalid published statistics")
        if key in ("action_std", "proprio_std") and not (value > 0).all():
            raise ValueError("Invalid normalization divisor")
    return stats


def locate_row(index, counts):
    if type(index) is not int or index < 0 or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Invalid original row metadata")
    for name, count in sorted(counts.items()):
        if index < count:
            return {"path": name, "local_row": index}
        index -= count
    raise ValueError("Row outside original population")


def read_selected_row(path, index):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        if index < count:
            # Arrow may read a containing physical row group. Convert ONLY the
            # selected row to Python; no other row's state/video is interpreted.
            table = parquet.read_row_group(group,
                columns=["task", "seed", "episode", "states", "actions", "video"])
            return table.slice(index, 1).to_pylist()[0]
        index -= count
    raise ValueError("Missing selected parquet row")


def validate_raw_row(row, selected):
    states = torch.tensor(np.asarray(row["states"]), dtype=torch.float32)
    actions = torch.tensor(np.asarray(row["actions"]), dtype=torch.float32)
    if states.shape != (100, 39) or actions.shape != (99, 4):
        raise ValueError("Original state/action alignment changed")
    if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise ValueError("Nonfinite fitting row")
    if (row["task"] != selected["task"] or row["seed"] != selected["source_seed"] or
            row["episode"] != selected["source_episode"] or selected["length"] != 99 or
            _metaworld_trajectory_group(row["task"], states[:-1], actions) != selected["lineage_group"]):
        raise ValueError("Selected raw row is not the original fitting lineage")
    if not isinstance(row["video"], dict) or not isinstance(row["video"].get("bytes"), bytes):
        raise ValueError("Expected original MP4 bytes, not an auto-decoded dataset")
    return states[:-1], actions


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def rng_digest(state):
    return hashlib.sha256(pickle.dumps((state[0], state[1], state[2].numpy().tobytes()), protocol=4)).hexdigest()


def clip(observations, actions):
    return {"observations": {k: v[:90:5].unsqueeze(0) for k, v in observations.items()},
            "actions": actions[:90].reshape(1, 18, 20)}


def selected_native_parity(row, states, actions, stats, transform, reader_class, decode):
    """Actual native __getitem__ reference, without its full-pool __init__."""
    class OneRow:
        def __getitem__(self, index):
            if index != 0:
                raise IndexError("Only the selected fitting source row is accessible")
            return row
    native = object.__new__(reader_class)
    native.dataset, native.transform, native.with_reward = OneRow(), transform, False
    native.states = states[None]
    native.proprios = ((states[:, :4] - stats["proprio_mean"]) / stats["proprio_std"])[None]
    native.actions = ((actions - stats["action_mean"]) / stats["action_std"])[None]
    native.seq_lengths = torch.tensor([99])
    before = rng_state()
    try:
        native_observations, native_actions, _, _, _ = native[0]
        expected, native_rng = clip(native_observations, native_actions), rng_digest(rng_state())
        restore_rng(before)
        frames = decode(row["video"]["bytes"])
        if frames.shape[0] != 100 or frames.shape[-1] != 3:
            raise ValueError("Unexpected original video alignment")
        pixels = torch.tensor(frames[:99], dtype=torch.float32) / 255.0
        visual = transform(pixels.permute(0, 3, 1, 2))
        proprio = (states[:, :4] - stats["proprio_mean"]) / stats["proprio_std"]
        normalized_actions = (actions - stats["action_mean"]) / stats["action_std"]
        actual = clip({"visual": visual, "proprio": proprio}, normalized_actions)
        for key in ("visual", "proprio"):
            assert_bytes(actual["observations"][key], expected["observations"][key], "selected native " + key)
        assert_bytes(actual["actions"], expected["actions"], "selected native actions")
        if native_rng != rng_digest(rng_state()):
            raise ValueError("Selected native transform RNG differs")
        return actual, {"selected_native_reader_bytes_equal": True,
            "selected_native_reader_rng_equal": True, "full_dataset_constructor_parity": False,
            "published_vs_recomputed_statistics_bitwise_parity_claimed": False,
            "full_video_decodes": 2, "distinct_decoded_source_rows": 1}
    finally:
        restore_rng(before)


def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("This preparation requires CUDA hidden and uninitialized")
    torch.set_num_threads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    try:
        with deadline(600):
            if sha256(args.cohort) != args.cohort_sha256 or sha256(args.raw_proof) != RAW_PROOF_SHA256:
                raise ValueError("Original cohort/raw proof changed")
            cohort = json.loads(args.cohort.read_text()); validate_cohort(cohort)
            selected = next(r for r in cohort["fit"] if r["length"] >= 90)
            if cohort["task"] != "mw-reach" or selected["source_pool"] != "all" or selected["split"] != "fit":
                raise ValueError("Require the original first eligible Reach fitting row")
            vendor = use_vendor(args.vendor)
            if sha256(vendor / READER_SOURCE) != READER_SHA256:
                raise ValueError("Native selected reader source changed")
            stats = published_stats(vendor / STATS_SOURCE)
            manifest = bound_manifest("metaworld", args.manifest)
            import pyarrow.parquet as pq
            counts = {}
            for name, metadata in sorted(manifest.items()):
                path = safe_member(args.data_root, name)
                if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
                    raise ValueError("Raw parquet changed: " + name)
                counts[name] = pq.ParquetFile(path).metadata.num_rows
            if sum(counts.values()) != 12600:
                raise ValueError("Original global population changed")
            source = locate_row(selected["index"], counts)
            exclusive_json(output / "source_mapping.json", {
                "raw_manifest_sha256": MANIFEST_HASHES["metaworld"], "parquet_rows": counts,
                "selected_fit_row": selected, "selected_source": source,
                "other_rows_metadata_and_hashes_only": True})
            row = read_selected_row(safe_member(args.data_root, source["path"]), source["local_row"])
            states, actions = validate_raw_row(row, selected)
            from app.plan_common.datasets.metaworld_hf_dset import MetaworldHFDataset, decode_video_frames
            from app.plan_common.datasets.transforms import make_transforms
            config = cohort["reference_config"]
            transform = make_transforms(img_size=config["data"]["img_size"], **config["data_aug"])
            random.seed(234); np.random.seed(234); torch.manual_seed(234)
            value, parity = selected_native_parity(row, states, actions, stats, transform,
                MetaworldHFDataset, decode_video_frames)
            for name, payload in (("normalization.pt", stats), ("stimulus.pt", value)):
                with (output / name).open("xb") as stream:
                    torch.save(payload, stream)
            for name, source_path in (("official_manifest.json", args.manifest), ("raw_input_report.json", args.raw_proof)):
                with source_path.open("rb") as source_stream, (output / name).open("xb") as target:
                    shutil.copyfileobj(source_stream, target)
            exclusive_json(output / "normalization_source.json", {
                "normalization_sha256": sha256(output / "normalization.pt"),
                "published_stats_source": STATS_SOURCE, "published_stats_source_sha256": STATS_SOURCE_SHA256,
                "raw_manifest_sha256": MANIFEST_HASHES["metaworld"],
                "native_normalization_reused": True, "normalization_recomputed": False,
                "normalization_scope": NORMALIZATION_SCOPE, "previous_fit_normalization_parity_claimed": False})
            members = ("stimulus.pt", "normalization.pt", "normalization_source.json",
                "official_manifest.json", "raw_input_report.json", "source_mapping.json")
            receipt = {"schema_version": 1, "status": "frozen_normalized_fit_stimulus", "task": "mw-reach",
                "checkpoint_sha256": args.checkpoint_sha256, "cohort_sha256": args.cohort_sha256,
                "selected_fit_row": selected, "normalized_fit_payload_only": True,
                "new_full_pool_state_scan": False, "development_or_protected_outcomes_accessed": False,
                "normalization_scope": NORMALIZATION_SCOPE, "clip_start": 0, "frames": 18, "frameskip": 5,
                "files": {name: sha256(output / name) for name in members},
                "preparation_source_sha256": sha256(Path(__file__)), "native_reader_sha256": READER_SHA256,
                "image_transform": copy.deepcopy(config["data_aug"]), "parity": parity,
                "gpu_calls": 0, "model_constructed": False, "behavioral_launch_ready": False}
            exclusive_json(output / "STIMULUS.json", receipt)
            verify_stimulus(output, sha256(output / "STIMULUS.json"), args.data_root,
                            args.cohort_sha256, selected, args.checkpoint_sha256)
            load_stimulus(output)
            if torch.cuda.is_initialized() or sha256(args.cohort) != args.cohort_sha256:
                raise ValueError("CUDA initialization or cohort mutation during CPU preparation")
            exclusive_json(output / "DONE.json", {"stimulus_sha256": sha256(output / "STIMULUS.json"),
                "status": "selected_fit_stimulus_ready_not_gpu_or_behavioral_validation"})
            return receipt
    except BaseException as error:
        exclusive_json(output / "FAILED.json", {"error": repr(error), "behavioral_launch_ready": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "cohort", "raw-proof", "manifest", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("cohort-sha256", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == "__main__":
    main()
