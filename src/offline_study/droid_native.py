"""Native DROID stimulus audit and full-CEM engineering, not robot execution.

Uses the released 15-recording selection for code-replication engineering; audits
all 16 files without promoting the extra recording to fresh confirmation. Model
outcomes cannot change this population or select an intervention.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .dino3_prepare import SOURCE_COMMIT, WEIGHT_SHA256
from .droid_assets import CHECKPOINT_SHA256, verify_download
from .droid_contract import action_metrics, prepare
from .planning_native_smoke import SMOKE_SEED
from .protocol import sha256, write_json
from .vendor import use_vendor


def verified_report(root):
    done = json.loads((root / "DONE.json").read_text())
    if sha256(root / "report.json") != done["report_sha256"]:
        raise ValueError("Input completion/report binding changed")
    report = json.loads((root / "report.json").read_text())
    if sha256(root / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("Input protocol binding changed")
    return report, done["report_sha256"]


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def observation_digest(observation):
    visual, proprio = observation["visual"], observation["proprio"]
    if (visual.shape != (1, 3, 256, 256) or visual.dtype != torch.uint8 or
            proprio.shape != (1, 7) or not torch.isfinite(proprio).all() or
            visual.max() == visual.min()):
        raise ValueError("Invalid native DROID 256px observation or seven-dimensional pose")
    digest = hashlib.sha256()
    for key in ("visual", "proprio"):
        digest.update(key.encode() + array_hash(observation[key].detach().cpu().numpy()).encode())
    return digest.hexdigest()


def audit_recordings(asset_root, manifest, contract):
    import h5py

    rows, unreadable_unselected = [], []
    for entry in manifest["assets"]:
        path = asset_root / "downloads" / entry["repo_type"] / entry["filename"]
        verified = verify_download(path, entry)
        if entry["kind"] != "evaluation_recording":
            continue
        try:
            handle = h5py.File(path, "r")
        except OSError as exc:
            if entry["filename"] in contract["released_config_recordings"]:
                raise ValueError("Unreadable selected recording; cannot substitute or omit: " + entry["filename"]) from exc
            unreadable_unselected.append({"filename": entry["filename"],
                "published_checksum_verified": True, "sha256": verified["verified_content_sha256"],
                "bytes": path.stat().st_size, "error": str(exc),
                "selected_by_released_config": False})
            continue
        with handle:
            obs = handle["episode_data/observation"]
            poses = np.asarray(obs["cartesian_position"])
            gripper = np.asarray(obs["gripper_position"])
            camera = obs["exterior_image_2_left"]
            if (poses.ndim != 2 or poses.shape[1] != 6 or len(poses) <= 40 or
                    gripper.shape != (len(poses),) or camera.ndim != 4 or
                    camera.shape[0] != len(poses) or camera.shape[-1] != 3 or
                    camera.dtype != np.uint8 or not np.isfinite(poses).all() or
                    not np.isfinite(gripper).all()):
                raise ValueError("Malformed/too-short recording, no silent substitution: " + entry["filename"])
            rows.append({"filename": entry["filename"], "sha256": verified["verified_content_sha256"],
                "raw_frames": len(poses), "camera_shape": list(camera.shape),
                "state_sha256": array_hash(np.c_[poses, gripper]),
                "initial_state_sha256": array_hash(np.r_[poses[0], gripper[0]]),
                "first_frame_sha256": array_hash(camera[0]),
                "last_frame_sha256": array_hash(camera[-1]),
                "selected_by_released_config": entry["filename"] in contract["released_config_recordings"]})
    duplicates = {}
    for field in ("sha256", "state_sha256", "initial_state_sha256", "first_frame_sha256"):
        groups = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row["filename"])
        duplicates[field] = [files for files in groups.values() if len(files) > 1]
    if len(rows) + len(unreadable_unselected) != 16:
        raise ValueError("Audit must account for all 16 released recordings")
    return {"recordings": rows, "unreadable_unselected_recordings": unreadable_unselected,
        "exact_duplicate_groups": duplicates,
        "population_for_engineering": "released_config_15; extra file audited but not outcome-evaluated",
        "paper_16_versus_config_15_reason_known": False,
        "absence_of_exact_duplicates_does_not_prove_independent_families": True,
        "model_outcomes_accessed": False}


def strict_dataset_class():
    from app.plan_common.datasets.droid_dset import DROIDVideoDataset

    class StrictDROID(DROIDVideoDataset):
        def __getitem__(self, idx, debug=False, **kwargs):
            # Native debug=True disables its retry-on-another-recording loop.
            return super().__getitem__(idx, debug=True, **kwargs)

        def loadvideo_hf(self, path):
            result = super().loadvideo_hf(path)
            self.last_sample = {"path": str(path), "raw_frame_indices": result[-1].tolist(),
                                "sampled_state_sha256": array_hash(result[2])}
            return result

    return StrictDROID


def make_dataset(asset_root, contract, strict=True):
    from app.plan_common.datasets.droid_dset import DROIDVideoDataset
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms, make_inverse_transforms

    model = contract["config"]["model_kwargs"]
    transform = make_transforms(img_size=256, normalize=model["data_aug"]["normalize"],
        random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.), reprob=0., auto_augment=False, motion_shift=False)
    cls = strict_dataset_class() if strict else DROIDVideoDataset
    dset = cls(data_path=str(asset_root / "downloads/dataset"),
        camera_views=["exterior_image_2_left"], frameskip=1, action_skip=1,
        frames_per_clip=5, fps=4, transform=transform, camera_frame=None,
        normalize_action=False, mpk_dset=True,
        mpk_manifest_patterns=model["data"]["droid"]["mpk_manifest_patterns"],
        droid_to_rcasa_action_format=1, local=True, seed=234)
    actual = [str(Path(name).relative_to(asset_root / "downloads/dataset")) for name in dset.samples]
    if actual != contract["released_config_recordings"]:
        raise ValueError("Native glob order differs from the released recording panel")
    preprocessor = Preprocessor(**{name: getattr(dset, name) for name in
        ("action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std")},
        transform=transform, inverse_transform=make_inverse_transforms(img_size=256, **model["data_aug"]))
    return dset, preprocessor


def assert_same(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Changed mapping keys")
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError("Changed sequence length")
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError("Changed native tensor")
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError("Changed native array or RNG state")
    elif a != b:
        raise ValueError("Changed native value")


def stimulus_parity(asset_root, contract):
    def reset():
        random.seed(SMOKE_SEED)
        np.random.seed(SMOKE_SEED)
        torch.manual_seed(SMOKE_SEED)
    reset()
    native, _ = make_dataset(asset_root, contract, strict=False)
    native_records = [(native[i], copy.deepcopy(native.rng.get_state())) for i in (0, 7, 14)]
    native_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    reset()
    strict, _ = make_dataset(asset_root, contract)
    strict_records = [(strict[i], copy.deepcopy(strict.rng.get_state())) for i in (0, 7, 14)]
    strict_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert_same(native_records, strict_records)
    assert_same(native_rng, strict_rng)
    return {"native_and_strict_pixels_actions_states_exact": True,
        "constructor_and_post_sampling_rng_exact": True, "tested_recording_indices": [0, 7, 14],
        "silent_substitution_disabled": True, "model_outcomes_accessed": False}


@contextlib.contextmanager
def verified_native_encoder(source, encoder_root):
    report, report_hash = verified_report(encoder_root)
    weights = encoder_root / "native_state_from_official_hf.pth"
    if (report["status"] != "native_dinov3_encoder_weight_mapping_and_feature_parity_verified" or
            report["official_files_sha256"]["model.safetensors"] != WEIGHT_SHA256 or
            sha256(weights) != report["native_weights_sha256"] or
            report["native_source_commit"] != SOURCE_COMMIT or
            subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT or
            subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()):
        raise ValueError("Encoder provenance or numerical gate changed")
    original = torch.hub.load
    calls = []

    def load(repo, name, *args, **kwargs):
        if name != "dinov3_vitl16" or kwargs.get("source") != "local" or args:
            raise ValueError("Unexpected hub call in DROID constructor")
        calls.append(name)
        # Native source has an incorrect filename suffix, so bind the verified
        # equivalent serialization explicitly, never download that wrong file.
        encoder = original(str(source), name, source="local", pretrained=False)
        encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        return encoder

    torch.hub.load = load
    try:
        yield {"encoder_report_sha256": report_hash, "native_weights_sha256": report["native_weights_sha256"],
               "original_native_pth_bytes_available": False,
               "verified_native_constructor_and_explicit_strict_weight_load": True}
        if calls != ["dinov3_vitl16"]:
            raise ValueError("DROID constructor did not load exactly one verified encoder")
    finally:
        torch.hub.load = original


def load_model(vendor, asset_root, encoder_source, encoder_root, contract, preprocessor):
    checkpoint = asset_root / "downloads/model/jepa_wm_droid.pth.tar"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("DROID world-model checkpoint changed")
    config = contract["config"]["model_kwargs"]
    # Native DinoEncoder constructs its requested filename before our narrow
    # redirect. These env vars supply only paths, never authentication values.
    os.environ["JEPAWM_HOME"] = str(encoder_source.parent)
    os.environ["JEPAWM_OSSCKPT"] = str(encoder_root)
    with verified_native_encoder(encoder_source, encoder_root) as provenance:
        model = importlib.import_module(config["module_name"]).init_module(
            folder=str(checkpoint.parent), checkpoint=checkpoint.name,
            model_kwargs=copy.deepcopy(config["pretrain_kwargs"]),
            wrapper_kwargs=config["wrapper_kwargs"], cfgs_data=config["data"],
            device=torch.device("cuda:0"), action_dim=7, proprio_dim=7,
            preprocessor=preprocessor)
    model.eval().requires_grad_(False)
    # Upstream permits missing checkpoint keys. Independently require exact
    # predictor tensor coverage/identity before trusting the loaded model.
    from app.vjepa_wm.utils import clean_state_dict
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = clean_state_dict(state["predictor"])
    actual = model.model.predictor.state_dict()
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k].cpu(), expected[k]) for k in actual):
        raise ValueError("Released predictor did not load completely and exactly")
    if model.ctxt_window != 2 or model.action_dim != 7 or model.model.use_proprio:
        raise ValueError("Wrong DROID wrapper architecture")
    provenance.update(checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_epoch=state["epoch"],
                      predictor_tensors_exact=True, precision="float32_no_tf32")
    return model, provenance


@torch.no_grad()
def full_episode(cfg, model, dset, preprocessor, output, repetition):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.utils import prepare_obs

    calls = []
    original = model.unroll
    def observed(context, act_suffix=None, **kwargs):
        result = original(context, act_suffix=act_suffix, **kwargs)
        # No-proprio DROID returns a visual Tensor, unlike the other tasks'
        # visual/proprio TensorDict. Preserve the original type for the objective.
        if not isinstance(result, torch.Tensor) or not torch.isfinite(result).all():
            raise ValueError("Nonfinite DROID forecast")
        calls.append(list(act_suffix.shape[:2]))
        write_json(output / "progress.json", {"repetition": repetition, "unroll_calls": len(calls),
            "expected_calls": 30, "role": "engineering_not_efficacy"})
        return result
    model.unroll = observed
    env = None
    try:
        agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
        env = make_env(cfg)
        evaluator = PlanEvaluator(cfg, agent)
        _, info = env.reset(seed=SMOKE_SEED, task_idx=0)
        env.proprio_env.unwrapped._freeze_rand_vec = False
        env.proprio_env.unwrapped.seeded_rand_vec = True
        env.seed(SMOKE_SEED)
        initial, goal, _, _ = evaluator.set_episode(cfg, agent, env, SMOKE_SEED, task_idx=0)
        setup = {"initial_sha256": observation_digest(initial), "goal_sha256": observation_digest(goal),
                 "dataset_sample": dset.last_sample}
        agent.set_goal(prepare_obs(cfg.task_specification.obs, goal))
        def actor(obs, steps_left):
            action = agent.act(prepare_obs(cfg.task_specification.obs, obs), steps_left=steps_left)
            if len(agent._prev_losses) != 15 or not torch.isfinite(agent._prev_losses).all():
                raise ValueError("Changed or nonfinite DROID CEM iterations")
            return action
        _, _, planned, _, _, _ = evaluator.unroll_agent(env, initial, info, actor, preprocessor=preprocessor)
        if len(planned) != 1 or planned[0].shape != (3, 7) or calls != [[3, 300], [3, 1]] * 15:
            raise ValueError("Changed full native DROID CEM schedule")
        metrics = action_metrics(planned[0], evaluator.expert_actions[0])
        return {**setup, "planned_actions": planned[0].tolist(),
            "recorded_actions": evaluator.expert_actions[0].tolist(),
            "metrics": {k: float(v) for k, v in metrics.items()}, "unroll_calls": calls,
            "dummy_success_intentionally_omitted": True, "robot_executions": 0}
    finally:
        model.unroll = original
        if env is not None:
            from evals.simu_env_planning.envs.droid_dset_dummy_env import DummyEnv
            # The author's placeholder owns no simulator resources and defines
            # no close(). Do not let gym's forwarded close mask an earlier error.
            if type(env.unwrapped) is not DummyEnv or hasattr(env.unwrapped, "close"):
                env.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "assets", "manifest", "encoder-source", "encoder-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        assets, assets_hash = verified_report(args.assets)
        if assets["status"] != "official_droid_inputs_staged_and_verified":
            raise ValueError("Wrong DROID input stage")
        manifest = json.loads(args.manifest.read_text())
        contract = prepare(args.vendor, manifest)
        write_json(args.output / "protocol.json", {"role": "native_droid_engineering_not_confirmation",
            "contract": contract, "assets_report_sha256": assets_hash,
            "source_sha256": sha256(Path(__file__)), "repetitions": 2,
            "smoke_seed": SMOKE_SEED, "precision": "float32_no_tf32",
            "population": "released_config_15; no outcomes from extra 16th recording",
            "scientific_efficacy_measurement": False, "physical_robot_execution": False})
        write_json(args.output / "metadata_audit.json", audit_recordings(args.assets, manifest, contract))
        write_json(args.output / "stimulus_parity.json", stimulus_parity(args.assets, contract))
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        _, preprocessor = make_dataset(args.assets, contract)
        model, provenance = load_model(args.vendor, args.assets, args.encoder_source,
                                      args.encoder_root, contract, preprocessor)
        versions = [(name, p._version) for name, p in model.named_parameters()]
        from omegaconf import OmegaConf
        torch.cuda.reset_peak_memory_stats()
        records = []
        for repetition in range(2):
            random.seed(SMOKE_SEED)
            np.random.seed(SMOKE_SEED)
            torch.manual_seed(SMOKE_SEED)
            dset, preprocessor = make_dataset(args.assets, contract)
            cfg = OmegaConf.create(copy.deepcopy(contract["config"]))
            cfg.local_seed = cfg.meta.seed = SMOKE_SEED
            before = time.monotonic()
            result = full_episode(cfg, model, dset, preprocessor, args.output, repetition)
            records.append(result)
            write_json(args.output / f"repetition-{repetition}.json", {"result": result,
                "seconds": time.monotonic() - before})
        assert_same(records[0], records[1])
        if versions != [(name, p._version) for name, p in model.named_parameters()]:
            raise ValueError("Frozen DROID parameters changed")
        files = ("metadata_audit.json", "stimulus_parity.json", "repetition-0.json", "repetition-1.json")
        write_json(args.output / "report.json", {"status": "native_droid_full_cem_engineering_passed",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "files_sha256": {name: sha256(args.output / name) for name in files},
            "same_seed_actions_and_metrics_exact": True, "model_provenance": provenance,
            "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "scientific_efficacy_measurement": False, "fresh_confirmation": False,
            "robot_executions": 0, "dummy_success_reported": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


if __name__ == "__main__":
    main()
