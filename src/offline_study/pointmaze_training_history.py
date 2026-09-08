"""Full PointMaze histories, preserving its native data/update/validation schedule.

Shares the verified objective and model construction with Wall, but never Wall's
418-update schedule, image layout, validation cursor or checkpoint bindings.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import BatchSampler, DistributedSampler, default_collate
import yaml

from .behavioral_development import verified_report
from .droid_native import assert_same
from .navigation_input_check import CONFIGS, SelectedFrameSlicer
from .pointmaze_training_inputs import verify_inputs
from .protocol import sha256, write_json
from .training_history import checkpoint_payload, validation_step, verify_validation_reader
from .training_pilot import (VirtualRankBatchSampler, build_model, accumulated_update,
                             verify_accumulation)
from .vendor import use_vendor

SEEDS = (234, 235, 236)
UPDATES = 1139
VALIDATIONS_PER_EPOCH = 5
CONFIG_SHA = "c666b4251f72f56c08c69ab223cfe645625f39705f1c84285320edc98c205a66"


def training_config(vendor):
    path = vendor / CONFIGS["pointmaze"]
    if sha256(path) != CONFIG_SHA:
        raise ValueError("Pinned PointMaze training configuration changed")
    return yaml.safe_load(path.read_text())


def validation_batches(length=12200):
    """Native per-rank shuffle/padding, including the final partial batch.

    Padding is replicated only for native training monitoring; it does not
    increase independent n or change the separate disjoint behavioral schedule.
    Native validation samplers keep epoch=0 as the upstream training loop does.
    """
    streams = [list(BatchSampler(DistributedSampler(range(length), num_replicas=16,
        rank=rank, shuffle=True), batch_size=4, drop_last=False)) for rank in range(16)]
    if len({len(s) for s in streams}) != 1:
        raise ValueError("Native validation streams have different lengths")
    return [[stream[step] for stream in streams] for step in range(len(streams[0]))]


class ValidationFrames:
    def __init__(self, original):
        from app.plan_common.datasets.traj_dset import TrajSlicerDataset, TrajSubset
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        if (type(original) is not TrajSlicerDataset or original.num_frames != 8 or
                original.frameskip != 5 or original.action_skip != 1 or original.process_actions != "concat"):
            raise ValueError("Wrong native PointMaze validation slicer")
        base = original.dataset
        while type(base) is TrajSubset:
            base = base.dataset
        if type(base) is not PointMazeDataset:
            raise ValueError("PointMaze validation requires its own image layout")
        self.original, self.base = original, base

    def __getitem__(self, index):
        from app.plan_common.datasets.traj_dset import TrajSubset
        row, start, end = self.original.slices[index]
        subset = self.original.dataset
        while type(subset) is TrajSubset:
            row = int(subset.indices[row])
            subset = subset.dataset
        if subset is not self.base:
            raise ValueError("Validation row mapping changed")
        frames = list(range(start, end, 5))
        image = torch.load(self.base.data_path / f"obses/episode_{row:03d}.pth",
                           map_location="cpu", weights_only=True, mmap=True)[frames] / 255.
        obs = {"visual": self.base.transform(image.permute(0, 3, 1, 2)),
               "proprio": self.base.proprios[row, frames]}
        return obs, self.base.actions[row, start:end].reshape(8, -1), self.base.states[row, frames], torch.zeros(8)


def restore_checkpoint(data, model, cfg, scheduler, wd, loader_rng, binding):
    state = data["study_resume"]
    epoch = data["epoch"]
    if (state["binding"] != binding or not state["epoch_boundary_only"] or state["epoch"] != epoch or
            not isinstance(epoch, int) or not 1 <= epoch < 50 or
            state["scheduler_step"] != UPDATES * epoch or state["wd_step"] != UPDATES * epoch or
            state["validation_events"] != VALIDATIONS_PER_EPOCH * epoch or
            len(state["cpu_rngs"]) != 16 or len(state["cuda_rngs"]) != 16):
        raise ValueError("Changed PointMaze schedule, seed or non-epoch resume")
    model.predictor.load_state_dict(data["predictor"], strict=True)
    for name in ("action_encoder", "proprio_encoder"):
        module = getattr(model, name)
        external = module is not None and not cfg["model"][name].get(name + "_inpred", False)
        if external != (name in data):
            raise ValueError("Native checkpoint component coverage changed")
        if external:
            module.load_state_dict(data[name], strict=True)
    model.optimizer.load_state_dict(data["opt"])
    model.scaler.load_state_dict(data["scaler"])
    scheduler._step, wd._step = state["scheduler_step"], state["wd_step"]
    loader_rng.set_state(state["loader_rng"])
    return [r.clone() for r in state["cpu_rngs"]], [r.clone() for r in state["cuda_rngs"]], state["validation_events"]


@torch.no_grad()
def monitor(run, validation, batches, event, cpu_rngs, cuda_rngs):
    batch_ids, values = batches[event % len(batches)], []
    for rank, indices in enumerate(batch_ids):
        obs, action, state, reward = default_collate([validation[index] for index in indices])
        torch.set_rng_state(cpu_rngs[rank])
        torch.cuda.set_rng_state(cuda_rngs[rank], 0)
        obs = {key: value.to("cuda:0", dtype=torch.bfloat16) for key, value in obs.items()}
        result = run(obs, action.to("cuda:0", dtype=torch.bfloat16),
            state.to("cuda:0", dtype=torch.bfloat16), reward.to("cuda:0", dtype=torch.bfloat16), train=False)
        metrics = {key: float(value) for key, value in result[3].items() if key.startswith("data_traj/")}
        if not metrics or not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Native PointMaze validation failed")
        cpu_rngs[rank], cuda_rngs[rank] = torch.get_rng_state(), torch.cuda.get_rng_state(0)
        values.append({"logical_rank": rank, "validation_clip_indices": indices, "metrics": metrics})
    return {"event": event, "logical_batches": values, "global_clips": sum(map(len, batch_ids)),
            "source_step_train_false": True, "noisy_and_recorded_action_rollouts_retained": True,
            "native_partial_validation_batch_preserved": True}


def verify_saved_resume(path, model, cfg, scheduler, wd, loader_rng, binding, next_batch):
    from app.vjepa_wm.utils import load_checkpoint_state_dict
    data = torch.load(path, map_location="cpu", weights_only=True)
    reference, ref_schedule, ref_wd = copy.deepcopy((model, scheduler, wd))
    resumed, resumed_schedule, resumed_wd = copy.deepcopy((model, scheduler, wd))
    with torch.no_grad():
        for parameter in resumed.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    load_checkpoint_state_dict(data, predictor=resumed.predictor, action_encoder=resumed.action_encoder,
        proprio_encoder=resumed.proprio_encoder, opt=resumed.optimizer, scaler=resumed.scaler)
    assert_same(reference.state_dict(), resumed.state_dict())
    assert_same(reference.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_same(reference.scaler.state_dict(), resumed.scaler.state_dict())
    resumed_rng = torch.Generator()
    cpu, cuda, events = restore_checkpoint(data, resumed, cfg, resumed_schedule, resumed_wd, resumed_rng, binding)
    ref_cpu = [r.clone() for r in data["study_resume"]["cpu_rngs"]]
    cuda_before = torch.cuda.get_rng_state(0)
    expected = accumulated_update(reference, ref_schedule, ref_wd, next_batch, ref_cpu)
    torch.cuda.set_rng_state(cuda_before, 0)
    actual = accumulated_update(resumed, resumed_schedule, resumed_wd, next_batch, cpu)
    for expected_state, actual_state in ((reference.state_dict(), resumed.state_dict()),
            (reference.optimizer.state_dict(), resumed.optimizer.state_dict()),
            (reference.scaler.state_dict(), resumed.scaler.state_dict()), (ref_cpu, cpu),
            (expected, actual), (data["study_resume"]["cuda_rngs"], cuda)):
        assert_same(expected_state, actual_state)
    if events != 5 or not torch.equal(resumed_rng.get_state(), data["study_resume"]["loader_rng"]):
        raise ValueError("Lost PointMaze validation or loader cursor")
    torch.cuda.set_rng_state(cuda_before, 0)
    return {"status": "actual_native_checkpoint_loader_and_resumed_update_exact",
        "checkpoint_sha256": sha256(path), "actual_upstream_loader_used": True,
        "next_update_parameters_optimizer_scaler_cpu_rng_exact": True,
        "validation_cursor_cuda_rng_loader_state_restored": True,
        "extra_updates_were_discarded_engineering_clones_only": True}


def source_bindings(vendor):
    names = ("pointmaze_training_history.py", "pointmaze_training_inputs.py", "training_history.py",
             "training_pilot.py", "navigation_input_check.py")
    return {**{name: sha256(Path(__file__).with_name(name)) for name in names},
            "native_training.py": sha256(vendor / "app/vjepa_wm/train.py"),
            "native_dataset.py": sha256(vendor / "app/plan_common/datasets/point_maze_dset.py"),
            "native_dataset_utils.py": sha256(vendor / "app/plan_common/datasets/utils.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "data-root", "input-receipt", "input-check", "pilot", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--engineering-proof", type=Path)
    parser.add_argument("--engineering-only", action="store_true")
    args = parser.parse_args()
    if args.engineering_only and (args.resume_from or args.engineering_proof):
        raise ValueError("Engineering must start from initialization")
    if not args.engineering_only and (not args.resume_from or not args.engineering_proof):
        raise ValueError("Full history requires one complete verified engineering epoch")
    use_vendor(args.vendor)
    cfg = training_config(args.vendor)
    inputs, input_hash = verify_inputs(args.data_root, args.input_receipt)
    pilot, pilot_hash = verified_report(args.pilot)
    pp = json.loads((args.pilot / "protocol.json").read_text())
    audit, audit_hash = verified_report(args.input_check)
    ip = json.loads((args.input_receipt / "protocol.json").read_text())
    if (pilot["status"] != "native_training_accumulation_pilot_passed" or pilot["task"] != "pointmaze" or
            pilot["updates_per_epoch"] != UPDATES or pilot["training_clips"] != 145800 or
            pilot["protocol_sha256"] != sha256(args.pilot / "protocol.json") or
            pilot["parity_sha256"] != sha256(args.pilot / "PARITY.json") or
            pp["source_sha256"] != sha256(Path(__file__).with_name("training_pilot.py")) or
            pp["native_training_source_sha256"] != sha256(args.vendor / "app/vjepa_wm/train.py") or
            pp["native_config_sha256"] != CONFIG_SHA or pp["input_check_report_sha256"] != audit_hash or
            inputs["native_config_sha256"] != CONFIG_SHA or ip["input_check_report_sha256"] != audit_hash or
            audit["task_reports_sha256"]["pointmaze"] != sha256(args.input_check / "pointmaze.json")):
        raise ValueError("Missing PointMaze-specific input/objective/config proof")
    binding = {"task": "pointmaze", "seed": args.seed, "input_report_sha256": input_hash,
        "native_config_sha256": CONFIG_SHA, "source_sha256": source_bindings(args.vendor),
        "pilot_report_sha256": pilot_hash, "input_check_report_sha256": audit_hash}
    if not args.engineering_only:
        proof, _ = verified_report(args.engineering_proof)
        ep = json.loads((args.engineering_proof / "protocol.json").read_text())
        if (proof["status"] != "one_epoch_pointmaze_training_engineering_complete" or proof["seed"] != args.seed or
                proof["protocol_sha256"] != sha256(args.engineering_proof / "protocol.json") or
                proof["resume_parity_sha256"] != sha256(args.engineering_proof / "RESUME_PARITY.json") or
                any(ep[key] != value for key, value in binding.items())):
            raise ValueError("No matching complete seed-specific engineering proof")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {**binding, "role": "training_engineering_full_epoch" if args.engineering_only
        else "complete_pointmaze_training_history", "epochs": 50, "optimizer_updates_per_epoch": UPDATES,
        "global_training_batch": 128, "logical_training_batches": "16x8 native DistributedSampler/drop_last",
        "training_rows": 1800, "validation_rows": 200, "training_clips": 145800, "validation_clips": 12200,
        "validation_every_within_epoch_updates": 200, "validation_events_per_epoch": 5,
        "native_validation_batches_per_cycle": 191, "validation_batch_sizes": "190x64 then48; native sampler padding retained",
        "source_validation_noisy_and_recorded_action_H6_both": True,
        "checkpoint_policy": "all50 immutable epochs; primary41..50; no outcome selection",
        "planning_episodes_per_checkpoint_condition": 96, "planning_evaluation_is_separate_required_job": True,
        "training_seed_triplet": list(SEEDS), "authors_exact_seed_triplet_or_collective_order_claimed": False,
        "resume_checkpoint": str(args.resume_from) if args.resume_from else None, "fresh_confirmation": False})
    started = time.monotonic()
    try:
        from app.plan_common.datasets.transforms import make_transforms
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.traj_dset import get_train_val_sliced
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        dataset = PointMazeDataset(data_path=str(args.data_root), n_rollout=None, normalize_action=True,
            transform=make_transforms(img_size=224, **cfg["data_aug"]))
        train, val, train_clips, val_clips = get_train_val_sliced(dataset, train_fraction=.9, random_seed=234,
            num_frames=4, num_frames_val=8, frameskip=5, action_skip=1)
        audited = json.loads((args.input_check / "pointmaze.json").read_text())
        if ((len(dataset), len(train), len(val), len(train_clips), len(val_clips)) != (2000,1800,200,145800,12200) or
                list(train.indices) != audited["native_train_indices"] or list(val.indices) != audited["native_validation_indices"]):
            raise ValueError("PointMaze split/clip population changed")
        sampler, batches = VirtualRankBatchSampler(len(train_clips)), validation_batches(len(val_clips))
        if len(sampler) != UPDATES or len(batches) != 191 or sum(map(len, batches[-1])) != 48:
            raise ValueError("Native PointMaze sampler schedule changed")
        validation = ValidationFrames(val_clips)
        reader_proof = verify_validation_reader(val_clips, validation)
        loader_rng = torch.Generator().manual_seed(2026090725)
        loader = torch.utils.data.DataLoader(SelectedFrameSlicer(train_clips), batch_sampler=sampler,
            num_workers=4, pin_memory=True, persistent_workers=True, generator=loader_rng)
        model, scheduler, wd = build_model(cfg, dataset, len(sampler))
        encoder_versions = [(name, p._version) for name, p in model.encoder.named_parameters()]
        cpu = [torch.get_rng_state().clone() for _ in range(16)]
        cuda = [torch.cuda.get_rng_state(0).clone() for _ in range(16)]
        native_monitor = validation_step(args.vendor, model, cfg, scheduler, wd)
        start_epoch, events, history = 0, 0, []
        if args.resume_from:
            data = torch.load(args.resume_from, map_location="cpu", weights_only=True)
            cpu, cuda, events = restore_checkpoint(data, model, cfg, scheduler, wd, loader_rng, binding)
            start_epoch = data["epoch"]
            history = list(data["study_resume"].get("checkpoint_history", []))
            history.append({"epoch": start_epoch, "path": str(args.resume_from), "sha256": sha256(args.resume_from)})
            if [r["epoch"] for r in history] != list(range(1, start_epoch+1)):
                raise ValueError("Missing inherited PointMaze checkpoint history")
            for row in history:
                if sha256(Path(row["path"])) != row["sha256"]:
                    raise ValueError("Inherited PointMaze checkpoint changed")
            del data
        for epoch in range(start_epoch, 1 if args.engineering_only else 50):
            sampler.epoch = epoch
            epoch_started, losses = time.monotonic(), []
            for step, batch in enumerate(loader):
                before = time.monotonic()
                if epoch == 0 and step == 0:
                    parity = verify_accumulation(args.vendor, model, scheduler, wd, batch, cpu)
                    write_json(args.output / "PARITY.json", {"training": parity, "validation_reader": reader_proof})
                    values = {"loss": parity["loss"]}
                else:
                    values = accumulated_update(model, scheduler, wd, batch, cpu)
                losses.append(values["loss"])
                if (step+1) % 200 == 0:
                    measured = monitor(native_monitor, validation, batches, events, cpu, cuda)
                    write_json(args.output / f"validation-{events:03d}.json", {"epoch": epoch+1, "update": step+1, **measured})
                    events += 1
                if (step+1) % 20 == 0:
                    progress = {"epoch": epoch+1, "epoch_updates": step+1, "total_epochs": 50,
                        "global_updates": epoch*UPDATES+step+1, "last_update_seconds": time.monotonic()-before,
                        "loss": values["loss"], "seconds": time.monotonic()-started}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            if (len(losses) != UPDATES or events != (epoch+1)*VALIDATIONS_PER_EPOCH or
                    encoder_versions != [(name,p._version) for name,p in model.encoder.named_parameters()]):
                raise ValueError("Incomplete PointMaze epoch or frozen encoder changed")
            checkpoint = args.output / f"jepa-e{epoch}.pth.tar"
            payload = checkpoint_payload(model, cfg, epoch+1, cpu, cuda, scheduler, wd, loader_rng, binding, events)
            payload["study_resume"]["checkpoint_history"] = list(history)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary); temporary.replace(checkpoint)
            history.append({"epoch": epoch+1, "path": str(checkpoint), "sha256": sha256(checkpoint)})
            write_json(args.output / f"epoch-{epoch+1:03d}.json", {"epoch": epoch+1, "updates": UPDATES,
                "mean_training_loss": float(np.mean(losses)), "seconds": time.monotonic()-epoch_started,
                "checkpoint": history[-1], "validation_events_so_far": events})
            write_json(args.output / "CHECKPOINTS.json", {"history": history, "required_epochs": list(range(1,51))})
            print(json.dumps({"epoch_complete": epoch+1, "checkpoint_sha256": history[-1]["sha256"]}), flush=True)
        resume_hash = None
        if args.engineering_only:
            sampler.epoch = 1
            proof = verify_saved_resume(checkpoint, model, cfg, scheduler, wd, loader_rng, binding, next(iter(loader)))
            write_json(args.output / "RESUME_PARITY.json", proof)
            resume_hash = sha256(args.output / "RESUME_PARITY.json")
        write_json(args.output / "report.json", {"status": "one_epoch_pointmaze_training_engineering_complete" if args.engineering_only
            else "all50_pointmaze_training_epochs_complete", "protocol_sha256": sha256(args.output / "protocol.json"),
            "seed": args.seed, "checkpoint_manifest_sha256": sha256(args.output / "CHECKPOINTS.json"),
            "checkpoints": len(history), "validation_events": events, "seconds": time.monotonic()-started,
            "resume_parity_sha256": resume_hash, "training_history_complete": not args.engineering_only,
            "behavioral_evaluations_complete": False, "fresh_confirmation": False, "all_study_tasks_complete": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "training_history_complete": False})
        raise


if __name__ == "__main__":
    main()
