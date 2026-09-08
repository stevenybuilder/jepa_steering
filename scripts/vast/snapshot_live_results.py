"""Archive immutable completed records from explicitly owned live workers.

Does not pause jobs, copy partial checkpoints, or infer whole-study completion.
New records appearing after the manifest are left for the next snapshot.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

from backup_results_to_google import verify_archive


WORKERS = {50189244: "wall", 50231985: "navigation", 50233992: "metaworld", 50239185: "pusht",
           50259194: "droid_parallel"}


def select_files(root, kind):
    import json
    from pathlib import Path
    root = Path(root)
    chosen = set()

    def file(path):
        if not path.is_file() or path.is_symlink() or path.name.startswith("."):
            raise ValueError("Unsafe or missing snapshot member: " + str(path))
        if path.suffix in (".key", ".pem") or path.name in ("rclone.conf",):
            raise ValueError("Credential-like member")
        chosen.add(str(path.relative_to(root)))

    def tree(path):
        if not path.is_dir():
            raise ValueError("Missing explicit snapshot root: " + str(path))
        for p in path.rglob("*"):
            if "__pycache__" not in p.parts and not p.name.startswith("._") and p.is_file():
                file(p)

    def completed(path):
        if not (path / "DONE.json").is_file():
            raise ValueError("Required engineering root is incomplete: " + str(path))
        tree(path)

    if kind == "droid_parallel":
        base = root / "droid-coupling-behavior-20260908-v3"
        tree(root / "droid-coupling-code-20260908-v3")
        tree(base / "freeze")
        file(base / "LAUNCH.json")
        for gpu in range(4):
            completed(base / f"engineering-gpu{gpu}")
            file(base / f"queue-gpu{gpu}/LAUNCH.json")
            if (base / f"queue-gpu{gpu}/DONE.json").is_file():
                file(base / f"queue-gpu{gpu}/DONE.json")
        for shard in sorted((base / "conditions").glob("*/shard-*")):
            if (shard / "DONE.json").is_file():
                completed(shard)
            else:
                # Preserve only published episode/trace pairs, never mutable
                # progress or a partially serialized active episode.
                for episode in sorted(shard.glob("episode-*.json")):
                    trace = episode.with_name(episode.name.replace("episode-", "trace-"))
                    if not trace.is_file():
                        continue
                    try:
                        json.loads(episode.read_text()); json.loads(trace.read_text())
                    except json.JSONDecodeError:
                        continue
                    file(episode); file(trace); file(shard / "protocol.json")
        if (base / "PANEL_DONE.json").is_file():
            completed(base / "analysis")
            file(base / "PANEL_DONE.json")
    elif kind == "wall":
        tree(root / "wall-history-code-v1")
        for label in ("navigation-input-check-20260907-v1", "wall-training-inputs-20260907-v1",
                      "wall-training-accumulation-pilot-20260907-v2"):
            completed(root / label)
        base = root / "wall-training-history-20260907-v1/seed-234"
        completed(base / "epoch-one-engineering")
        output = base / "remaining-epochs"
        file(output / "protocol.json")
        epochs = sorted(output.glob("epoch-*.json"))
        if not epochs:
            raise ValueError("No completed training epoch")
        maximum = 0
        for p in epochs:
            record = json.loads(p.read_text())
            maximum = max(maximum, record["epoch"])
            file(p)
            checkpoint = Path(record["checkpoint"]["path"])
            if not checkpoint.is_relative_to(output) or checkpoint.suffix != ".tar":
                raise ValueError("Checkpoint outside the explicit training output")
            file(checkpoint)
        for event in range(2, maximum * 2):
            file(output / f"validation-{event:03d}.json")
        if (output / "DONE.json").is_file():
            for name in ("CHECKPOINTS.json", "report.json", "DONE.json"):
                file(output / name)
    elif kind == "droid_behavior":
        base = root / "droid-coupling-behavior-20260908-v2"
        tree(root / "droid-coupling-code-20260908-v2")
        completed(base / "engineering")
        tree(base / "freeze")
        for name in ("QUEUE_LAUNCH.json", "TRAINING_PAUSE.json"):
            file(base / name)
        handoff = root / "droid-device-handoff-20260908-v1"
        if not (handoff / "READY.json").is_file():
            raise ValueError("DROID shard-boundary handoff not verified")
        tree(handoff)
        file(root / "pause_droid_at_shard_v1.py")
        shards = sorted((base / "conditions").glob("*/shard-*"))
        if not shards:
            raise ValueError("No DROID scientific shards to preserve")
        for shard in shards:
            completed(shard)
    elif kind == "droid_coupling_preparation":
        tree(root / "droid-coupling-code-20260908-v2")
        completed(root / "droid-coupling-fit-20260908-v2")
        completed(root / "droid-native-replication-20260907-v1/shard-all")
        completed(root / "droid-native-engineering-20260907-v7")
        base = root / "droid-coupling-behavior-20260908-v2"
        tree(base / "freeze")
        for name in ("QUEUE_LAUNCH.json", "TRAINING_PAUSE.json"):
            file(base / name)
        if (base / "engineering/DONE.json").is_file():
            completed(base / "engineering")
        # Only completed checkpoints, never an active partial epoch.
        tree(root / "pointmaze-history-code-20260908-v1")
        history = root / "pointmaze-training-history-20260908-v1/seed-234"
        completed(history / "epoch-one-engineering")
        for epoch in (2, 3):
            marker = history / "remaining-epochs" / f"epoch-{epoch:03d}.json"
            row = json.loads(marker.read_text())
            checkpoint = Path(row["checkpoint"]["path"])
            if row["epoch"] != epoch or not checkpoint.is_relative_to(history / "remaining-epochs"):
                raise ValueError("Wrong preserved PointMaze checkpoint")
            file(marker); file(checkpoint)
        file(history / "remaining-epochs/protocol.json")
        for event in range(5, 15):
            file(history / "remaining-epochs" / f"validation-{event:03d}.json")
    elif kind == "droid_fit_preparation":
        for name in ("droid-fit-input-code-20260908-v1", "droid-fit-input-evidence-20260908-v1",
                     "droid-fit-audit-code-20260908-v1", "droid-fit-audit-code-20260908-v2",
                     "droid-fit-audit-20260908-v1"):
            tree(root / name)
        completed(root / "droid-fit-audit-20260908-v2")
        eligible = root / "droid-fit-native-eligible-20260908-v1"
        completed(eligible)
        original = root / "droid-fit-download-20260908-v1"
        if not (original / "DONE.json").is_file():
            raise ValueError("Original DROID input stage is incomplete")
        current = json.loads((eligible / "FILES.json").read_text())
        prior = json.loads((original / "FILES.json").read_text())
        # Eligible raw files are hard links to prior inputs. Store one copy per
        # source object; both manifests retain the exact reconstruction mapping.
        for path in original.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(original)
                if relative.parts[0] == "raw" and str(relative.relative_to("raw")) in current:
                    name = str(relative.relative_to("raw"))
                    if prior[name] != current[name]:
                        raise ValueError("Reused DROID source object identity changed")
                else:
                    file(path)
    elif kind == "pointmaze_preparation":
        tree(root / "pointmaze-input-code-20260908-v1")
        tree(root / "pointmaze-history-code-20260908-v1")
        completed(root / "pointmaze-training-inputs-20260908-v1")
        completed(root / "navigation-input-check-20260907-v1")
        file(root / "run_pointmaze_history_queue_v1.py")
        file(root / "pointmaze-training-preparation-receiving-20260908-v1.json")
        file(root / "pointmaze-training-history-20260908-v1/seed-234/QUEUE_LAUNCH.json")
    else:
        if kind == "metaworld":
            base = root / "fixed-response-behavior-20260908-v1"
            tree(root / "fixed-response-code-20260908-v4")
            for gpu in range(8):
                completed(root / f"fixed-response-worker-checks-20260908-v1/gpu-{gpu}")
        elif kind == "navigation":
            base = root / "navigation-coupling-behavior-20260908-v1"
            tree(root / "navigation-coupling-code-20260908-v2")
            for task in ("wall", "pointmaze"):
                completed(base / task / "engineering")
                tree(base / task / "freeze")
        elif kind == "pusht":
            base = root / "pusht-planning-native-20260908-v1"
            tree(root / "pusht-planning-code-20260908-v1")
            tree(root / "pusht-planning-evidence-20260908-v1")
            completed(base / "engineering")
            completed(base / "simulator-check")
            tree(base / "freeze")
            file(base / "LAUNCH.json")
            file(root / "run_pusht_native_queue_v1.py")
            assets = root / "pusht-planning-assets-20260908-v3"
            for name in ("receiving_report.json", "RECEIVING_DONE.json"):
                file(assets / name)
            for name in ("protocol.json", "files.json", "report.json", "DONE.json"):
                file(assets / "source" / name)
        else:
            raise ValueError("Unknown snapshot role")
        for p in base.rglob("episode-*.json"):
            # Writers publish complete JSON before starting the following episode.
            # If caught mid-write, leave that record for the next snapshot.
            try:
                record = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            file(p)
            file(p.parent / "protocol.json")
            if (p.parent / "DONE.json").is_file():
                file(p.parent / "DONE.json")
                file(p.parent / "report.json")
            if kind == "metaworld":
                calls = p.parent / f"calls-{record['episode']:03d}"
                file(calls / "unroll_calls.json")
                file(calls / "action_trace.json")
    if not chosen:
        raise ValueError("Empty snapshot")
    return sorted(chosen)


HASH_FILES = '''
import hashlib,json,pathlib,sys
root=pathlib.Path('/workspace/jepa-runtime')
names=json.loads(sys.stdin.read())
result={}
for name in names:
 p=root/name
 if not p.is_relative_to(root) or '..' in p.parts or p.is_symlink(): raise ValueError('Unsafe path')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 result[name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(result,sort_keys=True))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=int, choices=tuple(WORKERS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--drive", required=True)
    p.add_argument("--kind", choices=("pointmaze_preparation", "droid_fit_preparation", "droid_coupling_preparation", "droid_behavior"), help="Explicit CPU-only preparation snapshot on Virginia")
    p.add_argument("--local-spool", action="store_true", help="Retain archive for retryable file upload; requires local reserve")
    p.add_argument("--incremental-from", type=Path, help="Verified prior receipt directory; unchanged members are referenced, never recopied")
    args = p.parse_args()
    if args.kind and args.instance != 50189244:
        raise ValueError("Preparation snapshot is leased only on Virginia")
    kind = args.kind or WORKERS[args.instance]
    if not args.drive.startswith("gdrive:Research-Archives/JEPA-WM/live-"):
        raise ValueError("Require a separate grounded private live-snapshot folder")
    workers = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"], text=True))
    worker = next(x for x in workers if x["id"] == args.instance)
    if worker["actual_status"] != "running" or not worker["geolocation"].endswith(", US"):
        raise ValueError("Expected owned running US worker")
    if args.instance == 50259194 and worker["label"] != "jepa-droid-parallel-us-v3":
        raise ValueError("DROID receiving worker ownership label changed")
    args.output.mkdir(parents=True, exist_ok=False)
    ssh = ["ssh", "-i", "/tmp/jepa_vast_50123620_ed25519", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-p",
           str(worker["ports"]["22/tcp"][0]["HostPort"]), "root@" + worker["public_ipaddr"]]
    code = inspect.getsource(select_files) + "\nprint(__import__('json').dumps(select_files('/workspace/jepa-runtime'," + repr(kind) + ")))"
    names = json.loads(subprocess.check_output(ssh + [shlex.join(["python", "-c", code])], text=True))

    def inventory():
        return json.loads(subprocess.check_output(ssh + [shlex.join(["python", "-c", HASH_FILES])],
            input=json.dumps(names), text=True))

    manifest = inventory()
    inherited = None
    if args.incremental_from:
        previous = json.loads((args.incremental_from / "VERIFIED.json").read_text())
        prior = json.loads((args.incremental_from / "FILES.json").read_text())
        if (previous.get("status") != "immutable_live_snapshot_all_members_verified" or
                previous.get("instance") != args.instance or previous.get("files") != len(prior) or
                not previous.get("archive", "").startswith("gdrive:Research-Archives/JEPA-WM/live-")):
            raise ValueError("Incremental parent lacks required verified identity")
        if not set(prior) <= set(manifest) or any(manifest[name] != item for name, item in prior.items()):
            raise ValueError("Prior immutable members missing or changed on source")
        inherited = {"verified_parent": previous, "parent_files": prior,
            "parent_receipt_sha256": hashlib.sha256((args.incremental_from / "VERIFIED.json").read_bytes()).hexdigest(),
            "parent_manifest_sha256": hashlib.sha256((args.incremental_from / "FILES.json").read_bytes()).hexdigest(),
            "all_inherited_source_members_rechecked": True}
        names = [name for name in names if name not in prior]
        manifest = {name: manifest[name] for name in names}
        if not names:
            raise ValueError("No new immutable members for incremental archive")
    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    write("FILES.json", manifest)
    if inherited:
        write("INHERITED.json", inherited)
    write("PLAN.json", {"instance": args.instance, "kind": kind, "drive": args.drive,
        "snapshot_files": len(names), "bytes": sum(x["bytes"] for x in manifest.values()),
        "running_jobs_untouched": True, "partial_study_not_completion": True})
    print(json.dumps({"stage": "inventory", "files": len(names), "bytes": sum(x["bytes"] for x in manifest.values())}), flush=True)
    subprocess.run(["rclone", "mkdir", args.drive], check=True)
    archive = args.drive + f"/instance-{args.instance}-results.tar.gz"
    name_stream = tempfile.TemporaryFile()
    name_stream.write(b"\0".join(n.encode() for n in names) + b"\0"); name_stream.seek(0)
    tar = subprocess.Popen(ssh + ["tar -C /workspace/jepa-runtime --hard-dereference --no-recursion --null -czf - -T -"],
        stdin=name_stream, stdout=subprocess.PIPE)
    spool = args.output.parent / (args.output.name + ".tar.gz")
    if args.local_spool and shutil.disk_usage(args.output).free < sum(x["bytes"] for x in manifest.values()) + 4 * 1024**3:
        tar.terminate(); tar.wait(); name_stream.close()
        raise ValueError("Insufficient local space for archive plus4GiB reserve")
    upload = None if args.local_spool else subprocess.Popen(
        ["rclone", "rcat", archive, "--drive-chunk-size", "16M"], stdin=subprocess.PIPE)
    sink = spool.open("xb") if args.local_spool else upload.stdin
    h, size, last = hashlib.sha256(), 0, time.monotonic()
    try:
        for chunk in iter(lambda: tar.stdout.read(1 << 20), b""):
            h.update(chunk); size += len(chunk); sink.write(chunk)
            if time.monotonic() - last > 20:
                print(json.dumps({"stage": "upload", "bytes": size}), flush=True); last = time.monotonic()
        sink.close()
        if tar.wait() or (upload is not None and upload.wait()):
            raise ValueError("Snapshot transfer failed; source remains intact")
        if args.local_spool:
            # Catch unexpected link/member serialization before paying to upload.
            verify_archive(["cat", str(spool)], h.hexdigest(), size, manifest)
            print(json.dumps({"stage": "retryable_file_upload", "archive_bytes": size}), flush=True)
            subprocess.run(["rclone", "copyto", str(spool), archive, "--immutable", "--drive-chunk-size", "8M",
                "--retries", "3", "--low-level-retries", "3", "--timeout", "90s", "--contimeout", "15s",
                "--stats", "15s", "--stats-one-line", "--stats-log-level", "NOTICE", "--tpslimit", "2"], check=True)
    finally:
        tar.stdout.close()
        name_stream.close()
        for proc in (tar, upload):
            if proc is not None and proc.poll() is None:
                proc.terminate(); proc.wait()
    print(json.dumps({"stage": "verify_readback", "archive_bytes": size}), flush=True)
    verify_archive(["rclone", "cat", archive], h.hexdigest(), size, manifest)
    if inventory() != manifest:
        raise ValueError("Selected records changed during snapshot; no verified receipt")
    write("VERIFIED.json", {"status": "immutable_live_snapshot_all_members_verified",
        "archive": archive, "archive_sha256": h.hexdigest(), "archive_bytes": size,
        "files": len(names), "instance": args.instance, "jobs_untouched": True,
        "all_study_complete": False, "source_disks_retained": True,
        "inherited_manifest_sha256": hashlib.sha256((args.output / "INHERITED.json").read_bytes()).hexdigest() if inherited else None})
    subprocess.run(["rclone", "copy", str(args.output), args.drive + f"/receipt-{args.instance}", "--immutable"], check=True)
    subprocess.run(["rclone", "check", str(args.output), args.drive + f"/receipt-{args.instance}", "--download", "--one-way"], check=True)
    print(json.dumps({"status": "live_snapshot_verified", "instance": args.instance}), flush=True)


if __name__ == "__main__":
    main()
