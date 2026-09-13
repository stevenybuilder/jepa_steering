"""Build a portable, hash-bound four-task input/fit bundle; no model calls or rentals.

The source snapshot is taken at invocation, so build after runner edits finish.
Remote bootstrap downloads only pinned released weights, not datasets or new fits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/offline_study"
WALL_ARCHIVE = Path("/tmp/claude-501/-Users-stevenyang/fb83bc12-5dcf-42ce-af17-cd0e08dad5c5/scratchpad/wall-fit-v1.tgz")
CHECKPOINTS = {
    "metaworld": "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8",
    "pointmaze": "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc",
    "wall": "8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    files = {}
    def tree(directory, destination):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink() and not any(
                part in {"__pycache__", ".pytest_cache"} or part.startswith("._")
                for part in path.relative_to(directory).parts
            ):
                files[str(Path(destination) / path.relative_to(directory))] = path
    for relative in ("src", "configs", "vendor/jepa-wms"):
        tree(ROOT / relative, relative)
    for name in ("fresh-simulator-banks-20260912-v1", "four-task-source-exposure-20260912-v1"):
        tree(BASE / name, "artifacts/offline_study/" + name)
    for path in sorted((ROOT / "scripts/vast").glob("fresh_campaign*")):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = path
    runner = ROOT / "scripts/run_fresh_confirmation.py"
    if runner.is_file():
        files[str(runner.relative_to(ROOT))] = runner
    runtime = BASE / "protected-preparation-worker-20260912-v1/preparation-runtime-cache-20260912-v1.tgz"
    files["runtime-cache.tgz"] = runtime
    for task in ("reach", "reach-wall", "pointmaze", "wall"):
        if task.startswith("reach"):
            refined = BASE / f"fixed-response-20260908-v1/fits/{task}"
            coupling = BASE / f"primary-durable-20260907/fits-v1/bfloat16/{task}/vision_action_coupling"
        else:
            refined = BASE / "table-completion-20260911-v1/refined-pointmaze-worker-v1/restored/fit-v1"
            coupling = BASE / f"primary-durable-20260907/navigation-fits-20260907-v1/bfloat16/{task}/vision_action_coupling"
        tree(coupling, f"fits/{task}/coupling")
        files[f"fits/{task}/PARITY.json"] = coupling.parent / "PARITY.json"
        if task != "wall":
            tree(refined, f"fits/{task}/refined")
            done = json.loads((refined / "DONE.json").read_text())
            for name, digest in done.items():
                if name != "status" and sha(refined / name) != digest:
                    raise ValueError(f"Frozen refined artifact differs: {task}/{name}")
    payloads = {}
    with tarfile.open(WALL_ARCHIVE, "r:gz") as archive:
        for member in archive:
            if member.isfile():
                parts = Path(member.name).parts
                if len(parts) != 2 or parts[0] != "fit-v1":
                    raise ValueError("Unexpected Wall archive member")
                payloads["fits/wall/refined/" + parts[1]] = archive.extractfile(member).read()
    done = json.loads(payloads["fits/wall/refined/DONE.json"])
    if done.get("status") != "fit_only_complete_not_behavioral_clearance":
        raise ValueError("Wall fit did not complete")
    for name, digest in done.items():
        if name != "status" and hashlib.sha256(payloads["fits/wall/refined/" + name]).hexdigest() != digest:
            raise ValueError("Frozen Wall fit checksum differs: " + name)
    return files, payloads


def build(destination, freeze=None):
    files, payloads = inventory()
    if freeze is not None:
        for path in sorted(freeze.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files[str(Path("fresh-freeze") / path.relative_to(freeze))] = path
    manifest = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in files.items()}
    manifest.update({name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                     for name, data in payloads.items()})
    metadata = {
        "files": manifest, "scientific_launch_ready": False,
        "checkpoints": CHECKPOINTS,
        "checkpoint_repository": "facebook/jepa-wms",
        "checkpoint_revision": "9b9c41ef249466630dbf1a20e78391865d07b3b9",
        "wall_fit_source_archive_sha256": sha(WALL_ARCHIVE),
        "vendor_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "vendor/jepa-wms"), "rev-parse", "HEAD"], text=True).strip(),
    }
    payloads["STAGING.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    assets = {task: {"refined": f"fits/{task}/refined", "coupling": f"fits/{task}/coupling",
                     "checkpoint": "checkpoints/jepa_wm_" + ("metaworld" if task.startswith("reach") else task) + ".pth.tar"}
              for task in ("reach", "reach-wall", "pointmaze", "wall")}
    payloads["assets.json"] = (json.dumps(assets, indent=2) + "\n").encode()
    metadata["files"]["assets.json"] = {"bytes": len(payloads["assets.json"]),
        "sha256": hashlib.sha256(payloads["assets.json"]).hexdigest()}
    payloads["STAGING.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz", compresslevel=1) as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    receipt = {"archive": str(destination), "bytes": destination.stat().st_size,
               "sha256": sha(destination), "files": len(manifest),
               "scientific_launch_ready": False}
    with destination.with_suffix(destination.suffix + ".json").open("x") as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps(receipt))


def verify(root):
    metadata = json.loads((root / "STAGING.json").read_text())
    for name, expected in metadata["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("Receiving staging mismatch: " + name)
    print(json.dumps({"staging_verified": True, "files": len(metadata["files"]), "scientific_launch_ready": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    build(args.bundle, args.freeze) if args.bundle else verify(args.verify)
