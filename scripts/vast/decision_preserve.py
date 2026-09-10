"""Package/verify this diagnostic only; no provider lifecycle operations."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(archive):
    project = Path("/workspace/decision")
    run = project / "artifacts/offline_study/decision-diagnostic-20260910-v2-retry1"
    if not (run / "ANALYSIS_DONE.json").is_file() or (run / "FAILED.json").exists():
        raise ValueError("Only completed, analyzed diagnostic can use this closeout path")
    if archive.exists():
        raise ValueError("Never overwrite an existing preservation archive")
    packages = subprocess.check_output(["uv", "pip", "freeze", "--python", "/workspace/decision-python/bin/python"], text=True)
    files = [p for p in project.rglob("*") if p.is_file() and
             not any(x in (".git", "__pycache__", ".cache") or x.startswith("._") for x in p.relative_to(project).parts)]
    files += [Path("/workspace") / name for name in ("decision-bootstrap.log", "decision-run.log",
              "decision-run-retry1.log", "decision-inputs.tar.gz", "decision-packages.txt")]
    files = sorted(files)
    if any(p.is_symlink() for p in files):
        raise ValueError("Unexpected symlink requires explicit preservation handling")
    members = {str(p.relative_to("/workspace")): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}
    manifest = {"instance": 50531754, "members": members, "packages": packages,
        "protocol_sha256": digest(run / "protocol.json"), "analysis_sha256": digest(run / "analysis.json"),
        "omitted": "rebuildable Python environment, public checkpoints/hub cache, git database, bytecode, Apple resource-fork metadata",
        "full_six_task_study_claimed": False}
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
    with tarfile.open(archive, "x:gz") as out:
        entry = tarfile.TarInfo("PRESERVATION_MANIFEST.json")
        entry.size = len(payload)
        out.addfile(entry, io.BytesIO(payload))
        for path in files:
            out.add(path, arcname=str(path.relative_to("/workspace")), recursive=False)
    for path in files:
        if digest(path) != members[str(path.relative_to("/workspace"))]["sha256"]:
            raise ValueError("Source changed during packaging; retain archive and source")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest(archive), "members": len(members)}), flush=True)


def verify(archive):
    with tarfile.open(archive, "r:gz") as source:
        manifest = json.load(source.extractfile("PRESERVATION_MANIFEST.json"))
        found = set()
        for entry in source:
            if entry.name == "PRESERVATION_MANIFEST.json":
                continue
            if not entry.isfile() or entry.name in found or entry.name not in manifest["members"]:
                raise ValueError("Unexpected/duplicate/non-regular archive member")
            found.add(entry.name)
            h = hashlib.sha256()
            stream = source.extractfile(entry)
            for block in iter(lambda: stream.read(4 << 20), b""):
                h.update(block)
            expected = manifest["members"][entry.name]
            if entry.size != expected["bytes"] or h.hexdigest() != expected["sha256"]:
                raise ValueError("Preserved member differs: " + entry.name)
        if found != set(manifest["members"]):
            raise ValueError("Archive coverage incomplete")
    result = {"archive": str(archive), "sha256": digest(archive), "bytes": archive.stat().st_size,
              "verified_members": len(found), "analysis_sha256": manifest["analysis_sha256"],
              "protocol_sha256": manifest["protocol_sha256"]}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "verify"))
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    (pack if args.command == "pack" else verify)(args.archive)


if __name__ == "__main__":
    main()
