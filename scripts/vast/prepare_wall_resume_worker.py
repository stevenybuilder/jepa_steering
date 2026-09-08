"""Verify restored Wall history inputs before a separate, explicitly leased resume."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile

ROOT = Path("/workspace/jepa-runtime")
OUT = ROOT / "wall-resume-input-check-20260908-v1"
CHECKPOINTS = {
    235: "71cec9f441fd3eefa3e274810b5072499fb6128dcdf2610ec6ef39bd3b49d937",
    236: "150e39e43fd2a49f14d997be0947330981682e9e9976dd7c3fbcf83e39c3d9d8"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        source = ROOT / "wall-history-code-v1/src/offline_study/training_history.py"
        if digest(source) != "951731fde0aca7551ea1afa84e6f449e2849ecedf2514df71389f2ff2c8021c5":
            raise ValueError("Original training/resume implementation changed")
        receipt = ROOT / "wall-training-inputs-20260907-v1"
        expected = json.loads((receipt / "files.json").read_text())
        report = json.loads((receipt / "report.json").read_text())
        if (digest(receipt / "report.json") != json.loads((receipt / "DONE.json").read_text())["report_sha256"] or
                digest(receipt / "files.json") != report["files_sha256"] or len(expected) != 1924):
            raise ValueError("Original complete Wall data receipt changed")
        archive = ROOT / "navigation-assets-20260907-v1/downloads/dataset/wall/wall_single.zip"
        if digest(archive) != "2b4ae4ed0ad03b337efac637f17752e7e7e27f864fec39dc25b51fef490c980d":
            raise ValueError("Official Wall archive changed")
        target = ROOT / "navigation-assets-20260907-v1/extracted/wall"
        target.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if len({m.filename for m in members}) != len(members):
                raise ValueError("Duplicate archive member")
            expanded = sum(m.file_size for m in members)
            if expanded > 64 * 1024**3 or shutil.disk_usage(target).free < expanded + 16 * 1024**3:
                raise ValueError("Insufficient room for complete data plus checkpoint reserve")
            for member in members:
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in member.filename or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Unsafe archive path")
            z.extractall(target)
        for index, (name, wanted) in enumerate(expected.items(), 1):
            p = target / "wall_single" / name
            if p.stat().st_size != wanted["bytes"] or digest(p) != wanted["sha256"]:
                raise ValueError("Extracted Wall member differs from original training source: " + name)
            if index % 200 == 0:
                print(json.dumps({"verified_files": index, "target": 1924}), flush=True)
        checkpoints = {}
        for seed, wanted in CHECKPOINTS.items():
            base = ROOT / f"wall-training-history-20260907-v1/seed-{seed}"
            p = base / "remaining-epochs/jepa-e1.pth.tar"
            if digest(p) != wanted:
                raise ValueError("Restored epoch2 checkpoint differs from verified Drive archive")
            proof = base / "epoch-one-engineering"
            original = json.loads((proof / "report.json").read_text())
            if (digest(proof / "report.json") != json.loads((proof / "DONE.json").read_text())["report_sha256"] or
                    original["seed"] != seed or original["status"] != "one_epoch_training_engineering_complete" or
                    digest(proof / "RESUME_PARITY.json") != original["resume_parity_sha256"]):
                raise ValueError("Original seed-specific resume proof changed")
            checkpoints[str(seed)] = {"path": str(p), "sha256": wanted, "resume_epoch": 2}
        write("report.json", {"status": "receiving_wall_inputs_and_epoch2_histories_verified",
            "data_files": 1924, "source_sha256": digest(source), "checkpoints": checkpoints,
            "data_receipt_sha256": digest(receipt / "report.json"), "seconds": time.monotonic() - started,
            "training_started": False, "hardware_change_not_bitwise_author_training_claim": True})
        write("DONE.json", {"report_sha256": digest(OUT / "report.json")})
        print(json.dumps({"status": "wall_inputs_verified_not_training_complete"}), flush=True)
    except Exception as exc:
        write("FAILED.json", {"error": str(exc), "training_started": False})
        raise


if __name__ == "__main__":
    main()
