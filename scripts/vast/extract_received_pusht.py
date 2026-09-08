"""Verify the two pinned transferred Push-T inputs before receiving-side extraction."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path("/workspace/jepa-runtime/pusht-planning-assets-20260908-v2")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    archive = ROOT / "downloads/datasets/pusht_noise.zip"
    checkpoint = ROOT / "downloads/models/jepa_wm_pusht.pth.tar"
    if (archive.stat().st_size != 2785304515 or
            digest(archive) != "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08" or
            checkpoint.stat().st_size != 211639615 or
            digest(checkpoint) != "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"):
        raise ValueError("Receiving Push-T archive/checkpoint differs from pinned release")
    target = ROOT / "data"
    target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Duplicate archive member")
        expanded = sum(m.file_size for m in members)
        if expanded > 64 * 1024**3 or shutil.disk_usage(target).free < expanded + 16 * 1024**3:
            raise ValueError("Insufficient space for full input extraction and reserve")
        for member in members:
            p = PurePosixPath(member.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in member.filename or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe archive member")
        z.extractall(target)
    files = {str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in target.rglob("*") if p.is_file()}
    (ROOT / "receiving_files.json").write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
    report = {"status": "received_full_pusht_archive_and_checkpoint_verified", "files": len(files),
        "archive_sha256": digest(archive), "checkpoint_sha256": digest(checkpoint),
        "files_sha256": digest(ROOT / "receiving_files.json"), "planning_executed": False, "fresh_confirmation": False}
    (ROOT / "receiving_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (ROOT / "RECEIVING_DONE.json").write_text(json.dumps({"report_sha256": digest(ROOT / "receiving_report.json")}) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
