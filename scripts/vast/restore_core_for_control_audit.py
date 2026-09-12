"""Read back the immutable core archive onto worker storage for control audit.

No model calls, scientific changes, or GPU use. Full part/archive/member hashes
must match before this restoration is eligible for analysis.
"""
import hashlib
import json
from pathlib import Path
import tarfile

from restore_table_evidence import request, safe_path
from verify_drive_archive_by_id import PartsReader

ROOT = Path("/workspace/table-completion-20260911-v1")
PROOF = ROOT / "control-audit-inputs"
OUT = ROOT / "control-audit-restored"


def main():
    spec = json.loads((PROOF / "PARTS.json").read_text())
    manifest = json.loads((PROOF / "FILES.json").read_text())
    if hashlib.sha256((PROOF / "FILES.json").read_bytes()).hexdigest() != spec["member_manifest_sha256"]:
        raise ValueError("Core manifest identity differs")
    OUT.mkdir(exist_ok=False)
    reader = PartsReader(spec["parts_in_join_order"], "1SoXkmEJmrCZXBnKk3re6mX8WwJiWBokY", request)
    hashed, seen, count = hashlib.sha256(), set(), 0
    class Reader:
        def read(self, size):
            nonlocal count
            block = reader.read(size)
            count += len(block)
            hashed.update(block)
            return block
    stream = Reader()
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                name = member.name
                safe_path(name)
                if not member.isfile():
                    raise ValueError("Unexpected core member type")
                if name not in manifest:
                    # direct_gcs_archive.build_archive adds selected files only.
                    raise ValueError("Unexpected core member: " + name)
                if name in seen or member.size != manifest[name]["bytes"]:
                    raise ValueError("Duplicate or wrong-sized core member")
                target = OUT / safe_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with archive.extractfile(member) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(4 << 20), b""):
                        h.update(block)
                        output.write(block)
                if h.hexdigest() != manifest[name]["sha256"]:
                    raise ValueError("Core member checksum differs")
                seen.add(name)
                if len(seen) % 500 == 0:
                    print(json.dumps({"verified_members": len(seen), "compressed_bytes_read": count}), flush=True)
        while stream.read(4 << 20):
            pass
        if (count != spec["archive_bytes"] or hashed.hexdigest() != spec["archive_sha256"] or
                seen != set(manifest) or len(reader.verified) != len(spec["parts_in_join_order"])):
            raise ValueError("Incomplete core restoration")
        with (ROOT / "CONTROL_AUDIT_RESTORED.json").open("x") as output:
            json.dump({"files": len(seen), "archive_sha256": hashed.hexdigest(),
                "archive_bytes": count, "part_count": len(reader.verified),
                "scientific_outputs_unchanged": True}, output, indent=2)
        print(json.dumps({"status": "core_restored_verified", "files": len(seen)}), flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
