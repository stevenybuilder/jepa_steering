"""CPU-only Push-T static coupling preparation on the explicitly owned NJ worker."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile


INSTANCE = 50239185
ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/workspace/jepa-runtime"
CODE = "pusht-coupling-code-20260908-v2"
EVIDENCE = "pusht-coupling-evidence-20260908-v2"
PANEL = "pusht-coupling-behavior-20260908-v2"


def main():
    rows = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"], text=True))
    worker = next(r for r in rows if r["id"] == INSTANCE)
    if (worker["actual_status"] != "running" or not worker["geolocation"].endswith(", US") or
            worker["label"] != "jepa-pusht-wall-history-us-v1" or
            sum(r["instance"]["totalHour"] for r in rows) > 7):
        raise ValueError("Lease/geography/budget changed")
    ssh = ["ssh", "-i", "/tmp/jepa_vast_50123620_ed25519", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-p",
        str(worker["ports"]["22/tcp"][0]["HostPort"]), "root@" + worker["public_ipaddr"]]
    subprocess.run(ssh + [f"test ! -e {REMOTE}/{CODE} && test ! -e {REMOTE}/{EVIDENCE} && test ! -e {REMOTE}/{PANEL}"], check=True)
    members = {}
    for path in sorted((ROOT / "src/offline_study").glob("*.py")):
        members[f"{CODE}/src/offline_study/{path.name}"] = path
    for name in ("test_pusht_coupling_behavior.py", "test_pusht_planning.py"):
        members[f"{CODE}/tests/{name}"] = ROOT / "tests" / name
    members[f"{CODE}/docs/PUSHT_COUPLING_BEHAVIOR.md"] = ROOT / "docs/PUSHT_COUPLING_BEHAVIOR.md"
    evidence = ROOT / "artifacts/offline_study/utah-durable-20260907/pusht-author-replication-20260907"
    fit = evidence / "fits-v1/bfloat16/pusht"
    for name in ("protocol.json", "operator_bank.pt", "fit_receipt.json", "DONE.json", "source_protocol.json", "source_fit_receipt.json"):
        members[f"{EVIDENCE}/fits/vision_action_coupling/{name}"] = fit / "vision_action_coupling" / name
    members[f"{EVIDENCE}/fits/PARITY.json"] = fit / "PARITY.json"
    for path in sorted((evidence / "cohorts/pusht").iterdir()):
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            raise ValueError("Unexpected cohort member")
        members[f"{EVIDENCE}/cohort/{path.name}"] = path
    manifest = {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                for name, path in members.items()}
    output = ROOT / "artifacts/offline_study/pusht-coupling-preparation-20260908-v2"
    output.mkdir(parents=True, exist_ok=False)
    (output / "FILES.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
            for name, path in members.items():
                archive.add(path, arcname=name, recursive=False)
        stream.seek(0)
        subprocess.run(ssh + [f"tar --keep-old-files -C {REMOTE} -xzf -"], stdin=stream, check=True)
    check = """
import hashlib,json,pathlib,sys
root=pathlib.Path('/workspace/jepa-runtime')
files=json.load(sys.stdin)
for name,want in files.items():
 p=root/name
 if p.is_symlink() or '..' in p.parts or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:
  raise ValueError('Receiving member mismatch: '+name)
print(json.dumps({'status':'all_preparation_members_verified','files':len(files),'gpu_job_launched':False}))
"""
    result = subprocess.check_output(ssh + [shlex.join(["python", "-c", check])],
        input=json.dumps(manifest), text=True)
    (output / "RECEIVING.json").write_text(result)
    print(result, flush=True)
    env = ["env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        f"PYTHONPATH={REMOTE}/{CODE}/src:/workspace/jepa-python/lib/python3.10/site-packages",
        "LD_LIBRARY_PATH=/opt/conda/lib", "/workspace/jepa-planning-python/bin/python"]
    tests = env + ["-m", "unittest", "discover", "-s", f"{REMOTE}/{CODE}/tests", "-p", "test_pusht_coupling_behavior.py", "-v"]
    proof = subprocess.run(ssh + [shlex.join(tests)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "receiving-tests.log").write_text(proof.stdout); print(proof.stdout, flush=True)
    command = env + ["-m", "offline_study.pusht_coupling_behavior", "freeze",
        "--vendor", "/workspace/jepa_steering/vendor/jepa-wms",
        "--reference", f"{REMOTE}/pusht-planning-native-20260908-v1",
        "--reference-code", f"{REMOTE}/pusht-planning-code-20260908-v1",
        "--fit", f"{REMOTE}/{EVIDENCE}/fits/vision_action_coupling",
        "--cohort", f"{REMOTE}/{EVIDENCE}/cohort/cohort.json",
        "--data-root", f"{REMOTE}/pusht-planning-assets-20260908-v3/source/data/pusht_noise",
        "--output", f"{REMOTE}/{PANEL}/freeze"]
    subprocess.run(ssh + [shlex.join(command)], check=True, timeout=180)
    local_freeze = output / "freeze"
    local_freeze.mkdir()
    for name in ("protocol.json", "FROZEN.json"):
        value = subprocess.check_output(ssh + [shlex.join(["cat", f"{REMOTE}/{PANEL}/freeze/{name}"])])
        (local_freeze / name).write_bytes(value)
    protocol_hash = hashlib.sha256((local_freeze / "protocol.json").read_bytes()).hexdigest()
    if json.loads((local_freeze / "FROZEN.json").read_text())["protocol_sha256"] != protocol_hash:
        raise ValueError("Copied freeze checksum mismatch")
    (output / "DONE.json").write_text(json.dumps({"status": "cpu_preparation_and_pre_outcome_freeze_complete",
        "protocol_sha256": protocol_hash, "instance": INSTANCE, "source_files": len(members),
        "gpu_engineering_complete": False, "candidate_jobs_launched": False, "fresh_confirmation": False}, indent=2) + "\n")
    print(json.dumps({"status": "prepared_not_gpu_validated", "protocol_sha256": protocol_hash}), flush=True)


if __name__ == "__main__":
    main()
