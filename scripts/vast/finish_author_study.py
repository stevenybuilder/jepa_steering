"""Collect both US workers' complete offline evidence and run the fixed gates.

Never starts confirmation from an incomplete summary. Failed and partial records
remain on their originating workers. This coordinator does not terminate rentals.
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from offline_study.protocol import sha256, write_json

RUNTIME = Path("/workspace/jepa-runtime")
ORIGINAL = RUNTIME / "author-correction-20260907"
PUSHT = RUNTIME / "pusht-author-replication-20260907"
MW = RUNTIME / "metaworld-author-extension-20260907"
OUTPUT = RUNTIME / "three-task-offline-closure-20260907"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-i",
       str(RUNTIME / "transfer-to-50159352"), "-p", "17784", "root@136.36.139.116"]


def remote_ready():
    command = "test -f " + str(PUSHT / "logs/rebalance-worker-DONE.json")
    result = subprocess.run(SSH + [command], capture_output=True, text=True, timeout=30)
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot verify Push-T worker readiness: " + result.stderr[-500:])
    return result.returncode == 0


def collect_bfloat16():
    script = ("import json,hashlib; from pathlib import Path; "
              f"r=Path({str(PUSHT)!r}); "
              "paths=[p for d in ('evaluation-v1/bfloat16','analysis-v1/bfloat16') for p in (r/d).rglob('*') if p.is_file()]; "
              "print(json.dumps({str(p.relative_to(r)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))")
    command = "/workspace/jepa-py311/bin/python -c " + shlex.quote(script)
    response = subprocess.run(SSH + [command], check=True, capture_output=True, text=True, timeout=120)
    manifest = json.loads(response.stdout)
    if len([p for p in manifest if p.endswith("/DONE.json")]) != 15:
        raise ValueError("Need ten completed BF16 shards and five verified analyses")
    for name in manifest:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid remote artifact path")
    temporary = Path(tempfile.mkdtemp(prefix="pusht-bf16-transfer-", dir=RUNTIME))
    # SFTP avoids the pipe/compressor stall observed with this provider's reverse
    # SSH tar stream. Verify every byte against the source manifest before install.
    for directory in ("evaluation-v1", "analysis-v1"):
        target = temporary / directory
        target.mkdir()
        command = ["scp", "-q", "-r", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-i", str(RUNTIME / "transfer-to-50159352"), "-P", "17784",
            "root@136.36.139.116:" + str(PUSHT / directory / "bfloat16"), str(target)]
        subprocess.run(command, check=True, timeout=600)
    for name, expected in manifest.items():
        if sha256(temporary / name) != expected:
            raise ValueError("Peer artifact transfer changed bytes")
    for directory in ("evaluation-v1", "analysis-v1"):
        destination = PUSHT / directory / "bfloat16"
        if destination.exists():
            raise FileExistsError("Refusing to overwrite existing collected results")
        destination.parent.mkdir(exist_ok=True)
        shutil.move(str(temporary / directory / "bfloat16"), destination)
    write_json(OUTPUT / "bf16-transfer.json", {"source_instance": 50159352, "target_instance": 50125440,
        "artifact_sha256": manifest, "verified_byte_identical": True,
        "source_preserved": True, "staging_directory": str(temporary)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-relay-only", action="store_true",
                        help="Wait for verified relay receipt without contacting an already-stopped source worker")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=False)
    started, transferred = time.monotonic(), False
    try:
        while True:
            if time.monotonic() - started > 21600:
                raise TimeoutError("Six-hour offline closure cap")
            for failure in (MW / "closure-v1/FAILED.json", PUSHT / "logs/rebalance-primary-FAILED.json"):
                if failure.exists():
                    raise RuntimeError("A prerequisite failed: " + str(failure))
            if not transferred:
                relay = PUSHT / "relay-verification"
                if (relay / "DONE.json").exists():
                    done = json.loads((relay / "DONE.json").read_text())
                    if sha256(relay / "report.json") != done["report_sha256"]:
                        raise ValueError("Relayed artifact receipt changed")
                    receipt = json.loads((relay / "report.json").read_text())
                    if receipt["bf16_analysis_scopes"] != 5 or receipt["bf16_completed_shards"] != 10:
                        raise ValueError("Incomplete relayed evidence")
                    for name, expected in receipt["artifacts_sha256"].items():
                        if sha256(PUSHT / name) != expected:
                            raise ValueError("Relayed artifact checksum mismatch")
                    transferred = True
                elif not args.local_relay_only and remote_ready():
                    collect_bfloat16()
                    transferred = True
            ready = transferred and (MW / "closure-v1/DONE.json").exists() and (PUSHT / "logs/rebalance-primary-DONE.json").exists()
            write_json(OUTPUT / "progress.json", {"bf16_collected": transferred,
                "full_metaworld_analysis_complete": (MW / "closure-v1/DONE.json").exists(),
                "pusht_fp32_complete": (PUSHT / "logs/rebalance-primary-DONE.json").exists(),
                "seconds": time.monotonic() - started, "confirmation_jobs_launched": False})
            if ready:
                break
            time.sleep(20)
        commands = [
            [sys.executable, "-m", "offline_study.author_summary", "--root", str(ORIGINAL),
             "--metaworld-extension-root", str(MW), "--pusht-root", str(PUSHT), "--output", str(OUTPUT / "metrics")],
            [sys.executable, "-m", "offline_study.author_advancement", "--metaworld-extension-root", str(MW),
             "--pusht-root", str(PUSHT), "--output", str(OUTPUT / "advancement")],
        ]
        for index, command in enumerate(commands):
            with (OUTPUT / f"stage-{index}.log").open("x") as log:
                subprocess.run(command, check=True, timeout=1800, stdout=log, stderr=subprocess.STDOUT)
        write_json(OUTPUT / "report.json", {"status": "all_primary_offline_pools_and_fixed_advancement_verified",
            "trajectories": {"reach": 33, "reach-wall": 27, "pusht": 21},
            "metrics_sha256": sha256(OUTPUT / "metrics/report.json"),
            "advancement_sha256": sha256(OUTPUT / "advancement/report.json"),
            "full_study_complete": False, "fresh_confirmation": False, "confirmation_jobs_launched": False})
        write_json(OUTPUT / "DONE.json", {"report_sha256": sha256(OUTPUT / "report.json")})
    except Exception as exc:
        write_json(OUTPUT / "FAILED.json", {"error": str(exc), "full_study_complete": False})
        raise


if __name__ == "__main__":
    main()
