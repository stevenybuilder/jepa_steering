"""Transfer and bootstrap one smoke-passed, board-leased receiving host only."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from smoke_eight_gpu_hosts import api

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/workspace/fresh-four-20260912-v1"


def main(receipt, bundle):
    rec = json.loads(receipt.read_text())
    if rec["status"] != "smoke_pass_retained_for_staging":
        raise ValueError("Infrastructure smoke has not passed")
    instance = rec["instance_id"]
    board = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md").read_text()
    lines = [line for line in board.splitlines() if f"| {instance} |" in line]
    if len(lines) != 1 or "rep_geometry_transcoder" not in lines[0] or "LEASED" not in lines[0]:
        raise ValueError("Project board lease absent")
    rows = api("show", "instances", timeout=20)
    row = next(r for r in rows if r["id"] == instance)
    if (row["label"] != rec["label"] or row["actual_status"] != "running"
            or not row["geolocation"].endswith("US") or row["num_gpus"] not in (4, 8)):
        raise ValueError("Receiving identity/state differs")
    expected = json.loads(bundle.with_suffix(bundle.suffix + ".json").read_text())
    out = ROOT / f"artifacts/offline_study/fresh-campaign-stage-20260912-v1/receiving-{instance}"
    out.mkdir(exist_ok=False)
    host, port = rec["ssh"]["host"], str(rec["ssh"]["port"])
    options = ["-i", "/Users/stevenyang/.ssh/id_ed25519", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "UserKnownHostsFile=" + str(out / "known_hosts")]
    ssh = ["ssh", *options, "-p", port, "root@" + host]
    subprocess.run(ssh + [f"test ! -e {REMOTE} && mkdir {REMOTE}"], check=True, timeout=30)
    with (out / "TRANSFER.json").open("x") as handle:
        json.dump({"instance_id": instance, "root": REMOTE, "bundle": expected, "ssh": rec["ssh"]}, handle, indent=2)
    subprocess.run(["scp", *options, "-P", port, str(bundle), f"root@{host}:{REMOTE}/bundle.tgz"], check=True, timeout=600)
    command = f"cd {REMOTE} && echo {shlex.quote(expected['sha256'] + '  bundle.tgz')} | sha256sum -c - && tar --keep-old-files --no-same-owner -xzf bundle.tgz"
    subprocess.run(ssh + [command], check=True, timeout=120)
    code = """import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1])
with (r/'bootstrap.log').open('x') as log:
 p=subprocess.Popen(['bash',str(r/'scripts/vast/fresh_campaign_bootstrap.sh'),str(r),sys.argv[2]],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'bootstrap_pid':p.pid,'scientific_launch_ready':False,'model_calls':0}
with (r/'BOOTSTRAP_LAUNCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
"""
    result = json.loads(subprocess.check_output(ssh + [shlex.join(["python", "-c", code, REMOTE, str(row["num_gpus"])])], text=True, timeout=30))
    with (out / "BOOTSTRAP_LAUNCH.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"instance_id": instance, "root": REMOTE, "ssh": rec["ssh"], **result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    main(args.receipt, args.bundle)
