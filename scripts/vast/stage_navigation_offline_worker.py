"""Stage only checksum-bound navigation inputs and an existing runtime, US-to-US."""
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from offline_study.protocol import sha256


VAST = "/Users/stevenyang/.local/share/uv/tools/vastai/bin/vastai"
SOURCE, DESTINATION = 50189244, 50205763


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--rsync", action="store_true", help="Resume interrupted owned staging without retransmitting complete files")
    parser.add_argument("--wall-training-worker", action="store_true",
                        help="Stage the complete Wall source population to the existing California worker, without replacing its runtime")
    args = parser.parse_args()
    instances = json.loads(subprocess.check_output([VAST, "show", "instances", "--raw"], text=True))
    if sum(float(row["instance"]["totalHour"]) for row in instances) > 7:
        raise ValueError("Aggregate exceeds authorized hourly cap")
    by_id = {row["id"]: row for row in instances}
    endpoints = {}
    destination_id = 50195621 if args.wall_training_worker else DESTINATION
    for identifier in (SOURCE, destination_id):
        row = by_id[identifier]
        if row["actual_status"] != "running" or not row["geolocation"].endswith(", US"):
            raise ValueError("Source/destination must be live US-owned instances")
        endpoints[identifier] = row["public_ipaddr"], int(row["ports"]["22/tcp"][0]["HostPort"])
    runtime = "workspace/jepa-runtime"
    paths = [] if args.assets_only else ["workspace/jepa-python", "workspace/jepa-planning-python",
        "root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu", "root/.cache/torch/hub",
        "workspace/jepa_steering/vendor/jepa-wms", runtime + "/code-v14"]
    paths += [runtime + "/navigation-offline-cohorts-20260907-v1",
              runtime + "/navigation-input-check-20260907-v1"]
    for task, folder in (("wall", "wall_single"), ("pointmaze", "point_maze")):
        root = args.cohorts / task
        frozen = json.loads((root / "FROZEN.json").read_text())
        if sha256(root / "input_files.json") != frozen["input_files_sha256"]:
            raise ValueError("Local input manifest changed")
        prefix = f"{runtime}/navigation-assets-20260907-v1/extracted/{task}/{folder}/"
        for name in json.loads((root / "input_files.json").read_text()):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe manifest input path")
            paths.append(prefix + name)
        paths.append(f"{runtime}/navigation-assets-20260907-v1/downloads/model/jepa_wm_{task}.pth.tar")
    if args.wall_training_worker:
        # These destination paths were reserved and checked absent. No active
        # runtime, vendor source, model weights or MetaWorld outputs are replaced.
        paths = [f"{runtime}/navigation-assets-20260907-v1/extracted/wall/wall_single",
                 f"{runtime}/navigation-input-check-20260907-v1",
                 f"{runtime}/wall-training-accumulation-pilot-20260907-v2"]
        paths += [f"{runtime}/navigation-assets-20260907-v1/{name}.json" for name in ("protocol", "report", "DONE")]
    source, destination = endpoints[SOURCE], endpoints[destination_id]
    ssh_options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                   "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=6", "-o", "ConnectTimeout=15"]
    agent = subprocess.check_output(["ssh-agent", "-s"], text=True)
    env = dict(os.environ)
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        env[name] = re.search(name + r"=([^;]+);", agent).group(1)
    try:
        subprocess.run(["ssh-add", args.key], env=env, check=True, stdout=subprocess.DEVNULL)
        dest = ["ssh", *ssh_options, "-p", str(destination[1]), "root@" + destination[0],
                "set -o pipefail; gzip -d | tar -C / -xf -"]
        command = ("set -o pipefail; tar --exclude=__pycache__ --exclude='._*' -C / --null -T - -cf -"
                   " | gzip -1 | " + shlex.join(dest))
        if args.rsync:
            transport = shlex.join(["ssh", *ssh_options, "-p", str(destination[1])])
            command = shlex.join(["rsync", "-arzR", "--partial", "--timeout=180", "--info=stats2",
                "--exclude=__pycache__", "--exclude=._*", "--from0", "--files-from=-", "-e", transport,
                "/", "root@" + destination[0] + ":/"])
        subprocess.run(["ssh", "-A", "-i", args.key, *ssh_options, "-p", str(source[1]),
                        "root@" + source[0], command], input="\0".join(paths).encode() + b"\0",
                       env=env, check=True, timeout=2400)
        print(json.dumps({"status": "selected_navigation_inputs_and_runtime_copied_pending_gpu_checks",
                          "source": SOURCE, "destination": destination_id, "input_path_count": len(paths),
                          "complete_wall_training_data_staged": args.wall_training_worker}), flush=True)
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, stdout=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    main()
