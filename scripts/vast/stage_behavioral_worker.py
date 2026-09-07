"""Stream only the owned MetaWorld planning runtime and frozen inputs to a US worker.

No local dataset copy, stored private key, raw video, or GPU work during staging.
"""
import argparse
import json
import os
import re
import shlex
import subprocess


VAST = "/Users/stevenyang/.local/share/uv/tools/vastai/bin/vastai"
SOURCE_ID = 50125440
DESTINATION_ID = 50195621
PATHS = [
    "workspace/jepa-python", "workspace/jepa-planning-python",
    "root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu", "root/.cache/torch/hub",
    "workspace/jepa_steering/vendor/jepa-wms",
    "workspace/jepa-runtime/checkpoints/jepa_wm_metaworld.pth.tar",
    "workspace/jepa-runtime/author-correction-20260907/code-v33",
    "workspace/jepa-runtime/author-correction-20260907/code-v35",
    "workspace/jepa-runtime/behavioral-development-freeze-20260907-v1",
    "workspace/jepa-runtime/planning-panel-coupling-engineering-20260907-v1",
]
for task in ("reach", "reach-wall"):
    PATHS += [f"workspace/jepa-runtime/author-correction-20260907/fits-v1/bfloat16/{task}/{part}"
              for part in ("PARITY.json", "vision_action_coupling", "operator_rank")]


def endpoint(instance):
    if not instance.get("geolocation", "").endswith(", US") or instance["actual_status"] != "running":
        raise ValueError("Worker must be running and US-located")
    return instance["public_ipaddr"], int(instance["ports"]["22/tcp"][0]["HostPort"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--tar-stream", action="store_true", help="Fallback if rsync is unavailable")
    parser.add_argument("--pointmaze-to-training-worker", action="store_true",
                        help="Stage the verified isolated CA simulator/runtime to the US training worker")
    args = parser.parse_args()
    paths = (["workspace/jepa-runtime/reach-worker-fit-fixture-v1",
              "workspace/jepa-runtime/reach-wall-worker-fit-fixture-v1",
              "workspace/jepa-runtime/planning-env-smoke-v2"] if args.fixtures_only else PATHS)
    instances = json.loads(subprocess.check_output([VAST, "show", "instances", "--raw"], text=True))
    by_id = {row["id"]: row for row in instances}
    source_id, destination_id = SOURCE_ID, DESTINATION_ID
    if args.pointmaze_to_training_worker:
        if args.fixtures_only:
            raise ValueError("Fixture and navigation-runtime modes are separate")
        source_id, destination_id = 50195621, 50189244
        paths = ["workspace/jepa-python", "workspace/jepa-planning-python", "workspace/jepa-maze-python",
                 "root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu", "root/.cache/torch/hub",
                 "root/.mujoco/mujoco210", "workspace/jepa-runtime/pointmaze-setup-v1/D4RL",
                 "workspace/jepa-runtime/pointmaze-env-check-20260907-v2", "workspace/jepa-runtime/code-v37"]
        # All these destination paths were checked absent before reservation.
        # In particular, do not replace the running DROID environment or vendor.
    source = endpoint(by_id[source_id])
    destination = endpoint(by_id[destination_id])
    if sum(float(row["instance"]["totalHour"]) for row in instances) > 7:
        raise ValueError("Current aggregate cost exceeds the authorized cap")
    agent_output = subprocess.check_output(["ssh-agent", "-s"], text=True)
    env = dict(os.environ)
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        env[name] = re.search(name + r"=([^;]+);", agent_output).group(1)
    try:
        subprocess.run(["ssh-add", args.key], env=env, check=True, stdout=subprocess.DEVNULL)
        dest_ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=6",
                    "-o", "ConnectTimeout=15", "-p", str(destination[1]), "root@" + destination[0]]
        # Restartable transfer: interrupted files are completed on retry. No
        # --delete, in-place overwrite of running code, or local staging archive.
        command = shlex.join(["rsync", "-a", "--relative", "--partial", "--compress",
            "--timeout=180", "--info=stats2", "--exclude=__pycache__", "--exclude=._*",
            "-e", shlex.join(dest_ssh[:-1])] + ["/" + path for path in paths] +
            ["root@" + destination[0] + ":/"])
        if args.tar_stream:
            command = "set -o pipefail; tar --exclude=__pycache__ --exclude='._*' -C / -cf - "
            command += " ".join(map(shlex.quote, paths))
            command += " | gzip -1 | " + shlex.join(dest_ssh + ["set -o pipefail; gzip -d | tar -C / -xf -"])
        subprocess.run(["ssh", "-A", "-i", args.key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=6",
                        "-p", str(source[1]), "root@" + source[0], command],
                       env=env, check=True, timeout=1800)
        print(json.dumps({"status": "runtime_stream_copied_not_yet_gpu_validated", "source": source_id,
                          "destination": destination_id, "paths": paths}), flush=True)
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
