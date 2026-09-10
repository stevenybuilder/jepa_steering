"""Copy only public official inputs and prepared runtime via authenticated SSH.

Uses an ephemeral local SSH agent; never copies a private key to a worker.
Destinations are the already-owned workers, not newly rented instances.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--destination", action="append", required=True, help="proxy-host:port")
    parser.add_argument("--existing-archive", action="store_true")
    parser.add_argument("--archive")
    args = parser.parse_args()
    # Explicit user policy: existing/available does not mean geographically eligible.
    instances = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"], text=True))
    allowed = set()
    for instance in instances:
        if instance.get("geolocation") != "United States, US":
            continue
        allowed.add(f"{instance.get('ssh_host')}:{instance.get('ssh_port')}")
        for binding in (instance.get("ports") or {}).get("22/tcp", []):
            allowed.add(f"{instance.get('public_ipaddr')}:{binding['HostPort']}")
    if any(destination not in allowed for destination in args.destination):
        parser.error("US-only policy: destination is not a live verified US instance endpoint")
    output = subprocess.check_output(["ssh-agent", "-s"], text=True)
    env = dict(os.environ)
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        env[name] = re.search(name + r"=([^;]+);", output).group(1)
    try:
        subprocess.run(["ssh-add", args.key], env=env, check=True, stdout=subprocess.DEVNULL)
        source_ssh = ["ssh", "-A", "-i", args.key, "-o", "BatchMode=yes",
                      "-o", "ConnectTimeout=15", "-p", "45353", "root@209.146.116.50"]
        archive = args.archive or "/workspace/jepa-runtime/support-sweeps-20260907/runtime-inputs.tar.gz"
        if args.archive and not args.existing_archive:
            parser.error("Custom archives must already exist")
        paths = ["workspace/jepa-python", "root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu",
                 "root/.cache/torch/hub", "workspace/jepa_steering/vendor/jepa-wms",
                 "workspace/jepa-runtime/data/pusht_noise", "workspace/jepa-runtime/checkpoints",
                 "workspace/jepa-runtime/registries",
                 "workspace/jepa-runtime/runs/pusht-inventory-v3-development-exposed"]
        command = "mkdir -p /workspace/jepa-runtime/support-sweeps-20260907 && test ! -e " + archive
        command += " && tar -C / -czf " + archive + " " + " ".join(map(shlex.quote, paths))
        command += " && sha256sum " + archive
        if not args.existing_archive:
            subprocess.run(source_ssh + [command], env=env, check=True)
        children = []
        for destination in args.destination:
            host, port = destination.rsplit(":", 1)
            command = f"scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -P {int(port)}"
            filename = PurePosixPath(archive).name if args.archive else "support-runtime-inputs.tar.gz"
            command += " " + shlex.quote(archive) + " " + shlex.quote(f"root@{host}:/workspace/{filename}")
            children.append(subprocess.Popen(source_ssh + [command], env=env))
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError("A worker transfer failed; partial archives retained")
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, stdout=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    main()
