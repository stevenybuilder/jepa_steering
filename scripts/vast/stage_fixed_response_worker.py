"""Stage an existing verified runtime on the explicitly leased successor worker."""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

SOURCE, DESTINATION = 50189244, 50229002
PATHS = ["workspace/jepa-python", "workspace/jepa-planning-python",
         "root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu",
         "root/.cache/torch/hub", "workspace/jepa_steering/vendor/jepa-wms",
         "workspace/jepa-runtime/fixed-response-assets-20260908-v1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key")
    parser.add_argument("--assets-only-to-indiana", action="store_true")
    parser.add_argument("--wall-archive-only", action="store_true")
    parser.add_argument("--pusht-archives-only", action="store_true")
    parser.add_argument("--destination", type=int, choices=(DESTINATION, 50231985, 50233992, 50239185), default=DESTINATION)
    parser.add_argument("--source", type=int, choices=(SOURCE, 50231985, 50205763), default=SOURCE)
    args = parser.parse_args()
    key, assets_only = args.key, args.assets_only_to_indiana
    destination_id = 50205763 if assets_only else args.destination
    paths = PATHS[-1:] if assets_only else PATHS
    if args.wall_archive_only:
        if args.destination != 50239185 or args.source != SOURCE or assets_only:
            raise ValueError("Wall archive is reserved for Virginia-to-New-Jersey staging")
        paths = ["workspace/jepa-runtime/navigation-assets-20260907-v1/downloads/dataset/wall/wall_single.zip"]
    if args.pusht_archives_only:
        if args.destination != 50239185 or args.source != 50205763 or assets_only or args.wall_archive_only:
            raise ValueError("Push-T archives are reserved for Indiana-to-New-Jersey staging")
        paths = ["workspace/jepa-runtime/pusht-planning-assets-20260908-v2/downloads"]
    rows = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"], text=True))
    workers = {r["id"]: r for r in rows}
    if sum(r["instance"]["totalHour"] for r in rows) > 7:
        raise ValueError("Aggregate authorized budget exceeded")
    def connection(number):
        row = workers[number]
        if row["actual_status"] != "running" or not row["geolocation"].endswith(", US"):
            raise ValueError("Require the leased running US worker")
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=6",
                "-p", str(row["ports"]["22/tcp"][0]["HostPort"]), "root@" + row["public_ipaddr"]]
    label = {50231985: "jepa-fixed-offline-us-v1", 50233992: "jepa-fixed-behavior-us-v1",
        50239185: "jepa-pusht-wall-history-us-v1"}.get(
        destination_id, "jepa-fixed-response-us-v1")
    if not assets_only and workers[destination_id].get("label") != label:
        raise ValueError("Destination ownership label changed")
    source, destination = connection(args.source), connection(destination_id)
    agent = subprocess.check_output(["ssh-agent", "-s"], text=True)
    env = dict(os.environ)
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        env[name] = re.search(name + r"=([^;]+);", agent).group(1)
    try:
        subprocess.run(["ssh-add", key], env=env, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(destination + [" && ".join("test ! -e " + shlex.quote("/" + p) for p in paths)],
                       env=env, check=True)
        remote = "set -o pipefail; nice -n 10 tar --exclude=__pycache__ --exclude='._*' -C / -cf - "
        remote += " ".join(map(shlex.quote, paths))
        remote += " | nice -n 10 gzip -1 | " + shlex.join(destination + ["set -o pipefail; gzip -d | tar -C / -xf -"])
        subprocess.run(source[:1] + ["-A", "-i", key] + source[1:] + [remote],
                       env=env, check=True, timeout=1200)
        print(json.dumps({"status": "runtime_staged_not_gpu_validated", "source": args.source,
                          "destination": destination_id, "paths": paths}))
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
