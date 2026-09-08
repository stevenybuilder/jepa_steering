"""Stage immutable Push-T runtime/evidence on the expressly leased US worker."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

SOURCE, DESTINATION = 50239185, 50245262
RUNTIME = ["workspace/jepa-python", "workspace/jepa-planning-python",
    "root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu",
    "root/.cache/torch/hub", "workspace/jepa_steering/vendor/jepa-wms"]
EVIDENCE = ["pusht-coupling-code-20260908-v2", "pusht-coupling-evidence-20260908-v2",
    "pusht-coupling-behavior-20260908-v2/freeze", "pusht-planning-code-20260908-v1",
    "pusht-planning-assets-20260908-v3", "pusht-planning-assets-20260908-v2/downloads/models",
    "pusht-planning-native-20260908-v1/freeze", "pusht-planning-native-20260908-v1/engineering",
    "pusht-planning-native-20260908-v1/simulator-check"]


def main():
    rows = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"], text=True))
    if sum(r["instance"]["totalHour"] for r in rows) > 7:
        raise ValueError("Budget exceeded")
    workers = {r["id"]: r for r in rows}
    key = "/tmp/jepa_vast_50123620_ed25519"
    def connection(number):
        r = workers[number]
        if r["actual_status"] != "running" or not r["geolocation"].endswith(", US"):
            raise ValueError("Expected owned running US worker")
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=6",
            "-p", str(r["ports"]["22/tcp"][0]["HostPort"]), "root@" + r["public_ipaddr"]]
    if workers[DESTINATION]["label"] != "jepa-pusht-coupling-us-v2":
        raise ValueError("Wrong destination lease")
    source, destination = connection(SOURCE), connection(DESTINATION)
    env = dict(os.environ)
    agent = subprocess.check_output(["ssh-agent", "-s"], text=True)
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        env[name] = re.search(name + r"=([^;]+);", agent).group(1)
    try:
        subprocess.run(["ssh-add", key], env=env, check=True, stdout=subprocess.DEVNULL)
        # Only completed native streams are eligible. Later streams are copied by
        # an independent coordinator when DONE/report/episode hashes verify.
        command = "import pathlib,json; r=pathlib.Path('/workspace/jepa-runtime/pusht-planning-native-20260908-v1/native'); print(json.dumps([str(p.relative_to('/')) for p in sorted(r.glob('shard-*')) if (p/'DONE.json').is_file()]))"
        streams = json.loads(subprocess.check_output(source + [shlex.join(["python", "-c", command])], env=env, text=True))
        paths = RUNTIME + ["workspace/jepa-runtime/" + p for p in EVIDENCE] + streams
        subprocess.run(destination + [" && ".join("test ! -e " + shlex.quote("/" + p) for p in paths)], env=env, check=True)
        transfer = "set -o pipefail; nice -n 10 tar --exclude=__pycache__ --exclude='._*' -C / -cf - "
        transfer += " ".join(map(shlex.quote, paths))
        transfer += " | nice -n 10 gzip -1 | " + shlex.join(destination + ["set -o pipefail; gzip -d | tar --keep-old-files -C / -xf -"])
        subprocess.run(source[:1] + ["-A"] + source[1:] + [transfer], env=env, check=True, timeout=1500)
        print(json.dumps({"status": "runtime_and_initial_evidence_staged_not_gpu_validated",
            "source": SOURCE, "destination": DESTINATION, "paths": paths}), flush=True)
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, stdout=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    main()
