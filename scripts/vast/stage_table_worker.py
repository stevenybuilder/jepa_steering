"""Stage small archive proofs and start restoration/bootstrap on owned50561030.

Large inputs never pass through laptop storage. Only an expiring Drive access
token reaches the worker process; no refresh/provider credentials are copied.
"""
import configparser
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "artifacts/offline_study/table-completion-20260911-v1"
REMOTE = "/workspace/table-completion-20260911-v1"


def main():
    from restore_table_evidence import TARGETS, BASE
    lease = json.loads((ROOT / "REPLACEMENT_LEASE.json").read_text())
    rows = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"], text=True))
    row = next(r for r in rows if r["id"] == lease["instance"])
    if (row["id"] != 50561030 or row["actual_status"] != "running" or
            row["label"] != lease["label"] or row["geolocation"] != lease["geolocation"]):
        raise ValueError("Receiving lease is not owned/running US worker")
    ssh = ["ssh", "-i", "/Users/stevenyang/.ssh/id_ed25519", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-p", str(row["ports"]["22/tcp"][0]["HostPort"]),
        "root@" + row["public_ipaddr"]]
    paths = [BASE / str(i) / batch / "DRIVE_VERIFIED.json" for i, batch in TARGETS]
    paths += [PROJECT / "artifacts/offline_study/pusht-native-recovery-20260908-v1/PUBLIC_INPUTS.json"]
    with subprocess.Popen(ssh + ["tar --keep-old-files -C " + REMOTE + "/code -xf -"],
                          stdin=subprocess.PIPE) as remote:
        with tarfile.open(fileobj=remote.stdin, mode="w|") as archive:
            for path in paths:
                archive.add(path, arcname=path.relative_to(PROJECT).as_posix(), recursive=False)
        remote.stdin.close()
        if remote.wait():
            raise RuntimeError("Proof staging failed")
    # Existing auth refresh is local; nothing is printed or written into proofs.
    subprocess.run(["rclone", "about", "gdrive:", "--json"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read("/Users/stevenyang/.config/rclone/rclone.conf")
    access = json.loads(cfg["gdrive"]["token"])["access_token"]
    remote_code = '''import json,os,subprocess,sys
from pathlib import Path
root=Path("/workspace/table-completion-20260911-v1")
data=json.load(sys.stdin)
env=dict(os.environ,JEPA_DRIVE_ACCESS_TOKEN=data["access"],JEPA_TABLE_RESTORE_ROOT=str(root/"restored"))
jobs={}
with (root/"restore.log").open("x") as log:
 p=subprocess.Popen(["python3","-u",str(root/"code/scripts/vast/restore_table_evidence.py"),"--workers","2"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs["restore_pid"]=p.pid
with (root/"bootstrap.log").open("x") as log:
 p=subprocess.Popen(["bash",str(root/"code/scripts/vast/bootstrap_table_completion.sh")],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs["bootstrap_pid"]=p.pid
with (root/"SETUP_LAUNCH.json").open("x") as out:json.dump(jobs,out)
print(json.dumps(jobs))
'''
    result = subprocess.check_output(ssh + [shlex.join(["python3", "-c", remote_code])],
        input=json.dumps({"access": access}), text=True, timeout=60)
    with (ROOT / "SETUP_LAUNCH.json").open("x") as stream:
        json.dump({"instance": row["id"], "remote_response": result,
                   "scientific_episodes_launched": False}, stream, indent=2)
    print(result, flush=True)


if __name__ == "__main__":
    main()
