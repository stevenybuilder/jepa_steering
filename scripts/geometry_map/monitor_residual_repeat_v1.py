"""Monitor/copy already-launched repeat jobs; never launch or stop GPU work."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def inspect(row):
    code = """import json
from pathlib import Path
p=Path(ROOT); out={'episode':EP,'done':False,'failed':False}
if (p/'results-v1'/'DONE.json').exists():out['done']=True
if (p/'results-v1'/'FAILED.json').exists():out['failed']=True
q=p/'results-v1'/'progress.json'
out['completed_arms']=len(json.loads(q.read_text())['rows']) if q.exists() else 0
lines=(p/'job-v1.log').read_text().splitlines()
events=[json.loads(s) for s in lines if s.startswith('{"event":')]
if events:
 e=events[-1];out['event']={k:v for k,v in e.items() if k!='metrics'}
 if 'metrics' in e:out['event'].update(arm=e['metrics']['arm'],raw_steps=e['metrics'].get('raw_steps'))
print(json.dumps(out))
""".replace("ROOT", repr(row["root"])).replace("EP", repr(row["episode"]))
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=12",
           "-p", str(row["port"]), "root@"+row["host"], "/opt/conda/bin/python -c "+shlex.quote(code)]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=35)
        return json.loads(out)
    except Exception as exc:
        return {"episode": row["episode"], "inspect_error": str(exc)}


def preserve(row, script_root, failed):
    command = [sys.executable, str(script_root/"archive_residual_search_v1.py"),
               "--source-host", row["host"], "--source-port", str(row["port"]),
               "--source-root", row["root"]+"/results-v1", "--target-host", row["backup_host"],
               "--target-port", str(row["backup_port"]), "--target-root", row["backup_root"],
               "--local-root", row["local_root"]]
    if failed:
        command.append("--allow-failed")
    done = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(done.stdout[-4000:])
    return json.loads((Path(row["local_root"])/"REMOTE_BACKUP_VERIFIED.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); rows = config["workers"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pending, verified, failed = {}, {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as checks, concurrent.futures.ThreadPoolExecutor(max_workers=2) as copies:
        while len(verified)+len(failed) < len(rows):
            states = list(checks.map(inspect, rows))
            for row, state in zip(rows, states):
                key = str(row["episode"])
                if (state.get("done") or state.get("failed")) and key not in pending and key not in verified and key not in failed:
                    pending[key] = copies.submit(preserve, row, Path(__file__).parent, state.get("failed", False))
                if key in pending and pending[key].done():
                    try:
                        receipt = pending.pop(key).result()
                        (verified if receipt["complete"] else failed)[key] = receipt
                    except Exception as exc:
                        failed[key] = {"copy_error": str(exc)}
                        pending.pop(key, None)
            status = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "states": states,
                      "copying": sorted(pending), "verified": sorted(verified), "failed": failed}
            (args.output_dir/"PROGRESS.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            if len(verified)+len(failed) < len(rows):
                time.sleep(40)
    if failed:
        (args.output_dir/"NEEDS_ATTENTION.json").write_text(json.dumps(failed, indent=2))
        raise RuntimeError("Run or backup failed; sources preserved, no automatic experimental restart")
    summary = args.output_dir/"repeat_summary.json"
    subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_residual_search_physics.py")),
                    "--inputs", *[str(Path(r["local_root"])/"full_episodes.json") for r in rows],
                    "--episodes", "8", "9", "10", "11", "--seen-development-repeat", "--output", str(summary)], check=True)
    receipt = {"complete": True, "source_and_second_host_all_verified": True, "workers": verified,
               "summary": str(summary), "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
               "monitor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir/"DONE.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__": main()
