"""Copy only unchanged inputs actually read by Push-T fit checks and planning.

The full archive stays on Indiana. Exclude unused precomputed tokens/train videos;
retain all validation videos, full state/action arrays, and the one fit-check video.
"""
import json
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

KEY = "/tmp/jepa_vast_50123620_ed25519"
SOURCE = "/workspace/jepa-runtime/pusht-planning-assets-20260908-v2"
TARGET = "/workspace/jepa-runtime/pusht-planning-assets-20260908-v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    def endpoint(port, host):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), "root@" + host]
    source = endpoint(17755, "99.22.20.240")
    dest = endpoint(50578, "71.104.167.38")
    agent = subprocess.check_output(["ssh-agent", "-s"], text=True)
    env = dict(os.environ)
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        env[name] = re.search(name + r"=([^;]+);", agent).group(1)
    try:
        subprocess.run(["ssh-add", KEY], env=env, check=True, stdout=subprocess.DEVNULL)
        source = source[:1] + ["-A", "-i", KEY] + source[1:]
        script = '''import json,hashlib
from pathlib import Path
r=Path(''' + repr(SOURCE) + ''')
m=json.loads((r/'files.json').read_text()); report=json.loads((r/'report.json').read_text())
assert hashlib.sha256((r/'report.json').read_bytes()).hexdigest()==json.loads((r/'DONE.json').read_text())['report_sha256']
assert hashlib.sha256((r/'files.json').read_bytes()).hexdigest()==report['files_sha256']
assert report['archive_sha256']=='442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08'
assert report['status']=='full_pinned_pusht_inputs_verified_and_extracted'
needed={f'pusht_noise/{pool}/{name}' for pool in ('train','val') for name in ('states.pth','rel_actions.pth','velocities.pth','seq_lengths.pkl')}
needed.update(f'pusht_noise/val/obses/episode_{i:03d}.mp4' for i in range(21))
needed.add('pusht_noise/train/obses/episode_10810.mp4')
needed.update(n for n in m if n in ('pusht_noise/train/shapes.pkl','pusht_noise/val/shapes.pkl'))
assert needed<=m.keys()
print(json.dumps({'names':sorted('data/'+n for n in needed)+['protocol.json','files.json','report.json','DONE.json'],'bytes':sum(m[n]['bytes'] for n in needed),'input_files':len(needed)}))
'''
        selected = json.loads(subprocess.check_output(source + [shlex.join(["python", "-c", script])], env=env, text=True))
        print(json.dumps({"stage": "source_manifest_verified", **{k:v for k,v in selected.items() if k != "names"}}), flush=True)
        if not args.verify_only:
            subprocess.run(dest[:1] + ["-i", KEY] + dest[1:] + ["mkdir -p " + TARGET + "/source"], env=env, check=True)
        remote = "set -o pipefail; tar -C " + SOURCE + " --null -cf - -T - | gzip -1 | " + shlex.join(
            dest + ["set -o pipefail; gzip -d | tar -C " + TARGET + "/source -xf -"])
        if not args.verify_only:
            subprocess.run(source + [remote], env=env, input=b"\0".join(n.encode() for n in selected["names"]) + b"\0", check=True, timeout=600)
        check = '''import json,hashlib
from pathlib import Path
r=Path(''' + repr(TARGET) + ''')
m=json.loads((r/'source/files.json').read_text()); files={}
for p in (r/'source/data').rglob('*'):
 if not p.is_file(): continue
 n=str(p.relative_to(r/'source/data')); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 assert h.hexdigest()==m[n]['sha256'] and p.stat().st_size==m[n]['bytes'],n
 files[n]=m[n]
assert len(files)==''' + str(selected['input_files']) + '''
report={'status':'all_native_planning_and_fit_check_input_bytes_verified','files':files,'unused_precomputed_tokens_and_other_train_videos_not_staged':True,'full_archive_preserved_on_indiana':True,'planning_executed':False,'fresh_confirmation':False}
(r/'receiving_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+chr(10))
(r/'RECEIVING_DONE.json').write_text(json.dumps({'report_sha256':hashlib.sha256((r/'receiving_report.json').read_bytes()).hexdigest()})+chr(10))
print(json.dumps({'status':report['status'],'files':len(files)}))
'''
        subprocess.run(dest[:1] + ["-i", KEY] + dest[1:] + [shlex.join(["python", "-c", check])], env=env, check=True)
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, stdout=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    main()
