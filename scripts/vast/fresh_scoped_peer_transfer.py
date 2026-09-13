"""One authorized read-only archive capability; no account/user private keys copied."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
KEY = '/Users/stevenyang/.ssh/id_ed25519'
BASE = ROOT / 'artifacts/offline_study/fresh-campaign-stage-20260912-v1'
SOURCE = (50827072, 'ssh8.vast.ai', 27072)
DEST = (50833790, 'ssh1.vast.ai', 33790)
ARCHIVE = '/workspace/prepared-runtime-inputs-v2.tgz'
EXPECTED = 'd3712d0a01c7eb19e3f2cf115d33516e00e4ff229d0c6bad8bf1edd5db7d78cf'
STAMP = str(int(time.time()))
REMOTE_KEY = '/tmp/fresh-peer-readonly-' + STAMP
OUTPUT = '/workspace/peer-runtime-' + STAMP + '.tgz'


def remote(host, code, timeout=30):
    instance, address, port = host
    command = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=6', '-o', 'UserKnownHostsFile=' + str(BASE / f'receiving-{instance}/known_hosts'),
               '-p', str(port), 'root@' + address, 'python3 -']
    if timeout > 100:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        process.stdin.write(code)
        process.stdin.close()
        lines = []
        for output_line in process.stdout:
            print(output_line.rstrip(), flush=True)
            lines.append(output_line)
        if process.wait():
            raise RuntimeError('Peer receiver failed: ' + ''.join(lines)[-1000:])
        return ''.join(lines)
    result = subprocess.run(command, input=code, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{instance}: {result.stderr[-1000:]} {result.stdout[-1000:]}')
    return result.stdout


def main():
    source_check = remote(SOURCE, f'import hashlib,json\np={ARCHIVE!r}\nh=hashlib.sha256()\nf=open(p,"rb")\nfirst=f.read(4194304);h.update(first)\nfor b in iter(lambda:f.read(8388608),b""):h.update(b)\nprint(json.dumps({{"sha":h.hexdigest(),"probe_sha":hashlib.sha256(first).hexdigest()}}))\n', 45)
    check = json.loads(source_check)
    assert check['sha'] == EXPECTED
    print(json.dumps({'source_sha_verified': True}), flush=True)
    pub = remote(DEST, f'import subprocess,pathlib\nsubprocess.run(["ssh-keygen","-q","-t","ed25519","-N","","-f",{REMOTE_KEY!r}],check=True)\nprint(pathlib.Path({(REMOTE_KEY + ".pub")!r}).read_text().strip())\n').strip()
    assert pub.startswith('ssh-ed25519 ')
    expiry = int(time.time()) + 600
    forced = f'/bin/sh -c \'test "$(date +%s)" -le {expiry} && exec /bin/cat {ARCHIVE}\''
    line = 'restrict,command="' + forced.replace('\\', '\\\\').replace('"', '\\"') + '" ' + pub
    source_known = (BASE / 'receiving-50827072/known_hosts').read_text().splitlines()
    keys = [s.split()[1:3] for s in source_known if s and not s.startswith('#')]
    assert keys
    known = '\n'.join(f'[{h}]:{p} {kind} {key}' for h,p in [('137.175.22.196',11363),('ssh8.vast.ai',27072)] for kind,key in keys) + '\n'
    installed = False
    try:
        remote(SOURCE, f'from pathlib import Path\np=Path("/root/.ssh/authorized_keys")\ns=p.read_text()\nline={line!r}\nassert line not in s.splitlines()\np.write_text(s+("" if s.endswith("\\n") else "\\n")+line+"\\n")\nprint("CAPABILITY_INSTALLED")\n')
        installed = True
        remote_code = f'''
import subprocess,pathlib,json,time,hashlib
key={REMOTE_KEY!r}; output={OUTPUT!r}
known=key+'.known_hosts';pathlib.Path(known).write_text({known!r})
prefix=['ssh','-i',key,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=6','-o','UserKnownHostsFile='+known]
chosen=None
for host,port in [('137.175.22.196',11363),('ssh8.vast.ai',27072)]:
 cmd=prefix+['-p',str(port),'root@'+host,'read-archive']
 start=time.monotonic()
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 import selectors
 s=selectors.DefaultSelector();s.register(p.stdout,selectors.EVENT_READ);data=bytearray()
 while len(data)<4194304 and time.monotonic()-start<25:
  if not s.select(timeout=1):continue
  b=__import__('os').read(p.stdout.fileno(),min(65536,4194304-len(data)))
  if not b:break
  data.extend(b)
 p.terminate()
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:p.kill();p.wait()
 elapsed=time.monotonic()-start
 ok=len(data)==4194304 and hashlib.sha256(data).hexdigest()=={check['probe_sha']!r}
 print(json.dumps({{'probe_host':host,'bytes':len(data),'elapsed':elapsed,'sha_pass':ok}}),flush=True)
 if ok and elapsed<2.1:chosen=cmd;break
if chosen is None:raise SystemExit('No peer route exceeds approximately2MB/s; no full writer started')
start=time.monotonic()
with open(output,'xb') as f:r=subprocess.run(chosen,stdout=f,stderr=subprocess.PIPE,timeout=420)
assert r.returncode==0,r.stderr.decode()[-500:]
h=hashlib.sha256()
with open(output,'rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()=={EXPECTED!r},h.hexdigest()
print(json.dumps({{'full_sha_pass':True,'output':output,'bytes':pathlib.Path(output).stat().st_size,'elapsed':time.monotonic()-start}}),flush=True)
'''
        print(remote(DEST, remote_code, 490), flush=True)
    finally:
        if installed:
            print(remote(SOURCE, f'from pathlib import Path\np=Path("/root/.ssh/authorized_keys")\nlines=p.read_text().splitlines(keepends=True)\ntarget={line!r}\nassert sum(x.rstrip("\\r\\n")==target for x in lines)==1\np.write_text("".join(x for x in lines if x.rstrip("\\r\\n")!=target))\nprint("EXACT_CAPABILITY_REMOVED")\n'), flush=True)
        print(remote(DEST, f'from pathlib import Path\nfor p in [{REMOTE_KEY!r},{(REMOTE_KEY + ".pub")!r},{(REMOTE_KEY + ".known_hosts")!r}]:Path(p).unlink(missing_ok=True)\nprint("EPHEMERAL_KEY_REMOVED")\n'), flush=True)


if __name__ == '__main__':
    main()
