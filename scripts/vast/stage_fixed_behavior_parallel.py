"""Finish a preserved partial runtime copy using four disjoint file streams.

Run locally; only ownedNebraska50231985 to ownedCalifornia50233992. No GPU work,
private-key copying, delete/inplace, or changes to scientific protocols.
"""
import concurrent.futures
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from stage_fixed_response_worker import PATHS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=int, choices=(50233992,50239185), default=50233992)
    args = parser.parse_args()
    rows = {r['id']: r for r in json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))}
    if sum(r['instance']['totalHour'] for r in rows.values()) > 7:
        raise ValueError('Current budget exceeded')
    def endpoint(number, label):
        r = rows[number]
        if r['actual_status'] != 'running' or not r['geolocation'].endswith(', US') or r['label'] != label:
            raise ValueError('Wrong leased US staging endpoint')
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=6',
                '-p', str(r['ports']['22/tcp'][0]['HostPort']), 'root@' + r['public_ipaddr']]
    source = endpoint(50231985, 'jepa-fixed-offline-us-v1')
    destination = endpoint(args.destination, {50233992:'jepa-fixed-behavior-us-v1',
        50239185:'jepa-pusht-wall-history-us-v1'}[args.destination])
    agent = subprocess.check_output(['ssh-agent', '-s'], text=True)
    env = dict(os.environ)
    for name in ('SSH_AUTH_SOCK', 'SSH_AGENT_PID'):
        env[name] = re.search(name + r'=([^;]+);', agent).group(1)
    try:
        subprocess.run(['ssh-add', '/tmp/jepa_vast_50123620_ed25519'], env=env, check=True, stdout=subprocess.DEVNULL)
        source = source[:1] + ['-A', '-i', '/tmp/jepa_vast_50123620_ed25519'] + source[1:]
        inventory = '''import json,stat
from pathlib import Path
roots = ''' + repr(PATHS if args.destination == 50233992 else PATHS[:-1]) + '''
rows=[]
for name in roots:
 root=Path('/')/name
 if not root.exists(): raise ValueError('Missing source root: '+name)
 for p in ([root] if root.is_file() else root.rglob('*')):
  if '__pycache__' in p.parts or p.name.startswith('._'): continue
  if p.is_file() or p.is_symlink(): rows.append((str(p.relative_to('/')),p.lstat().st_size))
print(json.dumps(rows))
'''
        command = shlex.join(['/workspace/jepa-python/bin/python', '-c', inventory])
        files = json.loads(subprocess.check_output(source + [command], env=env, text=True))
        groups, sizes = [[] for _ in range(4)], [0] * 4
        for name, size in sorted(files, key=lambda r: (-r[1], r[0])):
            index = min(range(4), key=lambda i: sizes[i])
            groups[index].append(name); sizes[index] += size
        if len({p for group in groups for p in group}) != len(files):
            raise ValueError('Overlapping staging files')
        print(json.dumps({'stage': 'four_disjoint_runtime_streams', 'files': len(files), 'bytes_per_stream': sizes}), flush=True)
        def copy(index):
            remote = shlex.join(['nice', '-n', '10', 'rsync', '-aRz', '--checksum', '--partial',
                '--timeout=120', '--from0', '--files-from=-', '-e', shlex.join(destination[:-1]),
                '/', destination[-1] + ':/'])
            result = subprocess.run(source + [remote], env=env,
                input=b'\0'.join(p.encode() for p in sorted(groups[index])) + b'\0',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
            if result.returncode:
                raise RuntimeError('Preserved partial stream failed: ' + result.stderr.decode()[-2000:])
            print(json.dumps({'completed_stream': index, 'files': len(groups[index])}), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(copy, range(4)))
        print(json.dumps({'status': 'parallel_runtime_copy_complete_not_gpu_readiness', 'destination': args.destination}), flush=True)
    finally:
        subprocess.run(['ssh-agent', '-k'], env=env, stdout=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    main()
