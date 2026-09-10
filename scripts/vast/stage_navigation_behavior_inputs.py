"""CPU/network-only copies between explicitly leased US workers, no GPU jobs."""
import json
import os
import re
import shlex
import subprocess


def main():
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    if sum(r['instance']['totalHour'] for r in rows) > 7:
        raise ValueError('Aggregate budget exceeded')
    rows = {r['id']: r for r in rows}
    def connection(number, label):
        row = rows[number]
        if row['label'] != label or row['actual_status'] != 'running' or not row['geolocation'].endswith(', US'):
            raise ValueError('Only the specifically leased running US worker is allowed')
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=6',
                '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    destination = connection(50231985, 'jepa-fixed-offline-us-v1')
    source = connection(50205763, 'jepa-navigation-offline-indiana')
    maze = connection(50189244, 'jepa-us-droid-history-20260907')
    groups = [(source, [f'workspace/jepa-runtime/navigation-assets-20260907-v1/downloads/model/jepa_wm_{task}.pth.tar'
                        for task in ('wall', 'pointmaze')]),
              (maze, ['workspace/jepa-maze-python', 'root/.mujoco/mujoco210'])]
    agent = subprocess.check_output(['ssh-agent', '-s'], text=True)
    env = dict(os.environ)
    for name in ('SSH_AUTH_SOCK', 'SSH_AGENT_PID'):
        env[name] = re.search(name + r'=([^;]+);', agent).group(1)
    try:
        subprocess.run(['ssh-add', '/tmp/jepa_vast_50123620_ed25519'], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        for origin, paths in groups:
            subprocess.run(destination + [' && '.join('test ! -e ' + shlex.quote('/' + p) for p in paths)],
                           env=env, check=True)
            receive = 'set -o pipefail; gzip -d | tar -C / -xf -'
            remote = "set -o pipefail; nice -n 10 tar --exclude=__pycache__ --exclude='._*' -C / -cf - "
            remote += ' '.join(map(shlex.quote, paths)) + ' | nice -n 10 gzip -1 | '
            remote += shlex.join(destination + [receive])
            subprocess.run(origin[:1] + ['-A', '-i', '/tmp/jepa_vast_50123620_ed25519'] + origin[1:] + [remote],
                           env=env, check=True, timeout=900)
            print(json.dumps({'copied_paths': paths, 'verified_for_execution': False}), flush=True)
    finally:
        subprocess.run(['ssh-agent', '-k'], env=env, stdout=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    main()
