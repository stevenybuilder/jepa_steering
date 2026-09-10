"""CPU/network-only Virginia-to-Indiana pinned PointMaze training preparation."""
import os
import re
import shlex
import subprocess

KEY='/tmp/jepa_vast_50123620_ed25519'


def main():
    def ssh(host,port):
        return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
            '-o','StrictHostKeyChecking=accept-new','-p',str(port),'root@'+host]
    source=ssh('98.86.102.84',40083); target=ssh('99.22.20.240',17755)
    agent=subprocess.check_output(['ssh-agent','-s'],text=True); env=dict(os.environ)
    for name in ('SSH_AUTH_SOCK','SSH_AGENT_PID'):
        env[name]=re.search(name+r'=([^;]+);',agent).group(1)
    try:
        subprocess.run(['ssh-add',KEY],env=env,check=True,stdout=subprocess.DEVNULL)
        source=source[:1]+['-A','-i',KEY]+source[1:]
        direct=target[:1]+['-i',KEY]+target[1:]
        root='/workspace/jepa-runtime/pointmaze-history-transfer-20260908-v1'
        subprocess.run(direct+['test ! -e '+root+' && mkdir '+root],env=env,check=True)
        names=['navigation-assets-20260907-v1/'+n for n in ('protocol.json','report.json','DONE.json',
            'downloads/dataset/point_maze/point_maze.zip')]
        names+=['navigation-input-check-20260907-v1','pointmaze-training-inputs-20260908-v1',
                'pointmaze-history-code-20260908-v1']
        tar=shlex.join(['tar','-C','/workspace/jepa-runtime','--exclude=__pycache__','--exclude=._*',
            '-czf','-']+names)
        subprocess.run(source+['set -o pipefail; '+tar+' | '+shlex.join(target+['tar -xzf - -C '+root])],
            env=env,check=True,timeout=900)
        subprocess.run(direct+['nohup /workspace/jepa-planning-python/bin/python -u '
            '/workspace/jepa-runtime/receive_pointmaze_history_inputs_v1.py > '
            '/workspace/jepa-runtime/pointmaze-history-receiving-20260908-v1.log 2>&1 < /dev/null &'],env=env,check=True)
    finally:
        subprocess.run(['ssh-agent','-k'],env=env,stdout=subprocess.DEVNULL,check=False)


if __name__=='__main__': main()
