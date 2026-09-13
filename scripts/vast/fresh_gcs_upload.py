"""Give one owned receiver a single-object GCS upload session, never account keys."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import urllib.parse
import urllib.request


REMOTE = '''import hashlib,json,sys,urllib.request
from pathlib import Path
request=json.loads(sys.stdin.read()); path=Path(request['path'])
h=hashlib.sha256()
with path.open('rb') as source:
 for block in iter(lambda:source.read(1048576),b''):h.update(block)
if h.hexdigest()!=request['sha256']:raise ValueError('Local archive hash mismatch')
with path.open('rb') as source:
 req=urllib.request.Request(request['session'],data=source,method='PUT',headers={'Content-Length':str(path.stat().st_size),'Content-Type':'application/octet-stream'})
 with urllib.request.urlopen(req,timeout=600) as response:result=json.load(response)
print(json.dumps({'bucket':result['bucket'],'name':result['name'],'generation':result['generation'],'size':result['size'],'source_sha256':h.hexdigest()}))
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',required=True)
    parser.add_argument('--known-hosts',type=Path,required=True)
    parser.add_argument('--remote-file',required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--object',required=True)
    args=parser.parse_args()
    if not args.object.startswith('fresh-campaign-20260912-v2/'):
        raise ValueError('Outside authorized fresh campaign prefix')
    token=subprocess.check_output(['gcloud','auth','print-access-token'],text=True).strip()
    url='https://storage.googleapis.com/upload/storage/v1/b/rgt-jepa-archive-2026/o?'+urllib.parse.urlencode({'uploadType':'resumable','name':args.object,'ifGenerationMatch':'0'})
    req=urllib.request.Request(url,data=b'{}',method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','X-Upload-Content-Type':'application/octet-stream'})
    with urllib.request.urlopen(req,timeout=30) as response:session=response.headers['Location']
    # Keep the scoped capability out of arguments, logs, and the local filesystem.
    ssh=['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(args.known_hosts),'-p',args.port,'root@'+args.host,shlex.join(['python','-c',REMOTE])]
    subprocess.run(ssh,input=json.dumps({'session':session,'path':args.remote_file,'sha256':args.sha256}),text=True,check=True,timeout=660)
