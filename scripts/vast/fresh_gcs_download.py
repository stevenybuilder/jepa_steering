"""Direct receiver download using a read-only, single-object downscoped token."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import urllib.error
import urllib.parse
import urllib.request

BUCKET = 'rgt-jepa-archive-2026'
OBJECT = 'fresh-campaign-20260912-v2/prepared-runtime-inputs-v2.tgz'
SHA = 'd3712d0a01c7eb19e3f2cf115d33516e00e4ff229d0c6bad8bf1edd5db7d78cf'
GENERATION = '1789246618367401'
URL = 'https://storage.googleapis.com/download/storage/v1/b/' + BUCKET + '/o/' + urllib.parse.quote(OBJECT, safe='') + '?alt=media&generation=' + GENERATION


def token():
    original = subprocess.check_output(['gcloud', 'auth', 'print-access-token'], text=True).strip()
    boundary = {'accessBoundary': {'accessBoundaryRules': [{
        'availableResource': '//storage.googleapis.com/projects/_/buckets/' + BUCKET,
        'availablePermissions': ['inRole:roles/storage.objectViewer'],
        'availabilityCondition': {'expression': "resource.name == 'projects/_/buckets/" + BUCKET + "/objects/" + OBJECT + "'"}}]}}
    fields = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange',
              'subject_token_type': 'urn:ietf:params:oauth:token-type:access_token',
              'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token',
              'subject_token': original, 'options': json.dumps(boundary)}
    request = urllib.request.Request('https://sts.googleapis.com/v1/token',
                                     data=urllib.parse.urlencode(fields).encode(), method='POST')
    with urllib.request.urlopen(request, timeout=30) as response:
        scoped = json.load(response)['access_token']
    # Verify the restriction before exposing even this limited token to a receiver.
    request = urllib.request.Request(URL, headers={'Authorization': 'Bearer ' + scoped, 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or len(response.read()) != 1:
            raise ValueError('Scoped runtime range check failed')
    denied = URL.replace(urllib.parse.quote(OBJECT, safe=''), urllib.parse.quote('fresh-campaign-20260912-v2/fresh-method-v2-source-freeze.tgz', safe='')).split('&generation=')[0]
    try:
        urllib.request.urlopen(urllib.request.Request(denied, headers={'Authorization': 'Bearer ' + scoped}), timeout=30)
    except urllib.error.HTTPError as error:
        if error.code != 403:
            raise
    else:
        raise ValueError('Downscoped token unexpectedly accessed another object')
    print(json.dumps({'scope_verified': True, 'allowed_object': OBJECT, 'other_object_denied': True}), flush=True)
    return scoped


REMOTE = '''import hashlib,json,sys,time,urllib.request
from pathlib import Path
request=json.loads(sys.stdin.read()); archive=Path('/workspace/prepared-runtime-inputs-v2.tgz')
start=last=time.monotonic(); count=0; digest=hashlib.sha256()
req=urllib.request.Request(request['url'],headers={'Authorization':'Bearer '+request['token']})
with urllib.request.urlopen(req,timeout=90) as response,archive.open('xb') as output:
 while True:
  block=response.read(1048576)
  if not block:break
  output.write(block);digest.update(block);count+=len(block)
  if time.monotonic()-last>10:
   print(json.dumps({'downloaded_bytes':count,'seconds':time.monotonic()-start}),flush=True);last=time.monotonic()
if digest.hexdigest()!=request['sha256']:raise ValueError('Runtime archive hash mismatch')
print(json.dumps({'archive':str(archive),'bytes':count,'sha256':digest.hexdigest(),'seconds':time.monotonic()-start,'download_verified':True}),flush=True)
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--port')
    parser.add_argument('--known-hosts',type=Path)
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    scoped=token()
    if not args.test_only:
        if not all((args.host,args.port,args.known_hosts)):
            parser.error('Host, port and verified known-hosts required')
        ssh=['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(args.known_hosts),'-p',args.port,'root@'+args.host,shlex.join(['python','-c',REMOTE])]
        subprocess.run(ssh,input=json.dumps({'token':scoped,'url':URL,'sha256':SHA}),text=True,check=True,timeout=660)
