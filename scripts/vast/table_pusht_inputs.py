"""Download the pinned release once; extract only exact historical planner inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT=Path('/workspace/table-completion-20260911-v1')
EXPECTED='442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    proof=json.loads((ROOT/'code/artifacts/offline_study/pusht-native-recovery-20260908-v1/PUBLIC_INPUTS.json').read_text())
    path=ROOT/'downloads/pusht_noise.zip';path.parent.mkdir(exist_ok=True)
    if not path.exists():
        url='https://huggingface.co/datasets/facebook/jepa-wms/resolve/6116f042ae7ae4c8e3f1fd2f194f432615664182/pusht/pusht_noise.zip?download=1'
        partial=path.with_suffix('.partial')
        with urllib.request.urlopen(url,timeout=120) as r,partial.open('xb') as out:
            shutil.copyfileobj(r,out,4<<20)
        assert partial.stat().st_size==2785304515 and sha(partial)==EXPECTED
        partial.rename(path)
    assert path.stat().st_size==2785304515 and sha(path)==EXPECTED
    verified={}
    with zipfile.ZipFile(path) as z:
        for historical,spec in proof.items():
            if '/source/data/' not in historical:continue
            member=historical.split('/source/data/',1)[1]
            assert not Path(member).is_absolute() and '..' not in Path(member).parts
            target=ROOT/'data'/member;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with z.open(member) as r,target.open('xb') as out:shutil.copyfileobj(r,out,4<<20)
            assert target.stat().st_size==spec['bytes'] and sha(target)==spec['sha256']
            verified[member]=spec
    with (ROOT/'PUSHT_INPUTS_VERIFIED.json').open('x') as f:
        json.dump({'archive_sha256':EXPECTED,'files':verified,'scientific_episodes':0},f,indent=2)
    print(json.dumps({'pusht_exact_input_files':len(verified),'ready_for_receiving_checks':True}),flush=True)


if __name__=='__main__':main()
