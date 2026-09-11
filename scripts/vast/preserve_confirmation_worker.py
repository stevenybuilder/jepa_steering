"""Package all bounded-worker research inputs/results/logs; no lifecycle calls."""
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    args = p.parse_args()
    workspace = Path('/workspace')
    files = []
    for root in (workspace/'confirmation', workspace/'confirmation-output'):
        if root.exists():
            files += [x for x in root.rglob('*') if x.is_file() and
                not any(part in ('.git','__pycache__') or part.startswith('._') for part in x.relative_to(root).parts)]
    files += [x for x in workspace.glob('confirmation-*.log') if x.is_file()]
    files += [x for x in (workspace/'confirmation-packages.txt', workspace/'confirmation-inputs.tar.gz',
                          workspace/'confirmation-launch-exit.json') if x.is_file()]
    if any(x.is_symlink() for x in files): raise ValueError('Unexpected research symlink')
    members = {str(x.relative_to(workspace)): {'bytes':x.stat().st_size,'sha256':digest(x)} for x in files}
    payload = json.dumps({'members':members,'omitted':'rebuildable environment, published checkpoint/DINO cache, vendor git history'},sort_keys=True).encode()
    with tarfile.open(args.archive,'x:gz') as out:
        entry = tarfile.TarInfo('PRESERVATION_MANIFEST.json'); entry.size=len(payload)
        out.addfile(entry,io.BytesIO(payload))
        for path in files: out.add(path,arcname=str(path.relative_to(workspace)),recursive=False)
    for path in files:
        if digest(path)!=members[str(path.relative_to(workspace))]['sha256']:
            raise ValueError('Research source changed during packaging')
    print(json.dumps({'archive':str(args.archive),'bytes':args.archive.stat().st_size,'sha256':digest(args.archive),'members':len(members)}))


if __name__ == '__main__':
    main()
