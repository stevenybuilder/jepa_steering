"""Audit archived input identity metadata without opening tensors or scoring outcomes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
SOURCE = 'cde8274dad2bbc91efea3c5baba2e2266217949f9993aa77ce695d3bbabd4d82'
BANK = 'artifacts/offline_study/fresh-simulator-banks-20260912-v1'
ARMS = {'native', 'native_repeat', 'fixed_rank4', 'matched_random_fixed_rank4',
        'coupling_only', 'matched_random_coupling', 'joint', 'visual_only',
        'action_condition_only'}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def audit(root, archive):
    metadata = {}
    with tarfile.open(archive) as source:
        manifest = json.load(source.extractfile('SNAPSHOT_MANIFEST.json'))
        members = {row['path']: row['sha256'] for row in manifest['members']}
        for name in members:
            p = Path(name)
            if (name.startswith('fresh-freeze-v2/') or p.name == 'STARTED.json'
                    or (p.suffix == '.json' and p.stem in ARMS)):
                payload = source.extractfile(name).read()
                if digest(payload) != members[name]:
                    raise ValueError('Archived metadata differs from verified member manifest')
                metadata[name] = json.loads(payload)
    protocol = metadata['fresh-freeze-v2/protocol.json']
    if (members['fresh-freeze-v2/protocol.json'] != FREEZE
            or metadata['fresh-freeze-v2/FROZEN.json']['protocol_sha256'] != FREEZE
            or protocol['source_sha256'] != SOURCE):
        raise ValueError('Unexpected frozen protocol/source identity')
    bank_hashes = {}
    for name in ('protocol.json', 'FROZEN.json'):
        path = BANK + '/' + name
        actual = digest((root / path).read_bytes())
        if actual != protocol['files_sha256'][path]:
            raise ValueError('Bank metadata changed')
        bank_hashes[name] = actual
    if json.loads((root / BANK / 'FROZEN.json').read_text())['protocol_sha256'] != bank_hashes['protocol.json']:
        raise ValueError('Bank freeze does not bind its protocol')
    starts, scientific, engineering, tensor_references = {}, 0, 0, set()
    for name, identity in metadata.items():
        if Path(name).name != 'STARTED.json':
            continue
        role = 'excluded_engineering' if identity['engineering'] else 'scientific_candidates'
        rows = protocol['tasks'][identity['task']]['records']
        expected = next(row for row in rows if row['role'] == role
                        and row['episode'] == identity['scenario']['episode'])
        if identity['scenario'] != expected or identity['freeze_sha256'] != FREEZE:
            raise ValueError('STARTED identity is not its exact frozen scenario')
        if protocol['files_sha256'][expected['tensor_path']] != expected['tensor_sha256']:
            raise ValueError('Scenario tensor reference is not hash-bound')
        starts[str(Path(name).parent)] = identity
        tensor_references.add(expected['tensor_path'])
    for name, record in metadata.items():
        if Path(name).suffix != '.json' or Path(name).stem not in ARMS:
            continue
        identity = starts[str(Path(name).parent)]
        expected = identity['scenario']
        if (record['scenario'] != expected or record['freeze_sha256'] != FREEZE
                or record['device_uuid'] != identity['device_uuid']
                or record['scientific_efficacy_measurement'] != (not identity['engineering'])
                or any(record['result'][key] != expected[key] for key in ('initial_sha256', 'goal_sha256'))):
            raise ValueError('Published arm input/device/role differs from STARTED and freeze')
        if identity['engineering']:
            engineering += 1
        else:
            scientific += 1
    return {'verified_utc': datetime.now(timezone.utc).isoformat(), 'archive': str(archive),
            'protocol_sha256': FREEZE, 'bound_source_sha256': SOURCE,
            'bank_metadata_sha256': bank_hashes, 'started_identities_verified': len(starts),
            'published_scientific_input_bindings_verified': scientific,
            'published_engineering_input_bindings_verified': engineering,
            'distinct_frozen_tensor_references_verified': len(tensor_references),
            'all_checked_bindings_match': True, 'tensor_or_model_files_rehashed': False,
            'scientific_source_rehashed': False, 'efficacy_values_not_reported': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.archive)))
