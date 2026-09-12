"""CPU-only verification of saved control directions and paired site budgets."""
import ast
import importlib.util
import json
from pathlib import Path

import torch

from audit_core_random_controls import ROOT, RAW, EVIDENCE, TASKS, digest, read, require


def main():
    read(ROOT / 'CONTROL_AUDIT_RESTORED.json')
    torch.set_num_threads(1)
    source = RAW / 'fixed-response-code-20260908-v4/src/offline_study'
    # Execute only the two archived, pure tensor helpers, avoiding simulator
    # imports and preserving the exact historical normalization implementation.
    tree = ast.parse((source / 'operator_fit.py').read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('_unit', 'orthogonal_random_control')]
    require(len(funcs) == 2, 'Missing archived helper')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source / 'operator_fit.py'), 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('audited_fixed_response',source / 'fixed_response.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks = {}
    for task in TASKS:
        fit = EVIDENCE / 'primary-durable-20260907/fits-v1/bfloat16' / task / 'vision_action_coupling'
        bank = torch.load(fit / 'operator_bank.pt', map_location='cpu', weights_only=True)
        protocol = read(fit / 'protocol.json')
        require(bank['protocol_sha256'] == digest(fit / 'protocol.json'), 'Unbound bank')
        directions = bank['global_tensors']
        rows = []
        for label, seed in (('visual', 2026090703), ('action', 2026090704)):
            native, random = directions[label+'_direction'], directions['random_'+label+'_direction']
            rebuilt = namespace['orthogonal_random_control'](native,seed)
            require(torch.equal(random,rebuilt), 'Saved coupling random direction does not reproduce exactly')
            rows.append({'site':label, 'seed':seed, 'bitwise_seed_reproduction':True,
                         'learned_norm':native.double().norm().item(), 'random_norm':random.double().norm().item(),
                         'cosine':torch.nn.functional.cosine_similarity(native.double().flatten(),random.double().flatten(),dim=0).item()})
        arms = {a['name']:a['edits'] for a in protocol['arms']}
        learned, random = arms['joint_equal_standardized_energy'], arms['matched_random_equal_standardized_energy']
        require(len(learned) == len(random) == 2, 'Wrong number of coupling sites')
        for a,b in zip(learned,random):
            require({k:v for k,v in a.items() if k != 'tensor'} == {k:v for k,v in b.items() if k != 'tensor'}, 'Different sites/budgets')
        fixed = torch.load(EVIDENCE/'fixed-response-20260908-v1/fits'/task/'operator_bank.pt',map_location='cpu',weights_only=True)
        module.validate_bank(fixed)
        basis = {}
        for arm, op in fixed['operators'].items():
            b = op['basis'].double().flatten(1)
            basis[arm] = {'rank':int(torch.linalg.matrix_rank(b).item()),
                         'max_orthonormal_error':(b@b.T-torch.eye(4)).abs().max().item(),
                         'map_singular_values':torch.linalg.svdvals(op['map'].double()).tolist()}
        tasks[task] = {'coupling': rows, 'coupling_site_time_and_scale_exact_match':True,
                       'fixed_dose':fixed['dose'], 'fixed_operators':basis,
                       'fixed_map_singular_values_are_not_claimed_matched':True}
    result = {'status':'coupling_seeds_reproduced_and_frozen_bank_geometry_checked', 'tasks':tasks,
              'new_model_calls':0,'fresh_confirmation':False,'random_seed_robustness_established':False,
              'source_sha256':digest(Path(__file__))}
    with (ROOT/'CONTROL_BANK_AUDIT.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
