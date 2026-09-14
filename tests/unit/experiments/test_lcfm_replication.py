import copy
import json
from pathlib import Path

import pytest

from offline_study.experiments.lcfm_replication import validate_protocol

PROTOCOL = Path(__file__).parents[3]/'paper/data/lcfm_replication_protocol.json'


def test_fixed_registry_separates_new_scenarios_from_engineering():
    p = json.loads(PROTOCOL.read_text())
    validate_protocol(p)
    science = {r['environment_seed'] for r in p['scenarios']}
    engineering = {r['environment_seed'] for r in p['engineering']}
    assert len(science) == 200 and len(engineering) == 2
    assert science.isdisjoint(engineering)


@pytest.mark.parametrize('mutation', ['missing_case', 'duplicate_case', 'wrong_precision', 'missing_arm', 'engineering_seed_overlap'])
def test_reject_changed_execution_contract(mutation):
    p = copy.deepcopy(json.loads(PROTOCOL.read_text()))
    if mutation == 'missing_case': p['scenarios'].pop()
    elif mutation == 'duplicate_case': p['scenarios'][-1] = p['scenarios'][0]
    elif mutation == 'wrong_precision': p['precision'] = 'bfloat16'
    elif mutation == 'missing_arm': p['arms'].pop()
    else: p['engineering'][0]['environment_seed'] = p['scenarios'][0]['environment_seed']
    with pytest.raises(ValueError): validate_protocol(p)
