import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from run_action_path_interchange import target_actions,recipient_replacements,METHODS


def test_target_actions_frozen_inside_support():
    raw=torch.zeros(9,30,2);raw[1]=.1;raw[2]=-.1
    torch.testing.assert_close(target_actions(raw,0,4,.25),raw[1])
    torch.testing.assert_close(target_actions(raw,0,4,-.25),raw[2])
    with pytest.raises(ValueError):target_actions(raw,0,4,0)


def test_oracle_is_separate_from_all_estimates():
    native=torch.zeros(256,400);donor=torch.ones_like(native)
    estimates={m:torch.full_like(native,i+2.) for i,m in enumerate(METHODS)}
    rows=recipient_replacements(native,donor,estimates)
    assert rows.shape==(8,256,400)
    assert torch.equal(rows[0],native) and torch.equal(rows[1],donor)
    assert torch.equal(rows[2],estimates[METHODS[0]])
