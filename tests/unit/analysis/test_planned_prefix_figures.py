"""Synthetic plot-contract tests; no fabricated scientific outputs."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('prefix_figures',ROOT/'scripts/build_planned_prefix_figures.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    f=[];p=[];s=[]
    for task in m.TASKS:
        for episode in range(4,32):
            for model in m.ARMS:
                for plan in m.ARMS:
                    f.append(dict(task=task,episode=episode,model_arm=model,plan_arm=plan,
                        predicted_goal_cost_weighted=episode/100,actual_goal_cost_weighted=episode/101))
            for arm in m.ARMS[1:]:
                for metric in m.METRICS:p.append(dict(task=task,episode=episode,arm=arm,metric=metric,effect=(episode-20)/100))
        for arm in m.ARMS[1:]:
            for metric in m.METRICS:s.append(dict(task=task,arm=arm,metric=metric,n=28,mean=-.025,bonferroni_family=12,bonferroni_95_low=-.05,bonferroni_95_high=.01))
    return pd.DataFrame(f),pd.DataFrame(p),pd.DataFrame(s)


class PrefixFigureTests(unittest.TestCase):
    def test_full_grid_and_negative_effects(self):m.validate_frames(*fixture())
    def test_missing_or_duplicate_forecast(self):
        f,p,s=fixture()
        for broken in (f.iloc[:-1],pd.concat([f,f.iloc[:1]])):
            with self.assertRaisesRegex(ValueError,'registered56'):m.validate_frames(broken,p,s)
    def test_nonfinite_cost_not_omitted(self):
        f,p,s=fixture();f.loc[0,'actual_goal_cost_weighted']=np.nan
        with self.assertRaisesRegex(ValueError,'goal costs'):m.validate_frames(f,p,s)
    def test_primary_mean_not_replaced(self):
        f,p,s=fixture();s.loc[0,'mean']=0
        with self.assertRaisesRegex(ValueError,'Primary mean'):m.validate_frames(f,p,s)
    def test_interval_family_required(self):
        f,p,s=fixture();s.loc[0,'bonferroni_family']=1
        with self.assertRaisesRegex(ValueError,'family12'):m.validate_frames(f,p,s)


if __name__=='__main__':unittest.main()
