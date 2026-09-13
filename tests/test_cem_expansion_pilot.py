import copy
import unittest
from types import SimpleNamespace

import torch
from offline_study.cem_expansion_pilot import (ARMS,CountedForecast,validate_payload,
    validate_registry,model_versions)


def payload():
    return {'selected_plan':torch.zeros(3,20),'final_mean':torch.zeros(6,20),
        'iterations':[{'iteration':i,'candidate_actions':torch.zeros(6,300,20),
            'objective_costs':torch.arange(300,dtype=torch.float32),
            'proposal_mean':torch.zeros(6,20),'proposal_std':torch.ones(6,20),
            'elite_indices':torch.arange(10)} for i in range(15)]}


class ExpansionContractTests(unittest.TestCase):
    def test_complete_trace_preserves_executed_prefix(self):
        p=payload();validate_payload(p)
        p['selected_plan']=torch.zeros(6,20)
        with self.assertRaises(ValueError):validate_payload(p)

    def test_rejects_missing_iteration(self):
        p=payload();p['iterations'].pop()
        with self.assertRaises(ValueError):validate_payload(p)

    def test_rejects_non_fp32_or_nonfinite(self):
        p=payload();p['iterations'][0]['candidate_actions']=p['iterations'][0]['candidate_actions'].half()
        with self.assertRaises(ValueError):validate_payload(p)
        p=payload();p['iterations'][0]['objective_costs'][0]=float('nan')
        with self.assertRaises(ValueError):validate_payload(p)

    def test_rejects_duplicate_elites_and_bad_scale(self):
        p=payload();p['iterations'][0]['elite_indices'][1]=0
        with self.assertRaises(ValueError):validate_payload(p)
        p=payload();p['iterations'][1]['proposal_std'][0,0]=0
        with self.assertRaises(ValueError):validate_payload(p)

    def test_exact_population_and_mean_call_counts(self):
        class Intervention:
            calls=0
            def __call__(self,context,act_suffix=None,**kwargs):
                self.calls+=1;return act_suffix
        counted=CountedForecast(Intervention())
        for _ in range(15):
            for n in (300,1):
                actions=torch.zeros(6,n,20)
                self.assertIs(counted(None,act_suffix=actions),actions)
        counted.validate()
        with self.assertRaises(ValueError):counted(None,act_suffix=torch.zeros(6,10,20))
        counted(None,act_suffix=torch.zeros(6,1,20))
        with self.assertRaises(ValueError):counted.validate()

    def test_registry_excludes_initial_eight(self):
        manifest={'scenarios':{t:list(range(4,32)) for t in ('reach','reach-wall')},'arms':list(ARMS)}
        protocol={'arms':list(ARMS),'new_episode_ids':list(range(4,32)),'new_cases':56,
            'counts':{'searches_per_case':6,'population300_forecasts_per_case':90,
            'mean1_forecasts_per_case':90,'callbacks_per_case':180,'new_cases':56}}
        validate_registry(manifest,protocol)
        manifest['scenarios']['reach'].insert(0,0)
        with self.assertRaises(ValueError):validate_registry(manifest,protocol)

    def test_model_version_detects_inplace_write(self):
        model=torch.nn.Linear(2,2);before=model_versions(model)
        with torch.no_grad():model.weight.add_(1)
        self.assertNotEqual(before,model_versions(model))


if __name__=='__main__':unittest.main()
