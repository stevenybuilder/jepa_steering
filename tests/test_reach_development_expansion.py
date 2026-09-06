import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from collect_reach_development_expansion_v1 import validate_inputs, replace_initializer, SHARDS


class DevelopmentExpansionTests(unittest.TestCase):
    def fixture(self):
        m={'episode_ids':list(range(66,74)),'split':'development_expansion_66_73_v1',
           'shards':{str(k):v for k,v in SHARDS.items()},'previous_full_episode_ids':list(range(66))}
        d={'complete':True,'model_execution':False,'planner_execution':False,
           'outputs':[{'episode':e,'environment_seed':2026090500+e,'planner_seed':90500+e,'path':f'input-{e:03d}.pt'} for e in range(62,112)]}
        return m,d
    def test_exact_disjoint_shards_before_load(self):
        m,d=self.fixture()
        self.assertEqual([x['episode'] for x in validate_inputs(m,d,49902461)],list(range(66,70)))
        self.assertEqual([x['episode'] for x in validate_inputs(m,d,49982193)],list(range(70,74)))
    def test_held_or_duplicate_sources_refused(self):
        m,d=self.fixture();m['episode_ids']=list(range(24,32))
        with self.assertRaises(ValueError):validate_inputs(m,d,49902461)
        m,d=self.fixture();d['outputs'].append(d['outputs'][4])
        with self.assertRaises(ValueError):validate_inputs(m,d,49902461)
    def test_seed_change_refused(self):
        m,d=self.fixture();d['outputs'][4]['planner_seed']+=1
        with self.assertRaises(ValueError):validate_inputs(m,d,49902461)
    def test_initializer_restored_even_on_failure(self):
        original=object();replacement=object();module=SimpleNamespace(reconstruct_fresh=original)
        with self.assertRaises(RuntimeError):
            with replace_initializer(module,replacement):
                self.assertIs(module.reconstruct_fresh,replacement)
                raise RuntimeError('test')
        self.assertIs(module.reconstruct_fresh,original)


if __name__=='__main__':unittest.main()
