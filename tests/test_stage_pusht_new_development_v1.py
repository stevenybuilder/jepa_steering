import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage',ROOT/'scripts/geometry_map/stage_pusht_new_development_v1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FrozenInputTests(unittest.TestCase):
    def manifest(self):
        return json.loads((ROOT/'artifacts/geometry_map/pusht_new_development_v1/manifest.json').read_text())
    def test_frozen_manifest(self):
        module.validate_manifest(self.manifest())
    def test_changed_panel_or_ids_rejected(self):
        for key,value in [('panel','held'),('clip_offset',1),('source_root','/some/val')]:
            x=self.manifest();x[key]=value
            with self.assertRaises(ValueError):module.validate_manifest(x)
        x=self.manifest();x['rows'][0]['source_id']=125
        with self.assertRaises(ValueError):module.validate_manifest(x)
    def test_duplicate_initial_rejected(self):
        x=self.manifest();x['rows'][1]['initial_state_sha256']=x['rows'][0]['initial_state_sha256']
        with self.assertRaises(ValueError):module.validate_manifest(x)

if __name__=='__main__':unittest.main()
