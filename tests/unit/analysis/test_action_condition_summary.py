import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("action_condition_summary",ROOT/"analysis/mechanism/action_condition_summary.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
frozen = m.frozen_module()


def fixture(directory):
    key=("reach",0)
    binding=dict(task=key[0],episode=key[1],candidate_seed=123,inputs_sha256="i"*64)
    visual=np.linspace(1,2,300,dtype=np.float32)
    proprio=np.linspace(2,3,300,dtype=np.float32)
    costs=dict(visual=visual,proprio=proprio,official=visual+np.float32(.1)*proprio)
    score={k:dict(costs=v.tolist(),elite_indices=list(range(10))) for k,v in costs.items()}
    agreement={k:frozen.agreement(v.tolist(),v.tolist(),list(range(10)),list(range(10))) for k,v in costs.items()}
    banks,receipts={},{}
    for bank in m.BANKS:
        receipts[bank]=dict(actions_shape=[6,300,20],dtype="torch.float32",forecast_calls=14,scientific_arms=13,
                            native_zero_full_forecast_and_score_byte_parity=True,candidate0_zero=True,
                            native_condition_sha256={str(l):str(l)*64 for l in range(6)},
                            actions_sha256=("a" if bank=="original" else "b")*64)
        banks[bank]={"native":dict(scores=copy.deepcopy(score))}
        for layer in range(6):
            for mode in ("permute","random"):
                audit=dict(requested_delta_l2=[1.]*300,delivered_delta_l2=[1.]*300,relative_norm_error=[0.]*300,
                           zero_target_count=0,max_relative_norm_error=0.,native_condition_sha256=str(layer)*64,
                           condition_shape=[300,2,8],random_seed=frozen.random_seed(*key,bank,layer) if mode=="random" else None)
                banks[bank][f"{mode}_B{layer}"]=dict(scores=copy.deepcopy(score),agreement=copy.deepcopy(agreement),condition_audit=audit)
    payload=dict(input_binding=binding,execution_manifest_sha256=m.MANIFEST_SHA,protocol_sha256=m.PROTOCOL_SHA,
                 banks=banks,bank_receipts=receipts)
    directory.mkdir(parents=True)
    (directory/"scores.json").write_text(json.dumps(payload))
    report=dict(status="complete_action_condition_case",input_binding=binding,execution_manifest_sha256=m.MANIFEST_SHA,
                protocol_sha256=m.PROTOCOL_SHA,scientific_forward_count=26,total_forward_count=28,
                global_rng_unchanged=True,parameters_unchanged=True,inputs_unchanged=True,fresh_bank_native_sampler_byte_and_rng_parity=True,
                physical_outcomes_measured=False,fresh_confirmation=False,training_performed=False,tf32_matmul=False,tf32_cudnn=False,
                backend_provenance=dict(dino_loader_source="verified_local_cache_no_network_branch_resolution",
                    checkpoint_sha256=m.CHECKPOINT_SHA,dino_source_sha256=m.DINO_SOURCE_SHA,dino_weights_sha256=m.DINO_WEIGHT_SHA,
                    precision="float32",allow_tf32=False),gpu_uuid="test-fixture-not-real",
                fresh_bank_seed=123^0x40000000,bank_receipts=receipts,
                files={"scores.json":m.sha(directory/"scores.json"),"actions.pt":"c"*64})
    (directory/"report.json").write_text(json.dumps(report))
    (directory/"DONE.json").write_text(json.dumps(dict(report_sha256=m.sha(directory/"report.json"),files=report["files"])))
    compact={name:m.sha(directory/name) for name in ("report.json","DONE.json","scores.json")}
    cloud=dict(gcs_download_sha256_verified=True,all_report_files_hash_verified=True,raw_preservation_pending=False,
               archive_scope="complete-cohort-shard-original-files",compact_sha256=compact,
               verified_archive_members={f"shard2/reach/episode-0/{name}":digest for name,digest in {**compact,"actions.pt":"c"*64}.items()},
               cloud_uri="gs://fixture-only/test",generation="1",sha256="d"*64)
    (directory/"CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
    return key,binding


class ActionConditionSummaryTests(unittest.TestCase):
    def test_complete_case_recomputes_all_metrics_norms_and_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/"reach/episode-0";key,binding=fixture(directory)
            payload,metrics,norms,source=m.validate_case(directory,key,binding,frozen,"test-fixture-not-real")
            self.assertEqual(len(metrics),72);self.assertEqual(len(norms),24)
            self.assertTrue(all(row["top10_overlap_count"]==10 for row in metrics))
            self.assertFalse(source["bulk_actions_locally_reloaded"])
            self.assertEqual(len(source["verified_archive_members"]),4)

    def test_all16_coverage_gate_before_any_case_json_read(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/"reach/episode-0";directory.mkdir(parents=True)
            (directory/"report.json").write_text("intentionally not JSON")
            with self.assertRaisesRegex(ValueError,"Incomplete16"):m.complete_paths(d)

    def test_compact_tamper_and_wrong_original_member_fail(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/"reach/episode-0";key,binding=fixture(directory)
            path=directory/"CLOUD_VERIFIED.json";cloud=json.loads(path.read_text())
            cloud["verified_archive_members"]["shard2/reach/episode-0/actions.pt"]="wrong"
            path.write_text(json.dumps(cloud))
            with self.assertRaisesRegex(ValueError,"full-archive member"):m.validate_case(directory,key,binding,frozen,"test-fixture-not-real")
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/"reach/episode-0";key,binding=fixture(directory)
            (directory/"scores.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"Compact source byte"):m.validate_case(directory,key,binding,frozen,"test-fixture-not-real")

    def test_raw_pending_cannot_be_full_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/"reach/episode-0";key,binding=fixture(directory)
            path=directory/"CLOUD_VERIFIED.json";cloud=json.loads(path.read_text())
            cloud["raw_preservation_pending"]=True;path.write_text(json.dumps(cloud))
            with self.assertRaisesRegex(ValueError,"fully verified"):m.validate_case(directory,key,binding,frozen,"test-fixture-not-real")

    def test_frozen_implementation_hash_required(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"source.py";path.write_text("raise RuntimeError('must not execute')")
            with self.assertRaisesRegex(ValueError,"implementation changed"):m.frozen_module(path)

    def test_complete16_frozen_contrasts_use_scenarios_and_all_families(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/"reach/episode-0";key,binding=fixture(directory)
            payload,_,_,_=m.validate_case(directory,key,binding,frozen,"test-fixture-not-real")
            cases=[]
            for task in m.TASKS:
                for episode in range(8):
                    item=copy.deepcopy(payload)
                    item["input_binding"].update(task=task,episode=episode)
                    cases.append(item)
            rows=frozen.summarize_complete(cases)
            self.assertEqual(len(rows),144)
            self.assertTrue(all(row["n"]==8 and row["mean"]==0 for row in rows))
            self.assertEqual({(r["task"],r["bank"],r["modality"]) for r in rows},
                             {(t,b,mod) for t in m.TASKS for b in m.BANKS for mod in m.MODALITIES})
            with self.assertRaises(ValueError):frozen.summarize_complete(cases[:-1])

    def test_plot_complete_all_layer_families_and_table_hash_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/"paper/data";data.mkdir(parents=True)
            rows=[dict(task=t,bank=b,modality="official",metric=metric,layer=l,n=8,n_defined=8,
                       mean=(l-2)*.01,simultaneous_95_low=(l-2)*.01-.015,simultaneous_95_high=(l-2)*.01+.015)
                  for t in m.TASKS for b in m.BANKS for metric in ("excess_spearman_loss","excess_top10_overlap_loss") for l in range(6)]
            table=data/"action_condition_contrasts.csv";pd.DataFrame(rows).to_csv(table,index=False)
            receipt=dict(status="complete16_exploratory_action_condition",scenarios=16,n_per_task=8,
                         outputs={str(table.relative_to(root)):dict(sha256=m.sha(table))})
            (data/"action_condition_summary.json").write_text(json.dumps(receipt))
            with patch.object(m,"ROOT",root),patch.object(m,"DATA",data):
                m.plots()
                for ext in ("png","svg","pdf"):
                    self.assertGreater((root/f"docs/figures/action_condition_specificity.{ext}").stat().st_size,1000)
                table.write_text("changed")
                with self.assertRaisesRegex(ValueError,"table hash changed"):m.plots()


if __name__ == "__main__":
    unittest.main()
