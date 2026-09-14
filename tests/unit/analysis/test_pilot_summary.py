import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("pilot_summary",Path(__file__).resolve().parents[3]/"analysis/mechanism/pilot_summary.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def score_record(cost):
    return {"objective_costs":cost.tolist(),"elite_indices":np.argsort(cost)[:10].tolist()}


def replay_fixture():
    native = np.linspace(1,2,300)
    scores = {"native":score_record(native)}
    summaries = []
    for arm in m.ARMS:
        full = native+.01+.002*np.sin(np.arange(300))
        common = native+.01+.0015*np.sin(np.arange(300))
        centered = native+.0005*np.sin(np.arange(300))
        for component,values in (("full",full),("cached_full",full),("zero",native),("mean_only",common),("centered_only",centered)):
            name = f"{arm}-{component}"
            scores[name] = score_record(values)
            summaries.append(dict(arm=name,**m.score_metrics(native,values,scores["native"]["elite_indices"],scores[name]["elite_indices"])))
        summaries.append(dict(arm=arm,component_audit=True,cached_full_exact_forecast_and_score_parity=True,
                              zero_exact_forecast_and_score_parity=True,no_component_renormalization=True,
                              common_field_energy_fraction=.99,centered_field_energy_fraction=.01,
                              common_centered_cost_reconstruction=m.reconstruction(native,full,common),
                              centered_centered_cost_reconstruction=m.reconstruction(native,full,centered)))
    return {"summaries":summaries},scores


class PilotSummaryTests(unittest.TestCase):
    def test_only_complete_frozen_cohorts(self):
        self.assertEqual(len(m.expected_cases(64)),64)
        self.assertEqual(len(m.expected_cases(8)),8)
        with self.assertRaises(ValueError): m.expected_cases(16)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root/"reach"/"episode-0"
            path.mkdir(parents=True)
            (path/"report.json").write_text("not even read before coverage passes")
            with self.assertRaisesRegex(ValueError,"Incomplete frozen cohort"):
                m.discover(root,m.expected_cases(8))

    def test_duplicate_case_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for parent in ("one","two"):
                path = Path(d)/parent/"reach"/"episode-0"
                path.mkdir(parents=True); (path/"report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                m.discover(d,m.expected_cases(8))

    def test_separate_actual_cem_addon_is_not_a_duplicate_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for parent in ("development-v1", "steered-cem-v2"):
                for task, episode in m.expected_cases(8):
                    path = root/parent/task/f"episode-{episode}"
                    path.mkdir(parents=True)
                    for name in ("report.json", "DONE.json", "CLOUD_VERIFIED.json"):
                        (path/name).write_text("{}")
            cases = m.discover(root,m.expected_cases(8))
            self.assertEqual(len(cases),8)
            self.assertTrue(all(p.parent.parent.name == "development-v1" for p in cases.values()))

    def test_reconstruction_not_clamped_and_zero_explicit(self):
        native = np.zeros(3)
        full = np.array([-1.,0,1])
        self.assertEqual(m.reconstruction(native,full,full),1.)
        self.assertEqual(m.reconstruction(native,full,-full),-3.)
        self.assertIsNone(m.reconstruction(native,np.ones(3),full))

    def test_actual_elite_ties_preserved_but_bad_set_rejected(self):
        costs = np.ones(300)
        ids = list(range(20,30))
        self.assertEqual(m.score_metrics(costs,costs,ids,ids)["native_elite_overlap_count"],10)
        with self.assertRaises(ValueError):
            m.score_metrics(np.arange(300),np.arange(300),list(range(10)),list(range(20,30)))

    def test_all_attention_heads_layers_horizons(self):
        payload = {"output_byte_parity":True,"same_input_rng_unchanged":True,"attention":[]}
        for h in range(1,7):
            for layer in range(6):
                payload["attention"].append(dict(layer=layer,forecast_horizon=h,T=h,H=16,W=16,action_tokens=0,
                    spatial_distance_patches=[[2.]*16],temporal_distance_frames=[[.5]*16],
                    conditioning_mass=[[0.]*16],visual_mass=[[1.]*16]))
        rows = m.attention_rows(payload,("reach",0))
        self.assertEqual(len(rows),576)
        self.assertEqual({r["head"] for r in rows},set(range(16)))
        payload["attention"].pop()
        with self.assertRaises(ValueError): m.attention_rows(payload,("reach",0))

    def test_negative_gaussian_entropy_is_valid_and_recomputed(self):
        std = np.full((6,1,20),.01)
        costs = np.arange(300,dtype=float)
        row = dict(proposal_entropy_nats=float(np.sum(np.log(std)+.5*np.log(2*np.pi*np.e))),
                   proposal_std_mean=.01,best_cost=0.,best_runnerup_margin=1.,elite_boundary_margin=1.,
                   proposal_std=std.tolist(),objective_costs=costs.tolist(),elite_indices=list(range(10)))
        values = [dict(row,iteration=i) for i in range(15)]
        result = m.cem_rows(values,("reach",0))
        self.assertLess(result[0]["proposal_entropy_nats"],0)
        self.assertTrue(all(r["compact_array_recalculation"] for r in result))
        values[0]["proposal_entropy_nats"] += 1
        with self.assertRaises(ValueError): m.cem_rows(values,("reach",0))

    def test_component_replay_preserves_all_arms_and_identities(self):
        report,scores = replay_fixture()
        rows,audit = m.replay_rows(report,scores,("reach",0))
        self.assertEqual(len(rows),10); self.assertEqual(len(audit),2)
        self.assertAlmostEqual(audit[0]["additive_components_cost_reconstruction"],1.)
        scores["fixed_rank4-cached_full"]["objective_costs"][10] += .0001
        with self.assertRaises(ValueError): m.replay_rows(report,scores,("reach",0))

    def test_scenario_unit_and_undefined_values_not_dropped(self):
        frame = pd.DataFrame({"task":["reach"]*4,"episode":range(4),"effect":[1.,1.,1.,1.]})
        result = m.summarize(frame,["task"],["effect"],4).iloc[0]
        self.assertEqual(result["mean"],1.)
        self.assertEqual(result.marginal_95_low,1.)
        frame.loc[2,"effect"] = np.nan
        result = m.summarize(frame,["task"],["effect"],4).iloc[0]
        self.assertEqual(result.n_defined,3)
        self.assertTrue(pd.isna(result["mean"]))
        with self.assertRaises(ValueError): m.summarize(pd.concat([frame,frame]),["task"],["effect"],4)

    def test_compact_hash_and_completion_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            (directory/"report.json").write_text(json.dumps({"files":{}}))
            (directory/"DONE.json").write_text(json.dumps({"report_sha256":"wrong","files":{}}))
            (directory/"CLOUD_VERIFIED.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"completion hash"):
                m.validate_case(directory,("reach",0),{},[])

    def test_complete_compact_case_source_binding(self):
        report,scores = replay_fixture()
        attention = {"input_binding":{},"output_byte_parity":True,"same_input_rng_unchanged":True,
                     "attention":[dict(layer=l,forecast_horizon=h,T=h,H=16,W=16,action_tokens=0,
                         spatial_distance_patches=[[2.]*16],temporal_distance_frames=[[.5]*16],
                         conditioning_mass=[[0.]*16],visual_mass=[[1.]*16]) for h in range(1,7) for l in range(6)]}
        cem = [dict(iteration=i,proposal_entropy_nats=-20.,proposal_std_mean=.2,best_cost=1.,
                    best_runnerup_margin=.01,elite_boundary_margin=.02) for i in range(15)]
        report.update(status="canonical_development_mechanism_case_complete",input_binding={},physical_outcomes_measured=False,
                      attention_candidate_count=1,attention_blocks=6,attention_horizons=6,cem_iterations=15,
                      cem_candidates_per_iteration=300,shared_candidate_count=300,shared_candidate_banks=1,
                      base_arms=2,components_per_arm=5,cem_output_and_rng_parity=True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,payload in (("attention.json",attention),("scores.json",scores),("cem_summary.json",cem)):
                (root/name).write_text(json.dumps(payload))
            report["files"]={n:m.sha(root/n) for n in ("attention.json","scores.json")}
            report["files"]["native-cem-trace.pt"]="archived_trace_hash"
            (root/"report.json").write_text(json.dumps(report))
            (root/"DONE.json").write_text(json.dumps(dict(report_sha256=m.sha(root/"report.json"),files=report["files"])))
            cloud=dict(all_report_files_hash_verified=True,gcs_download_sha256_verified=True,
                       compact_sha256={"cem_summary.json":m.sha(root/"cem_summary.json")},parent_trace_sha256="archived_trace_hash",
                       cloud_uri="gs://test/example",sha256="archived_payload_hash")
            (root/"CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            result=m.validate_case(root,("reach",0),{},["initial_manifest_hash"])
            self.assertEqual([len(x) for x in result[:4]],[576,15,10,2])
            self.assertTrue(result[-1]["legacy_execution_binding"])
            self.assertFalse(result[-1]["raw_preservation_pending"])
            # Operational compact-only readiness must not masquerade as raw preservation.
            members={name:m.sha(root/name) for name in ("report.json","DONE.json","attention.json","scores.json","cem_summary.json")}
            cloud.update(archive_scope="compact-derived-only",compact_files_hash_verified=True,raw_preservation_pending=True,
                         all_report_files_hash_verified=False,original_worker_report_files_sha256_verified=True,
                         verified_compact_members=members,compact_sha256=members)
            (root/"CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            result=m.validate_case(root,("reach",0),{},["initial_manifest_hash"])
            self.assertTrue(result[-1]["raw_preservation_pending"])
            cloud["all_report_files_hash_verified"]=True
            (root/"CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            with self.assertRaisesRegex(ValueError,"Ambiguous compact-only"):
                m.validate_case(root,("reach",0),{},["initial_manifest_hash"])
            cloud["all_report_files_hash_verified"]=False
            (root/"CLOUD_VERIFIED.json").write_text(json.dumps(cloud))
            # The derived summary is not trusted just because the bulk archive exists.
            (root/"cem_summary.json").write_text(json.dumps(cem)+" ")
            with self.assertRaisesRegex(ValueError,"Compact-cloud member hash"):
                m.validate_case(root,("reach",0),{},["initial_manifest_hash"])

    def test_public_table_plot_smoke_uses_temporary_synthetic_data(self):
        # Synthetic values stay in a temporary tree, never the public result paths.
        attention=[]; cem=[]; components=[]
        for task in m.TASKS:
            for metric in m.ATTENTION:
                for layer in range(6):
                    for horizon in range(1,7):
                        for head in range(16):
                            attention.append(dict(task=task,metric=metric,layer=layer,horizon=horizon,head=head,
                                                  mean=1+layer/6+head/16+horizon/6,n=4,marginal_95_low=0,marginal_95_high=4))
            for metric in m.CEM:
                for iteration in range(15):
                    value = 10-iteration if metric=="proposal_entropy_nats" else 1/(1+iteration)
                    cem.append(dict(task=task,metric=metric,iteration=iteration,mean=value,marginal_95_low=value-.1,marginal_95_high=value+.1,n=4))
            for arm in m.ARMS:
                for metric in ("common_field_energy_fraction","centered_field_energy_fraction",
                               "common_centered_cost_reconstruction","centered_centered_cost_reconstruction"):
                    value=.8 if metric.startswith("common") else .2
                    components.append(dict(task=task,arm=arm,metric=metric,mean=value,marginal_95_low=value-.02,marginal_95_high=value+.02,n=4))
        old_root,old_data=m.ROOT,m.DATA
        try:
            with tempfile.TemporaryDirectory() as d:
                m.ROOT=Path(d); m.DATA=m.ROOT/"paper/data"
                m.DATA.mkdir(parents=True); (m.ROOT/"docs").mkdir()
                outputs={}
                for name,rows in (("attention",attention),("cem",cem),("component",components)):
                    path=m.DATA/f"pilot_summary_{name}_summary.csv"
                    pd.DataFrame(rows).to_csv(path,index=False)
                    outputs[str(path.relative_to(m.ROOT))]={"sha256":m.sha(path)}
                (m.DATA/"pilot_summary.json").write_text(json.dumps(dict(outputs=outputs,cohort=8,n_per_task=4)))
                m.plots()
                self.assertEqual(len(list((m.ROOT/"docs/figures").glob("*.pdf"))),4)
                self.assertEqual(len(list((m.ROOT/"docs/figures").glob("*.svg"))),4)
                self.assertEqual(len(list((m.ROOT/"docs/figures").glob("*.png"))),4)
        finally:
            m.ROOT,m.DATA=old_root,old_data


if __name__ == "__main__":
    unittest.main()
