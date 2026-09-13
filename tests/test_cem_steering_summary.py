import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("cem_steering_summary",Path(__file__).resolve().parents[1]/"analysis/mechanism/cem_steering_summary.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def trace(offset=0.):
    return {"selected_plan":np.full((3,20),offset).tolist(),"final_mean":np.full((6,20),offset).tolist(),"iterations":[dict(iteration=i,
        proposal_mean=np.full((6,1,20),offset if i else 0.).tolist(),proposal_std=np.full((6,1,20),.1).tolist(),
        objective_costs=np.arange(300,dtype=float).tolist(),elite_indices=list(range(10)),
        candidate_actions_sha256=("b" if offset and i else "a")*64,candidate_actions_shape=[6,300,20],
        candidate_actions_dtype="float32") for i in range(15)]}


class CEMSteeringSummaryTests(unittest.TestCase):
    def test_all_iterations_and_both_arms(self):
        payload={"traces":{"native":trace(),"fixed_rank4":trace(.1),"matched_random_fixed_rank4":trace(.2)}}
        rows,final=m.compare_case(payload,("reach",0))
        self.assertEqual(len(rows),30);self.assertEqual(len(final),2)
        self.assertAlmostEqual(final[0]["selected_prefix_delta_rms"],.1)
        self.assertAlmostEqual(final[0]["selected_prefix_delta_l2"],.1*np.sqrt(60))

    def test_returned_prefix_is_exact3by20_not_proposal_or_reshaped(self):
        value=trace()
        self.assertEqual(m.validate_trace(value).shape,(3,20))
        for shape in ((6,20),(60,),(1,3,20)):
            value=trace();value["selected_plan"]=np.zeros(shape).tolist()
            with self.assertRaises(ValueError):m.validate_trace(value)
        value=trace();value["final_mean"]=np.zeros((3,20)).tolist()
        with self.assertRaises(ValueError):m.validate_trace(value)

    def test_overlap_only_on_verified_initial_actions(self):
        native,arm=trace(),trace(.1)
        first=m.iteration_metrics(native["iterations"][0],arm["iterations"][0],0)
        later=m.iteration_metrics(native["iterations"][1],arm["iterations"][1],1)
        self.assertEqual(first["shared_population_elite_overlap_count"],10)
        self.assertIsNone(later["shared_population_elite_overlap_count"])
        arm["iterations"][0]["candidate_actions_sha256"]="z"*64
        with self.assertRaises(ValueError):m.iteration_metrics(native["iterations"][0],arm["iterations"][0],0)

    def test_initial_identity_requires_evidence_not_equal_seeds(self):
        native,arm=trace()["iterations"][0],trace()["iterations"][0]
        del arm["candidate_actions_sha256"]
        with self.assertRaises(ValueError):m.same_candidates(native,arm)

    def test_negative_entropy_and_divergence(self):
        a,b=trace()["iterations"][0],trace(.1)["iterations"][1]
        value=m.iteration_metrics(a,b,1)
        self.assertLess(value["native_proposal_entropy_nats"],0)
        self.assertEqual(value["proposal_entropy_delta_nats"],0)
        self.assertAlmostEqual(value["proposal_mean_delta_rms"],.1)

    def test_missing_trace_iteration_and_bad_elites_fail(self):
        value=trace();value["iterations"].pop()
        with self.assertRaises(ValueError):m.validate_trace(value)
        value=trace();value["iterations"][0]["elite_indices"]=list(range(20,30))
        with self.assertRaises(ValueError):m.validate_trace(value)

    def test_only_scenario_units(self):
        frame=pd.DataFrame(dict(task=["reach"]*4,arm=["fixed_rank4"]*4,episode=range(4),effect=[.1]*4))
        result=m.aggregate(frame,["task","arm"],["effect"]).iloc[0]
        self.assertAlmostEqual(result["mean"],.1);self.assertEqual(result.n,4)
        with self.assertRaises(ValueError):m.aggregate(frame.iloc[:3],["task","arm"],["effect"])

    def test_hash_bound_plot_smoke_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/"paper/data";data.mkdir(parents=True)
            rows=[dict(task=task,arm=arm,iteration=i,metric=metric,n=4,mean=.1*i,
                       marginal_95_low=.05*i,marginal_95_high=.15*i)
                  for task in m.TASKS for arm in m.ARMS for i in range(15)
                  for metric in ("proposal_mean_delta_rms","proposal_entropy_delta_nats")]
            table=data/"cem_steering_iteration_summary.csv"
            pd.DataFrame(rows).to_csv(table,index=False)
            receipt=dict(status="complete_fixed8_instrumentation_only",scenarios=8,n_per_task=4,
                         outputs={str(table.relative_to(root)):dict(sha256=m.sha(table))})
            (data/"cem_steering_summary.json").write_text(json.dumps(receipt))
            with patch.object(m,"ROOT",root),patch.object(m,"DATA",data):
                m.plots()
                for extension in ("png","svg","pdf"):
                    self.assertGreater((root/f"docs/figures/cem_steering_search.{extension}").stat().st_size,1000)
                table.write_text("tampered")
                with self.assertRaisesRegex(ValueError,"hash changed"):m.plots()


if __name__=="__main__":unittest.main()
