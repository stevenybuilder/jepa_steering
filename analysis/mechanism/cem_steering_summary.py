"""Separate fixed-eight actual-steered-CEM instrumentation summary; CPU only.

Compact-case schema: report/DONE/CLOUD_VERIFIED plus steered-cem-summary.json containing
input_binding and traces keyed by native/fixed_rank4/matched_random_fixed_rank4.
Each trace has selected_plan and15 iterations with actual proposal mean/std,
costs, elite IDs, and candidate_actions SHA256/shape/dtype (or candidate arrays).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"paper/data"
PROTOCOL=DATA/"cem_steering_protocol.json"
TASKS=("reach","reach-wall")
ARMS=("fixed_rank4","matched_random_fixed_rank4")


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value,shape=None):
    array=np.asarray(value,dtype=float)
    if (shape is not None and array.shape!=shape) or not np.isfinite(array).all():
        raise ValueError("Unexpected trace metric shape or nonfinite value")
    return array


def validate_trace(trace):
    rows=trace["iterations"]
    if len(rows)!=15 or [r["iteration"] for r in rows]!=list(range(15)):
        raise ValueError("Require every actual CEM iteration0..14")
    plan=finite(trace["selected_plan"],(3,20))
    if "final_mean" in trace:
        finite(trace["final_mean"],(6,20))
    for row in rows:
        mean,std=finite(row["proposal_mean"]),finite(row["proposal_std"])
        if mean.shape!=std.shape or mean.size!=120 or np.any(std<=0):
            raise ValueError("Invalid native diagonal-Gaussian proposal")
        costs=finite(row["objective_costs"],(300,))
        elites=np.asarray(row["elite_indices"])
        if elites.shape!=(10,) or elites.dtype.kind not in "iu" or len(set(elites))!=10 or min(elites)<0 or max(elites)>=300:
            raise ValueError("Invalid actual elite IDs")
        if costs[elites].max()>np.delete(costs,elites).min():
            raise ValueError("Actual elites disagree with costs")
        if "proposal_entropy_nats" in row:
            entropy=float(np.sum(np.log(std)+.5*math.log(2*math.pi*math.e)))
            if not np.isclose(row["proposal_entropy_nats"],entropy,rtol=1e-6,atol=1e-10):
                raise ValueError("Proposal entropy disagrees with actual std")
    return plan


def same_candidates(a,b):
    if "candidate_actions" in a and "candidate_actions" in b:
        x,y=finite(a["candidate_actions"],(6,300,20)),finite(b["candidate_actions"],(6,300,20))
        return np.array_equal(x,y)
    fields=("candidate_actions_sha256","candidate_actions_shape","candidate_actions_dtype")
    if any(k not in a or k not in b for k in fields):
        raise ValueError("Iteration0 lacks candidate-array identity evidence")
    if any(re.fullmatch(r"[0-9a-f]{64}",r["candidate_actions_sha256"]) is None or r["candidate_actions_shape"]!=[6,300,20] for r in (a,b)):
        raise ValueError("Malformed candidate-array fingerprint")
    return all(a[k]==b[k] for k in fields)


def iteration_metrics(native,other,iteration):
    output={}
    for name in ("mean","std"):
        a,b=finite(native[f"proposal_{name}"]),finite(other[f"proposal_{name}"])
        if a.shape!=b.shape:
            raise ValueError("Proposal coordinate layout changed")
        delta=b-a
        output[f"proposal_{name}_delta_l2"]=float(np.linalg.norm(delta))
        output[f"proposal_{name}_delta_rms"]=float(np.sqrt(np.mean(delta**2)))
    entropy=[]; margins=[]
    for row in (native,other):
        std=finite(row["proposal_std"])
        entropy.append(float(np.sum(np.log(std)+.5*math.log(2*math.pi*math.e))))
        costs=np.sort(finite(row["objective_costs"],(300,)))
        margins.append((float(costs[1]-costs[0]),float(costs[10]-costs[9])))
    output.update(native_proposal_entropy_nats=entropy[0],arm_proposal_entropy_nats=entropy[1],
                  proposal_entropy_delta_nats=entropy[1]-entropy[0],
                  native_best_runnerup_margin=margins[0][0],arm_best_runnerup_margin=margins[1][0],
                  best_runnerup_margin_delta=margins[1][0]-margins[0][0],
                  native_elite_boundary_margin=margins[0][1],arm_elite_boundary_margin=margins[1][1],
                  elite_boundary_margin_delta=margins[1][1]-margins[0][1])
    if iteration==0:
        if not same_candidates(native,other):
            raise ValueError("Registered first population is not exactly the same actions")
        delta=finite(other["objective_costs"])-finite(native["objective_costs"])
        output.update(same_candidate_actions_verified=True,
                      shared_population_elite_overlap_count=len(set(native["elite_indices"])&set(other["elite_indices"])),
                      shared_population_cost_delta_mean=float(delta.mean()),
                      shared_population_cost_delta_rms=float(np.sqrt(np.mean(delta**2))))
    else:
        # Deliberately do not inspect elite-ID overlap after adaptive proposals.
        output.update(same_candidate_actions_verified=None,shared_population_elite_overlap_count=None,
                      shared_population_cost_delta_mean=None,shared_population_cost_delta_rms=None)
    return output


def compare_case(payload,key):
    traces=payload["traces"]
    if set(traces)!={"native",*ARMS}:
        raise ValueError("Incomplete fixed learned/random/native trace set")
    plans={name:validate_trace(trace) for name,trace in traces.items()}
    iterations,final=[] ,[]
    for arm in ARMS:
        if plans[arm].shape!=plans["native"].shape:
            raise ValueError("Final action-plan layout mismatch")
        delta=plans[arm]-plans["native"]
        final.append(dict(task=key[0],episode=key[1],arm=arm,selected_prefix_delta_l2=float(np.linalg.norm(delta)),
                          selected_prefix_delta_rms=float(np.sqrt(np.mean(delta**2)))))
        for iteration,(native,other) in enumerate(zip(traces["native"]["iterations"],traces[arm]["iterations"])):
            iterations.append(dict(task=key[0],episode=key[1],arm=arm,iteration=iteration,
                                   **iteration_metrics(native,other,iteration)))
    return iterations,final


def aggregate(frame,groups,metrics):
    protocol=read(PROTOCOL)
    weights=np.random.default_rng(protocol["bootstrap_seed"]).multinomial(4,[.25]*4,size=protocol["bootstrap_replicates"])/4
    rows=[]
    for key,block in frame.groupby(groups,sort=True):
        if len(block)!=4 or set(block.episode)!=set(range(4)):
            raise ValueError("Require all four scenarios per task/cell; no candidate pseudoreplication")
        block=block.sort_values("episode")
        for metric in metrics:
            values=finite(block[metric].to_numpy())
            draws=weights@values
            rows.append(dict(zip(groups,key),metric=metric,n=4,mean=float(values.mean()),
                             marginal_95_low=float(np.quantile(draws,.025)),marginal_95_high=float(np.quantile(draws,.975))))
    return pd.DataFrame(rows)


def run(args):
    # Read the prospective manifest before any source outcomes.
    if sha(args.execution_manifest)!=args.execution_manifest_sha256:
        raise ValueError("Addon execution-manifest hash mismatch")
    manifest,protocol=read(args.execution_manifest),read(PROTOCOL)
    if args.execution_manifest_sha256!=protocol["addon_execution_manifest_sha256"]:
        raise ValueError("Unexpected prospective addon execution manifest")
    if manifest["scenarios"]!=protocol["scenarios"]:
        raise ValueError("Unexpected addon scenario set")
    if manifest["native_execution_manifest_sha256"]!=protocol["native_execution_manifest_sha256"]:
        raise ValueError("Native comparator manifest mismatch")
    cases={(task,episode):args.compact_root/task/f"episode-{episode}" for task in TASKS for episode in range(4)}
    if not all((p/"CLOUD_VERIFIED.json").exists() and (p/"DONE.json").exists() for p in cases.values()):
        raise ValueError("Incomplete fixed8 addon preservation; no partial aggregate")
    iteration_rows,final_rows,sources=[],[],[]
    for key,directory in sorted(cases.items()):
        report,done,cloud=read(directory/"report.json"),read(directory/"DONE.json"),read(directory/"CLOUD_VERIFIED.json")
        if done["report_sha256"]!=sha(directory/"report.json") or done["files"]!=report["files"] or report["execution_manifest_sha256"]!=args.execution_manifest_sha256:
            raise ValueError("Addon completion or execution binding mismatch")
        if report.get("physical_outcomes_measured") is not False:
            raise ValueError("Unclear addon physical-outcome scope")
        for arm in ARMS:
            identity=report["parities"][arm]
            if any(identity.get(k) is not True for k in ("selected_plan_byte_equal","local_generator_byte_equal","global_rng_unchanged","native_iteration0_actions_byte_equal")):
                raise ValueError("Missing exact per-arm CEM parity")
            if identity["untraced_forecast_calls"]!=30 or identity["traced_forecast_calls"]!=30:
                raise ValueError("Unexpected actual steered-CEM forecast count")
        if cloud.get("gcs_download_sha256_verified") is not True or cloud.get("all_report_files_hash_verified") is not True:
            raise ValueError("Missing verified addon archive")
        compact=directory/"steered-cem-summary.json"
        digest=sha(compact)
        if cloud.get("compact_sha256",{}).get(compact.name)!=digest or report["files"][compact.name]!=digest:
            raise ValueError("Unbound compact trace comparison")
        payload=read(compact)
        if payload["input_binding"]!=report["input_binding"] or (payload["input_binding"]["task"],payload["input_binding"]["episode"])!=key:
            raise ValueError("Addon compact input mismatch")
        native_directory=args.native_compact_root/key[0]/f"episode-{key[1]}"
        native_report,native_done=read(native_directory/"report.json"),read(native_directory/"DONE.json")
        if native_done["report_sha256"]!=sha(native_directory/"report.json"):
            raise ValueError("Original native comparator completion mismatch")
        if native_report["input_binding"]!=payload["input_binding"]:
            raise ValueError("Addon changed original native scenario/seed binding")
        if payload["traces"]["native"]["source_trace_sha256"]!=native_report["files"]["native-cem-trace.pt"]:
            raise ValueError("Addon comparator is not the original native trace")
        for arm in ARMS:
            if payload["traces"][arm]["source_trace_sha256"]!=report["files"][arm+"-cem-trace.pt"]:
                raise ValueError("Addon steered trace parent mismatch")
        i,f=compare_case(payload,key)
        iteration_rows.extend(i);final_rows.extend(f)
        sources.append(dict(task=key[0],episode=key[1],report_sha256=sha(directory/"report.json"),
                            comparison_sha256=digest,cloud_receipt_sha256=sha(directory/"CLOUD_VERIFIED.json"),
                            cloud_uri=cloud["cloud_uri"],cloud_archive_sha256=cloud["sha256"],
                            native_trace_sha256=payload["traces"]["native"]["source_trace_sha256"],
                            original_native_report_sha256=sha(native_directory/"report.json"),
                            arm_trace_sha256={a:payload["traces"][a]["source_trace_sha256"] for a in ARMS},
                            input_binding=payload["input_binding"]))
    frames={"iteration_cases":pd.DataFrame(iteration_rows),"selected_prefix_cases":pd.DataFrame(final_rows)}
    primary=[m for m in protocol["primary_metrics"] if not m.startswith("selected_prefix")]
    frames["iteration_summary"]=aggregate(frames["iteration_cases"],["task","arm","iteration"],primary)
    frames["selected_prefix_summary"]=aggregate(frames["selected_prefix_cases"],["task","arm"],["selected_prefix_delta_l2","selected_prefix_delta_rms"])
    frames["shared_iteration0_summary"]=aggregate(frames["iteration_cases"].query("iteration==0"),["task","arm"],
        ["shared_population_elite_overlap_count","shared_population_cost_delta_mean","shared_population_cost_delta_rms"])
    outputs={}
    for name,frame in frames.items():
        path=DATA/f"cem_steering_{name}.csv";frame.to_csv(path,index=False)
        outputs[str(path.relative_to(ROOT))]=dict(sha256=sha(path),rows=len(frame))
    result=dict(status="complete_fixed8_instrumentation_only",scenarios=8,n_per_task=4,protocol_sha256=sha(PROTOCOL),
                analysis_source_sha256=sha(__file__),execution_manifest_sha256=args.execution_manifest_sha256,
                precision=manifest["precision"],
                sources=sources,outputs=outputs,physical_outcomes_measured=False,fresh_confirmation=False,
                caveats=["Later iterations contain different candidate actions; no later candidate-ID overlap statistic",
                         "Same initialseed is not same candidate population after proposal adaptation",
                         "Eight historical development scenarios are instrumentation/exploratory, not efficacy",
                         "Preclip Gaussian differential entropy is coordinate-dependent and may be negative"])
    (DATA/"cem_steering_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def plots():
    """Render only source-hash-bound complete-eight public tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    receipt=read(DATA/"cem_steering_summary.json")
    if receipt.get("status")!="complete_fixed8_instrumentation_only" or receipt.get("scenarios")!=8 or receipt.get("n_per_task")!=4:
        raise ValueError("Plot requires the complete fixed-eight receipt")
    for name,info in receipt["outputs"].items():
        if sha(ROOT/name)!=info["sha256"]:
            raise ValueError("Public addon table hash changed")
    frame=pd.read_csv(DATA/"cem_steering_iteration_summary.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.titlesize":10.5,
                         "axes.labelsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5,
                         "legend.fontsize":10.5,"svg.hashsalt":"cem-steering-search-v1",
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(6.8,5.8))
    fig.subplots_adjust(left=.13,right=.98,top=.80,bottom=.19,hspace=.43,wspace=.35)
    for row,task in enumerate(TASKS):
        for col,(metric,label) in enumerate((("proposal_mean_delta_rms","Proposal-mean RMS change"),
                                           ("proposal_entropy_delta_nats","Proposal entropy change (nats)"))):
            ax=axes[row,col]
            for arm,color,title in zip(ARMS,("#187c80","#c57d36"),("Learned rank four","Calibrated random")):
                block=frame[(frame.task==task)&(frame.arm==arm)&(frame.metric==metric)].sort_values("iteration")
                if len(block)!=15 or set(block.iteration)!=set(range(15)) or set(block.n)!={4}:
                    raise ValueError("Incomplete all-iteration addon plot cell")
                x=block.iteration.to_numpy()+1
                ax.plot(x,block["mean"],color=color,label=title,lw=1.6)
                ax.fill_between(x,block.marginal_95_low,block.marginal_95_high,color=color,alpha=.16,lw=0)
            ax.axhline(0,color="#89929b",lw=.65)
            ax.grid(alpha=.15)
            ax.set_xticks([1,5,10,15])
            if row==0: ax.set_title(label)
            if row==1: ax.set_xlabel("CEM iteration")
            if col==0: ax.set_ylabel("Reach" if task=="reach" else "Reach-Wall")
    fig.suptitle("Steering changes adaptive search trajectories",fontsize=12,y=.98)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.53,.925),ncol=2,frameon=False)
    fig.text(.5,.09,"All 8 fixed cases; n=4/task. Marginal scenario-bootstrap 95% bands.",ha="center",fontsize=10.5)
    fig.text(.5,.045,"Differences from native search. Exploratory; no actions executed.",ha="center",fontsize=10.5)
    directory=ROOT/"docs/figures"
    directory.mkdir(parents=True,exist_ok=True)
    for extension in ("png","svg","pdf"):
        metadata={"Date":None} if extension=="svg" else {"CreationDate":None,"ModDate":None} if extension=="pdf" else None
        fig.savefig(directory/f"cem_steering_search.{extension}",dpi=200,facecolor="white",metadata=metadata)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-manifest",type=Path)
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--compact-root",type=Path)
    parser.add_argument("--native-compact-root",type=Path,default=ROOT/"artifacts/offline_study/layer-pilot-20260913-v1/compact/development-v1")
    parser.add_argument("--plots-only",action="store_true")
    args=parser.parse_args()
    if not args.plots_only:
        if not all((args.execution_manifest,args.execution_manifest_sha256,args.compact_root)):
            parser.error("Execution manifest, its SHA256, and compact root are required")
        result=run(args)
        print(result["status"])
    plots()
