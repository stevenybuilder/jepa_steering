"""Source-bound complete16 CPU analysis of frozen action-condition diagnostic B."""
from __future__ import annotations
from offline_study._paths import frozen_source_path

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/"artifacts/offline_study/layer-pilot-20260913-v1"
DATA = ROOT/"paper/data"
MANIFEST_SHA = "4bf08978fb5ae7da9a971527efd796035fda82dacfd44028e2acaff44b526341"
PROTOCOL_SHA = "0662a0b376ca15767ad4f1d0b95758fdcb5812125387aca0cd43616d26eb2dca"
SOURCE_SHA = "26158e136ac7b819c584f253b50ebf5b111975d957e6b005e68fdd30abc887e3"
CHECKPOINT_SHA = "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
DINO_SOURCE_SHA = "88b35b92ca99c27c3bd9c650d930f43e78c7e6341fb13ecfffdc26272fbf80a5"
DINO_WEIGHT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
TASKS = ("reach","reach-wall")
BANKS = ("original","fresh")
MODALITIES = ("visual","proprio","official")
EXPECTED = {(t,e) for t in TASKS for e in range(8)}
ARMS = {"native",*(f"{mode}_B{layer}" for mode in ("permute","random") for layer in range(6))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_module(path=None):
    path = path or frozen_source_path("action_condition_specificity.py", SOURCE_SHA)
    if sha(path) != SOURCE_SHA:
        raise ValueError("Frozen statistical/source implementation changed")
    spec = importlib.util.spec_from_file_location("frozen_action_condition_specificity",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_paths(root):
    paths = {(t,e):Path(root)/t/f"episode-{e}" for t,e in EXPECTED}
    # Check all preservation names before opening any experimental case values.
    if not all((p/name).is_file() for p in paths.values() for name in ("report.json","DONE.json","scores.json","CLOUD_VERIFIED.json")):
        raise ValueError("Incomplete16 preserved cases; no partial outcome aggregate")
    found = {(p.parent.parent.name,int(p.parent.name.split("-")[-1]))
             for p in Path(root).glob("*/episode-*/report.json")}
    if found != EXPECTED:
        raise ValueError("Unexpected or incomplete16 scenario directories")
    return paths


def finite(values,shape=(300,)):
    values = np.asarray(values,dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Invalid finite compact metric array")
    return values


def member_binding(cloud,key,name,digest):
    suffix = f"{key[0]}/episode-{key[1]}/{name}"
    matches = [(p,v) for p,v in cloud["verified_archive_members"].items() if p == suffix or p.endswith("/"+suffix)]
    if len(matches) != 1 or matches[0][1] != digest:
        raise ValueError("Missing unique full-archive member binding: "+suffix)
    return matches[0][0]


def validate_case(directory,key,binding,module,expected_gpu):
    report,done = read(directory/"report.json"),read(directory/"DONE.json")
    cloud_bytes = (directory/"CLOUD_VERIFIED.json").read_bytes()
    cloud = json.loads(cloud_bytes)
    if report.get("status") != "complete_action_condition_case" or report["input_binding"] != binding:
        raise ValueError("Case identity/status mismatch")
    if report["execution_manifest_sha256"] != MANIFEST_SHA or report["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Wrong scientific execution binding")
    if done["report_sha256"] != sha(directory/"report.json") or done["files"] != report["files"]:
        raise ValueError("Original completion hash mismatch")
    if set(report["files"]) != {"scores.json","actions.pt"}:
        raise ValueError("Unexpected complete payload members")
    if report["scientific_forward_count"] != 26 or report["total_forward_count"] != 28:
        raise ValueError("Scientific/engineering forecast counts differ")
    for field in ("global_rng_unchanged","parameters_unchanged","inputs_unchanged","fresh_bank_native_sampler_byte_and_rng_parity"):
        if report.get(field) is not True:
            raise ValueError("Missing execution-attested exact parity: "+field)
    for field in ("physical_outcomes_measured","fresh_confirmation","training_performed","tf32_matmul","tf32_cudnn"):
        if report.get(field) is not False:
            raise ValueError("Unexpected evidence/precision scope: "+field)
    if report["backend_provenance"].get("dino_loader_source") != "verified_local_cache_no_network_branch_resolution" or not report.get("gpu_uuid"):
        raise ValueError("Missing receiving encoder/GPU provenance")
    backend = report["backend_provenance"]
    if backend.get("checkpoint_sha256") != CHECKPOINT_SHA or backend.get("dino_source_sha256") != DINO_SOURCE_SHA or backend.get("dino_weights_sha256") != DINO_WEIGHT_SHA:
        raise ValueError("Unexpected checkpoint or external encoder identity")
    if backend.get("precision") != "float32" or backend.get("allow_tf32") is not False:
        raise ValueError("Unexpected backend precision")
    if report["gpu_uuid"].removeprefix("GPU-").lower() != expected_gpu.removeprefix("GPU-").lower():
        raise ValueError("Case GPU differs from frozen receiver assignment")
    if report["fresh_bank_seed"] != binding["candidate_seed"] ^ 0x40000000:
        raise ValueError("Fresh bank seed differs from prospective rule")
    if any(cloud.get(k) is not True for k in ("gcs_download_sha256_verified","all_report_files_hash_verified")) or cloud.get("raw_preservation_pending") is not False:
        raise ValueError("Require fully verified durable original archives")
    if cloud.get("archive_scope") != "complete-cohort-shard-original-files":
        raise ValueError("Unexpected archive verification scope")
    members = {}
    for name in ("report.json","DONE.json","scores.json"):
        digest = sha(directory/name)
        if cloud["compact_sha256"].get(name) != digest:
            raise ValueError("Compact source byte mismatch")
        members[name] = member_binding(cloud,key,name,digest)
    members["actions.pt"] = member_binding(cloud,key,"actions.pt",report["files"]["actions.pt"])
    if sha(directory/"scores.json") != report["files"]["scores.json"]:
        raise ValueError("Scores are not bound to original report")
    payload = read(directory/"scores.json")
    if payload["input_binding"] != binding or payload["execution_manifest_sha256"] != MANIFEST_SHA or payload["protocol_sha256"] != PROTOCOL_SHA or payload["bank_receipts"] != report["bank_receipts"]:
        raise ValueError("Scores identity or bank receipts mismatch")
    if set(payload["banks"]) != set(BANKS) or set(report["bank_receipts"]) != set(BANKS):
        raise ValueError("Missing original/fresh bank")
    if report["bank_receipts"]["original"]["actions_sha256"] == report["bank_receipts"]["fresh"]["actions_sha256"]:
        raise ValueError("Fresh bank duplicates original")
    metrics,norms = [],[]
    for bank,arms in payload["banks"].items():
        bank_receipt = report["bank_receipts"][bank]
        if set(arms) != ARMS or bank_receipt["actions_shape"] != [6,300,20] or bank_receipt["dtype"] != "torch.float32" or bank_receipt["forecast_calls"] != 14 or bank_receipt["scientific_arms"] != 13:
            raise ValueError("Incomplete bank arm/input contract")
        if bank_receipt["native_zero_full_forecast_and_score_byte_parity"] is not True or bank_receipt["candidate0_zero"] is not True:
            raise ValueError("Missing native zero/input parity")
        for arm,record in arms.items():
            scores = record["scores"]
            if set(scores) != set(MODALITIES):
                raise ValueError("Missing modality-resolved cost")
            vectors = {m:finite(scores[m]["costs"]) for m in MODALITIES}
            reconstructed = vectors["visual"].astype(np.float32)+np.float32(.1)*vectors["proprio"].astype(np.float32)
            if not np.array_equal(vectors["official"].astype(np.float32),reconstructed):
                raise ValueError("Official cost does not reproduce FP32 visual+.1proprio")
            for modality in MODALITIES:
                native = arms["native"]["scores"][modality]
                recomputed = module.agreement(native["costs"],scores[modality]["costs"],native["elite_indices"],scores[modality]["elite_indices"])
                if arm == "native":
                    continue
                saved = record["agreement"][modality]
                for name,value in recomputed.items():
                    if (value is None) != (saved[name] is None) or (value is not None and not np.isclose(value,saved[name],rtol=1e-11,atol=1e-13)):
                        raise ValueError("Saved rank/score metric disagrees with compact arrays")
                mode,layer = arm.split("_B")
                metrics.append(dict(task=key[0],episode=key[1],bank=bank,layer=int(layer),mode=mode,modality=modality,**recomputed))
            if arm == "native":
                continue
            audit = record["condition_audit"]
            requested,delivered,relative = (finite(audit[k]) for k in ("requested_delta_l2","delivered_delta_l2","relative_norm_error"))
            if np.any(requested < 0) or np.any(delivered < 0):
                raise ValueError("Negative intervention norm")
            positive = requested > 0
            recomputed = np.zeros(300)
            recomputed[positive] = np.abs(delivered[positive]-requested[positive])/requested[positive]
            if np.any(delivered[~positive] != 0) or recomputed.max() > 1e-5 or not np.allclose(recomputed,relative,atol=1e-14,rtol=1e-9):
                raise ValueError("Delivered norm audit failed")
            if audit["zero_target_count"] != int((~positive).sum()) or not np.isclose(audit["max_relative_norm_error"],recomputed.max(),rtol=1e-9,atol=1e-14):
                raise ValueError("Norm summary mismatch")
            if audit["native_condition_sha256"] != bank_receipt["native_condition_sha256"][layer] or audit["condition_shape"][:2] != [300,2]:
                raise ValueError("Condition cache/layout binding mismatch")
            expected_seed = module.random_seed(key[0],key[1],bank,int(layer)) if mode == "random" else None
            if audit["random_seed"] != expected_seed:
                raise ValueError("Private random seed mismatch")
            partner = arms[f"{'random' if mode=='permute' else 'permute'}_B{layer}"]["condition_audit"]
            if not np.array_equal(requested,finite(partner["requested_delta_l2"])):
                raise ValueError("Permutation and random requested norms differ")
            norms.append(dict(task=key[0],episode=key[1],bank=bank,layer=int(layer),mode=mode,
                              requested_l2_mean=float(requested.mean()),delivered_l2_mean=float(delivered.mean()),
                              max_relative_norm_error=float(recomputed.max()),zero_target_count=int((~positive).sum())))
    provenance = dict(task=key[0],episode=key[1],report_sha256=sha(directory/"report.json"),
                      scores_sha256=sha(directory/"scores.json"),actions_sha256=report["files"]["actions.pt"],
                      cloud_receipt_sha256=hashlib.sha256(cloud_bytes).hexdigest(),cloud_uri=cloud["cloud_uri"],
                      generation=cloud["generation"],cloud_archive_sha256=cloud["sha256"],verified_archive_members=members,
                      gpu_uuid=report["gpu_uuid"],input_binding=binding,
                      execution_attested=["full H1-H6 native zero-clone byte parity","unchanged experiment RNG/parameters/inputs","native sampler action-byte/private-RNG parity"],
                      cpu_verified=["all compact source bytes","all modality goal costs","rank/elite/score metrics","all delivered norm scalar audits"],
                      full_h6_retained=False,bulk_actions_locally_reloaded=False)
    return payload,metrics,norms,provenance


def run(args):
    # Exact prospective identities are read before any experimental case values.
    if sha(args.manifest) != MANIFEST_SHA or sha(args.protocol) != PROTOCOL_SHA:
        raise ValueError("Unexpected prospective execution/protocol hashes")
    manifest,protocol = read(args.manifest),read(args.protocol)
    if manifest["scenarios"] != {t:list(range(8)) for t in TASKS} or protocol["scenarios"] != manifest["scenarios"]:
        raise ValueError("Changed complete16 registry")
    if manifest["protocol_sha256"] != PROTOCOL_SHA or manifest["source_sha256"]["action_condition_specificity.py"] != SOURCE_SHA:
        raise ValueError("Frozen source/protocol mismatch")
    if manifest["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Unexpected checkpoint")
    if sha(args.input_manifest) != manifest["input_manifest_sha256"]:
        raise ValueError("Input manifest hash mismatch")
    registry = read(args.input_manifest)["records"]
    bindings = {(r["task"],r["episode"]):r for r in registry}
    if len(bindings) != len(registry) or not EXPECTED <= set(bindings):
        raise ValueError("Missing or duplicate canonical inputs")
    paths = complete_paths(args.compact_root)
    assignments = [(r["task"],r["episode"],manifest["receivers"][slot]["gpu_uuid"])
                   for slot,records in manifest["slots"].items() for r in records]
    receivers = {(t,e):gpu for t,e,gpu in assignments}
    if len(assignments) != 16 or set(receivers) != EXPECTED:
        raise ValueError("Incomplete or duplicate physical receiver assignments")
    module = frozen_module()
    payloads,metrics,norms,sources = [],[],[],[]
    for key,directory in sorted(paths.items()):
        payload,rows,audits,source = validate_case(directory,key,bindings[key],module,receivers[key])
        payloads.append(payload);metrics.extend(rows);norms.extend(audits);sources.append(source)
    frames = {"case_metrics":pd.DataFrame(metrics),"case_metrics_norms":pd.DataFrame(norms),
              "contrasts":pd.DataFrame(module.summarize_complete(payloads))}
    if len(frames["case_metrics"]) != 1152 or len(frames["case_metrics_norms"]) != 384 or len(frames["contrasts"]) != 144:
        raise ValueError("Incomplete all-arm/layer/modality exported coverage")
    outputs = {}
    for stem,frame in frames.items():
        path = DATA/f"action_condition_{stem}.csv"
        frame.to_csv(path,index=False)
        outputs[str(path.relative_to(ROOT))] = dict(sha256=sha(path),rows=len(frame))
    result = dict(schema_version=1,status="complete16_exploratory_action_condition",scenarios=16,n_per_task=8,
                  banks_per_scenario=2,layers=6,scientific_arms_per_bank=13,total_forward_count=448,
                  execution_manifest_sha256=MANIFEST_SHA,protocol_sha256=PROTOCOL_SHA,frozen_source_sha256=SOURCE_SHA,
                  analysis_source_sha256=sha(__file__),input_manifest_sha256=sha(args.input_manifest),sources=sources,outputs=outputs,
                  all_raw_preservation_complete=True,physical_outcomes_measured=False,fresh_confirmation=False,
                  unique_cloud_archive_count=len({(s["cloud_uri"],s["generation"]) for s in sources}),
                  interpretation="Native candidate ranking is a reference, not physical action quality; no planning-emergence claim",
                  uncertainty="n8 paired scenarios per task,20k draws; simultaneous six-layer centered-bootstrap max-deviation bands within task/bank/modality/contrast family",
                  full_h6_tensors_retained=False)
    (DATA/"action_condition_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    receipt = read(DATA/"action_condition_summary.json")
    if receipt.get("status") != "complete16_exploratory_action_condition" or receipt.get("scenarios") != 16 or receipt.get("n_per_task") != 8:
        raise ValueError("Plot requires complete16 receipt")
    for name,info in receipt["outputs"].items():
        if sha(ROOT/name) != info["sha256"]:
            raise ValueError("Public action-condition table hash changed")
    frame = pd.read_csv(DATA/"action_condition_contrasts.csv")
    frame = frame[frame.modality=="official"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"svg.fonttype":"none",
                         "pdf.fonttype":42,"svg.hashsalt":"action-condition-specificity-v1"})
    fig,axes = plt.subplots(2,2,figsize=(6.8,6.1),sharex=True)
    fig.subplots_adjust(left=.12,right=.98,top=.80,bottom=.24,hspace=.38,wspace=.34)
    for row,task in enumerate(TASKS):
        for col,(metric,label) in enumerate((("excess_spearman_loss","Excess Spearman loss"),("excess_top10_overlap_loss","Excess top-10 overlap loss"))):
            ax = axes[row,col]
            family = frame[frame.metric==metric]
            finite_bounds = family[["simultaneous_95_low","simultaneous_95_high"]].to_numpy().ravel()
            finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
            low,high = (min(0.,finite_bounds.min()),max(0.,finite_bounds.max())) if len(finite_bounds) else (-.1,.1)
            pad = max((high-low)*.1,.002)
            for bank,color,title in zip(BANKS,("#147d83","#be762c"),("Original action bank","Fresh action bank")):
                block = family[(family.task==task)&(family.bank==bank)].sort_values("layer")
                if len(block)!=6 or set(block.layer)!=set(range(6)) or set(block.n)!={8}:
                    raise ValueError("Incomplete all-six-layer plot family")
                if block["mean"].notna().all():
                    ax.plot(block.layer,block["mean"],marker="o",ms=3,color=color,label=title,lw=1.5)
                    ax.fill_between(block.layer,block.simultaneous_95_low,block.simultaneous_95_high,color=color,alpha=.15,lw=0)
                else:
                    ax.text(.5,.8 if bank=="original" else .6,title+": undefined",transform=ax.transAxes,ha="center",fontsize=10.5,color=color)
            ax.axhline(0,color="#89929b",lw=.8)
            ax.set_ylim(low-pad,high+pad)
            ax.set_xticks(range(6),[f"B{i}" for i in range(6)])
            ax.grid(alpha=.15)
            ax.spines[["top","right"]].set_visible(False)
            if row==0:ax.set_title(label,fontsize=10.5)
            if row==1:ax.set_xlabel("H3 action-condition intervention")
            if col==0:ax.set_ylabel("Reach" if task=="reach" else "Reach-Wall")
    fig.suptitle("Action alignment versus norm-matched random edits",fontsize=12,y=.98)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.54,.93),ncol=2,frameon=False,fontsize=10.5)
    fig.text(.5,.105,"Positive: permutation disrupts native rankings more than random.",ha="center",fontsize=10.5)
    fig.text(.5,.060,"All 16 cases; n=8/task. Six-layer simultaneous 95% bands.",ha="center",fontsize=10.5)
    fig.text(.5,.017,"Official visual + 0.1 proprio goal cost. No actions executed.",ha="center",fontsize=10.5)
    out=ROOT/"docs/figures";out.mkdir(parents=True,exist_ok=True)
    for ext in ("png","svg","pdf"):
        meta={"Date":None} if ext=="svg" else {"CreationDate":None,"ModDate":None} if ext=="pdf" else None
        fig.savefig(out/f"action_condition_specificity.{ext}",dpi=220,facecolor="white",metadata=meta)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=BASE/"action-condition-execution-manifest-v2.json")
    parser.add_argument("--protocol",type=Path,default=DATA/"action_condition_protocol.json")
    parser.add_argument("--input-manifest",type=Path,default=BASE/"development-inputs-64/INPUT_MANIFEST.json")
    parser.add_argument("--compact-root",type=Path,default=BASE/"compact/action-condition-v2")
    parser.add_argument("--plots-only",action="store_true")
    args=parser.parse_args()
    if not args.plots_only:
        print(run(args)["status"])
    plots()
