"""Source-bound CPU summaries of complete frozen development mechanism cohorts.

Never starts a model or simulator. Eight cases are instrumentation-only; 64 must
contain every frozen ID. Bulk H6 tensors remain in the verified cloud archive.
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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data"
PROTOCOL = DATA / "pilot_summary_protocol.json"
BASE = ROOT / "artifacts/offline_study/layer-pilot-20260913-v1"
TASKS = ("reach", "reach-wall")
ARMS = ("fixed_rank4", "matched_random_fixed_rank4")
COMPONENTS = ("full", "cached_full", "zero", "mean_only", "centered_only")
ATTENTION = ("spatial_distance_patches", "temporal_distance_frames", "conditioning_mass", "visual_mass")
CEM = ("proposal_entropy_nats", "proposal_std_mean", "best_cost", "best_runnerup_margin", "elite_boundary_margin")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def expected_cases(cohort):
    if cohort not in (8, 64):
        raise ValueError("Only the frozen8 instrumentation or complete64 cohort is allowed")
    return {(task, episode) for task in TASKS for episode in range(cohort//2)}


def discover(root, expected):
    """Check complete names before reading any per-case results."""
    cases = {}
    root = Path(root)
    # The prospective actual-CEM addon shares episode IDs but is a separate
    # experiment. Never collect its reports into the development replay cohort.
    cohort_roots = [root/name for name in ("development-v1", "expansion-v2") if (root/name).is_dir()]
    paths = (path for directory in (cohort_roots or [root]) for path in directory.rglob("report.json"))
    for path in paths:
        match = re.fullmatch(r"episode-(\d+)", path.parent.name)
        if match and path.parent.parent.name in TASKS:
            key = (path.parent.parent.name, int(match.group(1)))
            if key in expected:
                if key in cases:
                    raise ValueError("Duplicate case directories")
                cases[key] = path.parent
    if set(cases) != expected:
        raise ValueError(f"Incomplete frozen cohort: {len(cases)}/{len(expected)}; no aggregate generated")
    if any(not (path/"DONE.json").exists() or not (path/"CLOUD_VERIFIED.json").exists() for path in cases.values()):
        raise ValueError("Incomplete frozen cohort preservation; no result values read")
    return cases


def finite(value, shape=None):
    array = np.asarray(value, dtype=float)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError("Unexpected shape or nonfinite metric")
    return array


def score_metrics(reference, costs, native_elites, elites):
    reference, costs = finite(reference, (300,)), finite(costs, (300,))
    for index, values in ((native_elites, reference), (elites, costs)):
        ids = np.asarray(index)
        if ids.shape != (10,) or ids.dtype.kind not in "iu" or len(set(ids)) != 10 or min(ids) < 0 or max(ids) >= 300:
            raise ValueError("Invalid actual elite IDs")
        # Native kernel determines ties; verify membership without replacing its IDs.
        if values[ids].max() > np.delete(values, ids).min():
            raise ValueError("Saved elites are not a valid minimum-cost set")
    delta = costs-reference
    centered = delta-delta.mean()
    ordered = np.sort(costs)
    return dict(score_change_mean=float(delta.mean()), score_change_rms=float(np.sqrt(np.mean(delta**2))),
                centered_score_change_rms=float(np.sqrt(np.mean(centered**2))),
                native_elite_overlap_count=len(set(native_elites)&set(elites)),
                best_runnerup_margin=float(ordered[1]-ordered[0]),
                elite_boundary_margin=float(ordered[10]-ordered[9]))


def reconstruction(reference, full, component):
    full_delta = np.asarray(full, float)-reference
    full_delta -= full_delta.mean()
    delta = np.asarray(component, float)-reference
    delta -= delta.mean()
    denominator = float(full_delta@full_delta)
    return None if denominator == 0 else float(1-((full_delta-delta)@(full_delta-delta))/denominator)


def attention_rows(payload, key):
    if payload.get("output_byte_parity") is not True or payload.get("same_input_rng_unchanged") is not True:
        raise ValueError("Attention parity failed or unavailable")
    rows, cells = [], set()
    for item in payload["attention"]:
        cell = item["layer"], item["forecast_horizon"]
        if cell in cells:
            raise ValueError("Duplicate attention cell")
        cells.add(cell)
        values = {m: finite(item[m], (1,16))[0] for m in ATTENTION}
        if any(np.any(values[m] < 0) for m in ATTENTION):
            raise ValueError("Negative attention distance or mass")
        if not np.allclose(values["visual_mass"]+values["conditioning_mass"], 1., rtol=1e-5, atol=1e-6):
            raise ValueError("Attention mass identity failed")
        for head in range(16):
            rows.append(dict(task=key[0], episode=key[1], layer=cell[0], horizon=cell[1], head=head,
                             T=item["T"], H=item["H"], W=item["W"], action_tokens=item["action_tokens"],
                             **{m: float(v[head]) for m,v in values.items()}))
    if cells != {(l,h) for l in range(6) for h in range(1,7)}:
        raise ValueError("Incomplete all-layer H1-H6 attention")
    return rows


def cem_rows(payload, key):
    if isinstance(payload, dict):
        payload = payload["iterations"]
    if len(payload) != 15 or [r["iteration"] for r in payload] != list(range(15)):
        raise ValueError("Incomplete native15-iteration CEM trace")
    rows = []
    for item in payload:
        values = {m: float(item[m]) for m in CEM}
        finite(list(values.values()))
        verified = False
        if "proposal_std" in item and "objective_costs" in item and "elite_indices" in item:
            std = finite(item["proposal_std"])
            if std.size != 120 or np.any(std <= 0):
                raise ValueError("Unexpected Gaussian action-space proposal")
            costs = finite(item["objective_costs"], (300,))
            measured = score_metrics(costs, costs, item["elite_indices"], item["elite_indices"])
            entropy = float(np.sum(np.log(std)+.5*math.log(2*math.pi*math.e)))
            for actual, expected in ((entropy, values["proposal_entropy_nats"]),
                                     (std.mean(), values["proposal_std_mean"]),
                                     (costs.min(), values["best_cost"]),
                                     (measured["best_runnerup_margin"], values["best_runnerup_margin"]),
                                     (measured["elite_boundary_margin"], values["elite_boundary_margin"])):
                if not np.isclose(actual, expected, rtol=1e-6, atol=1e-10):
                    raise ValueError("Compact CEM scalar disagrees with source arrays")
            verified = True
        if values["proposal_std_mean"] <= 0 or min(values["best_runnerup_margin"],values["elite_boundary_margin"]) < 0:
            raise ValueError("Invalid CEM margin or proposal std")
        rows.append(dict(task=key[0], episode=key[1], iteration=item["iteration"],
                         compact_array_recalculation=verified, **values))
    return rows


def replay_rows(report, scores, key):
    expected = {"native"}|{f"{arm}-{c}" for arm in ARMS for c in COMPONENTS}
    if set(scores) != expected:
        raise ValueError("Incomplete frozen replay arm registry")
    native = finite(scores["native"]["objective_costs"], (300,))
    native_elites = scores["native"]["elite_indices"]
    summaries = {r["arm"]: r for r in report["summaries"]}
    if len(summaries) != len(report["summaries"]):
        raise ValueError("Duplicate replay summary")
    rows, audit = [], []
    for arm in ARMS:
        full = finite(scores[f"{arm}-full"]["objective_costs"], (300,))
        values = summaries[arm]
        for check in ("component_audit", "cached_full_exact_forecast_and_score_parity",
                      "zero_exact_forecast_and_score_parity", "no_component_renormalization"):
            if values.get(check) is not True:
                raise ValueError("Replay parity or no-renormalization audit missing")
        common = finite(scores[f"{arm}-mean_only"]["objective_costs"], (300,))
        centered = finite(scores[f"{arm}-centered_only"]["objective_costs"], (300,))
        recomputed = {"common_centered_cost_reconstruction": reconstruction(native,full,common),
                      "centered_centered_cost_reconstruction": reconstruction(native,full,centered)}
        for metric, value in recomputed.items():
            saved = values[metric]
            if (value is None) != (saved is None) or (value is not None and not np.isclose(value,saved,rtol=1e-8,atol=1e-10)):
                raise ValueError("Cached-component cost reconstruction mismatch")
        energy = {m: float(values[m]) for m in ("common_field_energy_fraction","centered_field_energy_fraction")}
        if min(energy.values()) < 0 or not np.isclose(sum(energy.values()),1.,rtol=1e-5,atol=1e-7):
            raise ValueError("Common/centered energy identity failed")
        audit.append(dict(task=key[0],episode=key[1],arm=arm,**energy,**recomputed,
                          additive_components_cost_reconstruction=reconstruction(native,full,common+centered-native),
                          cached_full_score_exact=True,zero_score_exact=True,
                          forecast_parity_execution_attested=True,no_component_renormalization=True))
        for component in COMPONENTS:
            name = f"{arm}-{component}"
            costs = finite(scores[name]["objective_costs"], (300,))
            metrics = score_metrics(native,costs,native_elites,scores[name]["elite_indices"])
            for metric,value in metrics.items():
                if not np.isclose(value,summaries[name][metric],rtol=1e-8,atol=1e-10):
                    raise ValueError("Saved replay cost summary mismatch")
            if component in ("cached_full","zero") and not np.array_equal(costs, full if component=="cached_full" else native):
                raise ValueError("Exact score-cache identity failed")
            rows.append(dict(task=key[0],episode=key[1],arm=arm,component=component,**metrics))
    return rows,audit


def summarize(frame, groups, metrics, expected_n):
    """One value per scenario/cell; bootstrap scenarios, never heads/candidates."""
    protocol = read(PROTOCOL)
    rows = []
    weights = np.random.default_rng(protocol["bootstrap_seed"]).multinomial(
        expected_n,np.full(expected_n,1/expected_n),size=protocol["bootstrap_replicates"])/expected_n
    for key, block in frame.groupby(groups[0] if len(groups)==1 else groups, sort=True):
        key = key if isinstance(key,tuple) else (key,)
        block = block.sort_values("episode")
        if block.episode.duplicated().any() or len(block) != expected_n:
            raise ValueError("Wrong scenario-level aggregation unit")
        for metric in metrics:
            values = block[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                rows.append(dict(zip(groups,key),metric=metric,n=len(values),n_defined=int(np.isfinite(values).sum()),
                                 mean=None,marginal_95_low=None,marginal_95_high=None,
                                 undefined_reason="At least one frozen scenario has undefined zero-denominator reconstruction; no selective mean"))
                continue
            draws = weights@values
            rows.append(dict(zip(groups,key),metric=metric,n=len(values),n_defined=len(values),mean=float(values.mean()),
                             marginal_95_low=float(np.quantile(draws,.025)),marginal_95_high=float(np.quantile(draws,.975))))
    return pd.DataFrame(rows)


def load_manifest_set(execution_paths, input_paths, cohort):
    protocol = read(PROTOCOL)
    manifests = {sha(path):read(path) for path in execution_paths}
    inputs = {sha(path):read(path) for path in input_paths}
    expected = expected_cases(cohort)
    frozen = {protocol["initial_execution_manifest_sha256"]}
    if cohort==64:
        frozen.add(protocol["expanded_execution_manifest_sha256"])
    if set(manifests) != frozen:
        raise ValueError("Unexpected execution manifest set for frozen cohort")
    coverage, bindings, routes = set(), {}, {}
    for digest, manifest in manifests.items():
        if manifest["input_manifest_sha256"] not in inputs:
            raise ValueError("Missing exact input manifest")
        declared = {(task,int(episode)) for task,episodes in manifest["scenarios"].items() for episode in episodes}
        records = inputs[manifest["input_manifest_sha256"]]["records"]
        registry = {(r["task"],r["episode"]):r for r in records}
        if len(registry) != len(records) or not declared <= set(registry):
            raise ValueError("Input registry duplicates or lacks declared scenarios")
        for key in declared & expected:
            if key in bindings and bindings[key] != registry[key]:
                raise ValueError("Overlapping manifests disagree on immutable inputs")
            bindings[key] = registry[key]
            # The expansion includes original IDs in its inventory, but does
            # not authorize replacing their initial8 source/run bindings.
            if (key[1]<4) == (digest==protocol["initial_execution_manifest_sha256"]):
                routes.setdefault(key,[]).append(digest)
        coverage |= declared
    if not expected <= coverage:
        raise ValueError("Execution manifests do not cover frozen cohort")
    if cohort == 8 and protocol["initial_execution_manifest_sha256"] not in manifests:
        raise ValueError("Eight-case run lacks its frozen original execution manifest")
    checkpoints = {m["checkpoint_sha256"] for m in manifests.values()}
    if len(checkpoints) != 1:
        raise ValueError("Mixed checkpoint identities")
    for task in TASKS:
        if len({m["fit_bank_sha256"][task] for m in manifests.values()}) != 1:
            raise ValueError("Mixed task fitted-bank identities")
    if cohort==64:
        expanded = manifests[protocol["expanded_execution_manifest_sha256"]]
        if expanded["initial8_execution_manifest_sha256"] != protocol["initial_execution_manifest_sha256"] or expanded["initial8_reused_once"] is not True:
            raise ValueError("Missing initial8 reuse binding")
    return manifests, bindings, routes


def validate_case(directory,key,binding,routes):
    report_path = directory/"report.json"
    report, done = read(report_path), read(directory/"DONE.json")
    cloud_bytes = (directory/"CLOUD_VERIFIED.json").read_bytes()
    cloud = json.loads(cloud_bytes)
    cloud_digest = hashlib.sha256(cloud_bytes).hexdigest()
    if done["report_sha256"] != sha(report_path) or done["files"] != report["files"]:
        raise ValueError("Report completion hash mismatch")
    if report["input_binding"] != binding:
        raise ValueError("Case immutable input binding mismatch")
    if report.get("execution_manifest_sha256") is not None and report["execution_manifest_sha256"] not in routes:
        raise ValueError("Case execution-manifest mismatch")
    if key[1]>=4 and (report.get("execution_manifest_sha256") not in routes or report.get("global_rng_unchanged") is not True):
        raise ValueError("Expanded case lacks explicit execution or final RNG identity")
    if report["status"] != "canonical_development_mechanism_case_complete" or report["physical_outcomes_measured"] is not False:
        raise ValueError("Unexpected case status or physical-outcome scope")
    for field,value in (("attention_candidate_count",1),("attention_blocks",6),("attention_horizons",6),
                        ("cem_iterations",15),("cem_candidates_per_iteration",300),("shared_candidate_count",300),
                        ("shared_candidate_banks",1),("base_arms",2),("components_per_arm",5),("cem_output_and_rng_parity",True)):
        if report[field] != value:
            raise ValueError("Unexpected frozen diagnostic dimensions or CEM parity")
    compact_only=cloud.get("archive_scope")=="compact-derived-only"
    if cloud.get("gcs_download_sha256_verified") is not True:
        raise ValueError("Cloud preservation/hash audit unavailable")
    if compact_only:
        if (cloud.get("compact_files_hash_verified") is not True or cloud.get("raw_preservation_pending") is not True
                or cloud.get("original_worker_report_files_sha256_verified") is not True
                or cloud.get("all_report_files_hash_verified") is True):
            raise ValueError("Ambiguous compact-only versus fullraw preservation scope")
        names={"report.json","DONE.json","attention.json","scores.json","cem_summary.json"}
        members=cloud.get("verified_compact_members",{})
        if set(members)!=names or members!=cloud.get("compact_sha256"):
            raise ValueError("Incomplete verified compact-member inventory")
        if any(sha(directory/name)!=digest for name,digest in members.items()):
            raise ValueError("Compact-cloud member hash mismatch")
    elif cloud.get("all_report_files_hash_verified") is not True or cloud.get("raw_preservation_pending") is True:
        raise ValueError("Fullraw cloud preservation/hash audit unavailable")
    verified = {}
    for name in ("attention.json","scores.json"):
        digest = sha(directory/name)
        if digest != report["files"][name]:
            raise ValueError("Compact original source hash mismatch")
        verified[name] = digest
    cem_path = directory/"cem_summary.json"
    compact_hashes = cloud.get("compact_sha256",{})
    if compact_hashes.get("cem_summary.json") != sha(cem_path):
        raise ValueError("Derived compact CEM summary lacks exact preservation binding")
    if cloud.get("parent_trace_sha256") != report["files"]["native-cem-trace.pt"]:
        raise ValueError("Derived CEM summary parent trace mismatch")
    verified["cem_summary.json"] = sha(cem_path)
    attention = read(directory/"attention.json")
    if attention["input_binding"] != binding:
        raise ValueError("Attention input binding mismatch")
    provenance = dict(task=key[0],episode=key[1],report_sha256=sha(report_path),done_sha256=sha(directory/"DONE.json"),
                      cloud_receipt_sha256=cloud_digest,compact_verified=verified,
                      cloud_uri=cloud["cloud_uri"],cloud_archive_sha256=cloud["sha256"],
                      cloud_generation=cloud.get("generation"),
                      archive_scope="compact-derived-only" if compact_only else "full-report-payload",
                      raw_preservation_pending=compact_only,
                      original_worker_files_verified=cloud.get("original_worker_report_files_sha256_verified",not compact_only),
                      remote_payload_hashes=report["files"],execution_manifest_candidates=routes,
                      legacy_execution_binding=report.get("execution_manifest_sha256") is None,
                      input_binding=binding,forecast_parity="execution-attested; bulk tensors not locally reloaded",
                      global_rng_unchanged="execution completion attests final assertion; attention and CEM parity explicitly recorded")
    replay,audit = replay_rows(report,read(directory/"scores.json"),key)
    return attention_rows(attention,key),cem_rows(read(cem_path),key),replay,audit,provenance


def run(args):
    protocol = read(PROTOCOL)
    if sha(args.intent) != protocol["expansion_intent_sha256"]:
        raise ValueError("Frozen expansion intent changed")
    manifests,bindings,routes = load_manifest_set(args.execution_manifest,args.input_manifest,args.cohort)
    cases = discover(args.compact_root,expected_cases(args.cohort))
    attention,cem,replay,audit,sources = [],[],[],[],[]
    for key,directory in sorted(cases.items()):
        a,c,r,u,p = validate_case(directory,key,bindings[key],routes[key])
        attention.extend(a); cem.extend(c); replay.extend(r); audit.extend(u); sources.append(p)
    frames = {"attention_cases":pd.DataFrame(attention),"cem_cases":pd.DataFrame(cem),
              "replay_cases":pd.DataFrame(replay),"component_cases":pd.DataFrame(audit)}
    n = args.cohort//2
    frames["attention_summary"] = summarize(frames["attention_cases"],["task","layer","horizon","head"],ATTENTION,n)
    frames["cem_summary"] = summarize(frames["cem_cases"],["task","iteration"],CEM,n)
    replay_metrics = [k for k in frames["replay_cases"] if k not in ("task","episode","arm","component")]
    frames["replay_summary"] = summarize(frames["replay_cases"],["task","arm","component"],replay_metrics,n)
    component_metrics = [k for k in frames["component_cases"] if k.endswith(("energy_fraction","cost_reconstruction"))]
    frames["component_summary"] = summarize(frames["component_cases"],["task","arm"],component_metrics,n)
    paired = frames["component_cases"].pivot(index=["task","episode"],columns="arm",values=component_metrics)
    contrast = pd.DataFrame({metric:paired[metric][ARMS[0]]-paired[metric][ARMS[1]] for metric in component_metrics}).reset_index()
    frames["learned_minus_random_summary"] = summarize(contrast,["task"],component_metrics,n)
    outputs = {}
    for name,frame in frames.items():
        path = DATA/f"pilot_summary_{name}.csv"
        frame.to_csv(path,index=False)
        outputs[str(path.relative_to(ROOT))] = dict(sha256=sha(path),rows=len(frame))
    receipt = dict(schema_version=1,status="complete64_exploratory_development" if args.cohort==64 else "complete8_instrumentation_only",
                   cohort=args.cohort,n_per_task=n,protocol_sha256=sha(PROTOCOL),analysis_source_sha256=sha(__file__),
                   expansion_intent_sha256=sha(args.intent),execution_manifests={str(p):sha(p) for p in args.execution_manifest},
                   precision_by_execution_manifest={digest:manifest["precision"] for digest,manifest in manifests.items()},
                   input_manifests={str(p):sha(p) for p in args.input_manifest},sources=sources,outputs=outputs,
                   raw_preservation_pending_case_count=sum(s["raw_preservation_pending"] for s in sources),
                   all_raw_preservation_complete=not any(s["raw_preservation_pending"] for s in sources),
                   physical_outcomes_measured=False,fresh_confirmation=False,
                   scope="Native attention for one zero-action plan per scenario; separate actual native15-iteration CEM; cached-component shared300-action replay",
                   caveats=["Differential proposal entropy is pre-clipping; negative values valid and coordinate-dependent",
                            "Action token count zero does not remove AdaLN action conditioning",
                            "Attention distances are not attention entropy or causal-head measurements",
                            "Common/centered components are not energy-renormalized; reconstruction scores can be negative",
                            "Bulk H6 parity is execution-attested rather than independently reexecuted on this CPU",
                            "Historical development scenarios, no asserted fit-disjointness or new physical success"])
    (DATA/"pilot_summary.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    return receipt


def save_figure(fig,name):
    directory = ROOT/"docs/figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png","svg","pdf"):
        metadata = {"Date":None} if extension=="svg" else {"CreationDate":None,"ModDate":None} if extension=="pdf" else None
        fig.savefig(directory/f"{name}.{extension}",dpi=180,facecolor="white",metadata=metadata)


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.hashsalt":"pilot-mechanisms-v1","axes.spines.top":False,"axes.spines.right":False})
    receipt = read(DATA/"pilot_summary.json")
    for path,source in receipt["outputs"].items():
        if sha(ROOT/path) != source["sha256"]:
            raise ValueError("Public figure input hash mismatch")
    scope = f"n={receipt['n_per_task']} scenarios/task · "+("8-case instrumentation only" if receipt["cohort"]==8 else "all 64 frozen development scenarios")
    frame = pd.read_csv(DATA/"pilot_summary_attention_summary.csv")
    for metric,short,label in ((ATTENTION[0],"spatial","Spatial distance (patches)"),(ATTENTION[1],"temporal","Temporal distance (frames)")):
        selected = frame[frame.metric==metric]
        fig,axes = plt.subplots(2,6,figsize=(15,6),sharex=True,sharey=True)
        fig.subplots_adjust(top=.81,bottom=.15,right=.91,wspace=.1,hspace=.25)
        low,high = selected["mean"].min(),selected["mean"].max()
        for row,task in enumerate(TASKS):
            for col,horizon in enumerate(range(1,7)):
                matrix = selected[(selected.task==task)&(selected.horizon==horizon)].pivot(index="layer",columns="head",values="mean")
                im = axes[row,col].imshow(matrix,aspect="auto",vmin=low,vmax=high,cmap="viridis",origin="upper")
                axes[row,col].set_title(f"H{horizon}",fontsize=10)
                axes[row,col].set_xticks([0,5,10,15]); axes[row,col].set_yticks(range(6))
                if col==0: axes[row,col].set_ylabel(f"{task}\nBlock (zero-indexed)")
                if row==1: axes[row,col].set_xlabel("Head")
        fig.colorbar(im,cax=fig.add_axes([.93,.19,.012,.58]),label=label)
        fig.suptitle("Native attention across every predictor block and head",fontsize=16,y=.97)
        fig.text(.5,.885,f"{label} · {scope}",ha="center",fontsize=11)
        fig.text(.5,.035,"One archived zero-action plan/scenario; visual-query → visual-key mass-weighted distance.\n"
                 "Scenario means; cellwise 95% intervals in CSV. Action tokens=0 does not remove AdaLN conditioning. Not causal head evidence.",ha="center",fontsize=9,color="#555f69")
        save_figure(fig,f"pilot_attention_{short}"); plt.close(fig)
    frame = pd.read_csv(DATA/"pilot_summary_cem_summary.csv")
    fig,axes = plt.subplots(2,3,figsize=(12.4,6.9))
    fig.subplots_adjust(top=.81,bottom=.14,hspace=.48,wspace=.32)
    for row,task in enumerate(TASKS):
        for col,(metric,label) in enumerate((("proposal_entropy_nats","Pre-clipping proposal entropy (nats)"),("best_runnerup_margin","Best / runner-up cost margin"),("elite_boundary_margin","Elite-boundary cost margin"))):
            f = frame[(frame.task==task)&(frame.metric==metric)].sort_values("iteration")
            ax = axes[row,col]; x=f.iteration.to_numpy()+1
            ax.plot(x,f["mean"],color="#187c80"); ax.fill_between(x,f.marginal_95_low,f.marginal_95_high,color="#187c80",alpha=.18)
            ax.set(xlabel="Native CEM iteration",title=label); ax.set_xticks([1,5,10,15]); ax.grid(alpha=.15)
            if col==0: ax.set_ylabel(task); ax.axhline(0,color="#89929b",lw=.6)
    fig.suptitle("Native CEM proposal concentration and score margins",fontsize=15,y=.97)
    fig.text(.5,.895,scope+" · 15 iterations × 300 candidates, 10 elites",ha="center")
    fig.text(.5,.02,"Marginal scenario-bootstrap 95% bands. Gaussian differential entropy may be negative; measured before clipping/mean insertion.\n"
             "This is proposal entropy, not candidate-score softmax entropy, attention entropy, or physical-outcome uncertainty.",ha="center",fontsize=9,color="#555f69")
    save_figure(fig,"pilot_attention_cem"); plt.close(fig)
    frame = pd.read_csv(DATA/"pilot_summary_component_summary.csv")
    fig,axes = plt.subplots(2,2,figsize=(11.8,7.5))
    fig.subplots_adjust(top=.82,bottom=.14,hspace=.48,wspace=.29)
    for row,task in enumerate(TASKS):
        for col,suffix in enumerate(("field_energy_fraction","centered_cost_reconstruction")):
            for index,(arm,component) in enumerate((a,c) for a in ARMS for c in ("common","centered")):
                metric = f"{component}_{suffix}"
                f = frame[(frame.task==task)&(frame.arm==arm)&(frame.metric==metric)].iloc[0]
                color = "#187c80" if arm==ARMS[0] else "#c57d36"
                if np.isfinite(f["mean"]):
                    axes[row,col].errorbar(index,f["mean"],yerr=[[f["mean"]-f.marginal_95_low],[f.marginal_95_high-f["mean"]]],fmt="o",color=color,capsize=3)
                else:
                    axes[row,col].text(index,.08,"undefined",ha="center",rotation=90,fontsize=8,
                                       transform=axes[row,col].get_xaxis_transform())
            axes[row,col].set_xticks(range(4),["Learned\ncommon","Learned\ncentered","Random\ncommon","Random\ncentered"])
            axes[row,col].set_ylabel(task); axes[row,col].grid(axis="y",alpha=.15)
            axes[row,col].axhline(0,color="#89929b",lw=.6); axes[row,col].axhline(1,color="#89929b",lw=.6,ls="--")
            axes[row,col].set_title("Fraction of full activation-edit energy" if col==0 else "Centered score-change reconstruction: 1 − SSE / energy")
            if col==0: axes[row,col].set_ylim(-.025,1.025)
    fig.suptitle("Which cached component reproduces the full score change?",fontsize=15,y=.97)
    fig.text(.5,.905,scope+" · same 300 candidate actions per scenario",ha="center")
    fig.text(.5,.02,"No component-energy renormalization. Reconstruction can be negative; it is not a physical-success rate.\n"
             "Marginal paired-scenario 95% intervals; full-cache/zero score parity checked locally, forecast parity execution-attested.",ha="center",fontsize=9,color="#555f69")
    save_figure(fig,"pilot_replay_components"); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",type=int,choices=(8,64),default=64)
    parser.add_argument("--execution-manifest",type=Path,action="append")
    parser.add_argument("--input-manifest",type=Path,action="append")
    parser.add_argument("--compact-root",type=Path,default=BASE/"compact")
    parser.add_argument("--intent",type=Path,default=BASE/"expansion64-intent.json")
    parser.add_argument("--plots-only",action="store_true")
    args = parser.parse_args()
    if not args.plots_only:
        if not args.execution_manifest or not args.input_manifest:
            parser.error("Execution and input manifests required before any result is read")
        result = run(args)
        print(f"Completed {result['cohort']} cases: {result['status']}")
    plots()
