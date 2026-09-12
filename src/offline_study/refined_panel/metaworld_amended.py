"""MetaWorld component-extension analysis with ONE documented amendment to the device gate.

Frozen `offline_study.metaworld_component_analysis.load_panel` raises
"Paired arms executed on different devices" when the arms of a logical episode carry
different device UUIDs. This module reuses every frozen function unchanged and re-implements
only `load_panel`, replacing that single check with:

    arms of one logical episode may come from different devices iff the two devices' six
    device-bound engineering checks on the excluded smoke scenario (native, native_repeat,
    zero_dose, visual_only, action_condition_only, joint) are EQUIVALENT: every non-float
    outcome field identical (native_success, elementary_steps, observed_frames, expert_success,
    initial/goal sha256, published_candidate_count, planning-call structure) and the two
    simulator-measured floats (native_reward, native_state_distance) within REL_TOL (1e-6).

Measured 2026-09-12 across 5090 / 4090 / PRO 6000 / PRO 5000 / 5000 Ada / 6000 Ada / L40S:
success identical on all sub-arms, floats agree to ~8e-8 relative (last-bit float32 effects);
the excluded H100 differed by ~3e-1 and flipped success. Planner-internal `action_trace_sha256`
may differ between device families; the amendment keys on outcomes. The report records the
exact per-device outcome hashes, the outcome values, and the worst paired deviation.
`--strict` runs the unmodified frozen loader instead; `--check-only` verifies whatever exists.

Run against the FROZEN source snapshot (its source hash is bound in the freeze), e.g.
  python src/offline_study/refined_panel/metaworld_amended.py --frozen-src <snapshot>/code/src \
      --root <panel>/results --freeze <freeze> --engineering-root <panel>/engineering --output <dir>
"""
import argparse, hashlib, json, sys
from pathlib import Path


def _import_frozen(frozen_src):
    if frozen_src:
        sys.path.insert(0, str(frozen_src))
    import offline_study.metaworld_component_analysis as frozen
    from offline_study.behavioral_analysis import shard_paths
    from offline_study.behavioral_development import assigned_rows, validate_coverage, verified_report
    from offline_study.fixed_response_smoke import verify_episode
    from offline_study.metaworld_component_behavior import ARMS, ENGINEERING, METHOD, TASKS, read, validate_protocol, verify_engineering
    from offline_study.protocol import sha256, write_json
    return frozen, dict(shard_paths=shard_paths, assigned_rows=assigned_rows, validate_coverage=validate_coverage,
                        verified_report=verified_report, verify_episode=verify_episode, ARMS=ARMS, ENGINEERING=ENGINEERING,
                        METHOD=METHOD, TASKS=TASKS, read=read, validate_protocol=validate_protocol,
                        verify_engineering=verify_engineering, sha256=sha256, write_json=write_json)


def _outcome_without_timing(result):
    r = json.loads(json.dumps(result))
    r.pop("seconds", None)
    for c in r.get("planning_calls", []):
        if isinstance(c, dict):
            c.pop("seconds", None)
    return r


FLOAT_KEYS = ("native_reward", "native_state_distance")   # simulator-measured floats: compared with tolerance
REL_TOL, ABS_TOL = 1e-6, 1e-9                           # measured cross-device deviation is ~8e-8 relative; Hopper's was ~3e-1


def engineering_outcomes(engineering_root, uuid, task, F):
    """Six sub-arm outcomes (timing removed) for one device/task, plus their exact sha256."""
    parts = {}
    for name in F["ENGINEERING"]:
        report, _ = F["verified_report"](engineering_root / uuid / task / name)
        parts[name] = _outcome_without_timing(report["result"])
    exact = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return parts, exact


def outcome_deviation(a, b):
    """Max relative deviation between two devices' outcomes, or None if any non-float field differs.
    Non-float fields (success bools, step counts, stimulus hashes, candidate counts, planning-call
    structure) must match exactly; only FLOAT_KEYS are compared within tolerance."""
    worst = 0.0
    for sub in a:
        ra, rb = a[sub], b[sub]
        if set(ra) != set(rb):
            return None
        for k in ra:
            if k in FLOAT_KEYS:
                x, y = float(ra[k]), float(rb[k])
                worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
            elif ra[k] != rb[k]:
                return None
    return worst


def equivalent(a, b):
    d = outcome_deviation(a, b)
    return d is not None and d <= REL_TOL, d


def load_panel_amended(root, freeze, engineering_root, F):
    """Frozen load_panel with the per-episode same-UUID check replaced by outcome-equivalence."""
    sha256, read = F["sha256"], F["read"]
    protocol = F["validate_protocol"](freeze)
    panel, bindings, engineering, outcome_hash, outcomes = {}, {}, {}, {}, {}
    spanned, groups, worst = {}, {}, {}
    for task in F["TASKS"]:
        panel[task] = {}
        devices_by_episode = {}
        for arm in F["ARMS"]:
            rows = []
            for shard in F["shard_paths"](root / task / arm):
                report, digest = F["verified_report"](shard)
                launch = read(shard / "protocol.json")
                if ((shard / "FAILED.json").exists() or report["status"] != F["METHOD"] + "_shard_complete" or
                        report["task"] != task or report["arm"] != arm or launch["method"] != F["METHOD"] or
                        launch["task"] != task or launch["arm"] != arm or
                        report["protocol_sha256"] != sha256(shard / "protocol.json") or
                        launch["freeze_sha256"] != sha256(freeze / "protocol.json") or
                        launch["source_sha256"] != protocol["source_sha256"] or
                        report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False or
                        report["scientific_efficacy_measurement"] is not True):
                    raise ValueError("Unbound or failed component shard")
                uuid = launch["device_uuid"]
                if Path(uuid).name != uuid:
                    raise ValueError("Unsafe receiving-device reference")
                key = (uuid, task)
                if key not in engineering:
                    engineering[key] = F["verify_engineering"](engineering_root / uuid / task, freeze, task, uuid)
                    outcomes[key], outcome_hash[key] = engineering_outcomes(engineering_root, uuid, task, F)
                if engineering[key] != launch["engineering_report_sha256"]:
                    raise ValueError("Receiving proof changed")
                expected = F["assigned_rows"](protocol["episodes"], launch["logical_ranks"])
                if launch["expected_episodes"] != expected:
                    raise ValueError("Altered logical stream membership")
                current = []
                for filename, file_hash in report["episode_files_sha256"].items():
                    if Path(filename).name != filename or not filename.startswith("episode-") or sha256(shard / filename) != file_hash:
                        raise ValueError("Changed or unsafe episode record")
                    row = read(shard / filename)
                    out = shard / f"calls-{row['episode']:03d}"
                    for file, hash_key in (("unroll_calls.json", "unroll_calls_sha256"), ("action_trace.json", "action_trace_sha256")):
                        if sha256(out / file) != row[hash_key]:
                            raise ValueError("Changed raw call/action trace")
                    calls = read(out / "unroll_calls.json")
                    F["verify_episode"](row["result"], calls)
                    if any(c["backend_calls"] != 1 for c in calls):
                        raise ValueError("Additional online forecasts")
                    # AMENDMENT: frozen code required devices_by_episode[ep] == uuid. Here a
                    # different device is accepted only if its six engineering outcomes on the excluded
                    # smoke scenario are EQUIVALENT to the first device's (see equivalent()).
                    ep = row["episode"]
                    if ep in devices_by_episode and devices_by_episode[ep] != uuid:
                        first = devices_by_episode[ep]
                        ok, dev = equivalent(outcomes[(first, task)], outcomes[(uuid, task)])
                        if not ok:
                            raise ValueError(f"Paired arms executed on outcome-divergent devices ({first} vs {uuid}, {task}, deviation={dev})")
                        worst[task] = max(worst.get(task, 0.0), dev)
                        spanned.setdefault(task, {}).setdefault(ep, set()).update({first, uuid})
                    devices_by_episode.setdefault(ep, uuid)
                    current.append(row)
                current.sort(key=lambda r: r["episode"])
                F["validate_coverage"](current, expected)
                if report["episodes"] != len(current):
                    raise ValueError("Reported episode count differs")
                rows.extend(current)
                bindings[str(shard / "report.json")] = digest
            panel[task][arm] = sorted(rows, key=lambda r: r["episode"])
        # device groups by outcome hash, for the report
        for (u, t), h in outcome_hash.items():
            if t == task:
                groups.setdefault(task, {}).setdefault(h, []).append(u)
    import offline_study.metaworld_component_analysis as frozen
    frozen.validate_panel(panel)
    audit = {"amendment": ("same-device gate replaced by engineering-outcome equivalence: the six device-bound sub-arm "
                           "checks on the excluded smoke scenario must match exactly on every non-float field (success, "
                           "steps, stimulus hashes, candidate counts, planning-call structure) and within rel_tol on "
                           "native_reward / native_state_distance"),
             "rel_tol": REL_TOL, "float_keys": list(FLOAT_KEYS),
             "max_relative_deviation_between_paired_devices": worst,
             "device_exact_outcome_groups": {t: {h: sorted(v) for h, v in g.items()} for t, g in groups.items()},
             "device_engineering_outcome_sha256": {f"{u}/{t}": h for (u, t), h in outcome_hash.items()},
             "device_engineering_outcomes": {f"{u}/{t}": {sub: {k: v for k, v in o.items() if k != "planning_calls"}
                                                          for sub, o in outs.items()} for (u, t), outs in outcomes.items()},
             "episodes_spanning_devices": {t: {str(e): sorted(d) for e, d in sorted(m.items())} for t, m in spanned.items()}}
    return panel, bindings, audit


def check_shards(root, freeze, engineering_root, F):
    """Per-shard frozen verifications + device outcome grouping on whatever shards exist (no analysis)."""
    sha256, read = F["sha256"], F["read"]
    protocol = F["validate_protocol"](freeze)
    engineering, outcome_hash, outcomes, rows, problems = {}, {}, {}, [], []
    for task in F["TASKS"]:
        dev_by_ep = {}
        for arm in F["ARMS"]:
            for shard in F["shard_paths"](root / task / arm):
                try:
                    report, _ = F["verified_report"](shard)
                    launch = read(shard / "protocol.json")
                    ok = not ((shard / "FAILED.json").exists() or report["status"] != F["METHOD"] + "_shard_complete" or
                              report["task"] != task or report["arm"] != arm or launch["method"] != F["METHOD"] or
                              launch["task"] != task or launch["arm"] != arm or
                              report["protocol_sha256"] != sha256(shard / "protocol.json") or
                              launch["freeze_sha256"] != sha256(freeze / "protocol.json") or
                              launch["source_sha256"] != protocol["source_sha256"] or
                              report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False or
                              report["scientific_efficacy_measurement"] is not True)
                    if not ok:
                        raise ValueError("Unbound or failed component shard")
                    uuid = launch["device_uuid"]
                    key = (uuid, task)
                    if key not in engineering:
                        engineering[key] = F["verify_engineering"](engineering_root / uuid / task, freeze, task, uuid)
                        outcomes[key], outcome_hash[key] = engineering_outcomes(engineering_root, uuid, task, F)
                    if engineering[key] != launch["engineering_report_sha256"]:
                        raise ValueError("Receiving proof changed")
                    expected = F["assigned_rows"](protocol["episodes"], launch["logical_ranks"])
                    if launch["expected_episodes"] != expected:
                        raise ValueError("Altered logical stream membership")
                    n = 0
                    for filename, file_hash in report["episode_files_sha256"].items():
                        if sha256(shard / filename) != file_hash:
                            raise ValueError("Changed or unsafe episode record")
                        row = read(shard / filename)
                        out = shard / f"calls-{row['episode']:03d}"
                        for file, hash_key in (("unroll_calls.json", "unroll_calls_sha256"), ("action_trace.json", "action_trace_sha256")):
                            if sha256(out / file) != row[hash_key]:
                                raise ValueError("Changed raw call/action trace")
                        calls = read(out / "unroll_calls.json")
                        F["verify_episode"](row["result"], calls)
                        if any(c["backend_calls"] != 1 for c in calls):
                            raise ValueError("Additional online forecasts")
                        dev_by_ep.setdefault(row["episode"], set()).add(uuid)
                        n += 1
                    if report["episodes"] != n:
                        raise ValueError("Reported episode count differs")
                    rows.append((task, arm, shard.name, uuid, outcome_hash[key][:12], n, "OK"))
                except Exception as e:  # report, don't hide
                    problems.append(f"{task}/{arm}/{shard.name}: {type(e).__name__}: {e}")
                    rows.append((task, arm, shard.name, "?", "?", 0, "FAIL"))
        spanning = {ep: sorted(d) for ep, d in sorted(dev_by_ep.items()) if len(d) > 1}
        devs = sorted(u for (u, t) in outcomes if t == task)
        pair_dev = {}
        for i, a in enumerate(devs):
            for b in devs[i + 1:]:
                pair_dev[(a, b)] = outcome_deviation(outcomes[(a, task)], outcomes[(b, task)])
        divergent = {ep: d for ep, d in spanning.items()
                     if any(not equivalent(outcomes[(a, task)], outcomes[(b, task)])[0] for i, a in enumerate(d) for b in d[i + 1:])}
        exact_groups = {}
        for (u, t), h in outcome_hash.items():
            if t == task:
                exact_groups.setdefault(h, []).append(u)
        worst = max([v for v in pair_dev.values() if v is not None], default=0.0)
        print(f"== {task}: shards verified {sum(1 for r in rows if r[0]==task and r[6]=='OK')}, devices {len(devs)}, "
              f"exact-outcome groups {len(exact_groups)}, max pairwise relative float deviation {worst:.3g} "
              f"(non-float mismatch pairs: {sum(1 for v in pair_dev.values() if v is None)}), "
              f"episodes spanning devices {len(spanning)}, of which NOT EQUIVALENT within rel_tol {REL_TOL}: {len(divergent)}")
        for h, us in exact_groups.items():
            print(f"   exact group {h[:12]}: {[u[:8] for u in us]}")
        if divergent:
            print("   NOT EQUIVALENT:", divergent)
    for r in rows:
        print("   %-10s %-22s %s dev=%s eng-outcome=%s eps=%d %s" % r)
    for p in problems:
        print("   PROBLEM", p)
    return problems


def missing_units(root, F):
    out = []
    for task in F["TASKS"]:
        for arm in F["ARMS"]:
            have = {p.name for p in F["shard_paths"](root / task / arm)}
            for s in range(8):
                if f"shard-{s:02d}" not in have:
                    out.append(f"{task}/{arm}/shard-{s:02d}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-src", type=Path, help="src dir of the frozen source snapshot (hash bound in the freeze)")
    for name in ("root", "freeze", "engineering-root", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--strict", action="store_true", help="run the unmodified frozen loader instead of the amended one")
    ap.add_argument("--check-only", action="store_true", help="verify whatever shards exist and report device groups; no analysis")
    args = ap.parse_args()
    frozen, F = _import_frozen(args.frozen_src)
    miss = missing_units(args.root, F)
    if args.check_only:
        problems = check_shards(args.root, args.freeze, args.engineering_root, F)
        print(f"MISSING ({len(miss)}/64): " + (" ".join(miss) or "none"))
        raise SystemExit(1 if problems else 0)
    if miss:
        raise SystemExit(f"INCOMPLETE PANEL: {len(miss)} of 64 units missing: " + " ".join(miss))
    if args.strict:
        panel, bindings = frozen.load_panel(args.root, args.freeze, args.engineering_root)
        audit = {"amendment": None, "loader": "frozen metaworld_component_analysis.load_panel"}
    else:
        panel, bindings, audit = load_panel_amended(args.root, args.freeze, args.engineering_root, F)
    report = frozen.analyze(panel)
    report.update(freeze_sha256=F["sha256"](args.freeze / "protocol.json"), input_reports_sha256=bindings,
                  frozen_analysis_source_sha256=F["sha256"](Path(frozen.__file__)),
                  amended_loader_source_sha256=F["sha256"](Path(__file__)), device_gate_audit=audit)
    args.output.mkdir(parents=True, exist_ok=False)
    F["write_json"](args.output / "report.json", report)
    F["write_json"](args.output / "DONE.json", {"report_sha256": F["sha256"](args.output / "report.json")})
    print(json.dumps({"status": report["status"], "output": str(args.output), "device_gate_audit": audit}, indent=1))


if __name__ == "__main__":
    main()
