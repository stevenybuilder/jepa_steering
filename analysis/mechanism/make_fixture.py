"""Synthetic fresh-confirmation results tree in the exact DATA_CONTRACT.md schema.

Purpose: exercise every mechanism script BEFORE the real numbers land. Outputs are stamped
synthetic=true everywhere. Regimes control the planted effects:

  positive : refined edit +12 pp vs native on all tasks, its random control +0; coupling +8 pp,
             its control +1; pathway effects additive. Coefficient norm predicts rescue.
  mixed    : refined +12 pp on reach only, -6 pp on pointmaze, ~0 elsewhere; coupling +8 on wall only;
             random controls +4 on reach (so the learned edit does NOT beat its control there).
  negative : all learned arms -5 pp; random controls 0.
  null     : every arm identical success probability (noise only).

Usage: .venv/bin/python analysis/mechanism/make_fixture.py --regime positive --out /tmp/fx-positive

Known limit (2026-09-13, scenario_heterogeneity QA): the planted coefficient-norm -> rescue association in
`positive` is too weak to detect at n=96. The +0.12 refined shift is scaled by coef_scale ~ U(0.6, 1.4), which
gives a true within-native-failure Spearman(coef_norm_mean, rescue) of only ~0.12-0.14 per task (Stouffer z ~ 1.3
over four tasks), so no valid test can reach significance. To exercise the detection branch, widen coef_scale
(e.g. U(0.05, 1.95)) or raise the refined shift; that change is deliberately NOT made here because every
mechanism script's fixture outputs depend on this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ("reach", "reach-wall", "pointmaze", "wall")
ARMS = ("native", "fixed_rank4", "matched_random_fixed_rank4", "coupling_only",
        "matched_random_coupling", "joint", "visual_only", "action_condition_only")
BASE = {"reach": 0.45, "reach-wall": 0.30, "pointmaze": 0.80, "wall": 0.76}
# The 12 registered contrasts per task, copied from offline_study.experiments.fresh_confirmation.CONTRASTS
# (7 arm-vs-native + refined-random, coupling-random, joint-visual, joint-action, factorial-interaction);
# 12 x 4 tasks = family_size 48.
CONTRASTS = {f"{a}-native": {a: 1, "native": -1} for a in ARMS[1:]}
CONTRASTS.update({
    "refined-random": {"fixed_rank4": 1, "matched_random_fixed_rank4": -1},
    "coupling-random": {"coupling_only": 1, "matched_random_coupling": -1},
    "joint-visual": {"joint": 1, "visual_only": -1},
    "joint-action": {"joint": 1, "action_condition_only": -1},
    "factorial-interaction": {"joint": 1, "visual_only": -1, "action_condition_only": -1, "native": 1}})
ANALYSIS = dict(endpoint="official_binary_success", independent_unit="scenario", bootstrap_draws=20000,
                bootstrap_seed=2026091221, family_size=48,
                interval="paired_scenario_percentile_bootstrap_bonferroni_95", minimum_useful_gain_pp=5,
                complete_panel_required=True, automatic_selection=False, historical_results_pooled=False)


def frozen_results(matrix, tasks, episodes=96):
    """Same estimator as fresh_confirmation.analyze: one RNG stream over tasks in order, one index
    matrix per task, every registered contrast as values @ weights, Bonferroni-48 percentile bounds."""
    rng2 = np.random.default_rng(ANALYSIS["bootstrap_seed"])
    alpha = .05 / ANALYSIS["family_size"]
    results = {}
    for task in tasks:
        values = np.stack([np.asarray(matrix[task][a]) for a in ARMS], 1).astype(float)
        idx = rng2.integers(0, episodes, size=(ANALYSIS["bootstrap_draws"], episodes))
        contrasts = {}
        for name, weights in CONTRASTS.items():
            d = values @ np.asarray([weights.get(a, 0) for a in ARMS], dtype=float)
            iv = np.quantile(d[idx].mean(1), [alpha / 2, 1 - alpha / 2]) * 100
            contrasts[name] = dict(difference_pp=float(d.mean() * 100), simultaneous_95_interval_pp=iv.tolist())
        results[task] = dict(n=episodes, success_percent=dict(zip(ARMS, (values.mean(0) * 100).tolist())), contrasts=contrasts)
    return results
STATE_DIM = {"reach": 39, "reach-wall": 39, "pointmaze": 4, "wall": 4}

EFFECTS = {  # additive shift in success probability per arm, per task
    "positive": {t: {"fixed_rank4": .12, "matched_random_fixed_rank4": .0, "coupling_only": .08,
                     "matched_random_coupling": .01, "joint": .10, "visual_only": .06, "action_condition_only": .04}
                 for t in TASKS},
    "mixed": {"reach": {"fixed_rank4": .12, "matched_random_fixed_rank4": .04, "coupling_only": .02,
                        "matched_random_coupling": .03, "joint": .05, "visual_only": .06, "action_condition_only": -.02},
              "reach-wall": {"fixed_rank4": .01, "matched_random_fixed_rank4": -.03, "coupling_only": -.04,
                             "matched_random_coupling": .0, "joint": .0, "visual_only": .08, "action_condition_only": .07},
              "pointmaze": {"fixed_rank4": -.06, "matched_random_fixed_rank4": .0, "coupling_only": -.03,
                            "matched_random_coupling": .04, "joint": -.01, "visual_only": .01, "action_condition_only": .06},
              "wall": {"fixed_rank4": .02, "matched_random_fixed_rank4": .0, "coupling_only": .08,
                       "matched_random_coupling": .01, "joint": .02, "visual_only": -.02, "action_condition_only": .0}},
    "negative": {t: {"fixed_rank4": -.05, "matched_random_fixed_rank4": .0, "coupling_only": -.05,
                     "matched_random_coupling": .0, "joint": -.05, "visual_only": -.03, "action_condition_only": -.02}
                 for t in TASKS},
    "null": {t: {a: .0 for a in ARMS[1:]} for t in TASKS},
}


def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")


def build(regime, out, seed=7, tasks=TASKS, iters_scale=1.0, episodes=96):
    rng = np.random.default_rng(seed)
    out = Path(out)
    freeze_records, matrix = {}, {}
    for task in tasks:
        mw = task.startswith("reach")
        n_plan = 7 if mw else 1
        iters = 15 if mw else 30
        rec_iters = max(1, int(round(iters * iters_scale)))  # calls actually written per planning call
        rows = []
        for episode in range(episodes):
            init = rng.normal(size=STATE_DIM[task]).round(6).tolist()
            goal = rng.normal(size=STATE_DIM[task]).round(6).tolist()
            rows.append(dict(episode=episode, logical_rank=episode % 8, local_seed=2026091217,
                             environment_seed=int(rng.integers(1, 2**31 - 1)), role="scientific_candidates",
                             initial_sha256=h(task, episode, "init"), goal_sha256=h(task, episode, "goal"),
                             initial_state=init, goal_state=goal, rand_vec=init[:3], expert_frames=101,
                             expert_goal_success=1.0, included_regardless_of_expert_goal_success=True,
                             tensor_path=f"synthetic/{task}/{episode:03d}.pt", tensor_sha256=h(task, episode, "t")))
        for k in range(8):  # excluded engineering rows
            rows.append(dict(episode=k, logical_rank=k, local_seed=2026141217, environment_seed=int(rng.integers(1, 2**31 - 1)),
                             role="excluded_engineering", initial_sha256=h(task, k, "einit"), goal_sha256=h(task, k, "egoal"),
                             initial_state=[0.0] * STATE_DIM[task], goal_state=[0.0] * STATE_DIM[task], rand_vec=[0, 0, 0],
                             expert_frames=101, expert_goal_success=1.0, included_regardless_of_expert_goal_success=True,
                             tensor_path=f"synthetic/{task}/e{k:03d}.pt", tensor_sha256=h(task, k, "et")))
        freeze_records[task] = rows
        # latent scenario difficulty -> native success probability
        difficulty = rng.normal(size=episodes)
        base_logit = np.log(BASE[task] / (1 - BASE[task]))
        p_native = 1 / (1 + np.exp(-(base_logit - 1.2 * difficulty)))
        u_common = rng.uniform(size=episodes)  # shared noise -> paired outcomes correlate
        eff = EFFECTS[regime][task]
        succ = {}
        coef_scale = rng.uniform(0.6, 1.4, size=episodes)  # per-scenario coefficient magnitude
        for arm in ARMS:
            shift = eff.get(arm, 0.0)
            # in positive/mixed regimes, refined rescues concentrate where coefficient norm is large
            if arm == "fixed_rank4" and shift > 0:
                shift = shift * coef_scale / coef_scale.mean()
            p = np.clip(p_native + shift, 0.01, 0.99)
            noise = np.where(rng.uniform(size=episodes) < 0.85, u_common, rng.uniform(size=episodes))
            succ[arm] = noise < p
        matrix[task] = succ
        for episode in range(episodes):
            row = [r for r in rows if r["role"] == "scientific_candidates" and r["episode"] == episode][0]
            scen = out / task / f"scenario-{episode:03d}"
            native_actions = [h(task, episode, "act", i, "native") for i in range(n_plan)]
            hashes = {}
            for arm in ARMS:
                s = bool(succ[arm][episode])
                dist = float(max(0.02, rng.normal(0.25 if s else 0.55, 0.08)))
                reward = float(rng.normal(760 if s else 700, 25)) if mw else float(rng.normal(-dist * 10, 1))
                # action trace: diverge from native at a random call for edited arms (later if random control)
                if arm == "native":
                    acts = native_actions
                else:
                    first = int(rng.integers(0, n_plan)) if (s != bool(succ["native"][episode]) or rng.uniform() < 0.5) else n_plan
                    if "matched_random" in arm and rng.uniform() < 0.3:
                        first = n_plan
                    acts = [a if i < first else h(task, episode, "act", i, arm) for i, a in enumerate(native_actions)]
                calls, trace, planning_calls = [], [], []
                steps_left = 100 if mw else 30
                for pcall in range(n_plan):
                    for it in range(rec_iters):
                        cands = 300
                        energy = _energy(rng, arm, cands, coef_scale[episode])
                        calls.append(dict(horizon=6, candidates=cands, backend_calls=1,
                                          seconds=float(rng.normal(0.9, 0.05)), energy=energy))
                    calls.append(dict(horizon=6, candidates=1, backend_calls=1, seconds=0.05,
                                      energy=_energy(rng, arm, 1, coef_scale[episode])))
                    planning_calls.append(dict(steps_left=steps_left, returned_model_actions=3 if mw else 6,
                                               iterations=iters, seconds=float(rng.normal(40, 3))))
                    trace.append(dict(steps_left=steps_left, observation_sha256=dict(visual=h(task, episode, pcall, "v"),
                                      proprio=h(task, episode, pcall, "p")), actions_sha256=acts[pcall]))
                    steps_left -= 15 if mw else 30
                record = dict(scenario=row, arm=arm,
                              result=dict(initial_sha256=row["initial_sha256"], goal_sha256=row["goal_sha256"],
                                          expert_success=1.0, native_success=s, native_state_distance=dist,
                                          native_reward=reward, elementary_steps=100 if mw else 30,
                                          planning_calls=planning_calls, observed_frames=(101 if mw else 31),
                                          published_candidate_count=300),
                              calls=calls, action_trace=trace, source_parity=[[6, 300], [6, 1]],
                              device_uuid=f"GPU-synthetic-{episode % 16:02d}", numerical_checks=[],
                              seconds=float(sum(c["seconds"] for c in planning_calls)), freeze_sha256="synthetic",
                              scientific_efficacy_measurement=True, synthetic=True)
                write(scen / f"{arm}.json", record)
                write(scen / arm / "calls.json", calls)
                write(scen / arm / "actions.json", trace)
                hashes[arm] = h(json.dumps(record, sort_keys=True))
            write(scen / "STARTED.json", dict(task=task, scenario=row, freeze_sha256="synthetic",
                                              device_uuid=f"GPU-synthetic-{episode % 16:02d}", engineering=False))
            write(scen / "report.json", dict(task=task, scenario=row, records_sha256=hashes, device_uuid=f"GPU-synthetic-{episode % 16:02d}",
                                             parameters_unchanged=True, engineering_report_sha256="synthetic", synthetic=True))
            write(scen / "DONE.json", dict(report_sha256="synthetic"))
    # freeze protocol
    protocol = dict(method="fresh_four_task_eight_arm_confirmation_20260912_v2", synthetic=True, arms=list(ARMS),
                    tasks={t: dict(records=freeze_records[t], checkpoint_sha256="synthetic",
                                   assets=dict(checkpoint="synthetic", coupling="synthetic", refined="synthetic"),
                                   planning=dict(config={})) for t in tasks},
                    analysis=dict(ANALYSIS),
                    contrasts={k: dict(v) for k, v in CONTRASTS.items()},
                    scientific_episodes=3072, same_physical_gpu_per_scenario=True, source_sha256="synthetic")
    write(out / "freeze" / "protocol.json", protocol)
    # frozen-style analysis report (same estimator as fresh_confirmation.analyze, all 12 registered contrasts)
    results = frozen_results(matrix, tasks, episodes)
    write(out / "analysis" / "report.json", dict(method=protocol["method"], freeze_sha256="synthetic", synthetic=True,
                                                 regime=regime, results=results, analysis=protocol["analysis"],
                                                 source_reports_sha256={}, scientific_evaluations=3072,
                                                 single_released_checkpoint_per_task=True))
    write(out / "analysis" / "DONE.json", dict(report_sha256="synthetic"))
    return out


def _energy(rng, arm, cands, scale):
    if arm in ("fixed_rank4", "matched_random_fixed_rank4"):
        coef = (rng.normal(size=(cands, 4)) * (0.05 * scale if arm == "fixed_rank4" else 0.05)).round(4)
        req = np.linalg.norm(coef, axis=1) * 3.0
        return dict(coefficients=coef.tolist(), requested_l2=req.round(4).tolist(),
                    realized_l2=(req * 0.998).round(4).tolist(), active=[True] * cands,
                    response_probe_rollouts=0, native_shadow_rollouts=0, backend_calls=1, horizon=6)
    if arm == "native":
        return dict(response_probe_rollouts=0, native_shadow_rollouts=0, backend_calls=1, horizon=6)
    edited = cands
    req = float(cands * 0.04)
    return dict(horizon=6, candidates=cands, edited_candidates=edited, requested_squared_l2_sum=req,
                realized_squared_l2_sum=req * 0.997, response_probe_rollouts=0, native_shadow_rollouts=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=list(EFFECTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--iters-scale", type=float, default=0.2, help="fraction of CEM iterations whose unroll calls are written (schema-identical; 1.0 = full size, ~2.5 MB per refined record)")
    ap.add_argument("--episodes", type=int, default=96)
    a = ap.parse_args()
    root = build(a.regime, a.out, a.seed, a.tasks, a.iters_scale, a.episodes)
    print(json.dumps(dict(out=str(root), regime=a.regime, results=str(root), freeze=str(root / "freeze"),
                          analysis=str(root / "analysis"), synthetic=True)))
