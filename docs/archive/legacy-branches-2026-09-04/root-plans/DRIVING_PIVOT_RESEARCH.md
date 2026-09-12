# Driving-domain pivot: literature scoping (arXiv research agent, 2026-09-02 ~18:00 UTC)

User directive: egg/robot-sim collision is not a human-relevant safety behaviour; pivot the domain to driving while keeping the JEPA / representation-geometry method lineage (not orthogonal: driving latent predictors exist). Preregistered methodology carries over unchanged: hazard×action DiD with H0′ matched-displacement null, counterfactual-validity gate (recovered fraction 2608.11601; ARC/drift 2608.04653), behavioural-target gate (`behavior_gate.py`), retrieval/copying controls, scene-clustered sign-flip/max-T, sealed confirmation seeds, on-manifold patching with real donors.

## Recommendation (agent)

Go, on **MILE (Wayve, arXiv 2210.07729; github wayveai/mile, MIT, `mile.ckpt`) + CARLA 0.9.11**. Reasons: the only open-weight, action-conditioned *latent* driving world model trained on the same simulator that generates the factorial (no sim-render OOD, which dominated loop 1); native policy; BEV decoder with pedestrian/vehicle channels (a decodable hazard readout inside the model's own future); trivially fits 24 GB; measured hazard-conditional behaviour (Town05: 0.01 pedestrian collisions/km, 30% of driving from imagination). **WoTE** (2504.01941) as positive-control substrate (trajectory scoring with a collision reward is trained in). Do not start on Vista/DrivingWorld: 2608.11601 measured recovered fraction 0.38/0.31 on CARLA hazard counterfactuals (below our 0.5 gate) and Vista needs >40 GB.

Two structural facts:
1. Direct action-conditioned counterfactuals on the big real-video models fail the Rec gate; the fix in 2608.11601 (training-free abduction) has no code release.
2. Every open driving WM is expert-only trained (no collision futures) — the (H1,A1) cell is out of support exactly as with DROID. Only ReSim (2506.09981) addresses this (CARLA non-expert data; CARLA weights still pending). Consequence: make the behavioural target **continuous and in-support** (predicted min distance / decoded BEV overlap, brake-ranking flip), not contact.

## Ranked shortlist

| Rank | Model | Action entry / architecture | Data | 24 GB | Native planner | Licence | Hazard-conditional evidence |
|---|---|---|---|---|---|---|---|
| 1 | MILE 2210.07729 | RSSM (GRU h + Gaussian s); action conditions the prior transition; BEV decoder 8×192×192 | 32 h CARLA 0.9.11, Towns 01/03/04/06, RL expert | yes | policy head; collision cost from decoded BEV | MIT | Town05 DS 61.1, 0.01 ped-collisions/km |
| 2 | WoTE 2504.01941 | BEV latent WM rolled under each candidate trajectory; reward head (PDM collision/drivable) | NAVSIM | yes | trajectory scoring IS the model | Apache-2.0 | NC 98.5, TTC 94.9 (trained-in ranking → positive control) |
| 3 | Epona 2506.24113 | MST 1.3B latent + TrajDiT planner + VisDiT; trajectory modulates | nuPlan (+nuScenes) | yes ("single 4090") | generative planner (no ranking cost) | MIT | nuScenes collision 0.01/0.22/0.85 % |
| 4 | ReSim 2506.09981 | CogVideoX-2B DiT; waypoints as tokens | OpenDV + NAVSIM (+CARLA non-expert pending) | borderline | Video2Reward (not released) | Apache-2.0 / model licence | only model trained on hazardous actions |
| 5 | Drive-WM 2311.17918 | multiview diffusion; (Δx,Δy) MLP embedding | nuScenes | likely | planning tree w/ detector reward (code not released) | Apache-2.0 | collision 0.26 % vs 0.93 % random |
| 6 | Vista 2405.17398 / DrivingWorld 2412.19505 | SVD cross-attn actions / 1B GPT pose tokens | OpenDV 1,740 h / nuPlan + private | Vista no (66 GB) | variance reward / none | Apache-2.0 / MIT | Rec 0.38 / 0.31 on CARLA hazard counterfactuals (below gate) |

Excluded: Cosmos-Predict2 (32.5 GB, no driving action conditioning); Terra/ACT-Bench (CC-BY-NC, no planner; its "other agents stop when ego decelerates" causal-misalignment is a confound to control); LAW (WM is training-time only); Think2Drive/DreamerAD (no verifiable weights).

## Stimulus sources

- **CARLA / Bench2Drive / Leaderboard 2.0** scenario library: exact factorial, deterministic in synchronous mode with a seeded traffic manager; pedestrian-emergence scenarios exist. Re-verify replay bit-exactness as in the egg pilot. In-distribution only for MILE (and ReSim when its CARLA weights ship).
- **HUGSIM 2412.01718**: 3DGS reconstructions of nuScenes/Waymo/KITTI-360/PandaSet with inserted vehicle actors (incl. aggressive collision-seeking), deterministic, MIT — the real-video counterfactual source for nuScenes/nuPlan-trained models; vehicles only, no pedestrians.
- NAVSIM v2 pseudo-simulation (ego-action counterfactuals only); GEM 2412.11198 (object/pedestrian pose editing in real video; generator, not validated counterfactual).

## Prior art / novelty boundary

Linear probes + probe-direction steering of surrounding-vehicle predictions in GPUDrive/WOMD policies with near-collision windows (2606.31106); TopK-SAE concepts on NAVSIM trajectory scorers (2607.06328); control vectors/SAEs for speed/direction in motion transformers (2406.11624); physics steering in video WMs (2605.24322). No prior work does hazard×action counterfactual patching inside a driving *world model* with a planner-ranking endpoint. The claim must be the relational, action-conditional, planner-mediated state — "agents are decodable" is already known.

## Pilot plan (MILE + CARLA)

Per scene: H1 pedestrian in ego lane at TTC ≈ 2 s; H0 same pedestrian on the sidewalk; H0′ second sidewalk pose at matched pixel displacement; A0 brake chunk, A1 constant-throttle chunk; identical 1-s context; ground truth by replay (min ego–pedestrian distance, contact flag; contact only in (H1,A1)). Inject the fixed action chunks into the RSSM prior (verify the transition consumes external actions rather than the policy's own — the biggest risk), decode BEV per imagined step. Gate → B-gate (continuous: interaction NMSE on latents and on decoded min distance; brake-ranking flip ≥ 0.25) → retrieval (expert-episode kNN) → localization (encoder features, h, s, prior MLP, policy trunk) → patching. Strata: training towns (in-support) and Town05 (held-out). ≥16 discovery + ≥16 confirmation scenes (~64–80 seeds at 50% admission); n=16 gives ~80% power at d_z ≈ 0.8. Compute: < 2 GPU-hours for 80 seeds × 6 cells; RSSM sweeps are minutes.

Hypotheses: H1 B-gate passes on MILE; H2 brake ranking flips with hazard, not with H0′; H3 an RSSM site carries a patchable state (≥30% recovery, controls <10%); H4 predicted pedestrian trajectory is invariant to ego action (ACT-Bench misalignment absent).

Biggest risk: MILE's imagination is coupled to its own policy; if external action injection needs code changes, or the collision future is out of support and the WM predicts a near-miss regardless, the B-gate fails for the egg's reason — the honest next step is then a short non-expert fine-tune of MILE (training code released), not another interpretability method.

## BadDreamer addendum (arXiv 2606.21172; `references/BadDreamer_2606.21172.pdf`) — being folded in by the agent

VaViM (released width-768 autoregressive video WM, nuScenes/OpenDV/nuPlan) + VaVAM (flow-matching waypoint expert on the WM's future-aware representation H_fut). Trigger-erasure poisoning (inserted yellow rider vanishes from the predicted future) → 92.5% erasure, 90.3% unsafe-go; the **clean** pipeline yields ~82% on inserted-rider conflict windows (T-UGR endpoint), i.e. a large unambiguous hazard-conditional behaviour on real video; scene-matched control triggers (blue rider 24.6%, yellow bicycle 31.4%) = an H0′-style null. Design idea: hazard insertion into real nuScenes context frames as stimulus, T-UGR as planner currency, the representation interface H_fut as the first localization target, and a clean-vs-backdoored model contrast (same input, opposite behaviour) as an unusually clean mech-interp setup.

Sources listed by the agent: 2608.11601, MILE repo, WoTE repo, Epona HF, ReSim assets, Drive-WM, Vista ISSUES.md, DrivingWorld, ACT-Bench, DriveLaW 2512.23421, Cosmos performance.md, HUGSIM, NAVSIM, Bench2Drive, CARLA determinism docs, Leaderboard scenarios, 2606.31106, 2607.06328, 2406.11624, GEM, What-If World 2605.27589, Jaywalkers/DreamerV3 (MDPI).

---

# Revision 2 (agent, after verifying BadDreamer / VaViM-VaVAM and the driving-JEPA candidates) — supersedes the recommendation above

## Recommendation: two substrates, one per question

**Main (human-relevant, real video): VaViM-S + VaVAM-S** (Valeo, arXiv 2502.15672; github valeoai/VideoActionModel). Verified: weights on GitHub releases (S 185M / width 768, B 318M, L 1.2B; action experts 21M/38M/150M); code MIT, **weights research-only**; trained OpenDV 1,700 h → nuPlan 94 h + nuScenes 5.5 h; 8 context frames at 2 Hz, 512×288, LlamaGen tokens (576/frame, 4,608 max); fits a 4090 (agent estimate, VRAM undocumented). **Critical facts:** VaViM is *action-free*; VaVAM's flow-matching action expert reads VaViM hidden states via layer-wise joint attention across all 24 layers of the *context* frames, no future tokens generated at inference, no single layer chosen. BadDreamer's "H_fut" is its own abstraction (no equation, layer, pooling, or representation analysis in the paper). The real interface is 24 layers × 4,608 context tokens as read by VaVAM's attention — a localization *target space*, not a point.

**Replication (true 2×2 inside an action-conditioned WM): MILE** (2210.07729; CARLA 0.9.11; RSSM with action in the prior transition; BEV decoder with pedestrian/vehicle channels; MIT).

Design options on VaViM: **(A)** hazard × candidate-trajectory scoring inside VaVAM via the flow-matching energy E(a|x,c) = E_t‖v_θ(a_t,t,x,c) − (a−ε)‖² (derived, not native), DiD over {brake, go} × {H1, H0, H0′} — the analogue of our CEM ranking gate; **(B)** BadDreamer clean-vs-backdoored contrast as a **positive control with a ground-truth planted mechanism** (5% poisoning of 28,130 nuScenes windows fine-tuning VaViM-S is plausible on a 4090 in bf16, unverified; no code release). (B) does not answer the natural-safety question by itself.

Constraints: 2608.11601 measured Rec 0.38/0.31 for Vista/DrivingWorld (below gate; fix has no code); every open model except ReSim's pending CARLA checkpoint is expert-only trained → make the target continuous and in-support (min distance / lateral deviation, brake-vs-go energy), not contact; ACT-Bench's "causal misalignment" (other agents stop when ego decelerates) is a direct confound → test other-agent invariance.

## Ranked shortlist (revised)

| Rank | Model | Architecture / action entry | Data, context | 24 GB | Planner / cost | Licence | Hazard-conditional evidence |
|---|---|---|---|---|---|---|---|
| 1 | VaViM-S + VaVAM-S (2502.15672) | AR token WM (action-free) + flow-matching expert over all-layer hidden states | OpenDV/nuPlan/nuScenes; 8 fr @ 2 Hz | yes (est.) | native generator; ranking cost derived (energy) | MIT code; research-only weights | NeuroNCAP closed loop: VaVAM-L 56.8% frontal collision vs UniAD 78.8%, "seldom brakes"; BadDreamer clean T-UGR 17.5% on triggered windows |
| 2 | MILE (2210.07729) | RSSM; action in prior transition; BEV decoder | CARLA 32 h | yes (est.) | policy; BEV-derived cost | MIT | 0.01 ped-collisions/km; imagination driving |
| 3 | WoTE (2504.01941) | BEV latent WM rolled under each candidate; PDM reward head | NAVSIM | yes | native scoring | Apache-2.0 | NC 98.5, TTC 94.9 — by construction (positive control) |
| 4 | Epona (2506.24113) | MST latent + TrajDiT + VisDiT | nuPlan | yes | generative planner | MIT | collision 0.01/0.22/0.85 %, qualitative |
| 5 | ReSim (2506.09981) | CogVideoX-2B; waypoints as tokens | OpenDV+NAVSIM (+CARLA non-expert pending) | unverified | Video2Reward (not released) | Apache / model licence | only non-expert-trained model |
| 6 | Latent E2E predictors: LAW (2406.08481, ckpts), World4Drive (2507.00603, code only) | latent future features conditioned on ego trajectory | nuScenes | yes | W4D latent selector scores candidates | Apache-2.0 | policy metrics only (LAW coll. 0.21 %) |

Excluded after verification: AD-L-JEPA (LiDAR pretraining, not action-conditioned), DINO-Foresight (action-free), DINO-world 2507.19468 (action-conditioned fine-tune described, no weights), Drive-JEPA 2601.22032 (encoder + planner, no WM), Vista/DrivingWorld (Rec < 0.5; 66 GB), Cosmos, Terra, Think2Drive/DreamerV3-CARLA.

## Stimulus sources (revised)
- **NeuroNCAP** (2404.07762; atonderski/neuro-ncap, MIT; NeuRAD on nuScenes): scripted actor insertion (stationary/frontal/side; vehicles; pedestrians unverified) into real logs with closed-loop rendering; the field standard, already used for VaVAM; geometry-consistent across frames with ground-truth actor pose → preferred over BadDreamer's manual compositing.
- HUGSIM (vehicles only, deterministic); CARLA/Bench2Drive (exact factorial; in-distribution only for MILE).

## What BadDreamer does NOT give us
The clean model's 82.5% "yield" on triggered windows has no untriggered baseline on the same windows and no oracle procedure; the control-trigger rates (24.6%, 31.4%) are for the *poisoned* model. VaVAM's NeuroNCAP numbers (57% collisions, avoidance by deviation not braking) suggest the braking-specific interaction is modest. **The hazard×action DiD must be measured by us with H0/H0′ before any mechanism work; do not import 82.5% as an effect size.**

## Hypotheses, plan, metrics (revised)
1. B-gate: brake-vs-go energy DiD > 0 (sign-flip p < 0.05) and argmin flip rate ≥ 0.25 for H1 vs H0, not H0′, on ≥16 scenes; sampled-trajectory min distance reproduces ≥25% of the oracle interaction.
2. Interface localization: a layer band × actor/corridor token group whose real-donor patch (H1↔H0 hidden states) recovers ≥30% of the energy DiD, controls < 10%.
3. Positive control: the planted BadDreamer erasure is recovered at the same interface by the frozen pipeline.
4. Other-agent invariance: predicted actor tokens in VaViM's generated future do not change with the candidate action.

Per nuScenes val window: 8 context frames; H1 actor on the ego path (TTC ≈ 2–3 s), H0 same actor off-path at matched pixel displacement, H0′ second off-path pose, appearance control (blue rider) as nuisance; A0 decelerating 6-waypoint chunk, A1 constant-speed chunk. Readouts: energy over 10 timesteps × 10 noise seeds; sampled trajectory (final_x, lateral deviation, min distance); actor persistence in VaViM's generated future (decoded RGB + detector). Then retrieval baselines (kNN over nuScenes train windows — VaViM saw them), localization over 24 layers × token groups with LOSO projection, donor patching with sham/random/unrelated/main-effect-only/H0′ controls, discovery/confirmation split by log. Primary endpoint **min distance** (VaVAM avoids by deviation); T-UGR secondary. ≥16 + ≥16 scenes; sign-flip max-T over layer×group; exact binomial CI on flip rate.

## Right-question check and risks
VaViM answers the *planner-mediated* question (hazard × candidate ranking) on real human-relevant data, but not the *action-conditioned predictor* question — it has no action input, so the loop-1 "relational state from action modulation" claim can only be tested on MILE. Risks: (1) the derived energy cost may not track VaVAM's sampled behaviour (validate energy argmin against samples before gating); (2) the clean DiD may be small (VaVAM seldom brakes) — the B-gate decides; remedy is a stimulus change (frontal vehicle at short TTC), not another method; (3) research-only weights limit redistribution of patched checkpoints. Compute is small (VaViM-S forward passes are milliseconds); the cost is NeuRAD reconstruction — use NeuroNCAP's released scenarios first.
