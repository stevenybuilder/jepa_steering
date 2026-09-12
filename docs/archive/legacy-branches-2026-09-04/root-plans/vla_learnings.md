# VLA learnings (imported from the mechinterp-vla project, 2026-09-02)

Source: user-supplied transcript from the separate VLA (π0.5 / LIBERO-Safety) project. Kept verbatim in §1 for the record; §2 is what this project (JEPA / world-model sonar benchmark) takes from it and how each lesson is enforced here.

## 1. Transcript (verbatim)

```text
› Yeah, quite honestly, I think the new experiment is much more explicit because this experiment that was done previously was just whether the hand was visible or hidden. I don't think that it was
  properly set up because that doesn't directly answer a hypothesis of if there's a safety mechanism, just thinking about it from a common sense standpoint.

• I agree. The old experiment was fundamentally a visual-reliance test, not a clean safety-mechanism test.

  It asked: Does showing the hand change the robot's behavior?

  Even with blocking-versus-clear comparisons, a positive result could reflect: generic obstacle detection; visual salience or occlusion; an unfamiliar-object response; ordinary motor replanning;
  human-specific safety. Those explanations were never separated. The experiment was still useful as a preliminary check—it showed that the model sometimes uses the hand visually and is not merely
  replaying one fixed trajectory—but we should not label its activation geometry a "safety representation."

  The new experiment asks the correct common-sense question: When physical obstruction is held constant, does the model respond differently because the obstacle is human, and only when that human is
  actually endangered? That directly separates generic obstacle avoidance from a human-sensitive safety computation.

› This new experiment looks good, but I want to make sure that we don't run into the same problem as the old experiment, where we think that it's directly answering the hypothesis, but it is not.

• [hostile design audit]
  Freeze the exact hypothesis: At matched physical geometry and motor context, a visible human hand blocking the intended route causes additional avoidance relative to an equally obstructive inanimate
  object. A specific upstream internal computation mediates that additional behavior.

  Stages: 1 behavioral interaction passes; 2 held-out activation geometry passes (beyond pixels, position, motor output); 3 removal and insertion pass; 4 route-clear and unrelated behavior unchanged;
  5 a competent ordinary checkpoint lacks or weakens it; 6 matched safety/no-safety training arms differ. We must not skip levels. Behavioral evidence is not a mechanism; decodability is not causality.

  Controls: generic obstacle avoidance → geometry-matched human vs inanimate object; salience → multiple appearances, held-out identities; occlusion → visible/hidden and equal-area sham; route position →
  cross identities with blocking/edge/clear; generic motor correction → match initial actions/motor state; memorized task–asset pairing → novel asset–scene combinations, held-out tasks; freezing →
  measure progress/success/path length/timeout; trajectory copying → relocated hazards, fresh closed-loop replanning; statistical luck → frozen splits, task-level clustering, multiplicity correction.

  Hard behavioral gate: no activation search unless human blocking produces reliable collision reduction, exceeds the matched-object effect, disappears when route-clear, preserves task success, and
  replicates across held-out tasks/states/positions/identities. Ideally ≥10 independent task families. Target should be called a human-sensitive route-avoidance mechanism, not a generic "safety mechanism."

› Feel free to use the ARXIV research agent skill to see how state-of-the-art papers assess safety mechanisms in VLAs ...

• No established paper cleanly discovers and selectively edits an implicit, human-sensitive safety mechanism inside a VLA. Components: LIBERO-Safety (2606.23686), VLSA/AEGIS (2512.11891), SAFE-Dict
  (2602.01834), VLA-Trace (2605.30117), Action Atlas (2603.19233: action-correlated representations are often generic motor programs; edits can simply damage the policy), COAST (2605.17144).
  Three separate questions: generic route risk; human-sensitive safety; internal mechanism. Primary behavioral test = human-specific triple difference
  [(visible−hidden)_blocking − (visible−hidden)_clear]_human − [...]_object. Task family is the statistical cluster. Mechanism search only after behavior passes; sweep the full causal chain; keep
  token-level activations; residualize pixel/position/task/progress/action baselines; require transfer to held-out families; donor-free removal/insertion with strength sweep and equal-norm orthogonal,
  wrong-layer, wrong-token, reverse, sham controls; edits must preserve route-clear, object, and task competence.

  Interpretable outcomes: generic route effect fails → stop before interpretability; route passes but human-specific fails → generic avoidance; behavior passes but no representation transfers →
  wrong coordinate system or highly distributed; representation predicts but editing fails → correlate/redundancy/wrong operator; editing changes every task → generic motor subspace; selective
  donor-free editing succeeds → genuine causal evidence.

› Based on the experiments we ran so far, which of these failed and which passed?

• Generic route safety: failed the strict gate. Human-specific: not tested (no matched object). Transferable representation: partly passed (small subspace, may be generic motor). Causal editing:
  failed specificity (controls also changed actions). Selective repair: not achieved. Conclusion: the model notices the hand and reacts, but no safety-specific mechanism was isolated or controlled.

› ... if there was a safety-specific mechanism, it'd be implicit within the weights, so how would we find it? ... maybe we're looking in the wrong space.

• We may be searching for the wrong kind of object. The checkpoint may have learned a conditional computation (obstacle intersects intended motion → modify the motor plan) distributed across vision,
  attention, action denoising and replanning; it may never collapse into one stable "safety direction." Search for the earliest hazard × intended-route interaction; track it over layers, denoising time
  and replanning; test state-dependent subspaces and pathways; use matched safety-trained and ordinary checkpoints for any "training created it" claim.

› ... we were doing this based off a checkpoint ... / What does the π0.5 LIBERO-Safety checkpoint mean?

• Full π0.5 architecture, weights further fine-tuned on LIBERO-Safety demonstrations (base π0.5 → LIBERO-Safety fine-tuning → checkpoint). It may have learned avoidance implicitly from safe
  trajectories without an explicit "safety feature."

› ... doesn't that mean there was trajectory copying? I thought we controlled for this.

• We controlled experimental copying (no donor actions/activations; fresh states and noise) but not training memorization. Ruling that out needs held-out tasks, layouts, hazard poses and routes, plus
  nearest-training-trajectory checks.

› ... Do we even need to use that checkpoint? Isn't it better not to?

• Every runnable model is a checkpoint; the real choice is which trained weights. The released safety checkpoint alone cannot separate a general safety computation, memorized demonstrations, ordinary
  obstacle avoidance, and general fine-tuning gains. Cleanest: train two matched models from base π0.5 — SAFE (task + safety examples) and CONTROL (identical tasks/data volume/optimization/seeds, no
  safety signal) — evaluate on fully held-out tasks/layouts/routes/poses/identities, discover in SAFE, test absence in CONTROL, validate on the released checkpoint.

› How come we didn't catch this? How come you didn't catch it?

• The core mistake was conflating "a mechanism inside a safety-fine-tuned checkpoint" with "a safety-specific mechanism created by safety training." The matched ordinary model was treated as optional
  when it was necessary. Trajectory copying was defined too narrowly. The design became driven by what we could run on the released checkpoint instead of what would identify the hypothesis. Going
  forward: freeze the exact claim first; require matched SAFE/control models, behavioral separation, and compositional held-out tests before any expensive activation search.

› How did COAST even successfully do their experiments? ... https://arxiv.org/pdf/2605.17144

• COAST solved a substantially easier identification problem, and we applied its ideas before fixing our target. COAST: predict ordinary task success vs failure; simulator supplies an unambiguous label;
  any representation that improves success is useful; same-task train/test configurations acceptable; changing the entire motor strategy can count as success; does not ask why training created the
  subspace. It deliberately used imperfect checkpoints (π0.5 LIBERO step 2,000) for balanced contrastive data; 15 success/failure rollouts per task; fit a conceptor; select layer/aperture/strength on the
  fitting set; evaluate on 30 new configurations; count any completion increase as success (43% → 80%). It did not need to prove the subspace is a distinct semantic mechanism, the earliest causal source,
  independent of generic motor competence, created by a particular training signal, or harmless to unrelated behavior.
  Where we went wrong: sophisticated geometry on top of an ambiguous behavioral contrast; COAST had clean outcome labels before fitting geometry; we tried to answer several harder questions at once.
  Verdict: our safety question is genuinely harder, but the failures were not inevitable — redesign the behavioral and matched-training experiment first, then apply COAST-style geometry to the resulting
  factorial contrast.
```

## 2. What this project takes from it (and where it is enforced)

| Lesson | Status here | Enforcement |
|---|---|---|
| Freeze the exact claim before any geometry; name the target narrowly (not "safety mechanism") | Adopted. Our target is "a hazard × action relational state that changes the predicted future and the planner's action ranking", never "the model is safe" | `experiment_design.md` protocol v0.5; claim ladder L0–L5 |
| Behavioral gate before activation search; behaviour must be unambiguous and *large*, not merely significant | Adopted after loop 1 repeated the VLA mistake (arms run on a 5% interaction). B-gate: ≥16 scenes, model reproduces ≥25% of the true interaction, hazard-specific recovery > 0, planner flip ≥ 0.25 | `scripts/cgs_pilot/behavior_gate.py`; runner scripts stop on FAIL |
| Separate generic response from the specific one with matched physics; use difference-in-differences / triple differences | Adopted from the start: 2×2 hazard×action, H0′ matched-displacement null (relational DiD), main-effect-only donors, appearance controls | `protocol.py` (`paired_did`, `relational_did`); gate and patch controls |
| Training memorization is a central threat, not only donor-copying during intervention | Partly adopted: retrieval baselines (copy-delta, DROID-kNN, scene-blind ridge, persistence) and held-out seeds. Gap: no nearest-training-trajectory check against the actual training set for the driving substrate yet — required when we train our own model (keep a held-out hazard family and log nearest-training-clip distance per test scene) | `retrieval_baseline.py`; to add in driving plan |
| A "training created it" claim needs matched SAFE/CONTROL training arms; a single released checkpoint cannot support it | Adopted for the driving cell: because we train the JEPA driving WM ourselves, build matched arms from the start — WITH-collision-futures vs WITHOUT (same data volume, seeds, schedule) — so "in-support collision data creates the relational state" is testable; the released JEPA-WM/V-JEPA 2-AC checkpoints are transfer cells, not causal-origin evidence | driving substrate plan (pending) |
| Do not let the design be driven by what the released checkpoint can run | Violated in loop 1 (egg chosen for convenience; out-of-distribution for a DROID-only model). Corrected: stimulus retired; driving substrate chosen for the question, model trained to fit the stimulus | protocol v0.5 |
| Mechanism may be a distributed conditional computation, not a stable direction; search the earliest hazard × intended-route interaction and track it across layers / imagined time / replanning; state-dependent subspaces | Adopted: layer × imagined-step × token-group maps; cross-token action-Jacobian sonar; rotating-subspace tracking (principal angles, Procrustes transport); curved-code tests; conceptor quota. Result on egg: no such object, at any geometry class — consistent with no behaviour | `JEPA rep geometry.md`; `mech_w1/*` |
| COAST's endpoint (task success) is broad; ours must stay the factorial interaction; conceptors are one geometry class, fit only on interaction residuals, judged by selectivity | Adopted: conceptor arm fit on residualized DiD vectors with AND-NOT against action/null geometry; matched-spectrum and rank-one controls; selectivity with equivalence margins | `geometry_conceptor.py`, `conceptor_patch.py` |
| Edits must preserve unrelated behaviour; global damage is not a mechanism | Adopted: sham/random/unrelated-site/wrong-group controls; H0/H0′ equivalence margins; ARC/drift energy for global corruption | `patch_site.py`, `counterfactual_validity_gate.py` |
| Unit of inference = independent family (task family / scene), not seed; frozen splits; multiplicity correction | Adopted: scene-clustered bootstrap, sign-flip max-T, BY; stable hash split; sealed confirmation | `stats_utils.py`, `merge_heldout.py` |
| Interpretable-outcome table decided in advance | Adopted; loop-1 verdict maps to "generic response passes (action reproduced), specific effect fails (no hazard interaction) → stop before interpretability", which is where the B-gate now stops | `LOOP1_SUMMARY.md` |

Two rules added to this project because of the transcript:
1. **No cell may claim more than its contrast can identify.** A released checkpoint supports at most "this model does / does not build the relational state"; only matched training arms support "this training signal created it".
2. **Training-set nearest-neighbour audit is mandatory for any model we train ourselves**, in addition to the retrieval baselines, before a behavioural PASS is accepted.
