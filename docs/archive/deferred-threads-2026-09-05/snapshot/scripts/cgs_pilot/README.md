# `cgs_pilot` code map

The directory contains both the active released-world-model experiment and preserved code from retired research branches. Filename proximity does not imply that a script is eligible to run.

## Active Panel-P critical path

| File | Role |
|---|---|
| `public_panel_manifest.py` | Immutable, disjoint episode-pair manifests |
| `public_panel_eval.py` | Native unsteered behavioral evaluator |
| `public_panel_behavior_gate.py` | Fail-closed P0 class/support and realization gate |
| `public_panel_provenance.py` | Checkpoint, config, code, package, and data provenance |
| `public_panel_capture.py` | Hook-aligned activation and executed-control capture |
| `public_panel_factor_gate.py` | Basis, GMM/progress/HMM, and stability comparisons |
| `public_panel_coordinate_gate.py` | Othello-style absolute-versus-relational coordinate test |
| `public_panel_action_hmm.py` | Baum-Welch action-conditioned transitions, causal filtering, action/progress controls, and composition gate |
| `public_panel_frankenstein_math.py` | Probability, sparse-support, pattern-separation, attention-retrieval, and Procrustes primitives |
| `public_panel_frankenstein_gate.py` | Hash-bound outcome-free admission artifact for the additional recipe modules |
| `public_panel_sonar_math.py` | Conceptor, mixture-energy, metric-step, and Gaussian-OT primitives |
| `public_panel_sonar_fit.py` | Outcome-conditional operator fitting and family-specific gates |
| `public_panel_fit_apply_audit.py` | Fit/runtime population-equivalence audit |
| `public_panel_steer.py` | Eligible runtime intervention adapters and fail-closed HMM loading |
| `public_panel_development_gate.py` | Paired identity, action-dose, utility, cap-saturation, and static/HMM treatment-separation admission evidence |
| `public_panel_transport_bind.py` | Bind a passed donor operator into target factor coordinates for a runnable transport arm |
| `public_panel_arm_registry.py` | Exclusive-write final arm freeze |
| `public_panel_summarize.py` | Result summarization |

`public_panel_geometry.py` and `public_panel_coast.py` contain earlier diagnostics/reference work. They are not substitutes for the current factor, fit/apply, arm-registry, or behavioral gates.

## Implemented modules awaiting DINO-WM Reach evidence

- Action-conditioned HMM fitting and the physical-replan causal belief lifecycle are implemented. The HMM arm remains disabled unless sequence, action/progress, composition, outcome-routing, and treatment-separation gates all pass on DINO-WM Reach.
- Sparse support, matched-lure pattern preservation, blockwise probability calibration, bounded DINO Q/K capture, and paired attention-output restoration are implemented behind prospective gates.
- Relational factor transport has a held-out gate and runnable operator binder. It remains distinct from Gaussian OT and requires a second matched model/checkpoint capture.
- The native-coordinate adapter now covers both Reach and Reach-Wall with task-specific source hashes.
- The current behavior-only P0 run cannot license activation steering.

Until every module is evaluated and receives a prospective pass/fail record on DINO-WM Reach, results must not claim to test `Frankenstein-full`. Implemented modules that fail their gates become registered no-ops rather than being silently dropped.

## Legacy code

Scripts without the `public_panel_` prefix primarily belong to retired egg/cup RoboCasa, MetaDrive driving, V-JEPA transfer, CAFT, or earlier representation-geometry branches. Old `run_public_panel*.sh` pipelines are also obsolete because they can encode superseded automatic stage progression. Preserve them for provenance; do not execute them for the active experiment.

The governing plan is [WORLD_MODEL_SONAR_PLAN.md](../../WORLD_MODEL_SONAR_PLAN.md).
