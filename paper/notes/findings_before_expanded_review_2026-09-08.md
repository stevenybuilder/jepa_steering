# Findings

*Updated September 8, 2026 with the arXiv comparison and contribution assessment below. Execution status and compute estimates remain the user-provided snapshot; they were not refreshed for this literature update.*

## Execution status snapshot

The closed-loop intervention comparison is not finished or producing a reliable success-rate result yet.

- **Native baselines:** 96 episodes each completed for Reach, Reach-Wall, Wall, and PointMaze.
- **Revised operator:** full-episode engineering checks are running. The first unsteered 100-step episode finished in 5.2 minutes; its exact-repeat check is underway before testing the edit and random control.
- **The revised 96 episodes per task/condition have not started.** Earlier coupling comparisons still need the remaining input repair and final paired analysis.

The initial two-task comparison remains roughly **6–10 hours with 6–8 ready GPUs**, not with the currently available evaluation capacity. That is separate from the full study.

## Five promising paper directions

These are potential stories—not five established novel contributions. The efficacy numbers below belong to the original interventions, not yet the cheaper replacement.

| Direction | What we actually found | What the paper could responsibly argue |
|---|---|---|
| **1. Low-dimensional interventions can improve frozen world-model forecasts** | Original rank-4 edits reduced H6 proprioceptive embedding error by **2.89% on Reach** and **2.14% on Reach-Wall**. | Some forecast error is correctable through a small activation subspace without retraining model weights. Behavioral and multi-seed replication remain necessary for stronger claims. |
| **2. Vision–action coupling and activation correction provide complementary benefits** | The original combined edit improved Reach error by **4.82%**, with simultaneous 95% interval **[3.67%, 5.98%]**, and beat its random control and both component-removal arms. | This is our strongest existing positive story: combining the two components helps beyond either alone. That does not, by itself, establish mechanistic synergy or better planning. |
| **3. Spatial distribution and layer distribution are different questions** | On Reach, only the all-patch spatial arm passed the frozen gates. However, several single-layer edits also worked; block 0 improved error by **3.03%**. | Broad spatial support may matter without requiring edits across every layer. Our results do not support “single-layer edits don’t work.” |
| **4. Numerical precision can reverse a representational-geometry diagnostic** | In corrected Push-T results, cubic reconstruction had approximately **1,192× lower omitted-activation error than linear in FP32**, but approximately **35% higher error in BF16**. | Geometry conclusions can depend strongly on measurement precision. This qualifies the earlier “995×” headline: it is not a precision-independent benefit, nor an improvement over unsteered task performance. |
| **5. Steering benefits are task-dependent, with meaningful negative controls** | Push-T’s rank-4 improvement was inconclusive, while joint coupling worsened error by **0.64%** despite positive Reach results. | A useful study can identify where a steering recipe helps, fails, or harms—not merely advertise an average improvement. This is not yet evidence of intervention transfer. |

The positive-effect and localization summaries are documented in [corrected offline results](reports/CORRECTED_OFFLINE_RESULTS.md); I also checked the underlying precision-specific geometry reports.

The initial recommendation, before the prior-art comparison below, was **one focused four-page paper built around #1 and #2, with #3 and #5 as important boundaries**, followed by the completed behavioral comparison. The literature review sharpens that positioning toward identifying correctable error structure. A null behavioral result could still make an informative paper, but simply observing that forecast loss and planning success differ would not be a new contribution—the JEPA-WM authors already investigate that distinction. [Appendix G.3](https://arxiv.org/html/2512.24497v4#A7.SS3)

We can draft a clearly labeled development-results paper now. We cannot yet claim a repeatable improvement to the full JEPA-WM recipe, and HMM is still an untested direction, not a finding.

*The original supplied summary included the source assessment note: “Evidence checked using the provenance skill.”*

## Prior art: five papers to prioritize

**There is a plausible empirical contribution, but we have not yet established a novel, practical steering method.** The strongest question is:

> Which parts of a frozen JEPA world model's prediction error can be corrected through structured internal interventions—and when do those corrections improve the decisions made from its forecasts?

The comparison below uses arXiv versions available September 8, 2026. The descriptions summarize the papers; the proposed openings for our work are our interpretation. This focused comparison does not establish a priority or “first” claim.

| Paper | What it already establishes | Where our work could add something |
|---|---|---|
| **1. [JEPA-WM: What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/html/2512.24497v4)** — Terver et al.; December 2025, v4 September 2026 | Studies architecture, training, and planning choices for JEPA world models. Appendix G.3 explicitly investigates the imperfect relationship between prediction accuracy and planning success. | We intervene inside an already-trained dynamics predictor, holding its weights fixed, to investigate which forecast errors are correctable. Simply finding that prediction loss and success differ would add little. |
| **2. [Interpreting Physics in Video World Models](https://arxiv.org/html/2602.07050v1)** — Joseph et al.; February 2026 | Studies V-JEPA 2 and VideoMAE-v2 using layerwise probes, spatial analysis, attention ablations, and steering. Finds distributed physical representations; controlling decoded motion direction requires coordinated changes across many dimensions. | Our question concerns correcting action-conditioned future predictions. “Physics is distributed” is an established starting point. We could investigate whether useful forecast corrections occupy a much smaller subspace than the physical representations themselves. |
| **3. [COAST: Contrastive Conceptor Activation Steering](https://arxiv.org/html/2605.17144v1)** — Miao et al.; May 2026 | Fits success/failure conceptors offline and applies a multiplicative activation gate to frozen robot policies. Evaluates simulation, real robots, multiple policy architectures, and some transfer between tasks. | COAST steers the action generator. We edit the predicted consequences of candidate actions, which an unchanged planner then evaluates. That creates a research opportunity, but applying an existing steering idea to another component is not automatically a substantial contribution. |
| **4. [Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior](https://arxiv.org/html/2605.05115v1)** — Wurgaft et al.; May 2026 | Compares linear and manifold-following activation interventions in language models and a recurrent visual Mountain Car world model. Shows that respecting curved geometry can produce more coherent intermediate outputs. | We could establish when geometric fidelity actually improves forecast accuracy and action selection. Their world-model demonstration concerns coherent positional interpolation; it does not establish a general recipe for improving JEPA planning. |
| **5. [Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency](https://arxiv.org/html/2608.12939v1)** — An et al.; August 2026 | Measures how visually perturbed observations diverge under identical future actions. Connects that divergence to prediction error and planning-cost changes, with experiments on LeWM and PLDM. | ACPC diagnoses sensitivity and checkpoint robustness. Our interventions could identify where correctable errors arise and how to repair them. We would need evidence that the repair preserves distinctions between action consequences, beyond reducing one aggregate error metric. |

These papers converge on a shared problem: a representation can contain physical information, support a convincing probe, or look geometrically coherent without necessarily supporting reliable control. Our opportunity is to explain a specific part of that chain.

## Three contribution questions grounded in our findings

### 1. What determines whether an internal correction helps or harms?

**Strongest existing mechanistic lead.** Our spatial and pathway controls provide more specific evidence than a general claim that steering sometimes works.

- On Reach, the joint intervention beats its spatially permuted counterpart by **2.02% of native forecast error**, with simultaneous 95% interval **[1.09%, 2.94%]**. Both retain broad spatial support; the arrangement of the visual edit matters. [Reach coupling evidence](artifacts/offline_study/primary-durable-20260907/metaworld-author-extension-20260907/analysis-full-v1/bfloat16/reach/vision_action_coupling/report.json)
- On Push-T, the visual-only edit **increases error by 1.02%**, while the action-condition-only edit **reduces it by 0.34%**, below our useful-effect threshold. This identifies opposing component effects within the tested intervention. [Push-T coupling evidence](artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/vision_action_coupling/report.json)

**Potential contribution:** an experimentally supported explanation of which spatial and vision–action relationships make forecast correction effective.

We have evidence of this structure. We have not yet identified its physical meaning, such as object position, contact, or goal-relative motion. That is the drill-down that could turn an ablation result into a mechanism. These comparisons do not establish that joint steering is always necessary or that the two pathways exhibit mechanistic synergy.

### 2. Can a distributed physical representation admit a compact error correction?

**Strongest connection to rank 4.** The original rank-4 interventions reduce H6 proprioceptive embedding error by **2.89% on Reach and 2.14% on Reach-Wall**. The combined Reach intervention reaches **4.82%**. These are development results from the original operators. [Corrected offline results](reports/CORRECTED_OFFLINE_RESULTS.md)

The dimensionality needed to represent or arbitrarily control a physical variable need not equal the dimensionality needed to correct a particular prediction error. Consequently, our rank-4 result does not contradict Joseph et al.'s distributed representations. It also does not demonstrate that the world model represents physical state in four dimensions.

**Potential contribution:** characterize which errors permit compact correction and why. Merely reporting that four directions improve MSE would be a narrower empirical observation.

The original combined result is useful evidence, but its component-removal arms remove intervention energy as well as a component. Beating those arms does not establish equal-energy mechanistic synergy. [Combined development protocol and results](docs/COMBINED_DEVELOPMENT.md)

### 3. Which geometry measurements actually predict useful interventions?

**Promising boundary result.** In Push-T FP32, cubic reconstruction dramatically improves the omitted-activation reconstruction diagnostic, yet the corresponding forecast improvement is essentially zero. The reconstruction advantage reverses under BF16. These are distinct endpoints: local activation reconstruction and downstream H6 forecast error. [FP32 geometry evidence](artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/float32/pusht/action_response_geometry/report.json), [BF16 geometry evidence](artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/analysis-v1/bfloat16/pusht/action_response_geometry/report.json)

**Potential contribution:** identify a concrete failure mode in using local geometric reconstruction to select steering interventions.

This does not refute manifold steering: the geometries, interventions, and endpoints differ. Its value would come from explaining precisely why our diagnostic succeeds numerically while failing to identify a useful correction. We should not generalize this result to all nonlinear interventions or treat the reconstruction ratio as a task-performance gain.

## Revised paper positioning and limits

**Lead with question 1, use question 2 to motivate a compact intervention, and treat question 3 as a boundary on the recipe.** The defensible positioning today is:

> A controlled study of correctable error structure in frozen JEPA predictors.

A practical-method claim would require the cheaper, single distributed edit to retain benefits under a measured inference budget and improve completed planning evaluations. Rank alone does not establish efficiency: the costly response-probing procedure used to construct the original edit must be distinguished from applying the resulting low-rank correction. A cheaper successor remains a candidate until it is validated.

Claims of generalization, superior planning, or a new efficient steering algorithm remain unestablished. The completed primary evidence comprises **81 development trajectories: 33 Reach, 27 Reach-Wall, and 21 Push-T**. Repeated windows, arms, and numerical precisions do not multiply that independent sample count. Fresh confirmation and independent training-seed replication remain necessary for stronger claims. [Evidence scope and remaining work](reports/CORRECTED_OFFLINE_RESULTS.md)

The literature comparison motivates these research questions; it does not authorize a new experiment sweep or establish that every finding is independently novel.
