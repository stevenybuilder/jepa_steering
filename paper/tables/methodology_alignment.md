# Methodology alignment with JEPA-WM

This is an alignment check, not a claim that our interventions reproduce their ablations. Sources: [JEPA-WM Table 1 and Section 4](https://arxiv.org/html/2512.24497v4#S4), [our planning-alignment audit](../../reports/PLANNING_METHOD_ALIGNMENT.md), [our active protocol](../../docs/EXPERIMENT_PLAN.md), and [study configuration](../../configs/study.json).

| Component | JEPA-WM's design choices | Our corresponding treatment |
|---|---|---|
| Visual encoder | DINOv2 S/B/L; DINOv3 L; V-JEPA L; V-JEPA 2 L | Retain released task checkpoint; no encoder-family sweep |
| Action conditioning / positions | Feature, sequence, AdaLN, AdaLN-zero; sincos / RoPE | Retain architecture; intervene in vision/action paths |
| Predictor depth | 3, 6, 9, 12 | Retain six-block primary predictor; vary where activation edits are applied |
| Training rollout | 1, 2, 3, 6 steps | No base-weight changes for intervention comparisons; evaluate H6 forecasts |
| Context | 1, 2, 3, 5, 7, 9, 14 | Preserve corrected offline context 3; planning context is separately specified |
| Proprioception | With / without | Retain released input arrangement; measure both visual and proprioceptive embedding endpoints |
| Planning optimizer | CEM, NeverGrad, Adam, GD | Retain native CEM; no optimizer search in this intervention study |
| Planning distance | L1 / L2 | Retain released task-specific cost, including proprioceptive weight where specified |

The authors vary components against a reference and combine promising choices. We adopt that organizational logic for a different intervention design space, summarized in the draft's Table 1. Our layer-support ablation is not their predictor-depth ablation, and our vision-action edit is not their with/without-proprioception training comparison.

## Statistical and evaluation distinctions

- JEPA-WM reports three independently trained seeds for its final-model comparison, with 96 episodes for most tasks and separate counts/endpoints for robot-data evaluations. This does not mean every exploratory ablation used the complete final-model replication protocol.
- Our original primary comparison uses the full released 33/27/21 validation pools after authorized exposure, with many prefixes per trajectory and paired-lineage inference. This reproduces specified loader/context/metric behavior, not their complete training histories or an untouched confirmation cohort.
- The existing paper-alignment audit records a discrepancy between the paper's 96-episode description and released MetaWorld configurations containing 48. Our behavioral contract explicitly chooses 96. Matching that count alone is not matching statistical power.
- Their main quantitative task figures are Figures 3-7, and their final cross-task comparison is Table 2. Our current analogues show forecast-error contrasts. An equivalent success-rate comparison must wait for the completed behavioral panel.
- DROID uses a recorded-action endpoint. Do not place its action score on the same axis as simulated success percentages without explicitly distinguishing the metrics.

## What the current Table 1 means

The original five sweeps are fully measured for three primary tasks in BF16 and FP32. Temporal/HMM routing is a separate unfinished direction. Original combined/drop-one evidence exists for Reach; the fixed-response successor has a separate two-task offline evaluation. Neither is silently substituted for the five original sweeps.
