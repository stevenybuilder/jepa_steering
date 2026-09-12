# Introduction and contribution framing

## The conceptual picture

**Recommended analogy: a map that preserves the relationships needed to choose a route.** A map can have locally inaccurate distances yet be useful for navigation; a visually detailed map can be useless if a crossing is misplaced. A world-model representation similarly needs to preserve the consequences of actions that matter to the planner. Our evidence currently measures forecast correction, while the effect on route selection remains pending.

Use the analogy briefly, then translate it into measurable questions. Avoid making geometry sound like a newly discovered prerequisite of latent representations, or claiming that the current edit identifies a physical obstacle, contact relation, or semantic coordinate. We have not established those mechanisms.

Tong, Fan, and colleagues use Plato's cave to motivate moving from text toward richer visual experience in [Beyond Language Modeling](https://arxiv.org/html/2603.03276v1). Their rhetorical structure is useful: conceptual limitation, practical motivation, unresolved design choices, and controlled empirical findings. Our corresponding limitation is uncertainty about which features of an already learned predictor can be corrected usefully. We are not testing their claim about multimodal pretraining, and the draft does not reuse their prose or cave metaphor.

## Why this matters now

- **An actionable research junction:** pretrained world models, geometric interpretability, and frozen-policy steering provide concrete components for studying the missing connection between internal correction and decisions.
- **Practical motivation:** updating every base model for every task is costly. A small offline-fitted correction could offer a reusable interface, if benefit, latency, and transfer are verified together. Current fits remain task-specific.
- **Scientific motivation:** successful reconstruction or probing does not establish a useful correction. Matched controls and downstream endpoints can distinguish these claims and uncover where attractive geometric diagnostics fail.

These are motivations, not completed findings. They should not be used as evidence of generalization, production readiness, or an improvement to every JEPA model.

## Contribution bullets for the current draft

1. **A controlled study of correctable forecast structure.** Five intervention categories separate pathway, rank, depth, spatial support, and local response geometry, with native, zero-dose, permutation, and matched-random controls.
2. **An offline-fitted distributed correction without online response probes.** The implemented fixed-response operator has new, separately verified positive offline results on Reach and Reach-Wall. Its planning benefit remains pending.
3. **A boundary on geometric reconstruction as a steering diagnostic.** Numerical precision can reverse the local reconstruction comparison, and the better FP32 reconstruction does not establish a useful forecast improvement.

These are proposed empirical/methodological contributions. Do not label the underlying PCA, damped inverse, or amortization mathematics as newly invented. The literature review has not established an exclusive priority claim for this combination.

## Four-page structure

| Page | Purpose | Content |
|---|---|---|
| 1 | Explain the question and why it matters | Abstract; map analogy; prior-art gap; three bold contribution statements |
| 2 | Make the experiment understandable | Model/data/endpoint; Table 1; original and fixed-response operators; calibration scope |
| 3 | Present the evidence | Effects and intervals; original versus successor; pathway/spatial controls; depth; geometry boundary |
| 4 | Explain implications and limits | Relationship to prior art; full-system compute; pending behavior; sample/exposure limits; bounded future directions |

Figures, references, detailed methodology alignment, task coverage, and provenance notes are separate artifacts. The current PDF is a four-page development draft in a neutral single-column layout, not a claim of compliance with an unspecified venue template.

## What changes after behavioral completion

Use only the completed frozen comparison, including random controls and all registered arms. If it establishes a useful benefit, the headline can extend from forecast correction to measured planning improvement. If it is null, preserve the forecast finding and explain any diagnosed failure mechanism without claiming that the already-known loss/success mismatch is new. Fresh confirmation, model-history replication, and task transfer remain distinct steps in either case.
