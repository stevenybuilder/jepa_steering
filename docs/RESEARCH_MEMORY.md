# Sonar research memory

## Core question — Steven, 2026-09-09

"can you reduce the amount of data, retraining, or manual engineering needed when deployment conditions change?"

Steven explicitly confirmed this is the core question he has been trying to address.

## Motivation and intended scope

Steven's motivation is that JEPA world models can be difficult to productionize, with substantial manual engineering overhead and difficulties in training and fine-tuning, among other deployment issues. Treat this as the motivating problem to investigate and quantify, rather than a measured claim that all JEPA models have identical limitations. LeWM specifically addresses aspects of training stability and complexity; it does not by itself establish low-cost adaptation to changing deployment conditions.

Sonar's proposed approach is a repeatable discovery and selection procedure: discover interventions in a world model, select those appropriate to the current task/state/history, and apply them computationally efficiently to improve executed robotics outcomes. Task-conditioned memory and reuse may support this procedure. Representation geometry and activation patching are possible tools, not the end goal.

Preserve this broader deployment/adaptation objective when designing experiments. A frozen backbone is a candidate implementation constraint, not the user's ultimate objective. Changes in physical dynamics are one possible evaluation setting, not the entire research scope. Success must be established through control outcomes and measured adaptation-data, compute, retraining, and manual-design costs against strong alternatives.
