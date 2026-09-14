# Related work

JEPA-WM provides the predictive model and planner used throughout this project.
The surrounding literature helps distinguish what an internal intervention changes:
a decoded representation, a forecast, the planner's preferences, or an executed
robot trajectory.

## Predicting futures for planning

**Terver et al., JEPA-WM (2026; version 4).** JEPA-WM compares representation,
conditioning, training, and planning choices. Its planner scores candidate actions
through latent forecasts and their distance to an encoded goal. We retain the
released checkpoints and intervene during prediction. Published benchmark results
use the authors' checkpoint and evaluation aggregation; our intervention comparisons
use a concurrent unsteered baseline on paired starting states.
[Official implementation](https://github.com/facebookresearch/jepa-wms).

## Physical information across layers

**Joseph et al., Interpreting Physics in Video World Models (2026).** The
Physics Emergence Zone analysis studies how video encoders represent physical
variables across depth. It combines probes, subspace geometry, patch-level decoding,
and targeted attention ablations. This motivates examining layers and their
responses to intervention together. Our six-block action-conditioned predictor
has a different role from their video encoders, so a similar depth pattern alone
would not identify the same mechanism.

The project's [layer response maps](MECHANISMS.md#layer-response-map) measure
forecast-error changes. Its [attention analysis](PILOT_MECHANISMS.md) and
[precision controls](CONTROLLED_GEOMETRY.md) provide complementary measurements.

## Steering a policy and steering a predictor

**Miao et al., COAST (2026).** COAST uses success/failure activation statistics
to construct a conceptor that modifies a frozen policy's action-generation
computation. Our edits modify forecasts that an external planner subsequently
scores. This additional step makes candidate costs, rankings, and adaptive search
relevant measurements alongside prediction error and task success.

A conceptor's effective rank also differs from the rank of its activation change.
For a conceptor matrix C and strength β, the change is β(C − I)h. A low effective
rank for C does not imply a low-rank additive edit. Our rank-four operator and
its random-subspace control are defined explicitly in [Methods](METHODS.md).

## Geometry and intervention validity

**Wurgaft et al., Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior (2026).** This work compares curved and straight
paths through representations and their output consequences. It motivates testing
geometry through interventions while keeping their support and readout explicit.
Our sparse interpolation measurements do not establish a dense physical manifold.
[Numerical controls](CONTROLLED_GEOMETRY.md) show why precision must be specified
when comparing linear and cubic reconstruction.

**Zhang and Nanda, Towards Best Practices of Activation Patching in Language
Models: Metrics and Methods (2023).** Their analysis shows how patching conclusions
depend on the intervention and evaluation metric. Our [action-history
replication](LCFM_REPLICATION.md) checks a patch against a specified changed-input
forecast at multiple prediction steps, then measures the consequence for candidate
selection. It tests temporal consistency within JEPA-WM's two-frame context.

Together, these studies motivate checking the full chain from internal edit to
forecast to planning decision. The [experiment inventory](ANALYSIS_COMPLETION.md)
links the measurements used to examine each part of that chain here.
