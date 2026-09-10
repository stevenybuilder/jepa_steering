# Physics Emergence Zone: implications for JEPA-WM

## Immediate use in the geometry-map experiment

The Physics Emergence Zone (PEZ) result is a prior, not a rule that fixes the intervention layer. Physical variables often become usable at intermediate depth, but our target is JEPA-WM's six-block action-conditioned predictor rather than only a video encoder. We therefore measure layerwise emergence on a small candidate sweep and separately identify the low-dimensional subspaces inside those layers.

The current map is deliberately **task-conditioned**. Reach-Wall rollouts tell us where and how this frozen JEPA-WM represents Reach-Wall state, obstacle geometry, progress, and action consequences; they do not establish a universal JEPA coordinate system. The general object is the discovery procedure:

```text
(frozen JEPA-WM, task rollouts, local state/action/outcome signals)
    -> (useful layer, time, subspace, causal intervention)
```

Episode overfitting is controlled by complete-trajectory splits, different initial conditions, collection-seed-held-out probes, matched counterfactual actions, and untouched on-policy episodes. The exact Reach-Wall directions may not transfer to Push-T. A stronger reusable-method result would be that the same frozen procedure rediscovers a different useful subspace on Push-T and again produces a behavioral effect.

PEZ also warns that coordinates sharing a layer may occupy distinct or nearly orthogonal subspaces. Direction should be modeled circularly, such as `(sin(theta), cos(theta))`, and iterative orthogonal probes should test whether a variable is a population code rather than a single direction. Principal angles and subspace overlap should quantify whether progress, obstacle geometry, direction, and magnitude share geometry.

## Differentiator from PEZ

PEZ asks where predefined physical variables such as direction, speed, and plausibility are encoded. Our proposed method asks which task-conditioned internal coordinates are both used by the action-conditioned predictor and causally useful for planning. Decodability is only a screening stage; fixed-observation action patching, planner replay, snapshot forks, and finally closed-loop task behavior decide whether a subspace matters.

Outcome labels alone are not assigned to every frame. For world-model control, timestep-local simulator variables, candidate-action consequences, planner rank/cost, progress, and future return are more defensible signals. If outcome-conditioned geometry is used, it should compare matched states or value-to-go rather than labeling every early state of a failed episode as intrinsically bad.

A regularized Fisher-style screen can be one candidate subspace estimator:

```math
B_\ell v = \lambda (W_\ell + \gamma I)v,
```

where `B_l` measures matched between-outcome or between-advantage separation and `W_l` measures ordinary within-condition variation. Its top generalized eigenvectors form a subspace `U_l`; causal tests, not the eigenvalue alone, determine whether it is useful.

## Practical steering hypothesis

The headline is a reversible post-training control interface for a frozen world model, not an atlas for its own sake and not an unmeasured claim of lower compute. Discovery can be expensive, but a frozen low-rank runtime edit can be cheap:

```math
h' = h + \alpha Uv
```

or

```math
h' = h + UAU^\top h.
```

The practical claim must be judged by closed-loop behavior. A successful system would use cached task rollouts to discover a layer and subspace, validate that subspace causally, and apply a small task-specific adapter without changing base-model weights. It would also provide diagnosis: whether a failure comes from missing representation, wrong predicted dynamics, or planner misuse.

## Later, after the first behavioral effect

Once the core operator works, compare it honestly with a small LoRA or adapter on adaptation compute, final task success, inference overhead, and collateral change on unrelated tasks. Modularity, reversibility, inspectability, and preserving the frozen base model may be advantages even if discovery costs more than a small fine-tune. HMM/GMM routing is not part of the first test; it becomes eligible only if cached rollouts show distinct regimes, routing improves held outcome prediction over static routing, and static versus routed interventions are meaningfully different treatments.

The atlas is supporting infrastructure. The intended contribution is the repeatable interface:

> Given a frozen JEPA-WM and task rollouts, automatically find a causally useful internal control surface and use it to improve held-out planning behavior.

## Sources

- [Interpreting Physics in Video World Models (Physics Emergence Zone)](https://arxiv.org/abs/2602.07050)
- [Causal Physics Steering in Video World Models via Concept Activation Vectors](https://arxiv.org/abs/2605.24322)
