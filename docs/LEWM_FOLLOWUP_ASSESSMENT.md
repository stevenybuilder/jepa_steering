# Should the context experiment extend to LeWM?

**Recommendation: run a limited PushT pilot, then decide on a fresh replication.**
Research and source check: September 14, 2026. No LeWM inference or rental has
started. The completed JEPA-WM replication remains the evidence for the current
LCFM paper. Its 200-state results support failure of one-step validation, but do
not show that mismatch grows with context length.

LeWM can test whether the effect occurs in a second architecture and whether a
third occurrence of an action matters. Its released PushT predictor supports
**three frames**, compared with two in our JEPA-WM checkpoint. That is a modest
extension of explicit history. It does not establish long-context scaling.

## What the literature and source support

| Source | Relevant evidence | Consequence for this study |
|---|---|---|
| [LeWorldModel, v3, June 3, 2026](https://arxiv.org/html/2603.19312v3) | Jointly learned encoder and predictor; roughly 15M parameters in the paper. PushT and Cube use three-frame history. Published planning uses five model steps with five elementary controls per step. | A feasible second-model test, with a different representation and training recipe. Architecture and task differ from our MetaWorld experiment, so cross-model rate differences cannot identify a context-length effect. |
| [Official PushT checkpoint](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/config.json) and [pinned rollout](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py#L87) | Three-frame context, 192-wide embeddings, six predictor blocks, learned positional table of length three; rollout truncates history. | Do not change a config to 8 or 16 and describe the result as a native longer-context checkpoint. Count actual appearances in the predictor calls. |
| [Zhang and Nanda, 2023](https://arxiv.org/abs/2309.16042) | Activation-patching conclusions depend on measurement and intervention choices. | One-step fidelity is an incomplete validation target for this proposed rollout comparison. A time-indexed check is an extension of existing methodology, not grounds for claiming invention of causal patching. |
| [Fast LeWorldModel, v1, June 24, 2026](https://arxiv.org/abs/2606.26217v1) | Replaces local autoregressive rollout with action-prefix prediction and studies horizon-dependent latent error. | Ordinary rollout error accumulation is existing prior art. Our proposed claim concerns disagreement between two interventions within one frozen model. Do not conflate the two or claim novelty for error accumulation itself. |

The strongest argument against this follow-up is that reproducing every changed
input pathway should restore agreement by construction. That control alone is
not a discovery. The information gain would come from measuring the additional
error caused by omitting specific later appearances, its persistence after the
action leaves explicit history, and the effect on candidate selection. Neither
the papers above nor this targeted search establishes that no prior study has
examined temporal patch consistency.

## Smallest useful experiment

Use the released **PushT checkpoint only**, frozen, with its native context and
published five-step planning horizon. Choose the first imagined action as the
intervention, so its three appearances occur at predictions H1, H2, and H3, with
H4 and H5 measuring what remains after the action leaves explicit history. Keep
the official initial-observation convention and record the actual conditioning
sequence length at every call. If the evaluator starts with one observed frame,
report that convention explicitly; the predicted history fills the later window.
Do not substitute three observed frames without declaring a separate experiment.

For every state, use the same 300 candidate sequences in every condition. Define
the changed-input reference by the same cyclic donor rule as the JEPA experiment:
candidate i receives candidate (i+1) mod 300's first action block. Keep all other
actions fixed. Replace only the affected action-conditioning occurrence, at all
six blocks, before the block's AdaLN computation. The state history evolves
within each patched rollout. Do not copy the entire donor state or unrelated
actions, which would overwrite the discrepancy being measured.

Freeze these six arms before inference:

1. Unmodified rollout.
2. Capture-only / zero-change hook control.
3. Coherent changed-input reference.
4. Patch only the first appearance.
5. Patch the first two appearances.
6. Patch all three appearances.

Use **16 engineering/development states** to test implementation and estimate
variance, with no claim of confirmation. Keep them out of any subsequent
replication. No layer sweep, learned steering fit, task search, or selection of
favorable donor magnitudes is needed to answer this question.

## Falsifiable questions and controls

**Primary scientific question: does the one-time patch's discrepancy increase
while the action remains in context?** For each state, let D_h be the mean
squared latent distance between a patch and the changed-input reference at
horizon h. Define the paired contrast

    Delta = [D_H3(patch once, reference) - D_H2(patch once, reference)]
            / D_H5(unmodified, reference).

The same within-state denominator is used at both horizons, so a changing
normalizer cannot manufacture apparent growth. A positive mean suggests added
mismatch between the second and third appearances; zero or negative values
weaken the compounding claim. Report the raw squared-distance contrast as well.
A zero or numerically negligible denominator is undefined, retained in the
counts, and handled under a rule fixed using engineering inputs before the
replication. Report an unnormalized analysis over every state alongside it.

This contrast describes net evolution, not a causal isolation of history length:
autoregressive state propagation and the next action also change. The paired
first-two versus all-three condition isolates the omitted third occurrence at
H3; its exact all-three baseline is a construction check. First-only versus
first-two comparisons show the consequence of restoring the second occurrence.
H4/H5 curves distinguish continued growth from persistence or recovery after the
action leaves explicit history. Nonmonotonic curves remain valid results.

**Secondary question: does the mismatch affect candidate choice?** Measure the
fraction of states with a different minimum-cost candidate at H5, the cost of
that selection under the changed-input reference, rank correlation, and overlap
of the official elite set. Keep the official latent goal cost and elite count;
do not copy MetaWorld's proprioceptive term or top-ten rule into a different
planner. Include exact-tie and minimum-set membership checks. A changed winner
with negligible extra cost supports a narrower conclusion than a large cost
penalty. No selected-action execution means no claim about robot success.

**Engineering gate:** zero-change hooks reproduce the native rollout; patching
all appearances reproduces the reference throughout the horizon; patching the
first appearance matches the immediate forecast. Compare full tensors and costs
against deterministic repeat-run error. Freeze the numeric tolerance before
scientific outputs are examined. Failure means inspect the adapter and indexing,
not enlarge the cohort or relax the tolerance to obtain a desired outcome.

## Replication and power

If the engineering gate passes and the pilot shows measurable later mismatch,
freeze a fresh-state protocol, actual checkpoint hash, dataset revision, candidate
seeds, precision, tie handling, and analysis. Prefer **200 independent PushT
starting states** as an initial confirmatory design, with one sampled context per
source trajectory if using dataset trajectories. Bank resamples and candidate
counts do not increase the number of independent states. A second bank is a
paired robustness check. The released model is trained on its source dataset;
fresh project exposures are not necessarily unseen base-model training data.

The primary endpoint is the single prespecified paired Delta contrast above.
Use a two-sided 95% state-bootstrap interval with 20,000 resamples and report its
mean and raw-distance counterpart. Do not switch the primary endpoint to a
favorable horizon or task after the pilot. A practical target is a mean increase
of **0.05 in the fixed normalization**; this is a proposed study-design threshold,
not an effect found in LeWM. Evidence for a useful compounding result requires
an interval above zero and an estimated effect at least that large. Stronger
evidence for exceeding the threshold requires the lower bound above 0.05.

For planning, a rough paired-normal calculation gives
n = (1.96 + 0.84)^2 * (SD_Delta / 0.05)^2 for 80% power to reject zero if the true
mean is 0.05. An SD of 0.25 gives about 196 states; an SD of 0.50 gives about 784.
These are assumptions, not power claims for unknown LeWM data. Use pilot variance
with an uncertainty allowance to fix the final n before collecting fresh states;
check robustness with simulated resampling from the pilot. Increasing n cannot
remove architecture or task confounding.

For the secondary disagreement rate, worst-case marginal 95% binomial precision
is roughly +/-9.8 percentage points at n=100, +/-6.9 at n=200, and +/-4.9 at n=400.
These precision calculations do not justify treating banks as independent.
Secondary curves and condition comparisons use shared state-bootstrap draws;
report them as descriptive or freeze a corrected inferential family in advance.
No optional stopping based on observed significance.

## Compute and implementation recommendation

Start qualification on **one GPU**, with one complete state as the paired work
unit. The official PushT weights are advertised as 72.3MB, but the compressed
source dataset is 13.14GB; expanded storage and inference peak memory are not yet
measured. The full dataset is not repeatedly downloaded per worker. Stage a
verified minimal evaluation shard and required normalization data in GCS once
its provenance and sampling rule are fixed. Count the actual checkpoint
parameters rather than equating file size with parameter count.

A source audit found version-alignment work: the README conversion example passes
Hydra `_target_` fields to plain constructors, and current training data config
names Lance while evaluation instantiates HDF5. The current planner defaults to
one observed frame. Resolve these issues in an isolated pinned adapter before
renting a fleet. Source findings are not observed runtime failures.

After a real 16-state engineering pilot, use measured staging time, seconds per
complete state, and peak memory to compare a single GPU with disjoint replicas.
Eight workers are reasonable only if the remaining work and deadline repay
qualification and data transfer. No tensor parallelism is justified by the model
size. Preserve strict precision and paired-state ownership. Refresh live US-only
provider quotes and document campaign/hourly limits and shutdown reserves before
any lease, using the user's existing expanded spending authorization rather than
asking for it again. No price or runtime estimate is claimed from source alone.

## What would make the LCFM connection stronger?

A successful LeWM pilot would support cross-model generality and a third-action-
occurrence test. A stronger context-length claim needs a separate matched
training study, for example native histories 1, 2, 4, and 8 under the same model
family, data split, parameter budget, and training exposure, with multiple
training seeds. Evaluate a common rollout horizon independently of history
length. Treat token exposure, compute matching, and padding/masking as explicit
controls. This would be substantially more work and is not part of the proposed
checkpoint-only pilot or the current workshop paper.

The current recommendation would change to **defer** if native-hook parity cannot
be established, if source data cannot support a defensible independent cohort,
or if the pilot shows no meaningful discrepancy at later steps. A pilot that
shows mismatch without net growth still motivates an honest second-model
replication of temporal validity, but not the stronger claim that drift compounds.

Primary sources: [LeWM paper](https://arxiv.org/abs/2603.19312v3),
[official implementation](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac),
[checkpoint collection](https://huggingface.co/collections/quentinll/lewm),
[activation-patching methodology](https://arxiv.org/abs/2309.16042),
[Fast LeWM](https://arxiv.org/abs/2606.26217v1).
