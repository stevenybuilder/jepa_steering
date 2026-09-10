# Bounded planner-decision diagnostic — September 10, 2026

The user explicitly approved completing one diagnostic and the final write-up,
not reopening the six-task/family/history campaign. This is a new, prospectively
specified **development diagnostic**, not fresh confirmation, a replay of the
old planner's random-number stream, or a replication of full author CEM.
All completed results and original freezes remain unchanged.

## Question and fixed scope

Do the existing refined fixed-response and equal-budget coupling edits change
the ordering of identical candidate actions, and do their selected action
prefixes make better physical progress than native and matched-random edits?

- Tasks: Reach and Reach-Wall; all 96 existing canonical development scenarios
  per task, without selecting successes, failures, or large offline effects.
- Conditions: native, fixed_rank4, matched_random_fixed_rank4, coupling_only,
  matched_random_coupling; existing fitted banks, doses and H3/B3 hooks only.
- One checkpoint: the previously hashed MetaWorld release, strict FP32/no TF32.
- One decision per initial state; no new learned method, fitting or dose search.
- Use the exact pinned author's CEM code to capture its **first 300-candidate
  population**, before any outcome-dependent CEM updates: H6, variance 1,
  zero-mean candidate at index 0 and the configured clipping rules.
- Reset a dedicated candidate RNG per scenario, seeded by SHA256 of
  `jepa-decision-v1/<task>/<environment_seed>` (first 8 hexadecimal digits).
  This is a new disclosed diagnostic seed policy, not the historical full-CEM
  stream. Save the actual candidate tensor and its hash; share it across arms.
- Rank candidates using the native terminal visual MSE + 0.1 proprioceptive
  MSE objective. Select its minimum (lowest index breaks an exact tie).
- Execute only the chosen first three model actions / 15 elementary actions
  in the official simulator from the same reset state; no replanning.
  These are 960 scenario-condition short tests, not 960 full episodes.

The first population is a deliberately limited common-action test. It does not
sample the final optimized CEM population, execute the elite mean, diagnose late
episode failures, or establish full-task success improvement. H6 ranking and
15-step progress have different horizons; report this mismatch explicitly.

## Pairing, fidelity and engineering

Restore the canonical goal bank and original stimulus/source receipts. Verify
all 192 goal tensors and fitted-bank hashes before a run. Use existing initial
state/goal metadata and the source-native reset/warmup path. Initial physics and
actual observations must match across arms; no tolerance is used to accept
different physical states. Historical goal pixels remain the model input.

On an excluded engineering seed, require exact native repeat, zero-dose identity,
shared-action and initial-state identity, independent action conversion parity,
finite scores, one-pass fixed-map/coupling execution and unchanged parameters.
Never restore a full old queue to perform this check. Preserve failed attempts;
no retry may change scientific thresholds or exclude inconvenient scenarios.

Save every candidate action, every condition's 300 scores, selected index,
top-two score margin, top-ten overlap, rank agreement, initial/goal bindings,
requested/delivered intervention information, actual executed actions, simulator
state trajectory, terminal distances, timings and source/device metadata.
If two conditions select the identical candidate, its simulator outcome may be
reused only within that same scenario after reset repeatability is established;
record this reuse explicitly, not as an additional independent sample.

## Analysis fixed before new outcomes

The primary endpoint is terminal end-effector Euclidean distance to the expert
goal's end-effector position after 15 elementary actions (smaller is better).
Also retain the original full-state goal distance, simulator success at that
prefix (NOT full-episode success), reward and action norms as descriptive checks.

Four contrasts per task: refined versus native, refined versus its random,
coupling versus native, coupling versus its random. Positive differences are
reference distance minus treatment distance. Average paired differences over all
96 scenarios; 20,000 whole-scenario bootstrap draws, seed 2026091022, with
Bonferroni simultaneous 95% intervals across the eight fixed primary contrasts.
No statistical unit is created by the 300 candidates or repeated conditions.

Report the fraction of changed selections, score-rank agreement and top-ten
overlap against native. Changed-selection-only summaries are descriptive,
post-treatment-conditioned analyses, not causal subgroup effects. Do not infer
best-of-300 physical regret: only the selected candidates are simulated. No
forecast-to-success regression may join unmatched offline and simulator rows.
All 192 scenarios and five arms are required for a complete comparative report;
partial runs receive an explicitly incomplete report and never an efficacy claim.

## Execution budget and stopping

Use one US GPU, targeted at <=$0.75/hour, with a **$5 total new rental/transfer
budget**; do CPU development before leasing. Validate a short measured run before
the full diagnostic. The previous 51-minute estimate covers model scoring only,
not implementation, asset restoration, reset/rendering or analysis. It is not
a guaranteed completion ETA. Do not expand the sample or family after results.

Preserve raw results, code, logs and analysis in Google Drive and verify their
bytes before destroying the new disposable worker. Existing inaccessible source
disks are not part of this execution and must remain untouched.

## Possible non-steering research direction

Existing offline errors, lineage records, precision comparisons and behavioral
outcomes support exploratory evaluation research. The strongest proposed
extension is decision-quality auditing, using existing edits as controlled
perturbations rather than introducing a new steering method. The old action
trace saves hashes, not candidate scores or recoverable action arrays, so those
new outcomes cannot be reconstructed from the old trace alone.

Purely unsteered error-growth/precision analyses are possible secondary questions,
but no new benchmark, calibrated uncertainty predictor, causal failure taxonomy,
or general mechanism is established merely by reframing the project.
Relevant prior art already includes JEPA-WM Appendix G.3,
https://arxiv.org/html/2512.24497v4#A7.SS3,
and decision-fidelity evaluation such as WorldModelGym,
https://reka.ai/news/worldmodelgym. New work must demonstrate a specific finding;
the generic prediction-versus-decision gap is not itself a novelty claim.
