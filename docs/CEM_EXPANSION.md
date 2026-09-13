# Does the CEM search effect persist beyond the initial eight cases?

Status: **all 56 extension cases and their original trace archives are complete and verified**. The unchanged analysis ran after the complete preservation gate; its [result receipt](../paper/data/cem_expansion_summary.json), SHA256 `2952a0c0a96591834fcdeb7daa7fb01b641c8fa50758c9a55d55eadc91c263d7`, binds all seven exported CSVs and all 64 sources, including the separately retained original eight. The [analysis freeze](../paper/data/cem_expansion_analysis_freeze.json), recorded at 2026-09-13 19:11:17 UTC, binds source SHA256 `dcccc73c69a9248e83482376a6a1c50424b80db4bbbd6d2eb5517300e3868bc3`. The [frozen protocol](../paper/data/cem_expansion_protocol.json) fixes 56 new historical-development contexts, episode IDs 4–31 in Reach and Reach-Wall. Its SHA256 is `4766488e4907dff62f588894845f5b5aecf084e2c47a5ab4a1605cc2f07463db`. The original eight cases, whose effects were already viewed, remain separate and are not included in primary inference.

## Complete-cohort findings

Both learned and calibrated-random edits change CEM's search and returned prefixes. Mean returned-prefix RMS distance from concurrent native is **0.6974 versus 0.5777 in Reach**, and **0.4998 versus 0.2171 in Reach Wall**, for learned versus random. These are unsigned differences across all 60 prefix coordinates, not measures of action quality. The [prefix summary](../paper/data/cem_expansion_selected_prefix_summary.csv) and [individual cases](../paper/data/cem_expansion_selected_prefix_cases.csv) retain the full variation, including near-zero differences.

However, **all six prespecified learned-minus-random comparisons have family-corrected intervals spanning zero**. The enlarged sample does not establish a learned-specific excess search effect under this registered family. It also does not establish equivalence or absence of an effect.

| Task | Learned minus calibrated random | Mean | Marginal 95% interval | Family-6 interval |
| --- | --- | ---: | --- | --- |
| Reach | Returned-prefix RMS difference from native | +0.1197 | [−0.0634, +0.2924] | [−0.1284, +0.3487] |
| Reach | Mean proposal-mean RMS difference from native | +0.0403 | [−0.0412, +0.1122] | [−0.0717, +0.1353] |
| Reach | Mean proposal entropy difference, nats | +0.2631 | [−0.5480, +1.1339] | [−0.8152, +1.4570] |
| Reach Wall | Returned-prefix RMS difference from native | +0.2827 | [+0.0767, +0.4707] | [−0.0070, +0.5294] |
| Reach Wall | Mean proposal-mean RMS difference from native | +0.0809 | [−0.0146, +0.1648] | [−0.0541, +0.1929] |
| Reach Wall | Mean proposal entropy difference, nats | +0.3596 | [−0.3625, +1.0744] | [−0.6327, +1.3388] |

Every row uses n=28 paired contexts in the task. Values come from the six `extension56` rows of [primary_summary.csv](../paper/data/cem_expansion_primary_summary.csv). Reach Wall's prefix contrast excludes zero only in its uncorrected marginal interval; the registered family-6 interval includes zero. The exploratory initial-eight and descriptive pooled-64 rows are not substitutes for these primary results.

The first candidate bank is byte-bound across arms, but **all ten initial elites are shared in 102/112 edit-context comparisons**, not all 112. The remaining ten comparisons share nine elites. Later candidate banks differ adaptively, so matching later candidate IDs would not compare the same actions. This extends, rather than silently repeats, the initial-eight observation that all initial elite sets happened to match.

Source-bound figures show the complete registered sample: [search trajectories](figures/cem_expansion_search.png), [proposal entropy](figures/cem_expansion_entropy.png), and [returned prefixes](figures/cem_expansion_prefixes.png). Proposal curves use 120 coordinates; returned prefixes use 60. Curve bands are scenario mean ±1 SE; registered paired intervals remain in the table and CSVs. No physical actions were executed by this CEM-only study. The separately registered [physical-prefix replay](PLANNED_PREFIX_REPLAY.md) tests matched H3 endpoints and is not implied by these search differences.

## Cohorts and comparator

The primary cohort is the complete extension: n=28 scenarios per task. Each context reruns native, learned rank-four and calibrated-random CEM on the **same receiving GPU**, with the same scenario seed and frozen fitted operators. All three searches are also rerun untraced to verify selected-prefix and private-generator parity. Each search has 15 iterations, 300 population candidates, ten elites and 30 model callbacks: population and updated-mean forecasting on each iteration. The full extension has 336 searches and 10,080 callbacks; these are not independent observations.

The original eight cases use their original archived native comparator. They are reported as `initial8`, n=4 per task. The combined `pooled64`, n=32 per task, is descriptive, not a newly independent confirmatory cohort. Primary tests use only `extension56`. No simulator actions, intervention refits, new task checkpoints or protected evaluation are included.

## Primary comparisons and uncertainty

There are three per-scenario learned-minus-random contrasts, relative to concurrent native:

1. Returned-prefix RMS distance from native, using all **60 coordinates**, shape `(3,20)`.
2. Proposal-mean RMS divergence from native, averaged over all 15 iterations, each using **120 coordinates**, shape `(6,20)` with an optional singleton population dimension in the raw trace.
3. Preclip diagonal-Gaussian differential-entropy change from native, averaged over all 15 iterations.

An unsigned distance becoming larger is not a control benefit. Entropy changes can be negative and are measured in the checkpoint's action coordinates before clipping and deterministic mean insertion. They are not physical uncertainty, attention entropy or a score-softmax diagnostic.

The analysis uses 20,000 paired scenario-bootstrap draws, seed20260913. The same scenario weights apply to all arms, iterations and metrics within each task. Primary percentile intervals use Bonferroni family6 (two tasks × three metrics): quantiles `0.05/12` and `1−0.05/12`; marginal 95% intervals are also exported. These six intervals do not correct every point along the secondary curves. Scenario-level standard deviations and **one standard error**, `sample_SD/sqrt(n)`, support the requested plots; SE is explicitly not a confidence interval or training-seed variation.

The descriptive pooled bootstrap preserves the fixed 4+28 scenario strata. Initial and pooled summaries receive no primary Bonferroni inference labels. More scenarios improve precision but do not guarantee adequate power for a given effect; there is no outcome-based sample-size extension beyond the registry.

## Identity, provenance and completeness

Initial candidate arrays must have identical shape, dtype and SHA256 across the three arms. Initial elite overlap is **measured**, not required to equal ten: a real loss of shared elites is retained. Later populations adapt independently; later candidate IDs are not treated as the same actions, and no later elite-index overlap is calculated. The first differing action-array fingerprint can be reported descriptively without selecting cases or iterations.

The [CPU analysis](../analysis/mechanism/cem_expansion_summary.py) imports only hash-verified pure trace functions from the frozen eight-case analysis. It reads the scientific/execution/input manifests before case payloads and requires all56 extension cases plus all8 original cases, completion markers and full cloud-archive proof. It checks checkpoint, fit, input, source, GPU and strict-FP32/TF32-off provenance; all three traced/untraced 30-callback/RNG receipts; compact file hashes; and native/arm parent-trace archive membership. Original-eight native traces have their own complete full-raw preservation receipts. A compact-only or raw-pending receipt does not pass this study's final gate.

CPU recomputation verifies entropy, plan-coordinate differences, proposal differences, margins, valid elites and initial-array fingerprint identity. Full tensor equality and RNG checks remain execution-attested, backed by preserved raw traces; the laptop does not download all candidate tensors. Published original-eight CSVs and figures are never rewritten.

## Public table contract

| Output prefix `paper/data/cem_expansion_` | Keys and values |
| --- | --- |
| `iteration_cases.csv` | `cohort,task,episode,arm,iteration`; proposal-mean/std L2 and RMS, native/arm/delta entropy, native/arm/delta margins, initial-only elite overlap and score changes. All64 cases, both edit arms, all15 iterations: 1,920 rows. |
| `selected_prefix_cases.csv` | `cohort,task,episode,arm`; prefix L2/RMS, final-proposal-mean L2/RMS, explicit60/120 coordinate counts and first action-bank divergence iteration. 128 rows. |
| `primary_cases.csv` | `cohort,task,episode,metric,learned,random,effect`; three scenario-level contrasts. 192 rows. |
| `iteration_summary.csv` | `population,task,arm,iteration,metric,n,mean,scenario_sd,scenario_se,marginal_95_low,marginal_95_high`; all three populations. Curves are descriptive. |
| `selected_prefix_summary.csv` / `shared_iteration0_summary.csv` | Same summary fields without `iteration`; all three populations and both arms. |
| `primary_summary.csv` | `population,task,metric` plus summary fields; `bonferroni_family,bonferroni_95_low,bonferroni_95_high,inference_scope`. Only six `extension56` rows have family6 primary intervals. |
| `summary.json` | Complete status, cohort counts, all source/DONE/cloud and output CSV hashes, analysis/source/protocol freeze bindings, local-versus-GPU verification scope. |

For figures, use `population == "extension56"` as the main curve and `scenario_se` for visibly labelled ±1SE bands. `initial8` can appear as a distinct exploratory reference. Never silently mix `selected_prefix_delta_rms` with `proposal_mean_delta_rms` or `final_mean_delta_rms`. Every candidate/iteration/scenario remains present; no clipping long tails to simplify the story.

## Reproduction

Before outcome access, create the source-hash-bound analysis freeze with `--freeze paper/data/cem_expansion_analysis_freeze.json`; creation is exclusive, so an existing freeze cannot be silently replaced. Once the runtime owner delivers complete preservation, invoke the new analysis with the exact execution-manifest path/SHA and compact root. The default compact filename is `steered-cem-summary.json`; the original-eight root and original-native root remain separately specified. Tests use `python -m unittest discover -s tests -p 'test_cem_expansion_summary.py'`.

The complete registry passed verification without an analysis-source amendment. This result measures adaptive-search differences, not physical efficacy; it does not infer improved control from greater distance to the native search.
