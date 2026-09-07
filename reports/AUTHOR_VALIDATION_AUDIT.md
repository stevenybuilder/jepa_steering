# Author-validation coverage and rerun accounting — 2026-09-07

Status: metadata verified; no author-matched evaluation has run. Existing results
are preserved as custom offline development evidence. Counts below are prospective
full-pool coverage, not claimed sample counts behind the authors' published curves.

## Provenance

- Governing legacy plan/config introduced in `8f07ad2` at 2026-09-06 23:25:53 EDT.
  That initial configuration already set `authors_split_reproduced: false`.
- Paper: [JEPA-WM v4, Appendix G.2](https://arxiv.org/html/2512.24497v4#A7.SS2).
- Upstream commit: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- Inventory source revision: `6116f042ae7ae4c8e3f1fd2f194f432615664182`.
- Reference configs, relative to pinned upstream:
  `configs/vjepa_wm/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_AdaLN_ftprop_depth6_repro_2roll_save.yaml`
  and `configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml`.
- Metadata receipts: `artifacts/offline_study/author-validation-preflight-20260907/{metaworld,pusht}/preflight.json`.
  Each records hashes of its input manifest, config, optional registry and supplied fits.
- MetaWorld preflight SHA256: `82b3eb8b96d5a98d4b43eb590da71b88b3e8ef31b4b3332596df8934f2ce9420`.
- Push-T preflight SHA256: `73510375fbeede12905f5b9e444963289ee22850d76c10ec68f06b05e41c9028`.

## Counts, with denominators

| Task | Released source rows | Legacy development rows / groups | Author validation rows / groups | Available validation clips | H6 prefix rollouts per arm, full pass |
|---|---:|---:|---:|---:|---:|
| Reach | 300 | 28 / 28 | 33 / 33 | 330 | 3,960 |
| Reach-Wall | 300 | 25 / 24 | 27 / 27 | 270 | 3,240 |
| Push-T, train + val | 18,706 | 3,636 / 36 | 21 / 21 | 1,695 | 3,390 |
| Three-task total | 19,306 | 3,689 / 88 | 81 / 81 | 2,295 | 10,590 |

The pooled group totals are bookkeeping only, not a single task's statistical n.
MetaWorld uses the seed-234 torch permutation over all 12,600 rows BEFORE task
filtering; validation is the final 1,260 indices. Push-T uses its separate 21-row
val pool, not a new 10% split of its 18,685 training rollouts.

At the all-42-task baseline scope, the original corrected baseline measured 1,155
MetaWorld rows plus 3,636 Push-T rows = 4,791 unique source rows. The corresponding
full author-validation pool is 1,260 + 21 = 1,281 rows out of 31,306 released rows
(4.09%). The 1,260 MetaWorld validation rows contain 1,257 numerical lineage groups.

For the primary intervention scope, all 3,689 previously evaluated rows were measured
under the custom protocol: 100% of those old evaluations are non-transferable AS
author-matched evidence. They remain usable in their original scope. This does NOT
mean rerunning all 3,689 on the old cohort is the correction. The replacement full
official validation cohort is 81 different/partly overlapping source rows per arm:
0.42% of the 19,306 available primary-task rows, and 100% of the official validation
pool for those three tasks. Neither percentage is a GPU-time estimate.

There are five fixed sweeps (coupling, geometry, rank, layer, spatial) on three tasks:
15 task-by-sweep replacements. Their unique source coverage remains 81, not 405;
405 counts repeated trajectory-by-sweep passes before the arm/control multiplier.
Fit passes, diagnostic anchor evaluations, broad baselines and later closed-loop work
are additional work, not extra independent validation trajectories.

## Statistical interpretation

There is no identifiable single percentage of power lost from changing protocols.
The current corrected analysis already uses lineage-level uncertainty: 28 Reach,
24 Reach-Wall and 36 Push-T groups. The actual power depends on the specified effect,
variance of paired differences, dependence structure and multiplicity adjustment.
Repeated windows/replays can improve within-family estimation without supplying
the between-family diversity of equally many independent trajectories.

For scale only, the normal approximation for one two-sided paired comparison with
alpha=.05 and 80% power has standardized minimum detectable effect
`(1.959964 + 0.841621) / sqrt(n)`, in SDs of the family-level paired differences:

| Task | Legacy n / approximate MDE | Author-pool n / approximate MDE |
|---|---:|---:|
| Reach | 28 / 0.53 | 33 / 0.49 |
| Reach-Wall | 24 / 0.57 | 27 / 0.54 |
| Push-T | 36 / 0.47 | 21 / 0.61 |

These are illustrative, unadjusted sensitivity calculations, NOT observed power or
guaranteed power for our bootstrap analysis. Small n and multiple comparisons require
more cautious calculations. Equal family variance would make the old Reach/Wall
standard errors about 9%/6% larger than with the author-pool counts; Push-T actually
has more groups in the custom development pool. Matching the paper does not therefore
automatically increase power. No new protocol's variance has yet been measured.

## Refit and exposure gates

The audited coupling and geometry fits each overlap official validation in 17 Reach
and 11 Reach-Wall groups (out of 128 fit groups per task). They cannot be reused for
author-validation comparisons. The audited Push-T fits have zero such overlap, but
their context/preprocessing dependence must still be checked; do not reuse automatically.

Seven primary MetaWorld official-validation rows currently carry protected status:
Reach indices 10624, 10653, 10774, 10782; Reach-Wall indices 10201, 10350, 10476.
No outcomes were read for this metadata audit and no protections were changed.
Full matching needs an authorized, frozen one-shot evaluation; otherwise report
incomplete official coverage rather than quietly excluding these rows.

The official MetaWorld row split itself has 36 numerical lineage groups spanning
train and val, including one in each primary task. Preserve the published row pool
for replication, but exclude all its lineage groups from NEW intervention fitting
and report duplicate-aware uncertainty. This does not prove which exact rows were
seen during training of the released checkpoint.

Remaining implementation gates: official preprocessing, three-position offline
context, all within-clip prefixes, appropriate embedding metrics, separate decoded
state metrics if available, fit receipts, frozen analysis and exposure authorization.
Batch size four is a per-rank compute setting, not the statistical sample size. A
released final checkpoint cannot reconstruct the authors' multi-seed training history.
