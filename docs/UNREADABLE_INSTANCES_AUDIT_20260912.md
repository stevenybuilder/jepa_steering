# Provenance audit — five vast instances whose disks cannot be read (2026-09-12, read-only)

Question: does any result the project reports or depends on lack its raw data, receipts or logs because
it lives only on 50125440, 50205763, 50561030, 50588893 or 50592039?

**Answer: no.** Every product traced to these instances is present locally or on Drive with a matching
hash. One earlier-flagged dependency (seven Push-T window-metric files, 239.6 MB) is resolved: they are inside
the Utah closeout archives on Drive, hash-verified today. The only things that cannot be established are
negative facts about the dead disks themselves (whether stray files exist beyond the archived products).

## Checks performed today (files, not document claims, unless marked)

| Check | Result |
|---|---|
| 6 raw `native_fit.pt` captures (8,758,014,190 B) removed locally on 2026-09-07 after two-instance verification on 50125440/50189244 | all 6 on Drive as `raw-000..005-*-native_fit.pt`, sha256 match the relocation records (Drive-side hashsum) |
| 42 bulk `window_metrics.json` (1,710,558,874 B), same removal | 35 on Drive as `raw-006..047`, sha256 match; the other 7 (Push-T author-replication: distribution_layer ×2, distribution_spatial ×2, operator_rank shard-000, vision_action_coupling ×2) are inside `instance-50159352-batch-001/002.tar.gz` on Drive — extracted today, sha256 match all 7 |
| 50125440 Sept-8 shutdown archive (2,008 files) | `Research-Archives/JEPA-WM/shutdown-20260908T004500Z/instance-50125440-results.tar.gz` on Drive, sha256 0f3f713a… matches VERIFIED.json; local copy `restored-behavioral-inputs-20260908-v1/instance-50125440-results.tar.gz` (49 DONE dirs) |
| 50125440 combined-development analyses (BF16 846672df…/FP32) and Sept-7 outputs | covered by `google-storage-relocation-20260908T000807Z` (2,244 files, two-copy readback verified: Drive + GCS) |
| 50125440 Wall native 96 + input-repair streams | local `wall-native-completion-20260908-v1` (wall native shard-all, 96 episodes, report present) and the results tarball above |
| Codex coverage audit (Sept 10) | 741/741 known artifacts covered; `known_result_loss_confirmed: false`; `not_yet_mapped: {}` |
| 50205763 navigation-geometry results (431 files, 959 MB raw) | `Research-Archives/JEPA-WM/navigation-geometry-closure-20260908/navigation-geometry-results.tar.gz.part-aa/ab` on Drive; sha256 of both parts match INDIANA_PRIOR_ARCHIVE_PARTS.json; Codex readback "full_compressed_and_every_member_verified" |
| 50205763 seed-235 training | never ran there: doc records the Indiana job stopped, seed 236 untrained, corrected histories to start afresh (POINTMAZE_TRAINING_HISTORY.md §Historical). Nothing to preserve |
| Navigation final analysis (1,728 records, Wall+PointMaze) | 132 bound input reports all present locally under `table-completion-20260911-v1/navigation-final-analysis-v1/` with matching sha256 |
| Push-T final analysis v2 (864 records, 9 arms) | 82 bound reports: 73 present locally with matching sha256; the other 9 (restored 2026-09-08 native streams) are in Drive `instance-50239185-batch-000.tar.gz` (8 hash-matched by content; the 9th is the same set — 8 distinct shards bound under 9 paths) ; full panel also in 50588893 snapshots |
| 50588893 (Texas) final state | last snapshot `jepa-table-50588893-snapshot-1789149396272680183.tar.gz` = FINAL_DRIVE_VERIFIED (drive id 1oe5NcJ…, 5,455-member manifest, full byte readback); worker state done=True, failed=False, 492 episodes / 41 streams, all five queues (pusht, pointmaze-a/b, pointmaze-tail-a/b, pusht-complete-audit-v2) done. Snapshot holds 24 PointMaze DONE dirs + CONTINUATION_DONE |
| 50592039 (Maryland) final state | last snapshot `…50592039-snapshot-1789143445033923898.tar.gz` = FINAL_DRIVE_VERIFIED (289-member manifest, full readback); state done=True, 168 episodes / 14 streams |
| 50561030 (Washington) Wall continuation | 120 assigned episodes completed (doc); worker archive `wall-table-resume-20260911-v1.tar.gz` (204 members, sha 56cfc6d2…) on Drive with full-byte readback receipt (`wall-closeout/DRIVE_VERIFIED.json`) |

## Per instance

**50125440 (steering-deadline-0800, 8×5090; host offline since 2026-09-10 05:32).**
Ran: Sept-7 combined-development shards (8 shards, both precisions), canonical rank/combined streams, Wall native 96,
four input-repair streams; held primary copies of the raw fit captures / window metrics. Preserved: all of the above
(see table). Unaccounted: only "whatever may have been written after the Sept-8 00:45 shutdown archive" — Codex's four
inventory attempts on Sept 10 all timed out (0 files listed), so a final disk listing never existed. Docs note one
Reach coupling repair was "incomplete after shutdown"; its repaired streams are in the results tarball. No reported
result depends on anything not archived.

**50205763 (navigation-offline-indiana, 1×4090; stuck "loading" since 2026-09-10).**
Ran: Wall/PointMaze offline native baselines, 20 category/precision fits, navigation geometry (4 scopes, 8/8 shards each),
non-rank navigation comparisons; reserved for seed-235 training that never started. Preserved: 431-file navigation-geometry
archive (two parts, hashes verified today); readiness/activation/test receipts under `navigation-redistribution-20260908-v*`.
Unaccounted: no final disk inventory (stopped-disk copy API rejected the path). Nothing reported depends on it.

**50561030 (table-completion-0911-v1, 2×4090 Washington; cannot restart).** Ran the ten-stream Wall continuation
(120 episodes). Preserved: 204-member worker archive, Drive-verified; those Wall records are among the 132 locally
hash-verified navigation-analysis inputs. Unaccounted: nothing known.

**50588893 (table-0911-pusht-0-v2, 4×4090 Texas; cannot restart).** Ran 360 Push-T episodes, the Push-T complete
audit v2 (the frozen 864-record analysis), PointMaze receiving checks and the missing PointMaze control streams.
Preserved: final snapshot with 5,455-member manifest, Drive full-byte readback; worker state done, no failures.
Unaccounted: nothing known (state file marks every queue done before the final snapshot).

**50592039 (table-0911-pusht-4-v3, Maryland; cannot restart).** Ran 168 Push-T episodes / 14 streams. Preserved:
final snapshot, 289-member manifest, Drive readback; state done. Unaccounted: nothing known.

## What remains unknowable
Only the absence of extra files on the two dead hosts. No listing exists past their last archives (50125440: Sept 8
00:45 shutdown archive plus the Sept-10 raw relay; 50205763: Sept 8 navigation-geometry closure). Codex's coverage
audit and today's hash checks find every known artifact preserved. Destroying the five instances forfeits nothing
that is currently recoverable by any route other than Vast support restoring host access.

Notes: Drive hashes were read with `rclone hashsum sha256` (Drive-side); archive members were extracted by streaming
and hashed locally. No vast, ssh, or project files were touched; this report is the only file written.
