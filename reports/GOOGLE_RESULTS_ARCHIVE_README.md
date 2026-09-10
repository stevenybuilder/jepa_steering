# JEPA-WM result snapshot — September 7, 2026, 8:08 PM Eastern

This is a recovery snapshot, not a claim that the six-task study is complete.

**September 10 recovery update:** the snapshot below is historical. Use the
[final storage-release guide](VAST_FINAL_STORAGE_RELEASE_20260910.md) for later
results, exact archive IDs and verified rental deletions. This archive was
freshly streamed and all 2,244 files reverified during that closeout audit.

- `primary-durable-results.tar.gz`: all2,244 files in the laptop's
  `artifacts/offline_study/primary-durable-20260907` bundle at snapshot time.
  Original contents:2,314,736,088 bytes. Compressed archive:928,552,960 bytes.
- `receipt/FILES.json`: per-file original size and SHA256.
- `receipt/PLAN.json`: original location and explicit local offload targets.
- `receipt/VERIFIED.json`: completed two-destination readback checks.
- `receipt/DONE.json`: exact50 local duplicates removed,858,236,054 bytes freed.
- `summaries/`: readable methodological/status/result documents, separately dated.

Archive SHA256: `8fac5678983ec69883ecafdd29ac7ae4df507c58c259e94259f3e2f7952ce44e`.
The entire Drive download was checked, including every archive member's hash and
size; the independent GCS download matched the same full archive SHA256 and size.
Your Drive account owns the folder and no public sharing was enabled.

Independent copy:
`gs://rgt-jepa-archive-2026/rep_geometry_transcoder/20260908T000807Z/`.

To restore, download the archive, check its SHA256, and extract into a new
directory. Compare files to `receipt/FILES.json`. Do not overwrite newer experiment
output. Internal paths are relative to the original bundle root.

This snapshot does not contain everything on every worker: earlier remote-only
fit captures/bulk metrics, newly generated training checkpoints, and results
completed after this snapshot need separate archival batches. Original US-worker
copies remain untouched. Other laptop artifact roots are not included. Partial or
failed traces retain their labels and are not completed scientific measurements.
