# Retained Vast storage audit — September 8, 13:11 UTC

Read-only provider/board/existing-receipt audit. No stop, restart, destruction or
fresh Drive readback of these older archives was performed in this audit.
Eight stopped/created disks total **$0.737037/hour (~$17.69/day)**. Their advertised
`dph_total` includes inactive compute; use `storage_total_cost` for this breakdown.

| Instance | Storage/hour | What is preserved or recorded | Remaining release check |
|---|---:|---|---|
| Shanghai50135088 | $0.027778 | Ledger: no scientific jobs, partial public-input staging only; US-only exclusion | Explicit final relevant-content inventory/release receipt |
| Shanghai50135089 | $0.027778 | Same | Same |
| Shanghai50135090 | $0.027778 | Same | Same |
| US50125440 | $0.180556 | 2,008 files verified in selected-root shutdown archive | Older raw-fit/metric dependency below; selected roots do not prove entire relevant-disk coverage |
| Utah50159352 | $0.022222 | All85 referenced artifacts independently rehashed locally in this audit | Seven metric files absent from initial Google manifest; local copies remain, complete coverage reconciliation needed |
| Virginia50189244 | $0.400000 | All50 Wall checkpoints covered by38+12-epoch Drive snapshots; DROID fit/behavior archives | Explicit disk-retention reservation and older raw-fit/metric dependency below |
| OldUS50195621 | $0.041667 | 1,524 files in verified shutdown archive, including Wall235/236 epochs1–2 | Reconcile archived roots with all relevant contents and release retention |
| Unused50229002 | $0.009259 | Ledger: created bootstrap with no research code, input or output staged | Final owner inventory/release record; no destruction claimed |

## Important remaining storage dependency

Six raw native-fit captures (**8,758,014,190 bytes**) and42 bulk window metrics
(**1,710,558,874 bytes**) were removed locally after two-host verification on
50125440 and50189244. Reviewed selected-root Google manifests do not establish
independent durable coverage of those exact objects. Do **not** destroy both disks
based only on the more recent, narrower live snapshots. Copy/reconcile those
objects with durable storage before releasing their remaining preservation role.

Evidence directories under `artifacts/offline_study/`:

- `fit-capture-storage-relocation-20260907-v1/`
- `window-metrics-storage-relocation-20260907-v1/`
- `shutdown-20260908T004500Z/instance-50125440/`
- `shutdown-20260908T004500Z/instance-50195621/`
- `utah-durable-20260907/verification/`
- `live-20260908T063100Z/instance-50189244-retry-v2/`
- `live-20260908T091400Z/instance-50189244-incremental/`
- Virginia DROID snapshots: `live-20260908T100000Z/`, `105500Z/`, `111200Z/`

California50245262 is a different, externally restarted/quarantined resource and
is deliberately excluded from the stopped-disk table. At this audit it was running;
the complete provider rate was$6.962593/hour, before usage bandwidth. This audit
does not authorize touching that quarantined instance.
