# GPU compute playbook (vast.ai, JEPA-WM planner runs)

The one page to read before renting anything. Cross-project ownership rules are in
`~/Documents/GPU_RESOURCE_BOARD.md`; the roofline reasoning is in
[`jax_scaling_notes.md`](../jax_scaling_notes.md); per-campaign logs are
`docs/GPU_RUN_PLAN_<date>.md`. Legacy baseline-harness tuning is in
[`GPU_EFFICIENCY.md`](GPU_EFFICIENCY.md).

## 1. What the workload is

Frozen planner (CEM, 300 candidates, FP32, no TF32) stepping a MuJoCo/Push-T
simulator one episode at a time, batch 1, one process per GPU. It is memory-bandwidth
and latency bound; nothing in the protocol may be batched or sharded, so **the only
lever is more independent units in parallel**. Buy bandwidth per dollar, not FLOPs.

Measured seconds per episode (engineering episode, 2026-09-12):

| Card | MetaWorld reach/reach-wall | Refined tasks (Push-T/PointMaze/Wall) | Typical $/GPU-hr seen |
|---|---:|---:|---:|
| RTX 5090 | 242–258 | — | 0.59–0.93 |
| RTX PRO 6000 (WS) | 249–251 (= 5090) | ~100 | 2.22 (avoid) |
| RTX 4090 | 296–298 | — | 0.40–0.70 (2.00–3.33 in a dry market: avoid) |
| RTX 6000 Ada | ~390 | — | 0.63–0.73 |
| L40S | 418–423 | — | 0.80 |
| RTX PRO 5000 | 370–374 | — | 0.87–1.09 |
| RTX 5000 Ada | ~540 | — | 0.33–0.36 |
| A100 SXM4 / A800 | fails MetaWorld parity | 85–130 | 0.64–1.04; 8× box 7.26 |
| H100 PCIe | fails outcome parity | — | 2.47 (excluded) |

Unit sizes: MetaWorld unit = (task, arm, slot) = 12 episodes ≈ 48 min on a 5090.
Refined shard = 3 arms × 12 episodes ≈ 1 h on an A100. Engineering (device-bound,
once per task per GPU) = 25 min on a 5090, up to 55 min on slower cards.

## 2. Device parity (measured, not assumed)

Every GPU runs the protocol's six engineering checks on the excluded smoke scenario
before any scientific episode. Compare `action_trace_sha256` and outcomes across devices:

- Ada/Blackwell (5090, 4090, PRO 6000, PRO 5000, 5000 Ada, 6000 Ada, L40S): identical
  success/steps/stimulus hashes on all six checks; reward/distance agree to ≤ 8e-8 relative.
  Two planner-internal trace groups exist (PRO 5000/4090/L40S vs 5090/PRO 6000/5000 Ada);
  outcomes are identical. Safe to mix within a slot under the disclosed amended gate.
- Ampere (A100/A800): fails the stimulus-hash gate (`initial_sha256`). Refined panels only.
- Hopper (H100): passes the stimulus gate, flips success on two arms. Excluded.

Receipts: `engineering/<device-uuid>/<task>/*/report.json` on every host (collected with
results); `h100-parity-audit.tgz` on Drive. Write-up line: "All paired arms ran on GPUs
verified to reproduce a reference rollout; Ampere and Hopper were excluded by that check."

## 3. Renting

1. Search wide, by $/GPU-hr, before renting:
   ```bash
   vastai search offers 'gpu_name in [RTX_5090,RTX_4090,RTX_4080S,RTX_4080,RTX_5080,RTX_PRO_5000,RTX_PRO_6000,RTX_6000Ada,RTX_5000Ada,L40S] verified=true reliability>0.97 rentable=true disk_space>60 cuda_vers>=12.4' -o dph --raw
   ```
   then sort by `dph_total/num_gpus`, drop CN/HK, prefer US, prefer multi-GPU boxes.
2. Cap: ≤ ~1.5× the current 5090 rate for any card not faster than a 5090.
3. **One 8-GPU box beats eight singles**: one bootstrap, one checkpoint download, one host
   to fail, one collection. Use `scripts/vast/refined_multi_setup.sh` (per-GPU roots,
   `PUSHT=`/`WALL=`/`MW=` layouts, staggered driver launches).
4. Unverified/deverified hosts fail to attach GPUs ("unresolvable CDI devices"): don't.
5. Dead-provision signature: `intended=stopped` with empty status at 4 min, or no port at
   15 min. Healthy boxes flip to `running` within a minute. Replace, don't wait.
6. Image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, `--disk 80`, `--ssh --direct`.
   Bootstrap is the **component** bootstrap (`ops/component_rental_bootstrap.sh`), not
   `scripts/vast/bootstrap.sh` (that one demands FFmpeg 7 and exits).

## 4. Running

- One host per shard/unit. Whole 12-episode streams only; a partial dir is wiped and rerun.
- MetaWorld: `scripts/vast/mw_slot_driver_v2.sh task:slot[:arm+arm]`; move a unit between
  hosts by touching `results/<task>/<arm>/shard-NN.MOVED` on the old host *before* the new
  host starts it. A driver fixes its arm list when a slot starts; to add an arm later,
  launch a second driver (it waits for the running arm).
- Multi-GPU box: stagger driver launches by ~90 s (first-time torch.hub DINOv2 loads race
  on the shared cache).
- Remote one-liners: never `pkill -f <pattern>` inside an `ssh "…"` command — it matches
  the ssh shell itself. Kill by PID from `ps -eo pid,args | awk`.
- Old coordinators (`component_extension_queue.py`) kill their child on SIGTERM; stop them
  only between units.
- Quiet monitoring: `fleet_monitor.py` pattern (probe every 5 min; emit only FAILURE /
  DRIVER_DEAD / HOST_FINISHED; baseline known errors so restarts stay quiet).
- Times: boxes log UTC; report to the user in Eastern.

## 5. Collecting and releasing

1. Verify on the box: `DONE.json` present, `report_sha256` matches `report.json`, no `FAILED.json`.
2. Tar results **and** `engineering/` (device receipts are needed by the analysis) →
   laptop → `rclone copy` to Drive folder `14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48` →
   `rclone check --download`. Strip macOS `._*` files before hashing anything.
3. Release immediately after the Drive check. Never destroy a box with un-collected data
   without the user's explicit go-ahead.
4. Salvage rule: results without their device's engineering directory cannot pass the
   frozen analysis. Always collect `engineering/` with the first collection from a host.
5. Final validation before destroying legacy boxes: hash every on-box `report.json`
   against the assembled panel or the Drive archives (`closeout/compare_table_boxes.py`).

## 6. Cost accounting

```bash
vastai show invoices --raw   # per-instance charge lines: amount, quantity (GPU-h), rate
```
Attribute by instance since the campaign start; the dashboard shows UTC-day slices.
2026-09-12 night: $212 total, ~30% useful compute; the rest was premium cards, idle
holds, per-device engineering on 14 spread GPUs, and rework. Rules 2–5 above are the fix.
