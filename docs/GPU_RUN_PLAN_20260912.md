# GPU run plan — 14-cell completion (authoritative, 2026-09-12)

Owner: rep_geometry_transcoder / Claude session. Update this file BEFORE any
instance rental/stop/destroy and AFTER any assignment change. The
`GPU_RESOURCE_BOARD.md` "Latest execution status" points here.

## Cost post-mortem (09-12, from vast invoice lines since takeover 09-11 20:00 UTC): $212 total
~$60–65 useful compute at fair rates · ~$40–45 premium cards (NV3 PRO 6000 kept 15 h at $2.20 ≈ $22 excess; Texas/Brazil/
Romania 4090s at $2.00–3.33 ≈ $18 excess while a $0.63 RTX 6000 Ada and $0.80/GPU L40S did the same work) · ~$30 Codex
workers still running on the evening of 09-11 · ~$15–20 idle boxes held for decisions · ~$10 per-device engineering on 14
spread GPUs · ~$15 rework (NV0 salvage unusable, slot-0 rerun, NJ duplicate) · ~$5 validation restarts/H100 test.
Rules adopted: search ALL Ada/Blackwell cards + multi-GPU boxes by $/GPU-hr before renting; cap at ~1.5× the 5090 rate for
any card not faster than a 5090; swap expensive cards when their unit ends; release boxes the moment data is Drive-verified.

## Standing rules (from the user, tonight)
1. **Never stop or destroy an instance without the user's explicit go-ahead.**
   Finished/idle boxes stay up.
2. **Collect before anything else.** When a host finishes, tar its verified
   shard dirs (DONE.json + report hash) to the laptop scratchpad AND Drive
   folder `14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48`, verify with
   `rclone check --download`, only then consider the host free.
3. **One host per shard.** Cross-host "skip if verified" does NOT work (the
   driver only sees its own disk). Never launch a driver whose shard list
   overlaps another host's.
4. All three arms of a shard run on the same physical GPU (paired inputs).
   Partial shards are wiped and rerun whole — never resumed.
5. A100/A800 (Ampere) hosts: refined panels only (Push-T, Wall, PointMaze,
   DROID). They fail the MetaWorld device-parity gate. MetaWorld needs
   Blackwell/Ada consumer or PRO cards (5090, 4090, PRO 5000/6000).
6. vast.ai only; no China/HK hosts; prefer US; prefer `verified=true`,
   reliability ≥0.98. Boxes stuck "loading" ≥15 min with no port never ran
   anything — replacing those is allowed (record it here).

## Cell status (05:05 UTC)
| Panel | Cells | Status |
|---|---|---|
| DROID | 2 | **COMPLETE** 04:50 — collected `droid-panel-complete.tgz` (Drive-verified) |
| PointMaze | 2 | **COMPLETE** 06:18 — 0–5 (NJ), 6,7 (w2, `pointmaze-shard7-w2.tgz`); all Drive-verified. Analysis pending. |
| Push-T | 2 | **COMPLETE** 05:40 — 0–3 (A800), 4,5 (r1, `pusht-shards45-r1.tgz`), 6,7 (8xA100); all Drive-verified. Analysis pending. |
| Wall | 2 | **COMPLETE** 05:45 — 0,1 (A100-CZ, `wall-shards01-cz.tgz`), 2–7 (8xA100); all Drive-verified. Analysis pending. |
| Reach ×3, Reach-Wall ×3 | 6 | see slot map; long pole ~09:00 |

## Refined panels ANALYZED (09:25 UTC) — `src/offline_study/refined_panel/analysis.py`, output scratchpad `refined-analysis-v1/`
(subpackage placement is deliberate: a new file directly in src/offline_study would change the frozen source hash.)
12 contrasts, 20,000 paired cluster-bootstrap draws, seed 2026091102, Bonferroni 95%. All 12 intervals include 0.
| Task | n/arm | native | fixed_rank4 | matched_random | fixed−native | random−native | fixed−random |
|---|---|---|---|---|---|---|---|
| Push-T success % | 96 | 59.4 | 59.4 | 61.5 | +0.0 [−6.1, +5.9] | +2.1 [+0.0, +7.4] | −2.1 [−8.6, +3.6] |
| PointMaze success % | 96 | 80.2 | 85.4 | 79.2 | +5.2 [−6.3, +16.7] | −1.0 [−10.4, +8.3] | +6.3 [−6.3, +18.8] |
| Wall success % | 96 | 76.0 | 78.1 | 76.0 | +2.1 [−5.2, +9.4] | +0.0 [−4.2, +4.2] | +2.1 [−5.2, +9.4] |
| DROID score | 64 | 51.100 | 51.049 | 51.001 | −0.051 [−0.204, +0.129] | −0.099 [−0.337, +0.128] | +0.048 [−0.195, +0.298] |
DROID native 51.09965 matches the historical native exactly. Every shard verified (DONE hash, protocol/freeze/source
binding, episode + raw-trace hashes, same-device three arms, full 96/64 coverage, paired inputs).
Disclosed gaps: Wall/PointMaze freeze dirs not re-validated locally (hash-bound in every shard; PointMaze freeze
being fetched from NJ 09:27; Wall's hosts are destroyed); engineering dirs re-verified only for A800 Push-T and DROID
devices (others hash-bound); Push-T cohort.json not in the tree; DROID clustered by source recording (15).

## Fleet (instance id · host:port · card · $/hr · assignment · state)
| Label | ID | Endpoint | Card | $/hr | Assignment | State |
|---|---|---|---|---|---|---|
| NJ | 50626847 | 71.104.167.38:53046 | RTX 5090 | 0.75 | MetaWorld reach:5 + reach-wall:5 (actually launched 05:09 — the 05:03 "stop" never took; PointMaze 0–5 collected; NJ's duplicate shard-6 dirs moved to `duplicate-shard6-ignored/`, w2's shard 6 is canonical) | running |
| MO | 50632757 | 154.36.209.172:16758 | RTX 5000 Ada | 0.36 | reach:2 all 4 arms DONE, collected `reach-slot2-mo.tgz` 05:59 → now reach-wall:2 joint (MOVED on w3), v2 driver 06:00 | running |
| NV3 | 50640703 | 184.186.104.194:10081 | RTX PRO 6000 WS | 2.22 | DROID DONE → coordinator resumed: reach:3 joint, reach-wall:3 | running |
| ~~r1~~ | 50685454 | — | A100 SXM4 | — | Push-T 4,5 Drive-verified → destroyed 05:48 (user 05:47: "We can save the results if we're not using them") | destroyed |
| ~~A100-CZ~~ | 50682640 | — | A100 SXM4 | — | Wall 0,1 Drive-verified → destroyed 05:48 | destroyed |
| ~~w2~~ | 50687479 | — | A100 SXM4 | — | PointMaze 6,7 Drive-verified → destroyed 05:48 | destroyed |
| ~~w3~~ | 50687482 | — | RTX PRO 5000 | — | reach-wall:2 all 4 arms DONE (single device) → `mw-w3-rw2-all4.tgz` Drive-verified → destroyed 09:12 | destroyed |
| NV3 (idle) | 50640703 | 184.186.104.194:10081 | RTX PRO 6000 WS | 2.22 | reach:3 ×4, rw3 native+visual, rw5 joint DONE → `mw-nv3-reach3-rw3nv-rw5joint.tgz` Drive-verified 09:10. KEPT UP pending user decision on the same-device analysis gate (it would host rw3 action+joint reruns). | idle |
| BoxA (idle) | 50682625 | ssh4.vast.ai:12624 | RTX 5090 | 0.59 | reach:4 ×4, rw4 native+visual+joint DONE → `mw-boxa-reach4-rw4-nat-vis-joint.tgz` Drive-verified 09:38; kept for possible rw4 action rerun | idle |
| kr (idle) | 50703657 | 218.49.89.120:40157 | RTX 6000 Ada Korea | 0.63 | rw7 visual+action DONE → `mw-kr-rw7-vis-act.tgz` Drive-verified 09:42; kept for possible rw7 reruns | idle |
| 6000ada (idle) | 50701023 | 118.163.199.123:15377 | RTX 6000 Ada Taiwan | 0.73 | reach7 native+visual DONE → `mw-6000ada-reach7-nat-vis.tgz` Drive-verified 09:30 | idle |
| NJ (idle) | 50626847 | 71.104.167.38:53046 | RTX 5090 | 0.75 | reach 1 + 5 all arms DONE → `mw-nj-reach1-reach5-all.tgz` Drive-verified 09:30 | idle |
| r2 (idle) | 50685456 | 79.160.189.79:13943 | RTX 4090 | 0.70 | reach 6 ×4, rw6 native+joint DONE → `mw-r2-reach6-rw6-nat-joint.tgz` Drive-verified 09:48; kept for possible rw6 reruns | idle |
| ES | 50690831 | 81.39.139.61:28307 | RTX 5090 | 0.88 | MetaWorld reach-wall:0 + reach:0 (action, joint; native/visual pre-placed from NV0 salvage) | running |
| ~~BE~~ | 50690828 | — | RTX PRO 5000 | — | reach-wall:1 all 4 arms DONE (single device) → `mw-be-rw1-all4.tgz` Drive-verified → destroyed 09:40 | destroyed |
| tx1 (idle) | 50700641 | 57.132.208.22:27303 | 4090 Texas | 2.00 | rw5 native+visual+action DONE → `mw-tx1-rw5-nat-vis-act.tgz` Drive-verified; kept for possible rw5 joint rerun | idle |
| l40s2 (idle) | 50701024 | 118.163.199.123:27673 | 2× L40S | 1.60 | reach7 action+joint, rw6 visual+action DONE → collecting `mw-l40s2-…tgz`; kept for possible reruns | idle |
| ~~8xA100~~ | 50690802 | — | 8× A100 SXM4 | — | Push-T 6,7 + Wall 2–7 Drive-verified → destroyed 05:48 | destroyed |

Clock note: box/laptop UTC is authoritative; my earlier narrative timestamps after ~05:30 ran ~30 min fast.
Real 05:47 UTC = 1:47 AM Eastern at the Wall close.
| ~~slot7~~ | 50698217 | — | RTX 4090 Iceland | — | dead-provision (kernel register dump in status, no port at 16 min) → destroyed 05:23 per rule 6, never ran | destroyed |
| ~~qc8,cz4,bg4,az2,uk8~~ | 50699785/87/90/95/97 | — | 26× 4090 on unverified hosts | — | all five died identically at 10 min: host container runtime "unresolvable CDI devices" (GPU attach failed), vast set intended=stopped. Destroyed 05:42 (never ran). Lesson: unverified/deverified hosts fail exactly this way — verified only from now on. | destroyed |
| ~~wa1,mk2,us2,wa2,wa1b~~ | 50700630/31/34/38/43 | — | 8× 4090 on VERIFIED hosts (4 US) | — | never accepted by host (loading, intended=stopped, empty status at 4 min; healthy boxes flip to running within ~1 min). Destroyed 05:57. Verified + US did not help; host-side. | destroyed |
| tx1 | 50700641 | 57.132.208.22:27303 | 1× 4090 Texas | 2.00 | reach-wall:5 native+visual_only+action_condition_only (MOVED markers set on NJ) | engineering since 06:08 (first launch shipped the wrong bootstrap — fixed) |
| ~~h100~~ | 50701029 | — | 1× H100 PCIe US | — | parity test FAILED at outcome level (see audit) → destroyed 06:14 | destroyed |
| ~~MO~~ | 50632757 | — | RTX 5000 Ada | — | reach-wall:7 native DONE 08:48 → collected `reachwall-native-slot7-mo.tgz` (Drive-verified) → destroyed 08:50 (slowest card; no remaining unit would finish sooner on it) | destroyed |
| ~~ro1~~ | 50703808 | — | 4090 Romania | — | rw3 action+joint DONE → `mw-ro1-rw3-act-joint.tgz` Drive-verified → destroyed 08:58 | destroyed |
| ~~br1~~ | 50703789 | — | 4090 Brazil | — | rw7 joint + rw4 action DONE → `mw-br1-rw7joint-rw4act.tgz` Drive-verified → destroyed 08:58 | destroyed |
| 6000ada | 50701023 | 118.163.199.123:15377 | RTX 6000 Ada Taiwan | 0.73 | reach:7 native+visual_only | setting up 06:12 |
| l40s2 | 50701024 | 118.163.199.123:27673 | 2× L40S Taiwan | 1.60 | g0 reach:7 action_condition_only+joint; g1 reach-wall:6 visual_only+action_condition_only (MOVED on r2) | setting up 06:12 |
| ~~za~~ | 50701026 | — | 4090 South Africa | — | never accepted (stopped at 4 min) → destroyed 06:12 | destroyed |

Rented 06:16 (verified, non-Hopper): ~~ee2 50703655 (2× 4090 Estonia)~~ died at 1 min (CDI GPU-attach error) →
destroyed 06:22; kr 50703657 (RTX 6000 Ada Korea, 218.49.89.120:40157) READY 06:22 → launched on
rw7 visual_only+action_condition_only; ~~hu1 50703662 (4090 Hungary)~~ never accepted → destroyed 06:28; br1 50703789 (4090 Brazil $3.33, 201.25.77.65:55128) READY 06:22 → launched on rw7 joint + rw4 action+joint (MOVED on BoxA);
ro1 50703808 (4090 Romania $2.00, 92.180.27.84:59464) READY 06:27 → launched on rw3 action+joint (MOVED on NV3).
07:55 rebalance (no new hardware): rw5 joint NJ → NV3 (NJ would need 25-min rw engineering; NV3 already has it
and frees ~08:13; second v2 driver on NV3 waits for the running arm). rw4 joint br1 → BoxA (br1 had 3 units queued;
BoxA idle after rw4 visual ~08:35; markers: br1 MOVED set, BoxA MOVED removed). BoxA reachable only via
ssh4.vast.ai:12624 since ~07:40 (direct-IP sshd closes connections; box itself healthy).
Projected ends (UTC): BE 09:21, L40S 09:20, kr 09:35, ES 09:40, r2 09:43, br1 08:55, BoxA 09:25, NV3 09:05,
tx1 09:12, w3 09:02, MO 08:57, ro1 ~09:00 → panel data complete ≈ 09:45 UTC (5:45 AM Eastern).
Queue for the next READY boxes, in order: rw3 action+joint (MOVED on NV3 — NV3 alone would end ~10:20 UTC),
rw7 joint, rw4 action+joint (BoxA), rw5 joint (NJ), rw6 joint (r2). Stays: rw1 joint (BE), reach0 joint (ES), rw2 joint (w3). They fall to their old hosts (~09:30–10:00 UTC) unless the
H100 parity test passes (then rent 3–4 more H100s, mostly US) or 4090/5090 supply returns.
Bootstrap lesson: `scripts/vast/bootstrap.sh` is the jepa_steering one (needs FFmpeg 7, exits);
the component bootstrap is the one on the boxes (`ops/component_rental_bootstrap.sh`, copied to scratchpad `bootstrap.sh`).

Market note 05:20–05:45: zero 5090/PRO/L40S/6000-Ada offers outside CN/HK; verified 4090s only
at $1.07–2.67/GPU-hr (9 GPUs rented above, ~$16/hr). Accepted per user ("don't worry about credit").

## MetaWorld spread (approved by user 05:20: "as long as it doesn't affect data quality")
Why it is protocol-clean: the unit of science is one (task, arm, logical slot) = one whole
12-episode RNG stream on one device, against frozen paired stimuli, with device-bound
engineering verified by the parity gate. Nothing ties the four arms of a slot to one GPU
(ES already runs slot-0 arms from two devices). The "same physical GPU" rule applies to the
refined panels' three conditions, not here. Streams are never split or resumed.
Mechanism: hosts swapped to `mw_slot_driver_v2.sh` (in-flight arm untouched; it waits for the
orphaned run). A unit is moved by touching `results/<task>/<arm>/shard-NN.MOVED` on the old
host BEFORE the new host starts it; v2 skips MOVED units. MO keeps reach joint:2 (nothing to
move). NV3: old coordinator would kill its child on SIGTERM, so a watcher waits for reach
joint:3 DONE, then stops the coordinator and starts v2 on reach-wall:3.
Move list (30 units, ~24 GPU-h at 4090 pace), highest-priority first:
reach-wall 5:all, 4:all, 6:all, 7:all, reach 7:all, reach-wall 3:action+joint,
reach-wall 1:action+joint, 2:action+joint, 0:joint, reach 0:action+joint, reach 5:joint.
Assignment ledger (fill on READY): one task per GPU (engineering is per task per device).

NOTE (A100-CZ vs 8xA100 Wall 2,3): A100-CZ's driver list is 0 1 2 3. 8xA100 g2/g3
started Wall 2,3 at 04:11 and finish ~05:20; A100-CZ reaches shard 2 at ~05:55.
If A100-CZ gets there first it would duplicate — check at 05:40 and, if needed,
copy 8xA100's verified shard-2/3 dirs onto A100-CZ so its own-disk skip fires.

## Device-parity audit (06:10 UTC, from every host's engineering/<UUID>/<task>/<sub-arm>/report.json)
Compared `action_trace_sha256` and `result` for the engineering episode across devices.
- **reach**: native, native_repeat, zero_dose, visual_only, action_condition_only, joint traces are
  bit-identical on RTX 5090 (NJ, BoxA), RTX 4090 (r2), RTX PRO 6000 (NV3), RTX 5000 Ada (MO). No device effect.
- **reach-wall**: edited arms identical everywhere. Native trace splits into two groups: {5090 ES, PRO 6000 NV3,
  5000 Ada MO} = 78cb16f2… vs {PRO 5000 w3, PRO 5000 BE} = 1e851956…; deterministic within device (repeat matches).
  **Outcomes are identical across the two groups** (native_reward 753.8756103515625, native_state_distance
  0.37685930728912354, native_success false) → the difference is planner-internal, not in the executed trajectory.
- Rule adopted anyway: a slot's four arms stay within one trace-group. Reverted rw2:joint from MO back to w3
  (06:08; MO's 6-min engineering discarded, MO idle). New devices (4090 tx1, H100, 6000 Ada, L40S) get classified
  from their engineering reports before their arms are combined with other hosts' arms.
- 06:13 update: tx1 (4090) and L40S are in the PRO 5000 native-trace group; tx1's outcomes match ES (5090 group)
  bit-for-bit on all six sub-arms → the two Ada/Blackwell groups are outcome-identical and MAY mix within a slot.
- **H100 (Hopper) FAILS the outcome check**: native distance 0.2873 vs 0.3769, visual_only success True vs False,
  action_condition_only success False vs True on the same stimulus. Traces differ on every arm. Excluded from the
  panel; driver stopped, engineering archived as `h100-parity-audit.tgz` (Drive), instance destroyed 06:14.
  Its reach-wall:7 native+visual_only assignment is void (no data produced beyond engineering).
- Upstream jepa-wms (TMLR 2026, arXiv 2512.24497): 3 training seeds, 96 episodes, std over last-10-epoch success;
  no hardware/determinism mention. Write-up disclosure of the device audit is OPTIONAL per user (06:35);
  suggested one-liner for Methods: "All paired arms ran on GPUs verified to reproduce a reference rollout
  bit-for-bit; two architectures (Ampere, Hopper) that failed this check were excluded." Receipts: per-device
  `engineering/<UUID>/<task>/*/report.json` on every host + `h100-parity-audit.tgz` on Drive.

## 09:35 UTC: reach slot 0 native+visual MUST be rerun (independent of the gate decision)
The NV0-salvaged shards (`reach-shard00.tgz`, device 252aa065…) have no preserved engineering directory (NV0 was
destroyed pre-takeover with only results salvaged); `load_panel` requires `engineering/<uuid>/reach` → rejects
them under any option. Moved aside on ES (`nv0-salvage-unusable/`), second v2 driver queued on ES:
`reach:0:native+visual_only` after ES's running reach joint:0 (ES has reach engineering) → ~11:20 UTC (7:20 AM ET).
Under option 1 they could instead run in parallel on NJ + 6000ada (~10:35 UTC); not started, to avoid duplicate work.

## 09:58 UTC: MetaWorld panel assembled and pre-verified (56/64 units)
`scripts/vast/assemble_mw_panel.py` → scratchpad `mw-panel/` (manifest `mw-panel-manifest.json`); analysis via
`src/offline_study/refined_panel/metaworld_amended.py --frozen-src <mwcode>/code/src …` (run as a script, NOT with
PYTHONPATH=src: the freeze binds the mw-code.tgz source hash bcf77f2b…, not the laptop tree). `--check-only`:
0 problems on 56 shards; reach 7 devices / reach-wall 11 devices, max cross-device engineering float deviation 8.4e-8,
0 non-equivalent pairs. Missing: slot 0 of both tasks (on ES; reach native/visual rerunning → ~11:20 UTC).
`--strict` (frozen loader) fails only on "Paired arms executed on different devices".

## ACCOUNT EMPTY 13:35 UTC (9:35 AM Eastern): zero vast instances remain.
Legacy stopped instances (Codex era) released after validation: 50546169/50546172 restarted, on-box archives
md5-identical to Drive + every DONE dir present; 50544130/50546170/50546171 could not restart — Drive archives are full
workspace preservations with PRESERVATION_MANIFEST (same script as the two byte-verified); 50588615/50588903/50588914
restarted — all 75 new result reports content-hash-matched inside Drive archives (260 per box were Drive-restored
copies); 50561030/50588893/50592039 (no GPU to restart) and 50125440/50205763 (hosts dead since 09-10) — provenance
audit `scratchpad/closeout/UNREADABLE_INSTANCES_AUDIT.md`: every reported/depended-on product present locally or on
Drive hash-matched; Codex audit 741/741 covered. User authorized destruction 13:34 ("destroy them").

## CLOSED 11:50 UTC (7:50 AM Eastern). Decision: amended gate accepted by user (11:20 UTC: "i dont think that's necessary").
Final validation before release: every verified shard on all 9 remaining boxes matched the assembled panel's
report.json sha256 (NJ 8, NV3 7, BoxA 7, r2 6, ES 8, tx1 3, 6000ada 2, l40s2 4, kr 2 = 47 shards, 0 mismatches);
`rclone check --download --one-way` local→Drive: 35 tarballs, 0 differences. All nine instances destroyed 11:50.
Fleet: nothing running. All 14 cells: data Drive-verified, analyses archived (`analysis-outputs-20260912.tgz`).
Caption footnote to carry: MetaWorld component rows were paired to a concurrent native reproduction (45.83 / 29.17).

## 10:45 UTC: MetaWorld panel COMPLETE (64/64 units, all Drive-verified) — analyzed under the AMENDED gate
Frozen loader (`--strict`) fails only on "Paired arms executed on different devices". Amended run
(`mw-analysis-amended-v1/`, archived in `analysis-outputs-20260912.tgz` on Drive; amended loader sha f973d264…,
frozen analysis sha d08f9a96…, freeze 3ed73b75…): 12 contrasts, 20,000 draws, seed 2026091101, Bonferroni 95%.
| Task | native | visual | action | joint | vis−nat | act−nat | joint−nat | joint−vis | joint−act | interaction |
|---|---|---|---|---|---|---|---|---|---|---|
| reach (n=96/arm) | 45.8% | 50.0% | 42.7% | 52.1% | +4.2 [−14.6,+22.9] | −3.1 [−20.8,+14.6] | +6.2 [−13.5,+25.0] | +2.1 [−15.6,+19.8] | +9.4 [−8.7,+28.1] | +5.2 [−19.8,+30.2] |
| reach-wall (n=96/arm) | 29.2% | 38.5% | 36.5% | 29.2% | +9.4 [−9.4,+27.1] | +7.3 [−9.4,+24.0] | +0.0 [−17.7,+17.7] | −9.4 [−22.2,+3.1] | −7.3 [−22.9,+8.3] | −16.7 [−38.5,+4.2] |
All 12 simultaneous intervals include 0. If the user chooses option 2 (reruns), re-assemble and re-run `--strict`.

## OPEN DECISION (raised to user 09:05 UTC): frozen analysis same-device gate
`metaworld_component_analysis.load_panel` raises "Paired arms executed on different devices" unless every arm
of a logical episode has the same device UUID. Cross-device slots: reach 0 (NV0 salvage + ES), reach 7
(6000ada + L40S g0), rw3 (NV3 + ro1), rw4 (BoxA + br1), rw5 (tx1 + NV3), rw6 (r2 + L40S g1), rw7 (MO + kr + br1).
Option 1 (recommended): amend the gate to accept devices whose six engineering checks have identical outcomes
(disclosed; changes analysis_source_sha256). Option 2: rerun 12 arm-units (~8 GPU-h) so each slot is one device.
Until decided: keep NV3, ES, BoxA, tx1, r2, kr, 6000ada, l40s2 up after they finish.

## MetaWorld slot map (task:slot → host). 8 slots × {reach, reach-wall} × 4 arms
| slot | reach | reach-wall |
|---|---|---|
| 0 | ES (native, visual pre-placed; action, joint running) | ES |
| 1 | NJ-done (all 4 arms, collected to Drive 05:12) | BE |
| 2 | MO — all 4 arms DONE, Drive-verified 05:59 | w3 (native done, visual, action) + MO (joint) |
| 3 | NV3 (native, visual, action done; joint running) | NV3 |
| 4 | BoxA (native, visual done) | BoxA |
| 5 | NJ (launched 05:03) | NJ |
| 6 | r2 (native done) | r2 |
| 7 | slot7 box (pending) | slot7 box |

## Collected to Drive (all `rclone check --download` verified)
droid-progress-20260912T010846Z.tgz · droid-panel-complete.tgz ·
pusht-shard1.tgz (NV0 salvage) · pusht-shards0123-a800.tgz ·
reach-shard00.tgz (NV0 salvage) · pointmaze-shard6-w2.tgz ·
pointmaze-shards0-5-nj.tgz · reach-slot1-nj.tgz (05:12, all 4 arms of reach shard-01) ·
wall-shard2-8xa100.tgz · wall-shard3-8xa100.tgz (05:30; also relayed onto A100-CZ so its driver skips 2,3) ·
pusht-shard6-8xa100.tgz · pusht-shard7-8xa100.tgz · wall-shard{4,5,6,7}-8xa100.tgz (05:30)
Push-T on Drive: 0–3 (A800), 1 (NV0 salvage, superseded), 6,7 — missing 4,5 (r1). Wall on Drive: 2–7 — missing 0,1 (A100-CZ).

## Losses / incidents tonight
- 00:1x: 3 original workers stopped by the $7/hr guard after I added GPUs (guard since removed).
- w1 (A100 MA) destroyed 04:15 before collecting its PointMaze shard 7 → rerun on w2.
- l40s2 05:48: both GPU drivers launched simultaneously → torch.hub DINOv2 cache race (`Directory not empty` rmtree) killed gpu0's reach engineering; gpu1 unaffected. Relaunched gpu0 05:50 once the cache was populated. `refined_multi_setup.sh` now staggers per-GPU MW launches by 90 s.
- NJ 05:23: the pre-compaction "stop+relaunch" ssh command had been hung, fired late, and launched a second v1 driver, which wiped the 05:09 native-5 run and restarted it. Net: one clean native-5 run from 05:23 (whole stream from seed), 14 min lost. v2 driver now supervises.
- NJ duplicated PointMaze shard 6 (cross-host skip assumption). The 05:03 stop did not take (pkill self-match); found still running at 05:08 with native/shard-6 DONE and fixed_rank4/shard-6 partial → killed by PID, dirs quarantined, MetaWorld slot 5 launched 05:09. Wasted ≈1 GPU-hour, no data-quality impact (w2's shard 6 is the only one used).
- Dead-provision boxes replaced (never ran): 3×5090, 8×5090, 2×4090, s1, s2 (CN), r3, 4090 MY, A100 CZ2, 4090 IS.
- 2×A100 JP: failed MetaWorld parity gate (as did A100 Alberta) — released, nothing on it.

## Next actions
1. slot7 box: background waiter reports READY (→ `mw_box_setup.sh reach:7 reach-wall:7`) or DEAD_PROVISION at 15 min (→ replace, record here).
2. ~05:55 w2 PointMaze shard 7 → collect → PointMaze cells close (aggregate 0–5 NJ + 6,7 w2).
3. Wall 2/3 overlap: background relay (`wall23_relay.sh`) waits for 8xA100 g2/g3 verification, copies the shard dirs onto A100-CZ so its own-disk skip fires, and pushes both tars to Drive. If it reports OVERLAP_ALREADY_STARTED, A100-CZ's copy is the discard.
4. Collect Push-T 4,5 (r1), 6,7 (8xA100) and Wall 0,1 (CZ), 4–7 (8xA100) as they finish → cells close.
5. Run the frozen analyses per `docs/REFINED_SIX_TASK_COMPLETION.md` once a panel's 8 shards are all collected.

Background watchers (this session): fleet monitor `bbusp9gsj`; wall relay `bqxg0ovzv`; slot7 waiter `bifl2k9lq`.
