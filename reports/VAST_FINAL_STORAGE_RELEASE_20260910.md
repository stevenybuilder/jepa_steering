# Final Vast storage release — 12 rentals deleted; 2 inaccessible disks retained

User instruction: preserve all necessary JEPA research results/work/logs in
Google Drive, then end rental billing. No scientific jobs or GPU restarts.

## September 10 merge-time safety review

The closeout evidence below is historical and has not been rewritten. A later
static review found that the reusable release helper could promote historical
path/size matches without enforcing a current content hash. That is not a safe
general release gate: equal-size replacement content could be missed.
Before committing the helper, this fallback was removed. New plans recapture
historical matches and deduplicate only after hashing actual captured bytes;
changed existing coverage receipts now fail closed instead of being reused.
Four local regression tests exercise these guards with no provider calls.

This code correction does not retroactively add source evidence to old receipts,
establish data loss, or authorize another destruction. Source-hashed backup sets
remain verified as described below; metadata-only historical reconciliation is
not equivalent to proof of every current source byte. No recovery helper or
lifecycle action was run during the Git merge work.

Private final-preservation folder:
https://drive.google.com/drive/folders/14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48

All 960 core episodes, analysis, paused HMM outputs and restoration metadata are
already in the private completed-core folder:
https://drive.google.com/drive/folders/1SoXkmEJmrCZXBnKk3re6mX8WwJiWBokY

## Confirmed destruction

- 50135088, 50135089, 50135090: no scientific runs; complete bootstrap/code/log
  archives plus inventories passed full Drive byte readback before deletion.
- 50229002: unused image bootstrap, no research workspace; inventory archived
  and read back before deletion.
- 50233992: final core worker. 7,531 remaining source result/code files matched
  the two previously source-hashed, fully read-back archives; no later size or
  modification-time changes. Fifteen exact Drive archive objects rechecked;
  final top-level logs, inventory and release coverage separately uploaded and
  fully read back.
- 50159352: all 612 required workspace files preserved and fully read back;
  final restore index `1aghI05yHDd-wg3YD5odnxR8wEKt9bWFx`.
- 50195621: all 2,616 required files reconciled to new and prior verified
  archives; final restore index `14xJnFv4qGH-owv2JuDRjcimbSRR6Ss5N`.
- 50231985: all 5,237 required files preserved; final restore index
  `1UR7BVaeFQrCTDgEWLqy0Dx2aRpsFssYa`.
- 50259194: all 5,064 required files preserved, including unique checkpoints;
  final restore index `1qw1MI6VDdKluLZZEIDXPssdkEVTHhgml`.
- 50189244: all 5,693 required files accounted for, including 48 raw capture
  and metric objects, complete retained checkpoint histories and case-sensitive
  DROID metadata corrections. Final restore index
  `1U0gjkmR3CXbPd9hvYBb0I1qIQNQ6qij-`.
- 50239185: all 1,674 required files preserved, including older Wall training
  checkpoints, 91 backup batches and three explicit provenance supplements.
  Final restore index `1FgYM-S-EZGVw6OTrsCIBlVYSDMM97B1b`.
- 50245262: user explicitly resolved the stale other-thread ownership hold.
  All 825 required files (371,591,458 source bytes) reconciled to verified
  Drive copies, including an exact-hash existing copy of `states.pth`.
  Case audit found no aliases; eight excluded symlinks belonged to the
  rebuildable Python environment. Final restore index
  `1rirjE0x6eO7rRr0LOLVn4_gxRPrLypi9` passed full readback before deletion.

Provider deletion readbacks confirm all twelve IDs absent. California's
`50245262/DELETION_READBACK.json` records the additional deletion on September 10.

Destroyed disks are irreversible at Vast. Their necessary records remain
recoverable from Drive. All GPUs are stopped; no separate Vast volumes were found.
The earlier $0.245370/hour figure was a nominal storage-rate estimate, NOT a
verified measurement of charges accruing while the hosts were inaccessible.
The two retained disks have nominal rates totaling $0.217593/hour, but Vast's
documentation says offline machines and loading instances are not charged.
At 07:37:18 UTC, the billing API's `current` object reported charges 0, service
fee 0 and total 0. That field is not an hourly meter and does not prove future
billing will remain zero. Support was asked to confirm actual charges and
review any inaccessible-storage charges. The two contracts remain open.

Sources: [instance states and loading billing](https://docs.vast.ai/guides/instances/manage-instances),
[offline billing](https://docs.vast.ai/guides/instances/storage/volumes).

## Remaining work

All accessible, ownership-resolved study rentals have been preserved and
destroyed. No backup or scientific job remains running from this turn.

| Retained rental | Nominal storage/hour, not verified accrual | Blocker |
| --- | ---: | --- |
| 50125440 | $0.180556 | Host offline; stopped-disk reads time out. Earlier archives exist, but full remaining source coverage cannot be established. |
| 50205763 | $0.037037 | Stopped/loading; source-copy paths rejected and directory reads time out. Prior 431-file navigation archive now fully verified in Drive, but not a complete current disk inventory. |

The latest primary/Indiana retry began at 07:29:36 UTC and failed. Preserve
these disks until remaining source coverage is verified. Neither was deleted
or restarted. The user authorized support contact and explicitly confirmed
there is no other ongoing thread managing California.

An email was SENT to support@vast.ai requesting disk preservation, read-only
export/access recovery, exact last-contact/outage times, confirmation of any
loss, and billing review. It explicitly prohibits restarting paid compute,
deleting, reinitializing or overwriting disks without checking with the user.
Sent-message readback was verified; message/thread ID `1a08a3a8ccb810eb`.
The receipt and exact request are in `VAST_SUPPORT_SENT.json`.

## Access chronology and data-loss audit

- A September 8 shutdown snapshot associated with 22:06:48 UTC (18:06:48 EDT)
  listed both rentals stopped/exited. This is provider state, not a successful
  source-disk read at that moment.
- The earliest documented failed access probes identified in this audit began
  around September 10 05:32 UTC (01:32 EDT). This establishes when failures
  were observed, not when the outage actually began. Later probes also failed.
- Primary: all **741 canonical known artifact hashes** were mapped to verified
  backup members or raw objects; no known artifact was left unmapped.
- A fresh full Drive stream verified the primary archive's compressed SHA256,
  size and **all 2,244 files**, accounting for 542 directory entries. An initial
  attempt used a newer verifier that rejects directory members; the compatible
  full-member verification passed. That format mismatch was not corruption.
- Indiana: the prior **431-file navigation archive** passed complete Drive
  byte/member verification. This is a selected-results archive, not a complete
  inventory of the currently inaccessible source disk.
- **No loss was found in the audited backup sets.** The identified results are
  preserved, but unknown/uninventoried files on inaccessible disks cannot be
  certified intact or lost until access is restored or the provider investigates.

Audit receipts: `PRIMARY_KNOWN_ARTIFACT_COVERAGE_AUDIT.json`,
`PRIMARY_INITIAL_ARCHIVE_FRESH_READBACK.json`,
`INDIANA_PRIOR_ARCHIVE_FULL_DRIVE_READBACK.json`,
`BILLING_CURRENT_AUDIT_20260910.json`.

## Transfer correction

Vast's stopped-disk transfer works using C.<instance> for both the API source
and rsync module. The installed CLI's numeric-module path failed and its wrapper
did not propagate rsync failures; the preservation helper checks actual status.
No GPU/container restart is required.

Rclone's shared OAuth project 202264815644 repeatedly hit project request quota,
including metadata reads, not the user's Drive storage capacity. The connected
Drive uploader successfully supplied a fallback for small files. Browser upload
inspection stalled and was not used to transmit files.

On September 10, the Google Drive API was enabled on the EXISTING JEPA archive
project `project-flash-490419` (projectNumber 31043195041, independently confirmed
as the owner of bucket `rgt-jepa-archive-2026`). Backup requests now specify
`X-Goog-User-Project` for its own quota. First exact Drive checksum read succeeded.
No new credentials, IAM grants, file sharing, cloud project, bucket, or compute
were created. [Google's quota-project documentation](https://docs.cloud.google.com/docs/quotas/set-quota-project).

## Evidence and recovery

The full source inventory is also checked for case-only filename differences.
Virginia has four DROID metadata paths that alias on macOS. Each Linux source
is copied separately to a numbered file and archived under its original exact
name. `case-alias-corrections/DRIVE_VERIFIED.json` supersedes those members in
earlier batch archives; `FINAL_RELEASE_COVERAGE.json` selects the corrected
copies. Restore to a case-sensitive filesystem to preserve both names.

Local receipts: `artifacts/offline_study/final-storage-release-20260910-v1/`.
Every uploaded object records Drive ID, size, SHA256 and full byte readback.
Batch archives include original workspace-relative paths and a SHA256 member
manifest; single-file checkpoint/capture receipts map each source path to its
Drive object. Multipart fallback receipts specify ordered concatenation and
the whole-file checksum. Keep receipts with the final Drive index.

Restore instructions:

1. Download the newest `JEPA-final-storage-recovery-index-*.tar.gz` from the
   final-preservation folder. It contains manifests, deletion receipts, the
   resource ledger, these instructions, and preservation helper source.
2. Open the desired instance's `FINAL_RELEASE_COVERAGE.json`. Every required
   source path points to a batch receipt, an earlier archive/member proof, or
   a raw-object receipt. Follow those exact Drive IDs, not filenames alone.
3. Verify each object's size and SHA256. Concatenate multipart objects only
   in the manifest's recorded order, then verify the complete archive checksum.
4. Restore on a case-sensitive filesystem. Where the coverage map marks a
   correction/override, use that object's version of the member. Symlink targets
   are explicitly recorded; do not blindly extract absolute or escaping links.
5. Public datasets, released official checkpoints, Python environments and
   credentials are not research outputs. Recreate them from preserved source
   receipts/configs and supply credentials separately. No experiment resumes
   automatically from these archives.

Only verified local staging/archive duplicates were removed, with individual
receipts. NJ, Wall late-history, DROID-fit-v2 and DROID-coupling archive copies
were removed only after exact Drive-object checks and prior complete readback.
An older DROID first-attempt archive did NOT match the verified replacement
byte-for-byte and was retained locally. No broad local cleanup was performed.

All eight preservation helpers compile; four bounded-batch safety checks passed
(idempotent completion, incomplete-finalization refusal, worker-count limit,
and rejection of partially written reuse receipts). Actual source inventories,
member SHA256 checks, complete Drive byte readbacks and provider deletion
readbacks—not those software tests alone—gated each destruction.
