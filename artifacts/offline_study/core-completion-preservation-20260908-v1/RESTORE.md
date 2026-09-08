# Completed core panel: preservation and recovery

Private folder: https://drive.google.com/drive/folders/1SoXkmEJmrCZXBnKk3re6mX8WwJiWBokY

This archive preserves the completed **960-episode, two-task, five-arm** MetaWorld
behavioral-development comparison and its frozen paired analysis. It also retains
the paused HMM work. Partial HMM records are not completed comparisons or saved
random-number-generator state. No learned intervention passed the frozen success
improvement gates; this is not fresh confirmation or the full six-task study.

## Main archive

Download all fourteen `core-and-paused-hmm.tar.gz.part-0000` through `part-0013`
files together with `core-PARTS.json` and `core-FILES.json`. PARTS records exact
join order, Drive IDs, byte sizes and SHA256 hashes. Join in that order to a NEW
file named `core-and-paused-hmm.tar.gz`, checking each part before concatenation.
Never overwrite an existing archive or scientific output directory.

The joined archive must contain **1,366,522,673 bytes**, SHA256:

`2c3d61175cebf2db27794e607f5169c1ef8948e4511fdf962400505c5a238b38`

It contains **6,872 regular files** relative to `/workspace/jepa-runtime`:
episode/action/model-call records, all forty shard reports, exact analysis,
frozen protocol, fitted intervention assets, paired stimuli, released checkpoint,
executed source, receiving checks, and paused HMM v3 evidence. Use the full member
manifest rather than a directory count to verify recovery.

Before extracting to a new empty directory, check the full compressed hash and
EVERY archive member against `core-FILES.json`. The repository function
`scripts/vast/backup_results_to_google.py:verify_archive` performs this without
extracting. `core-DRIVE_VERIFIED.json` records full Drive byte/member readback;
per-part upload acknowledgements alone do not establish that verification.

An independently verified unsplit fallback exists in the same account:
`gs://rgt-jepa-archive-2026/rep_geometry_transcoder/core-completion-preservation-20260908-v1/core-and-paused-hmm.tar.gz`
with generation `1788903831492349`.

## Additional work and logs

`la-work-and-logs-supplement.tar.gz` preserves another **661 files**, including
the launcher/queue/worker logs outside the main result directories, earlier HMM
code/evidence/failed attempts, administrative scripts, and exact vendor working
tree. Its files are relative to `/workspace`, not `/workspace/jepa-runtime`.
Use `supplement-FILES.json` and `supplement-DRIVE_VERIFIED.json` to verify it.

Supplement size: **17,967,959 bytes**; SHA256:
`a730c5cd34fa282458a59f1fc32a54d0482160aae979640abb391a05c56872bd`

`ENVIRONMENT.json` records the two Python environment package inventories and
vendor commit. Credentials, rebuildable bytecode, Git internals, and duplicate
backup archives are excluded. The supplemental inventory separately hashes the
126 released MetaWorld dataset-cache files retained on the original volume;
these are downloadable official inputs, not newly collected outcome records.
The exact paired evaluation stimuli are included in the main archive.

## Other workers and limits

The four ancillary workers' selected outputs, logs, and latest complete training
resume checkpoints are preserved in the separate private folder:
https://drive.google.com/drive/folders/12r-UzKMTuyYm4wXKl9xPb3r5dcjfwsYr

That folder includes the verified twelve-part NJ archive and its restoration
metadata. Earlier completed DROID/navigation/offline archives and historical
snapshots remain under https://drive.google.com/drive/folders/1asQRvywFx8cZ6CeHJKGM9rV0V569P24q.

**Do not destroy the stopped source volumes using these receipts alone.**
Older training checkpoints not selected for the pause archives remain on those
volumes. Stopping compute retains them; it does not eliminate storage charges.
No worker results or checkpoints are deleted by this preservation operation.
All additional experiments remain paused pending a new user instruction.
