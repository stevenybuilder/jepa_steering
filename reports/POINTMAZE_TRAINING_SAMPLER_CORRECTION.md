# PointMaze training sampler correction — September 8, 2026

Confirmed at approximately12:37UTC using the pinned actual upstream `init_data`,
its unchanged caller statements and real2000-row PointMaze metadata on Texas.
This is an implementation defect, not a changed experimental choice.

The upstream PointMaze branch sets `shuffle=False` for both training and validation
DistributedSamplers. Our history/pilot wrappers used `shuffle=True`. Native rank0
training begins `[0,16,32,48,64,80,96,112]`; the old wrapper begins
`[133244,92312,105435,39110,21791,25642,144454,2244]`. Native validation begins
`[0,16,32,48]`; the old wrapper begins `[10644,7456,6547,6926]`.
The original train/validation populations1800/200 and clip counts145800/12200
are unchanged. Sample order, including epoch behavior, is not author-native.

Affected: newly trained PointMaze234/235 histories and the queued236 training
implementation. Unaffected by this defect: completed released-checkpoint offline
comparisons, existing released-checkpoint behavioral panels, DROID and Wall histories.
The released checkpoints were not produced by this training wrapper.

At12:38UTC the parent used identity-rechecked pidfd SIGTERM to stop only Indiana
queue7843, whose cleanup terminated its owned child8167; absence of both and an
empty GPU were verified. Texas PointMaze CPU waiters1830/2061 were childless and
cancelled before any queued training, leaving DROID1296–1299 running. All original
checkpoints, metadata, failed/cancellation records and immutable source remain.
No provider instance was stopped or destroyed and no result file was deleted.

Required correction: task-explicit native sampler policy, tests against actual
upstream `init_data` rather than a manually chosen shuffle flag, and a binding that
rejects resuming a shuffled history as a corrected-native run. Restart corrected
PointMaze seeds from their own initialization after receiving numerical checks;
do not relabel or continue old weights as an author-matched trajectory. Correction
is in progress; no corrected history or rerun completion is claimed here.
