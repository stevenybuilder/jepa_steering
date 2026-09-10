# Navigation geometry offline closure readiness

Read-only audit: **2026-09-08T14:49:32 UTC**, owned Indiana50205763
(`jepa-navigation-offline-indiana`, provider verified running/Indiana US).
No GPU/model calls, signals, scientific reruns, statistical recomputation or
remote writes. Only this local report was added.

## Finding

**All four Wall/PointMaze × BF16/FP32 action-response-geometry scopes are complete,
already aggregated, and freshly reverified. No raw shard is missing.** The
PointMazeFP32-ongoing statement in ABLATION_COVERAGE_STATUS is stale.

| Task | Precision | Complete shards | Trajectories | Prefixes/arm = diagnostic rows | Seven-arm measurement rows |
|---|---|---:|---:|---:|---:|
| Wall | BF16 |8/8, indices0–7|192|4,224|29,568|
| PointMaze | BF16 |8/8, indices0–7|200|24,400|170,800|
| Wall | FP32 |8/8, indices0–7|192|4,224|29,568|
| PointMaze | FP32 |8/8, indices0–7|200|24,400|170,800|

These are392 source trajectories across two tasks, reused in both precisions—not
784 independent trajectories. All seven frozen arms remain present: native,
zero-dose, equal-anchor linear, cubic, projected cubic, reflected curvature and
matched random. This is offline development/replication, **not** completed
geometry behavioral planning, untouched confirmation or training-history evidence.

## Exact existing evidence

All paths below are on Indiana under `/workspace/jepa-runtime`:

- Raw: `navigation-comparisons-20260907-v1/{precision}/{task}/action_response_geometry/shard-{0..7}`.
- Aggregate: `navigation-analysis-20260907-v1/{precision}/{task}/action_response_geometry`.
- Frozen fit/protocol: `navigation-fits-20260907-v1/{precision}/{task}/action_response_geometry`.
- Cohort: `navigation-offline-cohorts-20260907-v1/{task}/cohort.json`.
- Evaluation source: `navigation-evaluation-code-v1/src/offline_study`, SHA256
  `abb50325c78027ecba580c9eb5b222ac0735513b28f2e5cde12681f406b24c68`.
- Original CPU analysis source: `navigation-analysis-code-v1/src/offline_study`, SHA256
  `f2ff6ef8dfa860286987f071f3213c019aa200158bfd5bce38edfb5b1827ec9c`.

| Scope | Frozen protocol SHA256 | Existing aggregate report SHA256 |
|---|---|---|
| BF16 Wall |`fdefd9ece07075863cc7dca1461932f2cb0eea8370b00cff0308164cb311b5eb`|`643247c9860778d46c7deea714e045de15d4a401d8bc1b96a506f405d2def9bd`|
| BF16 PointMaze |`deb8968c73984379ace32729bc7f5b275f5aad4307efcb7b1a2e97e18fc2c089`|`34797270e44abf8937457060b8fda437fe02bb46726e1acf7e6b49e4beb0bfde`|
| FP32 Wall |`09d8124e8458f0a8d013a38f4438244c50dcd80dc5cea7ffb1b38127cee56a73`|`3f5315532e50c29bd794996dfc29f416e00cf7b717e0e4031cee6be736687077`|
| FP32 PointMaze |`7e453e87d02e78b37715c777f83de52f4b7254d621f9fdd34221b0223bc861fb`|`550fc44b126621b221d5abca79e718e96953de0455a4b4c2947ac4ce585b6d95`|

Every aggregate DONE→report hash passed; every status is
`verified_author_corrected_development_complete`. Each aggregate's40 input
bindings exactly equal its eight shards × five raw-file hashes. Latest aggregate
filesystem mtime is PointMazeFP32 at07:45:05UTC; this audit does not infer completion
time from filesystem mtime alone.

Fresh checks also established:

- All32 expected shard directories, no extra/missing shard and no shard FAILED.
- All160 DONE-bound report/window-metrics/selection/protocol/mechanism files intact.
- Exact task/precision/shard, checkpoint, cohort, fit receipt and protocol bindings;
  original evaluation-source hash, zero-dose identity and native-fidelity gates.
- Original frozen CPU `validate_cohort`, `verify_runtime_access`, `verify_fit`,
  `expected_shards`, `examples` and `check_coverage` checks passed. No duplicated
  or omitted trajectory/prefix/arm keys. Every diagnostic key is present once,
  including required `cubic_visual_native_fidelity_mse_h6`.
- No efficacy values or favorable-arm selection were reported by this audit.

The frozen geometry uses four anchors−1/−0.5/+0.5/+1, omits0, radius0.1 and
newest256 visual patches atP3/H3. Perturbed donors are model responses; the physical
target remains the recorded future of the original action. Protocols explicitly
warn that armwise energy normalization does not preserve raw projection/reflection
geometry. No current pending-template setting was substituted for those freezes.

## Preservation completed locally; Drive publication status

At15:25:41UTC the exact431 selected files (959,303,006 raw bytes) were archived
without remote writes, GPU calls or scientific recomputation. The125,518,931-byte
archive passed full compressed/member hash verification; complete source manifests
and pinned source selection were identical before and after transfer.
Archive SHA256 `63ff98086d7be6534494a4d45075b02d05eea52f4d9395e30a93c5e3cfffcb1a`.
[Local verified archive receipt](../artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/VERIFIED.json).

The connected Drive uploader rejected the unsplit file above its100MiB limit.
Two ordinary byte parts now exist in the verified private
[geometry folder](https://drive.google.com/drive/folders/1BOFRj7xdDWfD6yVkO-_drPtqb3Sp3CDp),
along with `PARTS.json`, `RESTORE.md`, `FILES.json` and the local verification
receipt. All six file sizes/parents/private permissions were checked. Local
rejoining preserves the exact original archive hash; splitting does not change
results or archive members. This is a transfer-format change, not a GPU speedup.

Direct Drive readback verified all67,108,864 bytes and providerSHA256 of part-aa.
Part-ab then hitHTTP403 quota; **full Drive rejoined/member readback is still
pending**, and the partial readback/failure evidence remains intact. Do not infer
permission to delete local or worker data from successful upload alone. Nothing
was deleted. A later bounded readback can complete this check independently of
GPU execution; no geometry result rerun is required.

## Historical safe next CPU step from the initial audit

Do **not** restart `run_navigation_nonrank_remaining.py`, `navigation_evaluate`,
or the five-category analyzer loop. Geometry aggregation already exists. Preserve
the four existing aggregates, their bound raw files/fits/cohorts and original
source snapshots; then update the stale coverage table. This audit found no local
geometry aggregate in the checked `navigation-analysis-compact-20260907-v1` or
`primary-durable-20260907/navigation-analysis-20260907-v1` roots; both only contain
earlier Wall-BF16 coupling evidence. Remote completion is verified, but a fresh
local/private-Drive geometry closure export remains the concrete preservation step.

The following read-only command can recheck the existing aggregate/raw binding
chain on Indiana before that separately authorized export. It neither writes nor
recomputes statistics:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 - <<'PY'
import hashlib, json
from pathlib import Path
r = Path('/workspace/jepa-runtime')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()
for precision in ('bfloat16', 'float32'):
    for task in ('wall', 'pointmaze'):
        a = r/'navigation-analysis-20260907-v1'/precision/task/'action_response_geometry'
        raw = r/'navigation-comparisons-20260907-v1'/precision/task/'action_response_geometry'
        report = json.loads((a/'report.json').read_text())
        assert json.loads((a/'DONE.json').read_text())['report_sha256'] == sha(a/'report.json')
        assert report['status'] == 'verified_author_corrected_development_complete'
        expected = {}
        for rank in range(8):
            p = raw/f'shard-{rank}'
            assert not (p/'FAILED.json').exists()
            done = json.loads((p/'DONE.json').read_text())
            for name in ('report', 'window_metrics', 'selection', 'protocol', 'mechanism_diagnostics'):
                f = p/(name+'.json'); h = sha(f)
                assert h == done[name+'_sha256']; expected[str(f)] = h
        assert report['input_sha256'] == expected
        print(task, precision, '8 shards / 40 bindings verified', sha(a/'report.json'))
PY
```
