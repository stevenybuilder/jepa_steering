# Reproduction and evidence access

## Public, CPU-only checks

Install `pip install -e '.[analysis,test]'`, then run:

```bash
python scripts/check_public_repository.py
python scripts/check_public_results.py
python scripts/build_readme_figures.py
python analysis/mechanism/steering_specificity.py
python -m analysis.mechanism.decision_geometry
python scripts/build_comparison_figures.py
python scripts/build_story_figures.py
python scripts/build_precision_story.py
python -m analysis.mechanism.pathway_geometry --plots-only
python -m analysis.mechanism.pilot_summary --plots-only
python scripts/build_paper_figures.py
python scripts/build_cem_expansion_figures.py
python scripts/build_planned_prefix_figures.py
```

Run the [portable CPU tests](../tests/README.md) with `python -m pytest tests/unit`
after restoring the two pinned source fixtures described there.

These checks verify committed aggregates, their hashes and derived figures. They do not
independently reproduce simulator outcomes. Public CI tests portable components;
full integration tests need the pinned vendor and private fixtures.

## Code guide

| Start with | Purpose |
|---|---|
| [model_loader.py](../src/offline_study/models/model_loader.py) and [backends.py](../src/offline_study/models/backends.py) | Load the frozen checkpoint and call its predictor |
| [interventions.py](../src/offline_study/interventions/interventions.py) and [fixed_response.py](../src/offline_study/interventions/fixed_response.py) | Apply activation edits and the fitted rank-four correction |
| [fresh_confirmation.py](../src/offline_study/experiments/fresh_confirmation.py) | Run paired protected scenarios and their frozen analysis |
| [action_counterfactual_pilot.py](../src/offline_study/experiments/action_counterfactual_pilot.py) and [lcfm_replication.py](../src/offline_study/experiments/lcfm_replication.py) | Compare input changes with one-time and persistent conditioning patches |
| [steered_cem_pilot.py](../src/offline_study/experiments/steered_cem_pilot.py) and [planned_prefix_replay.py](../src/offline_study/experiments/planned_prefix_replay.py) | Trace adaptive planning and replay selected actions in the simulator |
| [Mechanism analyses](../analysis/mechanism/) | Reconstruct statistics and figures from recorded experiments |

The repository separates configuration, execution, analysis, and presentation:

- `configs/` defines study settings and pinned input assets.
- [src/offline_study/](../src/offline_study/README.md) groups model interfaces,
  interventions, planning, fitting, experiments, and evaluation. `tasks/` separates
  MetaWorld, Push-T, navigation, and DROID implementations.
- `analysis/mechanism/` contains CPU analyses of completed experiments.
- `scripts/` provides entry points, result checks, and figure/media builders.
  `scripts/vast/` retains archive, transfer, profiling, and scientific audit utilities.
- `paper/data/` contains machine-readable result tables, protocols, and provenance.
  `reports/` documents results and methodological corrections; `fresh-confirmation/`
  and `wm-approaches/` hold the protected and earlier behavioral summaries.
- `docs/` explains methods and findings. `docs/figures/` and `docs/media/` contain
  the displayed assets and their reproduction information.
- [tests/](../tests/README.md) separates the portable CPU suite in `unit/` from
  explicitly selected `integration/` checks. [Public CI](../.github/workflows/tests.yml)
  runs the unit suite and all three published-result checkers.

## Source and data boundaries

The scientific runner is [fresh_confirmation.py](../src/offline_study/experiments/fresh_confirmation.py).
It binds source, banks, audited inputs, checkpoints, scenario RNG, pairing and
analysis. [study.json](../configs/study.json) describes the broader development
study; the final confirmation is a separate freeze, not that older config alone.

| Artifact | SHA256 |
|---|---|
| Final protocol | `7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90` |
| Executed scientific source snapshot | `cde8274dad2bbc91efea3c5baba2e2266217949f9993aa77ce695d3bbabd4d82` |
| Committed final report | `8123d71497835fc164f09f5094c430308647a3ee2462c66746baea32633a0b15` |
| Raw metadata collection archive | `06c4dad7951c5bd2b4cb4d889f145aeb25cc0b691272133fb5b4817c16fabbd6` |

The report in `reports/fresh-confirmation/report.json` is unchanged. The mechanism
audit is a separately generated recount. Figure scripts read these aggregates;
no numbers are manually entered into the plots.

Raw per-scenario results, freeze manifests, fitted banks, pinned runtime/source and
host logs are retained in private GCS/Drive archives. They are **not bundled** and
repository access does not confer storage access. The collection archive is
`metadata-analysis-final.tgz`, immutable GCS generation `1789276664373051`.
Archive identity and access can be requested from the repository owner. No claim
of fully public raw-data reproducibility is made.

## Replay with authorized archive access

The command below uses the original module name inside the restored snapshot.
Current package paths have been reorganized; source hash checks intentionally
reject substituting the reorganized checkout for an executed snapshot.

After restoring and verifying the archive hashes, use its frozen source snapshot
and matching result/freeze trees. The current checkout also contains later
diagnostic code; it must not be silently treated as the executed source snapshot.

```bash
python -m offline_study.fresh_confirmation analyze \
  --project RESTORED_PROJECT --freeze RESTORED_FREEZE \
  --results RESTORED_RESULTS --output NEW_ANALYSIS_DIRECTORY
```

See [analysis data contract](../analysis/mechanism/DATA_CONTRACT.md) for layouts and
[analysis runner](../analysis/mechanism/README.md) for post-confirmation diagnostics.
Real model execution additionally needs upstream [JEPA-WM](https://github.com/facebookresearch/jepa-wms),
its checkpoints, licensed data and environment dependencies; no paid launcher is
part of the CPU reproduction commands.

## Action-history replication

The [replication report](LCFM_REPLICATION.md) describes the separate 200-state
protocol and links its public per-state selection and reconstruction tables.
The two candidate banks share starting states; aggregation uses 100 independent
states per task. The older sixteen-state study remains separate.

```bash
python scripts/check_lcfm_replication.py
python scripts/build_lcfm_replication_figures.py
python scripts/write_lcfm_replication_report.py
```

The checker recounts all public secondary means and checks primary Wilson
intervals against SciPy. The figure builder requires the
complete, source-bound summary. Raw archives and per-case extraction require
authorized cloud access; their generation and download-SHA proofs are retained
in the public summary. The statistical aggregator rejects missing or repeated
states and requires all 200 registered cases.
