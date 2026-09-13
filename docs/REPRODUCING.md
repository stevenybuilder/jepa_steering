# Reproduction and evidence access

## Public, CPU-only checks

Install `pip install -e '.[analysis]'`, then run:

```bash
python scripts/check_public_results.py
python scripts/check_manuscript.py
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
python -m unittest discover -s tests -p test_decision_geometry.py -v
python -m unittest discover -s tests -p 'test_steering_specificity.py' -v
python -m unittest discover -s tests -p 'test_fresh_confirmation.py' -v
python -m unittest discover -s tests -p 'test_mechanism_energy_schema.py' -v
python -m unittest discover -s tests -p 'test_pilot_summary.py' -v
python -m unittest discover -s tests -p 'test_cem_steering_summary.py' -v
python -m unittest discover -s tests -p 'test_planned_prefix_summary.py' -v
python -m unittest discover -s tests -p 'test_planned_prefix_figures.py' -v
python -m unittest discover -s tests -p 'test_story_figures.py' -v
python -m unittest discover -s tests -p 'test_precision_story.py' -v
```

These check committed aggregates, their hashes and derived figures. They do not
independently reproduce simulator outcomes. Public CI tests portable components;
full integration tests need the pinned vendor and private fixtures.

## Source and data boundaries

The scientific runner is [fresh_confirmation.py](../src/offline_study/fresh_confirmation.py).
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

## Workshop manuscript

The curated LaTeX source, bibliography, official style and rendered draft are in
`paper/workshop/`. Figures reference the same committed `docs/figures/` assets as
the README. Build with `tectonic main.tex` from that directory after regenerating
figures. The author line is anonymous; this is a draft, not a claim of workshop
acceptance or a completed camera-ready submission.

## Repository curation

Scientific source, tests, configurations, aggregate evidence and figure builders
are retained. Historical operational notes, billing/host logs, scratch drafts and
bulk run archives are ignored and removed from the Git index, **not deleted from
local storage or cloud archives**. Existing Git history is not rewritten, so prior
commits may still contain those files. This is presentation cleanup, not retroactive
erasure of methodological amendments, negative results or execution limitations.
