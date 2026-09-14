# Scripts

Entry points grouped by purpose. Everything under "Checks" and "Figures" runs on CPU
from the committed aggregates in `paper/data/`; the rest documents how inputs were
prepared and how the GPU fleet was operated, and needs licensed assets to rerun.

**Checks** (run these first)

| Script | Purpose |
|---|---|
| `check_public_results.py` | Recompute every published aggregate, interval, and provenance hash |
| `check_lcfm_replication.py` | Recount the 200-state action-history replication from its row-level records |
| `check_public_repository.py` | Verify the tracked file set and every Markdown link |

**Figures** (deterministic, provenance-stamped)

| Script | Output |
|---|---|
| `build_publication_figures.py`, `build_readme_figures.py`, `build_story_figures.py` | README and benchmark figures |
| `build_lcfm_replication_figures.py`, `build_lcfm_history_figure.py`, `build_lcfm_ranking_figure.py`, `build_action_history_figure.py` | Action-history and second-read divergence figures |
| `build_cem_expansion_figures.py`, `build_planned_prefix_figures.py` | CEM search and physical-replay figures |
| `build_comparison_figures.py`, `build_precision_story.py`, `build_paper_figures.py` | Benchmark comparison, precision, and print-sized figures |

**Input preparation and audits** (no model calls)

`prepare_protected_inputs.py`, `prepare_fresh_simulator_banks.py`, `run_fresh_bank_preparation.py`,
`validate_protected_inputs_cpu.py`, `verify_four_task_source_exposure.py`, `lcfm_replication_exposures.py`,
`audit_confirmation_exposure.py`, `audit_omitted_exposure_records.py`, `audit_droid_native_inputs.py`,
`diagnose_droid_input_parity.py`, `capture_preparation_runtime.py`, `archive_fresh_preparation.py`,
`pilot_fresh_pusht_trajectories.py`.

**Execution and reporting**

`run_fresh_confirmation.py` launches the audited four-task confirmation. `write_lcfm_replication_report.py`
writes the replication report from the audited tables. `vast/` holds the fleet tooling: static queues,
parallel downloads, streaming backups, per-case draining, and direct archive to cloud storage.

**Media**

`capture_jepa_episode.py`, `render_jepa_episode.py`, `render_jepa_comparison.py`, `compose_jepa_episode_media.py`,
`render_qualitative_rollouts.py`, `render_task_demonstration.py` produce the recorded episodes in `docs/media/`.
