# Workshop manuscript

**Steering Latent World Predictions: From Internal Geometry to Planner Decisions**

Mechanism-focused preprint updated September 13, 2026. The official NeurIPS 2026 style is retained in preprint mode with an anonymous author line. No affiliation, workshop acceptance, or final page count is asserted.

## Scope and evidence

The manuscript separates the earlier rank-one layer sweep, later fixed-response rank-four operator, historical fixed-bank decision diagnostic, local pathway/geometry experiments, new 64-context attention/component replay, eight-context actual-CEM follow-up, and protected closed-loop confirmation. It does not infer an end-to-end causal mechanism across those populations. The completed GPU diagnostics use development inputs and do not execute robot actions.

| Evidence | Authoritative source | Scope |
|---|---|---|
| Protected 32 rates and 48 paired contrasts | `reports/fresh-confirmation/report.json` | 384 fresh scenarios × 8 arms; same-physical-GPU pairing within scenario |
| Protected coefficient and outcome audit | `reports/fresh-confirmation/mechanism_audit.json`, `paper/data/candidate_specificity.json` | Behavioral sensitivity and coefficient energy, not recovered candidate scores |
| Early rank-one layer response | `paper/data/layer_mechanism_grid.csv` | 576 recounted cells; independent units remain source lineages |
| Finite-bank margin certificate | `paper/data/decision_geometry.json`, `decision_geometry_scenarios.csv` | All 192 canonical development banks; 300 candidates each |
| Interpolation and pathway geometry | `paper/data/pathway_geometry.json`, `pathway_geometry_*summary.csv` | 17 verified source reports; precision-specific fits/doses; matched requested dose is not exact realized-energy equality |
| Native attention and cached-component replay | `paper/data/pilot_summary.json`, `pilot_summary_*.csv` | All 64 development contexts; exact cache/zero parity; native-energy components; bulk-preservation status separate from compact analysis readiness |
| Actual steered CEM | `paper/data/cem_steering_summary.json`, `cem_steering_*.csv` | Eight development contexts; both learned/random arms; 15 iterations; returned 60-coordinate prefix, no executed actions |
| Author and historical comparison | `paper/data/benchmark_comparison_sources.json`, `benchmark_comparison.csv` | External context and exposed development; never pooled into fresh confirmation |
| Literature | `docs/LITERATURE_MECHANISMS.md`, `references.bib` | Primary-source metadata and explicit endpoint boundaries |

Protected report SHA256: `8123d71497835fc164f09f5094c430308647a3ee2462c66746baea32633a0b15`.

The code/aggregate package is at `https://github.com/stevenybuilder/jepa_steering`, currently private. The manuscript does not claim that raw archives are publicly released. LeWorldModel, Memory Maze, and adaptation are future testbeds or hypotheses, not completed extensions. The systems appendix describes actual whole-scenario PyTorch replicas and cloud preservation, without claiming FSDP or a JAX rewrite.

The planner objective in the analyzed MetaWorld configuration is visual goal MSE + 0.1 × proprioceptive goal MSE. Full MetaWorld episodes have seven planning calls, not 33. Archived first-bank margins cannot explain protected outcome flips because cohort and planner RNG differ and protected numeric candidate costs were not saved. DROID remains 64 recorded-plan endpoints, not physical robot success.

## Figures and build

All current figures are referenced from `../../docs/figures`, not the old workshop `figures/` directory. Main figures: clean architecture, one separate ablation matrix, rank-one layer heatmap, candidate-margin geometry, precision/pathway geometry, and provenance-separated headline benchmark. Historical workshop PDFs remain on disk but are not included.

From the repository root, regenerate the architecture/table and benchmark figures with:

```bash
.venv/bin/python scripts/build_readme_figures.py --only architecture ablation
.venv/bin/python scripts/build_comparison_figures.py
.venv/bin/python scripts/build_paper_figures.py
.venv/bin/python scripts/check_public_results.py
.venv/bin/python scripts/check_manuscript.py
```

The decision/pathway figures come from their source-bound analysis scripts; rebuilding the manuscript does not require model inference or new GPU work.

Build from this directory:

```bash
tectonic main.tex
```

After building, render the PDF pages and check table/figure legibility and unresolved references. The print variants use larger labels than the wide README figures and are generated from the same verified tables.

`neurips_2026.sty` is the official 2026 file dated 2026-01-29, SHA256 `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`. The current source, bibliography, style, and PDF are versioned. Historical drafts and intermediate build files remain local and ignored; the rewrite does not change model weights or frozen results.
