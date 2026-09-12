# Paper artifacts

**Working title: Correcting Imagined Futures in Frozen JEPA World Models**

Development-results draft, September 8, 2026. The main text is approximately 2,200 words rendered as four core pages, followed by one references page. Figures and the evidence appendix are separate. No completed behavioral improvement or cross-task transfer is claimed.

## Read first

- [Main findings and paper context](../findings.md)
- [Extended abstract: editable Markdown](extended_abstract.md) · [PDF](extended_abstract.pdf)
- [Figure gallery](figures.md)
- [Research synthesis: hypotheses, novelty, circuit analysis, architecture, efficiency, and researchers](research_synthesis.md)
- [Introduction framing and contribution choices](framing.md)
- [JEPA-WM methodology alignment](tables/methodology_alignment.md) · [Task coverage](tables/task_coverage.md)
- [Evidence appendix](appendix.md) · [References](references.md)

## Available figures

1. Fixed-response intervention pipeline (conceptual).
2. Completed forecast effects by task, with matched random controls.
3. Pathway and spatial-permutation comparisons.
4. Separate depth and spatial-support sweeps.
5. Precision-dependent reconstruction versus forecast correction.
6. Actual fitted rank-4 basis loadings over the patch grid.

Every figure has an editable SVG, a PNG preview, and a plain-text caption in `figures/`. No success-rate bar is fabricated for pending comparisons. Figure 6 summarizes existing fitted tensors; it is not a semantic or circuit map.

## Provenance and reproducibility

[All reported contrasts](data/reported_contrasts.csv) exports 1,536 registered contrasts from 36 completed task/category/precision scopes, including both endpoints. The plotter verifies source report hashes against completion receipts and the primary closure manifest. [Figure provenance](data/figure_provenance.json) records source and output hashes. [Basis summary](data/basis_spatial_summary.json) records the two existing bank hashes and their spatial loading summaries. These checks establish provenance of the included report summaries, not a new end-to-end audit of all raw data.

The authors' released JEPA-WM plotting workflow uses **Matplotlib and Seaborn**, verified in the pinned repository files linked in [References](references.md#plotting-implementation-source). We use the same plotting libraries. **ReportLab** renders our PDF; its use is our choice, not a claim about the authors' manuscript workflow. PyMuPDF produces the review renders.

No model was trained or evaluated, and no partial behavioral outcome was read to produce these artifacts. Scientific settings, fitting banks, and experiment jobs are unchanged.

### Rebuild locally

Use an environment containing the pinned [plotting requirements](requirements.txt). The existing project environment supplies PyTorch only for reading the already fitted banks. The initial build used a temporary isolated plotting environment at `/tmp/rep-geometry-paper-venv`; no project dependency file was changed.

```bash
cd /Users/stevenyang/Documents/rep_geometry_transcoder
.venv/bin/python paper/scripts/export_basis.py
/tmp/rep-geometry-paper-venv/bin/python paper/scripts/build_figures.py
/tmp/rep-geometry-paper-venv/bin/python paper/scripts/build_pdf.py
```

To recreate the plotting environment if the temporary directory is gone:

```bash
uv venv /tmp/rep-geometry-paper-venv --python python3
uv pip install --no-cache --python /tmp/rep-geometry-paper-venv/bin/python -r paper/requirements.txt
```

The PDF builder verifies four core sections on their intended pages and saves renderings plus [layout checks](review/layout_check.json). Review the renderings after substantive text or layout changes. A future conference template may require typesetting changes; this is a readable four-page draft, not a claim of compliance with an unspecified venue style.

## Later additions

Only after the frozen paired behavioral analyses complete: task-success comparisons with the correct uncertainty and episode pairing, representative source-bound rollouts, and a measured full-planner cost/quality comparison. Add training-seed/history results when complete. Keep the original and successor operator results labeled separately.
