# LCFM workshop manuscript

**When Local Interventions Fail as Context Evolves in a Predictive World Model**

[PDF](main.pdf) · [Lead figure](../../docs/figures/lcfm_context_lifetime.png) ·
[Complete replication](../../docs/LCFM_REPLICATION.md) ·
[LeWM follow-up assessment](../../docs/LEWM_FOLLOWUP_ASSESSMENT.md)

This anonymous draft has **four main-text pages**, followed by references,
appendices, and the checklist. It centers the completed 200-state action-history
replication. The [full steering-study manuscript](../workshop/main.pdf) remains
separate. This PDF has not been submitted.

## What the paper establishes

A patch at an action's first appearance exactly reproduces the changed-input
forecast at that step. The original action returns in the next context window,
and the forecasts separate. The omission changes the lowest-cost candidate in
42/100 Reach states and 59/100 Reach-Wall states. Those selections have mean
extra reference costs of 1.81% and 1.87% in the primary bank. Patching both
appearances restores exact agreement.

The model uses **two frames of context and six imagined steps**. The reference is
the same frozen model with an input action changed. The study measures forecast
and selection consistency, with no new physical-outcome evaluation or
context-length scaling claim. The earlier sixteen-state study motivated the
prospective replication and is not pooled into its estimates.

The lead figure joins the conditioning sequence, measured forecast discrepancy,
and selection consequence. Appendices retain both banks, all six blocks, and
random controls. Primary rates use task-corrected Wilson intervals; secondary
means use paired state-bootstrap intervals. The separate LeWM assessment
recommends a small three-frame pilot and makes no claim of completed LeWM runs.

## Format and preparation status

The [workshop call](https://longcontextfm.github.io/), checked September 14, 2026,
accepts short papers up to four main-text pages and long papers up to eight,
excluding references and appendices. Its extended deadline is September 13 at
23:59 AoE (September 14, 11:59 UTC / 07:59 EDT). Review is anonymous and the
workshop is non-archival.

The draft uses the [workshop Overleaf template](https://www.overleaf.com/read/bpbmtzvcjnyh)
in `dblblindworkshop` mode. The exact `neurips_2026.sty` SHA256 is
`0bd02e695b26255c2c4cd1bff4d9c1e3d2a05e5a17fcc96b4cad81d9f28590ae`.
Margins and font sizes are unchanged. The checklist retains the template's
questions and guidelines, with answers revised for this paper's scope.

Remaining submission preparation is explicit in the checklist: an anonymous
code/raw-data package, a full asset-license inventory, and the author's ethics
attestation. The appendix documents the completed replication's GPU types,
representative runtimes, peak memory, and engineering exclusions; it does not
reconstruct all compute used by the earlier exploratory study.

## Build and check

From the repository root:

```bash
python scripts/check_lcfm_replication.py
python scripts/build_lcfm_replication_figures.py
python scripts/write_lcfm_replication_numbers.py
python scripts/check_lcfm_manuscript.py
tectonic paper/lcfm/main.tex
```

The replication checker verifies all 200 case identities, source/cloud receipts,
complete metric grids, and all 5,460 secondary means. It independently recounts
the primary selections and checks Wilson intervals against SciPy. Figure
creation verifies the source-table hash and requires the complete cohort.
The focused manuscript checker compares every generated result macro with the
immutable summary and checks figure paths, citations, and section references.
The original broad-study table checker remains unchanged.

## Editorial revision

The revision applies [blader/humanizer](https://github.com/blader/humanizer)
v3.0.0, commit `9862685f575c65a8247f90369951df1b3416e3d6`, as an editorial skill:
mark formulaic prose, rewrite, then check for unsupported additions or lost
technical meaning. It is not a scientific validation library.

The September 14 rewrite removes the unrelated learned-steering, geometry,
adaptive-search, and protected-behavior results from this LCFM manuscript. It
names the model-generated reference, follows one intervention through time, and
reports cost size beside selection frequency. The preceding broad draft is
preserved in the local campaign archive; the broader research remains available
in the separate full-study paper and repository reports.
