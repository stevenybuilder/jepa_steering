# LCFM workshop manuscript

**How Activation Edits Affect Predictions and Planning in JEPA-WM**

Separate anonymous draft for the NeurIPS 2026 Long Context Foundation Models workshop. The main text is eight pages, followed by references and appendices. The original [full-study manuscript](../workshop/main.pdf) remains separate. This PDF has not been submitted.

## Workshop fit and format

The [workshop call](https://longcontextfm.github.io/), checked September 13, 2026, accepts long papers up to eight main-text pages, excluding references and appendices, with anonymous review. Its extended deadline is September 13 at 23:59 AoE (September 14, 11:59 UTC / 07:59 EDT). It includes long-horizon agentic models, multimodal context, and planning.

This paper studies a **two-frame rolling predictor and six imagined steps**, not long-token retrieval or context-length scaling. Its workshop connection is the consistency of an intervention when action history is reused during sequential prediction and planning. The title and abstract do not claim a new long-context benchmark.

The [current Overleaf template](https://www.overleaf.com/read/bpbmtzvcjnyh) was downloaded through its browser interface and verified. This draft uses its exact `neurips_2026.sty` (SHA256 `0bd02e695b26255c2c4cd1bff4d9c1e3d2a05e5a17fcc96b4cad81d9f28590ae`) in `dblblindworkshop` mode. It differs from the generic style only in the workshop-specific review footer. The template's generic text says nine pages, while the current workshop call says eight; this paper follows the stricter workshop limit. No margins or font sizes are changed to fit.

The template requires the NeurIPS checklist. It is included after the appendices, with its questions and guidelines retained and answers completed from the available evidence. Remaining submission-preparation gaps are explicit: no anonymous code/raw-data package, incomplete per-experiment compute and asset-license inventories, and an author ethics attestation that has not yet been completed. This is a reviewable draft, not a claim that every submission requirement has been closed.

## Revision and focused analysis

The September 13 revision plan guides this version. The main results follow forecast accuracy, fixed-bank choice, adaptive search, and physical execution. The action-history experiment is a supporting diagnostic, with its sixteen-context sample and changed-input reference stated explicitly, followed by numerical geometry and protected evaluation. Identifying repository links are removed from the manuscript; the PDF metadata has no author name.

The new [context-history figure](../../docs/figures/lcfm_context_history.pdf) uses only the already completed action-counterfactual experiment. It retains both tasks, both original/fresh banks, and H3/H4/H6. All-block patches form the primary illustration. B1 is explicitly a post hoc single-block example, and all six layers and all 72 registered contrasts remain in the appendix. The raw-action reference and all-block persistent patch agree exactly; the B1 persistent patch does not.

The optional candidate-ranking analysis is now complete. It compares the persistent and H3-only patches against the coherent raw-action reference using H6 Spearman correlation, actual top-ten elite overlap, and winner agreement. All seven sites, both tasks and both banks are retained. The new protocol was committed before calculating these pairwise comparisons; the analysis is explicitly post hoc and descriptive. [Complete ranking report and reproduction](../../docs/LCFM_HISTORY_RANKING.md).

No new model runs, fitted operators, longer horizons, or protected trials were added. There are no new significance tests or selected favorable candidates. Existing reconstruction curves retain their marginal context-bootstrap intervals; the new ranking figure reports descriptive means from all eight contexts per task.
[Analysis protocol and complete results](../../docs/ACTION_COUNTERFACTUAL.md) · [Figure generator](../../scripts/build_lcfm_history_figure.py).

The original study's result boundaries remain: a shared activation correction can change candidate-relative costs; an unchanged first winner does not imply unchanged elite membership; and the development history test does not establish the cause of protected behavioral outcomes.

## Build and check

From the repository root:

```bash
python scripts/build_lcfm_history_figure.py
python scripts/check_manuscript.py --manuscript paper/lcfm/main.tex
tectonic paper/lcfm/main.tex
```

The figure builder verifies the immutable source-table hash, all sixty displayed task/bank/arm/horizon cells, finite intervals, and eight contexts per task. The manuscript checker validates the frozen protected report, 32 protected and 48 historical values, figure paths, and citation keys. Rendered pages are inspected for legibility, clipping, and the main-text page boundary.


## Plain-language revision

The manuscript applies [blader/humanizer](https://github.com/blader/humanizer)
version 3.0.0, commit `9862685f575c65a8247f90369951df1b3416e3d6`, as an editorial
skill: identify formulaic prose, rewrite, then check the revision for unsupported
additions and lost technical meaning. The tool does not check scientific evidence.
The September 13 revision changes the title, abstract, introduction, result order,
and discussion. It states what 17/32 counts, names the reference forecast, and
keeps the proposed larger study separate from completed results.
[Follow-up design and sample-size rationale](../../docs/LCFM_FOLLOWUP_DESIGN.md).
