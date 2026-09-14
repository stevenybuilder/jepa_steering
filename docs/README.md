# Documentation

Start with the four documents in the first table; the rest are the per-experiment
reports that the README and the results document link into.

| Start here | What it covers |
|---|---|
| [METHODS.md](METHODS.md) | Frozen checkpoints, intervention operators, dose definitions, fitting protocol, and evaluation contracts |
| [RESULTS.md](RESULTS.md) | Every reported number with its population, uncertainty, and provenance, including the protected four-task confirmation |
| [REPRODUCING.md](REPRODUCING.md) | CPU checks and figure builds from committed aggregates; what full model replay requires |
| [COMPUTE.md](COMPUTE.md) | Model sizes, dataset sizes, GPU fleet, precision trade-offs, and execution accounting |
| [RELATED_WORK.md](RELATED_WORK.md) | Position relative to video-model interpretability, activation steering, and world-model planning |

| Experiment report | Question it answers |
|---|---|
| [LCFM_REPLICATION.md](LCFM_REPLICATION.md) | Does an action patch that is exact at its first read stay exact when the action is read again as history? 200-state pre-registered replication |
| [LCFM_HISTORY_RANKING.md](LCFM_HISTORY_RANKING.md) | How the same divergence reorders the planner's candidate ranking at H6 |
| [ACTION_COUNTERFACTUAL.md](ACTION_COUNTERFACTUAL.md) | The sixteen-state exploratory study that first showed the effect, with action-range controls |
| [ACTION_CONDITION_SPECIFICITY.md](ACTION_CONDITION_SPECIFICITY.md) | Does alignment with the action encoder matter beyond edit magnitude? |
| [MECHANISMS.md](MECHANISMS.md) | From internal edits to planner decisions: the score bound, layer response map, and controls |
| [CEM_EXPANSION.md](CEM_EXPANSION.md) | Whether edits change the Cross-Entropy Method search beyond the first selection |
| [PLANNED_PREFIX_REPLAY.md](PLANNED_PREFIX_REPLAY.md) | Selected-prefix forecasts versus physical execution in the simulator |
| [PILOT_MECHANISMS.md](PILOT_MECHANISMS.md) | Attention structure, CEM proposal dynamics, and cached-component replay |
| [CONTROLLED_GEOMETRY.md](CONTROLLED_GEOMETRY.md) | Why cubic versus linear reconstruction flips between FP32 and BF16 |
| [PATHWAY_GEOMETRY.md](PATHWAY_GEOMETRY.md) | Five-task geometry and visual-action interaction tests |
| [ANALYSIS_COMPLETION.md](ANALYSIS_COMPLETION.md) | Inventory of completed experiments, figures, and remaining evidence gaps |

`figures/` holds every figure in PNG, SVG, and PDF, generated from `paper/data/` by the
scripts in `scripts/`. `media/` holds the recorded JEPA-WM episodes and their
verification receipts; see [media/README.md](media/README.md).
