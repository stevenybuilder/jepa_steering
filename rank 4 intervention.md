# Rank 4 intervention

*Summary of completed preliminary findings and implementation tradeoffs. Saved September 7, 2026 (EDT).*

**The two criticisms are computational cost and insufficient mechanistic specificity.** Rank 4 can have a useful effect while the ablation remains expensive and tells us too little about what information it changes. This is not a claim that rank 4 is ineffective.

## The three most interesting findings

| Finding | What we measured | Why it is interesting |
|---|---|---|
| **1. A small learned subspace can improve forecasts.** | Rank 4 reduces H6 proprioceptive embedding MSE by **2.887% on Reach** and **2.137% on Reach-Wall**, passing the registered native/random-control gates. On Reach, rank 1 also passes those gates, reducing error by **1.899%**, but is not established equivalent to the best eligible arm. Rank 4 is selected by the smallest-equivalent-eligible-rank rule; rank 8 is equivalent. | We have evidence that a compact correction is sufficient under this construction. That supports retaining a small output space. It does **not** identify four meaningful physical variables or prove that four is universally optimal. |
| **2. Coupling and rank correction both contribute to Reach’s combined result.** | Equal-budget coupling alone improves the error by **2.035%**. Coupling plus rank 4 improves it by **4.824%**, beating the combined random control and either component-removal arm. | There is additional forecast benefit from the combination. Cheap coupling is therefore both a promising component and an essential practical comparator. However, removing a component also removes energy, so this does **not** establish synergy at equal total energy. |
| **3. The effects are selective across task/model settings.** | Push-T rank 4 gives **0.198%**, with an interval spanning zero; joint coupling makes error **0.638% worse**. No action-response-geometry arm qualifies on any of the three primary tasks. | These interventions do not reliably improve every setting. A repeatable recipe needs a criterion for when its correction is relevant, including when to retain the native model. |

These are completed **development forecast results**, using 33 Reach, 27 Reach-Wall, and 21 Push-T trajectories. They do not establish improved task success. Repeated prefixes, arms, and precision runs do not increase the independent trajectory count. Sources: [primary results](reports/CORRECTED_OFFLINE_RESULTS.md#main-comparisons-with-the-unsteered-checkpoint), [combined analysis](docs/COMBINED_DEVELOPMENT.md#completed-measurements-and-non-routed-selection).

## Rank 4: useful parts and limitations

| Component | The good part | The limitation |
|---|---|---|
| **Four fitted directions** | Compact, reusable basis: approximately **1.56 MiB**. Produces one structured addition. | PCA selects activation variance; it does not explain which task-relevant information those directions repair. |
| **A fixed B3/H3 intervention site** | Clearly specifies where and when the intervention occurs. | Full spatial support leaves its mechanism unresolved. Our separate localization sweeps used **rank 1**, so they cannot identify the necessary patches or layers for rank 4. |
| **Candidate-specific coefficients** | Adapts the correction to how a candidate’s forecast responds. This is a reasonable mathematical motivation. | We have not established whether that expensive adaptation is necessary, or whether a cheap map would produce equally useful edits. |
| **Regularized 4×4 solve** | Small, well-defined linear-algebra problem. **The solve itself is cheap.** | Its inputs require repeated model execution in the current implementation. |
| **Energy-matched random controls** | Help distinguish learned structure from generic perturbation or simply increasing edit magnitude. | Planning retains the full learned/random rank-probe family and common degeneracy rule, even when requesting rank 4 alone. |
| **Forecast correction target** | Has demonstrated modest prediction improvements. | It targets visual residuals; the headline improvements are proprioceptive MSE. Neither establishes better action ranking or physical outcomes. |

## Exactly what gets edited

Rank 4 combines four fitted patterns into **one distributed addition**, applied at B3—the fourth predictor block—at imagined step H3:

\[
\Delta x = c_1b_1 + c_2b_2 + c_3b_3 + c_4b_4.
\]

Each pattern spans the newest **256 patches × 400 features**. The 102,400 entries describe the activation field; they are not 102,400 separately attempted ablations. The current rank-4 arm does not edit all six predictor blocks. In planning, the intervention applies only to full H6 forecasts; shorter forecasts remain native. “One addition” means one per eligible candidate forecast, with candidates evaluated in batches and repeatedly during planning.

Sources: [basis construction and response solve](src/offline_study/support_operator.py), [planning adapter](src/offline_study/planning_support.py).

## Where the computational expense comes from

The expensive operation is specifically **32 probe forecasts per candidate**: positive/negative probes along eight learned and eight random directions. At 300 candidates, that means **9,600 probe trajectories per full-H6 batch**, executed in batches and repeated within planning. Native-shadow and final forecasts add further work.

The measured fit-action batch took approximately **102 seconds versus 2.9 seconds native**, about 35× the time on that workload. These are forecast-batch measurements, not an end-to-end task-performance or deployment benchmark. [Implementation and timing](reports/PLANNING_METHOD_ALIGNMENT.md#rank-transfer-implementation-and-completed-fit-checks-1235-edt).

The retained probe family supports the source comparison and common degeneracy rule. Removing it requires an explicitly specified successor or demonstrated equivalence; it cannot silently be reported as the same frozen operator.

**What deserves preserving is the compact, controlled correction effect. What needs redesign is its coefficient-construction cost and its connection to identifiable, decision-relevant information.**

The separate [Rank Edit](<Rank Edit.md>) note records three proposed replacements and the broader compute audit. This summary introduces no new measurements or experimental changes.
