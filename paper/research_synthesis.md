# Research synthesis for the extended abstract

*September 8, 2026. Primary-paper review and local protocol/code audit. This document answers the research questions; proposed validations have not been run. See [Findings](../findings.md) for the concise paper-facing synthesis and [evidence appendix](appendix.md) for statistical limits.*

## 1. What hypothesis did we actually test?

The [experiment plan](../docs/EXPERIMENT_PLAN.md) states the objective: determine whether interventions in a frozen action-conditioned world model have reproducible, selective forecast effects, and whether useful effects survive candidate ranking and control. The labels below summarize its registered questions; they are not a claim that these exact hypothesis names were preregistered. The study tests a chain, and completion of one link does not establish the next.

| Question in the protocol | Why the chosen ablation discriminates | Completed evidence | Verdict and unresolved alternative |
|---|---|---|---|
| Is the useful correction compact? | Rank 1/4/8 changes capacity at fixed reference support, with a random basis at each rank and an equivalence/parsimony rule. | Original rank 4 helps both reaching tasks; Push-T is inconclusive. Reach rank 1 also passes eligibility. | Some compact corrections help. Four was a selected candidate in a bounded set, not an estimate of physical-state dimension or proof that smaller ranks fail. |
| Does spatial arrangement matter? | Permuting patch assignments preserves broad support and visual norm; 1/16/16/256 support arms separately test spatial concentration. | Reach joint beats permuted joint by 2.02% of native error; only all-patch spatial support passes its frozen gates. | Arrangement matters for these edits. We have not identified objects, contacts, or a semantic spatial code. Random location, covariance, and feature interactions need distinguishing before a physical explanation. |
| Must useful steering span depth? | B0-B5, B2+B3, and all-six arms hold total rank 1 and energy fixed. | Several singleton blocks help; Reach B0 improves 3.03%. | All-layer editing is unnecessary in this sweep. No unique causal layer or PEZ is identified; moving the rank-4 operator would be a new intervention. |
| Does vision-action coupling add useful structure? | Visual-only, internal action-conditioning-only, joint, equal-energy joint, and permutation arms separate pathway, energy, and alignment. | Push-T visual-only harms while action-only helps slightly. Reach equal-energy joint does not establish superiority to visual-only. Original combined rank/coupling beats its drop-one arms. | Pathways can oppose each other. Original combination evidence is positive but removal also removes energy; equal-energy mechanistic synergy is unproven. Raw simulator actions were not edited by the action-conditioning arm. |
| Does local nonlinear fidelity select better steering? | Linear/cubic reconstruction uses the same four noncentral anchors; projected/reflected controls interrogate curvature orientation. | Push-T reconstruction ordering reverses with precision; the cubic forecast endpoint is inconclusive. | This particular fidelity diagnostic is insufficient. One local action direction/radius does not test the whole latent manifold or disprove nonlinear steering. |
| Can the expensive response calculation be moved offline? | A separately specified successor replaces per-candidate response estimation with a mean-response map; native/zero/random arms remain. | BF16 H6 improvements: Reach 2.36%, Reach-Wall 2.19%; positive contrasts against matched random. | Supported on reused development trajectories. No direct old/new equivalence test, fresh confirmation, or average deployment-overhead estimate. |
| Do better forecasts support better decisions? | Frozen native CEM and paired scenario/condition evaluation keep the decision procedure comparable. | The full paired behavioral panel is incomplete in the checked local status. | Open. Recorded-action MSE can miss changes to candidate ordering or errors under planner-selected actions. |
| Should an expert/gate depend on history? | A history-based gate needs comparison with constant and memoryless gates under the same operator and budget. | Preparatory fitting exists; no completed efficacy claim in this paper. | Open. HMM preparation is not evidence of useful memory, routing, or MoE transfer. |

**How did we know these were the right ablations?** We knew what alternatives each could distinguish because the protocol held specified factors fixed and defined controls before those evaluations. We did not know which would be positive, and cannot establish that this was the globally optimal use of compute. The original online implementation retained response diagnostics that made planning expensive; sound comparisons did not excuse that engineering cost. A finite registered sweep is preferable to an unbounded search, but is still subject to development-set exposure and researcher choices.

The most useful remaining scientific validation is a prediction made from the mechanism hypothesis: for example, a specified spatial or pathway condition should predict when the edit helps on new scenarios. Another positive aggregate MSE alone would not explain the mechanism.

## 2. Why JEPA-WM, and what is frozen?

JEPA-WM gives this project released checkpoints, an action-conditioned predictor, native planning code, multiple task interfaces, and a published design-space study. These make a controlled intervention study practical: the base system and planner can stay fixed while the internal correction changes. This is a methodological justification supported by our pinned implementation, not a documented historical head-to-head proof that JEPA-WM was the best possible model. Its paper already studies prediction/planning mismatch, so that observation alone is not our contribution. [JEPA-WM, Sections 3-4 and G.3](https://arxiv.org/html/2512.24497v4)

| Stage/model | Teacher or EMA mechanism? | What this means for our project |
|---|---|---|
| JEPA-WM's world-model training | A pretrained visual encoder is frozen. The action encoder, optional proprio encoder, and predictor train. There is no moving EMA target-encoder update in the studied training recipe. | The target visual representation is fixed; it can loosely be called a fixed teacher, but it is not the online/EMA self-distillation loop. |
| Our intervention evaluation | All base-model modules are frozen; only separate correction artifacts were fitted offline. | We are not retraining a student or updating an EMA at steering time. |
| Earlier visual-encoder pretraining | DINOv2 and V-JEPA-family encoders have their own teacher/student pretraining lineage. | Loading a pretrained checkpoint does not run that pretraining machinery during predictor training or inference. |
| LeWM | Joint encoder/predictor training with prediction and SIGReg losses, without the EMA/stop-gradient anti-collapse scheme. | It changes how the world model is learned; it is not an update silently applied to our JEPA-WM checkpoint. |
| LeVJEPA | Video pretraining with a shared encoder/projector and invariance plus SIGReg, without target encoder, predictor, or stop-gradient. | Its removed predictor is the video-pretraining predictor, not evidence that action-conditioned dynamics are unnecessary for planning. |

Code evidence is explicit in the pinned [encoder initialization](../vendor/jepa-wms/app/vjepa_wm/utils.py), which sets encoder parameters non-trainable, and [training code](../vendor/jepa-wms/app/vjepa_wm/train.py), which constructs the trainable predictor/action/proprio modules. The optional checkpoint key `target_encoder` names weights saved by an upstream pretraining run; its presence is not an EMA update. A `.detach()` in a rollout also does not imply an EMA teacher. Our [fixed-response implementation](../src/offline_study/fixed_response.py) adds an activation hook to the existing predictor.

External checks: [JEPA-WM methodology](https://arxiv.org/html/2512.24497v4#S3), [DINOv2](https://arxiv.org/abs/2304.07193), [LeWM](https://arxiv.org/html/2603.19312v3), [LeVJEPA](https://arxiv.org/html/2608.27395v1). Teacher-free training is not a property of every method bearing the JEPA name.

## 3. What does our native-coordinate map look like?

At the current site, flatten the newest patch field to h in R^102400: 256 positions times 400 features. These are implementation coordinates. The internal block has mixed the input features; the last 16 channels should not automatically be interpreted as isolated physical-state variables merely because proprioception contributed 16 dimensions at input.

The implemented correction contains:

- A fitted mean and three PCA readout directions: z = P(h - mean) / scale.
- A four-vector output basis B in that same activation field, obtained from fitted activation structure.
- An intercept-plus-three feature vector phi = [1, z], a 4-by-4 coefficient matrix A, and dose normalization of B A phi before addition.
- Separate bindings to task, checkpoint, precision contract, and fitting population.

[Figure 6](figures/fig06_fitted_basis.png) visualizes the **actual saved basis**, not a hypothetical semantic map. Each heatmap sums squared loadings over the 400 features at each of 256 patches. The subspace-average map is invariant to rotations within that four-dimensional span. Individual direction maps are basis-dependent and are not comparable named physical axes across tasks. Their spatial mass is neither saliency nor the realized edit on a particular example.

We have a fitted linear coordinate system and a small correction map. We have not recovered a global intrinsic manifold, geodesic metric, canonical atlas, semantic concept dictionary, or circuit graph. PCA, maximum-covariance fitting, finite differences, and ridge inversion are established methods. The active steering operator is not a novel SAE or transcoder just because those words appear in the project's name or historical work. [ReFT](https://arxiv.org/abs/2404.03592) already provides low-rank representation adaptation in frozen language models; [Prisma](https://arxiv.org/abs/2504.19475) already supplies vision/video hooks and interpretability tooling.

To justify semantic coordinates later, a fitted axis or subspace must predict a specified physical variable on held-out examples, and interventions must change the targeted variable while preserving appropriate others. To justify a manifold claim, we would additionally need a fitted chart/metric and an on-manifold versus matched linear comparison. A PCA picture alone meets neither standard.

## 4. How this connects to Sonia's Physics Emergence Zone

Joseph et al. study physical information and interventions in **video encoders**, including V-JEPA 2 and VideoMAE-v2. Their PEZ is a layerwise emergence pattern in those encoders; their direction-steering results distinguish a readable low-dimensional projection from a larger causally effective subspace. Our work instead edits an **action-conditioned dynamics predictor downstream of a frozen DINOv2 encoder**. Our B3 is not an identified PEZ, and rank-4 error correction is not arbitrary control of a physical variable. [Interpreting Physics in Video World Models](https://arxiv.org/html/2602.07050v1)

The connection is a question: **can a distributed representation admit a compact correction for a particular error, even when full semantic control requires more dimensions?** Our reaching results are consistent with that possibility. They do not show that those four directions encode physics, or replicate her localization result. The next explanatory step is a specific read/write validation, not assigning physical names to the PCA directions.

The wider neuroscience connection is also substantive. Sadtler et al. manipulated mappings from neural population activity to cursor movement and studied learning constraints associated with the intrinsic manifold. This supplies experimental motivation for separating population geometry from the ability to control an output; it is not a demonstration about frozen artificial models. [Neural constraints on learning, Nature; author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC4393644/). Horoi, Lajoie, and Wolf similarly distinguish information in recurrent representations from its availability to an output through dynamics. [Internal representation dynamics and geometry](https://arxiv.org/abs/2001.03255)

## 5. Is circuit analysis required or useful for LLMs?

**Required for using, training, or demonstrating an effective steering method: no. Useful for explaining a particular internal computation: yes, when validated.** This is a judgment about the evidence needed for a claim, not a theorem that circuits are necessary or useless. A circuit is a selected subgraph of components and connections. A subspace intervention can be causal without revealing that entire graph.

| Prior work and validation | What it supports | What it does not buy us automatically |
|---|---|---|
| [Interpretability in the Wild](https://arxiv.org/abs/2211.00593): GPT-2 indirect-object identification, causal interventions, faithfulness/completeness/minimality checks. | Task-specific algorithms can be investigated beyond input/output correlations. | A complete explanation of arbitrary language behavior, or a requirement to map circuits before improving a model. |
| [Have Faith in Faithfulness](https://arxiv.org/html/2403.17806v2): tests EAP-IG against circuit-recovery baselines and evaluates recovered behavior. | Approximate attribution can reduce discovery cost; behavioral faithfulness matters more than graph overlap. | A similar-looking graph is not proof of the same mechanism, and an attribution score is not exact causal evidence. |
| [Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042): compares corruption and metric choices. | Intervention design materially changes localization conclusions. | A single ablation map does not uniquely locate a mechanism. |
| [Data-driven Circuit Discovery](https://arxiv.org/html/2605.09129v1): four task families, variations in syntax/complexity/domain, and mixed-task data. | Apparently faithful circuits can be dataset-specific; grouping computations can reveal distinct mechanisms. | One circuit per human task label, universal transfer, or an MoE architecture validated by circuit clustering. |

For our paper, distinguish three claims. An **efficacy claim** needs held-out outcomes and fair cost/control comparisons. A **selective-steering claim** also needs targeted changes, dose response, and preservation of other relevant variables. A **mechanistic explanation** needs evidence that the proposed internal mediator actually explains the effect, including appropriate removal/rescue or path interventions and generalization of that explanation. A comprehensive whole-model circuit is not mandatory for any narrower, clearly bounded claim.

The present data support controlled forecast interventions and some pathway/spatial specificity. They do not yet explain the physical computation. If tracing would not change the intervention choice, explain a failure, or predict a new outcome, its compute cost is difficult to justify here. A small causal test addressing the observed Reach/Push-T difference is more directly relevant than a full neuron-by-neuron search.

## 6. What would make this useful and efficient?

| Approach | Budget it targets | Implication for us |
|---|---|---|
| [LeWM](https://arxiv.org/html/2603.19312v3) | Learning a small model end-to-end with a simpler anti-collapse objective; inexpensive latent planning in the authors' setup. | A cheaper backbone may outperform the economics of repairing a large frozen one. We have not compared them at equal end-to-end budget. |
| [LeVJEPA](https://arxiv.org/html/2608.27395v1) | Video encoder pretraining, sparse observed tokens, simpler training components; authors report 5.6-20.8x lower compute in their matched-epoch comparison. | This is not a measured reduction in our CEM latency. Its causal encoder is a potential foundation for streaming models, requiring a compatible dynamics component. |
| [Fast-LeWM](https://arxiv.org/html/2606.26217) | Direct parallel predictions for action prefixes, replacing serial adjacent-state rollout; measured planner speed and success in its own setup. | Backbone/predictor architecture can remove costs that an activation hook cannot. This involves a trained dynamics-interface change. |
| Our fixed-response correction | Reusing a checkpoint and moving response estimation out of candidate forecasting. | Potentially cheap adaptation where an existing model must be retained; it does not remove native encoder, predictor, or CEM costs. |

The opportunity is **adaptation cost and controllable reuse**, not a claim that a large frozen model is intrinsically cheaper. A small world model plus an inexpensive correction could be complementary. Combining different methods' reported speedups by multiplication would be unjustified.

For an application budget, use T_total = T_calibration + N_decisions * (T_native_planner + T_edit_per_decision), with acquisition, inherited fitting, tuning, and failed runs separately disclosed. T_edit_per_decision includes every candidate scored by CEM. Compare with both leaving the model unchanged and adapting/training the strongest feasible smaller model. Count batch throughput, median/tail planner latency, memory, success, and useful control quality; a nominal parameter count is insufficient.

For the current D=102400 field, three dense projections plus four-direction expansion require approximately 7D+16 = 716816 multiply-accumulates per candidate application, before centering, scaling, normalization, copies, checks, and memory traffic. This arithmetic estimate is not a measured GPU latency. Storing mean, projection, and the learned basis in FP32 costs about 3.28 MB, plus small coefficients; the actual saved bank also includes random-control tensors and metadata. No dense D-by-D matrix is formed. At large CEM batches, memory traffic, synchronization, and batching can dominate this thin algebra.

The original cost was chiefly repeated world-model forecasts for estimating responses, including 32 online response probes in the retained diagnostic implementation. The successor performs zero such probes and no separate native shadow. The excluded engineering episode pair (314.28 s native, 313.03 s edited) validates integration on one scenario; it cannot establish speedup or average zero overhead. Offline calibration's approximately 100/95 seconds excludes the inherited basis/readout and research costs.

## 7. Architectural decisions and a plausible route to generalization

Already implemented: frozen base weights; explicit block/time/patch hooks; a task/checkpoint-bound bank; separation of offline response fitting from application; full candidate batching; FP32 coefficient math inside mixed-precision forecasts; dose auditing; unchanged native planner; explicit native/zero/random controls. These choices make a repeatable adaptation interface possible. They do not constitute a new base world-model architecture, nor do they remove data normalization, action-space, encoder, or simulator adapters.

A future multi-task system could have a shared frozen predictor, a small bank of correction experts, and a cheap router choosing one expert or identity from present state/action summaries. Experts should represent a validated correction regime, rather than merely memorizing task IDs. Top-1 application can retain one edit per forecast; evaluating every expert inside CEM would reintroduce the cost problem. This is a proposal, not an implementation or result.

MoE is not an unoccupied idea. Tong/Fan et al. investigate sparse experts in multimodal pretraining; that supports studying specialization at controlled active compute, not assuming it transfers to our predictor. [Beyond Language Modeling, Section 6](https://arxiv.org/html/2603.03276v1). [Branch-JEPA v3](https://arxiv.org/html/2607.05238v3) learns weighted sets of latent futures for multimodal prediction. Retaining several possible futures addresses uncertainty; selecting one small correction expert addresses adaptation cost. Averaging incompatible futures or edits can destroy their meaning. The paper was initially titled MoP-JEPA, but v3 substantially changes its framing and evaluation; use the current version rather than transplanting older headline numbers.

A defensible prospective comparison would hold active compute fixed and compare one shared correction, a state-based router with identity, and a parameter/budget-matched ungated alternative, then test unseen tasks or dynamics without selection on their outcomes. History-dependent routing additionally needs a memoryless comparator. This is a bounded research question, not authorization to begin another sweep.

What must generalize is the **procedure**: locate a useful site, fit/validate a compact correction using a bounded adaptation budget, and demonstrate prospective benefit. The same numerical basis need not transfer across independently trained checkpoints: latent coordinates can rotate or reorganize. Transfer claims should distinguish new scenarios, unseen tasks, new embodiments, and new independently trained models.

It is too broad to say JEPA models simply cannot scale. Representation pretraining, action-conditioned dynamics, long-horizon search, multimodal uncertainty, and task integration have different bottlenecks. Sparse experts address active capacity; cheaper encoders address representation cost; prefix prediction addresses rollout structure; standardized interfaces address engineering overhead. Our evidence currently addresses only a portion of adaptation and forecast evaluation.

## 8. Researchers and perspectives worth following

These are public research connections, not endorsements, collaborators, or people we have contacted. Status is based on the linked pages checked September 8, 2026; a lab's page may lag a career change.

| Researcher | Verified background/status | Why relevant |
|---|---|---|
| [Sonia Joseph](https://soniajoseph.github.io/) | McGill/Mila PhD candidate; FAIR JEPA researcher. Her [bio](https://www.soniajoseph.ai/about/) describes neuroscience/CS study and mouse visual-system research. | Direct physics/vision interpretability connection; readability versus causal control. |
| [Daniel Wurgaft](https://psychology.stanford.edu/people/daniel-wurgaft) | Stanford **Psychology PhD student**, cognitive area. | Co-first author of manifold steering; conceptual structure, behavior, and geometry through cognitive science. |
| [Can Rager](https://www.canrager.com/) | Independent interpretability researcher with a physics degree; his page does not claim current PhD enrollment. | Co-first author of manifold steering; geometry and causal intervention. |
| [Guillaume Lajoie](https://guillaumelajoie.com/) | Université de Montréal Mathematics and Statistics; neural computation research. | Dynamical systems, population geometry, and mathematical constraints on control/learning. |
| [Léo Choinière](https://guillaumelajoie.com/group/) | Listed as a neuroscience PhD student at Université de Montréal. | Brain-machine interfaces, motor control, and optimizing neurostimulation: practical intervention cost and target specificity. This is a methodological connection, not JEPA evidence. |
| [Colin Bredenberg](https://guillaumelajoie.com/group/) | Listed as a theoretical-neuroscience postdoc, co-supervised by Blake Richards. | Learning to control brain-computer interfaces; low-dimensional control and adaptation questions beyond CS. |

For this abstract, the closest direct papers remain JEPA-WM, Joseph et al., COAST, manifold steering, and ACPC. The additional efficiency/circuit/neuroscience papers sharpen what we can claim and what a future system would have to validate; they should not all be crammed into the four-page main text.
