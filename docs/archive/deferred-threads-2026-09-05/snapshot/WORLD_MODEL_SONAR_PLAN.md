# Frankenstein Sonar for latent world models

**Status:** active execution plan  
**Updated:** 2026-09-04  
**Scope:** frozen JEPA-WM/DINO-WM planning models only; improve ordinary robotics benchmark performance, not safety; no VLA-specific diffusion, action-expert, or language-token assumptions

This is the single source of truth for the active world-model experiment. Historical designs and failed branches are evidence, not instructions. The companion [paper-to-code audit](docs/WORLD_MODEL_MATH_CODE_AUDIT_2026-09-04.md) gives lower-level implementation findings; immutable manifests and result artifacts override status prose when they disagree.

## Objective

Test whether a bounded, distribution-aware intervention in a frozen latent world model improves native closed-loop robot-task success under fixed planner compute:

\[
\text{model-native coordinates}
\rightarrow
\text{subspace/regime distributions}
\rightarrow
\text{causal regime belief}
\rightarrow
\text{supported latent edit}
\rightarrow
\text{native behavioral effect}.
\]

The method family is **Frankenstein Sonar**. Mechanistic quantities build and falsify the controller; they do not substitute for improved behavior.

## Definition of the full Frankenstein recipe

The following are required modules in the research recipe, not a parking lot of optional ideas:

1. model-native relational coordinates;
2. probability geometry that can differ across factors/subspaces and regimes;
3. static multimodal regimes and causal temporal beliefs;
4. action-conditioned dynamics plus Markov-sufficiency falsification;
5. attention/Hopfield-style retrieval and Q/K distribution analysis;
6. covariance/conceptor and mixture-energy steering;
7. direction-versus-magnitude decomposition;
8. sparsity/superposition-aware feature support and controls;
9. matched-lure pattern separation for states requiring different successful actions;
10. Gaussian-OT and relational/Procrustes transport, kept mathematically distinct.

Every module must have an implementation, unit tests, a fit/development evaluation, a prespecified admission gate, and an ablation. **The experiment may not be described as a test of the full Frankenstein recipe while any module is merely mentioned in prose.**

Membership in the recipe is different from permission to deploy a module online. A module that fails its prospective performance/fidelity gate is recorded as implemented-and-rejected for this checkpoint/task and contributes an identity/no-op in the protected composition; it is not silently omitted or rescued on evaluation data. A module that remains unimplemented blocks a “full Frankenstein” claim, although simpler Sonar-core experiments may still be reported under narrower names.

Safety, collision, hazard, human-specific, obedience, and instruction-conflict objectives belong to archived branches. They are not labels, optimization targets, or secondary claims in this experiment.

## Non-negotiable experimental contract

1. The primary label is the released evaluator's native per-episode simulator success flag (`succ_def: simu`). No VLM judge or researcher-defined proxy may relabel an episode.
2. The primary comparison is the frozen checkpoint with Sonar versus that same checkpoint unsteered.
3. Every arm receives the same task, realized initial/goal state, environment seed, planner seed, episode order, horizon, CEM budget, and success definition.
4. Development, operator fitting, and protected evaluation use mutually disjoint episode pairs. Outcomes never cause episode replacement.
5. Regimes, bases, and sparse dictionaries are fitted without success/failure labels. Outcome labels enter only after representation discovery.
6. The protected evaluation set is opened once, after one primary recipe, dose, arm set, endpoints, and multiplicity rule are frozen.
7. A failed gate stops or simplifies the corresponding branch. It is never repaired by searching the protected set.

## Current behavioral experiment

### Active cell

- Model: released **DINO-WM** latent world model.
- Checkpoint: `mw_dino-wm.pth.tar`.
- Checkpoint SHA-256: `eab6e241f732db986f98ad224115ed7949ab4ff3f6cc8bc8f95b273cdb9e5d13`.
- Task: native MetaWorld Reach (`reach-v3-goal-observable`, manifest name `mw-reach`).
- Released configuration: `configs/evals/simu_env_planning/mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml`.
- Configuration SHA-256: `968aca1ae415a73be6d9ba72d8269c9422d513cc248dcc5f26b234618955b621`.
- Vendor commit: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- Planner: released CEM configuration: 15 iterations, 300 candidates, 10 elites, horizon 6, three actions executed per replan, 100 environment-step limit.
- Stimuli: simulator-generated initial/goal states and planner draws fixed by explicit `(environment seed, planner seed)` pairs; this is not a hand-curated image dataset.

### Frozen splits

Namespace: `panel-p-frankenstein-dino-reach-v1`.

| Split | Episodes | Purpose |
|---|---:|---|
| Development | 30 | Behavioral qualification, simple baselines, dose and recipe selection |
| Fit | 30 | Outcome-agnostic representation fitting followed by outcome-conditional operator fitting |
| Protected evaluation | 60 | One final paired behavioral comparison |

The stimulus manifests live in [docs/manifests/panel-p-frankenstein-dino-reach-v1](docs/manifests/panel-p-frankenstein-dino-reach-v1). The arm names in `manifest_meta.json` are a candidate inventory, not permission to run an intervention. A separate exclusive-write arm registry must be frozen after development gates pass.

### Status snapshot

The unsteered P0 run launched on 2026-09-04 and remains the only live Panel-P stage. Read the append-only remote `episodes.jsonl` and final `DONE.json` for counts rather than treating a partial number in a planning document as current. Partial outcomes are monitoring information, not a gate decision. No Panel-P activation intervention has run.

The prior JEPA-WM MetaWorld Reach-Wall cell finished 1/15 and failed class support. It is retired for this experiment; its protected set remains sealed. That is an operating-point failure, not evidence against Sonar.

## Registered model, checkpoint, and benchmark matrix

### Decision the matrix must support

The first question is not whether Sonar beats every published world model. It is whether one frozen checkpoint performs better when the qualified intervention is present than when the **same checkpoint** is unsteered. The replication matrix then asks whether that paired gain survives both a predictor-family change and a task/data change.

Only one released final checkpoint per model/environment is eligible. There is no training-epoch sweep and no checkpoint selection by observed intervention gain. Published success rates are context for expected headroom, not targets to reproduce and not criteria for changing seeds.

“Dataset” has two meanings that must stay separate:

- the offline trajectories used by the authors to train a frozen checkpoint; this project does not modify or retrain on them;
- the evaluation stimuli used for our causal comparison. MetaWorld creates initial and goal states in the simulator, whereas Push-T selects initial/goal material from the released dataset. In every case, paired arms reuse the exact realization and planner seed and verify realized-state hashes.

The released JEPA-WM study reports 12,600 MetaWorld episodes across 42 training tasks and 18,500 Push-T sequences/samples. Its evaluation uses simulator-derived initial states and expert goals for MetaWorld, dataset-derived goals for Push-T, and the native simulator success flag for both. The official repository releases matched DINO-WM and JEPA-WM environment checkpoints and planning code. See the [paper](https://arxiv.org/html/2512.24497) and [official release](https://github.com/facebookresearch/jepa-wms).

### Core 2 x 2 efficacy matrix

| Cell | Frozen model/checkpoint | Author training data | Closed-loop evaluation | Scientific role | Readiness |
|---|---|---|---|---|---|
| **C1** | DINO-WM `mw_dino-wm.pth.tar` | MetaWorld, 12,600 episodes/42 tasks | MetaWorld Reach; simulator initial state plus expert goal | Discovery cell and first protected causal test | P0 unsteered run active; no intervention has run |
| **C2** | JEPA-WM `mw_jepa-wm.pth.tar` | Same MetaWorld corpus | The exact C1 Reach realizations and planner seeds | Cross-model replication while holding task and visual encoder family fixed | Checkpoint present; matched Reach config/manifests and hook validation required |
| **C3** | DINO-WM `pt_dino-wm.pth.tar` | Push-T, 18,500 released sequences/samples | Push-T; released dataset-sourced initial/goal material | Cross-task/data replication of DINO-WM | Official config and checkpoint present; Push-T coordinate/lure adapter required |
| **C4** | JEPA-WM `pt_jepa-wm.pth.tar` | Same Push-T corpus | The exact C3 dataset items and planner seeds | Closes the model x task design | Official config and checkpoint present; JEPA hook plus Push-T adapter validation required |

The checkpoint hashes already verified on the execution host are:

| Checkpoint | SHA-256 |
|---|---|
| `mw_dino-wm.pth.tar` | `eab6e241f732db986f98ad224115ed7949ab4ff3f6cc8bc8f95b273cdb9e5d13` |
| `mw_jepa-wm.pth.tar` | `c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8` |
| `pt_dino-wm.pth.tar` | `8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f` |
| `pt_jepa-wm.pth.tar` | `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb` |

C1, C3, and C4 have author-released evaluation YAMLs. C2 is intentionally a matched-task configuration derived from the released JEPA-WM Reach-Wall YAML by changing only `task_specification.task` to `mw-reach`; `public_panel_eval.py` already records this as a deviation. That derived file must be materialized, reviewed, hashed, and behaviorally qualified before use. It must not inherit results or labels from C1.

The paper's aggregate reference rates are DINO-WM/JEPA-WM 44.8/58.2 on MetaWorld Reach and 66.0/70.2 on Push-T. They average repeated evaluations and training seeds rather than describing our exact downloaded file and seed stream. Our unsteered measurements in the pinned harness are the only baselines used for inference.

This core matrix is deliberately stronger than testing four unrelated models. Within each environment, DINO-WM and JEPA-WM use the same released data and DINOv2 ViT-S visual-encoder family with depth-six predictors; their action-conditioning/predictor recipes differ. Across environments, the matrix changes task geometry, stimulus source, and action dynamics. It therefore tests two distinct forms of transfer without importing VLA diffusion or language-token assumptions.

### Tests run in each core cell

| Test | C1 discovery | C2-C4 replication |
|---|---|---|
| Unsteered P0 and realization replay | Run first | Required independently; failure stops only that cell |
| Read-only capture identity | Exact action/outcome equality required | Same frozen tolerance |
| Model-native coordinates and probability/regime gates | Fit and select using C1 fit/development only | Refit local parameters with C1-selected families and frozen thresholds |
| Operator, site, and dose tournament | Allowed on C1 development only | No outcome-based reselection; use the frozen relative-depth site, recipe, and dose rule |
| All ten Frankenstein module decisions | Every module gets pass, fail, or explicit undecidable evidence | Apply the same gate functions; failed modules become registered no-ops |
| Protected behavioral comparison | 60 unused paired episodes | 60 unused paired episodes per eligible cell |
| Cross-model relational transport | C1 to C2 only after both local fits exist | Compare transported versus independently fitted C2 operator; likewise C3 to C4 |

The protected minimum arm set per cell is unsteered, identity, the frozen qualified Frankenstein composition, its orientation/action-dose-matched sham, the strongest frozen simple baseline, and the COAST-equation reference if numerically eligible. Static versus action-HMM routing is added only when all HMM gates and the minimum treatment-separation gate pass before protected evaluation.

### Splits, power, and family-level decision rule

C1 keeps its already frozen 30 development / 30 fit / 60 protected split. Future core cells use the same counts and disjoint cell-specific manifests. C1 and C2 intentionally share physical Reach stimulus IDs across models; C3 and C4 intentionally share Push-T dataset-item/planner IDs. The aggregate analysis resamples these shared stimulus IDs as blocks so cross-model reuse is not counted as independent evidence.

Every future P0 must pass the same 30-episode behavioral gate. Outcome-conditional local fitting additionally requires at least eight independent fit episodes in each outcome class; correlated replans do not count as extra episodes. A cell that passes behavioral P0 but lacks fit-class support can contribute only outcome-free descriptive geometry, not a Sonar intervention.

The first C1 protected result is a large-effect pilot. With 60 binary pairs, approximate 80% power is limited to an absolute effect around 17--22 percentage points when 25--40% of pairs are discordant. The complete four-cell matrix supplies 240 paired model-task episodes per arm and approximately 9--12-point sensitivity for the prespecified average effect under the same range. Exact inference uses paired outcomes, not these approximations.

The matrix-level primary estimand is the equal-cell-weighted paired risk difference for qualified Frankenstein Sonar versus unsteered. Use a hierarchical paired bootstrap that resamples stimulus IDs within task while retaining both models and all arms, plus an exact paired permutation/sign-flip check. Report per-cell risk differences and intervals, but reserve the unadjusted alpha for the one matrix-level primary contrast; use Holm correction for cell-specific efficacy claims and method comparators.

Matrix-level success requires all of the following:

- average paired success improvement at least +10 percentage points with a 95% interval above zero;
- nonnegative point estimates in at least three of four cells;
- no cell harmed by more than 10 points;
- no material violation of the fixed compute, support, cap-saturation, action-divergence, latency, or completion gates.

A broad cross-model/cross-task claim additionally requires at least three eligible cells spanning both model families and both task families. Fewer eligible cells can support only a cell-specific pilot claim. Failure of C1's behavioral or protected intervention gate triggers the reassessment ladder before any carpet-run of C2-C4.

### Deferred and excluded rows

| Row | Status and reason |
|---|---|
| MetaWorld Reach-Wall, JEPA-WM | Quarantined. The prior 1/15 run failed class support. It receives no intervention and cannot be reopened without a concrete harness/config root cause and a wholly preregistered rerun. |
| Wall with DINO-WM and JEPA-WM | Best secondary HMM/action-dynamics stress test: obstacle-mediated non-greedy planning, exact released paired configs, and checkpoint hashes `ff170be5...` / `8efb0623...`. It opens only after a valid C1 effect and a Wall-specific coordinate adapter. |
| PointMaze with DINO-WM and JEPA-WM | Ceiling/negative-control row, not a primary improvement benchmark: published reference rates are about 81.6/83.9. Useful later for testing whether steering harms an already strong planner. |
| DROID/RoboCasa and V-JEPA-2-AC | External-ecology replication only. They introduce DINOv3 or V-JEPA-2 encoders, much larger predictors, offline DROID endpoints or RoboCasa domain transfer, and different hook semantics. They do not belong in the first efficacy claim. |
| Orthogonal JEPA | Technique source, not a benchmark comparator. Its current paper evaluates newly trained state-based MuJoCo models and does not provide a compatible released visual-robotics checkpoint in the paper/release found on 2026-09-04. Use its factorization and magnitude-preservation ideas in our frozen-checkpoint gates; do not claim to outperform it. See [Orthogonal JEPA](https://arxiv.org/html/2608.20065). |
| Probabilistic-JEPA/HMM paper | Falsification source, not a checkpoint. Its claim ladder motivates filter/emission separation, innovation-history tests, and Chapman--Kolmogorov checks; it does not make a deterministic JEPA-WM automatically Markov-sufficient. See [the HMM/JEPA paper](https://arxiv.org/abs/2608.13621). |

The secondary released checkpoint hashes are retained for prospective use, not authorization to run: `wall_dino-wm` `ff170be5aec9249768be4a220d600b8f00a8589b2a78982ecf9273809f2767df`, `wall_jepa-wm` `8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e`, `mz_dino-wm` `b4dc463b5c7a1546c1e5edf2ef813005c136523fa373ba98bc207782860b3803`, and `mz_jepa-wm` `a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc`.

## Behavioral qualification before geometry

P0 passes only when all 30 development episodes are complete and:

- successes are at least 6;
- failures are at least 6;
- success rate is between 20% and 80%;
- both outcomes occur across at least two ordinal quartiles;
- every episode matches its manifest identity and realization-hash contract;
- every episode has at least two physical replans;
- there are no reset, capture, or completion errors.

If P0 fails, stop this cell and reassess the candidate rule. Do not change seeds, task difficulty, planner compute, horizon, or label to manufacture balance.

## The Frankenstein Sonar recipe

### 1. Capture the causal runtime population

Keep these axes distinct:

- physical replan `t`;
- CEM optimization iteration;
- candidate/elite index;
- predicted time/token position;
- transformer layer and hook site.

Fit data must match what the runtime hook edits: the same tensor location, pooling rule, token block, candidate structure, and physical replans. MetaWorld replans seven times in completed runs. Candidate rows within one CEM call are optimizer samples, not chronological HMM observations.

Only a common pre-outcome replan window may receive the terminal outcome label. Report early/late and divergence-time sensitivity because assigning a final episode outcome to early states remains imperfect credit assignment.

### 2. Test model-native coordinates before escalating complexity

Fit an outcome-agnostic coordinate map on fit episodes:

\[
x_t=\operatorname{diag}(s)^{-1}B^\top(h_t-c).
\]

Compare global whitened coordinates, PCA blocks, matched random-orthogonal blocks, learned predictive factors, and physically meaningful coordinate hypotheses. The Othello lesson is methodological: a representation that looks nonlinear, dense, or superposed in a human basis may become simple in the model's native relational basis.

For MetaWorld Reach, compare the same hand/goal variables in world, camera, and goal-relative frames with held-out whole episodes and shuffled-goal binding controls. A better readout nominates a coordinate system; it does not establish causal use.

An orthogonal rotation alone is not a contribution: a full Gaussian is rotation invariant. A factorization earns its role only through a registered restriction such as blockwise covariance, factor-specific dynamics, selective editing/protection, or improved held-out regularization.

### 3. Treat probabilities in different subspaces as distinct objects

Do not conflate:

\[
P(x^{(k)}\mid z=r),\qquad
P(z_t=r\mid x_{1:t}),\qquad
P(x^{(k)}\mid z=r,y),\qquad
P(j\mid i,z=r).
\]

They mean, respectively: a factor/subspace activation density, a causal belief over regimes, an outcome-conditional density, and an attention allocation. A conceptor score, Gaussian likelihood, HMM posterior, and attention softmax are not interchangeable probabilities.

The central distribution hypothesis is:

\[
P_{k,r}=P(x^{(k)}\mid z=r)
\]

may differ across factor blocks `k` and physical regimes `r` in center, covariance orientation, effective rank, tail shape, occupancy, and multimodality. This is stronger than saying that two PCA subspaces have different variance.

For every eligible block/regime, compare on held-out whole episodes:

- one Gaussian versus a static GMM;
- isotropic versus diagonal versus shrinkage-full covariance;
- held-out log score and calibration, not training likelihood;
- center displacement, principal angles/Grassmann distance, and regularized Gaussian Wasserstein/Bures distance;
- effective rank, conditioning, tail coverage, and episode-bootstrap stability;
- progress/event controls so regime geometry is not merely elapsed time.

If Gaussian calibration or tail coverage fails, use a bounded robust alternative on development data or fall back to a simpler operator. Covariance invertibility does not prove Gaussian adequacy.

For regime belief `b_t(r)`, the coherent mixture energy is

\[
E_y(x,b_t)=-\log\sum_r b_t(r)\exp[-E_{y,r}(x)],
\]

with responsibilities

\[
\rho_{y,r}(x,b_t)=
\frac{b_t(r)e^{-E_{y,r}(x)}}{\sum_jb_t(j)e^{-E_{y,j}(x)}}.
\]

Thus the edit weights depend on both temporal belief and how compatible the current activation is with each local density. Manually averaging local gradients with fixed weights is a separate ablation.

#### Attention as Hopfield-style retrieval

Transformer attention provides a useful retrieval interpretation, but it is not another name for the regime model. For one head,

\[
Q=HW_Q,\qquad K=HW_K,\qquad
A=\operatorname{softmax}(QK^\top/\sqrt{d_k}),\qquad O=AV.
\]

Modern Hopfield analysis interprets this softmax update as content-addressed associative retrieval. In this project, the bounded hypothesis is that regime-specific residual distributions induce regime-specific query, key, score, and retrieval distributions:

\[
P(h\mid z=r)\rightarrow P(Q,K,A,O\mid z=r).
\]

Measure whether a head amplifies regime separation using held-out distribution distances before and after its `W_Q/W_K` projections, and test attention-output restoration as mediation after a behavioral edit. Attention maps, entropy, or sharpness alone do not establish retrieval of a causal physical state. JEPA-WM/DINO-WM self-attention also is not homologous to LAVLA's action-decoder cross-attention, so LAVLA weighting is not copied literally.

### 4. Add temporal regimes only when sequence order earns them

Use a shared outcome-agnostic regime family for every local arm. Compare:

1. one Gaussian;
2. static `K=2` GMM;
3. exact progress/event bins;
4. Gaussian HMM with forward-backward joint transition posteriors and episode-equal weighting.

The online belief is causal:

\[
b_t(r)\propto p(x_t\mid r)\sum_qA_{qr}b_{t-1}(q).
\]

No future activations or smoothed posteriors may enter an intervention. HMM routing is eligible only if it beats the strongest static/progress alternative out of episode, has stable occupancy and transitions, improves outcome routing over static responsibilities with the same local operator bank, passes innovation-history and empirical Chapman-Kolmogorov tests, and produces a nontrivially different action treatment. Otherwise use static routing.

### 5. Test sparse superposition as a gated extension

The relevant hypothesis is not “later layers are dense.” It is:

\[
h_t=\mu_{z_t}+W_{z_t}f_t+\epsilon_t,
\qquad \|f_t\|_0\ll\dim(f),
\]

where an overcomplete, nonorthogonal dictionary can represent more possible features than residual dimensions, while only a small support

\[
S_t=\{i:f_{t,i}\ne0\}
\]

is active at one time. The testable object is regime-conditioned support:

\[
P(S_t\mid z_t=r).
\]

This module is now implemented in `public_panel_frankenstein_math.py` and `public_panel_frankenstein_gate.py` as an outcome-agnostic overcomplete sparse dictionary with the same deterministic ISTA encoder offline and online. The gate measures held-out reconstruction, next-state fidelity, executed-action readout fidelity, support frequency/entropy, matched PCA/random reconstruction, and split-half atom stability. `public_panel_steer.py --frankenstein-mode admitted` uses a passed support distribution as an additional abstention rule. It has not yet been fitted or admitted on the DINO-WM Reach capture, so no sparse behavioral claim exists.

If model-native coordinates plus Gaussian diagnostics leave reproducible interference or heavy tails, fit a post-hoc sparse dictionary/SAE without outcome labels. Before it can guide an edit, require:

- held-out activation reconstruction;
- next-state/predictive fidelity;
- downstream action-output fidelity after encode/decode;
- stable features under whole-episode resampling;
- noncollapsed activity and sensible feature-frequency distribution;
- improvement over matched PCA/random dictionaries.

Measure `P(f_i>0|z=r)`, feature magnitude, support entropy, regime-feature mutual information, and Jaccard/overlap of frequent supports. Intervene only on stable active features and compare against support-shuffled, dictionary-rotated, and wrong-feature controls.

Collateral behavioral effects do not prove superposition. They can also result from shared circuitry, nonlinear readout, off-manifold edits, regime mismatch, coordinate rotation, excessive dose, or bad token pooling.

Rectified `L_p`JEPA motivates the sparse-distribution hypothesis but its training objective is not copied into this frozen-checkpoint experiment.

#### Pattern-separation preservation

Pattern separation is distinct from sparsity. Construct matched **lure pairs**: states that are close in nuisance/visual geometry but require different successful actions or futures. With between-action and within-action scatter `S_B` and `S_W`, a regularized Fisher diagnostic solves

\[
S_Bv=\lambda(S_W+\epsilon I)v.
\]

The intervention should preserve the pre-edit separation or action margin of held-out lure pairs. This guards against a generic “move toward success” edit collapsing nearby states that demand different controls. The matched-lure constructor, regularized Fisher solve, action-label permutation null, split-half stability test, and runtime subspace-preservation projection are now implemented and unit tested. With a hash-bound coordinate report, nuisance matching uses audited simulator hand/goal geometry offline; otherwise the report explicitly marks a progress-only fallback. The projection activates only after the DINO-WM Reach gate passes; it has not yet been admitted behaviorally.

### 6. Fit outcome geometry after representation discovery

Outcome labels `y in {success,failure}` enter only now. Candidate operator families are:

- mean-difference vector;
- blockwise PCA/covariance direction;
- global or regime-local contrastive conceptors;
- success-minus-failure mixture-energy gradient;
- Gaussian optimal transport;
- activation patching when a valid on-manifold donor construction exists.

For a local contrastive conceptor:

\[
C_{y,r}=R_{y,r}(R_{y,r}+\alpha^{-2}I)^{-1},
\]

\[
D_r=C_{s,r}\wedge\neg C_{f,r},
\qquad
D_t=\sum_rb_t(r)D_r.
\]

Static and HMM Sonar must share the same `D_r`; only routing changes. A matched-spectrum sham preserves eigenvalues while scrambling orientation. Reports call the global comparator **COAST-equation reference**, not a faithful COAST reproduction.

The development phase chooses one primary operator. Frankenstein Sonar names the full structured hypothesis, not a requirement that every mathematical component be deployed.

### 6a. Keep two meanings of transport separate

**Gaussian optimal transport** maps a fitted failure distribution toward a success distribution within one coordinate system and is already an implemented candidate operator.

**Relational/Procrustes transport** asks whether the same functional geometry is expressed in rotated local coordinates across regimes, layers, checkpoints, or models:

\[
Q^*=\arg\min_{Q^\top Q=I}\|XQ-Y\|_F^2.
\]

For local operators, a passed alignment could support `D_s approximately Q^T D_r Q`. `public_panel_frankenstein_gate.py` now fits the map in matched factor coordinates with leave-one-episode-out scoring, and `public_panel_transport_bind.py` conjugates an admitted source operator into the target coordinate system while retaining the target model's routing, support, and dose geometry. Reconstruction, direction, covariance, and operator-orientation tests fix the row-vector convention. The transported arm still enters no protected comparison unless its paired-capture gate and development behavior beat independently fitted local operators.

### 7. Bound the edit in latent and behavioral geometry

For energy gradient `g`, use a capped metric step

\[
\delta=-\eta G^{-1}g
\]

with shrinkage, local-support rejection, a nuisance-whitened trust cap, and a downstream action-divergence term. Do not normalize every edit to the boundary: cap saturation on most states turns a confidence-sensitive field into a constant-length direction field.

Dose matching must report both latent metric norm and executed-action displacement. Every operator logs abstention, cap hits, support distance, residual norm, action divergence, planner compute, and latency.

## Experiment sequence

### P0 — behavioral substrate

Run the 30 unsteered development episodes and apply the fail-closed gate. No activation search or intervention occurs here.

### P1 — simple behavioral baselines

On development episodes only, run unsteered, identity, mean direction, PCA/covariance direction, matched random direction, equal-extra-compute CEM, and direct outcome/value reranking. This establishes whether the full stack beats obvious alternatives.

### P2 — full exploratory recipe

Capture fit-compatible activations, select coordinates/regimes without outcome labels, fit outcome operators, and implement/test all ten registered Frankenstein modules on fit/development data. Every module receives a pass/fail admission record and an ablation. The protected composition uses only modules that passed prospectively, but an incomplete module blocks the label “full Frankenstein.” Run stability, causal-fidelity, treatment-separation, and module-interaction checks before freezing the composition.

### P3 — freeze, protected evaluation, and simplification

Freeze one primary recipe and a separate arm registry. The minimum protected set is unsteered, identity, the selected Sonar arm, an orientation/dose-matched sham, the strongest simple baseline, and the COAST-equation reference if numerically eligible. Include static-versus-HMM only if HMM gates pass prospectively.

After a valid result, remove components systematically: HMM to static GMM to one Gaussian; full to diagonal to isotropic covariance; Mahalanobis to Euclidean/cosine; model-native to naive basis; sparse to dense; ordered to shuffled/reversed time; learned to matched-random geometry. The final method is the smallest object that retains the held-out behavioral effect.

Runtime ablations are executable rather than prose-only: `--frankenstein-ablate` independently disables probability support, sparse support, pattern protection, or model-native projection, while `--model-native-mode complement` runs the dose-matched orthogonal-complement control. Static/HMM, conceptor/energy/OT, angular/radial, attention restoration, and relational transport remain separate registered arms.

### P4 — locked cross-model and cross-task replication

Only after C1 yields a valid behavioral result or a prospectively defined promising-but-underpowered result, run C2--C4 in the registered order. Refit local distributions and operators on each cell's fit split, but freeze the algorithm, relative-depth site, hyperparameter/dose rule, module-gate thresholds, arm set, endpoint, and analysis code learned in C1. Do not replace an ineligible cell with Wall, PointMaze, Reach-Wall, RoboCasa, or a new checkpoint. The matrix-level result is reported after all eligible core rows complete; secondary rows cannot rescue it.

## Endpoint hierarchy

1. Primary: paired difference in native simulator success, selected Sonar versus unsteered.
2. Required preservation: compute/latency, action divergence, support distance, timeouts, and task-specific collateral behavior.
3. Method comparison: selected Sonar versus strongest simple baseline and eligible global COAST-equation reference.
4. Mechanism ablations: static versus HMM routing, local versus global geometry, density/covariance/sparse components.
5. Interpretation: coordinate meaning, regime meaning, attention mediation, cross-model transport.

Published JEPA-WM/DINO-WM success percentages are context, not acceptance thresholds. The experiment's baseline is the unsteered checkpoint measured in this exact harness.

## Idea coverage and status

| Idea | Role in the world-model study | Current status |
|---|---|---|
| Native simulator outcome | Primary label and utility endpoint | Active and implemented |
| Paired environment/planner seeds | Causal pairing and replay | Active and implemented |
| Model-native/Othello coordinates | Test whether relational coordinates simplify geometry | Reach and Reach-Wall adapters, fold-stable readout row space, runtime projection, and complement control implemented; DINO Reach decision pending |
| Predictive orthogonal factors | Candidate structured basis, never forced | Implemented with fallback |
| Different probability distributions by factor/regime | Central density hypothesis and mixture energy | Blockwise family selection, calibration/tail tests, regime KL, and runtime support gate implemented; DINO Reach decision pending |
| Full/diagonal/isotropic covariance | Tests whether anisotropy and interactions matter | Required development ablation |
| Static GMM versus progress versus HMM | Tests multimodality separately from temporal structure | Fixed/progress/action-HMM comparison and causal physical-replan lifecycle implemented; deployment remains gated on DINO Reach evidence |
| Action-conditioned transitions and Markov falsification | World-model-specific temporal validation | Implemented and unit/CUDA tested; DINO Reach fit decision pending |
| COAST soft conceptor geometry | Global comparator and possible local operator family | Implemented equation reference; not called faithful reproduction |
| Gaussian mixture energy | Confidence-sensitive local edit | Required module; implemented, with Gaussian adequacy still gated |
| Gaussian optimal transport | Distinct affine distribution comparator | Implemented development candidate |
| Sparse superposition and `P(S|z)` | Potentially selective feature-level routing/protection | Gate, matched controls, ISTA runtime support implemented; DINO Reach decision pending |
| Pattern separation on matched lures | Protect nearby states that require different successful actions from collapsing under a generic edit | Fisher/lure gate and runtime preservation implemented; DINO Reach decision pending |
| Attention/Hopfield retrieval and Q/K distributions | Tests whether self-attention retrieves and amplifies regime-relevant state | Bounded DINO Q/K/probability capture, retrieval gate, and paired clean-output mediation implemented; DINO Reach decision pending |
| Angular/radial decomposition | Tests direction versus magnitude | Required module; implemented development ablation |
| LAVLA cross-attention weighting | VLA/action-decoder-specific method | Not transferred literally |
| GIFT teacher losses | Training-time representation shaping | Excluded; only coordinate and angular/radial ideas transfer |
| CAFT | Weight-changing fine-tuning method | Excluded from frozen inference-time experiment |
| Relational/Procrustes transport | Align rotated local geometry across regimes/layers/models; distinct from Gaussian OT | Factor-space gate and runnable transported-operator binder implemented; paired model evidence pending |

## Code map

Active code is the `public_panel_*` family under [scripts/cgs_pilot](scripts/cgs_pilot). See its README for file-by-file status. The current critical path is:

1. `public_panel_manifest.py`
2. `public_panel_eval.py`
3. `public_panel_behavior_gate.py`
4. `public_panel_capture.py`
5. `public_panel_factor_gate.py` and `public_panel_coordinate_gate.py`
6. `public_panel_sonar_math.py` and `public_panel_sonar_fit.py`
7. `public_panel_fit_apply_audit.py`
8. `public_panel_steer.py`
9. `public_panel_development_gate.py`
10. `public_panel_arm_registry.py`

Legacy driving, RoboCasa safety, egg/cup, V-JEPA transfer, CAFT, and older auto-run shell pipelines are archived research branches. They must not launch or govern Panel P.

## Reassessment rule

If Sonar and the eligible geometry comparators fail against unsteered behavior, report the null and reassess the full chain:

1. Was there adequate behavioral headroom and power?
2. Was outcome-relevant information present before the outcome?
3. Did fit and runtime hook populations match?
4. Did the chosen coordinate system simplify held-out structure?
5. Were distributional assumptions calibrated?
6. Did the intervention create a meaningful but supported action change?
7. Did regime routing differ enough to test its hypothesis?

Do not answer a protected behavioral null by adding another operator to the same evaluation cohort.
