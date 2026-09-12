# VLA / World-Model Representation Geometry — Research Thread Reconstruction for Codex

## Purpose

This document reconstructs the **actual reasoning thread** from the conversation rather than presenting only a cleaned-up glossary.

It is organized around:

1. **Repeated user questions / intuitions**
2. **The corresponding technical answer or correction**
3. **The mathematical object involved**
4. **Why it matters experimentally**
5. **Which paper introduced or motivated the idea**
6. **Which ideas are new hypotheses generated in this chat rather than claims from the papers**

The core research direction that emerged is:

\[
\boxed{
\text{mechanistic localization}
\rightarrow
\text{latent-state geometry}
\rightarrow
\text{dynamical regimes}
\rightarrow
\text{regime-aware causal control}
}
\]

The broader project began as mechanistic interpretability of VLAs/world models, but the discussion increasingly reframed it as a **world-model interpretation of the residual stream**.


---

# Prompt-to-answer map: recurring intuitions from the conversation

This section explicitly ties recurring questions/statements from the conversation to the technical answers they generated.

| Repeated prompt / intuition | Technical answer that emerged | Main mathematical object | Experimental implication |
|---|---|---|---|
| “The behavioral gate keeps failing before we even reach the intervention.” | The upstream contrast is not identifiable yet; task/checkpoint/stimulus design must precede representation geometry. | \(P(y\mid x,\text{checkpoint})\), success/failure sampling | Run behavioral eligibility sweeps before residual extraction. |
| “Maybe we’re overindexing on one safety mechanism.” | A human-interpretable mechanism need not equal the best control variable. | \(h_{\ell,p}\), causal intervention operator \(M\) | Separate representation, causality, control, and selective control claims. |
| “If a layer/token predicts behavior, why doesn’t steering it cleanly improve the benchmark?” | Decodability/causal use do not guarantee safe global steering; shared circuitry, superposition, regime mismatch, nonlinear readout, and off-manifold shifts can interfere. | \(h'=h+\alpha v\), \(h'=Mh\) | Add unrelated-task controls and state-specific interventions. |
| “Middle/later layers may be denser.” | Caucheteux/King supports greater brain alignment/contextualization in intermediate layers, not density or superposition. | representation similarity across \(\ell\) | Measure sparsity/effective rank separately. |
| “Are image and language tokens superimposed?” | They are typically different token positions in a compatible feature space; multimodal mixing via attention is distinct from feature superposition. | \(h_{\ell,p}\in\mathbb R^d\) | Preserve modality/token identity in hooks and analysis. |
| “Is attention basically memory?” | Better analogy: content-addressable/associative retrieval. Memory/state, KV cache, weights, and attention are distinct. | \(A=\mathrm{softmax}(QK^\top/\sqrt d)\) | Analyze routing separately from latent-state geometry. |
| “Pattern separation seems relevant.” | The defensible analogy is geometric: similar sensory inputs may become more separable when their future consequences differ. | \(D(h_A^{(\ell,p)},h_B^{(\ell,p)})\) | Build matched lure-like VLA stimuli. |
| “Energy seems like probability / entropy.” | Related but distinct: \(P\propto e^{-E}\) for probabilistic EBMs; JEPA energy can just be compatibility; entropy describes distributional uncertainty. | \(E(h)\), \(P(h)\), \(H(P)\) | Define the target energy explicitly before doing gradient steering. |
| “Maybe one Gaussian isn’t enough.” | Multiple local activation clouds motivate a GMM. | \(\sum_r \pi_r\mathcal N(\mu_r,\Sigma_r)\) | Test multimodality before assuming one global ellipsoid. |
| “How does HMM relate to the GMM?” | HMM adds sequential transition structure to mixture components. | \(P(z_{t+1}\mid z_t)\), \(P(h_t\mid z_t)\) | Compare HMM to static GMM/time-bin baselines. |
| “The HMM itself has a belief state?” | Yes: posterior uncertainty over current hidden regime. | \(b_t(r)=P(z_t=r\mid h_{1:t})\) | Enables soft rather than hard state-conditioned control. |
| “NextLat belief states remind me of Othello.” | Othello shows model-native state coordinates; NextLat motivates compact predictive sufficient states. | predictive latent \(h_t\), \(b_t\) | Ask what model-native physical-state coordinate system a VLA learns. |
| “Regime switching doesn’t feel like it’s just time.” | Correct nuance: regime identity can be geometric; switching is still sequential. | \(z_t\to z_{t+1}\), \(\mathcal M_r\) | Do not equate elapsed time bins with regimes. |
| “Sparsity may depend on temporal encoding.” | Sparsity itself is not temporal, but active feature support can be regime/time-conditioned. | \(S_t=\{i:f_{t,i}\neq0\}\), \(P(S_t\mid z_t)\) | Test support changes across physical regimes. |
| “COAST success geometry may be an average over local geometries.” | Plausible new hypothesis, not a COAST result. | \(C_{\rm global}\approx\sum_r P(r)C_r\) | Compare global vs regime-specific conceptors. |
| “Could the same feature rotate between regimes/layers?” | Yes in principle; raw vector cosine can collapse while relational structure is conserved. | \(Q\), CKA/RSA, principal angles | Test relational correspondence rather than only vector identity. |
| “What exactly is transport?” | Transfer relational structure across coordinate systems rather than raw coordinates. | \(Q^*=\arg\min_Q\|QX_A-X_B\|_F^2\) | Fit held-out anchor transport tests. |
| “Could we transport a COAST subspace?” | Proposed extension: \(C_B\approx QC_AQ^\top\). | transported covariance/conceptor | Test whether local control geometry is related by basis rotation. |
| “Maybe the project is becoming a Frankenstein monster.” | Reduce to one or two falsifiable links: first establish local regime geometry, then local-vs-global control. | \((\mu_r,\Sigma_r,C_r)\) | Defer HMM/energy/transport until simpler tests succeed. |

---


---

# 0. Critical notation: three axes that must never be conflated

For VLA analysis, represent an activation as:

\[
\boxed{
h_{\tau,\ell,p}
}
\]

where:

- \(\tau\): physical trajectory/control timestep;
- \(\ell\): Transformer layer / block;
- \(p\): token position.

These are three different axes.

## Token position \(p\)

Which sequence slot are we examining?

Examples:

- image patch token;
- language token;
- proprioceptive/state token;
- action token;
- action-expert latent.

Attention primarily moves information **across token positions**.

## Layer \(\ell\)

How far through the current forward pass are we?

\[
h_{\tau,1,p}
\rightarrow
h_{\tau,2,p}
\rightarrow
\cdots
\rightarrow
h_{\tau,L,p}.
\]

Transformer blocks transform representations across depth.

## Trajectory timestep \(\tau\)

Where is the robot in the rollout?

\[
\tau
\rightarrow
\tau+1
\]

means the model acted, the world changed, and another observation/control step followed.

This axis is where:

- robot/world dynamics;
- latent regimes;
- HMM transitions;
- belief-state updates;

naturally live.

### Critical correction from the discussion

\[
\boxed{
\text{layer transition}
\neq
\text{trajectory-state transition}
}
\]

A regime-switching hypothesis should not accidentally use Transformer layer index as if it were physical time.

---

# 1. Repeated question thread: “Why do our behavioral gates keep failing before the geometry experiment?”

## User intuition repeated across the conversation

The experiment repeatedly reached a point where:

- token/layer associations looked promising;
- some activation geometry could be measured;
- but the **behavioral gate** did not cleanly instantiate the hypothesis;
- therefore the intended causal/steering experiment became uninterpretable.

Examples of upstream failure modes discussed:

- wrong checkpoint;
- task at floor or ceiling;
- insufficient success/failure examples;
- stimulus manipulation not changing the intended variable;
- dataset not providing the needed contrast;
- intervention affecting unrelated tasks;
- behavior metric not matching the claimed mechanism.

## Answer that emerged

The central bottleneck may be **experimental identifiability**, not the sophistication of the intervention.

The correct ordering is:

\[
\boxed{
\text{controlled behavioral contrast}
\rightarrow
\text{representation measurement}
\rightarrow
\text{geometric hypothesis}
\rightarrow
\text{causal intervention}
\rightarrow
\text{ablation}
}
\]

not:

\[
\text{interesting geometry}
\rightarrow
\text{invent intervention}
\rightarrow
\text{look for behavioral change}.
\]

## Why COAST matters here

COAST explicitly needs both successful and failed rollouts to construct:

\[
C_{\text{success}}
\quad\text{and}\quad
C_{\text{failure}}.
\]

Its core empirical claim is not merely that a direction is decodable; it fits **success-critical residual subspaces** and then intervenes on them.

This motivates a formal “behavioral eligibility gate” before any mechanistic experiment.

## Proposed behavioral eligibility table from the chat

| Gate | Question | Example pass criterion |
|---|---|---|
| Model | Can the policy express the target behavior? | non-zero positive examples |
| Checkpoint | Do both relevant outcome classes exist? | avoid 0/100% floor/ceiling |
| Task | Does the task expose the intended causal contrast? | measurable effect |
| Stimulus | Is the manipulated variable controlled? | matched nuisance variables |
| Metric | Does the score actually track the hypothesized behavior? | manual sanity check |
| Sample | Are there enough examples in each class? | pre-specified minimum |
| Stability | Does the contrast survive new seeds? | replicated effect |

### Research-process lesson

If a gate fails, **do not extract residuals yet**.

The failure itself can become evidence about which task/checkpoint/stimulus regime is scientifically usable.

---

# 2. Repeated question thread: “Are we conflating mechanistic interpretability with steering/control?”

## User intuition

The project began as:

> Find a safety-relevant mechanism inside a VLA/world model.

It gradually became:

> Intervene on the residual stream and improve benchmark performance, Othello-style or COAST-style.

The user noticed that these may be **different scientific objectives**.

## Answer

Yes. The conversation separated four claims that had been implicitly fused:

\[
\boxed{
\text{interpretable}
=
\text{causal}
=
\text{selectively steerable}
=
\text{benchmark-improving}
}
\]

There is no reason all four must coincide.

A cleaner decomposition is:

## Claim A — Representation

A variable is represented at a particular layer/token position.

Example:

\[
h_{\ell,p}
\rightarrow
\text{lock state / contact / target / action phase}
\]

may be decodable.

## Claim B — Causal use

Activation patching or ablation shows the representation affects downstream computation.

\[
do(h_{\ell,p}\leftarrow h'_{\ell,p})
\Rightarrow
\Delta \text{prediction/action}.
\]

## Claim C — Control geometry

A particular residual-space transformation moves the policy toward better behavioral states.

This transformation does not need to be the most human-interpretable feature.

## Claim D — Selective control

The intervention improves the target behavior **without collateral degradation**.

This is a much stronger claim than causality.

### Key conclusion

\[
\boxed{
\text{causal representation}
\neq
\text{optimal steering variable}
}
\]

This became one of the most important conceptual corrections in the thread.

---

# 3. Historical foundation I: RNNs

## What is an RNN?

A recurrent neural network explicitly carries a hidden state across sequence time.

\[
h_t
=
\phi(W_xx_t+W_hh_{t-1}+b).
\]

Conceptually:

```text
x1 --> h1 --> h2 --> h3 --> ...
       ^      ^      ^
       x2     x3     x4
```

The hidden state compresses history.

If:

\[
P(x_{t+1}\mid x_{1:t})
\approx
P(x_{t+1}\mid h_t),
\]

then \(h_t\) is behaving like a predictive sufficient state.

## Why this mattered to the later discussion

This provides historical roots for:

- latent state;
- Markov state;
- belief state;
- world-model state;
- dynamical systems.

The later conceptor work was originally developed for **recurrent neural dynamics**, not Transformer steering.

---

# 4. Historical foundation II: PCA and covariance

Suppose activations are:

\[
h_1,\ldots,h_N\in\mathbb R^d.
\]

Stack them:

\[
X\in\mathbb R^{N\times d}.
\]

Mean:

\[
\mu
=
\frac1N\sum_i h_i.
\]

Center:

\[
\tilde X=X-\mu.
\]

Covariance:

\[
\boxed{
\Sigma
=
\frac1N
\tilde X^\top\tilde X
}
\]

PCA eigendecomposes:

\[
\boxed{
\Sigma=V\Lambda V^\top
}
\]

where:

- \(v_i\) = eigenvector / principal direction;
- \(\lambda_i\) = variance along that direction.

## Ellipsoid interpretation

For an approximately elliptical activation cloud:

- eigenvectors = orientation;
- eigenvalues = axis variance;
- \(\sqrt{\lambda_i}\) roughly controls axis scale.

This is why the conversation repeatedly returned to an ellipsoid.

## Important correction repeated in the thread

\[
\boxed{
\text{PCA ellipsoid}
\neq
\text{nonlinear manifold}
}
\]

A manifold can be curved or multimodal.

PCA gives a linear second-order summary.

---

# 5. Historical foundation III: conceptors

Herbert Jaeger introduced conceptors to control recurrent neural-network dynamics.

Given a covariance/correlation-like matrix \(R\):

\[
\boxed{
C
=
R(R+\alpha^{-2}I)^{-1}
}
\]

where \(\alpha\) is the aperture.

If:

\[
R=V\Lambda V^\top,
\]

then:

\[
C
=
V
\operatorname{diag}
\left(
\frac{\lambda_i}
{\lambda_i+\alpha^{-2}}
\right)
V^\top.
\]

Therefore conceptors preserve the covariance eigenvectors but transform the eigenvalues:

\[
\boxed{
\lambda_i
\rightarrow
\frac{\lambda_i}
{\lambda_i+\alpha^{-2}}
}
\]

This behaves like a soft spectral gate.

High-variance directions approach \(1\).

Low-variance directions approach \(0\).

## Useful interpretation

\[
\boxed{
\text{conceptor}
\approx
\text{soft covariance-aware subspace / ellipsoid operator}
}
\]

## Why this feels like PCA

PCA might hard-select a top-\(k\) eigenspace.

A conceptor smoothly weights the entire eigenspectrum.

---

# 6. COAST: why it became the central inspiration

## Paper

**Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States**  
Miranda Muqing Miao, Subin Kim, Brandon Yang, Lyle Ungar  
arXiv:2605.17144

## Core idea

Fit success and failure conceptors:

\[
C_s
\]

and:

\[
C_f.
\]

COAST uses Boolean conceptor algebra to construct a contrastive success-critical operator roughly corresponding to:

\[
\boxed{
\text{success AND NOT failure}
}
\]

Then steer the residual stream multiplicatively:

\[
\boxed{
h'=hM^\top
}
\]

with:

\[
M=(1-\beta)I+\beta C_{\text{steer}}.
\]

## Why this is richer than one steering vector

Ordinary activation steering:

\[
h'=h+\gamma v.
\]

COAST instead modifies a multidimensional covariance-weighted subspace.

The published result reports that outcome-relevant geometry is **low-rank but not rank-one**, and that failure structure is more shared across tasks than success structure.

## Why the chat repeatedly admired the paper

Not only because of the math.

The discussion repeatedly identified four strengths:

1. **clean behavioral setup**;
2. **checkpoint/task selection that yields both successes and failures**;
3. **clear metrics and ablations**;
4. **mechanistic analysis after demonstrating behavioral efficacy**.

This was framed as “PI-quality experimental identifiability.”

---

# 7. COAST’s research genealogy

The conversation initially made COAST seem like a paper that suddenly combined unrelated techniques.

A better historical picture emerged.

## Step 1 — Conceptors

Jaeger develops conceptors for dynamical RNN state-space control.

## Step 2 — Residual-stream steering

**CORAL: Correctness-Optimized Residual Activation Lens**  
Miranda Muqing Miao, Young-Min Cho, Lyle Ungar  
arXiv:2602.06022

This uses distributed correctness-related residual activations for inference-time steering/calibration.

## Step 3 — Multidimensional semantic steering

**Conceptors for Semantic Steering**  
Triantafyllopoulos et al.  
arXiv:2605.04980

Core argument:

\[
\text{semantic concept}
\neq
\text{necessarily one vector}
\]

and conceptors preserve a full multidimensional concept subspace.

## Step 4 — VLA application

COAST asks roughly:

> What if the same multidimensional conceptor machinery is applied to VLA action states and success/failure rollouts?

### Research-process lesson

The apparent “cross-disciplinary leap” was more like:

\[
\boxed{
\text{existing toolbox}
+
\text{new problem with matching structure}
}
\]

than random idea combination.

---

# 8. Repeated question thread: “How did COAST authors come up with it, and what did the PI probably contribute?”

## Answer developed in the conversation

We cannot know private PI conversations.

But the paper structure suggests experienced research supervision likely emphasized questions such as:

- What is the **minimal falsifiable claim**?
- Does the behavioral contrast exist?
- What is the negative control?
- Are fitting and testing separated?
- Is checkpoint choice creating floor/ceiling effects?
- Could random projection produce the same gain?
- Does the result depend on one layer?
- Does the method generalize across architectures?
- What outcome would falsify the geometric interpretation?

## Important research-process point

The polished paper presents:

\[
\text{hypothesis}
\rightarrow
\text{method}
\rightarrow
\text{mechanism}.
\]

But actual research often looks like:

\[
\boxed{
\text{idea}
\rightarrow
\text{pilot}
\rightarrow
\text{failure}
\rightarrow
\text{sweep}
\rightarrow
\text{weird observation}
\rightarrow
\text{new hypothesis}
\rightarrow
\text{ablation}
\rightarrow
\text{simplify}
}
\]

This led directly to the proposal to track **weird observations** from the current project rather than treating every failed run as wasted work.

---

# 9. Weird-observation discipline proposed in the chat

Maintain a table like:

| Observation | Expected | Actual | Possible explanation | Cheap discriminating test |
|---|---|---|---|---|
| behavioral gate fails | target contrast exists | floor/ceiling | wrong checkpoint | checkpoint sweep |
| feature decodes well | steering helps target | collateral damage | shared/superposed feature | unrelated-task controls |
| global intervention helps early | helps throughout | hurts later | regime-specific geometry | temporal/regime split |
| same concept direction unstable | fixed \(v\) | rotates | coordinate change | relational geometry test |
| centers stable | whole cloud changes | covariance rotates | subspace transport | Procrustes / Grassmann |
| high reconstruction | behavior preserved | action breaks | aggregation loses token info | per-token test |

This is intended as a process analogue to COAST’s preliminary sweeps/ablation logic.

---

# 10. Transformer attention refresher

Given hidden state matrix:

\[
H\in\mathbb R^{n\times d},
\]

compute:

\[
Q=HW_Q,\qquad
K=HW_K,\qquad
V=HW_V.
\]

Attention:

\[
\boxed{
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
}
\]

Output:

\[
O=AV.
\]

For token \(i\):

\[
o_i
=
\sum_j
a_{ij}v_j.
\]

Useful intuition:

- query = what information is this position looking for?
- key = what kind of information does another position advertise?
- value = what information is returned if selected?

This is an intuition, not a literal semantic decomposition.

---

# 11. Attention vs memory

A repeated user question was whether Transformer attention maps onto human memory.

The answer was:

\[
\boxed{
\text{attention}
\neq
\text{memory}
}
\]

A better computational analogy is:

\[
\boxed{
\text{attention}
=
\text{content-addressable information retrieval/routing}
}
\]

A useful decomposition:

| Transformer object | Memory-like interpretation |
|---|---|
| learned weights \(\theta\) | long-term statistical knowledge |
| prompt/context | currently available information |
| KV cache | cached representations of earlier positions |
| residual state \(h_{\ell,p}\) | current computation/state |
| attention | retrieval/routing mechanism |

The analogy to associative memory has mathematical support from **Hopfield Networks is All You Need**, which shows a modern Hopfield update equivalent to Transformer attention.

This should not be interpreted as:

> the Transformer implements hippocampal biology.

---

# 12. VLM structure

A VLM generally contains:

1. vision encoder;
2. image-to-language-space projection/adaptation;
3. language model;
4. multimodal fusion mechanism.

## Vision tokens

For a ViT-like encoder, an image is divided into patches:

\[
I
\rightarrow
I_1,\ldots,I_m
\rightarrow
v_1,\ldots,v_m.
\]

The patch embeddings are often called vision tokens.

## Language tokens

Text:

```text
"pick up the red cup"
```

becomes token embeddings:

\[
e_1,\ldots,e_k.
\]

## Fusion

Architectures vary:

- concatenated vision + text token sequence;
- cross-attention;
- visual resamplers / latent queries;
- interleaved multimodal architectures.

---

# 13. Repeated question: “Are vision and language tokens superimposed?”

## Correction

Not in token-position space.

Vision token \(p_1\) and language token \(p_2\) are separate rows:

\[
h_{\ell,p_1}
\neq
h_{\ell,p_2}.
\]

But they may share a common feature dimension:

\[
h_{\ell,p}
\in
\mathbb R^d.
\]

Attention can mix cross-modal information.

After several layers, a language token such as `"cup"` may contain:

- lexical semantics;
- visual referent;
- position;
- color;
- task relevance.

## Separate concept: feature superposition

Inside any one vector:

\[
h_{\ell,p},
\]

many latent features can share the same finite-dimensional space.

Therefore:

\[
\boxed{
\text{multimodal integration}
\neq
\text{feature superposition}
}
\]

---

# 14. VLA structure

A generic VLA maps:

\[
\boxed{
\text{vision}
+
\text{language}
+
\text{robot state}
\rightarrow
\text{actions}
}
\]

Common inputs:

- image/video;
- instruction;
- proprioception;
- gripper state;
- joint state;
- action history.

## OpenVLA example

**OpenVLA: An Open-Source Vision-Language-Action Model**  
Kim et al., arXiv:2406.09246

OpenVLA combines visual features (DINOv2 + SigLIP) with a Llama-based language backbone and predicts robot actions through the VLA sequence framework.

## \(\pi_0\) example

**\(\pi_0\): A Vision-Language-Action Flow Model for General Robot Control**  
Black et al., arXiv:2410.24164

Important difference:

- pretrained VLM backbone;
- separate robotics-specific **action expert**;
- continuous action chunks;
- flow matching rather than ordinary autoregressive text-token generation.

### Why this matters for mechanistic work

“The VLA residual stream” may be underspecified.

One must distinguish:

- VLM stream;
- action-expert stream;
- visual pathway;
- action latents/tokens;
- output/control module.

The exact intervention hook matters.

---

# 15. Hippocampal pattern separation

A recurring user analogy came from the Mnemonic Similarity Task (MST).

At a high level:

\[
x_A\approx x_B
\]

but a pattern-separation computation makes their internal representations more distinguishable:

\[
h_A\not\approx h_B.
\]

The MST uses:

- repeated targets;
- highly similar lures;
- novel foils;

to place strong demands on hippocampal pattern separation.

Important nuance:

\[
\boxed{
\text{MST behavioral lure discrimination}
\neq
\text{direct measurement of the neural algorithm}
}
\]

It is a behavioral assay designed to tax that computation.

---

# 16. Proposed Transformer/VLA pattern-separation analogy

Suppose two observations are almost visually identical:

### A
Gripper is aligned enough for stable grasp.

### B
Gripper is only slightly displaced but will fail.

Early layer:

\[
\cos(h_A^{(2)},h_B^{(2)})
\approx0.98.
\]

Later layer:

\[
\cos(h_A^{(15)},h_B^{(15)})
\approx0.55.
\]

If the divergence appears **before** the behavioral difference and predicts the correct downstream action, this resembles computational pattern separation:

\[
\boxed{
\text{similar observation}
\rightarrow
\text{behaviorally distinct latent state}
}
\]

This does **not** imply an anatomical homology such as:

\[
\text{Transformer layer}
=
\text{dentate gyrus}.
\]

---

# 17. Novel experiment generated in the chat: VLA pattern-separation test

Construct matched lure-like robot stimuli.

At every relevant:

\[
(\ell,p),
\]

measure:

\[
D_{\ell,p}
=
D(
h_A^{(\ell,p)},
h_B^{(\ell,p)}
).
\]

Candidate metrics:

- cosine distance;
- Euclidean distance;
- Mahalanobis distance;
- probe margin;
- CKA/RSA;
- covariance overlap;
- principal-angle / Grassmann distance.

Main question:

\[
\boxed{
\text{Do near-identical observations with different future consequences}
\text{ become selectively separated across the forward pass?}
}
\]

This idea was generated in the conversation; it is not claimed as a result of the cited papers.

---

# 18. Middle-layer neuroscience discussion

## Paper

**Brains and algorithms partially converge in natural language processing**  
Charlotte Caucheteux & Jean-Rémi King  
Communications Biology (2022)

The important result discussed:

- brain scores across Transformer depth show an inverted-U pattern;
- intermediate layers align better with measured brain responses than input/output layers;
- middle representations are more contextualized than raw input embeddings.

## Repeated user statement

> Middle/later layers might be “denser” or more superposed.

## Correction

The Caucheteux/King result does **not** establish:

\[
\text{middle layers}
=
\text{denser representations}
\]

or:

\[
\text{middle layers}
=
\text{more superposition}.
\]

“More contextual/compositional” is a different property.

Possible combinations include:

\[
\text{high compositionality + sparse code}
\]

or:

\[
\text{high compositionality + dense superposition}.
\]

Density/sparsity/superposition must be measured separately.

---

# 19. Superposition

A stylized representation model:

\[
h=Wf
\]

where:

- \(h\in\mathbb R^d\) is the residual activation;
- \(f\in\mathbb R^m\) are latent features;
- \(m\gg d\).

Many possible features share a finite-dimensional residual space.

That is superposition.

Feature directions need not be orthogonal:

\[
w_i^\top w_j\neq0.
\]

This can create intervention interference.

---

# 20. Sparsity

Sparsity means few latent features are strongly active at once.

\[
\|f_t\|_0
\ll
m.
\]

This may make superposition more tractable.

## Repeated question

> Does sparsity depend on time / temporal encoding?

## Answer

Not intrinsically.

Sparsity is a property of the feature code at a particular state.

But the active support can evolve:

\[
S_\tau
=
\{i:f_{\tau,i}\neq0\}.
\]

And it may be **regime-conditioned**:

\[
\boxed{
P(S_\tau\mid z_\tau)
}
\]

Example:

Approach:

\[
S_A
=
\{\text{distance},\text{target location},\text{alignment}\}
\]

Contact:

\[
S_C
=
\{\text{contact},\text{force},\text{grasp stability}\}
\]

Transport:

\[
S_T
=
\{\text{goal},\text{grasp retention},\text{trajectory correction}\}
\]

Therefore:

\[
\boxed{
\text{sparsity}
\neq
\text{temporal encoding}
}
\]

but temporal/regime state can determine which sparse features are active.

---

# 21. Rectified \(L_p\)JEPA connection

## Paper

**Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations**  
Kuang et al., arXiv:2602.01456

The paper replaces isotropic-Gaussian distribution matching with a Rectified Generalized Gaussian target via RDMReg.

Important ideas:

- sparse non-negative representations;
- explicit control over expected \(\ell_0\) sparsity;
- maximum-entropy property under expected \(\ell_p\) constraints;
- distribution matching rather than just pointwise feature steering.

## Why it entered this conversation

It provides a direct bridge between:

- representation distributions;
- sparsity;
- entropy;
- JEPA-style latent representation learning.

It does **not** imply that every residual activation has a literal physical “activation energy.”

---

# 22. Energy, probability, and entropy: repeated confusion resolved

The user repeatedly linked:

- energy;
- probability;
- entropy;
- Gaussian activation distributions.

There is a real relationship, but they are not synonyms.

## Probabilistic energy model

\[
\boxed{
P(x)
=
\frac{e^{-E(x)}}{Z}
}
\]

so lower energy corresponds to higher probability density.

For a Gaussian:

\[
P(h\mid r)
=
\mathcal N(\mu_r,\Sigma_r),
\]

negative log-density yields:

\[
\boxed{
E_r(h)
=
\frac12
(h-\mu_r)^\top
\Sigma_r^{-1}
(h-\mu_r)
+
\frac12\log|\Sigma_r|
+
\text{const}
}
\]

The first term is a Mahalanobis-distance energy.

## JEPA-style energy

JEPA often uses “energy” more generally as a compatibility/prediction discrepancy between representations.

Thus:

\[
\boxed{
\text{low JEPA energy}
\neq
\text{explicit calibrated probability}
}
\]

## Entropy

Entropy is a property of a distribution:

\[
H(P)
=
-\mathbb E_{x\sim P}
[\log P(x)].
\]

It quantifies uncertainty/spread, not pointwise energy itself.

### Compact distinction

\[
\boxed{
\text{energy}
=
\text{compatibility score}
}
\]

\[
\boxed{
P\propto e^{-E}
\quad
\text{in a probabilistic EBM}
}
\]

\[
\boxed{
\text{entropy}
=
\text{uncertainty/diversity of the distribution}
}
\]

---

# 23. Gaussian mixture model

If one activation cloud is insufficient:

\[
h
\sim
\sum_{r=1}^K
\pi_r
\mathcal N(\mu_r,\Sigma_r).
\]

This is a GMM.

It represents multiple local ellipsoids.

Possible human labels:

- approach;
- alignment;
- contact;
- grasp;
- transport;
- recovery.

But these labels should not be assumed to match the model-native state.

## Critical distinction

A GMM is static.

It does not use ordering.

---

# 24. Markov chains

A first-order Markov state satisfies:

\[
\boxed{
P(z_{t+1}\mid z_{1:t})
=
P(z_{t+1}\mid z_t)
}
\]

This does not mean history is irrelevant.

It means all relevant history has been compressed into \(z_t\).

This is why Markov-state language connects naturally to learned world-model states.

---

# 25. HMM

An HMM introduces a hidden Markov state:

\[
z_t.
\]

Observed activations:

\[
h_t.
\]

Transitions:

\[
P(z_{t+1}\mid z_t).
\]

Emissions:

\[
P(h_t\mid z_t).
\]

For Gaussian emissions:

\[
h_t\mid z_t=r
\sim
\mathcal N(\mu_r,\Sigma_r).
\]

Thus:

\[
\boxed{
\text{GMM}
=
\text{multiple geometries}
}
\]

while:

\[
\boxed{
\text{HMM}
=
\text{multiple geometries + sequential transition structure}
}
\]

---

# 26. HMM belief state

Because \(z_t\) is hidden:

\[
\boxed{
b_t(r)
=
P(z_t=r\mid h_{1:t})
}
\]

Example:

\[
b_t
=
[0.05,\;0.70,\;0.20,\;0.05].
\]

This is uncertainty over the current hidden regime.

New evidence updates the posterior.

---

# 27. NextLat and belief states

## Paper

**Next-Latent Prediction Transformers Learn Compact World Models**  
arXiv:2511.05963

Core idea discussed:

Standard next-token Transformers can repeatedly consult full context.

Nothing necessarily forces one compact internal state to summarize history.

NextLat adds next-latent prediction:

\[
(h_t,x_{t+1})
\rightarrow
\hat h_{t+1}.
\]

The paper argues that the learned latents converge toward **belief-state-like compact representations** sufficient for predicting the future.

The conceptual target is:

\[
\boxed{
P(\text{future}\mid\text{history})
\approx
P(\text{future}\mid h_t)
}
\]

## Important nuance

The learned vector does not need to literally contain a normalized probability table.

“Belief state” can mean a predictive sufficient statistic of history.

---

# 28. Othello and model-native coordinate systems

## Paper

**Emergent Linear Representations in World Models of Self-Supervised Sequence Models**  
Neel Nanda, Andrew Lee, Martin Wattenberg  
arXiv:2309.00941

A central result discussed:

The board is more naturally linearly represented using:

\[
\boxed{
\{\text{mine},\text{yours},\text{empty}\}
}
\]

rather than the human-default:

\[
\{\text{black},\text{white},\text{empty}\}.
\]

This is a model-native relational coordinate system.

Causal vector interventions can alter move predictions.

## Why this matters for VLAs/world models

Human labels such as:

- approach;
- contact;
- grasp;
- failure;

may not be the actual internal coordinates.

A stronger research question is:

\[
\boxed{
\text{What coordinate system does the VLA use to encode its belief about physical state?}
}
\]

---

# 29. Othello + NextLat connection generated in the chat

Othello can be viewed as a particularly clean “belief-state” example because the move history almost fully determines the board.

The model needs an internal state sufficient for future legal-move prediction.

The interesting part is **how that state is parameterized**:

\[
\text{mine/yours}
\]

rather than:

\[
\text{black/white}.
\]

The VLA/world-model analog is harder because partial observability creates real uncertainty.

Thus the sequence:

\[
\boxed{
\text{Othello model-native state}
\rightarrow
\text{NextLat belief state}
\rightarrow
\text{VLA latent physical state}
}
\]

became a major conceptual bridge in the conversation.

This synthesis is a chat-generated interpretation, not a direct claim of either paper.

---

# 30. Regime switching: repeated correction

## User intuition

> Maybe regime switching isn't really switching in time; maybe it is switching in geometry.

## Correction

A **regime can be defined geometrically**:

\[
h_t\in\mathcal M_r.
\]

But regime **switching** is sequential:

\[
z_t
\rightarrow
z_{t+1}.
\]

Time orders the transitions.

Absolute elapsed time does not define the regime.

Two robot trajectories may enter “contact-like” geometry at different physical timestamps.

### Key distinction

Geometry answers:

> What regime does this state resemble?

Dynamics answers:

> How do regimes transition?

---

# 31. Switching dynamical system

A richer model than a simple HMM has:

- discrete regime \(z_t\);
- continuous state \(h_t\).

For example:

\[
\boxed{
h_{t+1}
=
A_{z_t}h_t
+
B_{z_t}a_t
+
\epsilon_t
}
\]

Different regimes have different local dynamics:

\[
A_{\text{approach}}
\neq
A_{\text{contact}}
\neq
A_{\text{transport}}.
\]

This is closer to robotics/world-model reasoning than a static mixture model.

---

# 32. Novel unified representation model generated in the chat

The conversation proposed:

\[
\boxed{
h_t
=
\mu_{z_t}
+
W_{z_t}f_t
+
\epsilon_t
}
\]

where:

- \(z_t\): latent regime;
- \(\mu_{z_t}\): regime center;
- \(W_{z_t}\): local feature basis;
- \(f_t\): sparse feature coefficients.

This one equation unifies:

- center;
- covariance;
- subspaces;
- sparse coding;
- superposition;
- regime switching.

If:

\[
W_A\neq W_B,
\]

the same semantic/world variable can be encoded by different directions in different regimes.

This was generated as a conceptual model in the chat; it is not a published result.

---

# 33. Center + covariance decomposition

A repeated thread questioned whether COAST-style covariance steering ignores the **center** of the activation cloud.

For a state distribution:

\[
(\mu,\Sigma)
\]

the two simplest moments are:

\[
\boxed{
\mu
=
\text{where the cloud is}
}
\]

and:

\[
\boxed{
\Sigma
=
\text{how it is oriented/spread}
}
\]

This motivated a clean experiment:

### A
No intervention

### B
Mean/center intervention only

### C
Covariance/subspace intervention only

### D
Center + covariance

### E
Matched random control

An affine transformation around a success center can be written:

\[
\boxed{
h'
=
\mu_s
+
M(h-\mu_s)
}
\]

This turns “the center might matter” into a falsifiable:

\[
\boxed{
\mu
\quad\text{vs}\quad
\Sigma
}
\]

question.

---

# 34. Global vs regime-conditioned COAST

The conversation proposed that one global success conceptor may pool several local success geometries.

Hypothesis:

\[
\boxed{
C_{\text{global}}
\approx
\sum_r
P(r)C_r
}
\]

where:

\[
C_r
\]

is a regime-specific control geometry.

This suggests:

\[
C_{\text{approach}},
\quad
C_{\text{contact}},
\quad
C_{\text{transport}}
\]

may differ.

A global intervention might help one phase and hurt another.

This became the proposed explanation for some collateral steering effects.

This is a **novel chat-generated hypothesis**, not a result established by COAST.

---

# 35. Belief-conditioned steering

If regime identity is uncertain, use the belief state:

\[
b_t(r)
=
P(z_t=r\mid h_{\le t}).
\]

Instead of selecting one hard operator:

\[
M_{z_t},
\]

construct:

\[
\boxed{
M_t
=
\sum_r
b_t(r)M_r.
}
\]

Example:

\[
b_t
=
[0.2,0.8]
\]

gives:

\[
M_t
=
0.2M_A
+
0.8M_B.
\]

This is a **soft belief-conditioned steering policy**.

Novel chat-generated idea.

---

# 36. Belief-conditioned energy intervention

Similarly define local energies:

\[
E_r(h).
\]

Then:

\[
\boxed{
E(h_t,b_t)
=
\sum_r
b_t(r)E_r(h_t)
}
\]

and intervene:

\[
\boxed{
h_t'
=
h_t
-
\eta
\nabla_h E(h_t,b_t).
}
\]

This connects:

- HMM belief state;
- local activation geometry;
- probabilistic/energy-based intuition;
- causal residual intervention.

Again, this is a research proposal generated in the conversation.

---

# 37. Attention and latent regime geometry

Suppose:

\[
P(h\mid A)
\neq
P(h\mid B).
\]

Since:

\[
Q=HW_Q,\qquad K=HW_K,
\]

different residual geometry can produce different query/key geometry.

Then:

\[
\boxed{
z_t
\rightarrow
h_t
\rightarrow
Q_t/K_t
\rightarrow
A_t
\rightarrow
\text{downstream computation}
}
\]

is a plausible causal chain.

Important: this chain must be experimentally established, not assumed.

---

# 38. Three distinct probability distributions that were repeatedly conflated

## Attention allocation

\[
\boxed{
P(j\mid i)
}
\]

via softmax attention.

This means:

> given query token \(i\), which key position \(j\) receives weight?

## Activation density

\[
\boxed{
P(h\mid z=r)
}
\]

This describes where activation vectors occur in representation space for regime \(r\).

## Belief over hidden regime

\[
\boxed{
P(z_t=r\mid h_{1:t})
}
\]

This is uncertainty about the latent state.

These are related but mathematically different.

---

# 39. Relational transport

## Paper

**Language Models Represent and Transform Concepts with Shared Geometry**  
Zhimin Hu, Lanhao Niu, Sashank Varma  
arXiv:2607.04525

The paper treats:

- concepts as contextual point-cloud representations;
- contextual transformations as structured displacement fields.

A concept-specific contextual displacement can be written:

\[
\phi^{(m)}(w,\tau)
=
r^{(m)}(w,\tau)
-
r^{(m)}(w,\tau_0).
\]

The key empirical idea:

> raw coordinates can differ across models, while relational displacement structure remains predictive across models.

## Important correction from the conversation

“Transport” here is **not necessarily literal differential-geometric parallel transport**.

The useful idea is:

\[
\boxed{
\text{do not transfer coordinates directly;}
\text{ transfer relational structure}
}
\]

---

# 40. Why transport became relevant to VLA regimes

Suppose a physical concept such as “alignment” is represented as:

\[
v_{\text{align}}^A
\]

in regime \(A\), but:

\[
v_{\text{align}}^B
\]

in regime \(B\).

It is possible that:

\[
\cos(v_{\text{align}}^A,v_{\text{align}}^B)
\approx0
\]

even though the concept has not disappeared.

The local coordinate system may have rotated.

The better question becomes:

\[
\boxed{
\text{Is the relationship between alignment and other landmarks preserved?}
}
\]

For example, relative relationships to:

- target distance;
- force;
- object position;
- contact;

may remain structurally similar.

---

# 41. Relational-weighting transport sketch

Suppose in source regime \(A\), an unknown feature has similarities:

\[
s^A=
[0.8,\;0.6,\;-0.1]
\]

to three anchor features.

If corresponding anchors in regime \(B\) are:

\[
v_1^B,v_2^B,v_3^B,
\]

predict:

\[
\boxed{
\hat v^B
\propto
0.8v_1^B
+
0.6v_2^B
-
0.1v_3^B.
}
\]

The goal is not to preserve raw coordinates.

The goal is to preserve relational structure.

---

# 42. Procrustes transport extension generated in the chat

Given matched anchor matrices:

\[
X_A
=
[v_1^A,\ldots,v_k^A]
\]

and:

\[
X_B
=
[v_1^B,\ldots,v_k^B],
\]

find an orthogonal map:

\[
\boxed{
Q^*
=
\arg\min_Q
\|QX_A-X_B\|_F^2
\quad
\text{s.t.}
\quad
Q^\top Q=I.
}
\]

Then:

\[
v_B
\approx
Qv_A.
\]

Orthogonal maps preserve:

\[
\|v\|,
\]

\[
v_i^\top v_j,
\]

\[
\cos(v_i,v_j),
\]

and angles.

This gives a concrete way to model a local basis rotation.

---

# 43. Transporting a COAST-style subspace

If regime \(A\) has conceptor:

\[
C_A,
\]

and the local coordinate change is \(Q\), then propose:

\[
\boxed{
\hat C_B
=
QC_AQ^\top.
}
\]

Likewise:

\[
\boxed{
\hat\Sigma_B
=
Q\Sigma_AQ^\top.
}
\]

With center transport:

\[
\mu_A
\rightarrow
\mu_B.
\]

The affine state map is:

\[
\boxed{
T_{A\rightarrow B}(h)
=
\mu_B
+
Q(h-\mu_A).
}
\]

This is a novel extension generated in the conversation.

---

# 44. Why transport is richer than ordinary feature steering

Ordinary feature steering assumes a concept is represented by one reusable vector:

\[
v.
\]

Intervention:

\[
h'=h+\alpha v.
\]

Transport instead assumes:

\[
v
\]

may be meaningful only relative to a local coordinate system.

So:

\[
v
\rightarrow
v^{(z)}.
\]

And corresponding local representations are connected by:

\[
T_{z\rightarrow z'}.
\]

Meaning becomes closer to:

\[
\boxed{
\{v^A,v^B,v^C,\ldots\}
}
\]

plus the relations that identify them as corresponding objects.

---

# 45. Is transport “richer than COAST”?

The conversation made an important distinction.

## Experimentally today

No.

COAST directly demonstrates behavioral VLA steering.

Relational Transport is primarily a representation-geometry result in LLMs.

## Conceptually

They answer different questions.

COAST:

\[
\boxed{
\text{What subspace should I steer toward?}
}
\]

Transport:

\[
\boxed{
\text{How does corresponding geometry change between coordinate systems?}
}
\]

Potential combined framework:

\[
\boxed{
\text{COAST gives local control geometry;}
\text{ transport maps corresponding control geometry across regimes.}
}
\]

This is a proposed synthesis, not a published result.

---

# 46. Action Atlas / “Not All Features Are Created Equal”

## Paper

**Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models**  
Bryce Grant, Xijia Zhao, Peng Wang  
arXiv:2603.19233

Methods discussed:

- cross-task activation injection;
- counterfactual prompting;
- sparse autoencoders;
- linear probes;
- feature ablation;
- feature steering.

Major result relevant to this conversation:

Cross-task activation injection can make a destination rollout move toward the **source task’s spatial coordinates** while degrading destination success.

This suggests internal action representations may encode:

\[
\boxed{
\text{spatially grounded motor programs}
}
\]

rather than abstract transferable task programs.

## Why this matters to our project

It supplies a concrete example of:

\[
\boxed{
\text{causal influence}
\not\Rightarrow
\text{good transferable control variable}
}
\]

An activation can strongly control behavior and still be destructive when transplanted into the wrong context.

This supports the regime/local-coordinate hypothesis.

---

# 47. Per-token vs mean-pooled processing

The Action Atlas discussion also motivated an important methodological caution.

A representation can have strong reconstruction metrics after token aggregation while losing behaviorally important information.

Thus:

\[
\boxed{
\text{high explained variance}
\not\Rightarrow
\text{behavioral fidelity}
}
\]

For VLA analysis, especially with action sequences, keep token structure explicit unless an aggregation method is behaviorally validated.

---

# 48. Repeated statement: “If I intervene on a causal layer/token feature, why does it affect unrelated downstream tasks?”

Possible explanations assembled in the discussion:

1. **superposition** — feature directions overlap;
2. **shared causal circuitry** — same representation genuinely supports multiple tasks;
3. **nonlinear downstream readout** — small residual change propagates broadly;
4. **off-manifold intervention** — state is pushed into unrealistic geometry;
5. **regime mismatch** — correct direction in one local state is wrong in another;
6. **coordinate rotation** — same conceptual variable is differently represented across regimes/layers;
7. **wrong intervention magnitude** — causal but too strong;
8. **token aggregation destroys local structure**.

Therefore collateral effects should not automatically be interpreted as proof of superposition.

---

# 49. Repeated statement: “Middle/later layers may be more behaviorally useful because they are denser.”

Correction:

The evidence discussed supports:

- greater contextualization;
- stronger brain alignment in intermediate layers;
- sometimes more task/action-specific structure later.

It does **not** directly establish “density.”

A useful empirical program is to separately measure:

\[
\text{sparsity}
\]

\[
\text{effective rank}
\]

\[
\text{feature overlap}
\]

\[
\text{contextual information}
\]

\[
\text{behavioral causal sensitivity}.
\]

---

# 50. Repeated statement: “Energy is basically probability / entropy.”

Correction:

Related, not identical.

For probabilistic EBMs:

\[
P(x)\propto e^{-E(x)}.
\]

But JEPA energy can be a compatibility function.

Entropy is a distribution-level uncertainty quantity.

This distinction matters because a proposed residual intervention such as:

\[
h'
=
h-\eta\nabla_hE(h)
\]

only has a probabilistic interpretation if the energy is actually calibrated to a density or clearly defined target geometry.

---

# 51. Repeated statement: “Regime switching may not be about time.”

Correction:

Regime **identity** can be geometric.

Regime **switching** requires sequential transition structure.

A static clustering method tests:

\[
P(h).
\]

An HMM tests:

\[
P(h_t,z_t,z_{t+1}).
\]

This distinction tells us when an HMM is justified.

---

# 52. Repeated statement: “The VLM’s image and language information may be superimposed.”

Correction:

Three separate phenomena:

1. distinct image/language token positions;
2. multimodal attention mixing across positions;
3. feature superposition within each residual vector.

These should be analyzed separately.

---

# 53. Repeated statement: “Attention is like memory.”

Correction:

The strongest defensible analogy is:

\[
\boxed{
\text{attention}
\approx
\text{associative/content-addressable retrieval}
}
\]

not memory storage itself.

The Hopfield-attention equivalence makes this analogy mathematically useful without implying biological equivalence.

---

# 54. Novel high-level hypothesis generated in the chat

The conversation eventually converged on:

\[
\boxed{
\begin{aligned}
&\text{VLA residual streams encode model-native latent world states;}\\
&\text{these states may function like predictive/belief states;}\\
&\text{the model traverses distinct local dynamical regimes;}\\
&\text{different regimes have different success/failure geometry;}\\
&\text{a global steering operator may blur incompatible local geometries;}\\
&\text{regime- or belief-conditioned steering may improve selectivity.}
\end{aligned}
}
\]

This is the strongest integrated research hypothesis produced in the thread.

It is **not** a claim made by any one cited paper.

---

# 55. Why this is more “world-model science” than ordinary feature steering

Ordinary feature steering asks:

\[
\boxed{
\text{Where is feature }f
\text{ and what happens if I add/remove it?}
}
\]

The emerging world-model framing asks:

\[
\boxed{
\text{What latent state does the model believe it is in?}
}
\]

\[
\boxed{
\text{How does that state evolve under action and observation?}
}
\]

\[
\boxed{
\text{What local geometry governs prediction/control in that state?}
}
\]

\[
\boxed{
\text{How does the geometry change when the system transitions regimes?}
}
\]

This shifts from static feature interpretation toward **latent dynamical systems**.

---

# 56. Minimal experiment proposed to reduce the “Frankenstein monster”

Before HMMs, transport, or complicated energy steering:

Choose one task that already passes the behavioral gate.

Define only two physically obvious regimes:

\[
A=\text{pre-contact}
\]

\[
B=\text{post-contact}.
\]

At a fixed:

\[
(\ell,p),
\]

fit:

\[
(\mu_A,\Sigma_A)
\]

and:

\[
(\mu_B,\Sigma_B).
\]

Then ask:

## Question 1 — Are they different?

Compare:

- center distance;
- covariance eigenspaces;
- principal angles;
- Grassmann distance;
- CKA/RSA;
- conceptor overlap.

## Question 2 — Is success/failure geometry regime-dependent?

Fit:

\[
C_s^A,\quad C_f^A
\]

and:

\[
C_s^B,\quad C_f^B.
\]

Ask whether:

\[
C_s^A
\neq
C_s^B.
\]

## Question 3 — Does a global conceptor blur the local operators?

Compare:

\[
C_{\text{global}}
\]

against local:

\[
C_A,\quad C_B.
\]

## Question 4 — Is local steering more behaviorally selective?

Test:

\[
\operatorname{Perf}(C_A\mid A)
\]

and:

\[
\operatorname{Perf}(C_B\mid B)
\]

against global steering and controls.

Only if the static two-regime geometry is clearly real should the project escalate to HMM inference.

---

# 57. Next-stage HMM experiment if justified

Collect sequence:

\[
h_1,\ldots,h_T
\]

at a fixed:

\[
(\ell,p).
\]

Fit \(K\)-state Gaussian-emission HMM:

\[
z_t\in\{1,\ldots,K\}.
\]

Inspect:

- state occupancy;
- transition matrix;
- emission centers;
- emission covariances;
- state duration;
- relationship to contact/grasp/recovery;
- relationship to success/failure.

Compare against a simpler baseline:

- fixed normalized-time bins;
- k-means/GMM without transition structure.

If HMM does not beat static clustering/time bins, the regime-dynamics hypothesis weakens.

---

# 58. Strong evidence ladder for “world-model interpretation of residual stream”

## Level 1 — Decodability

A physical/task state variable can be decoded from:

\[
h_{\tau,\ell,p}.
\]

## Level 2 — Geometric structure

Condition-specific states occupy reproducible directions/subspaces/distributions.

## Level 3 — Predictive sufficiency

The representation predicts relevant future state/action better than simpler baselines.

## Level 4 — Dynamical structure

The latent representation exhibits coherent sequential transition structure.

## Level 5 — Causality

Activation intervention alters predicted/action trajectory in the expected direction.

## Level 6 — Selective control

The intervention improves target behavior without broad collateral damage.

The discussion repeatedly emphasized that these levels must not be collapsed.

---

# 59. Papers explicitly mentioned or central to this conversation

## 59.1 Controlling Recurrent Neural Networks by Conceptors

Herbert Jaeger, 2014  
arXiv:1403.3369

Role in this thread:

- historical origin of conceptors;
- dynamical neural state regions;
- soft projection/operator view;
- direct ancestor of later conceptor steering.

---

## 59.2 Attention Is All You Need

Vaswani et al., 2017  
arXiv:1706.03762

Role:

- Transformer attention;
- query/key/value mechanism;
- separation between sequence routing and recurrent state.

---

## 59.3 Hopfield Networks is All You Need

Ramsauer et al., 2020  
arXiv:2008.02217

Role:

- modern Hopfield associative-memory interpretation;
- mathematical equivalence between Hopfield retrieval update and Transformer attention;
- useful bridge to memory without claiming neurobiological equivalence.

---

## 59.4 Emergent Linear Representations in World Models of Self-Supervised Sequence Models

Nanda, Lee, Wattenberg, 2023  
arXiv:2309.00941

Role:

- Othello model-native coordinate system;
- mine/yours/empty representation;
- causal vector intervention;
- inspiration for asking what coordinate system a VLA uses for physical state.

---

## 59.5 A Path Towards Autonomous Machine Intelligence

Yann LeCun, 2022

Role:

- JEPA/world-model architecture;
- energy-based modeling framing;
- predictive latent representations;
- hierarchical world-model intuition.

---

## 59.6 Brains and algorithms partially converge in natural language processing

Charlotte Caucheteux & Jean-Rémi King, 2022  
Communications Biology

Role:

- intermediate Transformer representations show strongest brain alignment;
- contextual/compositional representation discussion;
- explicit caution: this does not establish higher density or superposition.

---

## 59.7 Mnemonic Similarity Task: A Tool for Assessing Hippocampal Integrity

Stark, Kirwan, Stark, 2019  
Trends in Cognitive Sciences

Role:

- behavioral assay designed to tax hippocampal pattern separation;
- inspiration for matched “lure-like” VLA stimuli;
- distinction between behavioral discrimination and directly measuring neural pattern separation.

---

## 59.8 OpenVLA: An Open-Source Vision-Language-Action Model

Kim et al., 2024  
arXiv:2406.09246

Role:

- concrete VLA architecture example;
- vision-language backbone tied to robot actions;
- contrast with action-expert architectures.

---

## 59.9 \(\pi_0\): A Vision-Language-Action Flow Model for General Robot Control

Black et al., 2024  
arXiv:2410.24164

Role:

- VLM + separate action expert;
- flow-matching continuous action generation;
- demonstrates why “VLA residual stream” must be architecture-specific.

---

## 59.10 Next-Latent Prediction Transformers Learn Compact World Models

arXiv:2511.05963

Role:

- next-latent prediction;
- belief-state-like predictive sufficient representation;
- recurrent inductive bias in Transformers;
- bridge from static residual representation to compact world state.

---

## 59.11 CORAL: Correctness-Optimized Residual Activation Lens

Miao, Cho, Ungar, 2026  
arXiv:2602.06022

Role:

- residual-stream steering genealogy leading toward COAST;
- distributed correctness signals;
- inference-time intervention.

---

## 59.12 Rectified LpJEPA

Kuang et al., 2026  
arXiv:2602.01456

Role:

- sparse representation distributions;
- Rectified Generalized Gaussian;
- RDMReg;
- maximum-entropy representation constraints;
- distinction between sparsity, distribution matching, and pointwise steering.

---

## 59.13 Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models

Grant, Zhao, Wang, 2026  
arXiv:2603.19233

Role:

- Action Atlas;
- activation injection;
- per-token SAEs;
- probes;
- ablation/steering;
- spatially grounded motor programs;
- evidence that causal activation transplantation can destroy target behavior.

---

## 59.14 Conceptors for Semantic Steering

Triantafyllopoulos et al., 2026  
arXiv:2605.04980

Role:

- multidimensional semantic concept geometry;
- conceptor quota / layer-selection logic;
- Boolean conceptor operations;
- direct methodological precursor to COAST.

---

## 59.15 COAST

Miao, Kim, Yang, Ungar, 2026  
arXiv:2605.17144

Role:

- central inspiration;
- contrastive conceptors;
- success/failure VLA residual subspaces;
- multiplicative steering;
- cross-task failure geometry;
- bridge between interpretability and robotics performance.

---

## 59.16 Language Models Represent and Transform Concepts with Shared Geometry

Hu, Niu, Varma, 2026  
arXiv:2607.04525

Role:

- concept point clouds;
- contextual displacement vector fields;
- relational transport;
- shared geometry despite differing absolute coordinates;
- inspiration for regime-to-regime geometric correspondence.

---

# 60. Established results vs hypotheses generated here

## Established / paper-grounded

- Transformer attention uses Q/K/V weighted routing.
- Conceptors are covariance-derived soft projection operators.
- COAST uses contrastive conceptors to steer VLA hidden states.
- OthelloGPT has a linearly accessible mine/yours/empty board representation.
- NextLat explicitly targets compact predictive latent/belief-state representations.
- Rectified \(L_p\)JEPA explicitly regularizes toward sparse maximum-entropy representation distributions.
- Action Atlas reports strong causal effects from VLA activation injection and per-token feature interventions.
- Relational Transport shows shared relational transformation geometry across language models.
- Caucheteux/King report strongest brain alignment in intermediate Transformer layers.
- MST is designed to place strong demands on hippocampal pattern-separation processes.

## Hypotheses / ideas generated in this chat

These must be treated as **proposals**, not literature claims:

1. **World-model interpretation of the VLA residual stream**
2. **Regime-conditioned success/failure conceptors**
3. **Global COAST operator as an average/blurring of local control geometries**
4. **Center + covariance intervention**
5. **Belief-conditioned conceptor steering**
6. **Belief-conditioned energy intervention**
7. **Regime-conditioned sparsity**
8. **Matched “lure” stimuli for VLA pattern-separation analysis**
9. **Transport of features/subspaces between behavioral regimes**
10. **Procrustes transport of regime-specific conceptors**
11. **Use of HMM/switching dynamical systems over activation trajectories**
12. **Testing whether collision/interference from global steering is caused by regime mismatch rather than only superposition**
13. **Model-native behavioral regimes may differ from human labels such as approach/contact/grasp**
14. **Local representational geometry may be a better unit of control than a globally fixed feature vector**

---

# 61. Core equations cheat sheet

## RNN

\[
h_t=\phi(W_xx_t+W_hh_{t-1}+b)
\]

## Covariance

\[
\Sigma
=
\frac1N
(X-\mu)^\top(X-\mu)
\]

## PCA

\[
\Sigma=V\Lambda V^\top
\]

## Conceptor

\[
C
=
R(R+\alpha^{-2}I)^{-1}
\]

## Conceptor eigenvalue shrinkage

\[
\mu_i
=
\frac{\lambda_i}
{\lambda_i+\alpha^{-2}}
\]

## COAST-style residual operation

\[
h'=hM^\top
\]

\[
M=(1-\beta)I+\beta C_{\text{steer}}
\]

## Attention

\[
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
\]

\[
Q=HW_Q,\quad
K=HW_K,\quad
V=HW_V
\]

\[
O=AV
\]

## Superposition toy model

\[
h=Wf
\]

## Sparsity

\[
\|f_t\|_0\ll \dim(f)
\]

## Gaussian regime

\[
h_t\mid z_t=r
\sim
\mathcal N(\mu_r,\Sigma_r)
\]

## GMM

\[
P(h)
=
\sum_r
\pi_r
\mathcal N(h;\mu_r,\Sigma_r)
\]

## Markov property

\[
P(z_{t+1}\mid z_{1:t})
=
P(z_{t+1}\mid z_t)
\]

## HMM belief

\[
b_t(r)
=
P(z_t=r\mid h_{1:t})
\]

## Switching linear dynamics

\[
h_{t+1}
=
A_{z_t}h_t
+
B_{z_t}a_t
+
\epsilon_t
\]

## Unified regime/sparse representation hypothesis

\[
h_t
=
\mu_{z_t}
+
W_{z_t}f_t
+
\epsilon_t
\]

## Gaussian energy

\[
E_r(h)
=
\frac12
(h-\mu_r)^\top
\Sigma_r^{-1}
(h-\mu_r)
+
\frac12\log|\Sigma_r|
\]

## Probabilistic EBM

\[
P(h\mid r)
\propto
e^{-E_r(h)}
\]

## Belief-conditioned operator

\[
M_t
=
\sum_r
b_t(r)M_r
\]

## Belief-conditioned energy

\[
E(h_t,b_t)
=
\sum_r
b_t(r)E_r(h_t)
\]

## Energy steering

\[
h_t'
=
h_t
-
\eta\nabla_hE(h_t,b_t)
\]

## Affine center-aware intervention

\[
h'
=
\mu_s
+
M(h-\mu_s)
\]

## Procrustes transport

\[
Q^*
=
\arg\min_Q
\|QX_A-X_B\|_F^2
\quad
\text{s.t.}
\quad
Q^\top Q=I
\]

## Transported covariance

\[
\Sigma_B
\approx
Q\Sigma_AQ^\top
\]

## Transported conceptor

\[
C_B
\approx
QC_AQ^\top
\]

---

# 62. Recommended current project framing

The project should **not** try to prove the entire conceptual chain at once.

A cleaner framing is:

## Link 1 — Local state geometry

At one behaviorally valid task/checkpoint:

> Do pre-contact and post-contact states occupy measurably different residual geometries at behaviorally relevant layer/token positions?

## Link 2 — Local control geometry

> Does success/failure geometry differ between those states, and does state-conditioned steering outperform one global operator?

Everything else—HMMs, belief-conditioned interventions, transport, sophisticated energy models—becomes a follow-up **only if those first two links survive**.

---

# 63. Final conceptual summary

The conversation started with:

> Can mechanistic interpretability find a safety-relevant feature in a VLA/world model?

It evolved into:

\[
\boxed{
\text{What latent world state is represented in the residual stream?}
}
\]

Then:

\[
\boxed{
\text{Does that state move through distinct local dynamical regimes?}
}
\]

Then:

\[
\boxed{
\text{Does each regime have different success/failure geometry?}
}
\]

And finally:

\[
\boxed{
\text{Can control be made more selective by steering within the correct local geometry rather than applying one global feature/subspace intervention?}
}
\]

That is the central research thread to preserve.



---

# 64. Implementation-Ready Mathematics for the Novel Ideas Generated in This Chat

> **Status:** Everything in this section is a proposed research formulation unless explicitly labeled as coming from an existing paper. These equations are intended to make the brainstorm concrete enough for implementation and falsification.

The main proposed stack is:

\[
\boxed{
\text{model-native latent state}
\rightarrow
\text{regime inference}
\rightarrow
\text{regime-specific distribution geometry}
\rightarrow
\text{attention-routing consequences}
\rightarrow
\text{local causal intervention}
}
\]

The minimal implementation unit should remain:

\[
h_{\tau,\ell,p}\in\mathbb R^d
\]

for a fixed layer \(\ell\) and token position \(p\), measured across physical rollout time \(\tau\).

---

# 65. Othello \(\rightarrow\) Markov state \(\rightarrow\) HMM belief state

## 65.1 Othello as the clean deterministic special case

For Othello, the move sequence:

\[
a_{1:t}
\]

deterministically specifies a board state:

\[
s_t
=
F(a_{1:t}).
\]

If the model has internally compressed the relevant history, then some hidden representation:

\[
h_t
\]

should contain enough information to recover the board state or an equivalent model-native coordinate system.

The Othello paper's central interpretability result suggests a relational coordinate system:

\[
\psi(s_t)
=
\{\text{mine},\text{yours},\text{empty}\}
\]

is more linearly accessible than the human-default absolute-color coordinate system.

Because the board is essentially known exactly from the sequence, the corresponding Bayesian belief state is nearly degenerate:

\[
b_t(s)
=
P(s_t=s\mid a_{1:t})
\approx
\delta(s=s_t^\star).
\]

So Othello is a useful limiting case:

\[
\boxed{
\text{belief state with almost no uncertainty}
}
\]

but with a non-obvious **model-native coordinate system**.

---

## 65.2 VLA / world-model generalization

In robotics, the true world state:

\[
s_\tau
\]

is only partially observed through:

\[
o_{1:\tau}
\]

and depends on past actions:

\[
a_{1:\tau-1}.
\]

The classical belief state is:

\[
\boxed{
b_\tau(s)
=
P(
s_\tau=s
\mid
o_{1:\tau},
a_{1:\tau-1}
)
}
\]

A learned VLA hidden state:

\[
h_{\tau,\ell,p}
=
f_{\ell,p}
(
o_{\le \tau},
a_{<\tau}
)
\]

is **belief-state-like** if it is approximately sufficient for predicting relevant futures:

\[
P(
o_{\tau+1:T}
\mid
o_{\le\tau},
a_{<\tau}
)
\approx
P(
o_{\tau+1:T}
\mid
h_{\tau,\ell,p}
).
\]

### Proposed Othello-to-VLA question

Instead of assuming the correct human labels are:

\[
\{\text{approach},\text{contact},\text{grasp}\},
\]

search for a model-native coordinate map:

\[
\boxed{
\psi:
h_{\tau,\ell,p}
\rightarrow
\xi_\tau
}
\]

where \(\xi_\tau\) is a compact latent state whose relations predict:

- future observation;
- future action;
- success/failure;
- regime transition.

The direct analogy is:

\[
\boxed{
\text{black/white}
\rightarrow
\text{mine/yours}
}
\]

in Othello versus:

\[
\boxed{
\text{human physical labels}
\rightarrow
\text{model-native relational physical coordinates}
}
\]

in a VLA.

---

# 66. Testing whether a VLA latent is actually Markov-like

The Markov sufficiency hypothesis is:

\[
\boxed{
P(
h_{\tau+1}
\mid
h_{1:\tau},
a_\tau
)
\approx
P(
h_{\tau+1}
\mid
h_\tau,
a_\tau
)
}
\]

A practical predictive test:

### Model A — Markov predictor

\[
\hat h_{\tau+1}
=
F_\theta(
h_\tau,
a_\tau
)
\]

### Model B — history-aware predictor

\[
\hat h_{\tau+1}
=
G_\phi(
h_{\tau-k:\tau},
a_{\tau-k:\tau}
)
\]

If the history-aware model gives little held-out improvement:

\[
\Delta
=
L_{\text{Markov}}
-
L_{\text{history}}
\approx 0,
\]

then \(h_\tau\) is closer to a sufficient Markov-like state.

A stronger information-theoretic target is:

\[
\boxed{
I(
h_{\tau+1};
h_{1:\tau-1}
\mid
h_\tau,a_\tau
)
\approx0.
}
\]

This is difficult to estimate directly in high dimensions, so prediction-based approximations are more practical.

---

# 67. HMM formulation over VLA residual states

Fix a layer/token pair:

\[
(\ell,p).
\]

Let:

\[
x_\tau
=
h_{\tau,\ell,p}.
\]

Introduce a discrete latent regime:

\[
z_\tau
\in
\{1,\ldots,K\}.
\]

## Transition model

\[
\boxed{
P(
z_{\tau+1}=j
\mid
z_\tau=i
)
=
A_{ij}
}
\]

where:

\[
A
\in
\mathbb R^{K\times K}
\]

is the transition matrix.

## Gaussian emission model

\[
\boxed{
x_\tau
\mid
z_\tau=r
\sim
\mathcal N(
\mu_r,
\Sigma_r
)
}
\]

Each hidden regime therefore corresponds to its own local ellipsoid.

## HMM belief state

The filtered posterior is:

\[
\boxed{
b_\tau(r)
=
P(
z_\tau=r
\mid
x_{1:\tau}
)
}
\]

with the forward recursion:

\[
\tilde b_\tau(r)
=
P(
x_\tau
\mid
z_\tau=r
)
\sum_q
A_{qr}
b_{\tau-1}(q)
\]

and normalization:

\[
\boxed{
b_\tau(r)
=
\frac{
\tilde b_\tau(r)
}{
\sum_j
\tilde b_\tau(j)
}
}
\]

This is the explicit mathematical bridge between:

- a latent Markov chain;
- Gaussian activation subspaces;
- a belief state over those hidden regimes.

---

# 68. GMM versus HMM: the exact distinction to implement

## Gaussian mixture model

\[
\boxed{
P(x)
=
\sum_r
\pi_r
\mathcal N(
x;
\mu_r,
\Sigma_r
)
}
\]

No sequence order is used.

## HMM marginal

\[
P(
x_{1:T}
)
=
\sum_{z_{1:T}}
P(z_1)
\prod_{\tau=1}^{T}
P(
x_\tau
\mid
z_\tau
)
\prod_{\tau=1}^{T-1}
P(
z_{\tau+1}
\mid
z_\tau
).
\]

The HMM is justified only if transition structure adds predictive value.

### Required baseline test

Compare held-out sequence likelihood / predictive performance for:

1. one Gaussian;
2. GMM;
3. fixed normalized-time bins;
4. HMM;
5. optionally a switching linear dynamical system.

If HMM does not beat the GMM/time-bin baselines, there is no evidence that **Markov regime switching** is needed.

---

# 69. Proposed continuous/discrete switching dynamical system

A richer formulation is:

\[
z_\tau
\sim
P(
z_\tau
\mid
z_{\tau-1}
)
\]

and:

\[
\boxed{
h_{\tau+1}
=
A_{z_\tau}h_\tau
+
B_{z_\tau}a_\tau
+
c_{z_\tau}
+
\epsilon_\tau.
}
\]

Here each regime has its own local dynamics:

\[
A_r.
\]

This tests a stronger hypothesis than "there are different activation clusters."

It asks:

\[
\boxed{
\text{Do different regions of residual space obey different transition laws?}
}
\]

For physical AI, this is more naturally a **world-model hypothesis** than a static feature-steering hypothesis.

---

# 70. Proposed regime-conditioned sparse representation

The chat generated the following unified model:

\[
\boxed{
h_\tau
=
\mu_{z_\tau}
+
W_{z_\tau}f_\tau
+
\epsilon_\tau
}
\]

where:

- \(z_\tau\) = hidden dynamical regime;
- \(\mu_{z_\tau}\) = local center;
- \(W_{z_\tau}\) = local feature dictionary/basis;
- \(f_\tau\) = sparse feature coefficients.

Sparsity:

\[
\boxed{
\|f_\tau\|_0
\ll
\dim(f)
}
\]

The active support is:

\[
S_\tau
=
\{
i:
f_{\tau,i}\neq0
\}.
\]

The novel hypothesis is:

\[
\boxed{
P(
S_\tau
\mid
z_\tau=r
)
}
\]

differs across regimes.

For example:

\[
P(
\text{force feature active}
\mid
z=\text{contact}
)
\gg
P(
\text{force feature active}
\mid
z=\text{approach}
).
\]

### Implementation with an SAE

If a sparse autoencoder gives:

\[
f_\tau
=
\operatorname{SAEEnc}(h_\tau),
\]

measure per-regime:

- feature activation probability;
- mean feature magnitude;
- support overlap;
- entropy of active features;
- mutual information between feature activation and regime.

For feature \(i\):

\[
P(
f_i>0
\mid
z=r
).
\]

Support-overlap metric:

\[
\boxed{
J(r,s)
=
\frac{
|S_r\cap S_s|
}{
|S_r\cup S_s|
}
}
\]

where \(S_r\) is the set of features frequently active in regime \(r\).

This is one concrete way to test **regime-conditioned sparsity** rather than speaking about sparsity only qualitatively.

---

# 71. Measuring whether regime subspaces/distributions are actually different

Do not rely on one metric.

For regimes \(r\) and \(s\), compare:

## 71.1 Center displacement

\[
\boxed{
D_\mu(r,s)
=
\|
\mu_r-\mu_s
\|_2
}
\]

or a whitened version:

\[
D_M^2
=
(
\mu_r-\mu_s
)^\top
\Sigma_{\text{pooled}}^{-1}
(
\mu_r-\mu_s
).
\]

## 71.2 Covariance difference

\[
\boxed{
D_\Sigma(r,s)
=
\|
\Sigma_r-\Sigma_s
\|_F.
}
\]

## 71.3 Principal angles between top-\(k\) subspaces

If:

\[
U_r,U_s
\in
\mathbb R^{d\times k},
\]

compute singular values:

\[
\sigma_i
(
U_r^\top U_s
)
=
\cos\theta_i.
\]

The \(\theta_i\) are principal angles.

A Grassmann-style distance is:

\[
\boxed{
D_G(r,s)
=
\sqrt{
\sum_{i=1}^{k}
\theta_i^2
}.
}
\]

## 71.4 Gaussian 2-Wasserstein distance

For Gaussian approximations:

\[
\boxed{
W_2^2(
r,s
)
=
\|
\mu_r-\mu_s
\|_2^2
+
\operatorname{tr}
\left(
\Sigma_r
+
\Sigma_s
-
2
(
\Sigma_s^{1/2}
\Sigma_r
\Sigma_s^{1/2}
)^{1/2}
\right)
}
\]

This jointly measures center and covariance differences.

## 71.5 Conceptor overlap

For conceptors \(C_r,C_s\):

\[
\boxed{
\operatorname{sim}(C_r,C_s)
=
\frac{
\operatorname{tr}(C_rC_s)
}{
\sqrt{
\operatorname{tr}(C_r^2)
\operatorname{tr}(C_s^2)
}
}
}
\]

This is directly relevant to the COAST lineage.

---

# 72. Regime-specific success/failure conceptors

For each regime \(r\), gather successful and failed residuals separately.

Centered success covariance:

\[
R_{s,r}
=
\frac1{N_{s,r}}
\tilde X_{s,r}^\top
\tilde X_{s,r}.
\]

Success conceptor:

\[
\boxed{
C_{s,r}
=
R_{s,r}
(
R_{s,r}
+
\alpha^{-2}I
)^{-1}
}
\]

Failure conceptor:

\[
\boxed{
C_{f,r}
=
R_{f,r}
(
R_{f,r}
+
\alpha^{-2}I
)^{-1}
}
\]

Then construct a regime-specific contrastive operator using the same Boolean-conceptor machinery as COAST:

\[
\boxed{
C_{\text{steer},r}
=
C_{s,r}
\wedge
\neg C_{f,r}.
}
\]

The **new hypothesis** is not the conceptor equation itself; it is:

\[
\boxed{
C_{\text{steer},r}
\neq
C_{\text{steer},s}
}
\]

for physically/dynamically distinct regimes \(r\neq s\).

---

# 73. Is global COAST a blurred average of local operators?

This was one of the central novel ideas in the chat.

A first approximation is:

\[
\boxed{
\bar C
=
\sum_r
\pi_r
C_r
}
\]

where:

\[
\pi_r
=
P(z=r).
\]

Compare the actual global conceptor:

\[
C_{\text{global}}
\]

with:

\[
\bar C.
\]

Metrics:

\[
\boxed{
\Delta_C
=
\|
C_{\text{global}}
-
\bar C
\|_F
}
\]

and:

\[
\boxed{
\Delta_{\text{op}}
=
\|
C_{\text{global}}
-
\bar C
\|_2.
}
\]

This is not guaranteed to be exactly equal because conceptor construction is nonlinear:

\[
C(R)
=
R(R+\alpha^{-2}I)^{-1}.
\]

In general:

\[
\boxed{
C(
\mathbb E[R_r]
)
\neq
\mathbb E[
C(R_r)
].
}
\]

That nonlinearity is itself important.

The empirically testable question is:

> Does the global operator behave like a coarse compromise among incompatible local geometries?

A behavioral test:

\[
\operatorname{Perf}
(
C_r
\mid
z=r
)
>
\operatorname{Perf}
(
C_{\text{global}}
\mid
z=r
)
\]

with less collateral degradation.

---

# 74. Center-aware local steering

For regime \(r\), let success center be:

\[
\mu_{s,r}.
\]

Instead of applying a linear map around the origin:

\[
h'=M_rh,
\]

use an affine intervention:

\[
\boxed{
h'
=
\mu_{s,r}
+
M_r
(
h-\mu_{s,r}
).
}
\]

This separates:

- **translation toward a desirable center**;
- **rotation/shrinkage within the desirable covariance geometry**.

A clean ablation is:

### Center only

\[
h'
=
h
+
\gamma
(
\mu_{s,r}
-
\mu_{f,r}
).
\]

### Covariance only

\[
h'
=
M_rh.
\]

### Center + covariance

\[
h'
=
\mu_{s,r}
+
M_r
(
h-\mu_{s,r}
).
\]

This directly tests whether the behaviorally relevant information is primarily:

\[
\boxed{
\text{first moment }
\mu
}
\]

versus:

\[
\boxed{
\text{second moment }
\Sigma.
}
\]

---

# 75. Hard regime-conditioned steering

If an inferred regime is:

\[
\hat z_\tau
=
\arg\max_r
b_\tau(r),
\]

then choose:

\[
\boxed{
M_\tau
=
M_{\hat z_\tau}.
}
\]

Intervention:

\[
\boxed{
h_\tau'
=
\mu_{s,\hat z_\tau}
+
M_{\hat z_\tau}
(
h_\tau
-
\mu_{s,\hat z_\tau}
).
}
\]

This is simple but brittle when regime uncertainty is high.

---

# 76. Soft belief-conditioned steering

Use the HMM posterior directly.

Define:

\[
\boxed{
M_\tau
=
\sum_r
b_\tau(r)M_r.
}
\]

Similarly:

\[
\boxed{
\mu_\tau
=
\sum_r
b_\tau(r)
\mu_{s,r}.
}
\]

Then:

\[
\boxed{
h_\tau'
=
\mu_\tau
+
M_\tau
(
h_\tau-\mu_\tau
).
}
\]

This implements the novel idea:

\[
\boxed{
\text{steering strength/direction should depend on the model's inferred state}
}
\]

rather than using one global operator.

---

# 77. Energy-based local geometry

For Gaussian regime \(r\):

\[
h
\mid
r
\sim
\mathcal N(
\mu_r,
\Sigma_r
).
\]

The negative log-density is:

\[
\boxed{
E_r(h)
=
\frac12
(
h-\mu_r
)^\top
\Sigma_r^{-1}
(
h-\mu_r
)
+
\frac12
\log|
\Sigma_r
|
+
c.
}
\]

This gives an ellipsoidal energy landscape.

Low energy means:

\[
h
\]

lies in a high-density region under the regime model.

---

# 78. Contrastive success/failure energy

A useful proposed score is the log-density ratio:

\[
\boxed{
E_{\text{contrast},r}(h)
=
E_{s,r}(h)
-
\lambda
E_{f,r}(h).
}
\]

Depending on sign convention, minimize:

\[
E_{s,r}(h)
-
\lambda E_{f,r}(h)
\]

to move toward success while away from failure.

An equivalent discriminative score is:

\[
\boxed{
S_r(h)
=
\log
P(
h\mid
\text{success},r
)
-
\log
P(
h\mid
\text{failure},r
).
}
\]

This has a direct probabilistic interpretation under the Gaussian approximation.

---

# 79. Belief-conditioned energy: two versions

## 79.1 Simple weighted-energy approximation

\[
\boxed{
E_{\text{soft}}(h,b)
=
\sum_r
b(r)
E_r(h).
}
\]

Gradient:

\[
\boxed{
\nabla_h
E_{\text{soft}}
=
\sum_r
b(r)
\nabla_hE_r(h).
}
\]

Then:

\[
\boxed{
h'
=
h
-
\eta
\nabla_h
E_{\text{soft}}.
}
\]

## 79.2 More probabilistically principled mixture energy

If:

\[
P(h\mid b)
=
\sum_r
b(r)
P(h\mid r),
\]

then:

\[
\boxed{
E_{\text{mix}}(h,b)
=
-
\log
\sum_r
b(r)
e^{-E_r(h)}
+
c.
}
\]

Define posterior responsibilities:

\[
\boxed{
\rho_r(h,b)
=
\frac{
b(r)e^{-E_r(h)}
}{
\sum_j
b(j)e^{-E_j(h)}
}.
}
\]

Then:

\[
\boxed{
\nabla_h
E_{\text{mix}}
=
\sum_r
\rho_r(h,b)
\nabla_hE_r(h).
}
\]

This is an important refinement of the earlier chat idea.

The effective steering weights depend on both:

- the HMM belief \(b(r)\);
- how compatible the current activation \(h\) is with each regime.

---

# 80. Attention under regime-specific activation distributions

This directly addresses the repeated question:

> If each subspace/regime has a different probability distribution, can the Transformer attention mechanism account for that?

Attention does **not** explicitly fit Gaussian regimes.

But different residual distributions propagate through the learned linear \(Q/K\) maps and therefore induce different **attention-score distributions**.

Let:

\[
h_i
\mid
z_i=r
\sim
\mathcal N(
\mu_r,
\Sigma_r
)
\]

and:

\[
h_j
\mid
z_j=s
\sim
\mathcal N(
\mu_s,
\Sigma_s
).
\]

For one attention head:

\[
q_i
=
W_Qh_i
\]

\[
k_j
=
W_Kh_j.
\]

Therefore:

\[
\boxed{
q_i
\mid
r
\sim
\mathcal N(
W_Q\mu_r,
W_Q\Sigma_rW_Q^\top
)
}
\]

and:

\[
\boxed{
k_j
\mid
s
\sim
\mathcal N(
W_K\mu_s,
W_K\Sigma_sW_K^\top
).
}
\]

The pre-softmax attention score is:

\[
\boxed{
s_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d_k}
}.
}
\]

---

# 81. Expected attention score under two latent regimes

Define:

\[
A
=
\frac{
W_Q^\top W_K
}{
\sqrt{d_k}
}.
\]

Then:

\[
s_{ij}
=
h_i^\top
A
h_j.
\]

Assuming conditional independence of \(h_i\) and \(h_j\) given regimes \(r,s\):

\[
\boxed{
\mathbb E[
s_{ij}
\mid
r,s
]
=
\mu_r^\top
A
\mu_s.
}
\]

Thus different regime centers alone can systematically change attention logits.

---

# 82. Attention-score variance induced by regime covariances

Under the same conditional independence assumption:

\[
\boxed{
\operatorname{Var}
(
s_{ij}
\mid
r,s
)
=
\operatorname{tr}
(
A\Sigma_sA^\top\Sigma_r
)
+
\mu_r^\top
A\Sigma_sA^\top
\mu_r
+
\mu_s^\top
A^\top\Sigma_rA
\mu_s.
}
\]

This equation is extremely relevant to the chat's hypothesis.

It shows that **different covariance ellipsoids can alter not only the mean attention score but its variability** even if the network never explicitly names a regime.

So:

\[
\boxed{
(\mu_r,\Sigma_r)
\rightarrow
\text{different }Q/K\text{ distributions}
\rightarrow
\text{different attention-score distributions}.
}
\]

This is one rigorous mathematical answer to:

> Can attention "account for" different probability distributions in different subspaces?

It can **respond to them implicitly** because \(Q/K\) are functions of the residual state.

That is not the same as the Transformer explicitly computing:

\[
P(z=r\mid h).
\]

---

# 83. Softmax means the final attention expectation is not trivial

Attention weights are:

\[
a_{ij}
=
\frac{
e^{s_{ij}}
}{
\sum_m
e^{s_{im}}
}.
\]

In general:

\[
\boxed{
\mathbb E[
\operatorname{softmax}(s)
]
\neq
\operatorname{softmax}
(
\mathbb E[s]
).
}
\]

Therefore the regime-conditioned attention pattern should usually be estimated empirically or by Monte Carlo:

1. sample \(h_i\sim P(h\mid r)\);
2. sample key-token states \(h_j\sim P(h\mid s_j)\);
3. compute \(Q,K\);
4. compute attention;
5. estimate:

\[
\boxed{
\bar A^{(r)}
=
\mathbb E[
A
\mid
z_i=r
].
}
\]

Then compare regime attention patterns.

---

# 84. Quantifying whether attention changes across regimes

For attention distributions:

\[
a^{(r)}
\]

and:

\[
a^{(s)},
\]

candidate metrics include:

## Jensen-Shannon divergence

\[
\boxed{
D_{\text{JS}}
(
a^{(r)},
a^{(s)}
)
=
\frac12
D_{\text{KL}}
(
a^{(r)}
\|
m
)
+
\frac12
D_{\text{KL}}
(
a^{(s)}
\|
m
)
}
\]

where:

\[
m
=
\frac12
(
a^{(r)}
+
a^{(s)}
).
\]

## Attention cosine similarity

\[
\boxed{
\operatorname{cos}
(
a^{(r)},
a^{(s)}
)
=
\frac{
a^{(r)\top}a^{(s)}
}{
\|a^{(r)}\|
\|a^{(s)}\|
}.
}
\]

## Head-specific mutual information

Estimate:

\[
\boxed{
I(
z_\tau;
A_{\ell,h,\tau}
)
}
\]

to identify heads whose routing is strongly regime-dependent.

---

# 85. Direct causal test: does regime geometry control attention routing?

The proposed chain is:

\[
z
\rightarrow
h
\rightarrow
Q/K
\rightarrow
A
\rightarrow
\text{action}.
\]

This should be tested by intervention.

Suppose:

\[
h'
=
h+\delta h.
\]

Then:

\[
\delta q
=
W_Q\delta h.
\]

For a fixed key \(k_j\):

\[
\boxed{
\delta s_{ij}
\approx
\frac{
(
W_Q\delta h
)^\top
k_j
}{
\sqrt{d_k}
}.
}
\]

If the intervention also changes the key state:

\[
\delta k_j
=
W_K\delta h_j,
\]

then to first order:

\[
\boxed{
\delta s_{ij}
\approx
\frac{
\delta q_i^\top k_j
+
q_i^\top\delta k_j
}{
\sqrt{d_k}
}.
}
\]

Thus a residual-space intervention predicts a measurable attention-logit shift.

### Proposed mediation experiment

1. identify a regime-specific residual direction/subspace;
2. intervene on \(h\);
3. measure \(\Delta A\);
4. measure downstream \(\Delta\) action;
5. patch/restore the attention output;
6. test whether restoring attention attenuates the behavioral effect.

This asks whether attention mediates the causal effect of the residual geometry.

---

# 86. Regime-specific Q/K subspaces

The chat's "different probability distributions in each subspace" idea can be made more specific by fitting geometry not only in residual space but also after the learned projections.

For regime \(r\):

\[
Q_r
=
W_QX_r
\]

and:

\[
K_r
=
W_KX_r.
\]

Fit:

\[
(\mu^Q_r,\Sigma^Q_r)
\]

and:

\[
(\mu^K_r,\Sigma^K_r).
\]

Then compare:

\[
D_G(
U^Q_r,U^Q_s
)
\]

and:

\[
D_G(
U^K_r,U^K_s
).
\]

This tests whether the regime distinction is amplified or suppressed by the attention projections.

A useful layer/head statistic is:

\[
\boxed{
\Gamma_{\ell,h}
=
\frac{
D(
P(Q\mid r),
P(Q\mid s)
)
}{
D(
P(h\mid r),
P(h\mid s)
)
}.
}
\]

Interpretation:

- \(\Gamma>1\): the attention query map amplifies regime separation;
- \(\Gamma<1\): it compresses regime separation.

The same can be computed for keys.

---

# 87. Pattern separation as a generalized eigenvalue problem

The hippocampal/MST analogy can be made mathematically precise without claiming biological equivalence.

For two matched conditions \(A,B\):

Between-class scatter:

\[
\boxed{
S_B
=
(
\mu_A-\mu_B
)
(
\mu_A-\mu_B
)^\top.
}
\]

Within-class scatter:

\[
\boxed{
S_W
=
\Sigma_A+\Sigma_B.
}
\]

The Fisher/LDA direction solves:

\[
\boxed{
v^*
=
\arg\max_v
\frac{
v^\top S_Bv
}{
v^\top S_Wv
}.
}
\]

Equivalent generalized eigenproblem:

\[
S_Bv
=
\lambda
S_Wv.
\]

For each layer/token position, track:

\[
\lambda_{\max}^{(\ell,p)}.
\]

If:

\[
\lambda_{\max}
\]

increases across layers for highly similar but behaviorally distinct stimuli, that gives a concrete **pattern-separation-like statistic**.

This is richer than only measuring cosine similarity.

---

# 88. Predictive pattern separation

A stronger version asks whether separation is tied to different futures.

For matched observations \(A,B\), compute:

\[
D_h
=
D(
h_A,h_B
)
\]

and future divergence:

\[
D_{\text{future}}
=
D(
y_{A,\tau+1:T},
y_{B,\tau+1:T}
).
\]

Then test whether:

\[
\boxed{
D_h
\text{ predicts }
D_{\text{future}}
}
\]

better at particular layers/token positions.

This makes the pattern-separation analogy specifically **world-model-relevant**:

similar observations should separate when their future transition dynamics differ.

---

# 89. Othello-style model-native coordinates for VLA regimes

Do not force:

\[
z
=
\{\text{approach},\text{contact},\text{grasp}\}
\]

as the only possible state labels.

A more Othello-like strategy is to search over alternative relational coordinates.

For physical variables \(u_1,\ldots,u_m\), construct relational transforms such as:

\[
\xi_1
=
x_{\text{gripper}}
-
x_{\text{object}}
\]

\[
\xi_2
=
x_{\text{object}}
-
x_{\text{goal}}
\]

\[
\xi_3
=
\mathbf 1[
\text{contact}
]
\]

\[
\xi_4
=
\text{relative orientation}
\]

rather than only absolute coordinates.

Then compare probe/readout quality:

\[
R^2(
h\rightarrow
\text{absolute state}
)
\]

versus:

\[
R^2(
h\rightarrow
\text{relational state}
).
\]

The Othello-style hypothesis is:

\[
\boxed{
\text{relative / agent-centric variables may be more linearly accessible}
}
\]

than absolute human-default coordinates.

This would be a direct empirical test of a **model-native physical coordinate system**.

---

# 90. Relational transport between regimes

Suppose the same underlying physical concept is represented differently in regimes \(A\) and \(B\).

Matched anchor matrices:

\[
X_A
=
[
v_1^A,\ldots,v_k^A
]
\]

\[
X_B
=
[
v_1^B,\ldots,v_k^B
].
\]

Fit orthogonal Procrustes:

\[
\boxed{
Q_{A\rightarrow B}^*
=
\arg\min_Q
\|
QX_A-X_B
\|_F^2
\quad
\text{s.t. }
Q^\top Q=I.
}
\]

If:

\[
X_BX_A^\top
=
U\Sigma V^\top,
\]

then:

\[
\boxed{
Q^*
=
UV^\top.
}
\]

Held-out transport test:

\[
\boxed{
\hat v_*^B
=
Q^*
v_*^A.
}
\]

Evaluate:

\[
\cos(
\hat v_*^B,
v_*^B
)
\]

against random/identity baselines.

This tests whether a common **relation-preserving coordinate transform** maps one local representational regime to another.

---

# 91. Transporting centers, covariance, and conceptors

Under an orthogonal map \(Q\):

## Center

\[
\boxed{
\mu_B
\approx
Q\mu_A+c
}
\]

or in centered coordinates:

\[
h_B-\mu_B
\approx
Q(
h_A-\mu_A
).
\]

## Covariance

\[
\boxed{
\Sigma_B
\approx
Q\Sigma_AQ^\top.
}
\]

## Conceptor

\[
\boxed{
C_B
\approx
QC_AQ^\top.
}
\]

This gives a concrete interpretation of the earlier phrase:

> "preserve relational structure while transporting into a new local basis."

The raw coordinates change, but:

- lengths;
- angles;
- dot products;
- covariance eigenspectrum;

are preserved under orthogonal transport.

---

# 92. Testing whether transport is actually needed

Before introducing transport, compare:

### Identity baseline

\[
\hat v_B=v_A.
\]

### Mean-difference alignment

simple source/target average direction.

### Orthogonal Procrustes

\[
\hat v_B=Qv_A.
\]

### Unconstrained linear map

\[
\hat v_B=Mv_A.
\]

### Nonlinear map

small MLP if necessary.

If identity already works:

\[
v_A\approx v_B,
\]

there is no need for transport.

If Procrustes beats identity and generalizes to held-out features:

\[
\boxed{
\text{evidence for coordinate rotation}
}
\]

becomes much stronger.

---

# 93. Local geometry versus ordinary feature steering

Ordinary feature steering assumes:

\[
h'
=
h+\alpha v.
\]

The local-geometry hypothesis instead says:

\[
\boxed{
v
=
v(z)
}
\]

or more generally:

\[
\boxed{
\mathcal C
=
\mathcal C(z)
}
\]

where \(\mathcal C(z)\) is the regime-specific control subspace/operator.

This is richer because the represented object is not one static direction but an **equivalence class across local coordinate systems**:

\[
\boxed{
[
v^1,
v^2,
\ldots,
v^K
]
}
\]

connected by transport maps:

\[
Q_{r\rightarrow s}.
\]

---

# 94. Quantifying collateral damage from an intervention

Let a downstream behavior/task vector be:

\[
y
=
F(h).
\]

For a small intervention:

\[
\delta h,
\]

first-order effect:

\[
\boxed{
\delta y
\approx
J_F(h)
\delta h
}
\]

where:

\[
J_F(h)
=
\frac{
\partial F
}{
\partial h
}.
\]

Suppose:

- \(y_T\) = target behavior;
- \(y_U\) = unrelated behavior.

A selective steering direction can be formulated as:

\[
\boxed{
\max_{\delta h}
\quad
g_T^\top\delta h
-
\lambda
\|
J_U\delta h
\|_2^2
-
\eta
\|
\delta h
\|_2^2
}
\]

where:

\[
g_T
=
\nabla_h
y_T.
\]

This formalizes the conversation's recurring concern:

> "improve one downstream benchmark without affecting unrelated tasks."

A regime-conditioned version uses:

\[
J_U^{(r)}
\]

and:

\[
g_T^{(r)}.
\]

This turns behavioral selectivity into an explicit optimization criterion.

---

# 95. On-manifold / high-density regularization for steering

One explanation for destructive activation interventions is that:

\[
h'
\]

leaves the training distribution.

For regime \(r\), add a geometry penalty:

\[
\boxed{
\mathcal L_{\text{manifold}}
=
(
h'-\mu_r
)^\top
\Sigma_r^{-1}
(
h'-\mu_r
).
}
\]

Then optimize:

\[
\boxed{
\delta h^*
=
\arg\max_{\delta h}
\left[
\text{target gain}
-
\lambda
\text{collateral loss}
-
\gamma
\mathcal L_{\text{manifold}}
\right].
}
\]

This is a more principled version of saying:

> "steer behavior without pushing the residual state far outside the local activation distribution."

---

# 96. Regime-conditioned selectivity objective

For regime \(r\):

\[
\boxed{
\delta h_r^*
=
\arg\max_{\delta h}
\left[
g_{T,r}^\top\delta h
-
\lambda
\sum_{u\neq T}
\|
J_{u,r}\delta h
\|_2^2
-
\gamma
(
h+\delta h-\mu_r
)^\top
\Sigma_r^{-1}
(
h+\delta h-\mu_r
)
\right].
}
\]

This combines three major ideas from the conversation:

1. target causal effect;
2. collateral-behavior suppression;
3. local-regime geometric plausibility.

This is a new implementation-ready formulation generated from the chat.

---

# 97. Attention-aware steering objective

If attention routing is part of the causal pathway, add a term that preserves or modifies specific attention behavior.

Let:

\[
A(h)
\]

be the attention map induced by residual state \(h\).

For desired attention template:

\[
A_r^*,
\]

optimize:

\[
\boxed{
\mathcal L_{\text{attn}}
=
D_{\text{JS}}
(
A(h+\delta h),
A_r^*
).
}
\]

Then:

\[
\boxed{
\delta h^*
=
\arg\min_{\delta h}
\left[
-\text{target behavioral gain}
+
\lambda
\text{collateral loss}
+
\gamma
\mathcal L_{\text{manifold}}
+
\beta
\mathcal L_{\text{attn}}
\right].
}
\]

This explicitly ties the local activation-distribution hypothesis to the attention mechanism.

---

# 98. Does attention infer the HMM regime?

The Transformer is not explicitly running the HMM defined above.

But we can test whether its attention patterns contain enough information to infer the same regime.

Let:

\[
A_{\tau,\ell,h}
\]

be one attention head's matrix/row features.

Fit:

\[
\boxed{
\hat z_\tau
=
g(
A_{\tau,\ell,h}
)
}
\]

and compare with regime decoding from residual states:

\[
\hat z_\tau
=
f(
h_{\tau,\ell,p}
).
\]

Questions:

1. Does attention predict \(z_\tau\)?
2. Does attention add information beyond \(h_\tau\)?
3. Does intervention on the residual regime direction predictably change attention?
4. Does restoring attention remove the behavioral intervention effect?

Information-theoretically:

\[
I(
z;
A
)
\]

and:

\[
I(
z;
A
\mid
h
)
\]

are the quantities of interest.

If:

\[
I(
z;
A
\mid
h
)
\approx0,
\]

attention may mainly reflect regime information already present in the residual state.

---

# 99. Explicit Markov-chain / Othello / attention bridge

The full proposed causal/statistical picture can be written:

\[
\boxed{
s_\tau
\rightarrow
b_\tau
\rightarrow
h_{\tau,\ell,p}
\rightarrow
(Q,K,V)
\rightarrow
A_{\tau,\ell}
\rightarrow
a_\tau
\rightarrow
s_{\tau+1}.
}
\]

Where:

- \(s_\tau\): true physical state;
- \(b_\tau\): ideal belief state;
- \(h_{\tau,\ell,p}\): learned neural representation of that state;
- \(Q,K,V\): attention projections;
- \(A_{\tau,\ell}\): routing pattern;
- \(a_\tau\): robot action;
- \(s_{\tau+1}\): next physical state.

The hidden-state transition is:

\[
P(
z_{\tau+1}
\mid
z_\tau,a_\tau
).
\]

The observation update is:

\[
P(
z_{\tau+1}
\mid
z_\tau,a_\tau,o_{\tau+1}
).
\]

The Othello analogy corresponds to an almost perfectly observed/discrete version where:

\[
b_\tau
\]

collapses onto a known board state, while the interesting question becomes the **coordinate system used to represent that state**.

The VLA version adds:

- uncertainty;
- continuous geometry;
- action-conditioned transitions;
- multimodal attention;
- local control.

---

# 100. Proposed falsifiable hypothesis set

## H1 — Multimodal regime geometry exists

At fixed:

\[
(\ell,p),
\]

one Gaussian is worse than a multi-regime model:

\[
\mathcal L_{\text{heldout}}(
\text{GMM/HMM}
)
>
\mathcal L_{\text{heldout}}(
\text{single Gaussian}
).
\]

## H2 — Sequence structure matters

\[
\mathcal L_{\text{heldout}}(
\text{HMM}
)
>
\mathcal L_{\text{heldout}}(
\text{GMM}
).
\]

## H3 — Regimes have different success/failure operators

\[
D(
C_{\text{steer},r},
C_{\text{steer},s}
)
>
\text{null}.
\]

## H4 — Attention is regime-dependent

\[
D_{\text{JS}}
(
A^{(r)},
A^{(s)}
)
>
\text{null}.
\]

## H5 — Q/K geometry explains part of that difference

Predicted regime-conditioned score moments from:

\[
(
\mu_r,\Sigma_r
)
\]

match empirical attention-logit statistics.

## H6 — Local steering is more selective

\[
\text{target gain}_{\text{local}}
>
\text{target gain}_{\text{global}}
\]

at matched collateral degradation.

## H7 — Belief-conditioned steering beats hard state steering when regime uncertainty is high

Compare:

\[
M_{\arg\max b}
\]

versus:

\[
\sum_r
b(r)M_r.
\]

## H8 — Local coordinate transport exists

Held-out:

\[
\cos(
Q_{r\rightarrow s}v_*^r,
v_*^s
)
\]

beats identity/random baselines.

## H9 — Relational/agent-centric coordinates are more model-native than absolute coordinates

Probe/readout performance:

\[
R^2_{\text{relational}}
>
R^2_{\text{absolute}}.
\]

## H10 — Pattern separation is predictive rather than merely descriptive

Layer-wise separation of matched lures predicts future trajectory divergence and action choice.

---

# 101. Recommended implementation order for Codex

Do **not** implement every novel component simultaneously.

## Phase 1 — Establish geometry

1. pick one behaviorally valid task/checkpoint;
2. fix one or a small number of promising \((\ell,p)\);
3. collect rollout activation sequences;
4. fit one Gaussian, GMM, and simple pre/post-contact split;
5. compare centers, covariances, principal angles, Wasserstein distance;
6. determine whether multimodal/local geometry actually exists.

## Phase 2 — Establish sequence dependence

7. fit HMM;
8. compare held-out likelihood/prediction to GMM and time bins;
9. inspect transition matrix;
10. compare inferred regimes with physical events.

## Phase 3 — Connect geometry to attention

11. collect per-head Q/K and attention;
12. measure regime-conditioned Q/K moments;
13. compare empirical attention-score means/variances with the formulas above;
14. measure JS divergence of attention patterns;
15. identify heads with high \(I(z;A)\).

## Phase 4 — Establish local control

16. fit \(C_{s,r},C_{f,r}\);
17. compare local conceptors against global COAST-style conceptor;
18. test center-only vs covariance-only vs center+covariance;
19. evaluate target gain and collateral task loss.

## Phase 5 — Add uncertainty-aware control

20. compute HMM belief \(b_\tau\);
21. compare hard versus soft belief-conditioned steering;
22. optionally implement Gaussian mixture energy steering.

## Phase 6 — Add transport only if local bases rotate

23. identify corresponding anchor features across regimes;
24. fit Procrustes \(Q_{r\rightarrow s}\);
25. evaluate held-out feature/subspace transport;
26. test transported conceptors:

\[
QC_rQ^\top.
\]

---

# 102. Null results that should terminate branches

To avoid another "Frankenstein monster," stop specific branches when:

- GMM does not beat one Gaussian → no evidence for multiple local activation distributions.
- HMM does not beat GMM/time bins → no evidence that Markov transition structure is useful.
- local conceptors do not differ → no reason for regime-conditioned COAST.
- attention distributions do not differ by regime → do not build an attention-mediated story.
- Q/K moment predictions fail badly → the simple Gaussian propagation story is insufficient.
- Procrustes does not generalize to held-out features → no evidence for simple relation-preserving transport.
- local steering does not improve selectivity → global geometry may already be adequate.
- relational coordinates do not decode better than absolute variables → do not overfit the Othello analogy.

These are valuable scientific outcomes, not implementation failures.

---

# 103. Compact new-equation index

### Markov-like residual-state criterion

\[
P(
h_{\tau+1}
\mid
h_{1:\tau},a_\tau
)
\approx
P(
h_{\tau+1}
\mid
h_\tau,a_\tau
)
\]

### Conditional-information criterion

\[
I(
h_{\tau+1};
h_{1:\tau-1}
\mid
h_\tau,a_\tau
)
\approx0
\]

### HMM belief recursion

\[
b_\tau(r)
\propto
P(
x_\tau
\mid
z_\tau=r
)
\sum_q
A_{qr}
b_{\tau-1}(q)
\]

### Regime-conditioned sparse latent

\[
h_\tau
=
\mu_{z_\tau}
+
W_{z_\tau}f_\tau
+
\epsilon_\tau
\]

### Gaussian Wasserstein distance

\[
W_2^2
=
\|
\mu_r-\mu_s
\|^2
+
\operatorname{tr}
\left(
\Sigma_r+\Sigma_s
-
2
(
\Sigma_s^{1/2}
\Sigma_r
\Sigma_s^{1/2}
)^{1/2}
\right)
\]

### Regime conceptor

\[
C_{s,r}
=
R_{s,r}
(
R_{s,r}+\alpha^{-2}I
)^{-1}
\]

### Local contrastive conceptor

\[
C_{\text{steer},r}
=
C_{s,r}
\wedge
\neg C_{f,r}
\]

### Belief-conditioned operator

\[
M_\tau
=
\sum_r
b_\tau(r)M_r
\]

### Affine local steering

\[
h_\tau'
=
\mu_\tau
+
M_\tau
(
h_\tau-\mu_\tau
)
\]

### Mixture energy

\[
E_{\text{mix}}
=
-
\log
\sum_r
b(r)
e^{-E_r(h)}
\]

### Mixture-energy responsibility

\[
\rho_r
=
\frac{
b(r)e^{-E_r(h)}
}{
\sum_jb(j)e^{-E_j(h)}
}
\]

### Regime-conditioned query distribution

\[
q\mid r
\sim
\mathcal N(
W_Q\mu_r,
W_Q\Sigma_rW_Q^\top
)
\]

### Expected regime-conditioned attention logit

\[
\mathbb E[
s_{ij}
\mid
r,s
]
=
\mu_r^\top
\frac{
W_Q^\top W_K
}{
\sqrt{d_k}
}
\mu_s
\]

### Regime-conditioned attention-logit variance

\[
\operatorname{Var}
(
s_{ij}
\mid
r,s
)
=
\operatorname{tr}
(
A\Sigma_sA^\top\Sigma_r
)
+
\mu_r^\top
A\Sigma_sA^\top
\mu_r
+
\mu_s^\top
A^\top\Sigma_rA
\mu_s
\]

with:

\[
A
=
\frac{
W_Q^\top W_K
}{
\sqrt{d_k}
}.
\]

### Fisher pattern-separation direction

\[
v^*
=
\arg\max_v
\frac{
v^\top S_Bv
}{
v^\top S_Wv
}
\]

### Procrustes transport

\[
Q^*
=
\arg\min_Q
\|
QX_A-X_B
\|_F^2
\quad
\text{s.t.}
\quad
Q^\top Q=I
\]

### Transported conceptor

\[
C_B
\approx
QC_AQ^\top
\]

### Selective local control objective

\[
\delta h_r^*
=
\arg\max_{\delta h}
\left[
g_{T,r}^\top\delta h
-
\lambda
\sum_{u\neq T}
\|
J_{u,r}\delta h
\|^2
-
\gamma
(
h+\delta h-\mu_r
)^\top
\Sigma_r^{-1}
(
h+\delta h-\mu_r
)
\right]
\]

---


---

# Direct paper links for Codex

- Jaeger (2014), **Controlling Recurrent Neural Networks by Conceptors** — https://arxiv.org/abs/1403.3369
- Vaswani et al. (2017), **Attention Is All You Need** — https://arxiv.org/abs/1706.03762
- Ramsauer et al. (2020), **Hopfield Networks is All You Need** — https://arxiv.org/abs/2008.02217
- Nanda, Lee & Wattenberg (2023), **Emergent Linear Representations in World Models of Self-Supervised Sequence Models** — https://arxiv.org/abs/2309.00941
- LeCun (2022), **A Path Towards Autonomous Machine Intelligence** — https://openreview.net/forum?id=BZ5a1r-kVsf
- Caucheteux & King (2022), **Brains and algorithms partially converge in natural language processing** — https://www.nature.com/articles/s42003-022-03036-1
- Stark, Kirwan & Stark (2019), **Mnemonic Similarity Task: A Tool for Assessing Hippocampal Integrity** — https://pmc.ncbi.nlm.nih.gov/articles/PMC6991464/
- Kim et al. (2024), **OpenVLA: An Open-Source Vision-Language-Action Model** — https://arxiv.org/abs/2406.09246
- Black et al. (2024), **\(\pi_0\): A Vision-Language-Action Flow Model for General Robot Control** — https://arxiv.org/abs/2410.24164
- **Next-Latent Prediction Transformers Learn Compact World Models** — https://arxiv.org/abs/2511.05963
- Miao, Cho & Ungar (2026), **CORAL: Correctness-Optimized Residual Activation Lens** — https://arxiv.org/abs/2602.06022
- Kuang et al. (2026), **Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations** — https://arxiv.org/abs/2602.01456
- Grant, Zhao & Wang (2026), **Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models** — https://arxiv.org/abs/2603.19233
- Triantafyllopoulos et al. (2026), **Conceptors for Semantic Steering** — https://arxiv.org/abs/2605.04980
- Miao et al. (2026), **Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States** — https://arxiv.org/abs/2605.17144
- Hu, Niu & Varma (2026), **Language Models Represent and Transform Concepts with Shared Geometry** — https://arxiv.org/abs/2607.04525

