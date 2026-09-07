# Math insights for VLA Sonar experiments

## Scope

This note transfers the useful mathematical lessons from the world-model audit back to the VLA project. It is prospective: it must not alter an already frozen or running protected VLA cohort. The central research target remains behavioral improvement over the unsteered VLA; geometry, probes, HMMs, and attention analyses are mechanisms and operator-design tools, not substitute endpoints.

The strongest recipe is not “add every sophisticated method.” It is a sequence of falsifiable links:

\[
\text{behaviorally eligible checkpoint/task}
\rightarrow
\text{model-native coordinate}
\rightarrow
\text{stable local geometry}
\rightarrow
\text{supported bounded edit}
\rightarrow
\text{paired behavioral gain}.
\]

## 1. The Othello lesson: search for the model's coordinates

The important trick in OthelloGPT was not merely using a relational label. It compared the **same board variable** under two coordinate systems:

\[
\text{Black/White/Empty}
\quad\text{versus}\quad
\text{Mine/Yours/Empty given the current player}.
\]

The second frame changes with context and made the representation much more linearly accessible. A faithful VLA analogue should therefore compare the same physical quantity under competing frames, for example

\[
x_{ee}^{world}
\quad\text{versus}\quad
R_{object}^{\top}(x_{ee}-x_{object}),
\]

or

\[
x_{object}^{world}
\quad\text{versus}\quad
R_{goal}^{\top}(x_{object}-x_{goal}).
\]

Useful candidate frames include world/camera, robot-base, end-effector, object, goal, instruction-referenced object, and current-plan or waypoint-relative coordinates. Relative rotations should use a stable representation such as a 6-D rotation encoding rather than discontinuous Euler angles.

Three nuances are essential:

1. **A fixed affine transform is not a new ontology.** If the reference is constant, `wall - hand` is only a translation/sign change of `hand`. A centered linear probe should perform equivalently. It cannot support a model-native-coordinate claim.
2. **Binding must change across held-out conditions.** Hold out objects, goals, scenes, instructions, or viewpoint contexts so the decoder must generalize the relational rule. Random frame-level splits only test interpolation.
3. **Decodability only nominates a coordinate.** A coordinate becomes causally relevant only when an edit along its stable decoder row space changes the intended action or outcome more selectively than shuffled-reference, fixed-affine, and orthogonal-complement controls.

The clean metrics are cross-condition generalization performance (CCGP), held-out normalized error, row-space/principal-angle stability, and a matched causal edit. Compare identical target dimensionality and loss normalization; do not average unrelated coordinate errors into one “relational” score.

## 2. Preserve the VLA's actual axes

A VLA activation is not one unordered matrix. Depending on the architecture it may have axes for layer, modality/token position, action token, chunk position, diffusion or flow step, and physical policy call. These axes have different meanings and must be tagged before pooling.

In particular:

- diffusion/denoising steps are iterative inference time, not physical environment time;
- action-token or chunk position is not an independent episode;
- physical policy calls/replans are the chronological sequence relevant to an HMM;
- vision-language tokens and action-expert tokens may occupy different pathways;
- a mean over denoising states is not automatically a valid fit population for an edit applied to each individual denoising state.

Fit/apply equivalence must be checked at the exact hook population. If the intervention is applied at every denoising step and action token, capture and audit those same states, not only the final action or a mean representation. Compare density shift, regime agreement, edit cosine/norm, and downstream action effect between fit and runtime populations.

### Cross-attention-weighted geometry

LAVLA's useful VLA-specific idea is to treat decoder cross-attention as a relevance operator. For action queries and vision-language keys,

\[
\alpha=\operatorname{softmax}\!\left(\frac{Q_AK_{VL}^{\top}}{\sqrt{d_k}}\right),
\qquad
w_j=\mathbb E_{\text{heads, action tokens}}[\alpha_{\cdot j}],
\qquad
S'_{VL,j}=w_jS_{VL,j}.
\]

This can define a decoder-consulted representation before PCA, clustering, or conceptor fitting. But attention weight is not causal importance. It should be compared with uniform pooling, gradient/action-sensitivity weighting, and a weight-permuted control. Attention restoration or output patching is needed for a mediation claim.

## 3. COAST plus Sonar: preserve the operator while changing locality

For outcome class \(y\) and regime \(r\), fit episode-weighted, separately centered covariance

\[
\mu_{y,r}=\frac{\sum_iw_{ir}h_i}{\sum_iw_{ir}},
\qquad
R_{y,r}=\frac{\sum_iw_{ir}(h_i-\mu_{y,r})(h_i-\mu_{y,r})^\top}{\sum_iw_{ir}}.
\]

The conceptor is the soft spectral filter

\[
C_{y,r}=R_{y,r}(R_{y,r}+\alpha^{-2}I)^{-1}
=U\operatorname{diag}\!\left(\frac{\lambda_i}{\lambda_i+\alpha^{-2}}\right)U^\top.
\]

Unlike a hard top-\(k\) PCA projector, it retains the whole eigenspectrum with graded weights. Form the local success-critical operator with Moore–Penrose Boolean algebra:

\[
D_r=C_{success,r}\wedge\neg C_{failure,r}
=\operatorname{pinv}\!\left[
\operatorname{pinv}(C_{success,r})+
\operatorname{pinv}(I-C_{failure,r})-I
\right].
\]

The clean comparison changes one thing at a time:

\[
D_{global},
\qquad
D_t^{static}=\sum_rq_t(r)D_r,
\qquad
D_t^{HMM}=\sum_rb_t(r)D_r,
\]

\[
h'_t=h_t\left[(1-\beta)I+\beta D_t\right]^\top.
\]

Here \(q_t\) is an emission-only regime responsibility and \(b_t\) is a causal filtered belief. The static and HMM arms must share the same local matrices, aperture, site, token population, support rule, and dose. Their difference then isolates temporal routing. HMM is not an edit by itself.

Implementation details that matter:

- center success and failure separately when fitting covariance;
- apply the published multiplicative map about the origin unless an affine-center arm is explicitly named;
- expose the pseudoinverse cutoff and fail on an invalid spectrum instead of silently clipping it;
- report effective rank, quota, condition number, and episode-bootstrap operator stability;
- call this a COAST equation reference unless the authors' exact architecture, sites, preprocessing, and code are reproduced.

A regime-conditioned matched-spectrum sham

\[
\widetilde D_r=Q_r\Lambda_rQ_r^\top
\]

preserves eigenvalues, rank, trace, norm, routing, and nominal dose while destroying learned orientation. It is much stronger than an arbitrary random matrix.

## 4. Keep the probability objects separate

VLA experiments can contain at least five different probability-like quantities:

\[
\begin{aligned}
p(h_t\mid z_t=r) &\quad&\text{activation emission/density},\\
q_t(r)=p(z_t=r\mid h_t) &&\text{static regime responsibility},\\
b_t(r)=p(z_t=r\mid h_{1:t}) &&\text{causal filtered regime belief},\\
p(z_{t+1}\mid z_t,a_t,\Delta t_t) &&\text{transition law},\\
p(j\mid i) &&\text{attention allocation}.
\end{aligned}
\]

The supervised outcome estimate \(p(Y=success\mid h_{1:t})\) is a sixth object. JEPA compatibility energy, density energy, attention entropy, HMM entropy, sparsity, and outcome confidence must never be used as synonyms.

## 5. When an HMM is justified

A GMM describes local activation distributions without order:

\[
p(h_t)=\sum_r\pi_r p(h_t\mid z_t=r).
\]

An HMM adds a chronological transition constraint:

\[
b_t(r)\propto p(h_t\mid r)\sum_qb_{t-1}(q)A_{qr}.
\]

Use physical policy calls as \(t\), never diffusion iterations or CEM optimizer iterations. Fit transitions with forward-backward joint posteriors

\[
\xi_t(q,r)=p(z_t=q,z_{t+1}=r\mid h_{1:T}),
\]

and give each episode unit transition mass so long failures do not dominate merely by containing more steps.

The claim ladder should be strict:

1. One Gaussian versus static GMM establishes whether stable multimodality exists.
2. HMM versus one Gaussian, static GMM, and strong progress/event bins tests whether order adds held-out sequence information.
3. Older-history prediction of innovations can reject current-state Markov sufficiency; failure to reject does not prove sufficiency.
4. In robotics, compare fixed transitions with action- and interval-conditioned transitions

   \[
   A_t=A_\phi(a_t,\Delta t_t).
   \]

5. Compare chronological matrix composition with independently fitted multi-horizon prediction; algebraically multiplying a fixed matrix by itself is not an empirical Chapman–Kolmogorov test.
6. Require HMM routing to improve held-out outcome prediction over static routing using the same local operator bank.
7. Require treatment separation in operator and executed-action space. If HMM and static produce nearly identical edits/actions, their behavioral comparison is not an informative test.

Online control uses filtered beliefs only. Smoothed posteriors \(p(z_t\mid h_{1:T})\) use future states and are restricted to offline analysis.

## 6. Gaussian energy and exact bounded control

If local activation clouds are adequately modeled by Gaussians, use a coherent mixture density rather than manually averaging component energies:

\[
E_y(x)=-\log\sum_r\pi_{y,r}\mathcal N(x;\mu_{y,r},\Sigma_{y,r}).
\]

Its exact gradient is responsibility weighted:

\[
\nabla E_y(x)=\sum_r\gamma_{y,r}(x)\Sigma_{y,r}^{-1}(x-\mu_{y,r}),
\qquad
\gamma_{y,r}(x)=p(r\mid x,y).
\]

For the contrast \(g=\nabla(E_{success}-E_{failure})\), define a positive-definite metric \(G\) that can include whitening, action sensitivity, and local-support curvature. The magnitude-preserving capped quadratic step is

\[
\delta_0=-\eta G^{-1}g,
\qquad
\delta=\delta_0\min\!\left(1,
\frac{\epsilon}{\sqrt{\delta_0^\top G\delta_0}}
\right).
\]

This solves

\[
\min_\delta g^\top\delta+\frac{1}{2\eta}\delta^\top G\delta
\quad\text{subject to}\quad
\delta^\top G\delta\leq\epsilon^2.
\]

The boundary-only linear problem always uses the full radius for every nonzero gradient. It should be a named ablation, not silently called the capped quadratic solution. Euclidean radial clipping of an anisotropic solution is not an exact anisotropic trust-region solve.

Log cap-hit rate, support abstention, edit norm, and executed-action effect. If almost every state hits the cap, the field has become mostly a constant-length direction field and no longer tests whether energy/confidence magnitude matters.

Conceptor stability does not license energy or optimal transport. Bootstrap the actual precision matrices, energy-gradient field, and OT displacement field separately over whole episodes.

## 7. Comparing probability geometry between subspaces

For two regime/outcome clouds, useful comparisons answer different questions:

- center shift: \(\|\mu_1-\mu_2\|\);
- covariance orientation: principal angles or Grassmann distance;
- covariance-weighted overlap: normalized conceptor trace overlap;
- scale-normalized contrast: generalized eigenproblem

  \[
  \Sigma_1v=\lambda(\Sigma_2+\epsilon I)v;
  \]

- joint center/covariance shift: Gaussian 2-Wasserstein distance

  \[
  W_2^2=\|\mu_1-\mu_2\|^2+
  \operatorname{tr}\!\left(
  \Sigma_1+\Sigma_2-2(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2})^{1/2}
  \right);
  \]

- matched-row representational similarity: linear CKA;
- support/tail mismatch: held-out likelihood, calibration, and Mahalanobis-tail diagnostics.

Raw covariance magnitude depends on feature scale. Correlation is the appropriate pairwise scale-free measure; PCA/conceptors use the full covariance eigensystem to recover ellipsoid axes and strengths. High variance is not semantics, and Gaussian regularization is not evidence that the true distribution is Gaussian.

Different activation means/covariances can induce different attention-score distributions after the learned \(Q/K\) projections. That is a testable propagation hypothesis, not proof that attention explicitly infers the regime. Estimate the actual attention distributions and use matched residual controls.

## 8. Direction versus magnitude

GIFT motivates a clean geometric decomposition. For factor state \(x\),

\[
r=\|x\|,
\qquad
u=x/\max(r,\epsilon),
\qquad
\rho=\log(1+r).
\]

Decompose a proposed edit into

\[
\delta_{rad}=(u^\top\delta)u,
\qquad
\delta_{ang}=\delta-\delta_{rad}.
\]

Compare angular-only, radial-only, and joint edits after matching their metric norm and action-space dose. This can tell whether behavior depends on orientation, activation magnitude, or both. GIFT's teacher losses and privileged geometry are training interventions; they should not be described as part of a frozen-model Sonar edit.

## 9. Transport across regimes, layers, and models

If the same concept rotates between contexts, raw vector cosine can look poor even when relational geometry is preserved. With matched anchors \(X_A,X_B\), fit orthogonal Procrustes

\[
Q^*=\arg\min_{Q^\top Q=I}\|X_AQ-X_B\|_F^2.
\]

Then test on held-out anchors before transporting centers or conceptors:

\[
\mu_B\approx Q^\top\mu_A,
\qquad
C_B\approx Q^\top C_AQ.
\]

Procrustes, CKA, and Grassmann distance are descriptive alignment tools. Transport becomes causal only when a held-out transported edit works and beats identity, random-orthogonal, and independently fitted target-model controls. CKA requires matched rows; it is invalid for unrelated regime samples.

## 10. Behavioral design and controls

The most consequential mathematics is often experimental bookkeeping:

- choose a checkpoint/task distribution with a real native mixture of successes and failures and headroom for improvement;
- use the VLA's native simulator/state-derived success as primary, not an unaudited VLM judge;
- freeze task, checkpoint, evaluator, planner/control compute, initial states, goals, environment seeds, and policy-noise seeds;
- split development, fitting, and protected evaluation by independent episodes or scene clusters;
- use episodes, not frames/tokens/denoising steps, as the independent statistical unit;
- never replace failed or successful episodes after observing outcomes;
- fit regimes without outcome labels, then fit outcome geometry afterward;
- use a common causal pre-outcome window and test early/late stability to reduce reverse causality;
- require effective episode support per outcome and regime, not only thousands of correlated activations;
- match dose in executed-action space, not only latent norm;
- freeze a separate hash-bound arm registry before protected evaluation.

The baseline hierarchy should be:

1. selected VLA intervention versus unsteered checkpoint;
2. selected intervention versus disclosed COAST equation reference;
3. HMM versus static routing of the same local operator, if eligible;
4. identity, reverse-sign, label-shuffled, matched-spectrum/orientation, and equal-compute controls.

COAST is a useful VLA operator reference, not a universal source of performance metrics or sample-size guarantees. Its task/checkpoint numbers should not become the target for a different VLA architecture or benchmark.

## 11. Minimal acceptance checklist

Before a new VLA intervention cohort opens, require all applicable checks below:

- [ ] Native checkpoint/task behavioral eligibility passes.
- [ ] Exact tensor axes and hook lifecycle are documented.
- [ ] Fit and runtime populations match at each edited layer/token/denoising/policy-call state.
- [ ] Coordinate comparison uses matched variables, changing frames, held-out contexts, and binding shams.
- [ ] `K=2` beats `K=1` and progress/event controls and is stable under episode bootstrap; otherwise use `K=1`.
- [ ] Success/failure conceptors use separate centers and stable Moore–Penrose algebra.
- [ ] Energy, precision, and OT fields pass their own stability/calibration gates.
- [ ] HMM uses proper `xi`, causal filtering, action-conditioned tests, and beats static outcome routing.
- [ ] HMM and static treatments produce meaningfully different operators and actions.
- [ ] Action-space dose, cap saturation, support abstention, and unrelated-behavior effects are frozen.
- [ ] Identity/action hashes and paired stimulus hashes pass.
- [ ] The final arm registry is written before protected outcomes are opened.

If a gate fails, the right response is to simplify or reassess the causal chain—not to add HMMs, energies, attention weighting, or transport as a post-hoc rescue.

## Source notes

- [Math techniques](../mechinterp-vla/math_techniques.md)
- [Representational geometry paper concepts](../mechinterp-vla/representational_geometry_paper_concepts.md)
- [VLA/world-model research thread](<../mechinterp-vla/vla_world_model_research_thread_codex (1).md>)
- [Cross-model plan](<../mechinterp-vla/cross model plan.md>)
- [World-model paper-to-code audit](docs/WORLD_MODEL_MATH_CODE_AUDIT_2026-09-04.md)
