is higher covariance mean the features are more highly related? 

Sort of, but covariance is not the cleanest measure of “how related” two features are.

For two features \(X\) and \(Y\),

$$ \operatorname{Cov}(X,Y) = \mathbb E[(X-\mu_X)(Y-\mu_Y)]. $$

So:

large positive covariance → when \(X\) is above its mean, \(Y\) also tends to be above its mean
large negative covariance → when \(X\) is high, \(Y\) tends to be low
covariance near \(0\) → little linear co-variation

But the raw magnitude depends on the scale of the variables. If you multiply \(X\) by 100, the covariance increases by 100 even though the underlying relationship hasn't changed.

That is why correlation is better for “how strongly related?”:

$$ \rho_{XY} = \frac{\operatorname{Cov}(X,Y)} {\sigma_X\sigma_Y}. $$

Correlation is normalized to

$$ -1\leq\rho\leq1. $$

So:

$$ \boxed{\text{covariance = how much they move together}} $$

whereas

$$ \boxed{\text{correlation = how strongly they move together, normalized for scale}}. $$
In COAST specifically

This distinction matters.

COAST computes something like

$$ R=\frac{1}{N}X^\top X $$

after centering the activations.

So \(R\) is the activation covariance matrix.

For two activation dimensions \(i,j\),

$$ R_{ij} = \operatorname{Cov}(h_i,h_j). $$

Large off-diagonal entries mean those activation dimensions tend to vary together across the successful/failure trajectories.

But COAST is not primarily looking for pairs of highly correlated features.

It eigendecomposes the whole covariance structure:

$$ R=U\Lambda U^\top. $$

The eigenvectors \(u_i\) are directions through activation space, and the eigenvalues \(\lambda_i\) tell you:

How much the activations vary along that direction.

So a large \(\lambda_i\) means:

$$ \boxed{\text{this direction captures a lot of the structure/variance in this activation distribution}} $$

not:

“these two features are highly related.”

For example, if several neurons jointly move according to a latent variable like gripper alignment, PCA might find:

$$ v_{\text{alignment}} = 0.4h_1-0.2h_2+0.7h_3+\cdots $$

with a large eigenvalue.

None of the individual neurons necessarily corresponds to “alignment.” It's the joint covariance pattern that reveals the direction.

So the COAST intuition is more:

$$ \text{neurons covary} \rightarrow \text{structured covariance matrix} \rightarrow \text{principal directions/subspace} \rightarrow \text{behaviorally relevant geometry}. $$

That last jump—from pairwise covariance to the geometry of an entire subspace—is the important part.

Wednesday 10:04 PM

is this PCA? So geometrically you can think of the activation cloud as an ellipsoid:

                 activation space

                       /-------
                    /           \
                  /               \
                 |   success       |
                  \   manifold    /
                    \___________/

and C captures both:

the orientation of that ellipsoid;
how strongly each axis matters.

That's much richer than a steering vector.

Yes — that geometric picture is basically PCA-like, but COAST is not just “run PCA and keep the top components.”

PCA starts from the covariance matrix:

$$ R=\frac{1}{N}X^\top X $$

and eigendecomposes it:

$$ R=U\Lambda U^\top. $$

Geometrically:

columns of \(U\) = the ellipsoid’s principal axes
eigenvalues \(\lambda_i\) = how much variance lies along each axis

So your activation cloud:

$$ \text{cloud} \rightarrow \text{ellipsoid} $$

is exactly the standard PCA intuition.

The distinction is what happens next.

Ordinary PCA often uses a hard projection onto the top \(k\) components:

$$ P_k=U_kU_k^\top. $$

That effectively says:

$$ \text{top directions}=1,\qquad \text{everything else}=0. $$

COAST's conceptor instead uses

$$ C=R(R+\alpha^{-2}I)^{-1}. $$

Substitute

$$ R=U\Lambda U^\top $$

and you get

$$ C= U \operatorname{diag} \left( \frac{\lambda_i}{\lambda_i+\alpha^{-2}} \right) U^\top. $$

So each PCA axis gets a soft weight

$$ \mu_i= \frac{\lambda_i}{\lambda_i+\alpha^{-2}} $$

rather than a binary keep/delete decision.

So:

$$ \boxed{\text{PCA: find the axes}} $$

while

$$ \boxed{\text{conceptor: find the axes + softly gate each axis}} $$

For example:

$$ \lambda_1\gg\lambda_2\gg\lambda_3 $$

might produce something like

$$ \mu_1=.98,\quad \mu_2=.74,\quad \mu_3=.21. $$

Then applying \(C\) reshapes a hidden state according to all three axes:

$$ h' = Ch. $$

That's why “soft PCA projector” is a good intuition.

And COAST goes one step beyond even that because it computes separate covariance geometry for success and failure:

$$ C_{\text{success}},\qquad C_{\text{failure}} $$

then constructs something like

$$ C_{\text{success}}\wedge\neg C_{\text{failure}}. $$

So it's not merely asking:

What are the main PCA directions in successful trajectories?

It's closer to:

What multidimensional covariance structure is characteristic of success while excluding geometry also characteristic of failure?

That's the part that makes it more interesting than vanilla PCA.

Yesterday 8:14 AM
VLA Interp Tools

A genuinely relevant new release appeared yesterday: LAVLA — “Latent Cluster Analysis for Vision-Language-Action Models”, submitted September 2, 2026. It analyzes the GR00T N1.5 action decoder layer-by-layer and across all four diffusion timesteps, rather than stopping at the VLM backbone.

The interesting technical trick is cross-attention-weighted latent analysis. They extract the action decoder’s cross-attention matrix,

$$ \alpha=\operatorname{softmax}\left(\frac{Q_AK_{VL}^{T}}{\sqrt{d_k}}\right), $$

average attention over action tokens/heads to derive an importance weight \(w_j\) for each vision-language token, then reweight the embeddings before clustering:

$$ S'_{VL}=S_{VL}\odot W. $$

This is meant to suppress latent dimensions/tokens that the action decoder largely ignores and amplify the ones actually being consulted during action generation. It improves Silhouette score 62.1% without PCA and 24.0% with PCA, with roughly an 8% improvement in Davies–Bouldin score.

For your representation-geometry thread, the most relevant finding is temporal/layer structure: clusters become increasingly organized through the diffusion process, and cross-attention layers appear to repeatedly specialize in particular information across timesteps. They report disentanglement of spatiotemporal and kinematic features, with representations refining through intermediate layers and stabilizing toward the output. They also use Jaccard + overlap coefficients to follow clusters across layers/timesteps and Wasserstein distances to test whether clusters separate quantities like arm displacement and episode time.

Relative to COAST, this is much more descriptive than causal:

$$ \text{LAVLA: activations} \rightarrow \text{attention weighting} \rightarrow \text{PCA optional} \rightarrow \text{clustering} \rightarrow \text{concepts / temporal geometry} $$

versus

$$ \text{COAST: activations} \rightarrow \text{covariance} \rightarrow \text{conceptor subspace} \rightarrow \boxed{\text{causal steering}}. $$

So I would not treat LAVLA as a replacement for COAST or a full mech-interp toolkit. The potentially valuable new ingredient is using cross-attention itself as a relevance operator before doing representation geometry, especially in a diffusion/action decoder where early activations are noisy. The authors explicitly suggest this weighting could also improve other latent-analysis methods such as SAEs.

I did not find a public GitHub implementation linked from the arXiv entry yet.

LAVLA on arXiv

Today 8:23 AM
VLA Interp Tools

Two meaningful new items appeared since the last check.

GIFT — Guided Intermediate Feature Training was submitted September 3, 2026. It is not a mech-interp method in the COAST sense, but it is highly relevant to your representation-geometry/Sonar work because it explicitly shapes intermediate VLA and world-action-model representations rather than only supervising the final action. The framework adds three training objectives to intermediate visual tokens: geometric alignment, object-centric affordance prediction, and goal-region reconstruction, and applies the same principle across a direct-action VLA, a direct-action WAM, and an inverse-dynamics WAM.

The geometry loss is especially interesting. Rather than directly regressing a teacher representation, GIFT decomposes a frozen VGGT feature into direction and magnitude:

$$ d_i=\frac{g_i}{\|g_i\|},\qquad \rho_i=\log(1+\|g_i\|), $$

then separately optimizes angular alignment

$$ \mathcal L_{\rm ang} =\frac1N\sum_i(1-\hat d_i^\top d_i) $$

and scale alignment

$$ \mathcal L_{\rm scale} =\frac1N\sum_i(\hat\rho_i-\rho_i)^2. $$

So it explicitly preserves both orientation in representation space and feature magnitude.

The piece most relevant to your model-native-coordinate idea is its affordance representation. Objects/end-effectors are encoded in anchor-relative and object-relative coordinate frames, including relative translations and 6-D rotations; the authors explicitly say this reduces dependence on the global camera frame. That makes the distinction with Sonar fairly clean:

$$ \text{GIFT: prescribe useful physical coordinates} \rightarrow \text{train representations to encode them} $$

versus

$$ \text{Sonar: propose competing coordinates} \rightarrow \boxed{\text{discover which coordinate the pretrained model already uses}}. $$

That difference is important: GIFT strengthens the evidence that reference-frame choice inside intermediate VLA representations matters, while leaving your discovery problem intact.

Performance is substantive: zero-shot LIBERO-Plus improvements of +4.6, +12.6, and +5.2 pp across its three policy variants, and RoboCasa improvements of +12.6, +9.0, and +8.4 pp. Interestingly, the variants where the auxiliary predictions are not fed to the action generator perform better, suggesting the gains come from changing the internal representation itself rather than supplying extra inference-time information.

GIFT paper
GIFT project page

A secondary release worth noting for the closed-loop evaluation side is FailBench, also submitted September 3. It contains 2,197 robot manipulation attempts from 14 sources and evaluates 13 VLM-based success/failure judges. The best detector reaches only 0.77 balanced accuracy, dropping below 0.60 on some contact-intensive tasks.

That is directly relevant if you use automated evaluators to measure causal interventions: a COAST/Sonar intervention can genuinely change behavior while an external VLM judge mislabels the outcome. FailBench provides evidence that your closed-loop evaluation should ideally combine simulator/state-derived ground truth with video/VLM judging rather than treating the VLM evaluator as truth.

FailBench paper

Of the two, GIFT is the important one for your research direction. The object-relative/reference-frame supervision is unusually close to the coordinate-system question you are trying to turn into a discovery method.
