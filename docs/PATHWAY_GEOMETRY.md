# What the archived pathway and geometry tests actually identify

The archives support two concrete findings. Visual/action output nonadditivity is measurable separately from the cross-term introduced by squared loss. Local action interpolation also has a strong precision-dependent reconstruction result, but this does not translate into a comparably large, consistent recorded-action forecast improvement. Neither finding identifies a physical coordinate system, a dense manifold, or an effective robot controller.

This is a fixed exploratory CPU reanalysis, not a new model experiment or confirmation run. The [frozen protocol](../paper/data/pathway_geometry_protocol.json) enumerates all 17 registered coupling/geometry sources before the recount. The [receipt](../paper/data/pathway_geometry.json) binds the protocol, analysis source, raw report hashes, archive hash, and six derived tables. No GPU calls, refitting, outcome-selected subsets, or commits were used.

## Coverage and checks

The unit is the saved lineage group, not an individual window or action candidate. Reach, Reach-Wall, Push-T, Wall, and PointMaze have respectively 33, 27, 21, 192, and 200 lineages per available precision. Geometry covers all five tasks in BF16 and FP32. Coupling covers Reach, Reach-Wall, and Push-T in both precisions, plus BF16 Wall. The frozen registry contains no PointMaze coupling or FP32 Wall coupling source.

All 17 report hashes and protocol bindings match the public source registry; adjacent completion hashes are verified where available. Every exported arm mean matches its raw report and, where retained, the per-lineage recount. Native and zero-dose losses agree at every horizon, and all arms agree before the H3 edit at H1/H2. Duplicate IDs, missing pairs, and nonfinite metrics fail validation. All geometry line-valid and edit-eligible flags are one; no lineage is excluded.

The compact Wall coupling report retains per-lineage mechanism diagnostics but not per-arm lineage losses. Its H1–H6 loss factorial is therefore an aggregate estimate without a fabricated paired interval. Saved output diagnostics still support lineage summaries at H1/H3/H6. Their percentages use the fixed saved native mean because the native lineage denominator is unavailable.

Intervals elsewhere are marginal 95% paired-lineage percentile bootstrap intervals, with 20,000 draws and a fixed seed. Percentages bootstrap numerator and denominator together when both are retained. These are development-data descriptions, not multiplicity-corrected discoveries or independent training-seed replications. The MetaWorld tasks share a checkpoint but use task-specific fits; precision-specific fits and doses are not interchangeable.

## Output interaction is not loss interaction

Let native, visual-only, action-only, and joint outputs be `n`, `v`, `a`, and `j`; let the target be `y`. Define `dv=v−n`, `da=a−n`, and `r=j−v−a+n`. All products below are averaged across output coordinates before the saved lineage aggregation.

`D = MSE(j,y) − MSE(v,y) − MSE(a,y) + MSE(n,y)`

`D = 2 mean(dv*da) + 2 mean((n−y+dv+da)*r) + mean(r²)`

Even exactly additive outputs (`r=0`) can have a nonzero loss factorial through `2 mean(dv*da)`. Conversely, output nonadditivity can be substantial while the loss factorial is small because terms cancel. The archives retain `mean(r²)`, the additive quadratic cross-term, and the nonadditive remainder at H1/H3/H6. They do not retain the output vectors needed to reconstruct the direction of `r`, identify heads, or separate the two contributions inside the remainder. At H2/H4/H5, loss endpoints alone do not recover output nonadditivity.

The table below gives H6 proprioceptive-embedding quantities as percentages of native MSE. `D` is an interaction, not the intervention's net benefit. Positive `mean(r²)` measures nonadditivity, not successful control.

| Task / precision | Loss factorial D | Output mean(r²) | Quadratic cross | Nonadditive remainder |
| --- | ---: | ---: | ---: | ---: |
| Reach BF16 | +0.063710% | 0.419094% | +0.213827% | −0.150116% |
| Reach FP32 | +0.012203% | 0.000005% | +0.005026% | +0.007177% |
| Reach-Wall BF16 | +0.078028% | 0.511703% | +0.293606% | −0.215578% |
| Reach-Wall FP32 | +0.044801% | 0.000011% | +0.034126% | +0.010674% |
| Push-T BF16 | −0.048573% | 5.019563% | +2.460483% | −2.509056% |
| Push-T FP32 | −0.029321% | 0.002559% | −0.037744% | +0.008423% |
| Wall BF16 | +0.001583% | 0.281495% | +0.157518% | −0.155935% |

Source: [coupling summaries](../paper/data/pathway_geometry_coupling_summary.csv), filtered by `modality=proprio`, `horizon=6`. These are the original maximum-covariance visual/action directions: one fitted unit direction per site, with visual-predictor and B3 action-conditioning edits at H3, each dosed at 0.1 fit-split robust score sigma. They are not the later rank-four combined controller. Loss factorials retain both single-component doses in the joint arm. The separate standardized-energy joint scales both components by `1/sqrt(2)` and remains a separate column, alongside its matched-random arm, in the [lineage table](../paper/data/pathway_geometry_coupling_lineages.csv); substituting it for `joint` would change the factorial rather than improve its energy control. BF16 action-only realized-energy tolerance failures also limit a precision-isolated interpretation.

## What the local geometry construction tests

The source changes one H3 normalized-action direction at offsets −0.1, −0.05, +0.05, and +0.1, captures the B3/H3 activation, and reconstructs the omitted native center. The equal-anchor linear estimate averages the four donors. The cubic interpolant evaluates the center with weights `[-1/6, 2/3, 2/3, -1/6]`.

At this symmetric center those weights also equal a quadratic least-squares fit evaluated at zero. Consequently, a favorable result cannot identify third-order dynamics. This is a four-anchor, one-direction local reconstruction test, not a learned dense manifold or a demonstration of model-native physical coordinates. The edit spans the full newest 256-patch field: one dense residual vector is constructed per window and arm, not a shared fitted rank-one operator or a PCA subspace. Its common requested L2 is the fit-only median linear reconstruction residual norm. BF16 donor outputs are cast to FP32 for interpolation, but that cast does not undo upstream BF16 computation.

Three endpoints are kept distinct:

1. Omitted-activation MSE compares the raw reconstructed activation with the native activation; raw interpolation residual norms are unequal.
2. Native-forecast fidelity compares the resulting forecast with the model's native forecast. Both raw and requested-dose-normalized versions are retained; this measures preservation of the model, not physical accuracy.
3. Recorded-future MSE compares the forecast under the original recorded actions with encoded future targets. This is an offline latent error, not replanning, task success, or physical-state error.

All four registered geometry arms are retained: linear, cubic, projected cubic, and reflected curvature. The table and figure emphasize the prespecified cubic-versus-linear comparison; the complete [630-row summary](../paper/data/pathway_geometry_geometry_summary.csv) also reports cubic versus each other control, both modalities, all recorded-loss horizons, and H3/H6 native fidelity.

| Task | BF16 raw reconstruction advantage | FP32 raw reconstruction advantage | BF16 H6 proprio advantage | FP32 H6 proprio advantage |
| --- | ---: | ---: | ---: | ---: |
| Reach | −46.4289% | +94.1933% | +0.009638% | −0.000231% |
| Reach-Wall | −46.7716% | +93.9634% | +0.026390% | +0.000000123% |
| Push-T | −35.1464% | +99.9161% | −0.021238% | +0.005017% |
| Wall | −43.3160% | +99.9748% | −0.006288% | −0.000688% |
| PointMaze | −42.2175% | +99.9858% | +0.015918% | −0.014072% |

Positive means cubic has lower error. Raw reconstruction divides `linear−cubic` by linear reconstruction error; H6 future prediction divides `linear−cubic` by native H6 proprioceptive-embedding MSE. They are deliberately different denominators, not comparable effect-size scales. Query `control=equal_anchor_linear`, `intervention=cubic`, with metrics `omitted_activation_mse` and `recorded_proprio_mse_h6`.

![Precision-dependent local reconstruction and small, mixed forecast effects](figures/pathway_geometry_precision.png)

Native-forecast fidelity also need not follow raw reconstruction after normalization. For FP32 Push-T, raw cubic H6 proprio native-fidelity error is 99.89% lower than linear, but the requested-dose-normalized cubic native-fidelity error is 16.03% higher. This is a model-fidelity contrast, not a 16% change in physical prediction or success.

## Dose and precision caveats

The geometry arms have matched requested dose within a task/precision cell; this must not be described as exact matched realized energy. The archive checks each window with `abs(realized−requested) <= 1e−5 + 1e−3*abs(requested)`. FP32 Reach and Reach-Wall cubic pass fractions are 49.02% and 50.74%, versus 98.08% and 98.36% for linear. Their mean realized cubic L2 differs from mean requested L2 by only about 0.0106% and 0.0805%, but mean agreement does not establish per-window matching. The other eight geometry sources report full pass fractions for these arms.

Across precision, requested doses differ greatly: Reach cubic mean L2 is 7.4422 in BF16 versus 0.0065586 in FP32; Reach-Wall is 7.6218 versus 0.0064007. Thus the downstream precision contrast changes both numerical computation and delivered perturbation scale, alongside precision-specific fitted banks. The raw reconstruction reversal itself occurs before dose normalization, but its explanation is not isolated by these archives. Amplification of numerical noise by interpolation weights is a falsifiable candidate explanation, not a measured mechanism.

## Association, limits, and finite next tests

The [association table](../paper/data/pathway_geometry_associations.csv) contains every prespecified task/precision/modality/H3-or-H6 Pearson and Spearman comparison between cubic-minus-linear reconstruction error and cubic-minus-linear recorded-future error. Signs and magnitudes vary; no common monotone relationship is established. These are descriptive lineage correlations, not causal mediation estimates. The [cross-precision table](../paper/data/pathway_geometry_cross_precision.csv) pairs exact lineage IDs; matching identities does not remove the fit/dose confounds.

The finite priorities following this recount are:

1. For a common fixed candidate set, preserve full outputs for the native/V/A/joint factorial and repeat under a common requested and audited realized dose. This tests whether the precision-sensitive scalar interaction survives energy control and reveals its output direction.
2. At the same frozen direction and donors, separate model-computation precision from interpolation arithmetic and repeat the center plus held-out off-center coordinates. A shared bank/dose and a quadratic comparator distinguish precision effects from the specific cubic label. Freeze these cells before reading outcomes.
3. Follow the resulting forecast changes into candidate-score margins and rank crossings under one fixed scorer and candidate set before any closed-loop claim. Candidate-score softmax entropy, CEM proposal entropy, attention entropy, and coefficient-spectrum entropy are different measurements and must not be substituted for each other.

No archive-only calculation can establish that this geometry explains the fresh-control result, that a dense nonlinear physical manifold exists, or that mechanistic interpretability generally fails in robotics. Those remain broader hypotheses.

## Reproduce

With the archived sources available, run `.venv/bin/python analysis/mechanism/pathway_geometry.py`. For public-table-only figure rebuilding, append `--plots-only`. The same figure is exported as PNG, SVG, and PDF. Run `.venv/bin/python -m unittest discover -s tests -p 'test_pathway_geometry.py' -v` for the eight algebra, pairing, missing-data, source-binding, and coverage tests.
