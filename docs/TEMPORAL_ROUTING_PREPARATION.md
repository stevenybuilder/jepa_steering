# Conditional temporal-routing preparation

September8 update: native128-family fits, the archived-math port, family-disjoint
history diagnostics and causal successor adapter are now implemented. The user's
explicit request for HMM behavioral ablations supersedes the old eligibility-only
admission restriction for the new development panel; no positive history-value
claim follows. See [the separate paired contract](HMM_FIXED_RESPONSE_BEHAVIOR.md).
Full GPU engineering and paired behavioral results remain required.

The governing question is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): does native
imagined history H1/H2 add useful information beyond H2 alone when deciding a fixed
intervention at H3? A reproducible non-routed effect is necessary, not sufficient.
Incremental history value and treatment separation must also be established.

## Implemented native fit-history capture

`offline_study.routing_history` captures the unedited predictor B3 output at H1–H6,
pooling the newest 256 visual tokens in float32. Capture hooks do not alter tensors;
each captured rollout is checked bitwise against a separate uninstrumented forward.
The pooled field retains all 400 predictor features, including proprioceptive
feature conditioning; it is not the 384-dimensional visual encoder output. The
initial v1 runtime correctly stopped on an erroneous 384-feature shape assertion
before producing a completed capture. Its failed receipts are retained. The repair
checks the pinned predictor normalization width and retains all features without
truncation; a regression test covers the actual 400-dimensional shape.
The official corrected offline context, preprocessing, recorded actions and BF16
precision remain fixed. No target-error score is computed by this command.

The population is exactly the already-frozen 128 fitting lineage groups and four
prefixes per group, separately for Reach and Reach-Wall. Four disjoint trajectory
shards per task capture 32 groups / 128 prefixes each, with no padding. Each shard
writes its source/cohort/fit bindings and exact example list before capture, verifies
complete identity coverage and frozen parameters, then hashes the output tensor.
No validation or protected cohort is opened. Push-T has no qualifying non-routed
candidate and is not included in this conditional preparation.

Later H3–H6 features are retained only because the archived fitting method trains
emissions on native imagined sequences; they are forbidden inputs to a decision
made at H3. A prospective eligibility/fit contract must specify that boundary before
any estimator output is used. Successful capture is NOT HMM eligibility or efficacy.

## Still required

The archived reference uses two hidden states, a three-dimensional PCA projection,
diagonal Gaussian emissions, 20 deterministic EM iterations and causal filtering.
Its three reused development initial states are not three hidden regimes or a
current confirmation cohort. It is historical evidence, not a current protocol.

The new fit-only history-value assessment, treatment-separation criterion, comparator
normalization, underlying fixed operator and complete evaluation/analysis contract
still need freezing. No model/rank/regime/dose search is authorized. Constant,
memoryless and HMM gates must share the same underlying edit, with appropriate
matched-random controls and comparable budgets. The exact non-routed candidate
selection is already recorded, but routing qualification remains unresolved.
