# JEPA-WM causal action-mediation result

Verified **48 independent trajectory snapshots** across collection seeds 1–3. Action-pair rows were averaged within snapshots before inference.

| Block | Dose | Transfer: patch | Transfer: sham | Paired difference (95% CI) | Cosine paired difference (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.25 | 0.218 | 0.006 | 0.212 [0.198, 0.225] | 0.877 [0.821, 0.933] |
| 2 | 0.50 | 0.450 | 0.020 | 0.430 [0.404, 0.455] | 0.893 [0.839, 0.948] |
| 2 | 1.00 | 0.909 | 0.065 | 0.844 [0.798, 0.890] | 0.898 [0.848, 0.948] |
| 3 | 0.25 | 0.238 | -0.001 | 0.239 [0.234, 0.243] | 0.946 [0.919, 0.972] |
| 3 | 0.50 | 0.490 | 0.002 | 0.488 [0.480, 0.496] | 0.962 [0.936, 0.987] |
| 3 | 1.00 | 0.981 | 0.018 | 0.963 [0.946, 0.979] | 0.964 [0.940, 0.987] |
| 5 | 0.25 | 0.249 | 0.000 | 0.249 [0.247, 0.251] | 0.997 [0.990, 1.003] |
| 5 | 0.50 | 0.500 | 0.002 | 0.498 [0.494, 0.502] | 0.995 [0.988, 1.002] |
| 5 | 1.00 | 1.000 | 0.009 | 0.991 [0.984, 0.999] | 0.992 [0.985, 0.999] |

**Causal gate: PASS.**

Block 5 is a near-final-block transfer sanity check, not mechanism-selective evidence. Passing this gate establishes that an intermediate predictor representation causally mediates action-conditioned latent predictions; it does not yet establish closed-loop task improvement.
