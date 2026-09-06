# Identity-contrast geometry under the shared template (cgs-identity-contrast-geometry-v0.1)

Scope: descriptive/associational; template-projected; causal claims remain with patching. Everything is fitted on discovery scenes only (LOSO for every held-out score, the hazard-free Procrustes transport included); the identity-contrast vector I = DiD(ped) - DiD(cone) is algebraically invariant to subtracting one template vector from both identities, so the template removal changes its magnitude bookkeeping and the I_perp variant, not I itself; anti-alignment of I^A with the transported I^B is a geometric consequence of the reversed assignment and does not by itself show that either arm USES that direction.

Dumps: A = `/root/cgs-pilot/artifacts/drive_identity_geometry/self_test/null/arm_A`, B = `/root/cgs-pilot/artifacts/drive_identity_geometry/self_test/null/arm_B`; stimulus `/root/cgs-pilot/artifacts/drive_identity_geometry/self_test/null/stimulus`; 16 discovery scenes, 2 sites, step 0, groups corridor, hazard, hazard_corridor; runtime 9 s.

Prediction (reversed assignment): I^A = DiD(ped) - DiD(cone) in arm A and the transported I^B are ANTI-aligned; each arm's I has a consistent direction across scenes. `cos` = per-scene cosine of I^A with T(I^B) (mean [scene-bootstrap 95 % CI]); `p_perm` = scene-permutation one-sided p (null mean in brackets = shared-direction alignment); `q` = held-out projection onto the other arm's LOSO identity direction with the identity-relabel refit max-T p (registered); `within A/B` = LOSO direction-consistency t with max-T p; `template` = in-sample fraction of the four DiD fields' energy along the shared mean direction; `|I|/|tau|` = median identity-contrast norm over template norm (A / B); `asym` = template-amplitude asymmetry ped - cone in units of the template norm (A / B; predicted + / -); `rank` = hard rank for 80 % energy of the four raw fields -> of the shared-template residuals.

## group `corridor` -- 2 sites, 16 scenes

- I: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.063 .. 0.079
- I_perp_shared: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.065 .. 0.080
- I_perp_perarm: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.065 .. 0.079
- template energy fraction 0.937 .. 0.938; |I|/|tau| (A) 0.247 .. 0.250; template-amplitude asymmetry pattern (A > 0, B < 0) at 0/2 sites

| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \|I\|/\|tau\| | asym A / B | rank raw -> resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L03.mlp_out | 0.063 [-0.006, 0.132] | 0.9522 | 0.9960 [-0.015] | -0.072, -2.51, 0.0649 | 0.03 (0.5534) | 0.26 (0.4625) | 0.065 [-0.003, 0.136], 0.0529 | 0.938 | 0.25 / 0.25 | 0.00 / 0.01 | 1 -> 19 (I: 10/10) |
| L05.resid_post | 0.079 [0.010, 0.151] | 0.9762 | 1.0000 [-0.011] | -0.059, -2.12, 0.0999 | 0.04 (0.5524) | 0.09 (0.5035) | 0.080 [0.009, 0.155], 0.0849 | 0.937 | 0.25 / 0.25 | 0.00 / 0.01 | 1 -> 19 (I: 10/10) |

## group `hazard` -- 2 sites, 16 scenes

- I: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.115 .. 0.123
- I_perp_shared: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.117 .. 0.126
- I_perp_perarm: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.117 .. 0.125
- template energy fraction 0.973 .. 0.973; |I|/|tau| (A) 0.162 .. 0.164; template-amplitude asymmetry pattern (A > 0, B < 0) at 0/2 sites

| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \|I\|/\|tau\| | asym A / B | rank raw -> resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L03.mlp_out | 0.115 [0.060, 0.177] | 0.9998 | 0.9990 [0.006] | -0.000, -0.01, 0.5514 | 1.55 (0.1668) | -0.97 (0.7682) | 0.117 [0.059, 0.182], 0.5495 | 0.973 | 0.16 / 0.17 | -0.01 / 0.00 | 1 -> 15 (I: 10/10) |
| L05.resid_post | 0.123 [0.066, 0.192] | 1.0000 | 1.0000 [0.007] | -0.000, -0.01, 0.5524 | 1.19 (0.2358) | -1.15 (0.8012) | 0.126 [0.067, 0.198], 0.5465 | 0.973 | 0.16 / 0.17 | -0.01 / 0.00 | 1 -> 15 (I: 10/10) |

## group `hazard_corridor` -- 2 sites, 16 scenes

- I: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.153 .. 0.160
- I_perp_shared: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.157 .. 0.165
- I_perp_perarm: anti-alignment detected at 0/2 sites ; within-arm consistency A 0/2, B 0/2; cos range 0.157 .. 0.165
- template energy fraction 0.976 .. 0.976; |I|/|tau| (A) 0.156 .. 0.156; template-amplitude asymmetry pattern (A > 0, B < 0) at 0/2 sites

| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \|I\|/\|tau\| | asym A / B | rank raw -> resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L05.resid_post | 0.160 [0.098, 0.240] | 1.0000 | 1.0000 [0.002] | -0.024, -0.78, 0.3566 | 0.72 (0.3367) | -0.77 (0.7522) | 0.165 [0.101, 0.244], 0.3636 | 0.976 | 0.16 / 0.16 | -0.00 / 0.00 | 1 -> 14 (I: 10/10) |
| L03.mlp_out | 0.153 [0.091, 0.221] | 1.0000 | 1.0000 [0.003] | -0.020, -0.63, 0.3996 | 0.85 (0.3017) | -0.63 (0.7123) | 0.157 [0.096, 0.226], 0.3986 | 0.976 | 0.16 / 0.16 | -0.00 / 0.00 | 1 -> 14 (I: 10/10) |

## Transport diagnostics (held-out relative residual, transported vs raw)

- L03.mlp_out: 0.207 vs 1.460 (6380 matched hazard-free token rows)
- L05.resid_post: 0.207 vs 1.460 (6380 matched hazard-free token rows)

## Self-test

```
{
 "planted": false,
 "checks": {
  "no_anti_alignment": true,
  "cosine_band": true,
  "within_band": true
 },
 "passed": true,
 "n_sites": 6,
 "n_groups": 3,
 "counts": {
  "I": {
   "anti_alignment": [
    0,
    6
   ],
   "within_A": [
    0,
    6
   ],
   "within_B": [
    0,
    6
   ]
  },
  "I_perp_shared": {
   "anti_alignment": [
    0,
    6
   ],
   "within_A": [
    0,
    6
   ],
   "within_B": [
    0,
    6
   ]
  },
  "I_perp_perarm": {
   "anti_alignment": [
    0,
    6
   ],
   "within_A": [
    0,
    6
   ],
   "within_B": [
    0,
    6
   ]
  }
 },
 "max_abs_cross_cos_mean": {
  "I": 0.16047975026147618,
  "I_perp_shared": 0.16453098933610869,
  "I_perp_perarm": 0.1645233985427674
 },
 "max_abs_within_cos_mean_I": 0.04138331466463906,
 "identity_over_template_max_median": 0.25462704921887214,
 "template_fraction_range": [
  0.9374824969135699,
  0.9759940424969883
 ]
}
```
