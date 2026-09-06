# Identity-contrast geometry under the shared template (cgs-identity-contrast-geometry-v0.1)

Scope: descriptive/associational; template-projected; causal claims remain with patching. Everything is fitted on discovery scenes only (LOSO for every held-out score, the hazard-free Procrustes transport included); the identity-contrast vector I = DiD(ped) - DiD(cone) is algebraically invariant to subtracting one template vector from both identities, so the template removal changes its magnitude bookkeeping and the I_perp variant, not I itself; anti-alignment of I^A with the transported I^B is a geometric consequence of the reversed assignment and does not by itself show that either arm USES that direction.

Dumps: A = `/root/cgs-pilot/artifacts/drive_identity_geometry/self_test/planted/arm_A`, B = `/root/cgs-pilot/artifacts/drive_identity_geometry/self_test/planted/arm_B`; stimulus `/root/cgs-pilot/artifacts/drive_identity_geometry/self_test/planted/stimulus`; 16 discovery scenes, 2 sites, step 0, groups corridor, hazard, hazard_corridor; runtime 10 s.

Prediction (reversed assignment): I^A = DiD(ped) - DiD(cone) in arm A and the transported I^B are ANTI-aligned; each arm's I has a consistent direction across scenes. `cos` = per-scene cosine of I^A with T(I^B) (mean [scene-bootstrap 95 % CI]); `p_perm` = scene-permutation one-sided p (null mean in brackets = shared-direction alignment); `q` = held-out projection onto the other arm's LOSO identity direction with the identity-relabel refit max-T p (registered); `within A/B` = LOSO direction-consistency t with max-T p; `template` = in-sample fraction of the four DiD fields' energy along the shared mean direction; `|I|/|tau|` = median identity-contrast norm over template norm (A / B); `asym` = template-amplitude asymmetry ped - cone in units of the template norm (A / B; predicted + / -); `rank` = hard rank for 80 % energy of the four raw fields -> of the shared-template residuals.

## group `corridor` -- 2 sites, 16 scenes

- I: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.245 .. -0.231
- I_perp_shared: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.245 .. -0.231
- I_perp_perarm: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.245 .. -0.231
- template energy fraction 0.923 .. 0.923; |I|/|tau| (A) 0.348 .. 0.350; template-amplitude asymmetry pattern (A > 0, B < 0) at 0/2 sites

| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \|I\|/\|tau\| | asym A / B | rank raw -> resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L05.resid_post | -0.245 [-0.329, -0.154] | 0.0002 | 0.8691 [-0.277] | -0.492, -15.71, 0.0010 | 14.00 (0.0010) | 10.82 (0.0010) | -0.245 [-0.328, -0.155], 0.0010 | 0.923 | 0.35 / 0.35 | 0.01 / -0.02 | 1 -> 18 (I: 8/8) |
| L03.mlp_out | -0.231 [-0.315, -0.145] | 0.0002 | 0.8891 [-0.267] | -0.479, -15.25, 0.0010 | 13.14 (0.0010) | 10.15 (0.0010) | -0.231 [-0.316, -0.144], 0.0010 | 0.923 | 0.35 / 0.34 | 0.01 / -0.02 | 1 -> 19 (I: 8/8) |

## group `hazard` -- 2 sites, 16 scenes

- I: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.388 .. -0.373
- I_perp_shared: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.386 .. -0.371
- I_perp_perarm: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.386 .. -0.371
- template energy fraction 0.958 .. 0.959; |I|/|tau| (A) 0.264 .. 0.265; template-amplitude asymmetry pattern (A > 0, B < 0) at 2/2 sites

| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \|I\|/\|tau\| | asym A / B | rank raw -> resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L05.resid_post | -0.388 [-0.491, -0.287] | 0.0002 | 1.0000 [-0.473] | -0.670, -23.96, 0.0010 | 14.52 (0.0010) | 13.79 (0.0010) | -0.386 [-0.490, -0.287], 0.0010 | 0.958 | 0.26 / 0.26 | 0.02 / -0.02 | 1 -> 13 (I: 6/6) |
| L03.mlp_out | -0.373 [-0.479, -0.270] | 0.0002 | 1.0000 [-0.467] | -0.667, -23.62, 0.0010 | 14.76 (0.0010) | 13.82 (0.0010) | -0.371 [-0.479, -0.268], 0.0010 | 0.959 | 0.26 / 0.26 | 0.02 / -0.02 | 1 -> 13 (I: 6/6) |

## group `hazard_corridor` -- 2 sites, 16 scenes

- I: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.398 .. -0.383
- I_perp_shared: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.397 .. -0.381
- I_perp_perarm: anti-alignment detected at 2/2 sites ['L03.mlp_out', 'L05.resid_post']; within-arm consistency A 2/2, B 2/2; cos range -0.397 .. -0.381
- template energy fraction 0.961 .. 0.962; |I|/|tau| (A) 0.258 .. 0.262; template-amplitude asymmetry pattern (A > 0, B < 0) at 2/2 sites

| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \|I\|/\|tau\| | asym A / B | rank raw -> resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L05.resid_post | -0.398 [-0.512, -0.287] | 0.0002 | 1.0000 [-0.497] | -0.691, -24.38, 0.0010 | 14.98 (0.0010) | 14.03 (0.0010) | -0.397 [-0.513, -0.285], 0.0010 | 0.961 | 0.26 / 0.25 | 0.02 / -0.02 | 1 -> 12 (I: 5/5) |
| L03.mlp_out | -0.383 [-0.500, -0.269] | 0.0002 | 1.0000 [-0.490] | -0.686, -23.93, 0.0010 | 14.92 (0.0010) | 13.72 (0.0010) | -0.381 [-0.499, -0.269], 0.0010 | 0.962 | 0.26 / 0.25 | 0.02 / -0.02 | 1 -> 12 (I: 5/5) |

## Transport diagnostics (held-out relative residual, transported vs raw)

- L03.mlp_out: 0.206 vs 1.459 (6344 matched hazard-free token rows)
- L05.resid_post: 0.206 vs 1.459 (6344 matched hazard-free token rows)

## Self-test

```
{
 "planted": true,
 "checks": {
  "anti_alignment_I": true,
  "anti_alignment_I_perp": true,
  "within_arm_I": true,
  "template_dominates": true
 },
 "passed": true,
 "n_sites": 6,
 "n_groups": 3,
 "counts": {
  "I": {
   "anti_alignment": [
    6,
    6
   ],
   "within_A": [
    6,
    6
   ],
   "within_B": [
    6,
    6
   ]
  },
  "I_perp_shared": {
   "anti_alignment": [
    6,
    6
   ],
   "within_A": [
    6,
    6
   ],
   "within_B": [
    6,
    6
   ]
  },
  "I_perp_perarm": {
   "anti_alignment": [
    6,
    6
   ],
   "within_A": [
    6,
    6
   ],
   "within_B": [
    6,
    6
   ]
  }
 },
 "max_abs_cross_cos_mean": {
  "I": 0.39778422555611115,
  "I_perp_shared": 0.39701107382744866,
  "I_perp_perarm": 0.3970919223261827
 },
 "max_abs_within_cos_mean_I": 0.6944017133632381,
 "identity_over_template_max_median": 0.3497622077004965,
 "template_fraction_range": [
  0.9230677687816122,
  0.9617713451717631
 ]
}
```
