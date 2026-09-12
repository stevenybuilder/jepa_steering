# JAX Scaling Book notes for our JEPA-WM study

Rewritten 2026-09-12 — the prior version of this file grew to 1,570 lines of
repetitive session narration (a long-context Codex session logging its own
process rather than distilled findings). Replaced with the actual claims that
matter, each grounded in either the Scaling Book's own text or a direct
measurement on our hardware. Source pages fetched directly from
https://jax-ml.github.io/scaling-book/ (roofline and GPU chapters), not
paraphrased from memory or from the old notes.

## Core model (from the Scaling Book, roofline chapter)

Lower bound on execution time for an operation:

```
T_lower = max(T_math, T_comms)
T_math  = FLOPs / accelerator_FLOPs_per_sec
T_comms = bytes_moved / bandwidth_bytes_per_sec
```

**Arithmetic intensity** = FLOPs / bytes moved. Every accelerator has a ridge
point where peak FLOPs/s / peak bandwidth defines a critical intensity: below
it, the operation is memory-bandwidth-bound and extra compute capacity buys
nothing; above it, the operation is compute-bound and bandwidth is idle. The
book's own worked example: a bf16 matmul on TPU needs a batch size above
roughly 240 tokens to become compute-bound; below that it's bandwidth-bound
regardless of how fast the matrix unit is.

The book does not publish specs for consumer/workstation NVIDIA cards (RTX
4090/5090/PRO 6000 Ada/Blackwell) — it covers datacenter parts (A100/H100/B200/
TPU). The ridge-point reasoning still applies; only the numbers must come from
our own measurements.

## What we measured, and what it means

Per-episode wall time, same 300-candidate/15-iteration CEM planner, strict
FP32, on four different physical GPUs (measured from real consecutive episode
timestamps, not estimated):

| Worker | Card | $/hr | Measured s/episode | Task |
|---|---|---:|---:|---|
| NJ | RTX 5090 | 0.752 | ~242 | MetaWorld Reach (component behavior) |
| MO | RTX 5000 Ada | 0.356 | ~539–541 | MetaWorld Reach, visual_only arm |
| NV0 | RTX PRO 6000 WS | 2.062 | ~252–261 | Same task family |
| NV3 | RTX PRO 6000 WS | 2.220 | ~252–261 | Same task family |
| NV0 | RTX PRO 6000 WS | 2.062 | ~85–86 | Push-T / PointMaze / DROID refined-task behavior |

Reading this against the roofline model:

- **RTX PRO 6000 WS (much more compute, similar-generation bandwidth to the
  5090) delivers no speedup over the RTX 5090** on the same task (252–261s vs
  242s — within noise). That is the signature of a **bandwidth-bound**
  workload, not a compute-bound one: once you're at Blackwell-generation
  memory bandwidth, adding more FLOPs capacity (PRO 6000 has far more CUDA/
  tensor cores than a 5090) does nothing, because bandwidth was the binding
  constraint the whole time.
- **RTX 5000 Ada (older architecture, lower memory bandwidth) is ~2.2x
  slower** (541s vs 242s) on the identical task. That ratio is consistent
  with a bandwidth gap between an Ada-generation workstation card and a
  Blackwell-generation consumer/workstation card, not a compute gap — 300 CEM
  candidates is not a batch size large enough to make this compute-bound on
  any of these cards.
- The refined-task panels (Push-T/PointMaze/DROID) run ~6x faster per episode
  (~85s) than the MetaWorld component-behavior panel (~242–541s) on
  comparable hardware. That gap is a **task-inherent cost difference**
  (longer/heavier rollout in the MetaWorld component harness), not a hardware
  choice — switching hardware doesn't touch it.

**Practical GPU selection rule for this workload:** pick RTX 4090/5090-class
(or better-bandwidth) hardware; avoid older/lower-bandwidth workstation tiers
like the RTX 5000 Ada. Do not pay a premium for a bigger-compute card (RTX PRO
6000 WS) expecting a speedup — the measurements show it doesn't deliver one
here. This changes wall-clock GPU-hours consumed per episode; it does not
change the total episode count required by the frozen protocol (96 per
simulation condition, 64 for DROID) — more/faster GPUs divide that fixed work,
they never reduce it.

## Two separate questions — do not conflate them

1. **"Why does the remaining work need this many GPU-hours?"** Answer: fixed
   episode count × measured seconds/episode on whichever *already-owned* card
   runs it. Fixing this (routing future Reach/Reach-Wall slots to NJ/NV0/NV3
   instead of MO once they free up) costs nothing extra — those GPUs are
   already rented and already billing. Market price is irrelevant to this
   question.
2. **"Should we rent an additional (5th+) GPU to add a parallel lane and
   finish faster in wall-clock time?"** This is a separate, optional lever —
   more lanes divide the same fixed total GPU-hours across more wall-clock-
   parallel streams; it does not reduce the total. Whether it's worth doing
   depends on the price of new capacity, which does fluctuate with the spot
   market (see below) — but the price of *new* capacity has no bearing on
   question 1.

Renting rules, measured per-card rates and the device-parity results are in
[docs/GPU_COMPUTE_PLAYBOOK.md](docs/GPU_COMPUTE_PLAYBOOK.md).

## Cost accounting reminder

`remaining_GPU_hours = sum(remaining_episodes * measured_seconds_per_episode) / 3600`
is a fixed lower bound set by the protocol's required episode count and the
measured rate on whichever hardware runs it. Total dollar cost is
`GPU_hours × price_per_GPU_hour`, and is a separate, independent question from
GPU-hours — more parallel GPUs shortens wall-clock time to spend a fixed
budget, it does not create additional total compute-seconds. Keep these two
numbers (GPU-hours needed, dollars available) separate; conflating them is
what made the prior version of this document hard to act on.
