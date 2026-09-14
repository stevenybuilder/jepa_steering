# Choosing the next LCFM experiment

Historical design discussion, September 13, 2026. The 200-scenario replication
below was subsequently authorized and frozen in a separate
[execution protocol](../paper/data/lcfm_replication_protocol.json). The user later
expanded the compute budget while retaining the same scientific scope. The
original design and budget discussion below remain unchanged as provenance.
[Replication results](LCFM_REPLICATION.md).

## What the completed experiment tests

The reference is frozen JEPA-WM run with one input action replaced. Its predicted
future and candidate costs are the quantities that an internal patch tries to
reproduce. The unmodified input supplies a second reference for the size of the
action change. Neither reference is simulator ground truth.

The main comparison replaces action-derived conditioning once, at H3, or at both
appearances of that action, H3 and H4. Replacing the corresponding conditioning
at all blocks and both appearances reproduces the changed-input computation.
Exact equality is an implementation consistency check, not improved forecasting.

The 17/32 count measures disagreement with the changed-input reference's chosen
candidate. It is not task success, improvement over unsteered, or a comparison
against chance. There are only sixteen contexts, with two banks per context.
The model's context contains two frames; increasing the number of contexts in
the dataset does not increase the model's context length.

## A larger replication of the diagnostic

Use new trajectories, one prespecified starting context per trajectory, balanced
between Reach and Reach-Wall. Keep all comparisons within the same trajectory,
with identical candidate actions and random seeds. Freeze the action replacement
rule, intervention sites, and analysis before seeing the new outcomes. Existing
development contexts inform the design; they do not become fresh confirmation.

Estimate the frequency of changed candidate selection and report a confidence
interval. Choose one bank per context for the primary estimate; a second bank is
a paired robustness check. If both banks are aggregated, preserve the trajectory
as the sampling unit in uncertainty calculations. More candidates or windows
from a trajectory do not increase the independent sample size.

For a proportion near 50%, approximate 95% margins of error are:

| Independent trajectories per task | Approximate margin per task |
|---:|---:|
| 100 | ±9.8 percentage points |
| 200 | ±6.9 percentage points |
| 400 | ±4.9 percentage points |

These use 1.96 × sqrt(0.25/n), not a calculation of power to detect a planning
benefit. They assume independent sampling and one binary endpoint per trajectory;
they do not include simultaneous coverage across tasks or layer comparisons.
[NIST sample-size guidance](https://www.itl.nist.gov/div898/handbook/ppc/section3/ppc333.htm).

Do not test whether a mismatch rate is greater than zero as the main claim.
Instead report its magnitude and whether changed choices have meaningfully
different reference costs. Small winning margins can make a changed candidate
identity less consequential than the count suggests. Rank and elite agreement
remain secondary descriptive measures; avoid selecting the best layer again.

## The stronger forecasting and planning question

Before funding a larger behavior study, specify an actual steering rule with a
reason to improve predictions. The current input-reconstruction patch alone
does not supply that rule. A proposal to preserve a useful steering signal as
context advances is a new method and needs its own fixed definition and controls.

For such a method, compare unsteered JEPA-WM, the proposed edit, and a random edit
matched on site, rank, dose policy, and timing. Include one-time versus repeated
application as an ablation. Because repeating an edit also changes its total
application budget, define an appropriate budget control before attributing any
benefit specifically to history consistency.

Measure forecast accuracy against simulator observations for the same executed
actions. Separately run complete closed-loop episodes to measure task success.
Pair conditions by initial state and planner random seed. A more accurate match
to another model forecast is not a substitute for either endpoint.

Choose a primary comparison and a smallest useful effect before calculating
sample size. For example, a five-percentage-point task-success gain is a possible
design target, not a predicted result. For paired success outcomes, required
sample size depends on how often the two methods disagree, not just their
individual success rates. Use pilot disagreement rates with conservative
sensitivity ranges and simulate the planned paired test at 80–90% power.
Prespecify treatment of multiple tasks and secondary comparisons and keep the
evaluation size fixed, or register a sequential design before collecting data.
[NIST paired comparisons](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).

## Recommendation

Keep the current history experiment as supporting evidence in the README. A
modest independent replication could estimate how broadly the mismatch occurs.
Reserve a larger study for a specified forecasting or planning hypothesis with
simulator outcomes. Increasing the sample size of the exact-equality control
alone would add little scientific value.


## Concrete replication proposed for the next run

Use 100 fresh independent starting scenarios per task (200 total), with the
same 35 conditions and two 300-candidate banks as the existing action-history
experiment. Retain all six layers and random controls; do not select B1 based on
its development result. The primary endpoint is the all-block one-time patch's
H6 winner disagreement with the changed-input reference, using the original bank.
Report a 95% Wilson interval separately for each task, explicitly marginal; use
97.5% intervals if claiming simultaneous coverage of both task rates. The second
bank, rank correlation, elite overlap, and cost of the selected candidate under
the reference are secondary. The reference cost gap gives the size of a changed
choice in the model's own objective; it is still not physical ground truth.

Keep the action donor rule (next candidate cyclically), checkpoint, strict FP32,
H3/H4 edit timing, six-step horizon, and complete arm definitions unchanged.
Generate the new input manifest from a fixed seed registry and check initial
state and goal hashes against all recorded project exposures before inference.
Freeze the new source, registry, analysis, and manifest independently; the
original sixteen-context protocol and protected behavioral panel stay immutable.
No outcome-selected stopping, seed replacement, or layer selection is allowed.

The original sixteen cases took 186.7–190.7 seconds each for 70 full forecasts,
including encoding and checks, on their recorded GPU/runtime bindings. At that
throughput, 200 cases require approximately 10.4–10.6 GPU-hours. This is a planning
estimate from the actual workload, not a receiving measurement on another GPU.
The currently qualified video worker costs $0.4222/hour including its 40 GB disk;
that rate would imply about $4.4–$4.5 for the inference portion if throughput
transferred. Fresh input generation, environment qualification, archival, and
retries are additional. A proposed $10 total cap, $2.60 aggregate hourly cap, and
$2 closeout reserve would bound a small US-only fleet. A representative excluded
case must establish receiving throughput before scaling. No rental or scientific
launch is authorized by this document; the existing $5 allowance is for media.

This replication would make the existing diagnostic better measured. It would
not establish steering efficacy. A separate behavior study needs a specified
steering method and a power calculation for its paired physical outcomes.
