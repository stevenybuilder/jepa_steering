# Scientific writing and presentation

The audience is ML researchers. Establish frozen JEPA-WM, activation steering,
ablations, and representational-geometry questions upfront. Use ordinary direct
sentences without elementary definitions of embeddings or activation steering.
Introduce technical detail when it is needed to assess the experiment.

## Structural references

In the [Karpathy–Dwarkesh interview](https://www.dwarkesh.com/p/andrej-karpathy),
02:20–02:22, they discuss the difficulty experts have seeing a beginner's questions
and how a conversational explanation can be more precise than an abstract full
of jargon. The linked [short](https://www.youtube.com/watch?v=tKvceAJ-Dxk) is from
this discussion. Its transcript export was unavailable; the full interview's
published transcript supplies the source.

The [JEPA-WM paper](https://arxiv.org/html/2512.24497v4) establishes the model and
design choices before comparing experiments. We use that separation between
setup and evidence. [Interpreting Physics in Video World Models](https://arxiv.org/html/2602.07050v1),
which introduces the Physics Emergence Zone, organizes its findings around
physical questions and tests alternative explanations. We use that progression
from question to measurement to remaining uncertainty. These are structural
references; their physical claims and terminology do not become claims about
our action-conditioned predictor.

The manuscript's main sequence is prediction error, fixed candidate choice,
iterative search, short physical execution, and fresh full-task success. The
appendices preserve the detailed intervention equations, complete secondary
analyses, and historical tables. The order is explanatory, not a claim that
these different studies identify one causal chain.

## Tools reviewed

GitHub star counts were checked through the repository API on September 13, 2026.
Popularity is reported separately from suitability; none of these repositories
establishes that automatic rewriting improves scientific accuracy.

| Tool | Stars | Useful role | Limitation |
|---|---:|---|---|
| [blader/humanizer](https://github.com/blader/humanizer) | 47,632 | Flag inflated framing, formulaic transitions, and empty emphasis | An editing skill, not an ML evidence checker |
| [Orchestra AI Research Skills](https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/20-ml-paper-writing) | 12,613 | The ML writing module covers contribution, structure, figures, and citations | The repository contains many unrelated research workflows; using the writing guidance does not require installing them |
| [Vale](https://github.com/vale-cli/vale) | 6,103 | Offline, configurable prose linting for repeated style checks | Rules can find wording problems, but cannot decide what a result means |
| [academic-humanizer](https://github.com/AIScientists-Dev/academic-humanizer) | 1,523 | Academic editing guidance that keeps claims tied to evidence | Mechanical preservation of paragraph structure would retain the old draft's main problem |

The LCFM rewrite applies the humanizer 3.0.0 editing process at commit
`9862685f575c65a8247f90369951df1b3416e3d6`: identify patterns, rewrite, check for
remaining patterns and changed meaning, then write the final prose. No detector
score is used as a quality measure.

## Editing rules for this repository

- Start each result with the question, measurement, and answer. Explain the consequence immediately after the evidence.
- Use standard research terminology directly. Explain the experimental operation and measured quantity; avoid tutorial phrases such as “this is called.”
- Keep the measured quantity explicit. Forecast error, goal cost, action difference, physical distance, and task success are separate endpoints.
- Give one concrete example when it explains more than another abstract sentence. The Reach result is 52 successes in both conditions, with 21 rescues and 21 losses.
- Keep uncertainty with the claim it limits. An interval crossing zero leaves effects unresolved; it does not prove equality.
- Explain how to read each headline figure. Preserve all registered observations and comparisons; improve labels and ordering without changing the numbers.
- Use the concurrent baseline. Show post hoc best-arm comparisons as context, with the selection rule stated.

## Visual review

The final opening identifies the model, intervention families, geometry questions,
and Joseph et al.'s Physics Emergence Zone study. It summarizes measured findings
without making the lack of behavioral gain the entire research contribution.
The architecture, paired simulator GIF, decision-margin plot, and layer heatmap
remain visible. Geometry, history, and attention analyses remain visible; the full six-task
results table is expandable. This follows the supplied README review's advice on density while
rejecting an unsupported causal chain from development margins to protected success.

The README GIF and 1080p video show new unsteered JEPA-WM episodes on Reach and
Reach-Wall at simulation speed. Task and seed were fixed before capture. Every
saved simulator observation is checked against restored physics; labels stay
outside the robot view. The earlier three-arm, fifteen-action comparison remains
available in the [media documentation](media/README.md). Those earlier prefixes
cover 0.1875 seconds and use slowed playback; the full model clips do not.
