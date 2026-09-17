# Key sources

## Primary dataset
- Berg et al., "Sexual dimorphism in the complete connectome of the
  Drosophila male central nervous system", *Cell*, 2026. Companion site:
  https://male-cns.janelia.org/ — release notes at
  https://male-cns.janelia.org/release/, downloads at
  https://male-cns.janelia.org/download/.
- Google Research announcement (2026-09-03):
  https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/
- Janelia news release:
  https://www.janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-central-nervous-system
- neuPrint access: https://neuprint.janelia.org (Python client:
  `neuprint-python`; R client: `neuprintr` / `natverse::malecns`)

## Simulation validation reference
- Shiu et al., *Nature*, 2024 — connectome-constrained LIF model of the
  (female, FlyWire-derived) central brain; the firing-rate predictions from
  this paper are the standard other simulators (e.g. annel0/flybrain)
  validate against. We should do the same rather than inventing our own
  ad hoc validation.

## Directly relevant follow-on work (post-release, 2026)
- FLYNN, arXiv:2607.00025 — connectome-topology-derived RNN for
  vision-based navigation robustness; strongest evidence for "connectome
  structure alone confers useful inductive bias," relevant to our lesion/
  baseline experiments (Phases 9-10).
- "State of Brain Emulation Report 2025", arXiv:2510.15745 — broader context
  on what brain-emulation projects can and cannot currently claim; useful
  for `docs/scientific_assumptions.md` phrasing.
- Diverse food-sensing neurons trigger idiothetic local search in
  *Drosophila* (bioRxiv, 433771) and "Exploration and exploitation are
  flexibly balanced during local search in flies" (bioRxiv, 2024.06.26) —
  biological background on real fly foraging search strategy, useful when
  we design what "reasonable" food-seeking behavior looks like and what
  parts of our drone's search pattern are engineering abstraction vs
  biologically grounded.
- Gustatory connectome paper (ScienceDirect, S0092867426009438) — used a
  connectome-based model with single-neuron silencing to validate feeding
  circuit contributions; a template for how to run our lesion experiments
  credibly (silence one population at a time, check specific predicted
  behavioral change, not just "delete X% randomly").

## Existing simulators/frameworks worth knowing about (not necessarily used)
- FlyBrainLab / NeuroKernel / NeuroArch (eLife 2021,
  https://elifesciences.org/articles/62362) — GPU execution engine +
  interactive frontend, predates MaleCNS but built for this exact purpose.
- flybody (TuragaLab) / FlyGym (NeLy-EPFL) — anatomically detailed MuJoCo
  fly *body* biomechanics with RL examples. Not a drone, but the reference
  for how much biomechanical detail a "fly-controlled agent" project can
  reasonably include; we deliberately choose a simpler drone abstraction
  instead and say so explicitly.

This file should be updated as we read more deeply into specific pathway
identities (visual neuron types, descending neuron catalog) during Phase 2-3.
