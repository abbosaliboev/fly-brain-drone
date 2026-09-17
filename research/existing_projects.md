# Existing fly-connectome projects (surveyed 2026-09-16)

The male *Drosophila* CNS connectome (MaleCNS v1.0) was released by HHMI Janelia
FlyEM, the Cambridge Connectomics Group, and Google Research (published in
*Cell*, "Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system", Sept 2026). In the months since, a wave of hobbyist
and research projects has already explored almost every obvious application.
**This project must not claim novelty for the general idea of "connectome
controls embodied agent."** The list below is what we found; see
`docs/scientific_assumptions.md` for how this shapes our claims.

## GPU / CPU simulators
- **annel0/flybrain** — MIT-licensed, hand-written Triton kernels, leaky
  integrate-and-fire (LIF) model, connectome stored as CSR sparse matrix.
  Runs the full 166,700-neuron MaleCNS v1.0 at ~2.4x realtime on an RTX 3060
  (12GB). Validated against Shiu et al. 2024 (*Nature*) firing-rate
  predictions (median ratio 1.01, correlation 0.970). Explicitly documents
  what it does NOT establish: 57% of the network doesn't spike in a
  point-neuron model because the optic lobe uses graded potentials, <1% of
  cell types have measured physiology, and parameters are degenerate. This is
  the standard for scientific honesty we should match.
- **eonfathom/FastFly** — CUDA/CuPy simulator targeting real-time-or-faster
  execution.
- **eonsystemspbc/fly-brain** — FlyWire (female) whole-brain implementation,
  multiple simulation backends.
- **ruvnet/Connectome-OS** — Rust LIF runtime with a debugger for FlyWire
  circuits.
- **seohyunjun/mps-malecns-model** — PyTorch simulator targeting Apple
  Silicon (MPS backend).

## Visualization / analysis tooling
- **FlyBrainLab** (NeuroMynerva + NeuroArch + Neurokernel) — full interactive
  platform: 3D morphology viewer, circuit builder, multi-GPU circuit
  execution. The most complete prior art for "see the brain, see it compute."
- **navis-org/navis** — Python library for neuron morphology analysis/plotting.
  We will likely use this as a *utility*, not a competing solution.
- **murthylab/codex** — FlyWire Connectome Data Explorer.
- **flyconnectome/bigclust2** — high-dimensional connectomic clustering UI.

## Embodied control / robotics
- **skulitom/haltere** — MaleCNS v1.0 subset (30,000 neurons, 2.77M
  connections) piloting a real FPV drone in the game *Liftoff* via a virtual
  Xbox controller. Vision handled by a separate 5M-parameter CNN
  ("GateNet") that detects racing gates and feeds body-frame navigation
  goals into fly sensory neurons (haltere afferents, wing sensilla, lobula
  plate cells); motor output read from wing/haltere motor neurons through a
  premotor population (R² 0.7-0.8 command reconstruction). Live anatomical
  activity viewer. States real limitations: top speed ~3 m/s vs ~14 m/s
  human racing pace, inconsistent lap completion.
- **garyb9/fly-drone** — Full undiminished 166,700-neuron connectome (frozen,
  unmodified) + MuJoCo simulated quadrotor. A *learned decoder* (PPO/GAE)
  reads only the 2,022 descending/VNC motor neurons and maps activity to
  rotor commands through a cascaded PID/mixer. Visual input is two
  hand-engineered scalar statistics per eye, not raw pixels or a learned
  sensory encoding into the connectome itself. Live browser dashboard with
  anatomical activity graph. Explicitly labeled "a research simulator, not
  flight software." MIT-licensed code, CC-BY-4.0 data. This is the closest
  prior art to our Phase 4-6 architecture.
- **theajmalrazaq/drosophiladrone** — connectome model embedded in a real
  ROS2/PX4 quadrotor control loop (hardware-adjacent, not just simulation).
- **DrJimmFan/Neurofly** — connectome-driven six-axis robot arm, not a drone.
- **FLYNN** (arXiv:2607.00025) — connectome-*topology*-derived RNN (not a
  literal spiking sim) trained end-to-end with backprop for vision-based
  MuJoCo navigation. Key finding: comparable task performance to size-matched
  hand-crafted RNNs, but far more robust to out-of-distribution input and to
  total vision loss without retraining. This is the strongest evidence in
  the literature that connectome topology alone confers useful inductive
  bias — directly relevant to our Phase 9/10 lesion and baseline work.
- **Eon Systems "embodied brain emulation"** — LIF model of ~140,000 central
  brain neurons driving a simulated fly body toward a food source using
  *gustatory* (taste) cues, with emergent grooming behavior. Not a drone, not
  vision-driven, but the closest prior art to a "connectome finds food"
  demonstration.

## Games (not directly relevant to our roadmap, noted for completeness)
Doomfly (ViZDoom), Fly Dino (Chromium dino game, 80-neuron subcircuit),
Fly64 (Super Mario 64), FlyPong, Fly Chess Lab, fly-craftax.

## What this means for us
1. **"Connectome piloting a virtual drone" is not novel by itself.**
   garyb9/fly-drone already does this with the full connectome, a learned
   motor decoder, MuJoCo physics, and a live brain viewer — architecturally
   very close to Phases 1-6 of our own plan.
2. **No project we found combines**: the full undiminished connectome +
   closed-loop *vision-only* food-seeking behavior (not racing/stability) +
   a systematic lesion-ablation study + baseline-controller benchmarking
   (random / rule-based / small NN) with reproducible plotted results. That
   combination is our chosen angle (see `docs/scientific_assumptions.md` and
   `docs/experiments.md`).
3. We should credit garyb9/fly-drone, annel0/flybrain, haltere, and FLYNN by
   name in our README as prior art, and be specific about what we add on top.
4. We are building our own simulator from scratch (educational value, full
   control over assumptions) rather than forking flybrain/fly-drone, but we
   will use their published validation approach (compare against Shiu et al.
   2024 firing rates) as a sanity check on our own simulator.
