# Architecture (Phase 0 decision record)

## Chosen stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | The machine's default Python is 3.14.0, which is too new for PyTorch/CuPy prebuilt wheels as of Sept 2026. Python 3.11 is installed locally and has full ML-stack support. |
| GPU compute | PyTorch (CUDA) | Sparse matrix ops, autograd (useful later for learned decoders), one dependency instead of juggling CuPy + a separate ML framework. Machine has a TITAN RTX (24GB VRAM, driver supports CUDA 13.1) — plenty of headroom for a 166,700-neuron sparse LIF simulation (prior art fits this in 12GB). |
| Connectome access | `neuprint-python` against the public neuPrint MaleCNS server | Avoids downloading multi-TB EM imagery; gives typed, queryable neuron/connectivity data directly. |
| 3D brain visualization | VisPy (OpenGL, GPU-accelerated) embedded in a desktop app | Single Python process, no browser/WebSocket bridge needed, handles 166K-point scatter plots interactively at real-time framerates. Reassess only if a specific feature (e.g. shareable web demo) later needs a browser. |
| 2D real-time panels (activity bars, metrics, plots) | PyQtGraph | Native Qt integration alongside VisPy's Qt backend, GPU-light, built for real-time updating plots — avoids matplotlib's redraw cost. |
| Windowing/layout | PyQt5/PySide6 | Lets VisPy's 3D canvas and PyQtGraph's 2D panels live in one dashboard window (the split-panel layout in the brief). |
| Drone physics | MuJoCo (revised, Phase 4) | Originally planned PyBullet for "pure Python install, no compiler needed" -- that assumption was wrong on this machine: pybullet has no prebuilt wheel for this Python/Windows combination, and building it from source failed even with MSVC Build Tools + cmake installed (an unresolved distutils compiler-spawn error after two workaround attempts). MuJoCo ships an official prebuilt wheel (`pip install mujoco`, no compiler needed) and offers built-in offscreen rendering for the drone's-eye camera view. It's also what most of the prior-art projects we surveyed already use (garyb9/fly-drone, FLYNN, flybody, FlyGym) -- see research/existing_projects.md -- so our drone/camera code will look familiar if we ever want to compare against theirs. |
| Data interchange in-process | Plain NumPy / PyTorch tensors | Everything runs in one process for now (brain sim, decoder, physics, viz all in the same Python app) — no need for a message bus at this scale. |

We are **not** using a web frontend (Three.js/WebGL) for v1: a single desktop
Python app satisfies every visualization requirement in the brief, keeps the
loop tight for a solo incremental build, and avoids maintaining a
frontend/backend protocol. This can change later if we specifically want a
shareable browser demo.

We are **not** forking an existing simulator (annel0/flybrain, fly-drone):
building our own LIF simulator is slower but keeps every biological
assumption visible and owned, matching the brief's incremental/educational
intent. We will validate our simulator's output against the same published
reference (Shiu et al. 2024 firing rates) that annel0/flybrain used, rather
than inventing our own validation criteria.

## Closed-loop data flow (target end state, built up over Phases 1-7)

```
neuPrint MaleCNS  --(one-time fetch)-->  connectome tables (types, synapses)
                                              |
                                              v
                                     src/brain/connectome.py
                                     (typed neuron list + sparse
                                      weighted adjacency, CSR)
                                              |
                                              v
        +----------------------------  src/brain/simulator.py  <---------------+
        |                          (GPU LIF step function)                     |
        |                                    |                                 |
        |                                    v                                 |
        |                          src/brain/pathways.py                       |
        |                       (named visual/descending/motor                 |
        |                        populations, verified via neuPrint,           |
        |                        NOT invented)                                 |
        |                                    |                                 |
   drone camera                              v                                 |
   (PyBullet render)  --> src/vision/encoder.py --> injects current into       |
        ^                  (image -> visual neuron   identified visual         |
        |                   population encoding)     neurons                   |
        |                                                                      |
   src/drone/drone.py <-- src/control/motor_decoder.py <-- identified motor ---+
   (PyBullet physics,        (motor-neuron activity ->
    position/velocity)        LEFT/RIGHT/FWD/UP/DOWN)
        |
        v
   src/environment/world.py (food target, obstacles)
        |
        v
   src/visualization/dashboard.py (VisPy 3D brain + PyBullet 3D world +
                                    PyQtGraph activity/metrics panels)
```

The drone never receives target coordinates directly (per the brief) — only
its rendered camera image, which flows through the encoder into the
connectome's visual neurons.

## Phase 11: measured performance

Real profiling (2026-09-16, TITAN RTX, 32,684-neuron/561,133-connection
navigation network), not assumed numbers:

| What | Steps/sec |
|---|---|
| LIF simulator alone, GPU | 1,667 |
| LIF simulator alone, CPU | 427 (**3.9x** slower than GPU) |
| Full closed loop (vision encode + LIF + decode + MuJoCo physics), naive stimulus injection | 572 |
| Full closed loop, optimized stimulus injection (`StimulusInjector`) | ~594 (measured via smoke test; isolated injection-only comparison: 792 -> 1,126, a **1.42x** speedup) |

Where the time actually goes in the naive full loop: vision encoding 7.8%,
neural simulation 83.2%, decode+motor-mixing 7.8%, MuJoCo physics 1.1%.
The "neural simulation" share was *not* dominated by the LIF math itself --
profiling found it was mostly tensor-construction overhead (a fresh
`torch.full()` allocation plus 4-5 boolean-mask assignments every step).
`StimulusInjector` (`src/brain/simulator.py`) replaces that with a
persistent buffer and precomputed integer indices (`index_fill_`/
`index_put_`), applied in `src/experiments/benchmark.py`. The GPU vs CPU
comparison shows a real but modest (not "GPU is 100x faster") speedup at
this network size -- expected, since ~33k neurons / ~560k connections is
small enough that Python/CUDA per-call dispatch overhead is a bigger
factor than raw FLOPs, which is also *why* the tensor-bookkeeping
optimization mattered more than the GPU-vs-CPU choice itself. Even
unoptimized, 572 steps/sec is far above what's needed for real-time
operation at MuJoCo's own physics timestep (200Hz) or the dashboard's
render rate (~30-50fps) -- so simulation speed is not currently a
bottleneck for anything this project does; the numbers above are reported
because the brief asks for them, not because a performance problem was
found and needed fixing.

## What's deferred to later phases
- Multi-GPU / batched-experiment execution (Phase 11).
- Any web/video export tooling beyond OS-level screen recording (Phase 12).
- Reduced-connectome mode is a documented *fallback*, not the default: we
  default to the full connectome and only reduce if Phase 2 benchmarking
  shows real-time visualization can't keep up, in which case the reduction
  strategy and its scientific implications get written up explicitly.
