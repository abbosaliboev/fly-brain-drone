# Fruit-Fly Drone — Current Project Status

Last updated: 2026-09-17

## Goal

Use fruit-fly connectome-constrained visual processing to control a virtual
MuJoCo drone that searches for and reaches a food target from its own camera.

## Current working pipeline

```text
MuJoCo POV camera
  -> 91-ommatidium hexagonal eye
  -> R1-R6 broadband luminance
  -> R7 blue/UV proxy + R8 green proxy
  -> R1-R6/R7/R8 red-green opponent food salience
  -> pretrained FlyVis T4/T5 optic-motion signal
  -> yaw controller + stabilized drone body
  -> updated camera frame
```

World coordinates are not supplied to the controller. They are used only to
build the scene and measure whether the drone reached food.

## What works

- MaleCNS connectome data loading and caching.
- LIF simulation with optional delays, synaptic filtering, graded cells and
  experimental motion-correlation dynamics.
- Stable MuJoCo drone hover, planar motion and yaw-rate control.
- Correct forward-facing POV camera; food is a real rendered scene object.
- Official pretrained FlyVis 1.2.0 model loading and online recurrent inference.
- Opposite moving bars produce opposite T4/T5 horizontal-motion signs.
- Camera-only orange-pixel baseline reaches food.
- FlyVis motion damping reduced steps by about 6-7% in small paired tests.
- Multichannel colour-opponent receptor drive reaches food without calling the old
  orange RGB-threshold detector at runtime.
- Pure green and blue distractors are rejected in the controlled arena.
- Orange, red, amber and dim-orange food variants all succeeded in the first
  hue-robustness benchmark.

## Best current result: R7/R8 search

Controlled achromatic arena, three different start/food geometries:

| Case | Result | Steps | Final distance |
|---:|---|---:|---:|
| 0 | success | 994 | 0.598 m |
| 1 | success | 963 | 0.596 m |
| 2 | success | 1343 | 0.599 m |

Success rate: **3/3**.

Source: `src/experiments/phase26_r7_r8_generalization.py`  
Raw data: `experiments/results/r7_r8_generalization.csv`

## Other measured results

### Camera-only versus camera + FlyVis

Three paired yaw starts, fixed food location:

| Start yaw | Camera only | Camera + FlyVis |
|---:|---:|---:|
| 135 deg | 1284 | 1195 |
| 180 deg | 1123 | 1043 |
| 225 deg | 961 | 896 |

Mean: 1122.7 versus 1044.7 steps; FlyVis used 6.95% fewer steps.

### Position generalization with the RGB baseline

Mean: 1213.3 camera-only versus 1134.7 camera+FlyVis steps; 3/3 success
in both conditions.

## What failed, and what was learned

### Uniform hand-written LIF dynamics

The raw MaleCNS topology plus uniform spiking dynamics did not produce reliable
T4/T5 direction selectivity. Delays, graded transmission, three-branch Mi/Tm
dynamics and real synapse geometry improved results only to roughly 55%, not a
scientifically convincing solution.

### Simple R1-R6 neural salience

An R1-R6 contrast centroid localized isolated spots, but arena walls and ground
dominated it. The neural-only controller did not approach food.

### Learned linear R1-R6 food readout

Within one scene it appeared strong (94.4% visibility, 100% direction sign),
but this was background overfitting. With diverse training backgrounds and a
held-out deployment scene it fell to 27.8% visibility and 57.1% direction-sign
accuracy. It is not used as a successful result.

### First R7/R8 opponent with coloured background

Brown walls generated green-blue opponency, so a confidence threshold could
not separate food from the walls. Achromatic walls and ground remain part of
the controlled experiment. A second opponent dimension now separates orange
food from pure green and blue distractors, but arbitrary coloured backgrounds
and lighting are not yet validated.

## Scientific boundaries

- FlyVis is a task-optimized, connectome-constrained optic-lobe model; it is
  not the same individual connectome as the MaleCNS downstream network.
- RGB blue is used as a proxy for UV-sensitive R7. The virtual camera does not
  measure UV, and pale/yellow R7/R8 subtype diversity is not modeled.
- The colour-opponent centroid is an engineered readout of biologically motivated input
  channels, not a demonstrated endogenous fly food-choice circuit.
- Forward locomotion and low-level drone stabilization are engineered body
  controllers, not decoded fly motor programs.
- Current evidence supports useful visual processing, not fly cognition or
  consciousness.

## Tests

The current targeted suite has 10 passing regression tests covering:

- bearing and pixel encoders;
- configurable/achromatic MuJoCo world;
- camera orientation;
- yaw sign and reset behavior;
- T4/T5 opponent decoding;
- connection delays, synaptic filtering and graded transmission.

Windows virtual-memory pressure can intermittently prevent large FlyVis or
MuJoCo allocations. Online inference therefore uses a central 91-ommatidium
FlyVis lattice (`extent=5`) with the original learned type-shared parameters.

## Next experiments

1. Add shape/size or learned object evidence to distinguish warm distractors.
2. Add R7/R8 subtype/spectral response curves instead of direct blue/green.
3. Validate and polish the new interactive dashboard on the target display.
4. Run and monitor the committed Windows CI workflow on the remote repository.

## Latest update: coloured distractors

The first Phase 27 run used only `green - blue`. Green therefore looked like a
target: it took 2319 steps and approached the distractor to 0.219 m. Phase 28
added a second opponent computed from the receptor drives: an approximate red
channel `3 * luminance - green - blue`, followed by `red - green` salience.
Synthetic orange, green and blue stimuli verified that only orange produced a
non-zero response. Phase 27 was then repeated:

| Distractor | Food reached | Steps | Minimum distractor distance |
|---|---:|---:|---:|
| none | yes | 994 | n/a |
| blue | yes | 994 | 2.693 m |
| green | yes | 995 | 2.693 m |

Both distractors were rejected: the green case improved from 2319 to 995 steps
and its closest approach changed from 0.219 m to 2.693 m. The updated controller
also retained 3/3 position-generalization success (994, 963 and 1343 steps).
This solves the tested pure-green failure, but remains an engineered RGB
opponent rather than a validated biological spectral circuit.

The Phase 26 and 27 benchmark runners now reuse one loaded FlyVis instance
across their trials. Recurrent state is reset for every trial, while repeated
model allocations are avoided to reduce Windows virtual-memory pressure.

## Latest update: food-hue robustness

Phase 29 varied the rendered food material while keeping geometry, initial
pose and controller unchanged:

| Food colour | Food reached | Steps | Minimum food distance |
|---|---:|---:|---:|
| orange | yes | 994 | 0.598 m |
| red | yes | 994 | 0.598 m |
| amber | yes | 994 | 0.598 m |
| dim orange | yes | 995 | 0.598 m |

Result: **4/4 success**. This establishes tolerance to nearby warm RGB hues
and lower target intensity in the controlled achromatic arena. It does not yet
establish robustness to arbitrary illumination or photorealistic materials.

Source: `src/experiments/phase29_food_hues.py`  
Raw data: `experiments/results/food_hue_robustness.csv`

## Latest update: lighting and warm distractors

Phase 30 changed scene illumination and introduced food-like warm distractors:

| Case | Food reached | Steps | Minimum distractor distance |
|---|---:|---:|---:|
| 25% light | yes | 1012 | n/a |
| 50% light | yes | 996 | n/a |
| 150% light | yes | 994 | n/a |
| red distractor | yes | 2021 | 0.217 m |
| amber distractor | yes | 2062 | 0.217 m |

Lighting robustness passed **3/3**. Both warm distractors were mistaken for a
possible target before the drone recovered, roughly doubling search time. A
narrow orange-hue gate was also tested and rejected because it lost amber-food
robustness and caused the red-distractor trial to fail entirely. Colour alone
cannot define which of two warm spheres is “food”; the next discriminator must
use another cue such as shape, size, texture or learned object evidence.

Source: `src/experiments/phase30_lighting_and_warm_distractors.py`  
Raw data: `experiments/results/lighting_and_warm_distractors.csv`

## Reproducible command entry point

`python -m src.cli` now provides stable `search`, `demo`, `hues`, `robustness`,
and `verify` commands. The README records the exact interpreter required
for each workflow. Both `verify` and the full `search` path were executed from
this entry point; search reached food in 994 steps. The remaining
reproducibility gap is installation: the
Windows compatibility patches are now captured by the idempotent
`scripts/patch_flyvis_windows.py`; the remaining setup gap is automating the
environment installation and pretrained model download themselves.
The patcher was validated from pristine FlyVis 1.2.0/datamate 1.0.0 wheels:
after patching, both affected files matched the working environment exactly.
`requirements-flyvis.txt` pins these package versions and includes the base
project requirements.

`scripts/setup_flyvis_data.py` now automates/checks the official pretrained
model data. It delegates network transfer to FlyVis's own downloader and then
verifies the official archive SHA-256 plus the exact `flow/0000/000`
checkpoint SHA-256 used here. With package pins, Windows patches, data setup
and the unified CLI documented, the remaining gap is testing the entire setup
from an actually clean environment.

`scripts/clean_setup_smoke.ps1` and `.github/workflows/windows-smoke.yml` now
encode that clean setup test. The smoke path creates a fresh venv, installs
pinned requirements, applies/checks patches, ensures/checks model data, runs
the regression suite, and performs a real opposite-motion FlyVis inference.
The script can also be exercised against an existing interpreter for local
validation. The motion smoke deliberately uses the original retinal extent=15;
the online controller's extent=5 lattice is too coarse for this synthetic
full-field direction regression. Only the remote clean runner remains to be
observed.

The `demo` command launches MuJoCo's native live viewer around the successful
rendered-camera/FlyVis controller. It shares the exact Phase 25 control path,
is paced for human viewing, and holds the successful final scene until the
user closes the window. Closing it early is treated as a normal stop. This replaces the old
dashboard's simplified bearing sensor for demonstrations; telemetry panels are
still to be added.

An all-in-one `dashboard` command is now also implemented. It contains a
clickable top-down arena, rendered drone POV, 91-column hex-eye view, R1-R8
input bars, T4/T5 motion telemetry, food confidence, distance, trajectory,
Start/Pause/Reset controls, and a live optic-lobe activity diagram covering
retina, lamina, medulla and T4/T5 types. Brain-node brightness comes from real
baseline-relative FlyVis activity, not random animation. Arena clicks mutate only the MuJoCo food
geometry; the controller is not given the clicked coordinates.

The brain section now has two tabs. `3D MaleCNS brain model` renders the cached
141,781-neuron soma cloud as a mouse-rotatable software projection and overlays
27,523 matching visual-type neurons. `FlyVis pathway diagram` retains the
layer-by-layer view. Overlay brightness uses live FlyVis activity mapped by
cell type; it is explicitly labeled as a cross-model proxy because FlyVis and
MaleCNS are not the same individual connectome.

### Clean setup validation result

The smoke script was executed locally from a newly created Python 3.11 virtual
environment, not only against the development environment. It successfully:

1. installed `requirements-flyvis.txt` from PyPI;
2. applied and re-checked both Windows compatibility patches;
3. verified the pretrained archive and checkpoint checksums;
4. passed all 18 regression tests; and
5. recovered the expected FlyVis motion signs (`+0.020945` rightward,
   `-0.104056` leftward).

This run also exposed and fixed two clean-machine issues: the system default
was Python 3.14 rather than 3.11, and machine-wide pip configuration injected
a stale NVIDIA package index. The smoke script now selects Python 3.11
explicitly and uses the repository's isolated `.pip-flyvis.ini`. The workflow
file is ready; observing a hosted run requires pushing it to the remote repo.

## Dashboard visual upgrade (2026-09-17)

The dashboard now uses a cohesive dark laboratory theme, stronger visual
hierarchy, a live-system badge, polished controls, a gradient/grid arena and a
fly-shaped arena marker. MuJoCo renders the vehicle with a stylized fruit-fly
shell (head, compound eyes, thorax, banded abdomen, translucent wings and
legs), and the room now includes a checker floor, fill lighting, boundary trim
and an overview camera. The shell is cosmetic: flight physics remain the same
stabilized four-actuator quadrotor abstraction. In controlled food-search mode,
room surfaces and trim remain achromatic so they cannot leak a colour cue to
the engineered opponent readout.

Validation after the visual upgrade passed all 20 regression tests. A complete
rendered-camera + FlyVis search also still reached food in 994 simulation steps
at 0.599 m, confirming that the new shell and room did not break navigation.

The next dashboard pass moved the 3D MaleCNS brain into a permanent right-hand
panel, eliminating the very wide bottom row. The food target is now rendered as
a composite apple with a warm body, stem and leaf, and a matching apple icon is
used in the clickable arena. An optional dashboard-only challenge scene adds
low platforms, pylons, an arch and cool-colour beacons. Benchmark scenes keep
their original controlled geometry because `challenge_arena` defaults to off.

Post-change validation passed 22/22 regression tests. The full rendered-camera
FlyVis search still found the composite apple in 994 steps at 0.600 m.

### Windows low-virtual-memory startup fix

`motor_decoder.py` previously imported the full PyTorch simulator at runtime
even when the dashboard needed only the lightweight `MotorCommand` dataclass.
That dependency is now guarded by `TYPE_CHECKING`, so importing the dashboard
does not eagerly load PyTorch. On the development machine Windows nevertheless
reported WinError 1455 and OpenBLAS allocation failure when PyTorch itself was
tested: system Memory Compression was using about 8.4 GB, confirming global
commit/page-file exhaustion rather than a dashboard code exception. FlyVis
still legitimately requires PyTorch when its backend initializes; after a
reboot or a larger system-managed page file, no code workaround is required.

## Physical arena and fly-like search behavior (2026-09-18)

Dashboard pylons and arch pieces are now real MuJoCo collision geometry rather
than visual-only props. Five fly-relative forward rays detect physical scenery;
the apple is placed in a separate geometry group so it remains a goal rather
than being avoided as a wall. A new explicit `FlySearchBehavior` replaces the
stationary search spin with forward, multi-frequency casting curves. Visible
apple pursuit also retains forward motion, while a close looming surface takes
priority and causes braking plus a saccadic turn toward the clearer side.
Dashboard telemetry exposes `CASTING SEARCH`, `APPLE PURSUIT`, or
`LOOMING ESCAPE` and the nearest obstacle distance. The map now draws its
platforms, pylons and arch, and rotates the fly icon with its actual heading.

This behavior is biologically inspired, not a complete biomechanical fruit-fly
model: propulsion is still the documented stabilized quadrotor abstraction.
All 26 regressions passed, and a 3,000-step physical casting run entered
looming escape, remained inside the 4 m arena, and finished at
approximately (-1.99, -0.13, 1.00) m.

### Pillar jitter fix

The first looming controller recomputed its turn side every physics step. Near
a narrow pillar, adjacent rays alternated as the body rotated, producing
left/right command chatter and a discontinuous forward-force jump. The live
controller is now stateful: it commits to one escape direction for at least 90
physics steps and retains it until 1.35 m clearance, uses continuous braking,
and low-pass filters both yaw and forward commands. A 3,000-step challenge run
spent 272 steps in escape, recorded zero escape-direction flips, stayed inside
the arena, and finished at approximately (-1.30, -0.27, 1.00) m. All 27
regression tests pass.

Target identity remains colour-led. The engineered R7/R8 proxy computes warm
red-green opponency from RGB camera channels; FlyVis contributes optic-motion
damping, not apple recognition. The rendered stem, leaf and apple silhouette
are not yet used as shape evidence, and smell is not simulated. Consequently a
sufficiently similar warm-coloured object can still attract the controller.

## Apple Hunt rounds (2026-09-18)

Added `src/control/apple_hunt.py` for monotonic active wall-clock timing,
separate simulation time, pause/resume, score, contact/escape episodes, path
length, bounded ten-result history, and best times keyed by mode/difficulty.
Records are session-local; they are not written to disk. Moving the apple or
resetting the fly prepares a paused new round. Start resumes its timer.
Success requires three consecutive fresh neural-camera observations meeting
distance, confidence and rendered-apple visibility checks. Segmentation IDs
are used only by the scoring referee, never by the flight controller.

Dashboard includes HUD, result banner, green ring/sparkles, result glow,
mode-filtered history, random placement, clear results and difficulty controls.
Easy removes challenge props; Normal retains them; Hard adds two collidable
pillars and two distractors. Conservative placement clearance rejects walls
and pillars. This does not prove every randomly selected route can be solved
by the reactive controller. Best times are separate for each level and mode.

Engineered mode disables neural motion damping, Hybrid retains it, Neural
Only maps T4/T5 motion to yaw with no engineered forward search. It currently
has no learned forward command and is not expected to complete food searches.
No neural-contribution percentage is invented: result banners explicitly
state it is not quantified. Scores are presentation metrics, not scientific
benchmark results. Older environment benchmarks remain unchanged by the
optional `game_assets` flag.

Validation: 32 regression tests passed, including fake-clock pause/resume,
fresh-observation confirmation, scoring, bounded history and safe placement.
A dashboard construction/control smoke test passed without loading FlyVis.
Full neural end-to-end rounds and the broader presentation/camera overhaul
from the attached design brief have not been validated or completed here.

## Arrival, speed and observer-room fixes

Arrival no longer depends on the controller's adaptive colour confidence.
The referee checks all apple component IDs in fresh segmentation frames every
five physics steps and requires three consecutive visible, nearby observations.
Arrival radii are now 0.90/0.80/0.72 m: the camera sits 0.17 m in front of the
body and can enter the ~0.4 m apple at the older thresholds. Scoring IDs remain
isolated from motor decisions. The success banner and status bar show game and
real seconds. The active game clock, score and best times now use actual
simulation time, superseding the earlier wall-time scoring description.
1x/2x/5x selects 3/6/15 physical steps per timer callback; neural cadence and
physics timestep remain unchanged. Hardware load can limit real throughput.

Added NEURAL READOUT: R1-R6 activity contrast drives an explicit motor mapping
and retains safety avoidance. It is not apple identity recognition; NEURAL ONLY
still means the limited T4/T5 yaw-only experiment. The room now has wall ribs,
light strips, runway markings and island fixtures, plus a dedicated 640x480
observer renderer and 3D LAB tab. Observer colour styling is isolated from the
fly's neutral camera input. MAP remains the apple-placement view.

Validation: 34 regressions passed, including real rendering on approach and
facing away. `scripts/smoke_apple_hunt.py` checks arrival banners in three modes,
pause and exact 1x/2x/5x simulation increments with stubbed inference. A separate
real FlyVis NEURAL READOUT run started 0.77 m from the apple, produced neural
contrast confidence ~0.054 and displayed APPLE FOUND at 0.05 game seconds.
This validates nearby arrival, not general neural-only search performance.

### Dashboard allocation failure mitigation

After a reported model-compilation allocation failure, process inspection
showed one svchost at roughly 38 GB private memory and an independent model
probe failed in OpenBLAS allocation. Dashboard-only game assets now bound
MuJoCo's arena to 8 MB, use 512-pixel shadows and a 320x240 framebuffer.
Observer and retinal views share one renderer/context with 256 scene slots;
update_scene rebuilds each view so observer colours cannot leak into sensors.
The CLI defaults BLAS/OpenMP workers to one before imports, respecting explicit
environment overrides. Allocation failures produce a concise recovery message.
Reference: https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#size

Validation in .flyvis-venv: a 2,000-step game-assets physics run used 17,736
arena bytes at peak with no warnings, and the dashboard smoke passed arrival,
pause and speed checks (inference stubbed). These reductions cannot remedy
system-wide memory exhaustion; no Windows services were stopped or changed.

### Stable layout and interactive observer camera

Difficulty changes no longer resize the dashboard: result/status/history
heights, selector widths and column minimums are fixed. The full telemetry
stack now lives in a styled vertical scroll area, so metrics and all five bars
remain reachable on smaller screens. The oversized pill was replaced with a
compact `SYSTEM ONLINE` status line.

The 3D LAB view now starts closer and uses a free MuJoCo observer camera.
Left-drag changes azimuth/elevation and the mouse wheel zooms between 4.8 and
16 m. The single shared renderer remains in use. The dashboard smoke verifies
camera orbit/zoom, stable HARD-mode column widths and the telemetry scroll,
alongside arrival and speed controls. A fresh observer preview was visually
checked. The full verification run was interrupted by WinError 1455 while
loading torch (`cufft64_11.dll`) under the machine's existing page-file
pressure; affected UI smoke tests passed before that failure.
