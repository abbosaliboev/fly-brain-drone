# Experiments (planned; executed starting Phase 8)

This file will hold the concrete, reproducible experiment configs and
results once the closed-loop system exists (Phases 4-7). Placeholder for
now so the structure exists from Phase 0, per the project plan.

## Centerpiece experiment: does connectome structure reproduce fly-like search behavior?

The primary research question for this project (agreed 2026-09-16, see
`README.md` and `research/existing_projects.md` for why this specific
question, not just "does it find the food," is the novel contribution):

> When the drone loses sight of the food target, does a connectome-derived
> controller search in a way that statistically resembles real *Drosophila*
> idiothetic local search — and does that resemblance depend on the
> connectome's actual wiring, or would any reasonably competent controller
> produce similar statistics?

Real fly local-search behavior (per the bioRxiv sources in
`research/papers.md`: "Diverse food-sensing neurons trigger idiothetic
local search in Drosophila", and "Exploration and exploitation are flexibly
balanced during local search in flies") has measurable structure:
- a tight looping/turning search pattern centered on the last-known cue
  location
- search radius/loop size that grows over time (widening search)
- a turn-angle distribution characteristic of local vs. dispersal search
- a "give-up" time after which the agent switches to wider exploration

**Protocol (to be finalized once Phase 6-7 exist):**
1. Record the connectome-driven drone's trajectory after the food target is
   hidden/occluded mid-approach.
2. Compute the same summary statistics the biology papers use (turn-angle
   distribution, loop-radius-over-time, give-up latency).
3. Run the identical scenario with each baseline controller (random,
   rule-based, small NN — see Phase 10 of the main brief).
4. Compare each controller's statistics against both each other and against
   published fly values, and report where the connectome-driven controller
   sits — closer to real fly behavior, indistinguishable from baselines, or
   neither. **We do not assume in advance which way this comes out** — this
   is a real, open experimental question, not a foregone conclusion picked
   to make the connectome look good.

This is why the food-search task specifically needs an occlusion/loss
mechanic (the drone must be able to lose the target, not just approach it
directly) — noted for Phase 7 design.

## Full experiment matrix (Phase 8-10)

- A: random start position, fixed food location
- B: random food location
- C: multiple food objects
- D: visual noise
- E: partial sensory loss
- F: neuron lesion (random / visual / descending / motor populations, at
  10/20/30% removal)
- Baselines: random controller, rule-based controller, small NN, optional CNN

Planned experiment matrix (see main brief Phases 8-10):
- A: random start position, fixed food location
- B: random food location
- C: multiple food objects
- D: visual noise
- E: partial sensory loss
- F: neuron lesion (random / visual / descending / motor populations, at
  10/20/30% removal)
- Baselines: random controller, rule-based controller, small NN, optional CNN

Each run records: success rate, time to target, distance traveled, reaction
time, neural activity summary, FPS, GPU utilization — saved to
`experiments/results/` as CSV/JSON with auto-generated plots.

## Executed: camera/FlyVis paired ablation (Phase 18, 2026-09-17)

`src/experiments/phase18_flyvis_ablation.py` compared the identical rendered
POV-camera food controller with and without a recurrent pretrained FlyVis
T4/T5 motion-damping term. Food selection remained an explicit orange-pixel
salience readout; FlyVis did not recognize food. World coordinates were used
only to score arrival.

| start yaw | camera only | camera + FlyVis |
|---:|---:|---:|
| 135 deg | 1284 steps | 1195 steps |
| 180 deg | 1123 steps | 1043 steps |
| 225 deg | 961 steps | 896 steps |

Both conditions succeeded in 3/3 paired trials. Mean steps were 1122.7 and
1044.7 respectively (78 steps, or 6.95%, fewer with FlyVis). The sample is
small and deterministic, so this supports only the narrow claim that the
online FlyVis term was functional and consistently helpful in these starts.
Raw results: `experiments/results/flyvis_ablation.csv`.

## Executed: position generalization (Phase 19, 2026-09-17)

The same paired comparison was repeated across three different combinations
of drone start and food position. Both conditions succeeded in 3/3 trials.
Camera-only required 1123, 1067, and 1450 steps; camera+FlyVis required 1043,
1001, and 1360. Mean steps decreased from 1213.3 to 1134.7 (6.48%). This
shows the Phase 18 effect was not restricted to the original fixed food
coordinate, while remaining too small a sample for a broad generalization
claim. Raw results:
`experiments/results/flyvis_position_generalization.csv`.

## Executed: removing the RGB food detector (Phases 20-23, 2026-09-17)

An engineered centroid of FlyVis R1-R6 activity correctly localized isolated
synthetic spots (left `+0.0819`, centre `+0.0012`, right `-0.0793`) but failed
in the arena because wall and ground contrast was not food-specific. A neural-
only closed loop consequently remained at its start for all 5000 steps.

A ridge readout trained on 546 R1-R6 activities initially appeared successful
when training/test frames came from the same scene (94.4% visibility accuracy,
100% left/right sign accuracy). This was scene leakage/overfitting: after
training across diverse backgrounds and holding out the actual deployment
scene, visibility accuracy fell to 27.8%, bearing MAE rose to 4.196, and sign
accuracy fell to 57.1%. Phase 23 therefore did not replace the RGB food
detector. The current FlyVis input is grayscale and supplies motion/contrast,
not the orange-object semantics required by this task. A genuinely neural
replacement needs a colour/spectral pathway (or a separately trained visual
object-recognition objective), not a tuned threshold on R1-R6 activity.

## Executed: R7/R8 colour-opponent search (Phases 24-26, 2026-09-17)

The standard FlyVis input copies one grayscale movie into R1-R8. We added an
explicit spectral approximation: R1-R6 receive broadband luminance, R7 receives
the RGB camera's blue channel as a UV proxy, and R8 receives green. The final
readout derives an approximate red drive as `3 * luminance - green - blue` and
uses a retinotopic `red - green` centroid to select the orange target;
T4/T5 opponent activity continues to provide motion damping. No orange RGB
threshold is called by the runtime controller.

In isolated calibration, uniform input gave zero confidence; left, centre and
right orange spots produced errors `+0.667`, `+0.010`, and `-0.650`. Brown walls
also created chromatic opponency, so the controlled experiment uses achromatic
ground/walls while retaining orange food. Across three start/food geometries,
the updated controller succeeded in 3/3 trials in 994, 963, and 1343 steps. Raw results:
`experiments/results/r7_r8_generalization.csv`.

This is a materially more neural visual-selection path than the RGB threshold,
but not a fully biological colour model: an RGB blue channel is not UV, pale
and yellow Drosophila R7/R8 subtypes are not separately represented, and the
achromatic background removes colour distractors.

## Executed: coloured distractors (Phase 27, 2026-09-17)

The initial `green - blue` opponent reached food in 994 steps without a
distractor and with a blue distractor, but a green distractor delayed arrival
to 2319 steps and attracted the drone to 0.219 m. After adding the second
`red - green` opponent, the repeated results were 994, 994 and 995 steps for
none, blue and green; both coloured distractors stayed at least 2.693 m away.
Raw results: `experiments/results/r7_r8_distractors.csv`.

## Executed: multichannel colour calibration (Phase 28, 2026-09-17)

On synthetic uniform, orange, green and blue stimuli, the new opponent produced
confidence values `0.0000`, `0.5549`, `0.0000`, and `0.0000`. Orange produced
horizontal error `+0.6108`; pure green and blue produced zero. This calibration
explains the Phase 27 improvement, but it is an engineered RGB identity cue and
not evidence that FlyVis learned orange-food semantics.

## Executed: warm food-hue robustness (Phase 29, 2026-09-17)

The same closed-loop controller was tested with four rendered food materials,
using one shared FlyVis instance and resetting recurrent state between trials.
Orange, red and amber each reached food in 994 steps; dim orange reached it in
995 steps. All four finished at a minimum distance of 0.598 m, giving 4/4
success in the controlled achromatic arena. Raw results:
`experiments/results/food_hue_robustness.csv`.

This demonstrates tolerance to the tested nearby warm RGB hues and reduced
target intensity. It does not cover illumination colour shifts, shadows,
textured objects or arbitrary warm-coloured distractors.

## Executed: lighting and warm distractors (Phase 30, 2026-09-17)

At 25%, 50% and 150% of the baseline scene-light intensity, the controller
reached food in 1012, 996 and 994 steps. Thus the tested neutral illumination
range passed 3/3. With an additional red or amber sphere, it still eventually
reached food, but took 2021 and 2062 steps and approached each distractor to
0.217 m. Raw results:
`experiments/results/lighting_and_warm_distractors.csv`.

We tested a narrower green/red-ratio gate as a possible fix. It rejected amber
as food and made the red-distractor trial fail, so that change was reverted.
This negative control exposes an identity ambiguity rather than a threshold
bug: a colour-only controller cannot simultaneously generalize “food” across
warm hues and reject visually equivalent warm spheres. Shape, size, texture or
a learned object cue is required.

## Executed: clean-environment reproducibility smoke (2026-09-17)

A new Python 3.11 virtual environment was created from scratch and populated
from `requirements-flyvis.txt`. The scripted Windows patches, pretrained-data
checksum validation, 18 regression tests, and original-extent FlyVis motion
test all passed. The recovered horizontal responses were `+0.020945` for
rightward and `-0.104056` for leftward motion. A reduced extent=5 was tested
and correctly rejected for this particular full-field regression because it
did not retain sign reversal; online control still uses extent=5 as its tested
memory/performance tradeoff.
