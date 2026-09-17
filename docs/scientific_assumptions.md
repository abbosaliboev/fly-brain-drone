# Scientific assumptions and limitations

This document is the honesty ledger for the project. It must be updated
whenever we make a modeling choice that isn't directly read off the
connectome data. Every claim in the README should be traceable to a line
here.

## What comes directly from the connectome (MaleCNS v1.0)
- Neuron identity, type annotations, and count. **Verified against the live
  neuPrint instance (2026-09-16):** 176,422 total `:Neuron` nodes, 165,122
  with status "Traced" (close to, not identical to, the 166,700 figure in
  the press release/paper -- small drift is expected as proofreading
  continues), 141,781 with a known soma position (used for the Phase 1 3D
  viewer; the other ~24,000, disproportionately optic-lobe cells, have no
  soma position to plot).
- Synaptic connectivity: which neurons connect to which, and synapse counts
  per connection (used as connection weight magnitude — see caveat below).
- Neuron soma spatial position (used for 3D visualization layout).
- **Phase 2 pathway (verified 2026-09-16 via direct neuPrint query, see
  `src/brain/pathways.py`):** the elementary motion-detection circuit
  exists in MaleCNS under real type names -- photoreceptors `R1-R6`
  (combined type), `R7d/p/y`, `R8d/p/y`; lamina `L1-L5`; medulla `Mi1`,
  `Tm1`; direction-selective `T4a-d`, `T5a-d`. Induced subnetwork among just
  these types: 31,120 neurons, 406,462 (pre,post,ROI) connection rows. Left
  vs. right hemisphere is derived from the `instance` field's `_L`/`_R`
  suffix (covers all 31,120 neurons; cross-checked against the `somaSide`
  property where that property is populated -- it is null for the combined
  `R1-R6` type -- and agrees in every case checked).

## What is published biological knowledge, not from the connectome itself
- To be filled in as we identify specific visual/descending pathways in
  Phase 2-3 (e.g. lobula plate tangential cell optic-flow role is established
  in the literature; the *specific* MaleCNS body IDs for these cells must be
  confirmed via neuPrint query, not assumed from prior datasets like
  hemibrain/FlyWire, since IDs and even cell-type boundaries can differ
  between datasets).

## What is computationally modeled (engineering choice, biologically inspired)
- Neuron dynamics: leaky integrate-and-fire (LIF), the same simplification
  used by Shiu et al. 2024 and annel0/flybrain. This is a real modeling
  choice, not a fact about the fly: real neurons have much richer dynamics,
  and (per annel0/flybrain's own documented finding) roughly 57% of the
  MaleCNS network is optic-lobe neurons believed to use graded, non-spiking
  potentials, which a point-neuron LIF model does not capture correctly.
  We inherit this same limitation and will say so wherever we show optic
  lobe activity. **This directly affects Phase 2's chosen pathway**: R1-R6/
  R7/R8 photoreceptors, L1-L5 lamina neurons, and Mi1/Tm1 medulla neurons
  are all well-established in the literature to be predominantly graded-
  potential (non-spiking) cells in real Drosophila, not the discrete
  spiking units our LIF model treats them as. T4/T5 are closer to spiking/
  regenerative. So "spike counts" and "firing rates" reported for the
  early stages of our Phase 2 pathway are a modeling convenience for
  showing signal propagation, not a claim that these cells really spike
  the way our simulator's traces will show.
- Synaptic weights: the connectome gives synapse *counts* between neurons,
  not measured physiological weights or signs (excitatory/inhibitory) for
  most connections. Sign is typically inferred from predicted
  neurotransmitter identity (provided in the dataset), and magnitude from
  synapse count — both are standard modeling approximations, not measured
  physiology. Fewer than 1% of cell types have directly measured
  physiological parameters (time constants, thresholds) — we will use
  literature-typical values for the rest and document the source per
  parameter.

## What is engineering abstraction (no biological claim at all)
- Phase 2's "CENTER" stimulus: LEFT and RIGHT map onto a real anatomical
  fact (each compound eye projects to its own ipsilateral optic lobe, so
  driving only left- or right-hemisphere photoreceptors is biologically
  grounded). "CENTER" does not correspond to any single verified neuron
  population we identified in this pathway -- we implement it as
  simultaneous bilateral stimulation, an engineering simplification, not a
  claim about a real frontal/binocular-field circuit.
- The drone itself: a quadrotor is not a fly's body. There is no wing/haltere
  aerodynamics model. Motor output categories (LEFT/RIGHT/FORWARD/UP/DOWN)
  are our own discretization for driving a quadrotor, not a fly's actual
  motor repertoire.
- Visual encoding: how a rendered RGB camera frame gets converted into input
  current for identified visual neurons is an engineering pipeline we design
  and validate for task performance, not a reproduction of fly
  photoreceptor/lamina processing.
- Motor decoding: how motor-neuron (or descending-neuron) activity converts
  into drone motor commands is a decoder we design (see
  `src/control/motor_decoder.py`), explicitly labeled as an engineering
  abstraction, and swappable/replaceable.
- "Food-seeking" as a behavior: real flies forage using multimodal cues
  (vision, olfaction, gustation) over different search strategies than
  "fly toward the one visible target." Our vision-only, single-target task
  is a deliberately simplified proxy task, not a model of fly foraging
  ecology.

## Known current limitation: the decoded motor signal is not yet stimulus-reliable
Phase 3 (`src/experiments/phase3_motor_decoder.py`) traced a fully verified
circuit -- photoreceptors -> lamina -> medulla -> T4/T5 -> HS/VS -> DNa02
(the steering descending neuron, see `src/brain/pathways.py`) -- and
confirmed the decoder (`src/control/motor_decoder.py`) correctly reads
DNa02_L/DNa02_R activity into a yaw command. However: at the gain values
needed for any signal to reach DNa02 at all (5 synapses downstream of the
stimulus), the decoded left-vs-right difference is dominated by a real,
small structural asymmetry in this specific reconstruction (DNa02_R
receives 72 total synapse-count-weight from 4 connections in this
subnetwork; DNa02_L receives 66 from 3), not by which side was stimulated.
Concretely: at gain=0.3, DNa02_R fires more than DNa02_L regardless of
which side's photoreceptors were stimulated.

This is not a code bug (the structural numbers were independently verified
directly against `conn_df`) -- it is an honest finding about the current
*uncalibrated* model: our synaptic gain and the "ambient drive" placeholder
(see `src/experiments/phase2_visual_response.py` module docstring) have not
been fit to any real firing-rate data, so at this simulation depth, noise/
baseline structure can dominate over a genuine stimulus-driven signal. We
are proceeding to Phase 4 (drone) with this documented, not swept under the
rug -- the calibration and validation work planned for Phases 8-10
(baseline-controller comparison, lesion experiments, and fitting against
Shiu et al. 2024-style firing-rate data) is exactly what's needed to
resolve this, and doing it prematurely now (by hand-tuning parameters until
a "nice" result appears) would risk cherry-picking rather than validating.

## Phase 4: the closed loop exists, but only yaw is neurally driven
`src/experiments/phase4_closed_loop.py` closes the full loop -- drone
camera -> `src/vision/encoder.py` -> connectome (steering pathway) ->
`src/control/motor_decoder.py` -> `src/drone/physics.py` rotor mixer ->
MuJoCo physics -> new camera frame -- with no target coordinates given to
the drone at any point, per the brief's requirement. The drone (a standard
4-rotor MuJoCo quadrotor, verified: stable hover, clean single-axis yaw
under differential thrust, working offscreen camera) mechanically responds
to the connectome's output in real time. Given the Phase 3 finding above
(the yaw signal isn't yet reliably stimulus-locked), what this closed loop
currently demonstrates is that data flows correctly through every real
stage of the pipeline end-to-end, not that the drone's rotation is yet a
validated response to what it sees. Forward/altitude hold a fixed hover
(see motor_decoder.py) since no verified pathway for those axes exists yet.

## Phase 5: unified dashboard uses a 2-sensor model, not a rendered camera
`src/visualization/dashboard.py` combines the brain view, drone view, and
metrics in one window. While building it we found that MuJoCo's own
Renderer and VisPy's Qt-embedded OpenGL context conflict when both run in
one process: MuJoCo's rendered frames stopped changing at all in response
to the drone's orientation (confirmed byte-identical renders before/after
a 90-degree yaw) once VisPy's canvas was active in the same process, while
position changes still worked. We tried several fixes (a single shared
Renderer instead of two, copying the QImage buffer, moving the camera via
FlyCamera and TurntableCamera, moving the world instead of the camera) and
could not resolve it within reasonable effort. Rather than keep debugging
a cross-library GL conflict, or silently ship something that looks like a
camera but isn't really responding, `src/vision/encoder.py`'s
`encode_left_right_from_bearing` uses a simpler two-sensor (left/right
"light level") model derived directly from the drone's position and
orientation relative to a fixed landmark -- similar in spirit to how real
phototaxis robotics experiments are often modeled (two light sensors, not
a full camera). **This is an explicit downgrade from a rendered camera
image, stated plainly, not a silent one.** The real rendered-camera closed
loop is demonstrated working correctly in
`src/experiments/phase4_closed_loop.py` (MuJoCo's Renderer used standalone,
no Qt/VisPy in that process) -- that remains the reference implementation
for "a camera image drives the connectome," and the dashboard's simplified
model is for the unified-UI demo specifically, not a replacement finding.

## Phase 6: collision-avoidance pathway, and further physics simplifications
Verified (2026-09-16, live neuPrint queries) a second real circuit extending
directly off Phase 2/3's T4/T5: **T4/T5 -> LC4/LPLC2 (lobula columnar
looming detectors) -> DNp01, labeled "DNp01(GF)" in this dataset -- the
Giant Fiber escape neuron**, one of the most extensively characterized
single identified neurons in the fly nervous system (von Reyn et al. 2014).
No direct T4/T5->DNp01 connection exists; LC4/LPLC2 is confirmed as a real,
necessary intermediate stage, and T5d->LPLC2 (4,817 connections, 19,165
total synapse weight) is among the strongest connections found in any of
our pathway verification queries. GF's real role is a fast, symmetric
jump/takeoff escape response, not steering -- `AvoidanceDecoder` (see
`src/control/motor_decoder.py`) uses DNp01 activity as an *urgency*
magnitude only, and gets *direction* from the lateralized upstream LC4/
LPLC2 signal instead. This split is our own engineering design for
combining these two real-but-not-directional signals, not something read
off the connectome.

Since we don't compute real optic flow (see the Phase 5 entry above),
LC4/LPLC2 are driven the same way photoreceptors are: an engineered
external signal (`encode_looming_from_walls`, wall-proximity-based, not a
pixel computation) injected directly at the sensory-relay population,
documented as a substitute for genuine looming computation, not hidden.

**Drone physics needed real hardening to make Phase 6 possible, entirely
separate from the neuroscience:** an initial attempt to translate the drone
forward via pitch-tilting (like a real quadrotor) needed a full PD
attitude-stabilization loop to avoid an unbounded-pitch instability (a
constant differential thrust is a constant torque with nothing to stop it),
and even with that PD loop running, the closed loop diverged into a
runaway climb after a wall collision. We did not fully root-cause this and
are not claiming to have solved general quadrotor flight control -- instead:
forward locomotion is now a direct external force (`Drone.apply_forward_force`)
rather than tilt-based thrust vectoring, a pure-rate-damping term
(`Drone.apply_attitude_damping`, dissipative only, cannot itself cause
instability) replaces active attitude control, wall contacts were softened
(`solref`/`solimp` in `src/environment/world.py`), and `Drone.reset_if_unstable`
is an explicit safety net that resets to a safe hover if the simulation
still diverges, rather than letting a rare residual instability be
visually disruptive. All of this is drone-airframe engineering, unrelated
to the connectome/neural side of the project, and documented here so it
isn't mistaken for a more sophisticated flight controller than it is.

## Phase 7: food-search mechanic works; active search does not (yet) emerge
`src/experiments/phase7_food_search.py` builds the food-search task from
the brief: FOOD DISTANCE / TARGET ANGLE / SEARCH TIME are computed and
displayed (`food_distance_and_bearing` in `src/vision/encoder.py`) purely
for logging -- never fed into the connectome, which only ever receives the
two-scalar bearing drive, same as Phase 6. Two starting conditions were
tested (2026-09-16):
- **Facing toward the food (START_YAW_DEG=0):** reaches it reliably (~1.8s,
  255 steps in one run) -- confirms the approach/reach mechanic (distance
  tracking, the reached-radius check, the closed loop overall) works
  correctly.
- **Facing away from the food (START_YAW_DEG=180, the script's default):**
  this is the actually interesting case for the project's centerpiece
  question (does the connectome reproduce fly-like search when a target is
  lost from view? see README.md / docs/experiments.md). Currently, it does
  **not** produce anything resembling active search: with no directional
  visual cue (encode_left_right_from_bearing returns equal drive on both
  sides when the target is behind), the network's own residual/spontaneous
  yaw activity was too weak and non-directional to meaningfully reorient
  the drone -- it drifted a small amount, then stalled against a wall, and
  stayed there for 9,000+ simulation steps without ever turning back toward
  the food.

This is a real, reportable negative result, not a bug we're hiding: it is
consistent with the Phase 3 finding that the decoded yaw signal is not yet
reliably stimulus-locked at the current (unfit) gain, and with the fact
that we've made no attempt yet to calibrate the network against real
firing-rate data (Phases 8-10, by design). Whether a properly calibrated
version of this same pipeline *would* produce fly-like idiothetic local
search remains the open question the centerpiece experiment (Phase 8-10)
is designed to answer -- Phase 7 has built the mechanic needed to ask that
question (an occludable target, distance/bearing/search-time logging), not
answered it.

## Calibration attempt: gain scaling does not recover a stimulus-locked signal
Following the Phase 7 finding above, we tested whether the "known current
limitation" (docs entry earlier in this file) could be fixed by simply
scaling `gain`. Method: for each of 8 gain values (0.05 to 1.2), ran three
matched trials -- LEFT stimulus, RIGHT stimulus, and a NEUTRAL control (no
photoreceptor disinhibition on either side) -- and computed the
baseline-subtracted effect (yaw_LEFT - yaw_NEUTRAL) and
(yaw_RIGHT - yaw_NEUTRAL). This subtraction isolates genuine stimulus
effect from the constant structural L/R asymmetry documented earlier
(DNa02_R receives more raw synapse weight than DNa02_L in this specific
reconstruction), which a raw LEFT-vs-RIGHT comparison conflates.

**Result: at every gain tested, both baseline-subtracted effects were ~0**
(exactly 0 at most gains, one noisy non-zero blip at gain=0.15-0.2 that did
not replicate a consistent pattern). This is a materially different, more
specific finding than "the signal is dominated by noise": it means the
LEFT/RIGHT disinhibition signal at the photoreceptors is not preserved
through the 5-synapse chain (photoreceptor -> lamina -> medulla -> T4/T5
-> HS/VS -> DNa02) *at all*, regardless of overall synaptic gain -- scaling
gain up or down does not surface a hidden signal, because the signal
itself is not reaching DNa02 as a differential in our current model.
Plausible reasons (not distinguished by this test): our LIF model's binary
spike + short (150-step) averaging window may not preserve enough graded
information across 5 hops; this specific real circuit may not encode a
simple static bearing this way at all (T4/T5 and HS/VS are tuned for
*motion*, and our static disinhibition stimulus has no true motion
content); or our AMBIENT_DRIVE placeholder for "the rest of the brain"
may dominate whatever real signal exists.

**Decision (2026-09-16, discussed with user):** we are not pursuing a
deeper methodological fix (e.g., injecting stimulus directly at T4/T5
instead of photoreceptors, or a continuous rate-coded activity
representation instead of binary spikes) at this time. Instead, this is
documented as a real, reportable negative result, and Phases 8-10 proceed
with this limitation explicit: any behavioral difference (or lack thereof)
between the connectome-driven controller and baseline controllers must be
interpreted in light of the fact that the connectome controller's yaw
output is not currently known to be stimulus-driven at all -- it may be
behaviorally close to a random/constant-bias controller. Testing exactly
that comparison is itself the valid, honest point of the Phase 10 baseline
experiment, not a foregone conclusion to avoid.

## Bug found during Phase 8-10: yaw sign was inverted in the physics mixer
While investigating why `rule_based` (a hand-written, geometrically-correct
"turn toward the brighter side" controller) performed no better than
`random` in the first benchmark run, we found a real bug: `src/drone/physics.py`'s
`mix_to_rotors` applied `+d_yaw` to the FL/BR rotor pair, which we had
*assumed* (by right-hand-rule reasoning about the actuator gear-vector
signs in `src/drone/drone.py`) would rotate the drone's forward vector
toward `+Y` (our documented "left"). **Empirical testing showed the
opposite**: a sustained positive `command.yaw` rotated the forward vector
toward `-Y` -- i.e., a positive yaw command was actually steering RIGHT,
not LEFT, contradicting `SteeringDecoder`'s documented convention
everywhere else in the codebase. Fixed by negating `d_yaw` in the mixer;
re-verified empirically after the fix (positive command.yaw now correctly
rotates toward +Y).

**This means every closed-loop demo before this fix (Phases 4-7) had the
drone turning the opposite of whatever direction the controller
"intended."** It does not invalidate the Phase 3 calibration-attempt
finding above (that analysis measured `SteeringDecoder`'s raw decoded
`yaw` value directly, never passing it through the drone's physics mixer,
so the sign bug doesn't touch it: baseline-subtracted L/R effect at DNa02
was, and remains, ~0 regardless of this fix). It DOES mean the Phase 4-7
visual demonstrations were showing correct closed-loop *mechanics* with an
inverted *steering direction* -- worth knowing if revisiting those phases'
recordings/demos.

After the fix, the Phase 8-10 benchmark (`src/experiments/benchmark.py`,
N=6 trials/condition) gave: connectome 33%, connectome_lesion_10 17%,
connectome_lesion_20 33%, connectome_lesion_30 17%, rule_based 0%, random
0% food-reach success rate. **We are not treating this as "the connectome
wins"** -- N=6 is far too small for that (a single trial outcome swings
the rate by 17 percentage points), and rule_based scoring *worse* than
random despite using a geometrically-correct formula suggests its specific
gain constants need tuning, not that hand-crafted rules are worse in
principle. More repeats (Phase 8's own stated experiment design) are
needed before drawing a real conclusion; this run establishes the fixed,
correct-sign pipeline and a first-pass number, not a final result.

## Follow-up: found a real (weak) signal, but it didn't fix behavior
After a user question about *why* the signal disappears, we re-ran the
baseline-subtraction diagnostic (LEFT stim / RIGHT stim / NEUTRAL vs.
gain) separately **at each pathway stage** instead of only at DNa02.
Result: the disinhibition effect is real and correctly-signed at the
photoreceptor stage (by construction), fades through lamina/medulla (small,
wrong-signed relative to the naive disinhibition expectation), is
essentially zero at T4/T5's population mean, but **a real, correctly-signed
effect reappears at HS/VS** (+0.55 for LEFT stim on the left side, at
gain=0.2) -- and partially at DNa02, but only for the RIGHT condition,
consistent with the previously-documented structural asymmetry (DNa02_R:
72 synapse-weight from 4 connections; DNa02_L: 66 from 3) dominating
DNa02's single-pair readout.

**Fix attempted:** changed `SteeringDecoder` (`src/control/motor_decoder.py`)
to read HS/VS population activity (dozens of neurons per hemisphere,
averaged) instead of the 2-neuron DNa02 pair -- HS-cell activity driving
steering is independently documented in the literature (Borst and
colleagues' optic-flow work), not an invented substitute. Re-running the
gain-sweep calibration test with this decoder showed a real, consistently-
signed LEFT/RIGHT effect at gain=0.2 (our already-used value) -- a genuine
improvement over the flat-zero result with the DNa02-only decoder.

**However: re-running the Phase 8-10 benchmark with this improved decoder
showed no measurable improvement in food-search success rate** --
connectome, rule_based, and random all scored ~17% again (N=6). So we have
a real, if weak, population-level signal that does NOT (yet) translate
into better closed-loop behavior. Plausible reasons, not yet distinguished:
the signal may only be reliable for the two extreme "fully left" / "fully
right" disinhibition conditions tested in the diagnostic, not the
continuously-varying bearing angles the real closed-loop task produces;
the effect size may simply be too small relative to the drone's physical
dynamics and the small-N sampling noise; or both. This is a genuine,
reportable negative result on top of a genuine, reportable partial
positive one -- we are not overselling the diagnostic improvement as
having fixed the behavioral problem, because the behavioral test says it
hasn't, at least not yet at N=6.

## Converged finding: the problem is static-vs-motion stimulus, not network scope
Following the HS/VS decoder's behavioral non-improvement (previous entry),
the user asked to "connect the full brain" if that might help. We checked
two concrete, real (not invented) expansions:

1. **The central complex** (EPG, PEN1/PEN2, ER ring neurons, PFN/hDelta --
   the fly's actual head-direction/navigation system, confirmed present in
   MaleCNS) -- checked direct connectivity from our pathway (HS/VS, T4/T5)
   to it and back to our descending neurons: **zero connections found**
   (one single-synapse central-complex -> DNa11 connection exists, to a
   *different* descending neuron than DNa02, and not from anything in our
   pathway). The central complex is not anatomically reachable from this
   pathway in this data -- ruled out, not pursued further.
2. **The real descending-neuron family HS/VS actually projects to**: not
   just DNa02, but also DNp17 (12 neurons), DNpe008 (19), DNb03 (4), DNg41
   (2), DNg46 (2) -- all confirmed with real, substantial HS/VS input and
   clean bilateral L/R splits -- plus LPi lobula-plate intrinsic neurons
   (confirmed real gain-control input to HS/VS in this data). Built
   `EXPANDED_STEERING_PATHWAY` (`src/brain/pathways.py`) with this real,
   verified expansion (31,532 neurons total) and re-ran both diagnostics
   (gain-sweep baseline subtraction, continuous-bearing monotonicity) reading
   from this larger 20-21-neuron-per-side descending family instead of
   DNa02 alone.

**Result: the same failure signature as every previous readout point**
(DNa02 alone, HS/VS alone, and now this expanded family): only one gain
out of seven tested showed a "consistent" baseline-subtracted effect, and
the continuous-bearing sweep showed no monotonic relationship at all
(values: -0.29, -0.53, -0.48, -0.49, -0.54, -1.02, -0.40 for bearing 0.0
through 1.0 -- not increasing, not decreasing, just noisy). This is the
third independent readout point to show this exact pattern.

**Conclusion (2026-09-16):** this convergence across three different
readout populations (a 2-neuron pair, an 8-type tangential-cell
population, and a verified 6-type/41-neuron descending-neuron family) rules
out "the network we simulate is too small/isolated" as the explanation.
The much more likely cause, now well-supported rather than merely
speculated: **T4/T5 and everything downstream of it are functionally
motion detectors** (direction-selective correlators, per Maisak et al.
2013's characterization), and our stimulus is a **static** disinhibition
(a fixed left-vs-right brightness difference held constant during the
stimulus window), not genuine motion. Feeding a static signal into a
motion-differencing circuit is architecturally the wrong kind of input,
regardless of gain or which downstream population reads it out. **Actually
fixing this would require simulating genuine optic flow** -- a spatially
organized visual field with real moving patterns across it, not just two
aggregate left/right brightness scalars -- which is a substantially larger
engineering effort than anything attempted so far in this project (it
would mean modeling a 2D or panoramic array of photoreceptor inputs with
real spatiotemporal structure, not a 2-scalar Braitenberg-style sensor).
We are documenting this as the well-supported root cause rather than
attempting that larger rebuild in this session.

## The static-vs-motion hypothesis was tested directly, and also ruled out
The previous entry concluded the most likely cause was "T4/T5 are motion
detectors, our stimulus is static." We tested this directly rather than
leaving it as a hypothesis. Using neuprint-python's `fetch_mean_synapses`,
we obtained real per-neuron synapse-position data for all 3,377 R1-R6
photoreceptors (previously unavailable via soma position -- see Phase 1's
finding that most early visual neurons lack soma coordinates; synapse
position is a different, separately-queryable field that *is* available).
Binned photoreceptors into 6 real retinotopic position groups per
hemisphere (by synapse-position quantile) and implemented a genuine
moving-bar stimulus: a soft disinhibition window sweeping smoothly across
the 6 bins over 300 steps, in each of the two possible directions.

**Result: T4/T5 subtype activity showed no meaningful forward-vs-reverse
difference** (all subtypes: -0.011 to +0.024 spike-count difference over
400 steps -- noise-level, no direction-dependent pattern at all). Real
motion, not just a static disinhibition, still failed to elicit
direction-selective responses.

**This rules out the static-vs-motion hypothesis and points to a more
fundamental limitation of our simulator itself, not the stimulus design:**
real T4/T5 direction selectivity is understood to arise from comparing
*differently time-delayed* inputs from distinct medulla neuron types
(e.g. Mi1, Mi4, Mi9, Tm3, each with different intrinsic dynamics) --
essentially a Hassenstein-Reichardt correlator computation. Our simulator
(`src/brain/simulator.py`) gives every neuron the same membrane time
constant, uses a fixed one-timestep synaptic delay for every connection
regardless of type, and treats every neuron as a discrete spiking unit
even though (per the Phase 2 finding) the medulla neurons T4/T5 actually
compare are believed to be non-spiking, graded-potential cells. None of
that differential-delay machinery exists in our model, so there is no
mechanism by which it *could* produce direction selectivity, regardless of
how realistic the visual stimulus is.

**Conclusion:** fixing this is not a stimulus or network-topology problem
(both were tested and ruled out) -- it would require a substantially more
biophysically detailed simulator: per-neuron-type membrane time constants
fit to real data, explicit synaptic delay/filtering per connection type,
and/or graded (non-spiking) dynamics for the neuron classes documented as
non-spiking. This is a foundational upgrade to the simulation engine, not
a parameter tweak, and was not attempted in this session -- documented
here as the actual, specific, well-supported next step for anyone
continuing this work.

## Pretrained FlyVis rescue and its boundary (2026-09-17)

We subsequently integrated the official pretrained FlyVis model instead of
claiming that the hand-written uniform LIF dynamics had become biologically
adequate. Opposite moving bars produced opposite T4/T5 population signs, and
the recurrent model now runs online on rendered drone-camera frames. On this
memory-constrained Windows machine the online network uses a central
91-ommatidium lattice (`extent=5`); the same 734 learned type-shared parameters
are restored, but this reduced spatial extent is an engineering compromise.

Food recognition is **not** supplied by FlyVis: an explicit orange-pixel
salience detector provides target horizontal error. FlyVis contributes a
T4/T5 opponent optic-motion damping term to yaw. A three-start paired ablation
reached food in 3/3 trials in both conditions and reduced mean steps from
1122.7 to 1044.7 with FlyVis. This is evidence that the pretrained visual model
participates usefully in this loop, not that a fly connectome autonomously
learned food preference or search.

Attempts to remove the orange-pixel salience layer (Phases 20-23) did not
generalize. An R1-R6 centroid localized isolated spots, and a linear readout
looked accurate within one scene, but a background-held-out evaluation fell to
27.8% visibility accuracy and 57.1% left/right sign accuracy. The apparently
strong within-scene score was overfitting. Consequently the project continues
to label colour-based food selection as an engineering input; the pretrained
grayscale FlyVis network is used only for motion computation.

Phases 24-28 add an experimental multichannel spectral-input approximation. R7
gets camera blue as a UV proxy, R8 gets green, and R1-R6 supply luminance. The
readout derives `red ~= 3*luminance - green - blue` and uses `red - green`
salience. It located orange food in 3/3 geometrically distinct trials and
rejected the tested pure-green and pure-blue distractors. It must not be
described as a faithful fly colour system: the camera has no UV channel,
R7/R8 subtype diversity is omitted, the opponent equation is engineered, and
arbitrary hues, coloured backgrounds and lighting remain unresolved.

Phase 29 showed 4/4 closed-loop success for orange, red, amber and dim-orange
food materials in the same achromatic arena. This is a narrow rendered-colour
robustness result, not colour constancy: scene illumination was unchanged and
the target remained an untextured sphere.

Phase 30 passed neutral lighting scales from 0.25 to 1.5, but red and amber
distractor spheres attracted the controller before it recovered. A narrow hue
gate was tried and reverted after it traded away food-hue generalization and
caused one distractor trial to fail. The resulting limitation is fundamental
to the present observation model: colour opponency identifies a warm visual
category, not object identity or biological food value.

## What is not known / unresolved (update as we investigate)
- Whether MaleCNS annotations include a confirmed, complete set of
  "descending neurons" analogous to those characterized in the MANC
  connectome — to verify in Phase 2/3 via neuPrint queries, not assumed.
- Full sign (excitatory/inhibitory) coverage of the connectivity matrix.

## Explicit non-claims
We will not say or imply:
- "This is the actual fly brain" — it is a structural map used to constrain
  a computational model.
- "The fly/drone is thinking" — no claim about cognition or experience.
- "We reproduced fly intelligence" — we are testing whether connectome
  *structure* provides useful inductive bias for a control task, following
  the same framing as FLYNN (arXiv:2607.00025).
- "The drone is controlled exactly like a real fly" — see engineering
  abstractions above; a quadrotor is not a fly body.
- Novelty of the general concept "connectome controls a virtual drone" —
  see `research/existing_projects.md`; this has already been done by
  garyb9/fly-drone, haltere, and others. Our specific claimed contribution is
  the combination of full-connectome + vision-only food-seeking + systematic
  lesion/baseline benchmarking (see `docs/experiments.md`, written in later
  phases).
- The stylized fly visible in the dashboard is a rendering shell only. Its
  wings do not generate aerodynamic forces and its legs do not implement fly
  locomotion; the underlying body is still the stabilized four-actuator
  quadrotor model.
- Dashboard casting and looming-escape rules are biologically inspired
  behavioral motifs, not reconstructed fly motor commands. Five MuJoCo rays
  are an engineered proximity proxy for looming vision; they do not represent
  literal invisible range sensors possessed by a fruit fly.
