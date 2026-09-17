"""Named, verified neuron populations used as inputs/outputs for experiments.

Every type name here was confirmed to exist in the live MaleCNS v1.0
neuPrint instance before use (see research notes in this file's git history
/ the exploration scripts) -- nothing here is invented. Functional roles
(ON/OFF pathway, motion direction) are established Drosophila literature,
not something read directly out of the connectome itself; the connectome
only gives us structural connectivity. See docs/scientific_assumptions.md.

The elementary motion-detection circuit (photoreceptors -> lamina ->
medulla -> T4/T5) is one of the best-characterized circuits in the fly
brain, e.g. Takemura et al. 2013/2017 (lamina/medulla connectomics),
Maisak et al. 2013 (T4/T5 control optic flow), Shinomiya et al. (T4/T5
connectomics). We use it for Phase 2 because it is small, real, and has an
unambiguous, literature-supported input->output direction, not because the
MaleCNS connectome tags it as "the visual pathway" itself.
"""
from __future__ import annotations

# Outer photoreceptors (motion/luminance vision) and inner photoreceptors
# (color/UV/polarized light) -- the sensory transduction stage. Real type
# names confirmed in MaleCNS: 'R1-R6' is a single combined type in this
# dataset (not split into R1..R6 individually); R7/R8 are split into
# dorsal/pale/yellow subtypes reflecting the fly's stochastic ommatidial
# color-vision subtypes (dorsal rim, pale, yellow).
PHOTORECEPTORS = ["R1-R6", "R7d", "R7p", "R7y", "R8d", "R8p", "R8y"]

# Lamina monopolar cells: first synaptic relay after the photoreceptors.
# L1 is the principal input to the ON-motion pathway, L2 to the OFF-motion
# pathway (Joesch et al. 2010; Clark et al. 2011); L3 carries luminance
# information into both; L4/L5 have lateral/modulatory roles.
LAMINA = ["L1", "L2", "L3", "L4", "L5"]

# Medulla neurons directly postsynaptic to the ON/OFF lamina relays.
# Mi1 continues the ON pathway (from L1), Tm1 the OFF pathway (from L2).
MEDULLA = ["Mi1", "Tm1"]

# The four canonical direct inputs to T4 (ON-pathway motion detector),
# per Takemura et al. 2017 (eLife, "The comprehensive connectome of a
# neural substrate for 'ON' motion detection") and Arenz et al. 2017 /
# Haag et al. 2017 (eLife, "A common directional tuning mechanism..."):
# Mi1 and Tm3 are FAST, transient, band-pass-filtered inputs at the
# T4 dendrite's center; Mi9 and Mi4 are SLOW, sustained, low-pass-filtered
# inputs offset to either side (Mi9 null-direction side providing delayed
# disinhibition, Mi4 preferred-direction side providing delayed
# inhibition) -- this fast-center/slow-flank asymmetry, combined with a
# nonlinear (shunting-inhibition) interaction our simulator does not
# implement, is the real mechanism believed to produce T4 direction
# selectivity. Verified present in MaleCNS (2026-09-16) with strong real
# connectivity to T4a specifically (Mi1: 115,153 weight; Tm3: 50,467;
# Mi9: 37,468; Mi4: 22,018 -- matching the literature's ranking).
T4_INPUT_LINES = ["Mi4", "Mi9", "Tm3"]  # Mi1 already in MEDULLA above
FAST_INPUT_TYPES = ["Mi1", "Tm3"]   # per literature: fast/transient/band-pass
SLOW_INPUT_TYPES = ["Mi4", "Mi9"]   # per literature: slow/sustained/low-pass

# Direction-selective motion detectors: T4 for ON edges, T5 for OFF edges,
# each with four subtypes (a/b/c/d) tuned to one of four cardinal motion
# directions (Maisak et al. 2013). These project on to the lobula plate,
# which contains the tangential cells that pool direction-selective signal
# into wide-field optic-flow responses -- not yet included here; that is
# the natural next hop for a later phase, not invented for this one.
MOTION_DETECTORS = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

VISUAL_MOTION_PATHWAY = PHOTORECEPTORS + LAMINA + MEDULLA + T4_INPUT_LINES + MOTION_DETECTORS

# Lobula plate tangential cells: wide-field motion integrators postsynaptic
# to T4/T5, the classic optic-flow-sensing population (e.g. Maisak et al.
# 2013; Schnell et al. 2010). Verified present in MaleCNS under these type
# names via direct query (2026-09-16); confirmed receiving real T4/T5 input
# in this dataset (T4d/T5d -> VS is the single strongest connection found).
LOBULA_PLATE_TANGENTIAL = ["HSN", "HSE", "HSS", "HST", "VS", "VST1", "VST2", "VSm"]

# Steering descending neuron: DNa02 is characterized in Rayshubskiy et al.
# 2020 (eLife, "Neural circuit mechanisms for steering control in walking
# Drosophila") as receiving HS-cell visual-motion input and driving
# ipsiversive turning (activating one side's DNa02 turns the fly toward
# that side) during WALKING. Verified present in MaleCNS (2 neurons, a
# bilateral pair, matching the literature's description of DNa02 as an
# individually identifiable neuron) and verified to receive direct input
# specifically from HSS/HSE/HSN/HST and VS in this dataset, with clean
# ipsilateral wiring (HSS_L -> DNa02_L, HSS_R -> DNa02_R, no crossing found).
# No direct T4/T5 -> DNa02 connection exists in this data -- HS/VS is a
# real, necessary intermediate stage, not something we inserted.
#
# IMPORTANT: DNa02's proven behavioral role is STEERING DURING WALKING, not
# flight and not drone control. Using its L-vs-R activity difference to
# drive a drone's yaw (src/control/motor_decoder.py) is an ENGINEERING
# ABSTRACTION that reuses a real, verified sensorimotor logic (visual
# motion -> lateralized descending activity -> ipsiversive turning) for a
# different effector than the one it was characterized for. See
# docs/scientific_assumptions.md.
STEERING_DESCENDING = ["DNa02"]

FULL_STEERING_PATHWAY = VISUAL_MOTION_PATHWAY + LOBULA_PLATE_TANGENTIAL + STEERING_DESCENDING

# Looming/collision detection: lobula columnar neurons downstream of T4/T5,
# well-documented in the literature as the Drosophila visual looming
# pathway feeding the Giant Fiber escape circuit (e.g. von Reyn et al. 2014,
# Current Biology, "Feature integration drives probabilistic behavior in
# the Drosophila escape response"). Verified present in MaleCNS
# (2026-09-16) and verified receiving real, strong T4/T5 input in this
# dataset (e.g. T5d -> LPLC2: 4,817 connections, 19,165 total synapse
# weight -- among the strongest connections found in any of our pathway
# verification queries so far).
LOOMING_DETECTORS = ["LC4", "LC6", "LC16", "LPLC1", "LPLC2", "LC11", "LC12", "LC15"]

# The Giant Fiber (GF) escape descending neuron -- one of the most
# extensively characterized single identified neurons in the Drosophila
# nervous system, driving the fast escape takeoff/jump response to a
# looming threat. Confirmed present in MaleCNS labeled "DNp01(GF)" (its
# own instance name in this dataset explicitly identifies it as the Giant
# Fiber) and confirmed receiving direct, strong input from LC4 and LPLC2
# specifically (LC4->DNp01: ~6,362 combined synapse weight; LPLC2->DNp01:
# ~4,862). As with DNa02/steering, GF's real, published behavioral role
# (a fast jump/wing-flick escape takeoff) does not map onto a drone --
# using its activity to trigger an avoidance maneuver is an ENGINEERING
# ABSTRACTION that reuses a verified sensorimotor logic (looming -> urgent
# avoidance response), not a claim that a drone "jumps" the way a fly does.
ESCAPE_DESCENDING = ["DNp01"]

FULL_NAVIGATION_PATHWAY = FULL_STEERING_PATHWAY + LOOMING_DETECTORS + ESCAPE_DESCENDING

# --- Expanded steering network (2026-09-16) ---
# Added after diagnosing that DNa02's 2-neuron readout is dominated by a
# structural quirk (see docs/scientific_assumptions.md's signal-loss
# diagnostic) and a user question about whether connecting more of the real
# brain -- rather than our crude constant AMBIENT_DRIVE placeholder for
# "the rest of the brain" -- might help. We checked whether the central
# complex (the fly's actual head-direction/navigation system: EPG, PEN1/
# PEN2, ER ring neurons, PFN/hDelta -- all confirmed present in MaleCNS)
# connects to our pathway: it does NOT (zero direct HS/VS-or-T4/T5 ->
# central-complex connections found, and only a single-synapse central-
# complex -> DNa11 connection, not DNa02). So the central complex cannot
# be wired in without inventing a connection that doesn't exist in this
# data -- not pursued.
#
# Instead, verified what's REAL and directly connected: HS/VS receives
# substantial input from lobula-plate intrinsic (LPi) inhibitory
# interneurons -- these provide gain control/normalization for HS/VS in
# the real circuit (a well-established role for LPi cells in the fly optic
# flow literature) -- and HS/VS outputs to a WHOLE FAMILY of descending
# neurons, not just DNa02: DNp17 (12 neurons), DNpe008 (19 neurons), DNb03
# (4), DNg41 (2), DNg46 (2), all confirmed with clean bilateral L/R splits.
# DNp17/DNpe008/DNb03/DNg41/DNg46 do NOT have the same published behavioral
# characterization DNa02 has (no equivalent of Rayshubskiy et al. 2020 for
# these specifically, as far as we've verified) -- including them is
# justified by "real, substantial anatomical input from HS/VS," a weaker
# but still real and honestly-labeled basis, not invented.
LPI_INTERNEURONS = ["LPi34", "LPi21", "LPi14", "LPi12", "LPi3b", "LPi2b", "LPi3a"]

EXPANDED_STEERING_DESCENDING = ["DNa02", "DNp17", "DNpe008", "DNb03", "DNg41", "DNg46"]

EXPANDED_STEERING_PATHWAY = (
    VISUAL_MOTION_PATHWAY + LOBULA_PLATE_TANGENTIAL + LPI_INTERNEURONS + EXPANDED_STEERING_DESCENDING
)

# Verified neuron counts as of the query in this file's development
# (2026-09-16 snapshot); re-verify with fetch_subnetwork if this drifts.
# R1-R6: 3377, R7 (d/p/y): 82+332+482, R8 (d/p/y): 76+330+481,
# L1-L5: ~1770-1790 each, Mi1/Tm1: ~1773/1777,
# T4a-d: 1684/1690/1778/1709, T5a-d: 1664/1715/1720/1620
# Total subnetwork as fetched: 31,120 neurons, 406,462 (pre,post,roi) edges.


def hemisphere_stimulus_populations(neuron_df, side: str) -> "pd.Series":
    """Return the boolean mask selecting photoreceptors on the given side.

    side must be 'L' or 'R'. This is the entry point for the Phase 2
    LEFT/RIGHT visual stimulus: each compound eye projects only to its own
    (ipsilateral) optic lobe in real Drosophila neuroanatomy, so driving
    only one hemisphere's photoreceptors is a real anatomical statement, not
    an invented mapping. "CENTER" has no single verified neuron population
    in this pathway (no isolated binocular-frontal-field cell type was
    confirmed here) -- treat simultaneous bilateral stimulation as an
    explicit engineering simplification for "center", documented in
    docs/scientific_assumptions.md, not a claim about a real center-field
    circuit.
    """
    assert side in ("L", "R")
    is_photoreceptor = neuron_df["type"].isin(PHOTORECEPTORS)
    # fetch_adjacencies()'s neuron table doesn't include a somaSide column.
    # We derive hemisphere from the `instance` field's "_L"/"_R" suffix
    # instead, which covers all 31,120 neurons in this subnetwork (vs. the
    # somaSide property itself, which is null for the combined "R1-R6"
    # type) -- cross-checked against somaSide for the types where it *is*
    # populated (e.g. R8y, R8p) and it agrees in every sampled case.
    is_side = neuron_df["instance"].str.endswith(f"_{side}")
    return is_photoreceptor & is_side
