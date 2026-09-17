"""Motor decoder: connectome activity -> drone control signals.

Implements two verified pathways, both traced end-to-end in MaleCNS (see
src/brain/pathways.py for citations and verification queries):

1. STEERING: T4/T5 -> HS/VS (lobula plate tangential cells) -> DNa02.
   DNa02 is characterized (Rayshubskiy et al. 2020) for visual-motion-driven
   walking steering; using it for drone yaw is an explicit engineering
   abstraction, not a proven fly-to-drone mapping.
2. COLLISION AVOIDANCE: T4/T5 -> LC4/LPLC2 (looming detectors) -> DNp01
   (the Giant Fiber escape neuron). GF's real role is a fast jump/takeoff
   escape response (von Reyn et al. 2014); using its activity (and the
   lateralized LC4/LPLC2 signal feeding it) to trigger an avoidance turn is
   likewise an engineering abstraction.

FORWARD/UP are NOT grounded in any verified pathway -- we have not
identified one. `forward` in MotorCommand is a plain constant baseline set
by the caller (see src/experiments/phase6_navigation.py), not decoded from
neural activity, and is documented as such wherever it's set. We do not
invent a neural forward pathway to avoid saying so.

Each decoder is intentionally a thin, replaceable module (per the brief):
swap it for a different mapping without touching the simulator or the rest
of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Runtime import would load PyTorch even when callers only need the small
    # MotorCommand dataclass (as the FlyVis dashboard does).  Keeping this as
    # a type-only dependency lets the UI start before its neural backend.
    from src.brain.simulator import ConnectomeNetwork


@dataclass
class MotorCommand:
    yaw: float           # positive = turn left, negative = turn right (see below)
    left_turn: float      # non-negative magnitude, for a "LEFT TURN" activity bar
    right_turn: float     # non-negative magnitude, for a "RIGHT TURN" activity bar
    forward: float = 0.0   # NOT decoded from neural activity -- caller-set baseline only
    up: float = 0.0         # NOT grounded in a verified pathway yet -- always 0.0
    avoidance_urgency: float = 0.0  # DNp01 activity magnitude, for display/logging


class SteeringDecoder:
    """Decodes HS/VS (lobula plate tangential cell) population activity,
    averaged per hemisphere, into a yaw command.

    Changed from an earlier version that read DNa02_L/DNa02_R directly
    (2026-09-16). Diagnostic testing (per-stage baseline-subtracted LEFT/
    RIGHT effect, see docs/scientific_assumptions.md's "signal-loss
    diagnostic" entry) found a real, consistently-signed stimulus effect at
    HS/VS that did not reliably survive the last hop to DNa02 -- DNa02 gets
    input from only 3-4 real connections in this subnetwork (a structural
    asymmetry: DNa02_R has more total synapse weight than DNa02_L), so its
    single-pair readout is dominated by that structural quirk rather than
    the broader population signal. Reading a larger population (HS: HSN/
    HSE/HSS/HST, VS: VS/VST1/VST2/VSm, dozens of neurons per hemisphere
    instead of 1) averages that idiosyncrasy out.

    This is not an invented substitute: HS cell activity driving steering
    is itself independently documented in the literature (e.g. work from
    Borst and colleagues on HS-cell optic-flow responses correlating with
    turning), not just an intermediate relay we're reading out of
    convenience. Sign convention: HS/VS receiving more/disinhibited
    ipsilateral input drives turning toward that side, matching the same
    ipsiversive logic DNa02 was used for; verified empirically as the
    correctly-signed direction in the diagnostic referenced above.
    """

    STEERING_GAIN = 3.0  # compensates for population-mean activity having
    # smaller magnitude swings than a single neuron's raw spike count;
    # tuned so decoded yaw is on the same rough scale the rest of the
    # pipeline (src/drone/physics.py's YAW_GAIN etc.) already expects.

    def __init__(self, network: ConnectomeNetwork):
        from src.brain.pathways import LOBULA_PLATE_TANGENTIAL

        df = network.neuron_df
        hsvs = df[df["type"].isin(LOBULA_PLATE_TANGENTIAL)]
        left = hsvs[hsvs["instance"].str.endswith("_L")]
        right = hsvs[hsvs["instance"].str.endswith("_R")]
        if len(left) == 0 or len(right) == 0:
            raise RuntimeError(
                f"Expected HS/VS neurons on both sides, found {len(left)} left / "
                f"{len(right)} right. Re-verify against neuPrint -- do not guess."
            )
        self.left_idx = np.array([network.index_of(b) for b in left["bodyId"]], dtype=np.int64)
        self.right_idx = np.array([network.index_of(b) for b in right["bodyId"]], dtype=np.int64)

    def decode(self, activity) -> MotorCommand:
        """activity: array-like of per-neuron activity (e.g. a recent spike
        rate or smoothed spike count), indexed the same way as the network."""
        left_activity = float(activity[self.left_idx].mean())
        right_activity = float(activity[self.right_idx].mean())
        yaw = self.STEERING_GAIN * (left_activity - right_activity)
        return MotorCommand(
            yaw=yaw,
            left_turn=max(yaw, 0.0),
            right_turn=max(-yaw, 0.0),
        )


class AvoidanceDecoder:
    """Decodes the looming/Giant-Fiber pathway into an avoidance yaw term.

    DNp01 (bilateral pair) fires as a fairly global "threat" trigger rather
    than a steering signal in its own right (real GF activation drives a
    stereotyped symmetric takeoff, not a directional turn) -- so we use
    max(DNp01_L, DNp01_R) as the *urgency* magnitude, and the lateralized
    upstream looming signal (mean LC4+LPLC2 activity per hemisphere) to
    pick a *direction*: turn away from the side with the stronger looming
    signal (the nearer obstacle). This split (urgency from the descending
    neuron, direction from its upstream lateralized input) is our own
    engineering design for combining these two verified but not
    inherently-directional signals -- not something read directly off the
    connectome. See docs/scientific_assumptions.md.
    """

    def __init__(self, network: ConnectomeNetwork):
        df = network.neuron_df
        dnp01 = df[df["type"] == "DNp01"]
        left_dn = dnp01[dnp01["instance"].str.endswith("_L")]
        right_dn = dnp01[dnp01["instance"].str.endswith("_R")]
        if len(left_dn) != 1 or len(right_dn) != 1:
            raise RuntimeError(
                f"Expected exactly one DNp01_L and one DNp01_R, found "
                f"{len(left_dn)} left / {len(right_dn)} right. Re-verify against neuPrint."
            )
        self.dnp01_left_idx = network.index_of(int(left_dn.iloc[0]["bodyId"]))
        self.dnp01_right_idx = network.index_of(int(right_dn.iloc[0]["bodyId"]))

        from src.brain.pathways import LOOMING_DETECTORS
        looming = df[df["type"].isin(LOOMING_DETECTORS)]
        left_looming = looming[looming["instance"].str.endswith("_L")]
        right_looming = looming[looming["instance"].str.endswith("_R")]
        # plain numpy indices, not network.indices_of()'s GPU tensor: decode()
        # receives `activity` as a numpy array (already .cpu().numpy()'d by
        # the caller), not a torch tensor.
        self.looming_left_idx = np.array([network.index_of(b) for b in left_looming["bodyId"]], dtype=np.int64)
        self.looming_right_idx = np.array([network.index_of(b) for b in right_looming["bodyId"]], dtype=np.int64)

    def decode(self, activity) -> tuple[float, float]:
        """Returns (avoidance_yaw, urgency). avoidance_yaw follows the same
        sign convention as SteeringDecoder (positive = turn left)."""
        urgency = max(float(activity[self.dnp01_left_idx]), float(activity[self.dnp01_right_idx]))
        left_looming = float(activity[self.looming_left_idx].mean()) if len(self.looming_left_idx) else 0.0
        right_looming = float(activity[self.looming_right_idx].mean()) if len(self.looming_right_idx) else 0.0
        # turn away from the stronger (nearer) looming side
        avoidance_yaw = urgency * (right_looming - left_looming)
        return avoidance_yaw, urgency


class NavigationDecoder:
    """Combines SteeringDecoder (normal turning) and AvoidanceDecoder
    (urgent obstacle avoidance) into one MotorCommand. Avoidance is added
    on top of, not instead of, steering -- weighted by its own urgency, so
    it dominates only when the looming/Giant-Fiber pathway is strongly
    active."""

    def __init__(self, network: ConnectomeNetwork, avoidance_gain: float = 2.0):
        self.steering = SteeringDecoder(network)
        self.avoidance = AvoidanceDecoder(network)
        self.avoidance_gain = avoidance_gain

    def decode(self, activity, forward: float = 0.0) -> MotorCommand:
        base = self.steering.decode(activity)
        avoidance_yaw, urgency = self.avoidance.decode(activity)
        yaw = base.yaw + self.avoidance_gain * avoidance_yaw
        return MotorCommand(
            yaw=yaw,
            left_turn=max(yaw, 0.0),
            right_turn=max(-yaw, 0.0),
            forward=forward,
            avoidance_urgency=urgency,
        )


class SensoryRescueDecoder:
    """Temporary, explicit phototaxis readout for closed-loop validation.

    This reads real simulated photoreceptor population activity, but bypasses
    the currently non-direction-selective T4/T5 -> HS/VS computation.  It is
    therefore an engineering rescue controller, not evidence that the full
    biological motion pathway works.  Its purpose is to keep the embodied
    connectome pipeline usable while the graded temporal model is developed.
    """

    def __init__(self, network: ConnectomeNetwork, steering_gain: float = 12.0,
                 avoidance_gain: float = 3.0, search_yaw: float = 0.6,
                 dark_activity_threshold: float = 0.095,
                 alignment_activity_threshold: float = 0.02):
        from src.brain.pathways import PHOTORECEPTORS

        df = network.neuron_df
        photo = df[df["type"].isin(PHOTORECEPTORS)]
        left = photo[photo["instance"].str.endswith("_L")]
        right = photo[photo["instance"].str.endswith("_R")]
        self.left_idx = np.array([network.index_of(b) for b in left["bodyId"]], dtype=np.int64)
        self.right_idx = np.array([network.index_of(b) for b in right["bodyId"]], dtype=np.int64)
        if not len(self.left_idx) or not len(self.right_idx):
            raise RuntimeError("Sensory rescue requires bilateral photoreceptor populations")
        self.steering_gain = steering_gain
        self.search_yaw = search_yaw
        self.dark_activity_threshold = dark_activity_threshold
        self.alignment_activity_threshold = alignment_activity_threshold
        self.avoidance = AvoidanceDecoder(network)
        self.avoidance_gain = avoidance_gain

    def decode(self, activity, forward: float = 0.0) -> MotorCommand:
        # Light reduces tonic photoreceptor activity.  A target on the left
        # therefore means left activity < right activity and must yield +yaw.
        left_activity = float(activity[self.left_idx].mean())
        right_activity = float(activity[self.right_idx].mean())
        mean_activity = 0.5 * (left_activity + right_activity)
        if mean_activity >= self.dark_activity_threshold:
            # No bilateral light response: slowly scan until the target
            # enters the forward field. This fallback is deliberately
            # explicit; it is not claimed as emergent fly search behavior.
            sensory_yaw = self.search_yaw
            effective_forward = 0.0
        else:
            sensory_yaw = self.steering_gain * (right_activity - left_activity)
            # First rotate until the target is near the bilateral center;
            # translating while it is only at the field edge sends the body
            # past the target before the rate loop can finish the turn.
            aligned = abs(right_activity - left_activity) <= self.alignment_activity_threshold
            effective_forward = forward if aligned else 0.0
        avoidance_yaw, urgency = self.avoidance.decode(activity)
        yaw = sensory_yaw + self.avoidance_gain * avoidance_yaw
        return MotorCommand(
            yaw=yaw,
            left_turn=max(yaw, 0.0),
            right_turn=max(-yaw, 0.0),
            forward=effective_forward,
            avoidance_urgency=urgency,
        )
