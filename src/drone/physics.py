"""Motor mixer: MotorCommand -> 4 rotor thrust values (yaw + throttle only).

Standard quadrotor X-configuration mixing (real multirotor flight-control
math, not fly-specific): throttle moves all rotors together, yaw via
differential thrust across the two diagonal (same-spin-direction) rotor
pairs.

Only `command.yaw` is currently driven by a verified neural pathway (see
src/control/motor_decoder.py). `command.up` is always 0.0 from the decoder
(no verified pathway identified yet) -- not invented.

`command.forward` is NOT mixed into rotor thrust here. An earlier version
of this module tilted the airframe (pitch) to translate forward, the way a
real quadrotor does -- that needed a full PD attitude-stabilization loop to
avoid an unbounded-pitch instability (a constant differential thrust is a
constant torque with nothing to stop it, so pitch keeps accelerating until
the drone flips), and even with that PD loop in place, the closed loop
diverged into a runaway climb after a wall collision during Phase 6
testing. Tuning a robust flight controller is a substantial control-theory
project on its own and not what this project is actually about -- so
forward locomotion is applied as a direct external force instead (see
Drone.apply_forward_force in src/drone/drone.py), keeping the airframe
level. This mixer only ever handles yaw/throttle, which were already
proven stable in Phase 4.
"""
from __future__ import annotations

import numpy as np

from src.control.motor_decoder import MotorCommand
from src.drone.drone import HOVER_THRUST_PER_ROTOR

YAW_GAIN = 0.2  # bumped up from an initial 0.05 purely so the (currently
# weak, per docs/scientific_assumptions.md) decoded yaw signal is visible
# within a normal viewing session -- a demo-visibility tweak, not a
# scientific calibration; it does not change whether the signal is
# stimulus-locked, only how fast the drone turns per unit of it.
THROTTLE_GAIN = 0.05


def mix_to_rotors(command: MotorCommand) -> np.ndarray:
    # Positive command.yaw means a left turn: local +X must rotate toward
    # world/local +Y from the identity pose.  A direct MuJoCo regression
    # test caught that this sign had previously been inverted despite the
    # old comment claiming otherwise.
    d_yaw = YAW_GAIN * command.yaw
    base = HOVER_THRUST_PER_ROTOR + THROTTLE_GAIN * command.up
    fl = base + d_yaw
    fr = base - d_yaw
    bl = base - d_yaw
    br = base + d_yaw
    return np.clip([fl, fr, bl, br], 0.0, 4.0 * HOVER_THRUST_PER_ROTOR)
