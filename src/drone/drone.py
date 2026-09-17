"""A simple quadrotor drone simulated in MuJoCo.

Standard X-configuration quadrotor: 4 rotors at the corners of a box body,
each producing an upward thrust force plus a reaction yaw torque (adjacent
rotors spin in opposite directions, like a real quadrotor). Differential
thrust across rotors produces pitch (forward/backward tilt), roll
(left/right tilt), and yaw (turning) -- this is real, standard multirotor
physics, not something specific to flies. The fly-connectome part of this
project is the *controller* driving these 4 rotor commands, not the
airframe itself (see docs/scientific_assumptions.md: "the drone itself...
is an engineering abstraction, not a fly's body").
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

MASS_KG = 0.5
ARM_LEN = 0.15  # m, distance from center to each rotor
GRAVITY = 9.81
HOVER_THRUST_PER_ROTOR = MASS_KG * GRAVITY / 4

# Rotor layout (X configuration), spin direction for yaw reaction torque:
# FL(+,+) and BR(-,-) spin CCW (+yaw reaction), FR(+,-) and BL(-,+) spin CW (-yaw reaction)
ROTOR_OFFSETS = {
    "FL": (+ARM_LEN, +ARM_LEN, +1),
    "FR": (+ARM_LEN, -ARM_LEN, -1),
    "BL": (-ARM_LEN, +ARM_LEN, -1),
    "BR": (-ARM_LEN, -ARM_LEN, +1),
}
YAW_TORQUE_COEFF = 0.02  # N*m per N of thrust, typical small-prop ratio

def drone_xml(start_pos=(0.0, 0.0, 1.0), start_yaw_deg: float = 0.0) -> str:
    """Bakes `start_pos` (and optionally a starting yaw) into the compiled
    model as the drone's qpos0 -- needed because MuJoCo's own viewer
    "reset" (Backspace) reloads qpos0, not whatever a script assigns to
    data.qpos at runtime. Phase 6's walled arena passes a start position
    away from the walls for exactly this reason (a user-triggered viewer
    reset otherwise puts the drone back near the model's XML-default
    position, close to a wall, and immediately re-triggers the
    wall-collision instability -- see Drone.reset_if_unstable's docstring).
    Phase 7 additionally uses start_yaw_deg so the drone can start facing
    away from the food target."""
    half = np.radians(start_yaw_deg) / 2.0
    qw, qz = np.cos(half), np.sin(half)  # rotation about world +Z only
    return DRONE_XML_TEMPLATE.format(x=start_pos[0], y=start_pos[1], z=start_pos[2], qw=qw, qz=qz)


DRONE_XML_TEMPLATE = f"""
<body name="drone" pos="{{x}} {{y}} {{z}}" quat="{{qw}} 0 0 {{qz}}">
  <freejoint name="drone_free"/>
  <inertial pos="0 0 0" mass="{MASS_KG}" diaginertia="0.0023 0.0023 0.004"/>
  <!-- The physics remains the documented stabilized quadrotor abstraction,
       but the visible shell is a stylized fruit fly for the interactive demo. -->
  <geom name="drone_body" type="ellipsoid" size="0.10 0.075 0.065" rgba="0.25 0.16 0.08 1"/>
  <geom name="fly_abdomen" type="ellipsoid" pos="-0.14 0 -0.005" size="0.16 0.065 0.055" rgba="0.12 0.09 0.055 1" contype="0" conaffinity="0"/>
  <geom name="fly_abdomen_band_1" type="ellipsoid" pos="-0.10 0 -0.004" size="0.018 0.067 0.057" rgba="0.86 0.58 0.12 1" contype="0" conaffinity="0"/>
  <geom name="fly_abdomen_band_2" type="ellipsoid" pos="-0.17 0 -0.004" size="0.016 0.058 0.052" rgba="0.76 0.45 0.08 1" contype="0" conaffinity="0"/>
  <geom name="fly_head" type="sphere" pos="0.105 0 0.018" size="0.066" rgba="0.32 0.20 0.09 1" contype="0" conaffinity="0"/>
  <geom name="fly_eye_left" type="ellipsoid" pos="0.135 0.050 0.030" size="0.045 0.025 0.040" rgba="0.72 0.08 0.055 1" contype="0" conaffinity="0"/>
  <geom name="fly_eye_right" type="ellipsoid" pos="0.135 -0.050 0.030" size="0.045 0.025 0.040" rgba="0.72 0.08 0.055 1" contype="0" conaffinity="0"/>
  <geom name="fly_wing_left" type="ellipsoid" pos="-0.035 0.135 0.050" size="0.14 0.050 0.009" quat="0.7071 0 0 0.7071" rgba="0.72 0.90 1 0.42" contype="0" conaffinity="0"/>
  <geom name="fly_wing_right" type="ellipsoid" pos="-0.035 -0.135 0.050" size="0.14 0.050 0.009" quat="0.7071 0 0 0.7071" rgba="0.72 0.90 1 0.42" contype="0" conaffinity="0"/>
  <geom name="fly_leg_fl" type="capsule" fromto="0.04 0.05 -0.02 0.12 0.15 -0.10" size="0.006" rgba="0.12 0.08 0.04 1" contype="0" conaffinity="0"/>
  <geom name="fly_leg_fr" type="capsule" fromto="0.04 -0.05 -0.02 0.12 -0.15 -0.10" size="0.006" rgba="0.12 0.08 0.04 1" contype="0" conaffinity="0"/>
  <geom name="fly_leg_bl" type="capsule" fromto="-0.06 0.05 -0.02 -0.14 0.16 -0.10" size="0.006" rgba="0.12 0.08 0.04 1" contype="0" conaffinity="0"/>
  <geom name="fly_leg_br" type="capsule" fromto="-0.06 -0.05 -0.02 -0.14 -0.16 -0.10" size="0.006" rgba="0.12 0.08 0.04 1" contype="0" conaffinity="0"/>
  <site name="rotor_FL" pos="{ROTOR_OFFSETS['FL'][0]} {ROTOR_OFFSETS['FL'][1]} 0.01" size="0.02"/>
  <site name="rotor_FR" pos="{ROTOR_OFFSETS['FR'][0]} {ROTOR_OFFSETS['FR'][1]} 0.01" size="0.02"/>
  <site name="rotor_BL" pos="{ROTOR_OFFSETS['BL'][0]} {ROTOR_OFFSETS['BL'][1]} 0.01" size="0.02"/>
  <site name="rotor_BR" pos="{ROTOR_OFFSETS['BR'][0]} {ROTOR_OFFSETS['BR'][1]} 0.01" size="0.02"/>
  <!-- MuJoCo cameras look along local -Z.  These local X/Y axes make -Z
       coincide with the drone body's +X (nose-forward), with +Y upward. -->
  <camera name="drone_eye" pos="0.17 0 0.035" xyaxes="0 -1 0 0 0 1" fovy="90"/>
</body>
"""

DRONE_ACTUATORS_XML = "\n".join(
    f'<motor name="thrust_{name}" site="rotor_{name}" gear="0 0 1 0 0 {sign * YAW_TORQUE_COEFF}" ctrlrange="0 20"/>'
    for name, (dx, dy, sign) in ROTOR_OFFSETS.items()
)


@dataclass
class DroneState:
    position: np.ndarray
    quaternion: np.ndarray  # (w, x, y, z)
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


class Drone:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.rotor_actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrust_{name}")
            for name in ROTOR_OFFSETS
        ]
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")

    def apply_rotor_thrusts(self, thrusts: np.ndarray):
        """thrusts: array of 4 values [FL, FR, BL, BR] in Newtons."""
        for actuator_id, thrust in zip(self.rotor_actuator_ids, thrusts):
            self.data.ctrl[actuator_id] = max(0.0, float(thrust))

    def apply_forward_force(self, force_n: float, quaternion: np.ndarray):
        """Applies `force_n` Newtons along the drone's current local +X
        (forward) axis, as a direct external force via MuJoCo's
        xfrc_applied -- NOT via tilting the airframe.

        A real quadrotor can only move by tilting its thrust vector, and
        our first attempt at that (see src/drone/physics.py's git history /
        StabilizedMixer) needed a full PD attitude-stabilization loop to
        avoid an unbounded-pitch instability -- which then still diverged
        into a runaway climb after a wall collision during Phase 6 testing,
        for reasons we didn't fully track down. Tuning a robust flight
        controller is its own substantial control-theory project, and not
        what this project is actually about (the brief explicitly
        deprioritizes complex control/RL machinery in favor of showing the
        neural steering signal produces useful behavior) -- so we apply
        forward locomotion as a direct external force instead, keeping the
        airframe level. This is a further, explicit simplification of the
        drone's physics (on top of "it's a quadrotor, not a fly body" --
        see docs/scientific_assumptions.md), not a hidden one.
        """
        rot = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        forward = rot.apply([1.0, 0.0, 0.0])
        self.data.xfrc_applied[self.body_id, :3] = force_n * forward

    def apply_planar_velocity_control(
        self,
        forward_force_n: float,
        quaternion: np.ndarray,
        linear_velocity: np.ndarray,
        drag_gain: float = 1.5,
        max_force_n: float = 2.0,
    ):
        """Forward propulsion with horizontal velocity feedback.

        A constant external force accelerates forever in free space, which
        made the navigation benchmark test inertia rather than steering.
        This retains the documented direct-force abstraction but adds a
        viscous horizontal drag term, producing a finite terminal speed.
        """
        rot = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        forward = rot.apply([1.0, 0.0, 0.0])
        force = forward_force_n * forward
        force[:2] -= drag_gain * np.asarray(linear_velocity)[:2]
        norm = np.linalg.norm(force[:2])
        if norm > max_force_n:
            force[:2] *= max_force_n / norm
        force[2] = 0.0
        self.data.xfrc_applied[self.body_id, :3] = force

    def apply_yaw_rate_control(
        self,
        yaw_command: float,
        angular_velocity: np.ndarray,
        max_yaw_rate: float = 1.5,
        rate_gain: float = 0.08,
        level_gain: float = 0.1,
    ):
        """Treat ``yaw_command`` as a desired yaw rate and close the loop.

        Neural output still selects direction and magnitude; this method is
        the low-level body controller, analogous to the rate loop present on
        a real quadrotor.  It also damps roll/pitch rates so a collision does
        not turn into an uncontrolled tumble.
        """
        omega = np.asarray(angular_velocity)
        desired_yaw_rate = float(np.clip(yaw_command, -1.0, 1.0)) * max_yaw_rate
        torque = np.array([
            -level_gain * omega[0],
            -level_gain * omega[1],
            rate_gain * (desired_yaw_rate - omega[2]),
        ])
        self.data.xfrc_applied[self.body_id, 3:6] = torque

    def apply_attitude_damping(self, angular_velocity: np.ndarray, gain: float = 0.1):
        """Applies a torque opposing current angular velocity -- pure
        rate damping, no target-angle feedback. This is a deliberately
        minimal stability aid: since it can only ever remove rotational
        energy (never add it), it cannot itself cause the runaway
        instability we hit when trying full PD attitude control (see
        apply_forward_force's docstring). Without any damping, an
        unavoidable real-world case -- a wall collision imparting some
        spin -- had nothing to stop it from tumbling, per Phase 6 testing.
        Purely dissipative, so a "stronger" gain only settles faster, never
        diverges, but it also does not actively return the drone to level
        (no P term) -- it just stops it from spinning up further."""
        self.data.xfrc_applied[self.body_id, 3:6] = -gain * np.asarray(angular_velocity)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def reset_if_unstable(self, safe_position=(0.0, 0.0, 1.0), max_height: float = 8.0, max_speed: float = 15.0) -> bool:
        """Safety net, not a physics fix: if the simulation has diverged
        (seen in Phase 6 testing -- a wall-collision impulse occasionally
        triggered a numerically unstable contact response that launched the
        drone hundreds of meters into the air before MuJoCo's own solver
        forced it back to a plausible state on its own a few steps later).
        We didn't fully root-cause this and are not pretending to have
        fixed the underlying instability -- this just catches the
        divergence early and resets to a safe hover before it's visually
        disruptive, documented plainly as a safety net rather than hidden.
        Returns True if a reset was applied."""
        pos = self.data.xpos[self.body_id]
        speed = np.linalg.norm(self.data.cvel[self.body_id][3:])
        if pos[2] > max_height or speed > max_speed or not np.all(np.isfinite(pos)):
            self.data.qpos[0:3] = safe_position
            # Preserve the model's configured starting yaw.  Resetting every
            # trial to identity silently changed randomized benchmark trials
            # after an instability and made controller comparisons unfair.
            self.data.qpos[3:7] = self.model.qpos0[3:7]
            self.data.qvel[:] = 0.0
            self.data.xfrc_applied[self.body_id] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return True
        return False

    def get_state(self) -> DroneState:
        qpos = self.data.xpos[self.body_id].copy()
        quat = self.data.xquat[self.body_id].copy()
        linvel = self.data.cvel[self.body_id][3:].copy()
        angvel = self.data.cvel[self.body_id][:3].copy()
        return DroneState(position=qpos, quaternion=quat, linear_velocity=linvel, angular_velocity=angvel)

    def obstacle_rays(self, angles_deg=(-60.0, -30.0, 0.0, 30.0, 60.0)) -> tuple[np.ndarray, np.ndarray]:
        """Cast horizontal rays against physical scenery, excluding the fly.

        Geometry group 1 is reserved for the apple, so pursuit does not treat
        the goal itself as an obstacle. Distances are visual/looming proxies;
        MuJoCo contacts remain the final physical barrier.
        """
        state = self.get_state()
        rotation = Rotation.from_quat([
            state.quaternion[1], state.quaternion[2],
            state.quaternion[3], state.quaternion[0],
        ])
        origin = state.position.copy()
        geom_group = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        angles = np.radians(np.asarray(angles_deg, dtype=np.float64))
        distances = np.empty_like(angles)
        for index, angle in enumerate(angles):
            local = np.asarray([np.cos(angle), np.sin(angle), 0.0])
            direction = np.ascontiguousarray(rotation.apply(local), dtype=np.float64)
            direction[2] = 0.0
            direction /= np.linalg.norm(direction)
            distances[index] = mujoco.mj_ray(
                self.model, self.data, origin, direction, geom_group, True,
                self.body_id, None,
            )
        return angles, distances
