"""Hybrid closed loop: camera food salience plus recurrent FlyVis motion.

Food selection remains the explicit colour-pixel engineering readout from
Phase 16. FlyVis T4/T5 activity contributes an opponent motion-damping term
to yaw. Thus this experiment demonstrates real online model participation,
without falsely claiming that FlyVis itself recognizes food.
"""
from __future__ import annotations

import numpy as np
import mujoco

from src.control.motor_decoder import MotorCommand
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.environment.world import FOOD_POSITION, build_model
from src.vision.encoder import detect_food_in_frame
from src.vision.flyvis_backend import FlyVisBackend

START_POS = (0.0, 3.0, 1.0)
START_YAW_DEG = 180.0
MAX_STEPS = 5000
FLYVIS_INTERVAL = 100  # CPU model is supervisory, physics remains 200 Hz
FOOD_POS = np.asarray(FOOD_POSITION, dtype=np.float32)


def run_trial(backend: FlyVisBackend | None, start_yaw: float = START_YAW_DEG,
              motion_gain: float = 1.5, verbose: bool = True,
              start_pos=START_POS, food_pos=FOOD_POS) -> dict:
    food_pos = np.asarray(food_pos, dtype=np.float32)
    model = build_model(start_pos, start_yaw, food_position=food_pos)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    mujoco.mj_forward(model, data)
    if backend is not None:
        # Allocate the small OpenGL framebuffer before the neural steady
        # state claims most of this machine's constrained virtual memory.
        backend.reset_stream()
    flyvis_horizontal = 0.0
    flyvis_updates = 0

    for step in range(MAX_STEPS):
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        error, confidence = detect_food_in_frame(frame)
        if backend is not None and step % FLYVIS_INTERVAL == 0:
            motion = backend.infer_frame(frame)
            flyvis_horizontal = motion.horizontal
            flyvis_updates += 1

        target_yaw = 0.65 if confidence == 0.0 else 2.2 * error
        # A left body rotation makes the retinal scene move right (positive
        # FlyVis opponent signal), hence subtraction supplies visual damping.
        yaw = target_yaw - motion_gain * float(np.clip(flyvis_horizontal, -0.3, 0.3))
        forward = float(confidence > 0.0 and abs(error) < 0.18)

        state = drone.get_state()
        drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
        drone.apply_planar_velocity_control(1.5 * forward, state.quaternion, state.linear_velocity)
        drone.apply_yaw_rate_control(yaw, state.angular_velocity)
        drone.step()
        drone.reset_if_unstable(safe_position=start_pos, max_height=3.0)

        distance = float(np.linalg.norm(drone.get_state().position - food_pos))
        if distance < 0.6:
            result = dict(success=True, steps=step + 1, distance=distance,
                          flyvis_updates=flyvis_updates,
                          last_motion=flyvis_horizontal, start_yaw=start_yaw)
            if verbose:
                print(
                    f"FOOD REACHED steps={step + 1} distance={distance:.3f} "
                    f"flyvis_updates={flyvis_updates} last_motion={flyvis_horizontal:+.5f}"
                )
            renderer.close()
            return result
        if verbose and step % 500 == 0:
            print(
                f"step={step} distance={distance:.2f} pixel_error={error:+.2f} "
                f"flyvis_motion={flyvis_horizontal:+.5f}"
            )
    renderer.close()
    return dict(success=False, steps=MAX_STEPS, distance=distance,
                flyvis_updates=flyvis_updates, last_motion=flyvis_horizontal,
                start_yaw=start_yaw)


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    result = run_trial(backend)
    if not result["success"]:
        raise SystemExit("Food was not reached within MAX_STEPS")


if __name__ == "__main__":
    main()
