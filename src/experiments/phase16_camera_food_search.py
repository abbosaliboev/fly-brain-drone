"""Headless proof that the drone can find food from POV camera pixels only.

This is a perception/body baseline, not a biological controller. It never
reads the food's world coordinates for steering; coordinates are used only by
the evaluator to decide whether the trial succeeded.
"""
from __future__ import annotations

import numpy as np
import mujoco

from src.control.motor_decoder import MotorCommand
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.environment.world import FOOD_POSITION, build_model
from src.vision.encoder import detect_food_in_frame

START_POS = (0.0, 3.0, 1.0)
START_YAW_DEG = 180.0
MAX_STEPS = 5000
REACHED_RADIUS_M = 0.6
FOOD_POS = np.asarray(FOOD_POSITION, dtype=np.float32)


def main() -> None:
    model = build_model(START_POS, START_YAW_DEG)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    mujoco.mj_forward(model, data)

    seen_frames = 0
    for step in range(MAX_STEPS):
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        error, confidence = detect_food_in_frame(frame)
        if confidence == 0.0:
            yaw, forward = 0.65, 0.0
        else:
            seen_frames += 1
            yaw = 2.2 * error
            forward = 1.0 if abs(error) < 0.18 else 0.0

        state = drone.get_state()
        drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
        drone.apply_planar_velocity_control(1.5 * forward, state.quaternion, state.linear_velocity)
        drone.apply_yaw_rate_control(yaw, state.angular_velocity)
        drone.step()
        drone.reset_if_unstable(safe_position=START_POS, max_height=3.0)

        distance = float(np.linalg.norm(drone.get_state().position - FOOD_POS))
        if distance < REACHED_RADIUS_M:
            print(f"FOOD REACHED steps={step + 1} seen_frames={seen_frames} distance={distance:.3f}")
            return
        if step % 500 == 0:
            print(f"step={step} distance={distance:.2f} pixel_error={error:+.2f} confidence={confidence:.2f}")
    raise SystemExit("Food was not reached within MAX_STEPS")


if __name__ == "__main__":
    main()
