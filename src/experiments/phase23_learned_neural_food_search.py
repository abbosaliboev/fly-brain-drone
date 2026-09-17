"""Closed-loop food search with no RGB detector at runtime.

A ridge readout is calibrated on two separate food locations using labelled
FlyVis activity. The evaluation trial uses an unseen location; its controller
receives only neural activity. World coordinates are used only for success
scoring.
"""
from __future__ import annotations

import numpy as np
import mujoco
from pathlib import Path

from src.control.motor_decoder import MotorCommand
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.environment.world import FOOD_POSITION, build_model
from src.experiments.phase22_learned_neural_readout import predict
from src.vision.flyvis_backend import FlyVisBackend

START = (0.0, 3.0, 1.0)
FOOD = np.asarray(FOOD_POSITION, dtype=np.float32)
MAX_STEPS = 5000
NEURAL_INTERVAL = 20
READOUT_PATH = Path("data/cache/flyvis_food_readout.npz")


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    if not READOUT_PATH.exists():
        raise RuntimeError(
            f"Missing {READOUT_PATH}; run phase22_learned_neural_readout first"
        )
    saved = np.load(READOUT_PATH)
    readout = saved["mean"], saved["scale"], saved["weights"]

    model = build_model(START, 180.0, food_position=FOOD)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    mujoco.mj_forward(model, data)
    backend.reset_stream()

    predicted_error = predicted_visibility = motion_x = 0.0
    min_distance = float(np.linalg.norm(np.asarray(START) - FOOD))
    neural_updates = 0
    for step in range(MAX_STEPS):
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        if step % NEURAL_INTERVAL == 0:
            motion = backend.infer_frame(frame)
            motion_x = motion.horizontal
            estimate = predict(readout, backend.activity_features()[None])[0]
            predicted_error = float(np.clip(estimate[0], -1.0, 1.0))
            predicted_visibility = float(np.clip(estimate[1], 0.0, 1.0))
            neural_updates += 1

        visible = predicted_visibility > 0.5
        target_yaw = 2.2 * predicted_error if visible else 0.65
        yaw = target_yaw - 1.5 * float(np.clip(motion_x, -0.3, 0.3))
        forward = float(visible and abs(predicted_error) < 0.20)
        state = drone.get_state()
        drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
        drone.apply_planar_velocity_control(1.5 * forward, state.quaternion, state.linear_velocity)
        drone.apply_yaw_rate_control(yaw, state.angular_velocity)
        drone.step()
        drone.reset_if_unstable(safe_position=START, max_height=3.0)

        distance = float(np.linalg.norm(drone.get_state().position - FOOD))
        min_distance = min(min_distance, distance)
        if distance < 0.6:
            print(
                f"FOOD REACHED steps={step + 1} distance={distance:.3f} "
                f"neural_updates={neural_updates}"
            )
            return
        if step % 500 == 0:
            print(
                f"step={step} distance={distance:.2f} min={min_distance:.2f} "
                f"pred_error={predicted_error:+.3f} pred_visible={predicted_visibility:.3f}"
            )
    print(f"FAILED final_distance={distance:.3f} min_distance={min_distance:.3f}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
