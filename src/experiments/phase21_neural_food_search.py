"""Food search using only FlyVis neural salience and motion readouts.

Unlike Phases 16-19, this deliberately does not call the RGB orange-pixel
detector. It tests whether an untrained contrast-centroid readout of R1-R6 is
specific enough to select food in the rendered arena.
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

START = (0.0, 3.0, 1.0)
FOOD = np.asarray(FOOD_POSITION, dtype=np.float32)
MAX_STEPS = 5000
NEURAL_INTERVAL = 20
SALIENCE_THRESHOLD = 0.27


def main() -> None:
    model = build_model(START, 180.0)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    mujoco.mj_forward(model, data)
    backend = FlyVisBackend(retinal_extent=5)
    backend.reset_stream()

    error = confidence = motion_x = 0.0
    diagnostic_pixel_error = diagnostic_pixel_confidence = 0.0
    min_distance = float(np.linalg.norm(np.asarray(START) - FOOD))
    for step in range(MAX_STEPS):
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        if step % NEURAL_INTERVAL == 0:
            motion = backend.infer_frame(frame)
            motion_x = motion.horizontal
            error, confidence = backend.retinal_horizontal_salience()
            # Evaluator-only label: logged to measure neural alignment, never
            # used in visible/target_yaw/forward or any control decision.
            diagnostic_pixel_error, diagnostic_pixel_confidence = detect_food_in_frame(frame)

        visible = confidence >= SALIENCE_THRESHOLD
        target_yaw = 2.2 * error if visible else 0.65
        yaw = target_yaw - 1.5 * float(np.clip(motion_x, -0.3, 0.3))
        forward = float(visible and abs(error) < 0.04)
        state = drone.get_state()
        drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
        drone.apply_planar_velocity_control(1.5 * forward, state.quaternion, state.linear_velocity)
        drone.apply_yaw_rate_control(yaw, state.angular_velocity)
        drone.step()
        drone.reset_if_unstable(safe_position=START, max_height=3.0)

        distance = float(np.linalg.norm(drone.get_state().position - FOOD))
        min_distance = min(min_distance, distance)
        if distance < 0.6:
            print(f"FOOD REACHED steps={step + 1} distance={distance:.3f}")
            return
        if step % 500 == 0:
            print(
                f"step={step} distance={distance:.2f} min={min_distance:.2f} "
                f"neural_error={error:+.3f} confidence={confidence:.3f} "
                f"food_visible={diagnostic_pixel_confidence > 0} "
                f"food_error={diagnostic_pixel_error:+.3f}"
            )
    print(f"FAILED final_distance={distance:.3f} min_distance={min_distance:.3f}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
