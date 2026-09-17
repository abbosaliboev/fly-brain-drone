"""Closed-loop food search driven by R7/R8 neural colour opponency."""
from __future__ import annotations

from contextlib import nullcontext
import time
import numpy as np
import mujoco

from src.control.motor_decoder import MotorCommand
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.environment.world import FOOD_POSITION, build_model
from src.vision.flyvis_backend import FlyVisBackend

START = (0.0, 3.0, 1.0)
FOOD = np.asarray(FOOD_POSITION, dtype=np.float32)
MAX_STEPS = 5000
NEURAL_INTERVAL = 20


def run_trial(start=START, food=FOOD, start_yaw: float = 180.0,
              verbose: bool = True, distractors=(),
              backend: FlyVisBackend | None = None,
              food_rgba=(1.0, 0.35, 0.02, 1.0),
              light_level: float = 1.0,
              show_viewer: bool = False,
              viewer_step_delay: float = 0.0,
              hold_viewer_on_success: bool = False) -> dict:
    food = np.asarray(food, dtype=np.float32)
    model = build_model(
        start, start_yaw, food_position=food, food_rgba=food_rgba,
        achromatic_background=True,
        distractors=distractors, light_level=light_level,
    )
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    mujoco.mj_forward(model, data)
    if backend is None:
        backend = FlyVisBackend(retinal_extent=5)
    backend.reset_stream()

    error = confidence = motion_x = 0.0
    min_distance = float(np.linalg.norm(np.asarray(start) - food))
    distance = min_distance
    distractor_positions = [np.asarray(item[0], dtype=np.float32) for item in distractors]
    min_distractor_distance = float("inf")
    updates = 0
    if show_viewer:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
    else:
        viewer_context = nullcontext(None)

    stopped_by_user = False
    try:
        with viewer_context as live_viewer:
            for step in range(MAX_STEPS):
                if live_viewer is not None and not live_viewer.is_running():
                    stopped_by_user = True
                    break
                renderer.update_scene(data, camera="drone_eye")
                frame = renderer.render()
                if step % NEURAL_INTERVAL == 0:
                    motion = backend.infer_color_frame(frame)
                    motion_x = motion.horizontal
                    error, confidence = backend.r1_r7_r8_orange_salience()
                    updates += 1

                visible = confidence > 0.02
                target_yaw = 2.2 * error if visible else 0.65
                yaw = target_yaw - 1.5 * float(np.clip(motion_x, -0.3, 0.3))
                # extent=5 yields coarse horizontal bins (often 0.2-0.4 apart), so
                # requiring the fine pixel-controller tolerance would prevent motion.
                forward = float(visible and abs(error) <= 0.70)
                state = drone.get_state()
                drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
                drone.apply_planar_velocity_control(
                    1.5 * forward, state.quaternion, state.linear_velocity
                )
                drone.apply_yaw_rate_control(yaw, state.angular_velocity)
                drone.step()
                drone.reset_if_unstable(safe_position=start, max_height=3.0)
                if live_viewer is not None:
                    live_viewer.sync()
                    if viewer_step_delay > 0.0:
                        time.sleep(viewer_step_delay)

                distance = float(np.linalg.norm(drone.get_state().position - food))
                min_distance = min(min_distance, distance)
                if distractor_positions:
                    nearest = min(float(np.linalg.norm(drone.get_state().position - p))
                                  for p in distractor_positions)
                    min_distractor_distance = min(min_distractor_distance, nearest)
                if distance < 0.6:
                    result = dict(success=True, steps=step + 1, distance=distance,
                                  min_distance=min_distance, neural_updates=updates,
                                  stopped_by_user=False)
                    result["min_distractor_distance"] = min_distractor_distance
                    if verbose:
                        print(
                            f"FOOD REACHED steps={step + 1} distance={distance:.3f} "
                            f"neural_updates={updates}"
                        )
                    if live_viewer is not None and hold_viewer_on_success:
                        print("Food reached. Close the MuJoCo window to finish the demo.")
                        while live_viewer.is_running():
                            live_viewer.sync()
                            time.sleep(0.05)
                    return result
                if verbose and step % 500 == 0:
                    print(
                        f"step={step} distance={distance:.2f} min={min_distance:.2f} "
                        f"r7r8_error={error:+.3f} confidence={confidence:.3f}"
                    )
    finally:
        renderer.close()
    return dict(success=False, steps=MAX_STEPS, distance=distance,
                min_distance=min_distance, neural_updates=updates,
                min_distractor_distance=min_distractor_distance,
                stopped_by_user=stopped_by_user)


def main() -> None:
    result = run_trial()
    if not result["success"]:
        print(f"FAILED final_distance={result['distance']:.3f} min_distance={result['min_distance']:.3f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
