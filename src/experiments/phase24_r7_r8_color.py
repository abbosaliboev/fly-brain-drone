"""Calibrate the R7/R8 colour-opponent salience readout."""
from __future__ import annotations

import numpy as np
import mujoco

from src.experiments.phase20_neural_salience import spot_frame
from src.environment.world import build_model
from src.vision.encoder import detect_food_in_frame
from src.vision.flyvis_backend import FlyVisBackend


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    cases = (("uniform", None), ("left", -0.65), ("centre", 0.0), ("right", 0.65))
    for label, position in cases:
        backend.reset_stream()
        for _ in range(4):
            frame = (np.full((120, 120, 3), 90, dtype=np.uint8)
                     if position is None else spot_frame(position))
            backend.infer_color_frame(frame)
        error, confidence = backend.r7_r8_horizontal_salience()
        print(f"case={label:7s} error={error:+.4f} confidence={confidence:.4f}")

    print("arena yaw sweep:")
    model = build_model(drone_start_pos=(0.0, 3.0, 1.0))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=120, width=160)
    backend.reset_stream()
    for yaw in (np.arange(0.0, 360.0, 15.0) + 180.0) % 360.0:
        half = np.radians(yaw) / 2
        data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        backend.infer_color_frame(frame)
        error, confidence = backend.r7_r8_horizontal_salience()
        pixel_error, pixel_confidence = detect_food_in_frame(frame)
        print(
            f"yaw={yaw:5.0f} neural=({error:+.3f},{confidence:.3f}) "
            f"food=({pixel_error:+.3f},{pixel_confidence:.3f})"
        )
    renderer.close()


if __name__ == "__main__":
    main()
