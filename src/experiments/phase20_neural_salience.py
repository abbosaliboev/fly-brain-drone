"""Calibrate a food-location readout using FlyVis R1-R6 activity only."""
from __future__ import annotations

import numpy as np

from src.vision.flyvis_backend import FlyVisBackend


def spot_frame(position: float, size: int = 120) -> np.ndarray:
    frame = np.full((size, size, 3), 90, dtype=np.uint8)
    x = int((position + 1.0) * 0.5 * (size - 1))
    radius = 12
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - x) ** 2 + (yy - size // 2) ** 2 <= radius ** 2
    frame[mask] = (255, 90, 2)
    return frame


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    cases = [("uniform", None), ("left", -0.65), ("centre", 0.0), ("right", 0.65)]
    for label, position in cases:
        backend.reset_stream()
        # A few repeated frames let photoreceptor dynamics settle while the
        # recurrent state remains continuous.
        for _ in range(4):
            frame = (np.full((120, 120, 3), 90, dtype=np.uint8)
                     if position is None else spot_frame(position))
            backend.infer_frame(frame)
        error, confidence = backend.retinal_horizontal_salience()
        print(
            f"case={label:7s} neural_error={error:+.4f} "
            f"confidence={confidence:.4f}"
        )


if __name__ == "__main__":
    main()
