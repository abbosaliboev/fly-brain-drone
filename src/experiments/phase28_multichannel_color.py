"""Synthetic calibration of the R1-R6/R7/R8 orange opponent."""
from __future__ import annotations

import numpy as np

from src.vision.flyvis_backend import FlyVisBackend


def coloured_spot(rgb, position: float = -0.6, size: int = 120) -> np.ndarray:
    frame = np.full((size, size, 3), 90, dtype=np.uint8)
    x = int((position + 1.0) * 0.5 * (size - 1))
    yy, xx = np.ogrid[:size, :size]
    frame[(xx - x) ** 2 + (yy - size // 2) ** 2 <= 12 ** 2] = rgb
    return frame


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    cases = (
        ("uniform", None),
        ("orange", (255, 90, 2)),
        ("green", (12, 255, 12)),
        ("blue", (12, 25, 255)),
    )
    for name, colour in cases:
        backend.reset_stream()
        frame = (np.full((120, 120, 3), 90, dtype=np.uint8)
                 if colour is None else coloured_spot(colour))
        for _ in range(4):
            backend.infer_color_frame(frame)
        error, confidence = backend.r1_r7_r8_orange_salience()
        print(f"{name:7s} error={error:+.4f} confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
