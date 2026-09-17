"""Smoke-test the official pretrained FlyVis model on opposite bar motion."""
from __future__ import annotations

import numpy as np

from src.vision.flyvis_backend import FlyVisBackend


def moving_bar(left_to_right: bool, frames: int = 16, size: int = 405) -> np.ndarray:
    movie = np.full((frames, size, size), 0.5, dtype=np.float32)
    positions = np.linspace(size // 4, 3 * size // 4, frames).astype(int)
    if not left_to_right:
        positions = positions[::-1]
    width = max(3, size // 40)
    for frame, x in zip(movie, positions):
        frame[:, max(0, x - width): min(size, x + width)] = 1.0
    return movie


def main() -> None:
    # Direction-selectivity validation uses FlyVis's original retinal extent.
    # The reduced extent=5 online controller is a separate memory tradeoff and
    # is too coarse for this synthetic full-field motion regression.
    backend = FlyVisBackend(retinal_extent=15)
    rightward = backend.infer(moving_bar(True))
    leftward = backend.infer(moving_bar(False))
    print(f"rightward horizontal={rightward.horizontal:+.6f}")
    print(f"leftward  horizontal={leftward.horizontal:+.6f}")
    reversal = rightward.horizontal * leftward.horizontal < 0
    print(f"opponent sign reversal={reversal}")
    if not reversal:
        raise SystemExit("FlyVis motion smoke test failed: no opponent sign reversal")


if __name__ == "__main__":
    main()
