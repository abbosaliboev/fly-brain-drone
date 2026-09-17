"""Biologically inspired high-level flight modes for the live dashboard.

This is an explicit engineering controller, not a claim of complete fruit-fly
flight biomechanics.  It captures three observable motifs: forward casting
during odour/target search, smooth pursuit while a target is visible, and a
rapid saccadic turn away from looming obstacles.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FlightIntent:
    yaw: float
    forward_force: float
    mode: str
    nearest_obstacle: float


class FlySearchBehavior:
    """Stateful behavior with escape-direction hysteresis.

    A narrow pillar can alternate between adjacent rays as the body turns. A
    stateless controller then flips left/right every step. This controller
    commits to the first safe side until both a minimum saccade time and a
    wider clearance threshold have been satisfied.
    """

    def __init__(self, minimum_escape_steps: int = 90, release_distance: float = 1.35):
        self.minimum_escape_steps = minimum_escape_steps
        self.release_distance = release_distance
        self.escape_direction = 0.0
        self.escape_until_step = -1

    def reset(self) -> None:
        self.escape_direction = 0.0
        self.escape_until_step = -1

    def update(
        self, step: int, confidence: float, target_error: float,
        optic_motion: float, ray_angles: np.ndarray, ray_distances: np.ndarray,
    ) -> FlightIntent:
        base = fly_search_intent(
            step, confidence, target_error, optic_motion, ray_angles, ray_distances
        )
        nearest = base.nearest_obstacle
        if self.escape_direction:
            must_hold = step < self.escape_until_step
            if must_hold or nearest < self.release_distance:
                # Continuous braking removes the old +0.30/-0.65 speed jump.
                forward = float(np.clip((nearest - 0.42) * 0.72, -0.45, 0.34))
                return FlightIntent(
                    self.escape_direction, forward, "LOOMING ESCAPE", nearest
                )
            self.escape_direction = 0.0

        if base.mode == "LOOMING ESCAPE":
            self.escape_direction = 1.0 if base.yaw >= 0.0 else -1.0
            self.escape_until_step = step + self.minimum_escape_steps
            forward = float(np.clip((nearest - 0.42) * 0.72, -0.45, 0.34))
            return FlightIntent(
                self.escape_direction, forward, "LOOMING ESCAPE", nearest
            )
        return base


def fly_search_intent(
    step: int,
    confidence: float,
    target_error: float,
    optic_motion: float,
    ray_angles: np.ndarray,
    ray_distances: np.ndarray,
) -> FlightIntent:
    """Select casting, pursuit, or looming-escape flight."""
    distances = np.where(np.asarray(ray_distances) < 0.0, np.inf, ray_distances)
    angles = np.asarray(ray_angles, dtype=np.float64)
    nearest = float(np.min(distances)) if distances.size else float("inf")

    # Looming escape wins over food pursuit. Positive yaw turns left, so turn
    # toward the side whose rays report the larger free-space distance.
    if nearest < 1.05:
        left = float(np.mean(np.clip(distances[angles > 0.0], 0.0, 3.0)))
        right = float(np.mean(np.clip(distances[angles < 0.0], 0.0, 3.0)))
        if abs(left - right) < 0.05:
            turn = 1.0 if math.sin(step * 0.071) >= 0.0 else -1.0
        else:
            turn = 1.0 if left > right else -1.0
        # Brake hard at the surface, but retain a little forward motion while
        # there is room so the maneuver draws an arc instead of spinning.
        forward = float(np.clip((nearest - 0.42) * 0.72, -0.45, 0.34))
        return FlightIntent(turn, forward, "LOOMING ESCAPE", nearest)

    if confidence > 0.02:
        yaw = 2.0 * float(target_error) - 1.35 * float(np.clip(optic_motion, -0.3, 0.3))
        alignment = max(0.15, 1.0 - abs(float(target_error)))
        return FlightIntent(float(np.clip(yaw, -1.0, 1.0)), 1.45 * alignment, "APPLE PURSUIT", nearest)

    # Fruit flies search with moving, curved casts rather than a stationary
    # turn. Two incommensurate waves prevent an endlessly repeated circle.
    t = step * 0.005
    yaw = 0.68 * math.sin(2.1 * t) + 0.28 * math.sin(0.73 * t + 1.2)
    return FlightIntent(float(yaw), 0.62, "CASTING SEARCH", nearest)
