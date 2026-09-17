"""Presentation-only round bookkeeping; never supplies motor commands."""
from collections import deque
from time import perf_counter
import numpy as np


class AppleHunt:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.history = deque(maxlen=10)
        self.best = {}
        self.round = 0
        self.new_round()

    def new_round(self, mode="HYBRID", difficulty="NORMAL"):
        self.round += 1
        self.mode, self.difficulty = mode, difficulty
        self.accumulated = self.sim_time = self.path = 0.0
        self.started = None
        self.finished = False
        self.escapes = self.collisions = self.confirmations = 0
        self.previous_escape = self.previous_contact = False

    @property
    def elapsed(self):
        return self.accumulated + (self.clock() - self.started if self.started is not None else 0)

    def start(self):
        if not self.finished and self.started is None:
            self.started = self.clock()

    def pause(self):
        self.accumulated = self.elapsed
        self.started = None
        self.confirmations = 0

    @property
    def score(self):
        return max(0, round(max(0, 3000 - self.sim_time * 100)
                           + max(0, 1000 - self.path * 40)
                           - self.collisions * 250 - self.escapes * 40))

    def physics(self, dt, displacement, escape, contact):
        self.sim_time += dt
        self.path += float(displacement)
        self.escapes += int(escape and not self.previous_escape)
        self.collisions += int(contact and not self.previous_contact)
        self.previous_escape, self.previous_contact = escape, contact

    def observe(self, distance, confidence, target_visible):
        # Camera is 0.17 m ahead of the body; inside ~0.57 m it enters
        # the apple mesh. Confirm arrival before that rendering singularity.
        radius = {"EASY": .90, "NORMAL": .80, "HARD": .72}[self.difficulty]
        valid = distance < radius and confidence > .02 and target_visible
        self.confirmations = self.confirmations + 1 if valid else 0
        if self.finished or self.started is None or self.confirmations < 3:
            return None
        self.pause()
        self.finished = True
        result = (self.round, self.mode, self.difficulty, self.sim_time,
                  self.path, self.escapes, self.collisions, self.score)
        self.history.append(result)
        key = (self.mode, self.difficulty)
        self.best[key] = min(self.sim_time, self.best.get(key, float("inf")))
        return result


def safe_apple_position(model, position):
    """Conservative apple + approach clearance around static collision geoms."""
    import mujoco
    p = np.asarray(position)
    if not np.all(np.isfinite(p)) or np.any(np.abs(p[:2]) > 3.1):
        return False
    for i in range(model.ngeom):
        if model.geom_bodyid[i] != 0 or not model.geom_contype[i]:
            continue
        kind = model.geom_type[i]
        center, size = model.geom_pos[i], model.geom_size[i]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            if np.all(np.abs(p - center) < size + np.array([.65, .65, .55])):
                return False
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            if abs(p[2] - center[2]) < size[1] + .55 and np.linalg.norm(p[:2] - center[:2]) < size[0] + .65:
                return False
    return True
