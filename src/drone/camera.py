"""VisPy-based 3rd-person display of the drone + world.

We deliberately do NOT use MuJoCo's own Renderer here even though MuJoCo
computes the physics: in this dashboard process, MuJoCo's Renderer (WGL/
GLFW-backed on Windows) and VisPy's Qt-embedded OpenGL context conflict.
Confirmed by scratch testing several approaches (direct pixel diffing,
FlyCamera.rotation/rotation1, TurntableCamera.azimuth/elevation set per
frame, moving the world instead of the camera) -- in every case, changing
the drone's *orientation* produced a byte-identical render once VisPy's
canvas was active in the same process, while *position* changes worked
fine. We did not find a fix within reasonable effort, so for the unified
dashboard (src/visualization/dashboard.py) we only need this module for
DISPLAY (an external, user-controlled orbit view -- it doesn't need to
track the drone's exact orientation) and use a simpler two-sensor model
(src/vision/encoder.py's encode_left_right_from_bearing) instead of a
rendered POV camera for the actual neural input. The real rendered-camera
closed loop is demonstrated working correctly in
src/experiments/phase4_closed_loop.py, using MuJoCo's Renderer standalone
(no Qt/VisPy in that process) -- that remains the reference implementation
for "a camera image drives the connectome," not silently dropped.

The world drawn here (ground plane + one fixed colored landmark sphere) is
a placeholder test scene for validating the vision pipeline -- it is NOT
the food-search environment (that's Phase 6-7).
"""
from __future__ import annotations

import numpy as np
from vispy import scene
from vispy.scene import visuals
from scipy.spatial.transform import Rotation
from src.environment.world import FOOD_POSITION

LANDMARK_POS = np.array(FOOD_POSITION, dtype=np.float32)
GROUND_SIZE = 20.0
ARENA_HALF_SIZE = 4.0  # must match src/environment/world.py's ARENA_HALF_SIZE


def _build_world_scene(view: scene.ViewBox):
    ground = visuals.Plane(width=GROUND_SIZE, height=GROUND_SIZE, direction="+z", color=(0.25, 0.3, 0.28, 1))
    view.add(ground)
    landmark = visuals.Sphere(radius=0.4, color=(1.0, 0.5, 0.1, 1))
    landmark.transform = scene.transforms.STTransform(translate=LANDMARK_POS)
    view.add(landmark)

    s = ARENA_HALF_SIZE
    corners = np.array([[s, s, 0], [-s, s, 0], [-s, -s, 0], [s, -s, 0], [s, s, 0]], dtype=np.float32)
    walls = visuals.Line(pos=corners, color=(0.5, 0.4, 0.35, 1), width=3)
    view.add(walls)


class DroneVisual:
    """A simple box+4-colored-rotor drone representation, drawn in a given
    VisPy view and repositioned/reoriented each frame from MuJoCo state.
    (Object-transform updates like this work reliably -- it's specifically
    VisPy's *camera* classes that didn't accept per-frame orientation, see
    module docstring.)"""

    ROTOR_OFFSETS = [(0.15, 0.15, (1, 0, 0, 1)), (0.15, -0.15, (0, 1, 0, 1)),
                      (-0.15, 0.15, (0, 0, 1, 1)), (-0.15, -0.15, (1, 1, 0, 1))]

    def __init__(self, view: scene.ViewBox):
        self.body = visuals.Box(width=0.24, height=0.06, depth=0.24, color=(0.2, 0.6, 0.9, 1))
        view.add(self.body)
        self.rotors = []
        for dx, dy, color in self.ROTOR_OFFSETS:
            s = visuals.Sphere(radius=0.05, color=color)
            self.rotors.append((s, np.array([dx, dy, 0.02])))
            view.add(s)

    def update(self, position: np.ndarray, quaternion: np.ndarray):
        """quaternion: (w, x, y, z) as returned by Drone.get_state().

        Builds the transform via VisPy's own .rotate()/.translate() calls
        (the same pattern VisPy's TurntableCamera uses internally) rather
        than hand-constructing a 4x4 matrix -- a hand-built matrix here
        moved the box correctly but never visibly rotated it, most likely
        a row/column-convention mismatch with what MatrixTransform expects.
        """
        rot = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        rotvec = rot.as_rotvec()
        norm = np.linalg.norm(rotvec)
        angle = np.degrees(norm)
        axis = rotvec / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

        transform = scene.transforms.MatrixTransform()
        transform.rotate(angle, axis)
        transform.translate(position)
        self.body.transform = transform
        for sphere, offset in self.rotors:
            world_offset = rot.apply(offset) + position
            sphere.transform = scene.transforms.STTransform(translate=world_offset)


def build_display_canvas():
    canvas = scene.SceneCanvas(bgcolor="#87ceeb", keys="interactive")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, distance=6, elevation=20)
    _build_world_scene(view)
    drone_visual = DroneVisual(view)
    return canvas, drone_visual
