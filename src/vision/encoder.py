"""Camera frame -> visual-neuron stimulation encoding.

This is explicitly an engineering pipeline (see docs/scientific_assumptions.md):
it does not reproduce real photoreceptor/lamina transduction. It converts a
rendered camera frame into a LEFT/RIGHT drive signal for the verified
photoreceptor populations in src/brain/pathways.py, using the same
tonic-drive/disinhibition logic validated in Phase 2
(src/experiments/phase2_visual_response.py): brighter = more "light" =
LESS photoreceptor tonic drive on that side (since real photoreceptors are
histaminergic and tonically active in the dark; light reduces that).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def detect_food_in_frame(frame: np.ndarray, min_pixels: int = 8) -> tuple[float, float]:
    """Return ``(horizontal_error, confidence)`` for the orange food pixels.

    The error is normalized to [-1, 1], with positive meaning food is left
    of image centre. This is an explicit engineered colour-salience readout,
    not a claim about a named fly neural pathway. Crucially, it receives only
    camera pixels -- never food or drone world coordinates.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("frame must be an RGB image")
    rgb = image[..., :3].astype(np.float32)
    if rgb.max(initial=0.0) <= 1.0:
        rgb *= 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (red > 140) & (red > 1.45 * green) & (green > 1.25 * blue)
    ys, xs = np.nonzero(mask)
    if xs.size < min_pixels:
        return 0.0, 0.0
    centre_x = 0.5 * (image.shape[1] - 1)
    horizontal_error = (centre_x - float(xs.mean())) / max(centre_x, 1.0)
    confidence = min(1.0, xs.size / max(1.0, 0.02 * image.shape[0] * image.shape[1]))
    return float(np.clip(horizontal_error, -1.0, 1.0)), float(confidence)


def encode_left_right_from_bearing(
    drone_position: np.ndarray, drone_quaternion: np.ndarray, landmark_position: np.ndarray, tonic_drive: float,
) -> tuple[float, float]:
    """Two-sensor (Braitenberg-vehicle-style) light model, used by
    src/visualization/dashboard.py instead of a rendered camera.

    Why: src/experiments/phase4_closed_loop.py already demonstrates the
    real requirement -- a rendered camera image driving the connectome --
    using MuJoCo's own Renderer in a standalone script, and that works
    correctly (verified: hover stable, yaw rotates cleanly, camera frames
    change as expected). But MuJoCo's Renderer and VisPy's Qt-embedded
    OpenGL context conflict when both run in the same process (confirmed:
    MuJoCo's renderer returns byte-identical frames regardless of drone
    orientation once VisPy's canvas is active in the same process -- see
    src/drone/camera.py's docstring for the debugging history). Rather than
    build a separate-process IPC bridge just to keep pixel-perfect
    rendering inside the unified dashboard, we use this simpler two-sensor
    model here instead, honestly documented as a substitution, not a
    silent downgrade. It is functionally similar to how real phototaxis
    robotics experiments are often modeled (two light sensors, not a full
    camera), not an invented shortcut -- but it IS a step down from "a
    rendered image," so we say so explicitly.
    """
    rot = Rotation.from_quat([drone_quaternion[1], drone_quaternion[2], drone_quaternion[3], drone_quaternion[0]])
    forward = rot.apply([1.0, 0.0, 0.0])
    left_axis = rot.apply([0.0, 1.0, 0.0])

    to_landmark = landmark_position - drone_position
    distance = np.linalg.norm(to_landmark) + 1e-6
    direction = to_landmark / distance

    forward_component = np.dot(direction, forward)
    if forward_component <= 0:  # landmark behind the drone: not visible
        return tonic_drive, tonic_drive

    lateral_component = np.dot(direction, left_axis)  # + = landmark to the left
    falloff = 1.0 / (1.0 + 0.1 * distance**2)
    # Two forward-facing sensors with overlapping fields of view.  The old
    # implementation used only the lateral component, which made a target
    # straight ahead produce zero brightness on both sensors -- exactly the
    # same code as a target behind the drone.  Keeping the common forward
    # component lets bilateral brightness encode visibility/distance while
    # the lateral component still supplies the steering difference.
    left_brightness = np.clip(forward_component + lateral_component, 0.0, 1.0) * falloff
    right_brightness = np.clip(forward_component - lateral_component, 0.0, 1.0) * falloff

    left_drive = tonic_drive * (1.0 - left_brightness)
    right_drive = tonic_drive * (1.0 - right_brightness)
    return left_drive, right_drive


def food_distance_and_bearing(
    drone_position: np.ndarray, drone_quaternion: np.ndarray, food_position: np.ndarray,
) -> tuple[float, float, bool]:
    """Distance (m), bearing angle (degrees, + = left) and visibility, for
    display/logging only (src/experiments/phase7_food_search.py) -- NOT fed
    to the connectome. The connectome only ever gets
    encode_left_right_from_bearing's two drive scalars; this function lets
    us print/record "FOOD DISTANCE" / "TARGET ANGLE" the way the brief asks
    for without smuggling coordinates into the controller.
    """
    rot = Rotation.from_quat([drone_quaternion[1], drone_quaternion[2], drone_quaternion[3], drone_quaternion[0]])
    forward = rot.apply([1.0, 0.0, 0.0])
    left_axis = rot.apply([0.0, 1.0, 0.0])

    to_food = food_position - drone_position
    distance = float(np.linalg.norm(to_food))
    direction = to_food / (distance + 1e-6)

    forward_component = np.dot(direction, forward)
    lateral_component = np.dot(direction, left_axis)
    bearing_deg = float(np.degrees(np.arctan2(lateral_component, forward_component)))
    visible = forward_component > 0
    return distance, bearing_deg, visible


def encode_looming_from_walls(
    drone_position: np.ndarray, drone_quaternion: np.ndarray, arena_half_size: float,
) -> tuple[float, float]:
    """Wall-proximity-based looming signal, for LC4/LPLC2 (see
    src/brain/pathways.py's LOOMING_DETECTORS).

    Our simulation injects engineered signals directly at identified
    sensory-relay populations rather than computing real optic flow from
    rendered pixels (see encode_left_right_from_bearing's docstring for
    why, and docs/scientific_assumptions.md) -- there is no simulated
    motion-energy computation that would let a genuine T4/T5->LC4 looming
    response emerge on its own from a static bearing signal. So, exactly as
    we drive photoreceptors directly with a bearing-based drive, we drive
    LC4/LPLC2 directly with this wall-proximity-based drive as an explicit,
    documented substitute for true looming computation, not an attempt to
    hide the substitution.

    Returns (left_drive, right_drive): EXTRA excitatory current added on
    top of LC4/LPLC2's normal ambient drive, higher when a wall is nearer
    on that side (unlike the photoreceptor encoder, this is a positive
    drive, not a tonic-drive subtraction -- LC4/LPLC2 are cholinergic,
    not histaminergic, so there's no tonic-release-in-the-dark biology to
    mirror here).
    """
    rot = Rotation.from_quat([drone_quaternion[1], drone_quaternion[2], drone_quaternion[3], drone_quaternion[0]])
    left_axis = rot.apply([0.0, 1.0, 0.0])

    x, y = drone_position[0], drone_position[1]
    wall_dists = {
        "east": (arena_half_size - x, np.array([1.0, 0.0, 0.0])),
        "west": (arena_half_size + x, np.array([-1.0, 0.0, 0.0])),
        "north": (arena_half_size - y, np.array([0.0, 1.0, 0.0])),
        "south": (arena_half_size + y, np.array([0.0, -1.0, 0.0])),
    }
    left_signal, right_signal = 0.0, 0.0
    for dist, direction in wall_dists.values():
        dist = max(dist, 0.05)
        proximity = 1.0 / (1.0 + dist**2)
        lateral = np.dot(direction, left_axis)
        if lateral > 0:
            left_signal += proximity * lateral
        else:
            right_signal += proximity * (-lateral)
    return left_signal, right_signal


def encode_left_right_drive(frame: np.ndarray, tonic_drive: float) -> tuple[float, float]:
    """frame: (H, W, 3) uint8 RGB image, e.g. from the drone's camera.

    Returns (left_drive, right_drive): the external input current to apply
    to LEFT-hemisphere and RIGHT-hemisphere photoreceptors respectively.
    Brightness is averaged over each half of the image; brighter -> lower
    drive (more "light", less tonic dark-state drive), matching Phase 2's
    disinhibition logic. This is a simple, engineered proxy for "where is
    it bright," not a model of real photoreceptor optics.
    """
    h, w = frame.shape[:2]
    left_half = frame[:, : w // 2]
    right_half = frame[:, w // 2 :]
    left_brightness = left_half.mean() / 255.0
    right_brightness = right_half.mean() / 255.0
    left_drive = tonic_drive * (1.0 - left_brightness)
    right_drive = tonic_drive * (1.0 - right_brightness)
    return left_drive, right_drive
