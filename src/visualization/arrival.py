"""Render-only game referee. Its result must never select motor commands."""
import mujoco
import numpy as np


def visible_apple(renderer, data, apple_ids):
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera="drone_eye")
        labels = renderer.render()
        mask = np.isin(labels[..., 0], apple_ids) & (labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        pixels = int(np.count_nonzero(mask))
        # ID confidence is independent of recurrent adaptation and controller
        # mode. It confirms the rendered target, not a similarly coloured prop.
        return pixels >= 8, min(1.0, pixels / max(1, mask.size * .02))
    finally:
        renderer.disable_segmentation_rendering()
