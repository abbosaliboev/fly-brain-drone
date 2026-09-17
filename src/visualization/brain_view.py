"""Interactive 3D viewer for the MaleCNS connectome neuron point cloud.

Phase 1 scope: show every neuron with a known soma position as a point in
3D space, colored by broad region (brain vs. VNC, by z-position), with
mouse zoom/rotate/pan. No activity/simulation yet -- that starts in Phase 2,
at which point per-neuron color will be driven by simulated firing instead
of static region color.
"""
from __future__ import annotations

import sys

import numpy as np
from vispy import app, scene
from vispy.scene import visuals

sys.path.insert(0, str(__file__.rsplit("src", 1)[0]))
from src.brain.connectome import get_client, fetch_all_neurons  # noqa: E402


def _region_colors(z: np.ndarray) -> np.ndarray:
    """Color neurons by z-position as a rough brain-vs-VNC proxy.

    This is a visualization convenience, not a real neuropil-region
    classification (that would come from ROI queries in a later phase).
    """
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    colors = np.zeros((len(z), 4), dtype=np.float32)
    colors[:, 0] = 0.3 + 0.5 * z_norm       # more red toward one end
    colors[:, 1] = 0.5 + 0.3 * (1 - z_norm) # more green toward the other
    colors[:, 2] = 0.9 - 0.4 * z_norm       # blue-ish overall
    colors[:, 3] = 0.85
    return colors


def build_canvas(neurons_df) -> scene.SceneCanvas:
    positions = neurons_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    # Center at origin so the default camera distance is sane regardless of
    # this dataset's absolute voxel-coordinate range. (.copy() because
    # parquet-backed columns can hand back a read-only array.)
    positions = positions.copy() - positions.mean(axis=0)

    colors = _region_colors(neurons_df["z"].to_numpy(dtype=np.float32))

    canvas = scene.SceneCanvas(
        title=f"Fly Brain Drone -- MaleCNS v1.0 connectome ({len(neurons_df):,} neurons)",
        keys="interactive",
        bgcolor="#0b0f14",
        size=(1280, 800),
        show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, distance=positions.std() * 4)

    scatter = visuals.Markers()
    scatter.set_data(positions, face_color=colors, size=3, edge_width=0)
    view.add(scatter)

    axis = visuals.XYZAxis(parent=view.scene)
    axis.transform = scene.transforms.STTransform(scale=(positions.std(),) * 3)

    info = scene.Text(
        f"{len(neurons_df):,} neurons shown (soma-position subset of 165,122 traced;\n"
        "176,422 total :Neuron nodes, live neuPrint query)\n"
        "~125,000,000 synapses in the full connectome (Berg et al., Cell 2026 --\n"
        "not re-queried live, that aggregate query times out server-side)\n"
        "drag = rotate, scroll = zoom, shift+drag = pan",
        parent=canvas.scene,
        color="white",
        anchor_x="left",
        anchor_y="top",
        pos=(10, 10),
        font_size=10,
    )
    info.order = 10

    return canvas


def main():
    client = get_client()
    neurons_df = fetch_all_neurons(client)
    canvas = build_canvas(neurons_df)  # noqa: F841 -- keep alive
    app.run()


if __name__ == "__main__":
    main()
