"""Optional pretrained FlyVis visual-motion backend.

FlyVis supplies a learned, connectome-constrained optic-lobe motion estimate;
our MaleCNS model remains the downstream controller. This boundary is explicit:
it is not an end-to-end simulation of one individual fly connectome.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MOTION_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
PHOTORECEPTOR_TYPES = ("R1", "R2", "R3", "R4", "R5", "R6")


@dataclass(frozen=True)
class FlyVisMotion:
    horizontal: float
    vertical: float
    confidence: float
    populations: Mapping[str, float]


def decode_t4_t5_motion(
    activity: np.ndarray, layer_indices: Mapping[str, Sequence[int]]
) -> FlyVisMotion:
    """Decode opponent T4/T5 populations without importing FlyVis."""
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim == 2:
        values = values[-1]
    if values.ndim != 1:
        raise ValueError("activity must have shape (neurons,) or (frames, neurons)")

    populations: dict[str, float] = {}
    for cell_type in MOTION_TYPES:
        index = np.asarray(layer_indices[cell_type], dtype=np.int64)
        if index.size == 0:
            raise ValueError(f"FlyVis layer {cell_type} is empty")
        populations[cell_type] = float(values[index].mean())

    # Official FlyVis preferences: b=0, a=180, c=90, d=270 degrees.
    horizontal = 0.5 * (
        populations["T4b"] + populations["T5b"]
        - populations["T4a"] - populations["T5a"]
    )
    vertical = 0.5 * (
        populations["T4c"] + populations["T5c"]
        - populations["T4d"] - populations["T5d"]
    )
    return FlyVisMotion(
        horizontal=horizontal,
        vertical=vertical,
        confidence=float(np.hypot(horizontal, vertical)),
        populations=populations,
    )


class FlyVisBackend:
    """Lazy wrapper around one official pretrained FlyVis network."""

    def __init__(self, root_dir: str | Path = "data/flyvis",
                 network_name: str = "flow/0000/000", dt: float = 1 / 50,
                 retinal_extent: int = 15):
        self.root_dir = Path(root_dir).resolve()
        self.network_name = network_name
        self.dt = dt
        self.retinal_extent = retinal_extent
        self._network = None
        self._eye = None
        self._layer_indices = None
        self._stream_state = None
        self._last_activity = None
        self._baseline_activity = None
        self._last_r7r8_drive = None
        self._last_color_drives = None

    def load(self) -> None:
        if self._network is not None:
            return
        os.environ["FLYVIS_ROOT_DIR"] = str(self.root_dir)
        try:
            from flyvis import NetworkView
            from flyvis.datasets.rendering.eye import BoxEye
            from flyvis.network.network_view import CheckpointedNetwork

            view = NetworkView(self.network_name)
            # FlyVis parameters are shared by cell/edge type across columns.
            # A smaller central lattice is therefore useful for constrained
            # online CPU inference without changing learned parameter values.
            config = view.dir.config.network.to_dict()
            config["connectome"]["extent"] = self.retinal_extent
            checkpointed = CheckpointedNetwork(
                view.network_class, config, view.name, view.get_checkpoint("best"),
                view.recover_fn,
            )
            self._network = checkpointed.recover()
            self._network.eval()
            self._eye = BoxEye(extent=self.retinal_extent, kernel_size=13)
            self._layer_indices = {
                name: np.asarray(self._network.connectome.nodes.layer_index[name][:])
                for name in MOTION_TYPES
            }
        except MemoryError as exc:
            raise RuntimeError(
                "FlyVis connectome construction exhausted Windows virtual memory. "
                "Close memory-heavy applications or increase the page file, then retry; "
                "the downloaded pretrained weights are intact."
            ) from exc

    def infer(self, frames: np.ndarray) -> FlyVisMotion:
        """Run RGB/greyscale frames through FlyVis and decode T4/T5 motion."""
        self.load()
        import torch

        movie = np.asarray(frames)
        if movie.ndim == 3:
            grey = movie
        elif movie.ndim == 4 and movie.shape[-1] in (3, 4):
            grey = movie[..., :3].mean(axis=-1)
        else:
            raise ValueError("frames must be (T,H,W) or (T,H,W,3/4)")
        if grey.shape[0] < 2:
            raise ValueError("motion inference requires at least two frames")
        grey = grey.astype(np.float32)
        if grey.max(initial=0.0) > 1.0:
            grey /= 255.0

        tensor = torch.as_tensor(grey[None], dtype=torch.float32)
        hex_movie = self._eye(tensor)
        with torch.inference_mode():
            response = self._network.simulate(hex_movie, dt=self.dt)
        return decode_t4_t5_motion(
            response[0].detach().cpu().numpy(), self._layer_indices
        )

    @staticmethod
    def _greyscale_frames(frames: np.ndarray) -> np.ndarray:
        movie = np.asarray(frames)
        if movie.ndim == 3:
            grey = movie
        elif movie.ndim == 4 and movie.shape[-1] in (3, 4):
            grey = movie[..., :3].mean(axis=-1)
        else:
            raise ValueError("frames must be (T,H,W) or (T,H,W,3/4)")
        grey = grey.astype(np.float32)
        if grey.max(initial=0.0) > 1.0:
            grey /= 255.0
        return grey

    def reset_stream(self, grey_value: float = 0.5) -> None:
        """Initialize the recurrent state once for low-rate online use."""
        self.load()
        import torch

        with torch.inference_mode():
            self._stream_state = self._network.steady_state(
                1.0, self.dt, batch_size=1, value=grey_value
            )
        self._baseline_activity = (
            self._stream_state.nodes.activity[0].detach().cpu().numpy().copy()
        )

    def infer_frame(self, frame: np.ndarray) -> FlyVisMotion:
        """Advance the recurrent FlyVis network by one camera frame."""
        self.load()
        import torch

        if self._stream_state is None:
            self.reset_stream()
        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            grey = image[..., :3].mean(axis=-1)
        elif image.ndim == 2:
            grey = image
        else:
            raise ValueError("frame must be (H,W) or (H,W,3/4)")
        grey = grey.astype(np.float32)
        if grey.max(initial=0.0) > 1.0:
            grey /= 255.0
        tensor = torch.as_tensor(grey[None, None], dtype=torch.float32)
        hex_frame = self._eye(tensor)
        self._network.stimulus.zero(1, 1)
        self._network.stimulus.add_input(hex_frame)
        with torch.inference_mode():
            state = self._network(
                self._network.stimulus(), self.dt,
                state=self._stream_state, as_states=True,
            )[-1]
        self._stream_state = state
        activity = state.nodes.activity[0].detach().cpu().numpy()
        self._last_activity = activity
        return decode_t4_t5_motion(activity, self._layer_indices)

    def infer_color_frame(self, frame: np.ndarray) -> FlyVisMotion:
        """Advance with separate broadband, R7-blue and R8-green inputs.

        RGB blue is only a proxy for the fly's UV-sensitive R7 channel; this
        is an explicit engineering approximation, not spectral calibration.
        """
        self.load()
        import torch

        if self._stream_state is None:
            self.reset_stream()
        image = np.asarray(frame)[..., :3].astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image /= 255.0
        luminance = image.mean(axis=-1)
        channels = [luminance, image[..., 2], image[..., 1]]
        hex_channels = [
            self._eye(torch.as_tensor(channel[None, None], dtype=torch.float32))[0, 0, 0]
            for channel in channels
        ]
        self._last_r7r8_drive = (
            hex_channels[2] - hex_channels[1]
        ).detach().cpu().numpy()
        self._last_color_drives = tuple(
            channel.detach().cpu().numpy() for channel in hex_channels
        )
        stimulus = self._network.stimulus
        stimulus.zero(1, 1)
        # input_index order follows connectome.input_cell_types: R1..R8.
        for row in range(6):
            stimulus.buffer[0, 0, stimulus.input_index[row]] = hex_channels[0]
        stimulus.buffer[0, 0, stimulus.input_index[6]] = hex_channels[1]  # R7 blue/UV proxy
        stimulus.buffer[0, 0, stimulus.input_index[7]] = hex_channels[2]  # R8 green
        stimulus._nonzero = True
        with torch.inference_mode():
            state = self._network(
                stimulus(), self.dt, state=self._stream_state, as_states=True
            )[-1]
        self._stream_state = state
        activity = state.nodes.activity[0].detach().cpu().numpy()
        self._last_activity = activity
        return decode_t4_t5_motion(activity, self._layer_indices)

    def r7_r8_horizontal_salience(self) -> tuple[float, float]:
        """Decode green-vs-blue opponent salience from R7/R8 activity."""
        if self._last_activity is None:
            raise RuntimeError("infer_color_frame must be called first")
        types = self._network.connectome.nodes.type[:].astype(str)
        u_all = self._network.connectome.nodes.u[:]
        v_all = self._network.connectome.nodes.v[:]
        site_values: dict[tuple[int, int], dict[str, float]] = {}
        for cell_type in ("R7", "R8"):
            indices = np.nonzero(types == cell_type)[0]
            for index in indices:
                key = (int(u_all[index]), int(v_all[index]))
                value = float(self._last_activity[index])
                if self._baseline_activity is not None:
                    value -= float(self._baseline_activity[index])
                site_values.setdefault(key, {})[cell_type] = value
        sites = [key for key, value in site_values.items() if len(value) == 2]
        v = np.asarray([key[1] for key in sites], dtype=np.float64)
        opponent = np.asarray([
            site_values[key]["R8"] - site_values[key]["R7"] for key in sites
        ])
        # Retain positive green-over-blue chromatic outliers.
        weights = np.maximum(opponent - np.percentile(opponent, 75.0), 0.0)
        total = float(weights.sum())
        if total < 1e-8:
            return 0.0, 0.0
        extent = max(float(np.max(np.abs(v))), 1.0)
        error = -float(np.dot(v, weights) / total) / extent
        confidence = min(1.0, total / max(len(v) * 0.02, 1e-8))
        return float(np.clip(error, -1.0, 1.0)), float(confidence)

    def r7_r8_input_salience(self) -> tuple[float, float]:
        """Decode colour salience at the hex-sampled receptor-drive stage."""
        if self._last_r7r8_drive is None:
            raise RuntimeError("infer_color_frame must be called first")
        r7_indices = self._network.connectome.nodes.layer_index["R7"][:]
        v = np.asarray(self._network.connectome.nodes.v[:][r7_indices], dtype=np.float64)
        opponent = np.asarray(self._last_r7r8_drive, dtype=np.float64)
        weights = np.maximum(opponent - 0.05, 0.0)
        total = float(weights.sum())
        if total < 1e-8:
            return 0.0, 0.0
        extent = max(float(np.max(np.abs(v))), 1.0)
        error = -float(np.dot(v, weights) / total) / extent
        confidence = min(1.0, total / max(len(v) * 0.02, 1e-8))
        return float(np.clip(error, -1.0, 1.0)), float(confidence)

    def r1_r7_r8_orange_salience(self) -> tuple[float, float]:
        """Decode red-over-green salience from broadband/R7/R8 drives.

        With broadband defined as RGB mean, ``red = 3L-G-B``. The opponent
        score ``red-green`` therefore separates orange from pure green while
        retaining the same hexagonal receptor sampling.
        """
        if self._last_color_drives is None:
            raise RuntimeError("infer_color_frame must be called first")
        luminance, blue, green = self._last_color_drives
        red_proxy = 3.0 * luminance - green - blue
        opponent = red_proxy - green
        r7_indices = self._network.connectome.nodes.layer_index["R7"][:]
        v = np.asarray(self._network.connectome.nodes.v[:][r7_indices], dtype=np.float64)
        weights = np.maximum(np.asarray(opponent, dtype=np.float64) - 0.08, 0.0)
        total = float(weights.sum())
        if total < 1e-8:
            return 0.0, 0.0
        extent = max(float(np.max(np.abs(v))), 1.0)
        error = -float(np.dot(v, weights) / total) / extent
        confidence = min(1.0, total / max(len(v) * 0.03, 1e-8))
        return float(np.clip(error, -1.0, 1.0)), float(confidence)

    def retinal_horizontal_salience(self) -> tuple[float, float]:
        """Decode a contrast centroid from FlyVis R1-R6 neural activity.

        Returns ``(horizontal_error, confidence)`` with positive error on
        the image-left side. This reads neural activity and retinotopic node
        coordinates, not RGB values, but remains an engineered readout.
        """
        if self._last_activity is None:
            raise RuntimeError("infer_frame must be called before salience decoding")
        type_values = self._network.connectome.nodes.type[:].astype(str)
        v_values = self._network.connectome.nodes.v[:]
        mask = np.isin(type_values, PHOTORECEPTOR_TYPES)
        activity = self._last_activity[mask]
        if self._baseline_activity is not None:
            activity = activity - self._baseline_activity[mask]
        u = np.asarray(self._network.connectome.nodes.u[:][mask], dtype=np.int64)
        v = np.asarray(v_values[mask], dtype=np.float64)
        # Average the six co-located receptor channels before measuring
        # spatial contrast, otherwise their type baselines dominate.
        sums: dict[tuple[int, int], list[float]] = {}
        for u_coord, v_coord, value in zip(u, v.astype(int), activity):
            sums.setdefault((int(u_coord), int(v_coord)), []).append(float(value))
        sites = sorted(sums)
        coords = np.asarray([site[1] for site in sites], dtype=np.float64)
        response = np.asarray([np.mean(sums[site]) for site in sites])
        spatial_response = response - np.median(response)
        raw_contrast = np.abs(spatial_response)
        # Recurrent adaptation produces a weak global response; retain the
        # most spatially selective quartile so it cannot pull the centroid
        # back toward zero.
        floor = float(np.percentile(raw_contrast, 75.0))
        contrast = np.maximum(raw_contrast - floor, 0.0)
        total = float(contrast.sum())
        if total < 1e-8:
            return 0.0, 0.0
        centroid = float(np.dot(coords, contrast) / total)
        extent = max(float(np.max(np.abs(coords))), 1.0)
        error = -centroid / extent  # negative v is image-left in BoxEye layout
        confidence = min(1.0, total / max(len(coords) * 0.1, 1e-8))
        return float(np.clip(error, -1.0, 1.0)), float(confidence)

    def activity_features(self, cell_types: Sequence[str] = PHOTORECEPTOR_TYPES) -> np.ndarray:
        """Return the latest activity for selected types in node-index order."""
        if self._last_activity is None:
            raise RuntimeError("infer_frame must be called before reading features")
        types = self._network.connectome.nodes.type[:].astype(str)
        return np.asarray(self._last_activity[np.isin(types, tuple(cell_types))], dtype=np.float32)

    def retinal_snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return hex coordinates and latest L/R7/R8 receptor drives for UI use."""
        if self._last_color_drives is None:
            raise RuntimeError("infer_color_frame must be called first")
        r7_indices = self._network.connectome.nodes.layer_index["R7"][:]
        u = np.asarray(self._network.connectome.nodes.u[:][r7_indices], dtype=np.float32)
        v = np.asarray(self._network.connectome.nodes.v[:][r7_indices], dtype=np.float32)
        drives = np.stack(self._last_color_drives, axis=-1).astype(np.float32, copy=True)
        return u, v, drives

    def activity_by_type(self, cell_types: Sequence[str]) -> dict[str, float]:
        """Mean absolute baseline-relative activity for requested cell types."""
        if self._last_activity is None:
            raise RuntimeError("neural inference must be called first")
        values = np.asarray(self._last_activity, dtype=np.float32)
        if self._baseline_activity is not None:
            values = values - np.asarray(self._baseline_activity, dtype=np.float32)
        layers = self._network.connectome.nodes.layer_index
        output: dict[str, float] = {}
        for cell_type in cell_types:
            if cell_type not in layers:
                output[cell_type] = 0.0
                continue
            indices = np.asarray(layers[cell_type][:], dtype=np.int64)
            output[cell_type] = float(np.mean(np.abs(values[indices])))
        return output
