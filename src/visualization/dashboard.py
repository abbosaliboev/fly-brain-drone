"""Phase 5: unified real-time dashboard.

    +------------------------+------------------------+
    |      3D BRAIN VIEW     |    3D DRONE WORLD       |
    |  (steering pathway,    |  (VisPy 3rd-person      |
    |   glowing on spike)    |   view, interactive)    |
    +------------------------+------------------------+
    | Visual input (L/R) | Neural activity | Motor out |
    +----------------------------------------------------+
    | FPS | Sim Hz | GPU | Latency                       |
    +----------------------------------------------------+

Closes the same loop as Phase 4 (src/experiments/phase4_closed_loop.py),
in one window with the brain visualization from Phase 2 alongside it, but
uses a two-sensor bearing-based vision model instead of a rendered camera
image -- see src/vision/encoder.py's encode_left_right_from_bearing and
src/drone/camera.py for why (MuJoCo's Renderer and VisPy's Qt-embedded
OpenGL context conflict in one process). Phase 4's script remains the
reference implementation for "a rendered camera image drives the
connectome." See docs/scientific_assumptions.md for what is/isn't
validated in this pipeline -- this dashboard doesn't change any of that,
just visualizes it together.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from vispy import scene
from vispy.scene import visuals

from src.brain.connectome import get_client, fetch_subnetwork, fetch_predicted_nt, fetch_all_neurons
from src.brain.neurons import LIFParams
from src.brain.pathways import (
    FULL_STEERING_PATHWAY, PHOTORECEPTORS, LAMINA, MEDULLA, MOTION_DETECTORS,
    LOBULA_PLATE_TANGENTIAL, STEERING_DESCENDING, hemisphere_stimulus_populations,
)
from src.brain.simulator import ConnectomeNetwork, LIFSimulator
from src.control.motor_decoder import SteeringDecoder
from src.drone.drone import Drone
from src.drone.camera import build_display_canvas, LANDMARK_POS
from src.drone.physics import mix_to_rotors
from src.environment.world import build_model
from src.vision.encoder import encode_left_right_from_bearing

TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.2
ACTIVITY_DECAY = 0.9
GLOW_DECAY = 0.85

STAGE_COLORS = {
    "photoreceptor": (0.7, 0.3, 0.9),
    "lamina": (0.3, 0.8, 0.9),
    "medulla": (0.95, 0.85, 0.2),
    "T4/T5": (0.95, 0.5, 0.2),
    "HS/VS": (0.4, 0.95, 0.5),
    "DNa02": (1.0, 0.2, 0.2),
}


def stage_of(type_: str) -> str:
    if type_ in PHOTORECEPTORS:
        return "photoreceptor"
    if type_ in LAMINA:
        return "lamina"
    if type_ in MEDULLA:
        return "medulla"
    if type_ in MOTION_DETECTORS:
        return "T4/T5"
    if type_ in LOBULA_PLATE_TANGENTIAL:
        return "HS/VS"
    if type_ in STEERING_DESCENDING:
        return "DNa02"
    return "other"


def gpu_stats() -> tuple[float, float]:
    """Returns (utilization %, memory used MB), or (0, 0) if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.0,
        )
        util, mem = out.stdout.strip().split(",")
        return float(util), float(mem)
    except Exception:
        return 0.0, 0.0


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Brain Drone -- Phase 5 dashboard")
        self.resize(1600, 1000)

        client = get_client()
        neuron_df, conn_df = fetch_subnetwork(client, FULL_STEERING_PATHWAY, "steering")
        nt_df = fetch_predicted_nt(client, FULL_STEERING_PATHWAY, "steering")
        all_neurons = fetch_all_neurons(client)

        self.net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
        self.sim = LIFSimulator(self.net, LIFParams(), gain=GAIN)
        self.decoder = SteeringDecoder(self.net)
        self.activity_ema = np.zeros(self.net.n, dtype=np.float32)

        df = self.net.neuron_df
        df["stage"] = df["type"].map(stage_of)
        df["hemisphere"] = df["instance"].str[-1]

        pos_lookup = all_neurons.set_index("bodyId")[["x", "y", "z"]]
        has_pos = df["bodyId"].isin(pos_lookup.index).to_numpy()
        self.plot_idx = np.nonzero(has_pos)[0]
        positions = pos_lookup.loc[df["bodyId"].to_numpy()[self.plot_idx]].to_numpy(dtype=np.float32)
        self.brain_positions = positions.copy() - positions.mean(axis=0)
        self.base_colors = np.array(
            [STAGE_COLORS.get(s, (0.5, 0.5, 0.5)) for s in df["stage"].to_numpy()[self.plot_idx]],
            dtype=np.float32,
        )
        self.glow = np.zeros(self.net.n, dtype=np.float32)

        self.photoreceptor_mask = torch.tensor(df["type"].isin(PHOTORECEPTORS).to_numpy(), device=self.net.device)
        self.left_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "L").to_numpy(), device=self.net.device)
        self.right_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "R").to_numpy(), device=self.net.device)

        self.stage_groups = {}
        for stage in ["photoreceptor", "lamina", "medulla", "T4/T5", "HS/VS"]:
            self.stage_groups[stage] = np.nonzero((df["stage"] == stage).to_numpy())[0]

        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.drone = Drone(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_gpu_check = 0.0
        self.gpu_util, self.gpu_mem = 0.0, 0.0
        self.frame_times = []

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # target ~50 fps

    def _build_ui(self):
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        brain_canvas = scene.SceneCanvas(bgcolor="#0b0f14", keys="interactive")
        view = brain_canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, distance=self.brain_positions.std() * 4)
        self.scatter = visuals.Markers()
        self.scatter.set_data(self.brain_positions, face_color=self.base_colors, size=4, edge_width=0)
        view.add(self.scatter)
        brain_box = QtWidgets.QGroupBox("3D BRAIN -- steering pathway (glows on spike)")
        brain_layout = QtWidgets.QVBoxLayout(brain_box)
        brain_layout.addWidget(brain_canvas.native)
        top.addWidget(brain_box, stretch=1)

        drone_box = QtWidgets.QGroupBox("3D DRONE WORLD (drag to rotate view)")
        drone_layout = QtWidgets.QVBoxLayout(drone_box)
        self.drone_canvas, self.drone_visual = build_display_canvas()
        drone_layout.addWidget(self.drone_canvas.native)
        top.addWidget(drone_box, stretch=1)
        outer.addLayout(top, stretch=3)

        bottom = QtWidgets.QHBoxLayout()

        visual_box = QtWidgets.QGroupBox("VISUAL INPUT (2-sensor bearing model, see docs)")
        visual_layout = QtWidgets.QVBoxLayout(visual_box)
        self.left_bright_bar = QtWidgets.QProgressBar(); self.left_bright_bar.setFormat("LEFT %p%")
        self.right_bright_bar = QtWidgets.QProgressBar(); self.right_bright_bar.setFormat("RIGHT %p%")
        visual_layout.addWidget(self.left_bright_bar)
        visual_layout.addWidget(self.right_bright_bar)
        bottom.addWidget(visual_box)

        neural_box = QtWidgets.QGroupBox("NEURAL ACTIVITY (spikes/step, this population)")
        neural_layout = QtWidgets.QVBoxLayout(neural_box)
        self.neural_bars = {}
        for stage in ["photoreceptor", "lamina", "medulla", "T4/T5", "HS/VS"]:
            bar = QtWidgets.QProgressBar(); bar.setFormat(f"{stage} %p%")
            self.neural_bars[stage] = bar
            neural_layout.addWidget(bar)
        bottom.addWidget(neural_box)

        motor_box = QtWidgets.QGroupBox("MOTOR OUTPUT (DNa02-decoded)")
        motor_layout = QtWidgets.QVBoxLayout(motor_box)
        self.left_turn_bar = QtWidgets.QProgressBar(); self.left_turn_bar.setFormat("LEFT TURN %p%")
        self.right_turn_bar = QtWidgets.QProgressBar(); self.right_turn_bar.setFormat("RIGHT TURN %p%")
        forward_label = QtWidgets.QLabel("FORWARD/UP: not implemented (no verified pathway yet)")
        motor_layout.addWidget(self.left_turn_bar)
        motor_layout.addWidget(self.right_turn_bar)
        motor_layout.addWidget(forward_label)
        bottom.addWidget(motor_box)

        metrics_box = QtWidgets.QGroupBox("METRICS")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_box)
        self.metrics_label = QtWidgets.QLabel("")
        self.metrics_label.setStyleSheet("font-family: Consolas, monospace;")
        metrics_layout.addWidget(self.metrics_label)
        bottom.addWidget(metrics_box)

        outer.addLayout(bottom, stretch=1)
        self.setCentralWidget(central)

    def update_frame(self):
        t0 = time.perf_counter()

        state = self.drone.get_state()
        left_drive, right_drive = encode_left_right_from_bearing(
            state.position, state.quaternion, LANDMARK_POS, TONIC_DRIVE
        )

        ext = torch.full((self.net.n,), AMBIENT_DRIVE, device=self.net.device)
        ext[self.photoreceptor_mask] = TONIC_DRIVE
        ext[self.left_photo_mask] = left_drive
        ext[self.right_photo_mask] = right_drive
        spikes = self.sim.step(ext).cpu().numpy()
        self.activity_ema = ACTIVITY_DECAY * self.activity_ema + (1 - ACTIVITY_DECAY) * spikes

        command = self.decoder.decode(self.activity_ema)
        rotor_thrusts = mix_to_rotors(command)
        self.drone.apply_rotor_thrusts(rotor_thrusts)
        self.drone.step()
        self.step_count += 1

        self.glow = np.clip(self.glow * GLOW_DECAY + spikes.astype(np.float32), 0, 1)
        if len(self.plot_idx):
            glow_sub = self.glow[self.plot_idx][:, None]
            colors = self.base_colors * (0.25 + 0.75 * (1 - glow_sub)) + glow_sub * np.array([1.0, 1.0, 1.0])
            self.scatter.set_data(self.brain_positions, face_color=np.clip(colors, 0, 1), size=4, edge_width=0)

        new_state = self.drone.get_state()
        self.drone_visual.update(new_state.position, new_state.quaternion)

        self.left_bright_bar.setValue(int(100 * (1 - left_drive / TONIC_DRIVE)))
        self.right_bright_bar.setValue(int(100 * (1 - right_drive / TONIC_DRIVE)))
        for stage, idx in self.stage_groups.items():
            rate = spikes[idx].mean() if len(idx) else 0.0
            self.neural_bars[stage].setValue(int(100 * rate))
        self.left_turn_bar.setValue(min(100, int(command.left_turn * 20)))
        self.right_turn_bar.setValue(min(100, int(command.right_turn * 20)))

        now = time.time()
        if now - self.last_gpu_check > 1.0:
            self.gpu_util, self.gpu_mem = gpu_stats()
            self.last_gpu_check = now

        dt = time.perf_counter() - t0
        self.frame_times.append(dt)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times) + 1e-9)
        self.metrics_label.setText(
            f"FPS:        {fps:6.1f}\n"
            f"Sim step:   {self.step_count:6d}\n"
            f"GPU util:   {self.gpu_util:5.0f}%\n"
            f"GPU mem:    {self.gpu_mem:6.0f} MB\n"
            f"Latency:    {dt*1000:5.1f} ms/frame\n"
            f"Gain:       {GAIN} (unfit, see docs)"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
