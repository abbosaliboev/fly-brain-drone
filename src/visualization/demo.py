"""Phase 12: polished demo mode.

    INTRO SCREEN (title, one-line pitch, START EXPERIMENT)
            |
            v
    DASHBOARD -- brain (glowing) + drone world (interactive) +
                 FOOD DISTANCE / TARGET ANGLE / SEARCH TIME +
                 neural/motor activity bars + metrics +
                 a banner on success

This extends Phase 5's dashboard with Phase 6-7's food-search task
(steering + collision avoidance + a reachable food target) instead of
Phase 5's steering-only loop, and adds a "demo mode" checkbox that hides
the more technical labels (gain values, "not implemented" notes, raw
metrics) for a cleaner recording -- the underlying pipeline and its
documented limitations (docs/scientific_assumptions.md) are unchanged;
this only changes what's shown on screen.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import scene
from vispy.scene import visuals

from src.brain.connectome import get_client, fetch_subnetwork, fetch_predicted_nt, fetch_all_neurons
from src.brain.neurons import LIFParams
from src.brain.pathways import (
    FULL_NAVIGATION_PATHWAY, PHOTORECEPTORS, LAMINA, MEDULLA, MOTION_DETECTORS,
    LOBULA_PLATE_TANGENTIAL, STEERING_DESCENDING, LOOMING_DETECTORS, ESCAPE_DESCENDING,
    hemisphere_stimulus_populations,
)
from src.brain.simulator import ConnectomeNetwork, LIFSimulator, StimulusInjector
from src.control.motor_decoder import SensoryRescueDecoder, MotorCommand
from src.drone.drone import Drone
from src.drone.camera import build_display_canvas, LANDMARK_POS as FOOD_POS
from src.drone.physics import mix_to_rotors
from src.environment.world import build_model, ARENA_HALF_SIZE
from src.vision.encoder import encode_left_right_from_bearing, encode_looming_from_walls, food_distance_and_bearing

TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.2
ACTIVITY_DECAY = 0.9
GLOW_DECAY = 0.85
FORWARD_FORCE_N = 1.5
REACHED_RADIUS_M = 0.6
START_POS = (-3.0, -3.0, 1.0)

STAGE_COLORS = {
    "photoreceptor": (0.7, 0.3, 0.9), "lamina": (0.3, 0.8, 0.9), "medulla": (0.95, 0.85, 0.2),
    "T4/T5": (0.95, 0.5, 0.2), "HS/VS": (0.4, 0.95, 0.5), "DNa02": (1.0, 0.2, 0.2),
    "LC": (0.3, 0.9, 0.6), "DNp01": (1.0, 0.1, 0.6),
}


def stage_of(type_: str) -> str:
    if type_ in PHOTORECEPTORS: return "photoreceptor"
    if type_ in LAMINA: return "lamina"
    if type_ in MEDULLA: return "medulla"
    if type_ in MOTION_DETECTORS: return "T4/T5"
    if type_ in LOBULA_PLATE_TANGENTIAL: return "HS/VS"
    if type_ in STEERING_DESCENDING: return "DNa02"
    if type_ in LOOMING_DETECTORS: return "LC"
    if type_ in ESCAPE_DESCENDING: return "DNp01"
    return "other"


class IntroScreen(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #0b0f14;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("FLY BRAIN DRONE")
        title.setStyleSheet("color: white; font-size: 42px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)

        subtitle = QtWidgets.QLabel("166,700-neuron adult male Drosophila connectome (MaleCNS v1.0)\ncontrolling a virtual drone")
        subtitle.setStyleSheet("color: #aaaaaa; font-size: 16px;")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)

        diagram = QtWidgets.QLabel("BRAIN  -->  NEURAL SIGNAL  -->  DRONE  -->  FOOD")
        diagram.setStyleSheet("color: #4fc3f7; font-size: 20px; font-family: Consolas, monospace;")
        diagram.setAlignment(QtCore.Qt.AlignCenter)

        question = QtWidgets.QLabel('"Can a fly brain fly a drone toward food?"')
        question.setStyleSheet("color: #ffb74d; font-size: 18px; font-style: italic;")
        question.setAlignment(QtCore.Qt.AlignCenter)

        caveat = QtWidgets.QLabel(
            "Honest framing: a connectome is a wiring diagram, not a\n"
            "reproduction of fly cognition. See docs/scientific_assumptions.md."
        )
        caveat.setStyleSheet("color: #666666; font-size: 11px;")
        caveat.setAlignment(QtCore.Qt.AlignCenter)

        start_btn = QtWidgets.QPushButton("START EXPERIMENT")
        start_btn.setStyleSheet(
            "QPushButton { background-color: #4fc3f7; color: #0b0f14; font-size: 18px; "
            "font-weight: bold; padding: 14px 28px; border-radius: 8px; }"
            "QPushButton:hover { background-color: #81d4fa; }"
        )
        start_btn.clicked.connect(self.start_requested.emit)

        for w in (title, subtitle, diagram, question, caveat):
            layout.addSpacing(16)
            layout.addWidget(w)
        layout.addSpacing(30)
        layout.addWidget(start_btn, alignment=QtCore.Qt.AlignCenter)


class FoodSearchDashboard(QtWidgets.QMainWindow):
    def __init__(self, demo_mode: bool = True):
        super().__init__()
        self.demo_mode = demo_mode
        self.setWindowTitle("Fly Brain Drone -- Find the Food")
        self.resize(1600, 1000)

        client = get_client()
        neuron_df, conn_df = fetch_subnetwork(client, FULL_NAVIGATION_PATHWAY, "navigation")
        nt_df = fetch_predicted_nt(client, FULL_NAVIGATION_PATHWAY, "navigation")
        all_neurons = fetch_all_neurons(client)

        self.net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
        self.sim = LIFSimulator(self.net, LIFParams(), gain=GAIN)
        self.decoder = SensoryRescueDecoder(self.net)
        self.activity_ema = np.zeros(self.net.n, dtype=np.float32)

        df = self.net.neuron_df
        df["stage"] = df["type"].map(stage_of)

        # Full-brain context layer: all 141,781 neurons with a known soma
        # position (the same data Phase 1's viewer used), shown dim in the
        # background so the brain's real anatomical shape is visible --
        # not just the ~31k-neuron pathway we actually simulate. Both
        # layers are centered on the FULL brain's mean position so they
        # align spatially (our simulated pathway is a real spatial subset
        # of this cloud, not a separate dataset).
        full_positions_raw = all_neurons[["x", "y", "z"]].to_numpy(dtype=np.float32)
        brain_center = full_positions_raw.mean(axis=0)
        self.full_brain_positions = full_positions_raw - brain_center
        self.full_brain_colors = np.tile(np.array([0.35, 0.38, 0.45, 0.35], dtype=np.float32),
                                          (len(self.full_brain_positions), 1))

        pos_lookup = all_neurons.set_index("bodyId")[["x", "y", "z"]]
        has_pos = df["bodyId"].isin(pos_lookup.index).to_numpy()
        self.plot_idx = np.nonzero(has_pos)[0]
        positions = pos_lookup.loc[df["bodyId"].to_numpy()[self.plot_idx]].to_numpy(dtype=np.float32)
        self.brain_positions = positions.copy() - brain_center
        self.base_colors = np.array(
            [STAGE_COLORS.get(s, (0.5, 0.5, 0.5)) for s in df["stage"].to_numpy()[self.plot_idx]], dtype=np.float32
        )
        self.glow = np.zeros(self.net.n, dtype=np.float32)

        photoreceptor_mask = torch.tensor(df["type"].isin(PHOTORECEPTORS).to_numpy(), device=self.net.device)
        left_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "L").to_numpy(), device=self.net.device)
        right_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "R").to_numpy(), device=self.net.device)
        looming_mask = df["type"].isin(LOOMING_DETECTORS)
        left_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_L")).to_numpy(), device=self.net.device)
        right_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_R")).to_numpy(), device=self.net.device)
        self.injector = StimulusInjector(self.net, photoreceptor_mask, left_photo_mask, right_photo_mask,
                                          left_loom_mask, right_loom_mask)

        self.stage_groups = {}
        for stage in ["photoreceptor", "lamina", "medulla", "T4/T5", "HS/VS", "LC", "DNp01"]:
            self.stage_groups[stage] = np.nonzero((df["stage"] == stage).to_numpy())[0]

        self.model = build_model(drone_start_pos=START_POS)
        import mujoco
        self.data = mujoco.MjData(self.model)
        self.drone = Drone(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.trial_start = time.time()
        self.last_visible_time = None
        self.reached = False
        self.frame_times = []

        self._build_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)

        self.success_banner = QtWidgets.QLabel("")
        self.success_banner.setStyleSheet("color: #00e676; font-size: 22px; font-weight: bold; padding: 6px;")
        self.success_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.success_banner.hide()
        outer.addWidget(self.success_banner)

        top = QtWidgets.QHBoxLayout()
        brain_canvas = scene.SceneCanvas(bgcolor="#0b0f14", keys="interactive")
        view = brain_canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, distance=self.full_brain_positions.std() * 4)

        # Full-brain anatomical context (dim, static -- not simulated, just
        # showing real shape/scale) drawn first so the glowing pathway
        # scatter renders on top of it.
        self.full_brain_scatter = visuals.Markers()
        self.full_brain_scatter.set_data(self.full_brain_positions, face_color=self.full_brain_colors,
                                          size=2, edge_width=0)
        view.add(self.full_brain_scatter)

        self.scatter = visuals.Markers()
        self.scatter.set_data(self.brain_positions, face_color=self.base_colors, size=5, edge_width=0)
        view.add(self.scatter)
        brain_box = QtWidgets.QGroupBox(
            f"FLY BRAIN -- {len(self.full_brain_positions):,} neurons shown (real anatomy, dim) / "
            f"{len(self.brain_positions):,} simulated (bright, glows on activity)"
        )
        b_layout = QtWidgets.QVBoxLayout(brain_box)
        b_layout.addWidget(brain_canvas.native)
        top.addWidget(brain_box, stretch=1)

        drone_box = QtWidgets.QGroupBox("DRONE + FOOD (drag to rotate view)")
        d_layout = QtWidgets.QVBoxLayout(drone_box)
        self.drone_canvas, self.drone_visual = build_display_canvas()
        d_layout.addWidget(self.drone_canvas.native)
        top.addWidget(drone_box, stretch=1)
        outer.addLayout(top, stretch=3)

        bottom = QtWidgets.QHBoxLayout()

        task_box = QtWidgets.QGroupBox("FOOD SEARCH")
        t_layout = QtWidgets.QVBoxLayout(task_box)
        self.distance_label = QtWidgets.QLabel("FOOD DISTANCE: -- m")
        self.angle_label = QtWidgets.QLabel("TARGET ANGLE: -- deg")
        self.search_label = QtWidgets.QLabel("SEARCH TIME: -- s")
        for lbl in (self.distance_label, self.angle_label, self.search_label):
            lbl.setStyleSheet("color: white; font-size: 14px; font-family: Consolas, monospace;")
            t_layout.addWidget(lbl)
        bottom.addWidget(task_box)

        neural_box = QtWidgets.QGroupBox("NEURAL ACTIVITY")
        n_layout = QtWidgets.QVBoxLayout(neural_box)
        self.neural_bars = {}
        for stage in ["photoreceptor", "T4/T5", "HS/VS", "DNa02", "LC", "DNp01"]:
            bar = QtWidgets.QProgressBar(); bar.setFormat(f"{stage} %p%")
            self.neural_bars[stage] = bar
            n_layout.addWidget(bar)
        bottom.addWidget(neural_box)

        motor_box = QtWidgets.QGroupBox("MOTOR OUTPUT")
        m_layout = QtWidgets.QVBoxLayout(motor_box)
        self.left_turn_bar = QtWidgets.QProgressBar(); self.left_turn_bar.setFormat("LEFT TURN %p%")
        self.right_turn_bar = QtWidgets.QProgressBar(); self.right_turn_bar.setFormat("RIGHT TURN %p%")
        m_layout.addWidget(self.left_turn_bar)
        m_layout.addWidget(self.right_turn_bar)
        if not self.demo_mode:
            m_layout.addWidget(QtWidgets.QLabel(
                "RESCUE MODE: photoreceptor readout + explicit scan; T4/T5 temporal model pending"
            ))
        bottom.addWidget(motor_box)

        if not self.demo_mode:
            metrics_box = QtWidgets.QGroupBox("METRICS")
            metrics_layout = QtWidgets.QVBoxLayout(metrics_box)
            self.metrics_label = QtWidgets.QLabel("")
            self.metrics_label.setStyleSheet("font-family: Consolas, monospace; color: white;")
            metrics_layout.addWidget(self.metrics_label)
            bottom.addWidget(metrics_box)
        else:
            self.metrics_label = None

        outer.addLayout(bottom, stretch=1)
        self.setCentralWidget(central)

    def update_frame(self):
        if self.reached:
            return
        t0 = time.perf_counter()
        state = self.drone.get_state()
        left_drive, right_drive = encode_left_right_from_bearing(state.position, state.quaternion, FOOD_POS, TONIC_DRIVE)
        left_loom, right_loom = encode_looming_from_walls(state.position, state.quaternion, ARENA_HALF_SIZE)
        ext = self.injector.build(AMBIENT_DRIVE, TONIC_DRIVE, left_drive, right_drive, left_loom * 5.0, right_loom * 5.0)
        spikes = self.sim.step(ext).cpu().numpy()
        self.activity_ema = ACTIVITY_DECAY * self.activity_ema + (1 - ACTIVITY_DECAY) * spikes

        command = self.decoder.decode(self.activity_ema, forward=1.0)
        hover_command = MotorCommand(yaw=0.0, left_turn=0.0, right_turn=0.0)
        self.drone.apply_rotor_thrusts(mix_to_rotors(hover_command))
        self.drone.apply_planar_velocity_control(
            FORWARD_FORCE_N * command.forward, state.quaternion, state.linear_velocity
        )
        self.drone.apply_yaw_rate_control(command.yaw, state.angular_velocity)
        self.drone.step()
        self.drone.reset_if_unstable(safe_position=START_POS, max_height=3.0)
        self.step_count += 1

        self.glow = np.clip(self.glow * GLOW_DECAY + spikes.astype(np.float32), 0, 1)
        if len(self.plot_idx):
            glow_sub = self.glow[self.plot_idx][:, None]
            colors = self.base_colors * (0.25 + 0.75 * (1 - glow_sub)) + glow_sub * np.array([1.0, 1.0, 1.0])
            self.scatter.set_data(self.brain_positions, face_color=np.clip(colors, 0, 1), size=4, edge_width=0)

        new_state = self.drone.get_state()
        self.drone_visual.update(new_state.position, new_state.quaternion)

        distance, bearing_deg, visible = food_distance_and_bearing(new_state.position, new_state.quaternion, FOOD_POS)
        now = time.time()
        if visible:
            self.last_visible_time = now
        search_time = 0.0 if self.last_visible_time is None else now - self.last_visible_time
        self.distance_label.setText(f"FOOD DISTANCE: {distance:5.2f} m")
        self.angle_label.setText(f"TARGET ANGLE: {bearing_deg:+6.1f} deg  ({'visible' if visible else 'hidden'})")
        self.search_label.setText(f"SEARCH TIME: {search_time:4.1f} s")

        for stage, idx in self.stage_groups.items():
            if stage in self.neural_bars and len(idx):
                self.neural_bars[stage].setValue(int(100 * spikes[idx].mean()))
        self.left_turn_bar.setValue(min(100, int(command.left_turn * 20)))
        self.right_turn_bar.setValue(min(100, int(command.right_turn * 20)))

        if self.metrics_label is not None:
            dt = time.perf_counter() - t0
            self.frame_times.append(dt)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            fps = 1.0 / (sum(self.frame_times) / len(self.frame_times) + 1e-9)
            self.metrics_label.setText(f"FPS: {fps:.1f}\nSim step: {self.step_count}\nGain: {GAIN} (unfit, see docs)")

        if distance < REACHED_RADIUS_M:
            self.reached = True
            total_time = time.time() - self.trial_start
            self.success_banner.setText(f"FOOD REACHED -- {total_time:.1f}s, {self.step_count} steps")
            self.success_banner.show()


def main():
    app = QtWidgets.QApplication(sys.argv)

    intro_window = QtWidgets.QMainWindow()
    intro_window.setWindowTitle("Fly Brain Drone")
    intro_window.resize(900, 700)
    intro = IntroScreen()
    intro_window.setCentralWidget(intro)

    # Keeps the QMainWindow instance (and its QTimer) alive after on_start()
    # returns -- without this reference, nothing outside the Qt event loop
    # holds the FoodSearchDashboard object, and it would be garbage
    # collected (stopping its timer) once on_start() exits.
    state = {"dashboard": None}

    def on_start():
        state["dashboard"] = FoodSearchDashboard(demo_mode=True)
        state["dashboard"].show()
        intro_window.close()

    intro.start_requested.connect(on_start)
    intro_window.show()
    app.exec_()


if __name__ == "__main__":
    main()
