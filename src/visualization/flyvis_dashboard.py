"""Interactive dashboard for the rendered-camera FlyVis food-search controller."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from src.control.motor_decoder import MotorCommand
from src.control.flight_behavior import FlySearchBehavior
from src.control.apple_hunt import AppleHunt, safe_apple_position
from src.visualization.arrival import visible_apple
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.environment.world import ARENA_HALF_SIZE, FOOD_POSITION, build_model
from src.vision.flyvis_backend import FlyVisBackend, FlyVisMotion


START = np.asarray((0.0, 3.0, 1.0), dtype=np.float64)
INITIAL_FOOD = np.asarray(FOOD_POSITION, dtype=np.float64)
NEURAL_INTERVAL = 20
PHYSICS_STEPS_PER_TICK = 3

APP_STYLE = """
QMainWindow, QWidget { background-color: #07111f; color: #dce8f5; }
QGroupBox {
    background-color: #0d1a2b; border: 1px solid #223754;
    border-radius: 12px; margin-top: 12px; padding: 12px 8px 8px 8px;
    font-weight: 600; color: #9fc5e8;
}
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 7px; }
QPushButton {
    background-color: #147ca6; color: white; border: none; border-radius: 8px;
    padding: 10px 18px; font-weight: 700; min-height: 20px;
}
QPushButton:hover { background-color: #1b9acb; }
QPushButton:pressed { background-color: #0d6386; }
QPushButton:disabled { background-color: #26384a; color: #718497; }
QProgressBar {
    background-color: #08121e; border: 1px solid #263b54; border-radius: 7px;
    min-height: 22px; text-align: center; color: white;
}
QProgressBar::chunk { border-radius: 6px; background-color: #18a7d3; }
QTabWidget::pane { border: 1px solid #223754; border-radius: 8px; }
QTabBar::tab {
    background: #0a1524; color: #8298ad; padding: 8px 18px;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
}
QTabBar::tab:selected { background: #17304b; color: #eef8ff; }
QLabel { background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background:#08121e; width:9px; margin:0; border-radius:4px;
}
QScrollBar::handle:vertical { background:#29445f; min-height:28px; border-radius:4px; }
"""


def world_from_arena_pixel(
    x: float, y: float, width: float, height: float,
    arena_half_size: float = ARENA_HALF_SIZE,
) -> tuple[float, float]:
    """Map a dashboard click to MuJoCo x/y coordinates."""
    margin = 18.0
    usable_w = max(width - 2 * margin, 1.0)
    usable_h = max(height - 2 * margin, 1.0)
    world_x = ((x - margin) / usable_w * 2.0 - 1.0) * arena_half_size
    world_y = (1.0 - (y - margin) / usable_h * 2.0) * arena_half_size
    limit = arena_half_size - 0.45
    return float(np.clip(world_x, -limit, limit)), float(np.clip(world_y, -limit, limit))


class OrbitViewLabel(QtWidgets.QLabel):
    """Mouse-controlled observer surface for the MuJoCo 3D lab camera."""

    orbit_changed = QtCore.pyqtSignal(float, float)
    zoom_changed = QtCore.pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        self._last_mouse: QtCore.QPoint | None = None
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setToolTip("Drag to orbit · mouse wheel to zoom")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._last_mouse = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last_mouse is None:
            return
        delta = event.pos() - self._last_mouse
        self._last_mouse = event.pos()
        self.orbit_changed.emit(float(delta.x()), float(delta.y()))

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:
        self._last_mouse = None
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.zoom_changed.emit(float(event.angleDelta().y()))
        event.accept()


class ArenaWidget(QtWidgets.QWidget):
    food_moved = QtCore.pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 280)
        self.food = INITIAL_FOOD.copy()
        self.drone = START.copy()
        self.heading_deg = 0.0
        self.success_at = None
        self.difficulty = "NORMAL"
        self.trail: list[np.ndarray] = []
        self.setCursor(QtCore.Qt.CrossCursor)

    def _screen(self, position: np.ndarray) -> QtCore.QPointF:
        margin = 18.0
        x = margin + (position[0] / ARENA_HALF_SIZE + 1.0) * 0.5 * (self.width() - 2 * margin)
        y = margin + (1.0 - (position[1] / ARENA_HALF_SIZE + 1.0) * 0.5) * (self.height() - 2 * margin)
        return QtCore.QPointF(float(x), float(y))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            x, y = world_from_arena_pixel(
                event.position().x() if hasattr(event, "position") else event.x(),
                event.position().y() if hasattr(event, "position") else event.y(),
                self.width(), self.height(),
            )
            self.food_moved.emit(x, y)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        gradient = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QtGui.QColor("#0d1b2a"))
        gradient.setColorAt(1.0, QtGui.QColor("#071019"))
        painter.fillRect(self.rect(), gradient)
        border = self.rect().adjusted(18, 18, -18, -18)
        painter.setPen(QtGui.QPen(QtGui.QColor("#29445f"), 2))
        painter.drawRect(border)
        painter.setPen(QtGui.QPen(QtGui.QColor(35, 61, 82, 110), 1))
        for index in range(1, 8):
            x = border.left() + index * border.width() / 8
            y = border.top() + index * border.height() / 8
            painter.drawLine(QtCore.QPointF(x, border.top()), QtCore.QPointF(x, border.bottom()))
            painter.drawLine(QtCore.QPointF(border.left(), y), QtCore.QPointF(border.right(), y))
        painter.setPen(QtGui.QColor("#90a4ae"))
        painter.drawText(28, 38, "CHALLENGE ARENA — click to place the apple")

        # Challenge geometry mirrors the MuJoCo room so routes and occlusions
        # are legible from the game map, not surprising invisible barriers.
        painter.setOpacity(0.0 if self.difficulty == "EASY" else 1.0)
        painter.setPen(QtGui.QPen(QtGui.QColor("#52677a"), 2))
        painter.setBrush(QtGui.QColor("#1a2936"))
        for x, y, rx, ry in ((-2.55, 2.25, 0.65, 0.55), (2.45, -2.20, 0.75, 0.62)):
            center = self._screen(np.asarray((x, y, 1.0)))
            sx = rx / ARENA_HALF_SIZE * border.width() * 0.5
            sy = ry / ARENA_HALF_SIZE * border.height() * 0.5
            painter.drawRoundedRect(QtCore.QRectF(center.x() - sx, center.y() - sy, 2*sx, 2*sy), 6, 6)
        center = self._screen(np.asarray((0.0, 0.0, 1.0)))
        radius = 0.72 / ARENA_HALF_SIZE * border.width() * 0.5
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(QtGui.QColor("#607487"))
        for x, y in ((-2.25, -0.70), (1.85, 1.20)):
            point = self._screen(np.asarray((x, y, 1.0)))
            painter.drawEllipse(point, 6, 6)
        arch_a = self._screen(np.asarray((-1.45, -2.35, 1.0)))
        arch_b = self._screen(np.asarray((0.15, -2.35, 1.0)))
        painter.setPen(QtGui.QPen(QtGui.QColor("#7890a5"), 5))
        painter.drawLine(arch_a, arch_b)
        if self.difficulty == "HARD":
            for x, y in ((-1, 1), (1, -1)):
                painter.drawEllipse(self._screen(np.array([x, y, 1])), 8, 8)
            painter.setBrush(QtGui.QColor("#da8035"))
            for x, y in ((-2.5, -2.7), (2.7, 0)):
                painter.drawEllipse(self._screen(np.array([x, y, 1])), 6, 6)
        painter.setOpacity(1.0)

        if len(self.trail) > 1:
            painter.setPen(QtGui.QPen(QtGui.QColor("#62e6bd") if self.success_at is not None else QtGui.QColor(80, 180, 255, 130), 2))
            painter.drawPolyline(QtGui.QPolygonF([self._screen(p) for p in self.trail]))

        food_point = self._screen(self.food)
        painter.setBrush(QtGui.QColor(255, 72, 45, 45))
        painter.drawEllipse(food_point, 24, 24)
        painter.setBrush(QtGui.QColor("#ef5038"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(food_point + QtCore.QPointF(-4, 1), 9, 11)
        painter.drawEllipse(food_point + QtCore.QPointF(4, 1), 9, 11)
        painter.setPen(QtGui.QPen(QtGui.QColor("#6d4325"), 3))
        painter.drawLine(food_point + QtCore.QPointF(0, -10), food_point + QtCore.QPointF(2, -16))
        painter.setBrush(QtGui.QColor("#58b66b"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(food_point + QtCore.QPointF(7, -14), 6, 3)
        if self.success_at is not None:
            phase = min((time.perf_counter() - self.success_at) / 2.0, 1.0)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(80, 240, 160, int(255*(1-phase))), 3))
            painter.drawEllipse(food_point, 20 + phase*70, 20 + phase*70)
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                point = food_point + QtCore.QPointF(np.cos(angle)*phase*80, np.sin(angle)*phase*80)
                painter.drawEllipse(point, 2, 2)

        drone_point = self._screen(self.drone)
        painter.save()
        painter.translate(drone_point)
        painter.rotate(-self.heading_deg)
        painter.setBrush(QtGui.QColor("#d99a32"))
        painter.drawEllipse(QtCore.QPointF(0, 0), 10, 6)
        painter.setBrush(QtGui.QColor(180, 225, 245, 170))
        painter.drawEllipse(QtCore.QPointF(-2, -7), 8, 3)
        painter.drawEllipse(QtCore.QPointF(-2, 7), 8, 3)
        painter.restore()
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(drone_point + QtCore.QPointF(10, -8), "fly")
        painter.end()


class HexEyeWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(260)
        self._u = self._v = self._drives = None

    def set_snapshot(self, u: np.ndarray, v: np.ndarray, drives: np.ndarray) -> None:
        self._u, self._v, self._drives = u, v, drives
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#0b0f14"))
        painter.setPen(QtGui.QColor("#b0bec5"))
        painter.drawText(10, 20, "FLY EYE — luminance / R7-blue / R8-green")
        if self._drives is None:
            painter.drawText(10, 45, "Waiting for first neural frame…")
            painter.end()
            return
        scale = min(self.width() / 14.0, (self.height() - 35) / 12.0)
        cx, cy = self.width() * 0.5, self.height() * 0.55
        for u, v, drive in zip(self._u, self._v, self._drives):
            px = cx + scale * (v + 0.5 * u)
            py = cy - scale * 0.86 * u
            lum, blue, green = np.clip(drive, 0.0, 1.0)
            colour = QtGui.QColor.fromRgbF(float(lum), float(green), float(blue))
            painter.setBrush(colour)
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1))
            painter.drawEllipse(QtCore.QPointF(px, py), scale * 0.38, scale * 0.38)
        painter.end()


class BrainActivityWidget(QtWidgets.QWidget):
    """Schematic optic-lobe map driven by actual FlyVis type activity."""

    STAGES = (
        ("RETINA", ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")),
        ("LAMINA", ("L1", "L2", "L3", "L4", "L5")),
        ("MEDULLA", ("Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm9")),
        ("MOTION", ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")),
    )
    CELL_TYPES = (
        "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
        "L1", "L2", "L3", "L4", "L5",
        "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm9",
        "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    )

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(265)
        self.values = {name: 0.0 for name in self.CELL_TYPES}
        self.stage_scales = {stage: 1e-5 for stage, _cells in self.STAGES}

    def set_activity(self, values: dict[str, float]) -> None:
        self.values.update(values)
        for stage, cells in self.STAGES:
            peak = max((self.values[cell] for cell in cells), default=0.0)
            self.stage_scales[stage] = max(
                peak, self.stage_scales[stage] * 0.94, 1e-5
            )
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#090d12"))
        painter.setPen(QtGui.QColor("#cfd8dc"))
        painter.drawText(
            12, 20,
            "FLYVIS OPTIC-LOBE ACTIVITY — brightness = actual |Δ activity|",
        )
        stage_width = self.width() / len(self.STAGES)
        top, bottom = 48.0, self.height() - 18.0

        # The lines show the processing direction, not individual synapses.
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 90, 105), 2))
        for index in range(len(self.STAGES) - 1):
            x1 = (index + 0.72) * stage_width
            x2 = (index + 1.28) * stage_width
            painter.drawLine(QtCore.QPointF(x1, (top + bottom) / 2),
                             QtCore.QPointF(x2, (top + bottom) / 2))

        for stage_index, (stage, cells) in enumerate(self.STAGES):
            center_x = (stage_index + 0.5) * stage_width
            painter.setPen(QtGui.QColor("#90a4ae"))
            painter.drawText(
                QtCore.QRectF(stage_index * stage_width, 27, stage_width, 18),
                QtCore.Qt.AlignCenter, stage,
            )
            spacing = (bottom - top) / max(len(cells), 1)
            for row, cell_type in enumerate(cells):
                y = top + (row + 0.5) * spacing
                normalized = float(np.clip(
                    self.values[cell_type] / self.stage_scales[stage], 0.0, 1.0
                ))
                colour = QtGui.QColor.fromRgbF(
                    0.12 + 0.88 * normalized,
                    0.18 + 0.68 * normalized,
                    0.25 - 0.18 * normalized,
                )
                radius = 6.0 + 7.0 * normalized
                painter.setBrush(colour)
                painter.setPen(QtGui.QPen(QtGui.QColor(210, 220, 225, 150), 1))
                painter.drawEllipse(QtCore.QPointF(center_x - 20, y), radius, radius)
                painter.setPen(QtGui.QColor("#eceff1"))
                painter.drawText(QtCore.QPointF(center_x, y + 4), cell_type)
        painter.end()


class MaleCNSBrainWidget(QtWidgets.QWidget):
    """Interactive software-rendered MaleCNS soma cloud with activity overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        cache = Path(__file__).resolve().parents[2] / "data" / "cache" / "neurons.parquet"
        frame = pd.read_parquet(cache, columns=["type", "x", "y", "z"])
        # Parquet-backed columns may expose a read-only NumPy view.
        positions = frame[["x", "y", "z"]].to_numpy(dtype=np.float32).copy()
        positions -= positions.mean(axis=0)
        positions /= max(float(np.std(positions)), 1.0)
        types = frame["type"].fillna("").astype(str).to_numpy()

        # A deterministic display subset keeps software rotation responsive;
        # every tracked visual type remains present in the active overlay.
        stride = max(1, len(positions) // 45000)
        self.context_positions = positions[::stride]
        self.active_positions: list[np.ndarray] = []
        self.active_keys: list[str] = []
        for male_type in np.unique(types):
            key = self._flyvis_key(male_type)
            if key is None:
                continue
            selected = positions[types == male_type]
            if len(selected):
                self.active_positions.append(selected)
                self.active_keys.extend([key] * len(selected))
        self.active_positions_array = (
            np.concatenate(self.active_positions, axis=0)
            if self.active_positions else np.empty((0, 3), dtype=np.float32)
        )
        self.active_keys_array = np.asarray(self.active_keys, dtype=object)
        self.activity = {name: 0.0 for name in BrainActivityWidget.CELL_TYPES}
        self.activity["R1-6"] = 0.0
        self.scale = 1e-5
        self.yaw = -0.45
        self.pitch = 0.15
        self.last_mouse: QtCore.QPoint | None = None

    @staticmethod
    def _flyvis_key(male_type: str) -> str | None:
        if male_type == "R1-6":
            return "R1-6"
        if male_type.startswith("R7"):
            return "R7"
        if male_type.startswith("R8"):
            return "R8"
        if male_type in BrainActivityWidget.CELL_TYPES:
            return male_type
        return None

    def set_activity(self, values: dict[str, float]) -> None:
        self.activity.update(values)
        self.activity["R1-6"] = float(np.mean([
            values.get(name, 0.0) for name in ("R1", "R2", "R3", "R4", "R5", "R6")
        ]))
        peak = max(self.activity.values(), default=0.0)
        self.scale = max(peak, self.scale * 0.95, 1e-5)
        self.update()

    def _rotation(self) -> np.ndarray:
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        yaw = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float32)
        pitch = np.asarray(((1, 0, 0), (0, cp, -sp), (0, sp, cp)), dtype=np.float32)
        return yaw @ pitch

    def _project(
        self, positions: np.ndarray, span: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        rotated = positions @ self._rotation().T
        if span is None:
            span = max(float(np.percentile(np.abs(rotated[:, :2]), 99.5)), 1e-5)
        scale = 0.43 * min(self.width(), self.height()) / span
        x = np.rint(self.width() * 0.5 + rotated[:, 0] * scale).astype(np.int32)
        y = np.rint(self.height() * 0.54 - rotated[:, 1] * scale).astype(np.int32)
        return x, y, span

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        width, height = max(self.width(), 1), max(self.height(), 1)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (8, 13, 20)
        x, y, span = self._project(self.context_positions)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        image[y[valid], x[valid]] = (48, 68, 92)

        if len(self.active_positions_array):
            ax, ay, _ = self._project(self.active_positions_array, span=span)
            intensities = np.asarray([
                self.activity.get(str(key), 0.0) / self.scale
                for key in self.active_keys_array
            ], dtype=np.float32)
            intensities = np.clip(intensities, 0.0, 1.0)
            colours = np.column_stack((
                100 + 155 * intensities,
                50 + 205 * intensities,
                25 + 45 * (1.0 - intensities),
            )).astype(np.uint8)
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                px, py = ax + dx, ay + dy
                valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                image[py[valid], px[valid]] = colours[valid]

        qimage = QtGui.QImage(
            image.data, width, height, image.strides[0], QtGui.QImage.Format_RGB888
        ).copy()
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, qimage)
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(12, 20, "MaleCNS v1.0 soma cloud — drag to rotate")
        painter.setPen(QtGui.QColor("#ffca70"))
        painter.drawText(
            12, 40,
            "Glow = FlyVis type activity mapped onto matching MaleCNS cell types (proxy)",
        )
        painter.setPen(QtGui.QColor("#90a4ae"))
        painter.drawText(
            12, height - 10,
            f"{len(self.context_positions):,} context points / "
            f"{len(self.active_positions_array):,} visual-type neurons",
        )
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.last_mouse = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.last_mouse is None:
            return
        delta = event.pos() - self.last_mouse
        self.last_mouse = event.pos()
        self.yaw += delta.x() * 0.01
        self.pitch = float(np.clip(self.pitch + delta.y() * 0.01, -1.4, 1.4))
        self.update()

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:
        self.last_mouse = None
        self.setCursor(QtCore.Qt.OpenHandCursor)


class FlyVisDashboard(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fruit-Fly Neural Lab — Interactive Apple Hunt")
        self.resize(1480, 900)
        self.setStyleSheet(APP_STYLE)

        self.model = build_model(
            START, 180.0, food_position=INITIAL_FOOD, achromatic_background=True,
            challenge_arena=True, game_assets=True,
        )
        self.data = mujoco.MjData(self.model)
        self.drone = Drone(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=240, width=320, max_geom=256)
        # Camera views are sequential; reuse one context and framebuffer.
        self.world_renderer = self.renderer
        self.observer_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.observer_camera)
        self.observer_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.observer_camera.lookat[:] = (0.0, 0.0, 0.65)
        self.observer_camera.distance = 8.2
        self.observer_camera.azimuth = 132.0
        self.observer_camera.elevation = -38.0
        self.food_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "food"
        )
        self.food_parts = {}
        for name in ("food", "food_lobe", "food_stem", "food_leaf"):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.food_parts[geom_id] = self.model.geom_pos[geom_id].copy() - INITIAL_FOOD
        self.food = INITIAL_FOOD.copy()
        self.backend = FlyVisBackend(retinal_extent=5)
        self.backend_ready = False
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.running = False
        self.reached = False
        self.error = self.confidence = self.motion_x = 0.0
        self.motion: FlyVisMotion | None = None
        self.flight_mode = "READY"
        self.nearest_obstacle = float("inf")
        self.flight_behavior = FlySearchBehavior()
        self.commanded_yaw = 0.0
        self.commanded_forward = 0.0
        self.game = AppleHunt()
        self.scene_defaults = (self.model.geom_contype.copy(), self.model.geom_conaffinity.copy(), self.model.geom_rgba.copy())
        self.neural_error = self.neural_confidence = 0.0
        for i in range(self.model.ngeom):
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("hard_"):
                self.model.geom_contype[i] = self.model.geom_conaffinity[i] = 0
                self.model.geom_rgba[i, 3] = 0
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(15)
        self._render_camera()
        self._render_world()
        self._update_metrics()
        self.start_button.setEnabled(False)
        self.status.setText("Loading FlyVis model… please wait")
        QtCore.QTimer.singleShot(50, self._initialize_backend)

    def _initialize_backend(self) -> None:
        try:
            self.backend.reset_stream()
        except Exception as exc:
            self.status.setText(f"FlyVis failed to load: {exc}")
            return
        self.backend_ready = True
        self.start_button.setEnabled(True)
        self.status.setText("Ready — place the apple, then press START")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(central)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(8, 2, 8, 2)
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("FRUIT-FLY NEURAL FLIGHT LAB")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#f4f9ff;")
        subtitle = QtWidgets.QLabel(
            "MaleCNS anatomy · FlyVis optic lobe · closed-loop MuJoCo behavior"
        )
        subtitle.setStyleSheet("font-size:12px; color:#7694ad;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch(1)
        system_status = QtWidgets.QLabel("●  SYSTEM ONLINE")
        system_status.setFixedSize(142, 30)
        system_status.setAlignment(QtCore.Qt.AlignCenter)
        system_status.setStyleSheet(
            "color:#62e6bd; border-bottom:2px solid #23665f; "
            "font-size:11px; font-weight:750; letter-spacing:1px;"
        )
        header_layout.addWidget(system_status, 0, QtCore.Qt.AlignVCenter)
        grid.addWidget(header, 0, 0, 1, 3)
        self.hud = QtWidgets.QLabel("APPLE HUNT")
        self.hud.setStyleSheet("font-size:16px; color:#62e6bd; padding:8px;")
        title_box.addWidget(self.hud)

        self.arena = ArenaWidget()
        self.arena.food_moved.connect(self.move_food)
        arena_box = QtWidgets.QGroupBox("INTERACTIVE FLIGHT ARENA")
        arena_layout = QtWidgets.QVBoxLayout(arena_box)
        self.world_camera = OrbitViewLabel()
        self.world_camera.setMinimumSize(320, 240)
        self.world_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.world_camera.orbit_changed.connect(self.orbit_world_camera)
        self.world_camera.zoom_changed.connect(self.zoom_world_camera)
        self.world_tabs = QtWidgets.QTabWidget()
        self.world_tabs.addTab(self.world_camera, "3D LAB / OBSERVER")
        self.world_tabs.addTab(self.arena, "MAP / PLACE APPLE")
        arena_layout.addWidget(self.world_tabs)
        self.result_banner = QtWidgets.QLabel()
        self.result_banner.setWordWrap(True)
        self.result_banner.setFixedHeight(92)
        self.result_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.result_banner.setStyleSheet("font-size:20px; color:#62e6bd;")
        arena_layout.addWidget(self.result_banner)
        self.history_table = QtWidgets.QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(["Round", "Mode", "Level", "Time", "Path", "Escapes", "Hits", "Score"])
        self.history_table.setFixedHeight(132)
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        arena_layout.addWidget(self.history_table)
        note = QtWidgets.QLabel("Game score is a presentation metric. Scientific evaluation uses controlled benchmark trials.")
        note.setWordWrap(True)
        arena_layout.addWidget(note)
        grid.addWidget(arena_box, 1, 0, 2, 1)

        camera_box = QtWidgets.QGroupBox("FLY POV — live compound-eye camera")
        camera_layout = QtWidgets.QVBoxLayout(camera_box)
        self.camera = QtWidgets.QLabel()
        self.camera.setMinimumSize(320, 240)
        self.camera.setAlignment(QtCore.Qt.AlignCenter)
        self.camera.setStyleSheet("background:#000;")
        camera_layout.addWidget(self.camera)
        grid.addWidget(camera_box, 1, 1)

        self.eye = HexEyeWidget()
        eye_box = QtWidgets.QGroupBox("RETINAL INPUT")
        eye_layout = QtWidgets.QVBoxLayout(eye_box)
        eye_layout.addWidget(self.eye)
        grid.addWidget(eye_box, 2, 1)

        controls = QtWidgets.QGroupBox("CONTROL + NEURAL TELEMETRY")
        controls_outer = QtWidgets.QVBoxLayout(controls)
        telemetry_scroll = QtWidgets.QScrollArea()
        telemetry_scroll.setWidgetResizable(True)
        telemetry_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        telemetry_content = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(telemetry_content)
        panel.setContentsMargins(3, 3, 6, 3)
        telemetry_scroll.setWidget(telemetry_content)
        controls_outer.addWidget(telemetry_scroll)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("START")
        self.start_button.clicked.connect(self.toggle_running)
        reset_button = QtWidgets.QPushButton("RESET FLY")
        reset_button.clicked.connect(self.reset_drone)
        buttons.addWidget(self.start_button)
        buttons.addWidget(reset_button)
        panel.addLayout(buttons)
        options = QtWidgets.QHBoxLayout()
        self.mode_select = QtWidgets.QComboBox()
        self.mode_select.addItems(["HYBRID", "ENGINEERED", "NEURAL READOUT", "NEURAL ONLY"])
        self.mode_select.setFixedWidth(135)
        self.mode_select.setToolTip("HYBRID: engineered search + neural motion damping. ENGINEERED: no neural motor input. NEURAL READOUT: neural contrast with engineered motor mapping and safety; not apple recognition. NEURAL ONLY: T4/T5 yaw only, no forward drive.")
        self.difficulty_select = QtWidgets.QComboBox()
        self.difficulty_select.addItems(["NORMAL", "EASY", "HARD"])
        self.difficulty_select.setFixedWidth(82)
        options.addWidget(self.mode_select)
        options.addWidget(self.difficulty_select)
        self.speed_select = QtWidgets.QComboBox()
        self.speed_select.addItems(["1x", "2x", "5x"])
        self.speed_select.setFixedWidth(54)
        self.speed_select.setToolTip("Simulation throughput multiplier. GAME time follows physics; REAL time is separate. Actual speed depends on hardware.")
        options.addWidget(self.speed_select)
        panel.addLayout(options)
        self.mode_select.currentTextChanged.connect(self.new_round)
        self.difficulty_select.currentTextChanged.connect(self.change_difficulty)
        actions = QtWidgets.QGridLayout()
        for index, (label, callback) in enumerate((("NEW ROUND", self.new_round), ("RANDOM APPLE", self.random_apple), ("CLEAR RESULTS", self.clear_results))):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            actions.addWidget(button, index // 2, index % 2)
        panel.addLayout(actions)
        self.status = QtWidgets.QLabel("Click arena to place the apple, then press START")
        self.status.setWordWrap(True)
        self.status.setFixedHeight(46)
        self.status.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        panel.addWidget(self.status)
        self.metrics = QtWidgets.QLabel()
        self.metrics.setStyleSheet("font-family: Consolas; font-size: 13px;")
        panel.addWidget(self.metrics)
        self.bars: dict[str, QtWidgets.QProgressBar] = {}
        for name in ("R1-R6 luminance", "R7 blue/UV", "R8 green", "T4/T5 motion", "food confidence"):
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat(name + "  %p%")
            self.bars[name] = bar
            panel.addWidget(bar)
        panel.addStretch(1)
        grid.addWidget(controls, 2, 2)

        self.brain_model = MaleCNSBrainWidget()
        self.brain = BrainActivityWidget()
        brain_box = QtWidgets.QGroupBox("BRAIN ACTIVITY")
        brain_layout = QtWidgets.QVBoxLayout(brain_box)
        brain_tabs = QtWidgets.QTabWidget()
        brain_tabs.addTab(self.brain_model, "3D MaleCNS brain model")
        brain_tabs.addTab(self.brain, "FlyVis pathway diagram")
        brain_layout.addWidget(brain_tabs)
        brain_box.setMinimumWidth(390)
        grid.addWidget(brain_box, 1, 2)

        grid.setColumnStretch(0, 4)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 4)
        grid.setColumnMinimumWidth(0, 390)
        grid.setColumnMinimumWidth(1, 360)
        grid.setColumnMinimumWidth(2, 390)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 2)
        self.setCentralWidget(central)

    @QtCore.pyqtSlot(float, float)
    def move_food(self, x: float, y: float) -> None:
        if not safe_apple_position(self.model, (x, y, 1.0)):
            self.status.setText("Choose an open location with room around the apple.")
            return
        self.food[:] = (x, y, 1.0)
        self._place_food_model()
        mujoco.mj_forward(self.model, self.data)
        self.arena.food = self.food.copy()
        self.reached = False
        self.status.setText(
            f"Apple moved to ({x:+.2f}, {y:+.2f}). Controller receives no coordinates."
        )
        self.arena.update()
        self._render_camera()
        self.new_round()

    def new_round(self, *_):
        self.running = self.reached = False
        self.game.new_round(self.mode_select.currentText(), self.difficulty_select.currentText())
        self.arena.difficulty = self.game.difficulty
        self.start_button.setText("START")
        self.result_banner.clear()
        self.statusBar().clearMessage()
        self.result_banner.setGraphicsEffect(None)
        self.arena.success_at = None
        self.arena.trail.clear()
        self.flight_behavior.reset()
        self.commanded_yaw = self.commanded_forward = 0.0
        self.error = self.confidence = self.motion_x = 0.0
        self.step_count = 0
        self.neural_error = self.neural_confidence = 0.0
        self.status.setText("New round ready. Press START." if self.game.mode != "NEURAL ONLY" else "NEURAL ONLY: motion-to-yaw experiment; no forward drive. Choose NEURAL READOUT for contrast-driven flight.")
        self.refresh_history()
        self._update_metrics()

    def refresh_history(self):
        entries = [r for r in self.game.history if r[1:3] == (self.game.mode, self.game.difficulty)]
        self.history_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            for col, value in enumerate(entry):
                self.history_table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{value:.2f}" if isinstance(value, float) else str(value)))

    def random_apple(self):
        rng = np.random.default_rng()
        for _ in range(1000):
            x, y = rng.uniform(-3.0, 3.0, 2)
            if safe_apple_position(self.model, (x, y, 1)) and np.linalg.norm(self.drone.get_state().position[:2] - (x, y)) > 1.5:
                self.move_food(float(x), float(y))
                return
        self.status.setText("No safe random position found. Try another level.")

    def change_difficulty(self, *_):
        self.running = False
        mujoco.mj_resetData(self.model, self.data)
        self._place_food_model()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("hard_"):
                enabled = self.difficulty_select.currentText() == "HARD"
                self.model.geom_contype[i] = self.scene_defaults[0][i] if enabled else 0
                self.model.geom_conaffinity[i] = self.scene_defaults[1][i] if enabled else 0
                self.model.geom_rgba[i, 3] = self.scene_defaults[2][i, 3] if enabled else 0
            if name.startswith(("pylon_", "arch_", "platform_", "island_", "beacon_")):
                enabled = self.difficulty_select.currentText() != "EASY"
                self.model.geom_contype[i] = self.scene_defaults[0][i] if enabled else 0
                self.model.geom_conaffinity[i] = self.scene_defaults[1][i] if enabled else 0
                self.model.geom_rgba[i, 3] = self.scene_defaults[2][i, 3] if enabled else 0
        mujoco.mj_forward(self.model, self.data)
        self.random_apple()

    def clear_results(self):
        self.game.history.clear()
        self.game.best.clear()
        self.history_table.setRowCount(0)
        self._update_metrics()

    def _place_food_model(self) -> None:
        """Move all visible apple components as one dashboard object."""
        for geom_id, offset in self.food_parts.items():
            self.model.geom_pos[geom_id] = self.food + offset

    def toggle_running(self) -> None:
        if not self.backend_ready or self.reached:
            return
        self.running = not self.running
        self.game.start() if self.running else self.game.pause()
        self.start_button.setText("PAUSE" if self.running else "START")
        self.status.setText("Searching from camera…" if self.running else "Paused")

    def reset_drone(self) -> None:
        was_running = self.running
        self.running = False
        mujoco.mj_resetData(self.model, self.data)
        self._place_food_model()
        mujoco.mj_forward(self.model, self.data)
        if self.backend_ready:
            self.backend.reset_stream()
        self.step_count = 0
        self.reached = False
        self.error = self.confidence = self.motion_x = 0.0
        self.flight_behavior.reset()
        self.commanded_yaw = self.commanded_forward = 0.0
        self.arena.trail.clear()
        self.running = was_running
        self.start_button.setText("PAUSE" if self.running else "START")
        self.status.setText("Fly reset. Searching…" if self.running else "Fly reset.")
        self._render_camera()
        self._update_metrics()
        self.new_round()
        self._render_world()

    def _render_camera(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="drone_eye")
        frame = self.renderer.render().copy()
        image = QtGui.QImage(
            frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.camera.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.camera.setPixmap(pixmap)
        return frame

    @QtCore.pyqtSlot(float, float)
    def orbit_world_camera(self, dx: float, dy: float) -> None:
        self.observer_camera.azimuth = (self.observer_camera.azimuth - dx * 0.45) % 360.0
        self.observer_camera.elevation = float(np.clip(
            self.observer_camera.elevation - dy * 0.35, -85.0, -8.0
        ))
        self._render_world()

    @QtCore.pyqtSlot(float)
    def zoom_world_camera(self, wheel_delta: float) -> None:
        factor = 0.86 if wheel_delta > 0 else 1.16
        self.observer_camera.distance = float(np.clip(
            self.observer_camera.distance * factor, 4.8, 16.0
        ))
        self._render_world()

    def _render_world(self):
        self.world_renderer.update_scene(self.data, camera=self.observer_camera)
        # Presentation colours affect only the spectator scene. The fly's
        # camera retains the neutral room used by its colour readout.
        for geom in self.world_renderer.scene.geoms[:self.world_renderer.scene.ngeom]:
            if geom.objtype != int(mujoco.mjtObj.mjOBJ_GEOM) or geom.objid < 0:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom.objid) or ""
            if name.startswith('wall_'):
                geom.rgba[:3] = (.16, .24, .32)
            elif name.startswith(('lab_light_', 'room_trim_', 'beacon_')):
                geom.rgba[:3] = (.12, .85, .95)
            elif name.startswith(('arch_', 'pylon_', 'hard_pillar')):
                geom.rgba[:3] = (.30, .48, .58)
            elif name.startswith('platform_'):
                geom.rgba[:3] = (.12, .25, .30)
        frame = self.world_renderer.render().copy()
        image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.world_camera.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.world_camera.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _neural_update(self, frame: np.ndarray) -> None:
        self.motion = self.backend.infer_color_frame(frame)
        self.motion_x = self.motion.horizontal
        self.error, self.confidence = self.backend.r1_r7_r8_orange_salience()
        if self.mode_select.currentText() == "NEURAL READOUT":
            self.neural_error, self.neural_confidence = self.backend.retinal_horizontal_salience()
        u, v, drives = self.backend.retinal_snapshot()
        self.eye.set_snapshot(u, v, drives)
        brain_activity = self.backend.activity_by_type(BrainActivityWidget.CELL_TYPES)
        self.brain.set_activity(brain_activity)
        self.brain_model.set_activity(brain_activity)
        means = drives.mean(axis=0)
        self.bars["R1-R6 luminance"].setValue(int(np.clip(means[0] * 100, 0, 100)))
        self.bars["R7 blue/UV"].setValue(int(np.clip(means[1] * 100, 0, 100)))
        self.bars["R8 green"].setValue(int(np.clip(means[2] * 100, 0, 100)))
        self.bars["T4/T5 motion"].setValue(int(np.clip(abs(self.motion_x) * 300, 0, 100)))
        self.bars["food confidence"].setValue(int(np.clip(self.confidence * 100, 0, 100)))

    def _physics_step(self) -> None:
        state = self.drone.get_state()
        ray_angles, ray_distances = self.drone.obstacle_rays()
        mode = self.mode_select.currentText()
        intent = self.flight_behavior.update(
            self.step_count, self.confidence, self.error, self.motion_x if mode == "HYBRID" else 0.0,
            ray_angles, ray_distances,
        )
        if mode == "NEURAL ONLY":
            from src.control.flight_behavior import FlightIntent
            intent = FlightIntent(float(np.clip(-1.35*self.motion_x, -1, 1)), 0.0, "NEURAL MOTION ONLY", intent.nearest_obstacle)
        elif mode == "NEURAL READOUT":
            from src.control.flight_behavior import FlightIntent
            # Contrast-driven neural activity, decoded by an explicit engineered
            # motor mapping. No apple identity or range is passed to this rule.
            if intent.mode != "LOOMING ESCAPE":
                yaw = np.clip(2*self.neural_error - 1.35*self.motion_x, -1, 1)
                drive = min(1.0, self.neural_confidence*8) * max(.15, 1-abs(self.neural_error))
                intent = FlightIntent(float(yaw), float(drive), "NEURAL CONTRAST + MOTOR READOUT", intent.nearest_obstacle)
        self.flight_mode = intent.mode
        self.nearest_obstacle = intent.nearest_obstacle
        self.drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
        yaw_alpha = 0.35 if intent.mode == "LOOMING ESCAPE" else 0.20
        self.commanded_yaw += yaw_alpha * (intent.yaw - self.commanded_yaw)
        if intent.forward_force < 0.0:
            self.commanded_forward = intent.forward_force
        else:
            self.commanded_forward += 0.16 * (intent.forward_force - self.commanded_forward)
        self.drone.apply_planar_velocity_control(
            self.commanded_forward, state.quaternion, state.linear_velocity,
            drag_gain=2.1,
        )
        self.drone.apply_yaw_rate_control(self.commanded_yaw, state.angular_velocity)
        self.drone.step()
        recovered = self.drone.reset_if_unstable(safe_position=START, max_height=3.0)
        contact = any(self.model.geom_bodyid[c.geom1] == self.drone.body_id or self.model.geom_bodyid[c.geom2] == self.drone.body_id for c in self.data.contact)
        self.game.physics(self.model.opt.timestep, 0 if recovered else np.linalg.norm(self.drone.get_state().position-state.position), intent.mode == "LOOMING ESCAPE", contact)
        if recovered:
            self.game.pause()
            self.running = False
            self.status.setText("RECOVERY: round interrupted; start a new round.")
        self.step_count += 1

    def _tick(self) -> None:
        self.arena.update()
        self._update_metrics()
        if not self.backend_ready or not self.running or self.reached:
            return
        for _ in range(PHYSICS_STEPS_PER_TICK * int(self.speed_select.currentText()[0])):
            if self.step_count % NEURAL_INTERVAL == 0:
                frame = self._render_camera()
                self._neural_update(frame)
            if self.step_count % 5 == 0:
                # Scoring uses rendered geometry IDs only as a referee. They
                # are never supplied to the flight controller.
                visible, referee_confidence = visible_apple(self.renderer, self.data, list(self.food_parts))
                result = self.game.observe(float(np.linalg.norm(self.drone.get_state().position-self.food)), referee_confidence, visible)
                if result is not None:
                    self.reached, self.running = True, False
                    self.start_button.setText("START")
                    self.arena.success_at = time.perf_counter()
                    self.status.setText("APPLE FOUND! Place the apple for the next round.")
                    glow = QtWidgets.QGraphicsDropShadowEffect(self.result_banner)
                    glow.setColor(QtGui.QColor("#62e6bd"))
                    glow.setBlurRadius(18)
                    glow.setOffset(0)
                    self.result_banner.setGraphicsEffect(glow)
                    self.result_banner.setText(f"APPLE FOUND!\nFound in {self.game.sim_time:.2f} game seconds\nReal time: {self.game.elapsed:.2f}s\nPath: {self.game.path:.2f} m | Escapes: {self.game.escapes}\nScore: {self.game.score:,}")
                    self.statusBar().showMessage(f"APPLE FOUND in {self.game.sim_time:.2f}s! Real: {self.game.elapsed:.2f}s", 0)
                    self.refresh_history()
                    break
            self._physics_step()
            if not self.running:
                break
        state = self.drone.get_state()
        self.arena.drone = state.position.copy()
        qw, qx, qy, qz = state.quaternion
        self.arena.heading_deg = float(np.degrees(np.arctan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        )))
        if not self.arena.trail or self.step_count % 12 == 0:
            self.arena.trail.append(state.position.copy())
            self.arena.trail = self.arena.trail[-500:]
        self.arena.update()
        self._render_camera()
        self._render_world()
        distance = float(np.linalg.norm(state.position - self.food))
        self._update_metrics(distance)

    def _update_metrics(self, distance: float | None = None) -> None:
        state = self.drone.get_state()
        if distance is None:
            distance = float(np.linalg.norm(state.position - self.food))
        populations = self.motion.populations if self.motion else {}
        t4t5 = "  ".join(f"{k}:{populations.get(k, 0.0):+.2f}" for k in ("T4a", "T4b", "T5a", "T5b"))
        self.metrics.setText(
            f"step: {self.step_count}\n"
            f"food distance: {distance:.3f} m\n"
            f"food bearing error: {self.error:+.3f}\n"
            f"food confidence: {self.confidence:.3f}\n"
            f"horizontal motion: {self.motion_x:+.4f}\n"
            f"flight mode: {self.flight_mode}\n"
            f"nearest obstacle: {self.nearest_obstacle:.2f} m\n"
            f"{t4t5}"
        )
        best = self.game.best.get((self.game.mode, self.game.difficulty))
        best_text = f"{best:.2f}s" if best is not None else "--"
        status = "FOUND" if self.reached else "SEARCHING" if self.running else "READY / PAUSED"
        self.hud.setText(f"APPLE HUNT | ROUND {self.game.round:02d} | GAME {self.game.sim_time:.2f}s | REAL {self.game.elapsed:.2f}s | {self.speed_select.currentText()}\nBEST {best_text} | DIST {distance:.2f}m | SCORE {self.game.score:,} | {self.game.mode} / {self.game.difficulty} | {status}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        self.renderer.close()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = FlyVisDashboard()
    except (ValueError, MemoryError, mujoco.FatalError) as exc:
        if not isinstance(exc, MemoryError) and 'memory' not in str(exc).lower():
            raise
        print("Dashboard could not allocate memory. Close unused applications or restart Windows, then retry. If this persists, check Windows virtual memory/page-file settings.", file=sys.stderr)
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
