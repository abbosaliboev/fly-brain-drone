"""UI/referee smoke using real MuJoCo; FlyVis inference deliberately stubbed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization.flyvis_dashboard import FlyVisDashboard, QtWidgets
import mujoco
from unittest.mock import patch


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with patch.object(FlyVisDashboard, '_initialize_backend', lambda self: None):
        window = FlyVisDashboard()
        window.timer.stop()
        window.backend_ready = True
        window._neural_update = lambda frame: None
        try:
            original_columns = [window.centralWidget().layout().columnMinimumWidth(i) for i in range(3)]
            original_distance = window.observer_camera.distance
            window.orbit_world_camera(20, -10)
            window.zoom_world_camera(120)
            assert window.observer_camera.distance < original_distance
            assert window.observer_camera.azimuth != 132
            window.difficulty_select.setCurrentText('HARD')
            assert [window.centralWidget().layout().columnMinimumWidth(i) for i in range(3)] == original_columns
            scrolls = window.findChildren(QtWidgets.QScrollArea)
            assert scrolls and scrolls[0].widgetResizable()
            for mode in ('HYBRID', 'NEURAL ONLY', 'NEURAL READOUT'):
                window.mode_select.setCurrentText(mode)
                window.new_round()
                window.data.qpos[:3] = (2.23, 3, 1)
                window.data.qpos[3:7] = (1, 0, 0, 0)
                window.data.qvel[:] = 0
                mujoco.mj_forward(window.model, window.data)
                window.toggle_running()
                for _ in range(6):
                    window._tick()
                assert window.reached, mode
                assert 'APPLE FOUND!' in window.result_banner.text(), mode
                assert 'game seconds' in window.result_banner.text(), mode
                assert window.history_table.rowCount() == 1, mode
            for speed in (1, 2, 5):
                window.new_round()
                window.data.qpos[:3] = (0, 0, 1)
                window.data.qvel[:] = 0
                mujoco.mj_forward(window.model, window.data)
                window.speed_select.setCurrentText(f'{speed}x')
                window.toggle_running()
                window._tick()
                assert abs(window.game.sim_time - speed * .015) < 1e-9
                window.toggle_running()
                before = window.game.sim_time
                window._tick()
                assert window.game.sim_time == before
            window._render_world()
            destination = Path('experiments/results/dashboard_lab_preview.png')
            destination.parent.mkdir(parents=True, exist_ok=True)
            window.world_camera.pixmap().save(str(destination))
            print('PASS: visible arrival banner in three modes; 1x/2x/5x physics time; pause. Neural inference was stubbed.')
        finally:
            window.close()


if __name__ == '__main__':
    main()
