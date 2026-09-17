"""Fit and held-out-test a tiny food readout on FlyVis neural activity."""
from __future__ import annotations

import numpy as np
import mujoco
from pathlib import Path

from src.environment.world import build_model
from src.vision.encoder import detect_food_in_frame
from src.vision.flyvis_backend import FlyVisBackend

READOUT_PATH = Path("data/cache/flyvis_food_readout.npz")


def ridge_dual(x: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    mean = x.mean(0, keepdims=True)
    # Near-constant neurons otherwise create enormous coefficients and
    # catastrophic online extrapolation after a different initial yaw.
    scale = np.maximum(x.std(0, keepdims=True), 1e-2)
    z = (x - mean) / scale
    z = np.column_stack([z, np.ones(len(z))])
    if len(z) <= z.shape[1]:
        weights = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y)
    else:
        weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ y)
    return mean, scale, weights


def predict(model, x: np.ndarray) -> np.ndarray:
    mean, scale, weights = model
    z = (x - mean) / scale
    return np.column_stack([z, np.ones(len(z))]) @ weights


def collect_sweep(backend: FlyVisBackend, food_position,
                  start_yaw: float = 0.0,
                  drone_position=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    model = build_model(drone_start_pos=drone_position, food_position=food_position)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=120, width=160)
    backend.reset_stream()
    features, labels = [], []
    for yaw in (np.arange(0.0, 360.0, 5.0) + start_yaw) % 360.0:
        half = np.radians(yaw) / 2
        data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        backend.infer_frame(frame)
        error, confidence = detect_food_in_frame(frame)
        features.append(backend.activity_features())
        labels.append([error, float(confidence > 0.0)])
    renderer.close()
    return np.asarray(features), np.asarray(labels)


def metrics(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    visible = truth[:, 1] > 0.5
    visibility_accuracy = float(np.mean((estimate[:, 1] > 0.5) == visible))
    bearing_mae = float(np.mean(np.abs(estimate[visible, 0] - truth[visible, 0])))
    sign_accuracy = float(np.mean(
        np.sign(estimate[visible, 0]) == np.sign(truth[visible, 0])
    ))
    return visibility_accuracy, bearing_mae, sign_accuracy


def main() -> None:
    model = build_model(drone_start_pos=(0.0, 3.0, 1.0))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=120, width=160)
    backend = FlyVisBackend(retinal_extent=5)
    backend.reset_stream()

    features, labels = [], []
    yaws = np.arange(0.0, 360.0, 5.0)
    for yaw in yaws:
        half = np.radians(yaw) / 2
        data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="drone_eye")
        frame = renderer.render()
        backend.infer_frame(frame)
        error, confidence = detect_food_in_frame(frame)
        features.append(backend.activity_features())
        labels.append([error, float(confidence > 0.0)])
    x, y = np.asarray(features), np.asarray(labels)

    # Alternating angular samples prevent evaluating on exact training frames.
    train = np.arange(len(yaws)) % 2 == 0
    test = ~train
    learned = ridge_dual(x[train], y[train])
    estimate = predict(learned, x[test])
    visibility_accuracy, bearing_mae, sign_accuracy = metrics(estimate, y[test])
    print(f"train={train.sum()} test={test.sum()} features={x.shape[1]}")
    print(f"visibility_accuracy={visibility_accuracy:.3f}")
    print(f"visible_bearing_mae={bearing_mae:.3f}")
    print(f"visible_sign_accuracy={sign_accuracy:.3f}")
    renderer.close()

    # Harder split: train on two food locations, evaluate on a third unseen
    # location. All labels are used only to fit/score the readout.
    scenarios = (
        ((0.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
        ((-1.5, 1.0, 1.0), (2.0, 1.5, 1.0)),
        ((1.0, -1.5, 1.0), (-2.0, 2.0, 1.0)),
        ((-2.0, -1.0, 1.0), (1.5, 2.5, 1.0)),
        ((1.5, 1.0, 1.0), (-2.5, -1.5, 1.0)),
        ((0.5, 2.0, 1.0), (-2.5, 2.8, 1.0)),
    )
    train_sets = [
        collect_sweep(backend, food, start_yaw, drone)
        for drone, food in scenarios
        for start_yaw in (0.0, 180.0)
    ]
    train_x = np.concatenate([item[0] for item in train_sets])
    train_y = np.concatenate([item[1] for item in train_sets])
    test_x, test_y = collect_sweep(
        backend, (3.0, 3.0, 1.0), 180.0, (0.0, 3.0, 1.0)
    )
    position_model = ridge_dual(train_x, train_y)
    position_estimate = predict(position_model, test_x)
    visibility_accuracy, bearing_mae, sign_accuracy = metrics(position_estimate, test_y)
    print("position-held-out:")
    print(f"visibility_accuracy={visibility_accuracy:.3f}")
    print(f"visible_bearing_mae={bearing_mae:.3f}")
    print(f"visible_sign_accuracy={sign_accuracy:.3f}")
    READOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        READOUT_PATH, mean=position_model[0], scale=position_model[1],
        weights=position_model[2],
    )
    print(f"saved_readout={READOUT_PATH}")


if __name__ == "__main__":
    main()
