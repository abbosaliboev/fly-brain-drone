"""Paired FlyVis ablation across different food/start locations."""
from __future__ import annotations

import csv
from pathlib import Path

from src.experiments.phase17_flyvis_camera_search import run_trial
from src.vision.flyvis_backend import FlyVisBackend

CASES = (
    ((0.0, 3.0, 1.0), (3.0, 3.0, 1.0), 180.0),
    ((0.0, 0.0, 1.0), (-2.5, 2.5, 1.0), 0.0),
    ((-2.0, -2.0, 1.0), (2.5, -1.5, 1.0), 180.0),
)
RESULT = Path("experiments/results/flyvis_position_generalization.csv")


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    rows = []
    for case_id, (start, food, yaw) in enumerate(CASES):
        for condition in ("camera_only", "camera_flyvis"):
            result = run_trial(
                backend if condition == "camera_flyvis" else None,
                start_yaw=yaw, start_pos=start, food_pos=food, verbose=False,
            )
            result.update(condition=condition, case_id=case_id,
                          start_pos=start, food_pos=food)
            rows.append(result)
            print(
                f"case={case_id} {condition:13s} success={result['success']} "
                f"steps={result['steps']} distance={result['distance']:.3f}"
            )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()
