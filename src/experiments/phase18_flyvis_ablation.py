"""Paired camera-only versus camera+FlyVis yaw-damping benchmark."""
from __future__ import annotations

import csv
from pathlib import Path

from src.experiments.phase17_flyvis_camera_search import run_trial
from src.vision.flyvis_backend import FlyVisBackend

YAW_STARTS = (135.0, 180.0, 225.0)
RESULT = Path("experiments/results/flyvis_ablation.csv")


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    rows = []
    for yaw in YAW_STARTS:
        for condition in ("camera_only", "camera_flyvis"):
            active_backend = backend if condition == "camera_flyvis" else None
            result = run_trial(active_backend, start_yaw=yaw, verbose=False)
            result["condition"] = condition
            rows.append(result)
            print(
                f"{condition:13s} yaw={yaw:5.1f} success={result['success']} "
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
