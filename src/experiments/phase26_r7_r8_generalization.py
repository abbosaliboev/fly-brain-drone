"""Generalization test for R7/R8-driven neural colour search."""
from __future__ import annotations

import csv
from pathlib import Path

from src.experiments.phase25_r7_r8_food_search import run_trial
from src.vision.flyvis_backend import FlyVisBackend

CASES = (
    ((0.0, 3.0, 1.0), (3.0, 3.0, 1.0), 180.0),
    ((0.0, 0.0, 1.0), (-2.5, 2.5, 1.0), 0.0),
    ((-2.0, -2.0, 1.0), (2.5, -1.5, 1.0), 180.0),
)
RESULT = Path("experiments/results/r7_r8_generalization.csv")


def main() -> None:
    rows = []
    backend = FlyVisBackend(retinal_extent=5)
    for case_id, (start, food, yaw) in enumerate(CASES):
        result = run_trial(start, food, yaw, verbose=False, backend=backend)
        result.update(case_id=case_id, start=start, food=food, start_yaw=yaw)
        rows.append(result)
        print(
            f"case={case_id} success={result['success']} steps={result['steps']} "
            f"distance={result['distance']:.3f}"
        )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()
