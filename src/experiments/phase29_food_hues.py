"""Closed-loop robustness benchmark across nearby warm food colours."""
from __future__ import annotations

import csv
from pathlib import Path

from src.experiments.phase25_r7_r8_food_search import run_trial
from src.vision.flyvis_backend import FlyVisBackend

START = (0.0, 3.0, 1.0)
FOOD = (3.0, 3.0, 1.0)
CASES = (
    ("orange", (1.0, 0.35, 0.02, 1.0)),
    ("red", (1.0, 0.05, 0.02, 1.0)),
    ("amber", (1.0, 0.55, 0.02, 1.0)),
    ("dim_orange", (0.55, 0.18, 0.01, 1.0)),
)
RESULT = Path("experiments/results/food_hue_robustness.csv")


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    rows = []
    for name, rgba in CASES:
        result = run_trial(
            START, FOOD, 180.0, verbose=False, backend=backend,
            food_rgba=rgba,
        )
        result.update(food_hue=name, food_rgba=rgba)
        rows.append(result)
        print(
            f"{name:10s} success={result['success']} steps={result['steps']} "
            f"food_min={result['min_distance']:.3f}"
        )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()
