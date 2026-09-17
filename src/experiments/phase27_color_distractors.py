"""Test R7/R8 food search against blue and green distractors."""
from __future__ import annotations

import csv
from pathlib import Path

from src.experiments.phase25_r7_r8_food_search import run_trial
from src.vision.flyvis_backend import FlyVisBackend

START = (0.0, 3.0, 1.0)
FOOD = (3.0, 3.0, 1.0)
DISTRACTOR_POS = (-2.5, 2.0, 1.0)
CASES = (
    ("none", ()),
    ("blue", ((DISTRACTOR_POS, (0.05, 0.1, 1.0, 1.0)),)),
    ("green", ((DISTRACTOR_POS, (0.05, 1.0, 0.05, 1.0)),)),
)
RESULT = Path("experiments/results/r7_r8_distractors.csv")


def main() -> None:
    rows = []
    backend = FlyVisBackend(retinal_extent=5)
    for name, distractors in CASES:
        result = run_trial(
            START, FOOD, 180.0, verbose=False, distractors=distractors,
            backend=backend,
        )
        result["distractor"] = name
        rows.append(result)
        print(
            f"{name:5s} success={result['success']} steps={result['steps']} "
            f"food_min={result['min_distance']:.3f} "
            f"distractor_min={result['min_distractor_distance']:.3f}"
        )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()
