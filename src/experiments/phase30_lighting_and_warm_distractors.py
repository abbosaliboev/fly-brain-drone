"""Probe lighting robustness and same-category warm-colour distractors."""
from __future__ import annotations

import csv
from pathlib import Path

from src.experiments.phase25_r7_r8_food_search import run_trial
from src.vision.flyvis_backend import FlyVisBackend

START = (0.0, 3.0, 1.0)
FOOD = (3.0, 3.0, 1.0)
DISTRACTOR_POS = (-2.5, 2.0, 1.0)
RESULT = Path("experiments/results/lighting_and_warm_distractors.csv")
CASES = (
    ("light_025", 0.25, ()),
    ("light_050", 0.50, ()),
    ("light_150", 1.50, ()),
    ("red_distractor", 1.0, ((DISTRACTOR_POS, (1.0, 0.05, 0.02, 1.0)),)),
    ("amber_distractor", 1.0, ((DISTRACTOR_POS, (1.0, 0.55, 0.02, 1.0)),)),
)


def main() -> None:
    backend = FlyVisBackend(retinal_extent=5)
    rows = []
    for name, light_level, distractors in CASES:
        result = run_trial(
            START, FOOD, 180.0, verbose=False, backend=backend,
            distractors=distractors, light_level=light_level,
        )
        result.update(case=name, light_level=light_level)
        rows.append(result)
        print(
            f"{name:16s} success={result['success']} steps={result['steps']} "
            f"food_min={result['min_distance']:.3f} "
            f"distractor_min={result['min_distractor_distance']:.3f}"
        )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()
