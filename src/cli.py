"""Single command-line entry point for the current reproducible workflows."""
from __future__ import annotations

import argparse
import compileall
import importlib.util
import os
import sys
import unittest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Run the Fruit-Fly Drone search and validation workflows.",
    )
    parser.add_argument(
        "command",
        choices=("search", "demo", "dashboard", "hues", "robustness", "verify"),
        help=(
            "search: one closed-loop trial; demo: live MuJoCo viewer; "
            "dashboard: interactive visual dashboard; "
            "hues: warm-food benchmark; "
            "robustness: lighting/distractor benchmark; verify: compile and test"
        ),
    )
    return parser


def _require_flyvis() -> None:
    # Do not import FlyVis here: FlyVis resolves its data root at import time,
    # while FlyVisBackend deliberately sets FLYVIS_ROOT_DIR immediately before
    # importing it. An eager availability check would cache the wrong root.
    if importlib.util.find_spec("flyvis") is None:
        raise SystemExit(
            "This command requires the FlyVis environment. Run it with:\n"
            r"  .\.flyvis-venv\Scripts\python.exe -m src.cli <command>"
        )


def verify() -> int:
    compiled = compileall.compile_dir("src", quiet=1)
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if compiled and result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    command = build_parser().parse_args(argv).command
    if command == "verify":
        return verify()

    _require_flyvis()
    if command == "dashboard":
        # Small online frames do not need large BLAS/OpenMP worker pools.
        # Set defaults before importing numpy/torch; respect user overrides.
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        from src.visualization.flyvis_dashboard import main as run_dashboard

        return run_dashboard()
    if command in ("search", "demo"):
        from src.experiments.phase25_r7_r8_food_search import run_trial

        demo = command == "demo"
        result = run_trial(
            show_viewer=demo,
            viewer_step_delay=0.012 if demo else 0.0,
            hold_viewer_on_success=demo,
        )
        return 0 if result["success"] or result.get("stopped_by_user") else 1
    if command == "hues":
        from src.experiments.phase29_food_hues import main as run_hues

        run_hues()
        return 0

    from src.experiments.phase30_lighting_and_warm_distractors import main as run_robustness

    run_robustness()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
