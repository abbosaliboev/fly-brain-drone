"""Phase 8-10: baseline-controller comparison + lesion experiments.

Compares four "brains" driving the identical body/physics/sensors:
  - connectome: the full verified pathway (Phase 6-7's NavigationDecoder)
  - connectome_lesion_{10,20,30}: same, with that % of neurons randomly
    silenced (see src/brain/simulator.py's lesion_mask)
  - rule_based: a hand-written formula using the SAME two sensory signals
    (bearing left/right drive, wall-looming left/right) the connectome
    receives -- a fair comparison (same senses, different "brain"), not a
    strawman
  - random: a smoothed random walk for yaw, same forward force

IMPORTANT, given docs/scientific_assumptions.md's calibration-attempt
finding: the connectome controller's yaw is not currently known to be
stimulus-driven at all (baseline-subtracted LEFT vs RIGHT effect at DNa02
was ~0 across every gain tested). So this experiment's real question is
"does the connectome controller behave differently from random," not an
assumed "the connectome wins" -- we report whatever the numbers show.

Runs headless (no viewer) for speed; saves results to
experiments/results/benchmark.csv and a bar-chart PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import pandas as pd
import torch
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.brain.connectome import get_client, fetch_subnetwork, fetch_predicted_nt
from src.brain.pathways import FULL_NAVIGATION_PATHWAY, PHOTORECEPTORS, LOOMING_DETECTORS, hemisphere_stimulus_populations
from src.brain.simulator import ConnectomeNetwork, LIFSimulator, StimulusInjector
from src.brain.neurons import LIFParams
from src.control.motor_decoder import NavigationDecoder, SensoryRescueDecoder, MotorCommand
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.drone.camera import LANDMARK_POS as FOOD_POS
from src.environment.world import build_model, ARENA_HALF_SIZE
from src.vision.encoder import encode_left_right_from_bearing, encode_looming_from_walls

TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.2
ACTIVITY_DECAY = 0.9
FORWARD_FORCE_N = 1.5
REACHED_RADIUS_M = 0.6
MAX_STEPS = 5000  # 25 simulated seconds; includes hidden-target scan time
N_REPEATS = 6
RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"

RNG = np.random.default_rng(0)


def random_start(rng) -> tuple[tuple, float]:
    margin = 1.0
    x = rng.uniform(-ARENA_HALF_SIZE + margin, ARENA_HALF_SIZE - margin)
    y = rng.uniform(-ARENA_HALF_SIZE + margin, ARENA_HALF_SIZE - margin)
    yaw = rng.uniform(0, 360)
    return (x, y, 1.0), yaw


def rule_based_yaw(left_drive, right_drive, left_loom, right_loom, tonic_drive):
    left_bright = 1.0 - left_drive / tonic_drive
    right_bright = 1.0 - right_drive / tonic_drive
    steer = 3.0 * (left_bright - right_bright)       # turn toward the brighter side
    avoid = 2.0 * (right_loom - left_loom)            # turn away from the more-looming side
    return steer + avoid


def run_trial(condition: str, net, sim, decoder, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    start_pos, start_yaw = random_start(rng)
    model = build_model(drone_start_pos=start_pos, drone_start_yaw_deg=start_yaw)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    mujoco.mj_forward(model, data)

    df = net.neuron_df if net is not None else None
    if net is not None:
        photoreceptor_mask = torch.tensor(df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device)
        left_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "L").to_numpy(), device=net.device)
        right_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "R").to_numpy(), device=net.device)
        looming_mask = df["type"].isin(LOOMING_DETECTORS)
        left_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_L")).to_numpy(), device=net.device)
        right_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_R")).to_numpy(), device=net.device)
        # Phase 11 profiling: precomputed-index + persistent-buffer
        # injection measured 1.42x faster than rebuilding a boolean-mask-
        # indexed tensor every step (see StimulusInjector's docstring).
        injector = StimulusInjector(net, photoreceptor_mask, left_photo_mask, right_photo_mask,
                                     left_loom_mask, right_loom_mask)
        activity_ema = np.zeros(net.n, dtype=np.float32)

    random_yaw_state = 0.0
    prev_pos = np.array(start_pos)
    distance_traveled = 0.0
    min_distance = float(np.linalg.norm(prev_pos - FOOD_POS))
    t0 = time.time()

    reset_count = 0
    for step in range(MAX_STEPS):
        state = drone.get_state()
        left_drive, right_drive = encode_left_right_from_bearing(state.position, state.quaternion, FOOD_POS, TONIC_DRIVE)
        left_loom, right_loom = encode_looming_from_walls(state.position, state.quaternion, ARENA_HALF_SIZE)

        if condition.startswith("connectome"):
            ext = injector.build(AMBIENT_DRIVE, TONIC_DRIVE, left_drive, right_drive,
                                  left_loom * 5.0, right_loom * 5.0)
            spikes = sim.step(ext).cpu().numpy()
            activity_ema = ACTIVITY_DECAY * activity_ema + (1 - ACTIVITY_DECAY) * spikes
            command = decoder.decode(activity_ema, forward=1.0)
        elif condition == "rule_based":
            yaw = rule_based_yaw(left_drive, right_drive, left_loom, right_loom, TONIC_DRIVE)
            command = MotorCommand(yaw=yaw, left_turn=max(yaw, 0), right_turn=max(-yaw, 0))
        elif condition == "random":
            random_yaw_state = np.clip(random_yaw_state + rng.normal(0, 0.15), -2.0, 2.0)
            command = MotorCommand(yaw=random_yaw_state, left_turn=max(random_yaw_state, 0), right_turn=max(-random_yaw_state, 0))
        else:
            raise ValueError(condition)

        # The neural/rule controller supplies a desired yaw rate.  Hover and
        # the low-level rate loop are body control, not neural decisions.
        hover_command = MotorCommand(yaw=0.0, left_turn=0.0, right_turn=0.0)
        drone.apply_rotor_thrusts(mix_to_rotors(hover_command))
        forward_scale = command.forward if condition.startswith("connectome") else 1.0
        drone.apply_planar_velocity_control(
            FORWARD_FORCE_N * forward_scale, state.quaternion, state.linear_velocity
        )
        drone.apply_yaw_rate_control(command.yaw, state.angular_velocity)
        drone.step()
        was_reset = drone.reset_if_unstable(safe_position=start_pos, max_height=3.0)
        reset_count += int(was_reset)

        new_pos = drone.get_state().position
        distance_traveled += np.linalg.norm(new_pos[:2] - prev_pos[:2])
        prev_pos = new_pos

        dist_to_food = np.linalg.norm(new_pos - FOOD_POS)
        min_distance = min(min_distance, float(dist_to_food))
        if dist_to_food < REACHED_RADIUS_M:
            return dict(condition=condition, success=True, steps_to_target=step + 1,
                        wall_time_s=time.time() - t0, distance_traveled=distance_traveled,
                        final_distance=dist_to_food, min_distance=min_distance,
                        reset_count=reset_count, seed=seed)

    return dict(condition=condition, success=False, steps_to_target=MAX_STEPS,
                wall_time_s=time.time() - t0, distance_traveled=distance_traveled,
                final_distance=float(np.linalg.norm(prev_pos - FOOD_POS)),
                min_distance=min_distance, reset_count=reset_count, seed=seed)


def make_lesion_mask(net, fraction: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    mask = np.zeros(net.n, dtype=bool)
    n_lesion = int(round(fraction * net.n))
    idx = rng.choice(net.n, size=n_lesion, replace=False)
    mask[idx] = True
    return torch.tensor(mask)


def main():
    client = get_client()
    neuron_df, conn_df = fetch_subnetwork(client, FULL_NAVIGATION_PATHWAY, "navigation")
    nt_df = fetch_predicted_nt(client, FULL_NAVIGATION_PATHWAY, "navigation")
    net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
    decoder = NavigationDecoder(net)
    rescue_decoder = SensoryRescueDecoder(net)

    conditions = ["connectome", "connectome_rescue", "connectome_lesion_10", "connectome_lesion_20",
                  "connectome_lesion_30", "rule_based", "random"]

    results = []
    for condition in conditions:
        print(f"\n=== {condition} ===", flush=True)
        for rep in range(N_REPEATS):
            # Paired experimental design: every controller sees the exact
            # same start position/yaw for a given repetition.  Do not use
            # Python's randomized hash(), which changes across processes.
            seed = 10_000 + rep
            if condition in ("connectome", "connectome_rescue"):
                sim = LIFSimulator(net, LIFParams(), gain=GAIN)
            elif condition.startswith("connectome_lesion_"):
                frac = int(condition.rsplit("_", 1)[1]) / 100.0
                lesion_mask = make_lesion_mask(net, frac, seed=1000 + rep)
                sim = LIFSimulator(net, LIFParams(), gain=GAIN, lesion_mask=lesion_mask)
            else:
                sim = None
            active_decoder = rescue_decoder if condition == "connectome_rescue" else decoder
            result = run_trial(condition, net if condition.startswith("connectome") else None, sim, active_decoder, seed)
            results.append(result)
            print(f"  rep {rep}: success={result['success']} steps={result['steps_to_target']} "
                  f"dist_traveled={result['distance_traveled']:.1f} final_dist={result['final_distance']:.2f}", flush=True)

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'benchmark.csv'}")

    summary = df.groupby("condition").agg(
        success_rate=("success", "mean"),
        mean_steps_to_target=("steps_to_target", "mean"),
        mean_distance_traveled=("distance_traveled", "mean"),
        mean_final_distance=("final_distance", "mean"),
    ).reindex(conditions)
    print("\n" + summary.to_string())
    summary.to_csv(RESULTS_DIR / "benchmark_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    summary["success_rate"].plot.bar(ax=axes[0], color="steelblue")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title(f"Food-reach success rate (n={N_REPEATS} per condition)")
    axes[0].set_ylim(0, 1)
    summary["mean_final_distance"].plot.bar(ax=axes[1], color="indianred")
    axes[1].set_ylabel("Mean final distance to food (m)")
    axes[1].set_title("Lower = closer to food when trial ended")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "benchmark.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'benchmark.png'}")


if __name__ == "__main__":
    main()
