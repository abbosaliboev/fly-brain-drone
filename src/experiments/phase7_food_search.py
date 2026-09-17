"""Phase 7: find the food.

Same closed loop as Phase 6 (steering + collision avoidance), but framed
around the brief's centerpiece task: the drone must locate and reach a
food target using vision alone. The drone never receives the food's
coordinates -- src/vision/encoder.py's encode_left_right_from_bearing only
ever returns two scalar drive values, and food_distance_and_bearing (used
here only for the on-screen FOOD DISTANCE / TARGET ANGLE display, exactly
as the brief's mockup shows) is not fed into the network anywhere.

When the food is behind the drone (not currently "visible" to the bearing
sensor), encode_left_right_from_bearing returns equal drive on both sides
-- no directional cue at all. What happens next is not a programmed search
algorithm: it's whatever the connectome's own residual/spontaneous activity
does with an undifferentiated input. Whether that produces anything
resembling real fly idiothetic local search (see docs/experiments.md's
"Centerpiece experiment") is an open, unresolved question this script does
not answer by itself -- it only sets up the mechanic (a target that can be
out of view) and reports what's observed, honestly, without pre-deciding
the outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import torch
import mujoco
import mujoco.viewer

from src.brain.connectome import get_client, fetch_subnetwork, fetch_predicted_nt
from src.brain.pathways import FULL_NAVIGATION_PATHWAY, PHOTORECEPTORS, LOOMING_DETECTORS, hemisphere_stimulus_populations
from src.brain.simulator import ConnectomeNetwork, LIFSimulator
from src.brain.neurons import LIFParams
from src.control.motor_decoder import SensoryRescueDecoder, MotorCommand
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.drone.camera import LANDMARK_POS as FOOD_POS
from src.environment.world import build_model, ARENA_HALF_SIZE
from src.vision.encoder import encode_left_right_from_bearing, encode_looming_from_walls, food_distance_and_bearing

TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.2
ACTIVITY_DECAY = 0.9
FORWARD_FORCE_N = 1.5
REACHED_RADIUS_M = 0.6
START_POS = (0.0, 3.0, 1.0)  # 3m from food (at (3, 3)), same y so a pure
# 180-degree yaw faces exactly away from it
START_YAW_DEG = 180.0  # facing AWAY from food -- see
# docs/scientific_assumptions.md's Phase 7 entry for both findings:
# START_YAW_DEG=0 (facing toward food) reaches it in ~1.8s/255 steps,
# confirming the approach/reach mechanic works; START_YAW_DEG=180 (this
# default) is the actually interesting case for the centerpiece question --
# and currently the network's own residual dynamics do NOT produce
# anything resembling active search when the target starts hidden; the
# drone drifts a small amount and stalls against a wall instead.


def main():
    client = get_client()
    neuron_df, conn_df = fetch_subnetwork(client, FULL_NAVIGATION_PATHWAY, "navigation")
    nt_df = fetch_predicted_nt(client, FULL_NAVIGATION_PATHWAY, "navigation")
    net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
    sim = LIFSimulator(net, LIFParams(), gain=GAIN)
    # Explicit rescue mode: reads bilateral simulated photoreceptor activity
    # while the biophysical T4/T5 temporal model is still under development.
    decoder = SensoryRescueDecoder(net)

    df = net.neuron_df
    photoreceptor_mask = torch.tensor(df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device)
    left_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "L").to_numpy(), device=net.device)
    right_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "R").to_numpy(), device=net.device)
    looming_mask = df["type"].isin(LOOMING_DETECTORS)
    left_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_L")).to_numpy(), device=net.device)
    right_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_R")).to_numpy(), device=net.device)
    activity_ema = np.zeros(net.n, dtype=np.float32)

    model = build_model(drone_start_pos=START_POS, drone_start_yaw_deg=START_YAW_DEG)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    mujoco.mj_forward(model, data)

    trial_start = time.time()
    last_visible_time = None
    reached = False

    print(f"Network: {net.n} neurons, device={net.device}. Food at {FOOD_POS}. Close the viewer to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        t_last_print = time.time()
        while viewer.is_running() and not reached:
            state = drone.get_state()
            left_drive, right_drive = encode_left_right_from_bearing(
                state.position, state.quaternion, FOOD_POS, TONIC_DRIVE
            )
            left_loom, right_loom = encode_looming_from_walls(state.position, state.quaternion, ARENA_HALF_SIZE)

            ext = torch.full((net.n,), AMBIENT_DRIVE, device=net.device)
            ext[photoreceptor_mask] = TONIC_DRIVE
            ext[left_photo_mask] = left_drive
            ext[right_photo_mask] = right_drive
            ext[left_loom_mask] += left_loom * 5.0
            ext[right_loom_mask] += right_loom * 5.0

            spikes = sim.step(ext).cpu().numpy()
            activity_ema = ACTIVITY_DECAY * activity_ema + (1 - ACTIVITY_DECAY) * spikes

            command = decoder.decode(activity_ema, forward=1.0)
            hover_command = MotorCommand(yaw=0.0, left_turn=0.0, right_turn=0.0)
            drone.apply_rotor_thrusts(mix_to_rotors(hover_command))
            drone.apply_planar_velocity_control(
                FORWARD_FORCE_N * command.forward, state.quaternion, state.linear_velocity
            )
            drone.apply_yaw_rate_control(command.yaw, state.angular_velocity)
            drone.step()
            drone.reset_if_unstable(safe_position=START_POS, max_height=3.0)
            viewer.sync()
            step += 1

            distance, bearing_deg, visible = food_distance_and_bearing(state.position, state.quaternion, FOOD_POS)
            now = time.time()
            if visible:
                last_visible_time = now
            search_time = 0.0 if last_visible_time is None else now - last_visible_time

            if distance < REACHED_RADIUS_M:
                reached = True
                print(f"\n*** FOOD REACHED *** total time={now - trial_start:.1f}s, steps={step}")
                break

            if now - t_last_print > 1.0:
                print(
                    f"step={step} FOOD DISTANCE: {distance:5.2f} m | TARGET ANGLE: {bearing_deg:+6.1f} deg | "
                    f"visible={visible} | SEARCH TIME: {search_time:4.1f}s | urgency={command.avoidance_urgency:.2f}",
                    flush=True,
                )
                t_last_print = now

    if not reached:
        print("\nViewer closed before reaching the food.")


if __name__ == "__main__":
    main()
