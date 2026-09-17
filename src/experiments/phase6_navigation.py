"""Phase 6: simple navigation in a walled arena.

    - approach a visual landmark (steering pathway, same as Phase 4)
    - avoid the arena walls (looming/Giant-Fiber pathway, new in Phase 6)
    - maintain basic hover stability

FORWARD is a constant baseline set by this script, NOT decoded from neural
activity (no verified forward-drive pathway exists yet -- see
src/control/motor_decoder.py). Only yaw (steering + avoidance) is neurally
driven. This matches the brief's Phase 6 instruction to demonstrate the
neural controller producing useful behavior without complex RL -- here,
"useful behavior" specifically means the neurally-decoded steering/avoidance
signal, layered onto a constant forward drive, exactly like a classic
Braitenberg vehicle (constant motion + differential steering from sensors).
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
from src.control.motor_decoder import NavigationDecoder
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.drone.camera import LANDMARK_POS
from src.environment.world import build_model, ARENA_HALF_SIZE
from src.vision.encoder import encode_left_right_from_bearing, encode_looming_from_walls

TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.2
ACTIVITY_DECAY = 0.9
FORWARD_FORCE_N = 1.5  # constant direct force, not neurally decoded -- see
# Drone.apply_forward_force's docstring for why this isn't tilt-based


def main():
    client = get_client()
    neuron_df, conn_df = fetch_subnetwork(client, FULL_NAVIGATION_PATHWAY, "navigation")
    nt_df = fetch_predicted_nt(client, FULL_NAVIGATION_PATHWAY, "navigation")
    net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
    sim = LIFSimulator(net, LIFParams(), gain=GAIN)
    decoder = NavigationDecoder(net)

    df = net.neuron_df
    photoreceptor_mask = torch.tensor(df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device)
    left_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "L").to_numpy(), device=net.device)
    right_photo_mask = torch.tensor(hemisphere_stimulus_populations(df, "R").to_numpy(), device=net.device)

    looming_mask = df["type"].isin(LOOMING_DETECTORS)
    left_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_L")).to_numpy(), device=net.device)
    right_loom_mask = torch.tensor((looming_mask & df["instance"].str.endswith("_R")).to_numpy(), device=net.device)

    activity_ema = np.zeros(net.n, dtype=np.float32)

    START_POS = (-2.5, -2.5, 1.0)  # baked into the model itself (not just
    # runtime qpos) so MuJoCo's own viewer "reset" also returns here, away
    # from the walls -- see drone_xml()'s docstring
    model = build_model(drone_start_pos=START_POS)
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    mujoco.mj_forward(model, data)

    print(f"Network: {net.n} neurons, device={net.device}. Close the viewer to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        t_last_print = time.time()
        while viewer.is_running():
            state = drone.get_state()
            left_drive, right_drive = encode_left_right_from_bearing(
                state.position, state.quaternion, LANDMARK_POS, TONIC_DRIVE
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

            command = decoder.decode(activity_ema)
            rotor_thrusts = mix_to_rotors(command)
            drone.apply_rotor_thrusts(rotor_thrusts)
            drone.apply_forward_force(FORWARD_FORCE_N, state.quaternion)
            drone.apply_attitude_damping(state.angular_velocity)
            drone.step()
            was_reset = drone.reset_if_unstable(safe_position=START_POS, max_height=3.0)
            viewer.sync()

            step += 1
            if was_reset:
                print(f"step={step}: simulation diverged, reset to safe hover (see Drone.reset_if_unstable)", flush=True)
            if time.time() - t_last_print > 1.0:
                print(
                    f"step={step} pos={state.position.round(2)} yaw={command.yaw:+.2f} "
                    f"urgency={command.avoidance_urgency:.2f} "
                    f"dist_to_landmark={np.linalg.norm(state.position - LANDMARK_POS):.2f}",
                    flush=True,
                )
                t_last_print = time.time()


if __name__ == "__main__":
    main()
