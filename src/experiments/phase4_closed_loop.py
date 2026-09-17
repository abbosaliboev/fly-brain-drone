"""Phase 4: close the loop.

    drone camera -> vision encoder -> connectome (steering pathway)
        -> motor decoder -> rotor mixer -> drone physics -> new camera frame

The drone never receives target coordinates -- only its own rendered
camera image feeds the connectome. Only yaw is driven by a verified neural
pathway (DNa02); forward/altitude hold a stable hover (see
src/control/motor_decoder.py and src/drone/physics.py). Per Phase 3's
documented finding, the yaw signal is not yet reliably stimulus-locked at
this simulation's calibration -- this script demonstrates the closed loop
mechanically works end-to-end on real connectome data, not that it is a
validated flight controller.
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
from src.brain.pathways import FULL_STEERING_PATHWAY, PHOTORECEPTORS, hemisphere_stimulus_populations
from src.brain.simulator import ConnectomeNetwork, LIFSimulator
from src.brain.neurons import LIFParams
from src.control.motor_decoder import SteeringDecoder
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.environment.world import build_model
from src.vision.encoder import encode_left_right_drive

TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.2
ACTIVITY_DECAY = 0.9  # exponential moving average for the decoder's "activity" input


def main():
    client = get_client()
    neuron_df, conn_df = fetch_subnetwork(client, FULL_STEERING_PATHWAY, "steering")
    nt_df = fetch_predicted_nt(client, FULL_STEERING_PATHWAY, "steering")
    net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
    sim = LIFSimulator(net, LIFParams(), gain=GAIN)
    decoder = SteeringDecoder(net)

    photoreceptor_mask = torch.tensor(net.neuron_df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device)
    left_photo_mask = torch.tensor(hemisphere_stimulus_populations(net.neuron_df, "L").to_numpy(), device=net.device)
    right_photo_mask = torch.tensor(hemisphere_stimulus_populations(net.neuron_df, "R").to_numpy(), device=net.device)
    activity_ema = np.zeros(net.n, dtype=np.float32)

    model = build_model()
    data = mujoco.MjData(model)
    drone = Drone(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)

    print(f"Network: {net.n} neurons, device={net.device}. Starting closed loop -- close the viewer window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        t_last_print = time.time()
        while viewer.is_running():
            renderer.update_scene(data, camera="drone_eye")
            frame = renderer.render()
            left_drive, right_drive = encode_left_right_drive(frame, TONIC_DRIVE)

            ext = torch.full((net.n,), AMBIENT_DRIVE, device=net.device)
            ext[photoreceptor_mask] = TONIC_DRIVE
            ext[left_photo_mask] = left_drive
            ext[right_photo_mask] = right_drive

            spikes = sim.step(ext).cpu().numpy()
            activity_ema = ACTIVITY_DECAY * activity_ema + (1 - ACTIVITY_DECAY) * spikes

            command = decoder.decode(activity_ema)
            rotor_thrusts = mix_to_rotors(command)
            drone.apply_rotor_thrusts(rotor_thrusts)
            drone.step()
            viewer.sync()

            step += 1
            if time.time() - t_last_print > 1.0:
                state = drone.get_state()
                print(
                    f"step={step} pos={state.position.round(2)} yaw_cmd={command.yaw:+.3f} "
                    f"L_bright={1-left_drive/TONIC_DRIVE:.2f} R_bright={1-right_drive/TONIC_DRIVE:.2f}"
                )
                t_last_print = time.time()


if __name__ == "__main__":
    main()
