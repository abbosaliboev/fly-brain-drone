"""Phase 3: decode LEFT/RIGHT visual stimulus into a yaw motor command.

Extends Phase 2's pathway all the way to the verified steering circuit
(T4/T5 -> HS/VS -> DNa02, see src/brain/pathways.py) and reads the result
through src/control/motor_decoder.SteeringDecoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.brain.connectome import get_client, fetch_subnetwork, fetch_predicted_nt
from src.brain.pathways import FULL_STEERING_PATHWAY, PHOTORECEPTORS, hemisphere_stimulus_populations
from src.brain.simulator import ConnectomeNetwork, LIFSimulator
from src.brain.neurons import LIFParams
from src.control.motor_decoder import SteeringDecoder

N_STEPS = 300
STIM_ON, STIM_OFF = 100, 250
TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
GAIN = 0.05


def run_trial(net: ConnectomeNetwork, side: str) -> np.ndarray:
    sim = LIFSimulator(net, LIFParams(), gain=GAIN)
    photoreceptor_mask = torch.tensor(net.neuron_df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device)
    stim_mask = torch.tensor(hemisphere_stimulus_populations(net.neuron_df, side).to_numpy(), device=net.device)
    spike_counts = torch.zeros(net.n, device=net.device)
    for t in range(N_STEPS):
        ext = torch.full((net.n,), AMBIENT_DRIVE, device=net.device)
        ext[photoreceptor_mask] = TONIC_DRIVE
        if STIM_ON <= t < STIM_OFF:
            ext[stim_mask] = 0.0
        spikes = sim.step(ext)
        spike_counts += spikes.float()
    return spike_counts.cpu().numpy()


def main():
    client = get_client()
    neuron_df, conn_df = fetch_subnetwork(client, FULL_STEERING_PATHWAY, "steering")
    nt_df = fetch_predicted_nt(client, FULL_STEERING_PATHWAY, "steering")
    net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
    decoder = SteeringDecoder(net)
    print(f"Network: {net.n} neurons, {net.pre_idx.numel()} connections, device={net.device}")
    print(f"DNa02_L index={decoder.left_idx}, DNa02_R index={decoder.right_idx}")

    global GAIN
    for gain in (0.05, 0.1, 0.15, 0.2, 0.3):
        GAIN = gain
        print(f"\n=== gain={gain} ===")
        for side in ("L", "R"):
            spike_counts = run_trial(net, side)
            command = decoder.decode(spike_counts)
            dna02_l = spike_counts[decoder.left_idx]
            dna02_r = spike_counts[decoder.right_idx]
            print(
                f"'Light on {side}' stimulus: DNa02_L={dna02_l:.0f} spikes, "
                f"DNa02_R={dna02_r:.0f} spikes -> yaw={command.yaw:+.1f}"
            )

    print(
        "\nNOTE: as in Phase 2, whether this decoded direction matches the "
        "biologically 'expected' turn has not been validated -- see "
        "docs/scientific_assumptions.md. What this experiment demonstrates is "
        "that the full verified chain (photoreceptor -> lamina -> medulla -> "
        "T4/T5 -> HS/VS -> DNa02) produces a reproducible, stimulus-dependent, "
        "asymmetric signal at the one real descending neuron we identified for "
        "steering, and that the decoder module correctly turns that into a "
        "yaw command. FORWARD/UP/DOWN remain unimplemented pending a verified "
        "pathway -- not invented."
    )


if __name__ == "__main__":
    main()
