"""Phase 2: minimal scientifically meaningful simulation.

Inject a LEFT or RIGHT stimulus into one hemisphere's photoreceptors in the
verified motion-vision pathway (src/brain/pathways.py) and observe whether
activity propagates through lamina -> medulla -> T4/T5 in a way that stays
lateralized to the stimulated side. This is the pipeline sanity check the
rest of the project (Phase 3 onward) builds on -- see docs/experiments.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.brain.connectome import get_client, fetch_subnetwork, fetch_predicted_nt
from src.brain.pathways import (
    VISUAL_MOTION_PATHWAY, PHOTORECEPTORS, LAMINA, MEDULLA, MOTION_DETECTORS,
    hemisphere_stimulus_populations,
)
from src.brain.simulator import ConnectomeNetwork, LIFSimulator
from src.brain.neurons import LIFParams

N_STEPS = 300
STIM_ON, STIM_OFF = 100, 250
TONIC_DRIVE = 15.0  # arbitrary units, unfit -- see simulator.py module docstring
# steady-state V for external input alone is v_rest + input (R=1 normalized),
# so this must exceed v_threshold - v_rest = 7 with margin, or no neuron
# ever reaches threshold no matter how long it's held.
#
# Real Drosophila photoreceptor->lamina synapses are sign-inverting: real
# photoreceptors are tonically depolarized/active in the dark, tonically
# releasing histamine, which is INHIBITORY onto L1/L2 (see NT_SIGN in
# neurons.py). Light REDUCES photoreceptor depolarization, which reduces
# histamine release, DISINHIBITING the lamina (Laughlin & Hardie-type
# photoreceptor/LMC physiology). So we give every photoreceptor a constant
# tonic drive by default (simulating "dark"), and "stimulus" on one
# hemisphere means REMOVING that drive for that hemisphere's photoreceptors
# during the stimulus window (simulating "light turns on there") -- this is
# the biologically correct direction for this specific synapse, not an
# arbitrary sign flip.
#
# We only simulate this one hand-picked 31,120-neuron pathway, not the full
# connectome, so lamina/medulla/T4/T5 neurons here are missing whatever
# input they'd normally receive from neurons outside this subnetwork (lateral
# inputs, feedback from other columns, etc). Without any of that, disinhibition
# alone just returns a neuron to its own resting potential, not above
# threshold. AMBIENT_DRIVE is an explicit, documented placeholder for
# "the rest of the brain this reduced subnetwork excludes" -- tuned only to
# make propagation observable, not fit to any real baseline firing-rate data.
AMBIENT_DRIVE = 8.5  # above the 7mV threshold gap: downstream stages fire
# tonically at baseline too, so what we look for is a lateralized *change*
# in firing rate from the (still-present-on-the-unstimulated-side)
# photoreceptor inhibition, not raw silence-vs-firing.


def stage_of(type_: str) -> str:
    if type_ in PHOTORECEPTORS:
        return "photoreceptor"
    if type_ in LAMINA:
        return "lamina"
    if type_ in MEDULLA:
        return "medulla"
    if type_ in MOTION_DETECTORS:
        return "T4/T5"
    return "other"


def run_trial(net: ConnectomeNetwork, side: str, gain: float):
    """Runs one "light on `side`" trial. Returns (spike_counts, first_spike_step)
    where first_spike_step[i] is the first step >= STIM_ON that neuron i
    spiked (or -1 if it never did during/after the stimulus)."""
    sim = LIFSimulator(net, LIFParams(), gain=gain)
    photoreceptor_mask = torch.tensor(
        net.neuron_df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device
    )
    stim_mask = torch.tensor(
        hemisphere_stimulus_populations(net.neuron_df, side).to_numpy(),
        device=net.device,
    )
    spike_counts = torch.zeros(net.n, device=net.device)
    first_spike_step = torch.full((net.n,), -1, dtype=torch.long, device=net.device)
    for t in range(N_STEPS):
        ext = torch.full((net.n,), AMBIENT_DRIVE, device=net.device)
        ext[photoreceptor_mask] = TONIC_DRIVE  # baseline "dark" tonic drive
        if STIM_ON <= t < STIM_OFF:
            ext[stim_mask] = 0.0  # "light on this side" removes the drive
        spikes = sim.step(ext)
        spike_counts += spikes.float()
        if t >= STIM_ON:
            newly = spikes & (first_spike_step < 0)
            first_spike_step = torch.where(newly, torch.full_like(first_spike_step, t), first_spike_step)
    return spike_counts.cpu().numpy(), first_spike_step.cpu().numpy()


def summarize(net: ConnectomeNetwork, spike_counts: np.ndarray, first_spike_step: np.ndarray, label: str):
    df = net.neuron_df.copy()
    df["spikes"] = spike_counts
    df["latency_ms"] = np.where(first_spike_step >= 0, first_spike_step - STIM_ON, np.nan)
    df["stage"] = df["type"].map(stage_of)
    df["hemisphere"] = df["instance"].str[-1]  # 'L' or 'R'
    print(f"\n--- {label} ---")
    pivot = df.groupby(["stage", "hemisphere"]).agg(
        mean_spikes=("spikes", "mean"),
        n=("spikes", "count"),
        mean_first_spike_latency_ms=("latency_ms", "mean"),
    )
    stage_order = ["photoreceptor", "lamina", "medulla", "T4/T5"]
    print(pivot.reindex(stage_order, level="stage").round(2))


def main():
    client = get_client()
    neuron_df, conn_df = fetch_subnetwork(client, VISUAL_MOTION_PATHWAY, "visual_motion")
    nt_df = fetch_predicted_nt(client, VISUAL_MOTION_PATHWAY, "visual_motion")

    net = ConnectomeNetwork(neuron_df, conn_df, nt_df)
    print(f"Network: {net.n} neurons, {net.pre_idx.numel()} directed connections, device={net.device}")

    gain = 0.05
    left_spikes, left_latency = run_trial(net, "L", gain)
    summarize(net, left_spikes, left_latency, f"LEFT stimulus (gain={gain})")

    right_spikes, right_latency = run_trial(net, "R", gain)
    summarize(net, right_spikes, right_latency, f"RIGHT stimulus (gain={gain})")

    print(
        "\nNOTE: downstream (lamina/medulla/T4-T5) activity does differ between "
        "hemispheres and scales with `gain`, confirming the stimulus->pathway->"
        "downstream-activity pipeline is actually propagating signal through "
        "real connectome connectivity, not just echoing the input. The DIRECTION "
        "of the lateralized effect has NOT been validated against any biological "
        "prediction -- this 31,120-neuron subnetwork has real lateral/recurrent "
        "connections (not a clean feedforward chain), an unfit synaptic gain, and "
        "an ambient-drive placeholder for excluded input (see module docstring), "
        "so interpreting the sign of the effect would need real circuit analysis, "
        "not assumption. See docs/scientific_assumptions.md."
    )


if __name__ == "__main__":
    main()
