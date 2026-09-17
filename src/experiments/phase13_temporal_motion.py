"""Validate direction selectivity after the temporal/graded simulator upgrade.

Uses real R1-R6 synapse positions to construct six retinotopic bins per
hemisphere. A light bar sweeps through those bins in both directions. The
scientific gate is a repeatable forward-vs-reverse difference in T4 subtype
activity; food-search behavior is not evaluated until this gate passes.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from src.brain.connectome import CACHE_DIR
from src.brain.neurons import LIFParams
from src.brain.pathways import (
    PHOTORECEPTORS, MOTION_DETECTORS, FAST_INPUT_TYPES, SLOW_INPUT_TYPES,
)
from src.brain.simulator import (
    ConnectomeNetwork,
    LIFSimulator,
    visual_graded_mask,
    visual_temporal_parameters,
)

N_BINS = 6
BASELINE_STEPS = 100
STEPS_PER_BIN = 35
TAIL_STEPS = 80
TONIC_DRIVE = 15.0
AMBIENT_DRIVE = 8.5
TEMPORAL_GAIN = 0.002
MOTION_GATE_GAIN = 0.0
MOTION_OPPONENT_GAIN = 0.0
MOTION_OPPONENT_TAU_MS = 10.0
MOTION_THREE_BRANCH_GAIN = 8.0
MI4_TAU_MS = 15.0
MI9_TAU_MS = 25.0
SLOW_TAU_MS = 20.0
SLOW_DELAY_STEPS = 5


def load_network() -> ConnectomeNetwork:
    neurons = pd.read_parquet(CACHE_DIR / "navigation_neurons.parquet")
    connections = pd.read_parquet(CACHE_DIR / "navigation_connections.parquet")
    nt = pd.read_parquet(CACHE_DIR / "navigation_predicted_nt.parquet")
    return ConnectomeNetwork(
        neurons, connections, nt, device=os.environ.get("FRUIT_FLY_DEVICE", "cuda")
    )


def retinotopic_bins(net: ConnectomeNetwork) -> list[np.ndarray]:
    positions = pd.read_parquet(CACHE_DIR / "r1r6_synapse_positions.parquet")
    positions = positions.groupby(["bodyId", "instance"], as_index=False)[["x", "y", "z"]].mean()
    bins: list[np.ndarray] = []
    for side in ("L", "R"):
        side_df = positions[positions["instance"].str.endswith(f"_{side}")].copy()
        xyz = side_df[["x", "y", "z"]].to_numpy(dtype=float)
        xyz = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-9)
        # First anatomical principal axis gives a reproducible 1-D retinal
        # sweep without inventing body IDs or random bin assignments.
        _, _, vh = np.linalg.svd(xyz, full_matrices=False)
        side_df["axis"] = xyz @ vh[0]
        side_df["bin"] = pd.qcut(side_df["axis"], N_BINS, labels=False, duplicates="drop")
        for bin_id in range(N_BINS):
            body_ids = side_df.loc[side_df["bin"] == bin_id, "bodyId"]
            idx = [net.body_id_to_idx[b] for b in body_ids if b in net.body_id_to_idx]
            bins.append(np.asarray(idx, dtype=np.int64))
    return bins


def inferred_relay_coordinates(net: ConnectomeNetwork) -> np.ndarray:
    """Infer each neuron's receptive position from weighted upstream R1 paths."""
    positions = pd.read_parquet(CACHE_DIR / "r1r6_synapse_positions.parquet")
    positions = positions.groupby(["bodyId", "instance"], as_index=False)[["x", "y", "z"]].mean()
    coordinate = np.full(net.n, np.nan, dtype=np.float64)
    for side in ("L", "R"):
        side_df = positions[positions["instance"].str.endswith(f"_{side}")].copy()
        xyz = side_df[["x", "y", "z"]].to_numpy(dtype=float)
        xyz = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-9)
        _, _, vh = np.linalg.svd(xyz, full_matrices=False)
        axis = xyz @ vh[0]
        for body_id, value in zip(side_df["bodyId"], axis):
            idx = net.body_id_to_idx.get(body_id)
            if idx is not None:
                coordinate[idx] = value

    pre = net.pre_idx.cpu().numpy()
    post = net.post_idx.cpu().numpy()
    weight = net.weight.abs().cpu().numpy().astype(np.float64)
    fixed = np.isfinite(coordinate).copy()
    for _ in range(5):
        valid = np.isfinite(coordinate[pre])
        weighted_sum = np.bincount(
            post[valid], weights=weight[valid] * coordinate[pre[valid]], minlength=net.n
        )
        weight_sum = np.bincount(post[valid], weights=weight[valid], minlength=net.n)
        inferred = np.divide(
            weighted_sum, weight_sum, out=np.full(net.n, np.nan), where=weight_sum > 0
        )
        coordinate[~fixed & np.isfinite(inferred)] = inferred[~fixed & np.isfinite(inferred)]

    return coordinate


def inferred_relay_coordinates_2d(net: ConnectomeNetwork) -> np.ndarray:
    """Infer two-dimensional receptive coordinates through weighted paths."""
    positions = pd.read_parquet(CACHE_DIR / "r1r6_synapse_positions.parquet")
    positions = positions.groupby(["bodyId", "instance"], as_index=False)[["x", "y", "z"]].mean()
    coordinates = np.full((net.n, 2), np.nan, dtype=np.float64)
    for side in ("L", "R"):
        side_df = positions[positions["instance"].str.endswith(f"_{side}")].copy()
        xyz = side_df[["x", "y", "z"]].to_numpy(dtype=float)
        xyz = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-9)
        _, _, vh = np.linalg.svd(xyz, full_matrices=False)
        axes = xyz @ vh[:2].T
        for body_id, value in zip(side_df["bodyId"], axes):
            idx = net.body_id_to_idx.get(body_id)
            if idx is not None:
                coordinates[idx] = value

    pre = net.pre_idx.cpu().numpy()
    post = net.post_idx.cpu().numpy()
    weight = net.weight.abs().cpu().numpy().astype(np.float64)
    fixed = np.isfinite(coordinates[:, 0]).copy()
    for _ in range(5):
        for dim in range(2):
            valid = np.isfinite(coordinates[pre, dim])
            sums = np.bincount(
                post[valid], weights=weight[valid] * coordinates[pre[valid], dim], minlength=net.n
            )
            totals = np.bincount(post[valid], weights=weight[valid], minlength=net.n)
            inferred = np.divide(sums, totals, out=np.full(net.n, np.nan), where=totals > 0)
            coordinates[~fixed & np.isfinite(inferred), dim] = inferred[~fixed & np.isfinite(inferred)]
    return coordinates


def t4_offset_vectors(net: ConnectomeNetwork, coordinates: np.ndarray) -> pd.DataFrame:
    df = net.neuron_df
    types = df["type"].to_numpy()
    pre = net.pre_idx.cpu().numpy()
    post = net.post_idx.cpu().numpy()
    weight = net.weight.abs().cpu().numpy().astype(np.float64)
    motion_post = np.isin(types[post], ["T4a", "T4b", "T4c", "T4d"])

    centers = {}
    totals_by_group = {}
    for name, source_types in (("fast", FAST_INPUT_TYPES), ("slow", SLOW_INPUT_TYPES)):
        valid = np.isfinite(coordinates[pre, 0]) & motion_post & np.isin(types[pre], source_types)
        totals = np.bincount(post[valid], weights=weight[valid], minlength=net.n)
        center = np.full((net.n, 2), np.nan)
        for dim in range(2):
            sums = np.bincount(
                post[valid], weights=weight[valid] * coordinates[pre[valid], dim], minlength=net.n
            )
            center[:, dim] = np.divide(sums, totals, out=np.full(net.n, np.nan), where=totals > 0)
        centers[name] = center
        totals_by_group[name] = totals

    valid = np.isfinite(centers["fast"][:, 0]) & np.isfinite(centers["slow"][:, 0])
    valid &= np.isin(types, ["T4a", "T4b", "T4c", "T4d"])
    idx = np.flatnonzero(valid)
    delta = centers["slow"][idx] - centers["fast"][idx]
    return pd.DataFrame({
        "index": idx, "bodyId": df.iloc[idx]["bodyId"].to_numpy(),
        "type": types[idx], "dx": delta[:, 0], "dy": delta[:, 1],
        "magnitude": np.linalg.norm(delta, axis=1),
        "angle_deg": np.degrees(np.arctan2(delta[:, 1], delta[:, 0])),
    })


def inferred_relay_bins(net: ConnectomeNetwork) -> list[np.ndarray]:
    """Bin Mi/Tm cells using inferred receptive-field position."""
    coordinate = inferred_relay_coordinates(net)
    df = net.neuron_df
    relay = df["type"].isin(FAST_INPUT_TYPES + SLOW_INPUT_TYPES).to_numpy()
    bins: list[np.ndarray] = []
    for side in ("L", "R"):
        side_mask = df["instance"].str.endswith(f"_{side}").to_numpy()
        idx = np.flatnonzero(relay & side_mask & np.isfinite(coordinate))
        labels = pd.qcut(coordinate[idx], N_BINS, labels=False, duplicates="drop")
        for bin_id in range(N_BINS):
            bins.append(idx[np.asarray(labels) == bin_id])
    return bins


def synaptic_relay_bins(net: ConnectomeNetwork) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Build retinal sweep bins from real Mi/Tm -> T4 synapse locations."""
    syn = pd.read_parquet(CACHE_DIR / "t4_input_synapse_centroids.parquet")
    rows = []
    for body_id, group in syn.groupby("bodyId_pre"):
        xyz = np.average(
            group[["x_post", "y_post", "z_post"]], axis=0,
            weights=group["synapse_count"],
        )
        rows.append((body_id, *xyz))
    positions = pd.DataFrame(rows, columns=["bodyId", "x", "y", "z"])
    positions = positions.merge(net.neuron_df[["bodyId", "instance"]], on="bodyId")
    bins, axes = [], {}
    for side in ("L", "R"):
        side_df = positions[positions["instance"].str.endswith(f"_{side}")].copy()
        xyz = side_df[["x", "y", "z"]].to_numpy()
        centered = xyz - xyz.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        axes[side] = axis
        side_df["bin"] = pd.qcut(centered @ axis, N_BINS, labels=False)
        for bin_id in range(N_BINS):
            ids = side_df.loc[side_df["bin"] == bin_id, "bodyId"]
            bins.append(np.asarray([net.body_id_to_idx[b] for b in ids], dtype=np.int64))
    return bins, axes


def actual_t4_synaptic_vectors(net: ConnectomeNetwork) -> pd.DataFrame:
    syn = pd.read_parquet(CACHE_DIR / "t4_input_synapse_centroids.parquet")
    syn["group"] = np.where(syn["type_pre"].isin(FAST_INPUT_TYPES), "fast", "slow")
    rows = []
    for (body_id, type_, instance), group in syn.groupby(
        ["bodyId_post", "type_post", "instance_post"]
    ):
        centers = {}
        for name, subgroup in group.groupby("group"):
            centers[name] = np.average(
                subgroup[["x_post", "y_post", "z_post"]], axis=0,
                weights=subgroup["synapse_count"],
            )
        if len(centers) == 2 and body_id in net.body_id_to_idx:
            delta = centers["slow"] - centers["fast"]
            rows.append((net.body_id_to_idx[body_id], body_id, type_, instance, *delta))
    return pd.DataFrame(rows, columns=["index", "bodyId", "type", "instance", "dx", "dy", "dz"])


def spatial_edge_weight_tensor(net: ConnectomeNetwork) -> torch.Tensor:
    weights = pd.read_parquet(CACHE_DIR / "t4_edge_spatial_weights.parquet")
    lookup = {
        (int(pre), int(post)): float(value)
        for pre, post, value in weights[["bodyId_pre", "bodyId_post", "spatial_weight"]].itertuples(index=False)
    }
    pre_ids = net.neuron_df.iloc[net.pre_idx.cpu().numpy()]["bodyId"].to_numpy()
    post_ids = net.neuron_df.iloc[net.post_idx.cpu().numpy()]["bodyId"].to_numpy()
    values = np.fromiter(
        (lookup.get((int(pre), int(post)), 0.0) for pre, post in zip(pre_ids, post_ids)),
        dtype=np.float32,
        count=len(pre_ids),
    )
    return torch.tensor(values, device=net.device)


def t4_fast_slow_offsets(net: ConnectomeNetwork, coordinate: np.ndarray) -> pd.DataFrame:
    """Weighted fast-center minus slow-flank offset for every T4 neuron."""
    df = net.neuron_df
    types = df["type"].to_numpy()
    pre = net.pre_idx.cpu().numpy()
    post = net.post_idx.cpu().numpy()
    weight = net.weight.abs().cpu().numpy().astype(np.float64)
    valid_coord = np.isfinite(coordinate[pre])
    motion_post = np.isin(types[post], ["T4a", "T4b", "T4c", "T4d"])

    def weighted_center(source_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
        edge = valid_coord & motion_post & np.isin(types[pre], source_types)
        sums = np.bincount(
            post[edge], weights=weight[edge] * coordinate[pre[edge]], minlength=net.n
        )
        totals = np.bincount(post[edge], weights=weight[edge], minlength=net.n)
        center = np.divide(sums, totals, out=np.full(net.n, np.nan), where=totals > 0)
        return center, totals

    fast, fast_weight = weighted_center(FAST_INPUT_TYPES)
    slow, slow_weight = weighted_center(SLOW_INPUT_TYPES)
    t4 = np.isin(types, ["T4a", "T4b", "T4c", "T4d"])
    idx = np.flatnonzero(t4 & np.isfinite(fast) & np.isfinite(slow))
    return pd.DataFrame({
        "index": idx,
        "bodyId": df.iloc[idx]["bodyId"].to_numpy(),
        "type": types[idx],
        "instance": df.iloc[idx]["instance"].to_numpy(),
        "fast_center": fast[idx],
        "slow_center": slow[idx],
        "offset": slow[idx] - fast[idx],
        "fast_weight": fast_weight[idx],
        "slow_weight": slow_weight[idx],
    })


def run_sweep(net: ConnectomeNetwork, bins: list[np.ndarray], reverse: bool) -> np.ndarray:
    tau, delays = visual_temporal_parameters(
        net, slow_tau_ms=SLOW_TAU_MS, slow_delay_steps=SLOW_DELAY_STEPS
    )
    sim = LIFSimulator(
        net,
        LIFParams(),
        gain=TEMPORAL_GAIN,
        synaptic_tau_ms_per_neuron=tau,
        delay_steps_per_connection=delays,
        graded_mask=visual_graded_mask(net),
        motion_gate_gain=MOTION_GATE_GAIN,
        motion_opponent_gain=MOTION_OPPONENT_GAIN,
        motion_opponent_tau_ms=MOTION_OPPONENT_TAU_MS,
        motion_three_branch_gain=MOTION_THREE_BRANCH_GAIN,
        mi4_tau_ms=MI4_TAU_MS,
        mi9_tau_ms=MI9_TAU_MS,
    )
    df = net.neuron_df
    photo_mask = torch.tensor(df["type"].isin(PHOTORECEPTORS).to_numpy(), device=net.device)
    counts = torch.zeros(net.n, device=net.device)

    order = list(range(N_BINS))
    if reverse:
        order.reverse()
    # Sweep both eyes in the same anatomical direction.
    active_sequence = [(bins[i], bins[N_BINS + i]) for i in order]
    total = BASELINE_STEPS + N_BINS * STEPS_PER_BIN + TAIL_STEPS
    for step in range(total):
        external = torch.full((net.n,), AMBIENT_DRIVE, device=net.device)
        external[photo_mask] = TONIC_DRIVE
        sweep_step = step - BASELINE_STEPS
        if 0 <= sweep_step < N_BINS * STEPS_PER_BIN:
            left_idx, right_idx = active_sequence[sweep_step // STEPS_PER_BIN]
            external[torch.as_tensor(left_idx, device=net.device)] = 0.0
            external[torch.as_tensor(right_idx, device=net.device)] = 0.0
        counts += sim.step(external).float()
    return counts.cpu().numpy()


def run_relay_sweep(net: ConnectomeNetwork, bins: list[np.ndarray], reverse: bool) -> np.ndarray:
    """Drive inferred Mi/Tm retinotopic bins to isolate downstream motion math."""
    tau, delays = visual_temporal_parameters(
        net, slow_tau_ms=SLOW_TAU_MS, slow_delay_steps=SLOW_DELAY_STEPS
    )
    sim = LIFSimulator(
        net, LIFParams(), gain=TEMPORAL_GAIN,
        synaptic_tau_ms_per_neuron=tau,
        delay_steps_per_connection=delays,
        graded_mask=visual_graded_mask(net),
        motion_gate_gain=MOTION_GATE_GAIN,
        motion_opponent_gain=MOTION_OPPONENT_GAIN,
        motion_opponent_tau_ms=MOTION_OPPONENT_TAU_MS,
        motion_three_branch_gain=MOTION_THREE_BRANCH_GAIN,
        mi4_tau_ms=MI4_TAU_MS,
        mi9_tau_ms=MI9_TAU_MS,
        motion_edge_spatial_weight=spatial_edge_weight_tensor(net),
    )
    response = torch.zeros(net.n, device=net.device)
    order = list(range(N_BINS))
    if reverse:
        order.reverse()
    sequence = [(bins[i], bins[N_BINS + i]) for i in order]
    total = BASELINE_STEPS + N_BINS * STEPS_PER_BIN + TAIL_STEPS
    for step in range(total):
        external = torch.full((net.n,), AMBIENT_DRIVE, device=net.device)
        sweep_step = step - BASELINE_STEPS
        if 0 <= sweep_step < N_BINS * STEPS_PER_BIN:
            left_idx, right_idx = sequence[sweep_step // STEPS_PER_BIN]
            external[torch.as_tensor(left_idx, device=net.device)] += 6.0
            external[torch.as_tensor(right_idx, device=net.device)] += 6.0
        sim.step(external)
        # Continuous depolarization retains direction information that is
        # lost when weak responses are quantized to integer spike counts.
        normalized_v = torch.clamp(
            (sim.v - sim.p.v_rest) / (sim.p.v_threshold - sim.p.v_rest), 0.0, 1.0
        )
        response += normalized_v
    return response.cpu().numpy()


def main():
    net = load_network()
    bins = retinotopic_bins(net)
    print(f"network={net.n} neurons, device={net.device}; bin sizes={[len(x) for x in bins]}")
    forward = run_sweep(net, bins, reverse=False)
    reverse = run_sweep(net, bins, reverse=True)

    df = net.neuron_df.copy()
    df["forward"] = forward
    df["reverse"] = reverse
    df["difference"] = forward - reverse
    result = (
        df[df["type"].isin(MOTION_DETECTORS)]
        .groupby("type")[["forward", "reverse", "difference"]]
        .mean()
        .sort_index()
    )
    print("\nT4/T5 mean spikes per neuron:\n" + result.round(4).to_string())
    print(f"\nmax_abs_direction_effect={result['difference'].abs().max():.4f}")
    motion = df[df["type"].isin(MOTION_DETECTORS)]
    print(
        "individual_neuron_mean_abs_effect="
        f"{motion['difference'].abs().mean():.6f}; "
        f"individual_neuron_max_abs_effect={motion['difference'].abs().max():.4f}"
    )

    relay_bins = inferred_relay_bins(net)
    print(f"\ninferred Mi/Tm bin sizes={[len(x) for x in relay_bins]}")
    relay_forward = run_relay_sweep(net, relay_bins, reverse=False)
    relay_reverse = run_relay_sweep(net, relay_bins, reverse=True)
    relay_difference = relay_forward - relay_reverse
    motion_mask = df["type"].isin(MOTION_DETECTORS).to_numpy()
    print(
        "relay-driven individual mean/max abs effect="
        f"{np.abs(relay_difference[motion_mask]).mean():.6f}/"
        f"{np.abs(relay_difference[motion_mask]).max():.4f}"
    )

    coordinate = inferred_relay_coordinates(net)
    offsets = t4_fast_slow_offsets(net, coordinate)
    offset_summary = offsets.groupby("type")["offset"].agg(["count", "mean", "median", "std"])
    print("\nPer-T4 fast/slow receptive-field offsets:\n" + offset_summary.round(5).to_string())
    offsets.to_csv(CACHE_DIR / "t4_fast_slow_offsets.csv", index=False)
    vectors = t4_offset_vectors(net, inferred_relay_coordinates_2d(net))
    vector_summary = vectors.groupby("type").agg(
        count=("magnitude", "count"), mean_dx=("dx", "mean"),
        mean_dy=("dy", "mean"), mean_magnitude=("magnitude", "mean"),
    )
    vector_summary["population_angle_deg"] = np.degrees(
        np.arctan2(vector_summary["mean_dy"], vector_summary["mean_dx"])
    )
    print("\n2-D T4 offset vectors:\n" + vector_summary.round(5).to_string())
    vectors.to_csv(CACHE_DIR / "t4_fast_slow_offset_vectors.csv", index=False)
    # A bilateral/subtype mean cancels mirror-symmetric receptive fields.
    # Score each T4 neuron against the direction predicted by its own offset.
    t4_rows = vectors["index"].to_numpy(dtype=np.int64)
    observed = relay_difference[t4_rows]
    expected_sign = np.sign(vectors["dx"].to_numpy())
    informative = (observed != 0) & (expected_sign != 0)
    if informative.any():
        accuracy = np.mean(np.sign(observed[informative]) == expected_sign[informative])
        aligned_effect = np.mean(observed[informative] * expected_sign[informative])
    else:
        accuracy = float("nan")
        aligned_effect = 0.0
    print(
        f"geometry-aligned T4 direction accuracy={accuracy:.3f} "
        f"({informative.sum()} informative neurons); aligned_effect={aligned_effect:.6f}"
    )


if __name__ == "__main__":
    main()
