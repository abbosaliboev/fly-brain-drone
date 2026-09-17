"""Build per-T4 center/flank geometry from individual synapse locations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.brain.connectome import CACHE_DIR

FAST = {"Mi1", "Tm3"}


def analyze_subtype(subtype: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = CACHE_DIR / "t4_individual_synapses" / f"{subtype}.parquet"
    synapses = pd.read_parquet(path)
    rows = []
    edge_rows = []
    for (body_id, instance), group in synapses.groupby(["bodyId_post", "instance_post"]):
        xyz = group[["x_post", "y_post", "z_post"]].to_numpy(dtype=np.float64)
        if len(xyz) < 6:
            continue
        centered = xyz - xyz.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        local = centered @ axes.T
        types = group["type_pre"].to_numpy()
        fast = np.isin(types, list(FAST))
        mi4 = types == "Mi4"
        mi9 = types == "Mi9"
        if not (fast.any() and mi4.any() and mi9.any()):
            continue
        fast_center = local[fast].mean(axis=0)
        mi4_center = local[mi4].mean(axis=0)
        mi9_center = local[mi9].mean(axis=0)
        mi4_offset = mi4_center - fast_center
        mi9_offset = mi9_center - fast_center
        # Global preferred/null axis, pointing Mi4 -> Mi9. Project every
        # synapse onto it and aggregate per real pre->post neuron pair.
        fast_global = xyz[fast].mean(axis=0)
        mi4_global = xyz[mi4].mean(axis=0)
        mi9_global = xyz[mi9].mean(axis=0)
        axis_global = mi9_global - mi4_global
        axis_global /= np.linalg.norm(axis_global) + 1e-9
        projection = (xyz - fast_global) @ axis_global
        scale = np.std(projection) + 1e-9
        edge_data = group[["bodyId_pre", "bodyId_post", "type_pre"]].copy()
        edge_data["spatial_weight"] = projection / scale
        for (pre_id, post_id, pre_type), edge_group in edge_data.groupby(
            ["bodyId_pre", "bodyId_post", "type_pre"]
        ):
            edge_rows.append({
                "bodyId_pre": pre_id, "bodyId_post": post_id,
                "type_pre": pre_type,
                "spatial_weight": edge_group["spatial_weight"].mean(),
            })
        rows.append({
            "bodyId": body_id,
            "type": subtype,
            "instance": instance,
            "side": instance[-1],
            "n_synapses": len(group),
            **{f"mi4_d{i+1}": mi4_offset[i] for i in range(3)},
            **{f"mi9_d{i+1}": mi9_offset[i] for i in range(3)},
            "mi4_distance": float(np.linalg.norm(mi4_offset)),
            "mi9_distance": float(np.linalg.norm(mi9_offset)),
            "flank_opposition": float(np.dot(mi4_offset, mi9_offset) /
                                       (np.linalg.norm(mi4_offset) * np.linalg.norm(mi9_offset) + 1e-9)),
        })
    return pd.DataFrame(rows), pd.DataFrame(edge_rows)


def main():
    results = []
    edge_results = []
    for subtype in ("T4a", "T4b", "T4c", "T4d"):
        result, edges = analyze_subtype(subtype)
        results.append(result)
        edge_results.append(edges)
        print(f"{subtype}: {len(result)} neurons analyzed")
    all_results = pd.concat(results, ignore_index=True)
    output = CACHE_DIR / "t4_individual_dendrite_geometry.parquet"
    all_results.to_parquet(output, index=False)
    edge_output = CACHE_DIR / "t4_edge_spatial_weights.parquet"
    pd.concat(edge_results, ignore_index=True).to_parquet(edge_output, index=False)
    summary = all_results.groupby(["type", "side"]).agg(
        neurons=("bodyId", "count"),
        mi4_distance=("mi4_distance", "mean"),
        mi9_distance=("mi9_distance", "mean"),
        flank_opposition=("flank_opposition", "mean"),
        mi4_axis1=("mi4_d1", "mean"),
        mi9_axis1=("mi9_d1", "mean"),
    )
    print("\n" + summary.round(3).to_string())
    print(f"\nSaved {output}")
    print(f"Saved {edge_output}")


if __name__ == "__main__":
    main()
