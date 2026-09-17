"""Access to the MaleCNS v1.0 connectome via neuPrint.

We query neuPrint rather than bulk-downloading the raw EM/synapse dataset
(see docs/dataset.md for why). Results are cached to local parquet files
under data/cache/ so repeated runs (and later phases) don't re-hit the
server for the same query.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from neuprint import Client

DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# neuprint-python's fetch_neurons() also computes a full per-neuron-per-ROI
# synapse breakdown, which for this dataset's ~176k neurons blows up into a
# multi-million-row intermediate DataFrame and can exhaust memory during a
# pandas consolidation step. We only need identity + soma position for the
# 3D viewer, so we query neuPrint directly with a lean Cypher query instead.
_NEURON_QUERY = """
MATCH (n:Neuron)
WHERE n.somaLocation IS NOT NULL
RETURN n.bodyId AS bodyId,
       n.type AS type,
       n.instance AS instance,
       n.status AS status,
       n.size AS size,
       n.somaLocation AS somaLocation
"""


def get_client() -> Client:
    """Create a neuPrint client using a token from the environment.

    Set NEUPRINT_APPLICATION_CREDENTIALS to your neuPrint auth token
    (Account menu at https://neuprint.janelia.org), e.g. in a local
    untracked .env file loaded before running any script here.
    """
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError(
            "NEUPRINT_APPLICATION_CREDENTIALS is not set. Get your token from "
            "https://neuprint.janelia.org (Account menu, top right) and set it "
            "as an environment variable before running."
        )
    return Client(SERVER, dataset=DATASET, token=token)


def fetch_all_neurons(client: Client, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch (bodyId, type, instance, status, size, x, y, z) for every neuron
    that has a known soma position.

    NOTE: as of this query, the MaleCNS neuPrint instance has 176,422 total
    :Neuron nodes, 165,122 with status "Traced" (close to the 166,700 figure
    in the Cell paper/press release; small differences are expected as
    proofreading continues), and 141,781 with a somaLocation. Neurons without
    a soma position (~24,000, disproportionately optic-lobe cells) are
    excluded here since we have nothing to plot for them in a soma-position
    point cloud -- this is a real, documented limitation, not a random
    subsample. See docs/scientific_assumptions.md.

    Cached locally after the first fetch since this doesn't change often.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "neurons.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = client.fetch_custom(_NEURON_QUERY)
    coords = pd.DataFrame(
        [loc["coordinates"] for loc in df["somaLocation"]],
        columns=["x", "y", "z"],
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["somaLocation"]), coords], axis=1)
    df.to_parquet(cache_path)
    return df


def fetch_subnetwork(
    client: Client, types: list[str], cache_name: str, force_refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch neurons + induced ConnectsTo connectivity for a specific list of
    verified neuron types (e.g. a named pathway from pathways.py).

    Returns (neuron_df, conn_df). conn_df has one row per (bodyId_pre,
    bodyId_post, roi) with a synapse-count weight -- callers that want a
    single weight per neuron pair should group by (bodyId_pre, bodyId_post)
    and sum weight across ROIs.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    neuron_path = CACHE_DIR / f"{cache_name}_neurons.parquet"
    conn_path = CACHE_DIR / f"{cache_name}_connections.parquet"
    if neuron_path.exists() and conn_path.exists() and not force_refresh:
        cached_neurons = pd.read_parquet(neuron_path)
        cached_types = set(cached_neurons["type"].dropna().unique())
        missing_types = set(types) - cached_types
        if not missing_types:
            return cached_neurons, pd.read_parquet(conn_path)
        # Pathway definitions evolve. A cache name alone is not sufficient
        # identity: the old "navigation" cache silently omitted newly-added
        # Mi4/Mi9/Tm3 types and invalidated temporal experiments.
        print(f"Refreshing stale {cache_name!r} cache; missing types: {sorted(missing_types)}")

    # Aggregate on the Neo4j server and transfer one row per neuron pair.
    # fetch_adjacencies() first materializes millions of per-ROI rows; the
    # expanded visual pathway exhausted RAM/pagefile before pandas could
    # consolidate them, even though ConnectomeNetwork immediately performs
    # this exact aggregation itself.
    type_list = "[" + ",".join(f"'{t}'" for t in types) + "]"
    neuron_df = client.fetch_custom(f"""
        MATCH (n:Neuron) WHERE n.type IN {type_list}
        RETURN n.bodyId AS bodyId, n.type AS type,
               n.instance AS instance, n.status AS status
    """)
    conn_df = client.fetch_custom(f"""
        MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
        WHERE pre.type IN {type_list} AND post.type IN {type_list}
        RETURN pre.bodyId AS bodyId_pre, post.bodyId AS bodyId_post,
               sum(c.weight) AS weight
    """)
    neuron_df.to_parquet(neuron_path)
    conn_df.to_parquet(conn_path)
    return neuron_df, conn_df


def fetch_predicted_nt(
    client: Client, types: list[str], cache_name: str, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch per-neuron predicted neurotransmitter for a list of types.

    Returns a DataFrame with columns (bodyId, nt). Predictions are per
    individual neuron, not per type -- a handful of neurons in an otherwise
    consistent type do come back "unclear" (see docs/scientific_assumptions.md).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}_predicted_nt.parquet"
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        # NT rows do not carry type, so validate their body IDs against the
        # already-validated neuron cache when available.
        neuron_path = CACHE_DIR / f"{cache_name}_neurons.parquet"
        if neuron_path.exists():
            expected_ids = set(pd.read_parquet(neuron_path, columns=["bodyId"])["bodyId"])
            if expected_ids.issubset(set(cached["bodyId"])):
                return cached
        else:
            return cached

    type_list = "[" + ",".join(f"'{t}'" for t in types) + "]"
    query = f"""
    MATCH (n:Neuron) WHERE n.type IN {type_list}
    RETURN n.bodyId AS bodyId, n.predictedNt AS nt
    """
    df = client.fetch_custom(query)
    df["nt"] = df["nt"].fillna("unclear")
    df.to_parquet(cache_path)
    return df


def fetch_t4_input_synapse_centroids(client: Client, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch real postsynaptic locations for Mi/Tm -> T4 connections.

    Rows are aggregated per neuron pair on the server to avoid materializing
    every individual synapse locally. ``x/y/z_post`` are the mean T4
    postsynaptic-site coordinates and ``synapse_count`` is the number of
    actual SynapsesTo relationships represented by that centroid.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "t4_input_synapse_centroids.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    query = """
    MATCH (pre:Neuron)-[:Contains]->(pre_ss:SynapseSet)
          -[:ConnectsTo]->(post_ss:SynapseSet)<-[:Contains]-(post:Neuron),
          (pre_ss)-[:Contains]->(:Synapse)-[:SynapsesTo]->(post_syn:Synapse)
          <-[:Contains]-(post_ss)
    WHERE pre.type IN ['Mi1', 'Tm3', 'Mi4', 'Mi9']
      AND post.type IN ['T4a', 'T4b', 'T4c', 'T4d']
    RETURN pre.bodyId AS bodyId_pre, post.bodyId AS bodyId_post,
           pre.type AS type_pre, post.type AS type_post,
           post.instance AS instance_post,
           avg(post_syn.location.x) AS x_post,
           avg(post_syn.location.y) AS y_post,
           avg(post_syn.location.z) AS z_post,
           count(*) AS synapse_count
    """
    df = client.fetch_custom(query)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_t4_individual_synapses(
    client: Client, force_refresh: bool = False
) -> dict[str, Path]:
    """Cache individual Mi/Tm -> T4 postsynaptic sites by T4 subtype.

    Partitioning prevents the ~870k-row result from occupying memory twice
    during DataFrame/parquet conversion. Each returned file can be analyzed
    independently and discarded before loading the next subtype.
    """
    output_dir = CACHE_DIR / "t4_individual_synapses"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for subtype in ("T4a", "T4b", "T4c", "T4d"):
        path = output_dir / f"{subtype}.parquet"
        paths[subtype] = path
        if path.exists() and not force_refresh:
            continue
        query = f"""
        MATCH (pre:Neuron)-[:Contains]->(pre_ss:SynapseSet)
              -[:ConnectsTo]->(post_ss:SynapseSet)<-[:Contains]-(post:Neuron),
              (pre_ss)-[:Contains]->(pre_syn:Synapse)
              -[:SynapsesTo]->(post_syn:Synapse)<-[:Contains]-(post_ss)
        WHERE pre.type IN ['Mi1', 'Tm3', 'Mi4', 'Mi9']
          AND post.type = '{subtype}'
        WITH DISTINCT pre, post, pre_syn, post_syn
        RETURN pre.bodyId AS bodyId_pre, post.bodyId AS bodyId_post,
               pre.type AS type_pre, post.type AS type_post,
               post.instance AS instance_post,
               post_syn.location.x AS x_post,
               post_syn.location.y AS y_post,
               post_syn.location.z AS z_post,
               pre_syn.location.x AS x_pre,
               pre_syn.location.y AS y_pre,
               pre_syn.location.z AS z_pre
        """
        df = client.fetch_custom(query)
        df.to_parquet(path, index=False)
        del df
    return paths
