"""GPU leaky-integrate-and-fire (LIF) simulator over a connectome subnetwork.

This is a from-scratch implementation (see docs/architecture.md for why we
didn't fork an existing simulator). The synaptic weight scale (`gain`
below) has NOT been fit against any published firing-rate data yet -- that
calibration (the same kind of validation annel0/flybrain and Shiu et al.
2024 did) is future work, tracked in docs/scientific_assumptions.md. Until
then, treat absolute firing rates from this simulator as illustrative of
the input->propagation->output *pipeline*, not as validated predictions.
"""
from __future__ import annotations

import math

import pandas as pd
import torch

from .neurons import LIFParams, NT_SIGN


class ConnectomeNetwork:
    """Indexed, signed, GPU-resident representation of a neuron subnetwork."""

    def __init__(self, neuron_df: pd.DataFrame, conn_df: pd.DataFrame, nt_df: pd.DataFrame, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.neuron_df = neuron_df.reset_index(drop=True)
        self.n = len(self.neuron_df)
        self.body_id_to_idx = {bid: i for i, bid in enumerate(self.neuron_df["bodyId"])}

        # One weight per (pre, post) pair, summed across ROI rows.
        agg = conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
        agg = agg[agg["bodyId_pre"].isin(self.body_id_to_idx) & agg["bodyId_post"].isin(self.body_id_to_idx)]

        nt_lookup = dict(zip(nt_df["bodyId"], nt_df["nt"]))
        sign = agg["bodyId_pre"].map(lambda bid: NT_SIGN.get(nt_lookup.get(bid, "unclear"), 1.0))

        pre_idx = agg["bodyId_pre"].map(self.body_id_to_idx).to_numpy()
        post_idx = agg["bodyId_post"].map(self.body_id_to_idx).to_numpy()
        signed_weight = (agg["weight"].to_numpy() * sign.to_numpy()).astype("float32")

        self.pre_idx = torch.tensor(pre_idx, dtype=torch.long, device=self.device)
        self.post_idx = torch.tensor(post_idx, dtype=torch.long, device=self.device)
        self.weight = torch.tensor(signed_weight, device=self.device)

    def index_of(self, body_id: int) -> int:
        return self.body_id_to_idx[body_id]

    def indices_of(self, body_ids) -> torch.Tensor:
        return torch.tensor([self.body_id_to_idx[b] for b in body_ids], dtype=torch.long, device=self.device)


def visual_temporal_parameters(
    network: ConnectomeNetwork,
    fast_tau_ms: float = 3.0,
    slow_tau_ms: float = 20.0,
    default_tau_ms: float = 5.0,
    slow_delay_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build literature-motivated temporal parameters for the T4 inputs.

    Mi1/Tm3 are fast/transient and Mi4/Mi9 are slow/sustained.  Exact
    MaleCNS per-cell constants are unavailable, so these are explicit model
    parameters to calibrate, not measured values. Delays are assigned by the
    presynaptic cell type for every real connectome edge.
    """
    from .pathways import FAST_INPUT_TYPES, SLOW_INPUT_TYPES

    types = network.neuron_df["type"]
    tau = torch.full((network.n,), default_tau_ms, device=network.device)
    fast_mask = torch.tensor(types.isin(FAST_INPUT_TYPES).to_numpy(), device=network.device)
    slow_mask = torch.tensor(types.isin(SLOW_INPUT_TYPES).to_numpy(), device=network.device)
    tau[fast_mask] = fast_tau_ms
    tau[slow_mask] = slow_tau_ms

    delays = torch.ones(network.pre_idx.shape, dtype=torch.long, device=network.device)
    slow_pre = slow_mask[network.pre_idx]
    delays[slow_pre] = slow_delay_steps
    return tau, delays


def visual_graded_mask(network: ConnectomeNetwork) -> torch.Tensor:
    """Cells modeled as non-spiking graded relays in the motion circuit."""
    from .pathways import FAST_INPUT_TYPES, SLOW_INPUT_TYPES

    graded_types = set(FAST_INPUT_TYPES) | set(SLOW_INPUT_TYPES)
    return torch.tensor(
        network.neuron_df["type"].isin(graded_types).to_numpy(),
        dtype=torch.bool,
        device=network.device,
    )


class LIFSimulator:
    """Discrete-time LIF simulation driven by a ConnectomeNetwork."""

    def __init__(self, network: ConnectomeNetwork, params: LIFParams = LIFParams(), gain: float = 0.02,
                 lesion_mask: torch.Tensor | None = None, tau_m_per_neuron: torch.Tensor | None = None,
                 synaptic_tau_ms_per_neuron: torch.Tensor | None = None,
                 delay_steps_per_connection: torch.Tensor | None = None,
                 graded_mask: torch.Tensor | None = None,
                 motion_gate_gain: float = 0.0,
                 motion_opponent_gain: float = 0.0,
                 motion_opponent_tau_ms: float = 10.0,
                 motion_three_branch_gain: float = 0.0,
                 mi4_tau_ms: float = 15.0,
                 mi9_tau_ms: float = 25.0,
                 motion_edge_spatial_weight: torch.Tensor | None = None):
        """lesion_mask: optional bool tensor (n,), True = this neuron is
        lesioned. Lesioned neurons are forced to never spike (silenced) --
        the standard "remove a neuron" operation for a spiking simulation:
        they still receive input and their membrane potential still
        evolves, but they can never influence anything downstream. Used
        for the Phase 8-9 lesion experiments (docs/experiments.md).

        tau_m_per_neuron: optional float tensor (n,) of per-neuron membrane
        time constants, overriding params.tau_m_ms's single global value
        for whichever neurons it's set for. Added to test the
        static-vs-motion/simulator-biophysics finding in
        docs/scientific_assumptions.md: real T4/T5 direction selectivity is
        believed to depend on FAST (Mi1/Tm3) vs SLOW (Mi9/Mi4) input
        dynamics (see src/brain/pathways.py's T4_INPUT_LINES), which a
        single global tau_m cannot represent at all. A slower tau_m makes a
        neuron's own response lag a fast stimulus onset more than a fast-
        tau_m neuron's would -- an approximation of a "delay line" via
        intrinsic dynamics, not an explicit multi-step synaptic delay
        buffer (which this simulator still doesn't have)."""
        self.net = network
        self.p = params
        self.gain = gain  # unfit synaptic scale, see module docstring
        self.device = network.device
        n = network.n
        self.v = torch.full((n,), params.v_rest, device=self.device)
        self.refractory = torch.zeros(n, dtype=torch.long, device=self.device)
        self.spikes_prev = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.output_prev = torch.zeros(n, device=self.device)
        self.refractory_steps = max(1, round(params.refractory_ms / params.dt_ms))
        self.lesion_mask = (
            lesion_mask.to(self.device) if lesion_mask is not None
            else torch.zeros(n, dtype=torch.bool, device=self.device)
        )
        self.tau_m_ms = (
            tau_m_per_neuron.to(self.device) if tau_m_per_neuron is not None
            else torch.full((n,), params.tau_m_ms, device=self.device)
        )
        self.graded_mask = (
            graded_mask.to(self.device, dtype=torch.bool) if graded_mask is not None
            else torch.zeros(n, dtype=torch.bool, device=self.device)
        )
        self.motion_gate_gain = motion_gate_gain
        self.motion_opponent_gain = motion_opponent_gain
        self.motion_opponent_tau_ms = motion_opponent_tau_ms
        self.motion_three_branch_gain = motion_three_branch_gain
        self.mi4_tau_ms = mi4_tau_ms
        self.mi9_tau_ms = mi9_tau_ms
        self.motion_edge_spatial_weight = (
            motion_edge_spatial_weight.to(self.device)
            if motion_edge_spatial_weight is not None else None
        )
        if self.motion_edge_spatial_weight is not None and self.motion_edge_spatial_weight.shape != network.weight.shape:
            raise ValueError("motion_edge_spatial_weight must have one value per connection")
        self.motion_fast_edges = None
        self.motion_slow_edges = None
        self.motion_fast_norm = None
        self.motion_slow_norm = None
        self.motion_fast_trace = torch.zeros(n, device=self.device)
        self.motion_slow_trace = torch.zeros(n, device=self.device)
        self.motion_mi4_edges = None
        self.motion_mi9_edges = None
        self.motion_mi4_norm = None
        self.motion_mi9_norm = None
        self.motion_mi4_trace = torch.zeros(n, device=self.device)
        self.motion_mi9_trace = torch.zeros(n, device=self.device)
        if motion_gate_gain or motion_opponent_gain or motion_three_branch_gain:
            from .pathways import FAST_INPUT_TYPES, SLOW_INPUT_TYPES, MOTION_DETECTORS

            types = network.neuron_df["type"].to_numpy()
            pre_types = types[network.pre_idx.cpu().numpy()]
            post_types = types[network.post_idx.cpu().numpy()]
            motion_post = torch.tensor(
                pd.Series(post_types).isin(MOTION_DETECTORS).to_numpy(), device=self.device
            )
            self.motion_fast_edges = motion_post & torch.tensor(
                pd.Series(pre_types).isin(FAST_INPUT_TYPES).to_numpy(), device=self.device
            )
            self.motion_slow_edges = motion_post & torch.tensor(
                pd.Series(pre_types).isin(SLOW_INPUT_TYPES).to_numpy(), device=self.device
            )
            self.motion_mi4_edges = motion_post & torch.tensor(pre_types == "Mi4", device=self.device)
            self.motion_mi9_edges = motion_post & torch.tensor(pre_types == "Mi9", device=self.device)
            self.motion_fast_norm = torch.zeros(n, device=self.device)
            self.motion_slow_norm = torch.zeros(n, device=self.device)
            self.motion_fast_norm.scatter_add_(
                0, network.post_idx[self.motion_fast_edges], network.weight[self.motion_fast_edges].abs()
            )
            self.motion_slow_norm.scatter_add_(
                0, network.post_idx[self.motion_slow_edges], network.weight[self.motion_slow_edges].abs()
            )
            self.motion_mi4_norm = torch.zeros(n, device=self.device)
            self.motion_mi9_norm = torch.zeros(n, device=self.device)
            self.motion_mi4_norm.scatter_add_(
                0, network.post_idx[self.motion_mi4_edges], network.weight[self.motion_mi4_edges].abs()
            )
            self.motion_mi9_norm.scatter_add_(
                0, network.post_idx[self.motion_mi9_edges], network.weight[self.motion_mi9_edges].abs()
            )
        # Optional temporal dynamics.  The default remains the original
        # one-step, unfiltered model so old experiments are reproducible.
        self.synaptic_tau_ms = (
            synaptic_tau_ms_per_neuron.to(self.device)
            if synaptic_tau_ms_per_neuron is not None else None
        )
        self.synaptic_trace = torch.zeros(n, device=self.device)

        if delay_steps_per_connection is None:
            self.delay_steps = None
            self.spike_history = None
            self.history_cursor = 0
        else:
            delays = delay_steps_per_connection.to(self.device, dtype=torch.long)
            if delays.shape != network.pre_idx.shape:
                raise ValueError("delay_steps_per_connection must have one value per connection")
            if torch.any(delays < 1):
                raise ValueError("connection delays must be at least one simulation step")
            self.delay_steps = delays
            history_len = int(delays.max().item()) + 1
            self.spike_history = torch.zeros((history_len, n), device=self.device)
            self.history_cursor = 0

    def step(self, external_input: torch.Tensor) -> torch.Tensor:
        """Advance one dt. external_input: float tensor of shape (n,), one
        externally injected current value per neuron (0 for un-stimulated
        neurons). Returns the boolean spike tensor for this step."""
        n = self.net.n
        syn_current = torch.zeros(n, device=self.device)
        motion_gate_current = torch.zeros(n, device=self.device)
        if self.net.pre_idx.numel() > 0:
            presynaptic_output = self.output_prev
            if self.synaptic_tau_ms is not None:
                decay = torch.exp(-self.p.dt_ms / self.synaptic_tau_ms)
                self.synaptic_trace.mul_(decay).add_(presynaptic_output)
                presynaptic_output = self.synaptic_trace

            if self.spike_history is None:
                edge_output = presynaptic_output[self.net.pre_idx]
            else:
                # Store the latest completed step, then read each edge from
                # its own delayed history slot. Delay=1 is equivalent to the
                # original simulator's previous-step transmission.
                self.spike_history[self.history_cursor] = presynaptic_output
                slots = (self.history_cursor - (self.delay_steps - 1)) % self.spike_history.shape[0]
                edge_output = self.spike_history[slots, self.net.pre_idx]
                self.history_cursor = (self.history_cursor + 1) % self.spike_history.shape[0]
            syn_current.scatter_add_(0, self.net.post_idx, self.net.weight * edge_output)
            if self.motion_gate_gain or self.motion_opponent_gain or self.motion_three_branch_gain:
                fast = torch.zeros(n, device=self.device)
                slow = torch.zeros(n, device=self.device)
                # Use the undelayed relay output here. The opponent circuit
                # supplies its own matched fast/slow temporal filters.
                raw_edge_output = self.output_prev[self.net.pre_idx]
                fast.scatter_add_(
                    0,
                    self.net.post_idx[self.motion_fast_edges],
                    self.net.weight[self.motion_fast_edges] * raw_edge_output[self.motion_fast_edges],
                )
                slow.scatter_add_(
                    0,
                    self.net.post_idx[self.motion_slow_edges],
                    self.net.weight[self.motion_slow_edges] * raw_edge_output[self.motion_slow_edges],
                )
                fast = fast / self.motion_fast_norm.clamp_min(1.0)
                slow = slow / self.motion_slow_norm.clamp_min(1.0)
                if self.motion_gate_gain:
                    motion_gate_current += self.motion_gate_gain * fast * slow
                if self.motion_opponent_gain:
                    # Reichardt-style opponent pair. Traces are from the
                    # previous step, so order matters: reversing a moving
                    # edge flips the sign instead of producing equal energy.
                    opponent = fast * self.motion_slow_trace - slow * self.motion_fast_trace
                    motion_gate_current += self.motion_opponent_gain * opponent
                    decay = math.exp(-self.p.dt_ms / self.motion_opponent_tau_ms)
                    self.motion_fast_trace.mul_(decay).add_(fast, alpha=1.0 - decay)
                    self.motion_slow_trace.mul_(decay).add_(slow, alpha=1.0 - decay)
                if self.motion_three_branch_gain:
                    mi4 = torch.zeros(n, device=self.device)
                    mi9 = torch.zeros(n, device=self.device)
                    # Explicit signs are applied below; abs weights here
                    # represent branch strength without conflating Mi4 and
                    # Mi9 merely because both are glutamatergic.
                    if self.motion_edge_spatial_weight is None:
                        mi4_factor = torch.ones_like(self.net.weight[self.motion_mi4_edges])
                        mi9_factor = torch.ones_like(self.net.weight[self.motion_mi9_edges])
                    else:
                        # Axis points Mi4 -> Mi9: retain the anatomically
                        # appropriate half-space for each branch.
                        mi4_factor = torch.clamp(
                            -self.motion_edge_spatial_weight[self.motion_mi4_edges], min=0.0, max=3.0
                        )
                        mi9_factor = torch.clamp(
                            self.motion_edge_spatial_weight[self.motion_mi9_edges], min=0.0, max=3.0
                        )
                    mi4.scatter_add_(
                        0, self.net.post_idx[self.motion_mi4_edges],
                        self.net.weight[self.motion_mi4_edges].abs()
                        * raw_edge_output[self.motion_mi4_edges] * mi4_factor,
                    )
                    mi9.scatter_add_(
                        0, self.net.post_idx[self.motion_mi9_edges],
                        self.net.weight[self.motion_mi9_edges].abs()
                        * raw_edge_output[self.motion_mi9_edges] * mi9_factor,
                    )
                    mi4 = mi4 / self.motion_mi4_norm.clamp_min(1.0)
                    mi9 = mi9 / self.motion_mi9_norm.clamp_min(1.0)
                    branch = (
                        fast * self.motion_mi9_trace - mi9 * self.motion_fast_trace
                        - fast * self.motion_mi4_trace + mi4 * self.motion_fast_trace
                    )
                    motion_gate_current += self.motion_three_branch_gain * branch
                    mi4_decay = math.exp(-self.p.dt_ms / self.mi4_tau_ms)
                    mi9_decay = math.exp(-self.p.dt_ms / self.mi9_tau_ms)
                    self.motion_mi4_trace.mul_(mi4_decay).add_(mi4, alpha=1.0 - mi4_decay)
                    self.motion_mi9_trace.mul_(mi9_decay).add_(mi9, alpha=1.0 - mi9_decay)

        total_input = external_input + self.gain * syn_current + motion_gate_current
        not_refractory = self.refractory == 0

        dv = (self.p.dt_ms / self.tau_m_ms) * (-(self.v - self.p.v_rest) + total_input)
        self.v = torch.where(not_refractory, self.v + dv, self.v)

        # Graded cells transmit their membrane response continuously and do
        # not emit/reset on discrete spikes. Their normalized output is 0 at
        # rest and saturates at the nominal LIF threshold.
        spikes = (
            (self.v >= self.p.v_threshold) & not_refractory
            & ~self.lesion_mask & ~self.graded_mask
        )
        self.v = torch.where(spikes, torch.full_like(self.v, self.p.v_rest), self.v)
        graded_floor = self.p.v_rest - (self.p.v_threshold - self.p.v_rest)
        graded_ceiling = self.p.v_threshold
        self.v = torch.where(
            self.graded_mask,
            torch.clamp(self.v, min=graded_floor, max=graded_ceiling),
            self.v,
        )
        self.refractory = torch.where(spikes, torch.full_like(self.refractory, self.refractory_steps), self.refractory)
        self.refractory = torch.clamp(self.refractory - 1, min=0)

        self.spikes_prev = spikes
        threshold_gap = self.p.v_threshold - self.p.v_rest
        graded_output = torch.clamp((self.v - self.p.v_rest) / threshold_gap, 0.0, 1.0)
        self.output_prev = torch.where(self.graded_mask, graded_output, spikes.float())
        self.output_prev = torch.where(self.lesion_mask, torch.zeros_like(self.output_prev), self.output_prev)
        return spikes


class StimulusInjector:
    """Builds the per-step external-input tensor efficiently.

    Phase 11 profiling found that rebuilding this tensor with
    `torch.full()` + boolean-mask assignment every step (the pattern used
    in Phases 2-7's experiment scripts) spent ~83% of total closed-loop
    wall time in tensor bookkeeping, not the LIF computation itself.
    Precomputing integer indices once and reusing one persistent buffer
    with in-place ops (`fill_`/`index_fill_`/`index_put_`) measured a real
    1.42x throughput improvement in isolated profiling (792 -> 1126
    steps/sec on this network). This class packages that pattern so new
    code doesn't have to rediscover it.
    """

    def __init__(self, network: ConnectomeNetwork, photoreceptor_mask: torch.Tensor,
                 left_photo_mask: torch.Tensor, right_photo_mask: torch.Tensor,
                 left_loom_mask: torch.Tensor | None = None, right_loom_mask: torch.Tensor | None = None):
        self.device = network.device
        self.buf = torch.empty(network.n, device=self.device)
        self.photo_idx = photoreceptor_mask.nonzero(as_tuple=True)[0]
        self.left_photo_idx = left_photo_mask.nonzero(as_tuple=True)[0]
        self.right_photo_idx = right_photo_mask.nonzero(as_tuple=True)[0]
        self.left_loom_idx = left_loom_mask.nonzero(as_tuple=True)[0] if left_loom_mask is not None else None
        self.right_loom_idx = right_loom_mask.nonzero(as_tuple=True)[0] if right_loom_mask is not None else None

    def build(self, ambient_drive: float, tonic_drive: float, left_drive: float, right_drive: float,
              left_loom: float = 0.0, right_loom: float = 0.0) -> torch.Tensor:
        self.buf.fill_(ambient_drive)
        self.buf.index_fill_(0, self.photo_idx, tonic_drive)
        self.buf.index_fill_(0, self.left_photo_idx, float(left_drive))
        self.buf.index_fill_(0, self.right_photo_idx, float(right_drive))
        if self.left_loom_idx is not None and left_loom:
            self.buf.index_put_((self.left_loom_idx,), self.buf[self.left_loom_idx] + left_loom)
        if self.right_loom_idx is not None and right_loom:
            self.buf.index_put_((self.right_loom_idx,), self.buf[self.right_loom_idx] + right_loom)
        return self.buf
