"""Leaky integrate-and-fire (LIF) neuron parameters.

These are literature-typical values for insect central neurons, NOT
per-neuron measured physiology -- fewer than 1% of MaleCNS cell types have
directly measured physiological parameters (see
docs/scientific_assumptions.md). Every neuron in a given simulation
currently shares the same parameters; giving different neuron types
different time constants is a reasonable future refinement, not something
we claim to have justified per-type data for yet.
"""
from dataclasses import dataclass


@dataclass
class LIFParams:
    tau_m_ms: float = 15.0       # membrane time constant
    v_rest: float = -52.0        # mV, resting/reset potential
    v_threshold: float = -45.0   # mV, spike threshold
    refractory_ms: float = 2.0   # absolute refractory period
    dt_ms: float = 1.0           # simulation time step


# Predicted-neurotransmitter -> synaptic sign convention. This is the same
# type of approximation used in prior connectome-LIF work (e.g. Shiu et al.
# 2024): fast ionotropic acetylcholine receptors in insects are cation
# channels (excitatory), GABA-A and histamine-gated and glutamate-gated
# (GluClalpha) receptors are chloride channels (inhibitory). "unclear"
# predictions (a large fraction for photoreceptors specifically, see
# docs/scientific_assumptions.md) default to excitatory as an explicit,
# documented, arbitrary choice -- not a scientific claim about those
# synapses' real sign.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": +1.0,
    "serotonin": +1.0,
    "octopamine": +1.0,
    "unclear": +1.0,
}
