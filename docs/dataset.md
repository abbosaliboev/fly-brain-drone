# Dataset: MaleCNS v1.0

## What it is
The complete connectome of the adult male *Drosophila melanogaster* central
nervous system: central brain, both optic lobes, and the full ventral nerve
cord, with an intact neck connective (i.e. brain-to-VNC wiring is included,
unlike the earlier MANC/hemibrain releases which covered only one region).

- **Neurons:** 166,700
- **Neuron types:** 11,710 (annotated)
- **Synaptic connections:** ~125,000,000
- **Produced by:** FlyEM Project Team (HHMI Janelia), Cambridge Connectomics
  Group (MRC LMB), Google Research
- **Publication:** Berg et al., "Sexual dimorphism in the complete connectome
  of the Drosophila male central nervous system", *Cell*, 2026
- **License:** CC-BY 4.0 (attribution required, otherwise unrestricted)
- **Official site:** https://male-cns.janelia.org/

## Access paths
1. **neuPrint** (https://neuprint.janelia.org) — interactive web explorer and
   the `neuprint-python` client library for programmatic Cypher-style
   queries (neuron search, connectivity queries, ROI filtering). This is our
   primary access path for Phase 1-3: we don't need the raw EM imagery, we
   need neuron identities, types, positions, and the connectivity matrix.
2. **Bulk files** (Google Cloud Storage) — full EM volumes, segmentation,
   skeletons (SWC), synapse point clouds. Needed only if we later want
   literal 3D neuron morphology (dendrite/axon shape) rather than
   soma-position point clouds. Sizes: neuron annotations ~13MB, NT
   predictions ~42MB, connectivity weights ~1.1GB, synapse coordinates
   ~12.7GB, synapse partner pairs ~6.8GB. We will NOT bulk-download the
   multi-terabyte EM image volumes — irrelevant to this project.
3. **`natverse::malecns`** (R) — not used; we're a Python project.

## Can we simulate the complete network?
Yes, computationally: annel0/flybrain already demonstrates a full
166,700-neuron LIF simulation at >2x realtime on a 12GB RTX 3060 using a CSR
sparse connectivity matrix. Our TITAN RTX (24GB) has more than enough
headroom. **We will target the full connectome for the simulator itself**
(this is honest and achievable), but:

- **Visualization** of all 125M synapses simultaneously is not meaningful or
  renderable — even the source projects don't attempt this. We render all
  166,700 neuron somas as a GPU point cloud (feasible), and connections
  either (a) restricted to a selected pathway/neuron-of-interest, or (b)
  aggregated to neuropil-region-level edges for an overview graph.
- **Real-time interactivity** at full scale needs to be measured on our own
  hardware before we promise it — this is a Phase 1/2 benchmarking task, not
  an assumption.

## Known pathway identities (do not invent — verify against neuPrint)
The connectome's cell type annotations include established Drosophila visual
and descending neuron classes (e.g. lobula plate tangential cells for
optic-flow, descending neurons known from prior VNC/MANC work). We will
**query neuPrint directly** for the current type names and confirm counts
before writing any code that references a specific neuron type by name.
Any pathway we cannot verify in the actual dataset will be documented in
`docs/scientific_assumptions.md` as unconfirmed / engineering abstraction,
not asserted as fact.

## Computational requirements (working estimate, to be measured)
- GPU sim (LIF, sparse CSR, full connectome): fits comfortably in 24GB VRAM
  based on annel0/flybrain's reported footprint on 12GB cards.
- Connectivity/annotation data needed for a real-time build: low tens of GB
  at most (feather/parquet tables + skeleton SWCs for the neurons we render),
  not the full multi-TB EM dataset.

## FlyVis (a second, separate dataset/model used in this project)

Our own hand-built LIF simulation of the raw MaleCNS motion pathway did not
reproduce reliable direction selectivity (see
`docs/scientific_assumptions.md` for the full diagnostic story). The working
food-search demo instead uses **FlyVis**
(https://github.com/TuragaLab/flyvis), the official implementation of
Lappalainen et al., "Connectome-constrained networks predict neural activity
across the fly visual system," *Nature* (2024). FlyVis is a *task-trained*
deep network whose architecture is constrained by an earlier fly connectome
(not literally the same individual MaleCNS reconstruction we query via
neuPrint elsewhere in this project), fit end-to-end to reproduce real
recorded optic-lobe activity. Code: MIT license. Pretrained weights: see
FlyVis's own repository for terms; we do not redistribute them here, only a
verified SHA-256 of the checkpoint we tested against (`scripts/setup_flyvis_data.py`).

We use FlyVis strictly for its motion-processing output (T4/T5-analogous
signals); the "what counts as food" color-opponent readout on top of it is
our own engineered addition, not part of FlyVis or MaleCNS. See
`docs/scientific_assumptions.md` for exactly where this line is drawn.
