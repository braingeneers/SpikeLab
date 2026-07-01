# SpikeLab example — Diazepam dose-response analysis

Companion code for the manuscript:

> **SpikeLab: Agentic tools for spike data analysis**
> Manuscript: https://doi.org/10.64898/2026.04.25.720833
> Data (Zenodo): https://doi.org/10.5281/zenodo.19776254

This directory contains [`manuscript_analysis.ipynb`](manuscript_analysis.ipynb), a
Jupyter notebook that reproduces the analysis pipeline from the manuscript on a
diazepam dose-response experiment recorded from a human brain organoid on a
multi-electrode array (MEA).

---

## 1. System requirements

### Software dependencies

SpikeLab is a pure-Python package. The example notebook requires:

| Component | Version | Provided by |
|---|---|---|
| Python | ≥ 3.10 (tested on 3.10, 3.11, 3.12, 3.13) | — |
| numpy | ≥ 1.20 | `spikelab` (core) |
| scipy | ≥ 1.5 | `spikelab` (core) |
| matplotlib | ≥ 3.5 | `spikelab` (core) |
| h5py | ≥ 3.0 | `spikelab` (core) |
| scikit-learn | ≥ 1.0 | `spikelab[ml]` |
| umap-learn | ≥ 0.5 | `spikelab[ml]` |
| networkx | ≥ 2.6 | `spikelab[ml]` |
| python-louvain | ≥ 0.16 | `spikelab[ml]` |
| jupyter / notebook | any recent | install separately |

Section 5 (GPLVM latent-state analysis) is **optional** and additionally requires:

| Component | Version | Provided by |
|---|---|---|
| jax / jaxlib | ≥ 0.4.26 | `spikelab[gplvm]` |
| jaxopt | ≥ 0.8.2 | `spikelab[gplvm]` |
| optax | ≥ 0.2.2 | `spikelab[gplvm]` |
| poor-man-gplvm | latest | install from GitHub (see below) |

If JAX / `poor_man_gplvm` are not installed, the notebook detects this and skips
Section 5 gracefully — every other section runs unchanged.

### Operating systems

Platform-independent. Tested on:

- Linux (Ubuntu 22.04)
- macOS (13+, Intel and Apple Silicon)
- Windows 11

### Non-standard hardware

None. The notebook runs on a standard CPU-only desktop or laptop. No GPU is
required (JAX runs on CPU for the optional GPLVM section). Approximately 4 GB of
free RAM is sufficient for the example dataset.

---

## 2. Installation guide

### Instructions

We recommend a fresh virtual environment (conda or venv). To run everything
except the optional GPLVM section:

```bash
pip install "spikelab[ml]" jupyter
```

To also run the optional GPLVM section (Section 5):

```bash
pip install "spikelab[gplvm]"
pip install git+https://github.com/samdeoxys1/poor-man-GPLVM.git
```

Alternatively, install the full set of optional dependencies at once:

```bash
pip install "spikelab[all]" jupyter
# poor-man-gplvm is not on PyPI and must still be installed manually (see above)
```

### Typical install time

On a normal desktop with a broadband connection:

- `spikelab[ml]` + jupyter: **~2–4 minutes**
- `spikelab[all]`: **~5–10 minutes** (pulls heavier optional packages such as
  `spikeinterface` and `numba`)

---

## 3. Demo

### Instructions to run the demo

1. Download the five spike-sorted pickle files from Zenodo
   (https://doi.org/10.5281/zenodo.19776254) and place them in a `data/`
   directory next to the notebook:

   ```
   examples/
   ├── manuscript_analysis.ipynb
   ├── README.md
   └── data/
       ├── 200123_2953_D0.pkl
       ├── 200123_2953_D3.pkl
       ├── 200123_2953_D10.pkl
       ├── 200123_2953_D30.pkl
       └── 200123_2953_D50.pkl
   ```

2. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook manuscript_analysis.ipynb
   ```

3. Run all cells top to bottom (**Kernel → Restart & Run All**).

### Expected output

The notebook prints per-cell summaries and renders inline figures, including:

- Per-condition unit counts, durations, and spike totals
- Population-rate traces with detected network bursts, and a burst-sensitivity sweep
- Per-unit firing-rate, ISI, and spike-triggered population-rate summaries
- Pairwise FR-correlation and STTC matrices and their across-condition distributions
- A Louvain functional-community count from the STTC network
- Burst-aligned rasters, burst-to-burst and rank-order correlation heatmaps
- PCA embeddings of burst correlation structure and of the full firing-rate trace
- (Optional) a GPLVM latent-state overlay and state-entropy value
- An HDF5 workspace file (`workspace_200123_2953.h5`) written to disk

Figures are illustrative of the manuscript figures; exact values depend on the
dataset revision downloaded from Zenodo.

### Expected run time

On a normal desktop:

- Full notebook **without** Section 5 (GPLVM): **~3–6 minutes**
- Including the optional GPLVM section: **~6–12 minutes** (JAX first-call
  compilation adds overhead)

---

## 4. Instructions for use

### Running on your own data

The pipeline works on any `SpikeData` object. To adapt the notebook:

1. Load or construct a `spikelab.SpikeData` for each recording/condition (spike
   times are in **milliseconds**). You can build one from spike index/time
   arrays via `SpikeData.from_idces_times(...)`, from a raster via
   `SpikeData.from_raster(...)`, or load your own serialized objects.
2. Replace the `CONDITIONS` / `LABELS` / `COLORS` lists and the loading loop in
   Section 1 with your own recordings.
3. Re-run the remaining sections. Detection and analysis parameters
   (`thr_burst`, `min_burst_diff`, `delt`, `max_lag`, smoothing kernels, etc.)
   are exposed at each call site and can be tuned to your preparation.

See the full API reference at https://spikelab.braingeneers.gi.ucsc.edu/ for all
available data structures and methods.

### Reproduction instructions (optional)

To reproduce the manuscript analysis exactly:

1. Install the dependencies as in Section 2 (use `spikelab[all]` plus
   `poor-man-gplvm` for the GPLVM section).
2. Download the dataset from Zenodo (https://doi.org/10.5281/zenodo.19776254)
   into `data/` as shown above.
3. Run the notebook end to end with **Kernel → Restart & Run All**.

Shuffle-based steps (e.g. `rank_order_correlation`) use fixed random seeds in the
notebook, so those results are deterministic across runs.
