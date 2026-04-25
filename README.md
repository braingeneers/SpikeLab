# SpikeLab

[![Tests](https://github.com/braingeneers/SpikeLab/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/braingeneers/SpikeLab/actions/workflows/tests.yml?query=branch%3Amain) [![Black Formatting](https://github.com/braingeneers/SpikeLab/actions/workflows/black.yml/badge.svg)](https://github.com/braingeneers/SpikeLab/actions/workflows/black.yml)

SpikeLab is a Python library for loading, analyzing, visualizing, and exporting neuronal spike train data from multi-electrode array (MEA) electrophysiology experiments.

📖 **Documentation:** [spikelab.braingeneers.gi.ucsc.edu](https://spikelab.braingeneers.gi.ucsc.edu/)

## What SpikeLab can do

- **Load data** from common neuroscience formats (HDF5, NWB, KiloSort/Phy, SpikeInterface)
- **Represent spike trains** as `SpikeData` objects with per-unit spike times in milliseconds
- **Compute firing rates** as `RateData` objects (instantaneous firing rates binned over time)
- **Slice around events** to create `SpikeSliceStack` or `RateSliceStack` objects for event-aligned analysis
- **Conduct analyses** at the single unit, pairwise and population level
- **Export data** to KiloSort, NWB, and other formats
- **Store and organize results** using the `AnalysisWorkspace` for multi-stage analysis projects
- **Access programmatically** via a built-in MCP server for tool-based workflows
- **Run spike sorting** on electrophysiology recordings with built-in pipelines for Kilosort2, Kilosort4, and rt-sort (`spikelab.spike_sorting`)
- **Submit batch jobs** to remote Kubernetes clusters for compute-heavy workloads via `spikelab.batch_jobs`

## Installation

### Prerequisites

You need **Python 3.10 or later**. If you don't have Python installed, we recommend installing it via [Miniconda](https://docs.anaconda.com/miniconda/).

### Option 1: pip install (recommended)

```bash
pip install spikelab
```

This installs SpikeLab and its core dependencies (numpy, scipy, matplotlib, h5py).

### Option 2: conda environment

If you prefer a conda environment with all dependencies pre-configured:

```bash
git clone https://github.com/braingeneers/SpikeLab.git
cd SpikeLab
conda env create -f environment.yml
conda activate spikelab
pip install spikelab
```

### Option 3: install from source

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/braingeneers/SpikeLab.git
cd SpikeLab
pip install -e .
```

### Verify the installation

Open a Python prompt and run:

```python
from spikelab import SpikeData
print("SpikeLab is installed correctly!")
```

If you see the success message, you're ready to go.

### Optional dependencies

Some features require additional packages that are not installed by default. Install them by appending the extra in brackets:

```bash
pip install "spikelab[s3]"
pip install "spikelab[s3,ml,mcp]"   # multiple extras
pip install "spikelab[all]"         # everything except kilosort4
```

| Extra | Install command | What it enables |
|---|---|---|
| `mcp` | `pip install "spikelab[mcp]"` | Built-in MCP server for tool-based workflows |
| `sse` | `pip install "spikelab[sse]"` | SSE transport for the MCP server (uvicorn + starlette) |
| `s3` | `pip install "spikelab[s3]"` | Upload/download data from Amazon S3 (or any S3-compatible store) |
| `io` | `pip install "spikelab[io]"` | Extra I/O helpers (pandas) |
| `ml` | `pip install "spikelab[ml]"` | scikit-learn, UMAP, networkx, python-louvain |
| `neo` | `pip install "spikelab[neo]"` | NWB / neo / quantities for reading NWB files |
| `gplvm` | `pip install "spikelab[gplvm]"` | Gaussian Process Latent Variable Model fitting |
| `numba` | `pip install "spikelab[numba]"` | Numba-accelerated routines |
| `spike-sorting` | `pip install "spikelab[spike-sorting]"` (+ MATLAB for Kilosort2) | Kilosort2 / rt-sort pipelines via `spikelab.spike_sorting` |
| `kilosort4` | `pip install "spikelab[kilosort4]"` (+ PyTorch with CUDA, [installed separately](https://pytorch.org/get-started/locally/)) | Kilosort4 pipeline |
| `batch-jobs` | `pip install "spikelab[batch-jobs]"` | Submit jobs to remote Kubernetes clusters (`spikelab-batch-jobs` CLI) |
| `docs` | `pip install "spikelab[docs]"` | Sphinx + theme + autodoc-typehints for building the docs |
| `dev` | `pip install "spikelab[dev]"` | pytest, black, and other dev utilities |
| `all` | `pip install "spikelab[all]"` | All of the above except `kilosort4` |

When installing from a local source checkout, replace `spikelab` with `-e .` (e.g. `pip install -e ".[s3]"`).

## Quick start

```python
from spikelab import SpikeData
from spikelab.data_loaders import load_spikedata_from_nwb

# Load spike data from an NWB file
sd = load_spikedata_from_nwb("recording.nwb")

# Basic properties
print(f"Units: {sd.N}")
print(f"Duration: {sd.length} ms")

# Compute instantaneous firing rates (100 ms bins)
rates = sd.rates(bin_size=100.0)

# Get a binary spike raster (1 ms bins)
raster = sd.raster(bin_size_ms=1.0)

# Compute pairwise spike time tiling coefficients
sttc_matrix = sd.spike_time_tilings(delt=20.0)

# Export to KiloSort format
sd.to_kilosort("ks_output/", fs_Hz=20000.0)
```

## Key concepts

- **All spike times are in milliseconds** throughout the library.
- **`SpikeData`** holds per-unit spike times and is the starting point for all analyses.
- **`RateData`** holds binned instantaneous firing rates with shape `(units, time_bins)`.
- **`SpikeSliceStack` / `RateSliceStack`** hold event-aligned slices for comparative analysis.
- **`PairwiseCompMatrix`** holds an N x N comparison matrix (e.g., STTC between unit pairs).
- **`AnalysisWorkspace`** stores intermediate results across multi-stage analysis pipelines.

## AI-assisted analysis

SpikeLab includes a built-in skill that can guide you through data analysis interactively. The skill teaches your CLI agent of choice how to use the SpikeLab library, write analysis scripts, generate visualizations, and manage results — all through natural language conversation.

### How it works

The **spikelab-analysis-implementer** skill ships inside the installed package at `spikelab/agent/skills/spikelab-analysis-implementer/`. CLI agents that load skills from installed packages can pick it up automatically; alternatively, copy or symlink the skill into the agent's skills directory. The skill allows the agent to:

- Load your data files and set up an analysis workspace
- Write and run analysis scripts using SpikeLab methods
- Generate publication-quality figures
- Keep a running log of analyses and results
- Maintain up-to-date API documentation (repo maps)

Alternatively, MCP tools are available for all methods in the repository

## Directory structure

```
SpikeLab/
├── src/
│   └── spikelab/           # Installable Python package
│       ├── spikedata/          # Core data structures and analysis
│       │   ├── spikedata.py        # SpikeData class
│       │   ├── ratedata.py         # RateData class
│       │   ├── spikeslicestack.py  # SpikeSliceStack class
│       │   ├── rateslicestack.py   # RateSliceStack class
│       │   ├── pairwise.py         # PairwiseCompMatrix and PairwiseCompMatrixStack
│       │   ├── utils.py            # Shared utility functions
│       │   └── plot_utils.py       # Visualization helpers
│       ├── data_loaders/       # File I/O
│       │   ├── data_loaders.py     # Load from HDF5, NWB, KiloSort, SpikeInterface
│       │   ├── data_exporters.py   # Export to KiloSort, NWB, and other formats
│       │   └── s3_utils.py         # Amazon S3 upload/download utilities
│       ├── spike_sorting/      # Spike-sorting pipelines
│       │   ├── pipeline.py         # Top-level sorting pipeline + config
│       │   ├── ks2_runner.py       # Kilosort2 runner (requires MATLAB)
│       │   ├── ks4_runner.py       # Kilosort4 runner (PyTorch / CUDA)
│       │   ├── rt_sort/            # rt-sort runner
│       │   └── stim_sorting/       # Stimulation-aware sorting helpers
│       ├── workspace/          # Analysis workspace for storing intermediate results
│       │   ├── workspace.py        # AnalysisWorkspace class
│       │   └── hdf5_io.py          # HDF5 serialization for workspace objects
│       ├── mcp_server/         # MCP protocol server for programmatic access
│       │   ├── server.py           # MCP server implementation
│       │   └── tools/              # MCP tool definitions
│       ├── batch_jobs/         # Remote Kubernetes job submission
│       │   ├── cli.py              # spikelab-batch-jobs CLI
│       │   ├── session.py          # RunSession entry point
│       │   ├── policy.py           # Pre-submission policy checks
│       │   ├── profiles/           # Built-in cluster profiles (YAML)
│       │   └── templates/          # Jinja2 manifest templates
│       └── agent/              # Bundled agent skills (analysis-implementer, …)
│           └── skills/
├── tests/              # Test suite (pytest)
├── docs/               # Sphinx documentation source
├── examples/           # Example scripts and notebooks
├── environment.yml     # Conda environment specification
└── pyproject.toml      # Package configuration
```

## Running tests

```bash
git clone https://github.com/braingeneers/SpikeLab.git
cd SpikeLab
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please open an issue or pull request on the [GitHub repository](https://github.com/braingeneers/SpikeLab).

All code must be formatted with [Black](https://black.readthedocs.io/). You can check formatting with:

```bash
black --check .
```

## License

SpikeLab is released under the [MIT License](LICENSE).
