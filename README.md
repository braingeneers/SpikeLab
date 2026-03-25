# SpikeLab

[![Tests](https://github.com/braingeneers/SpikeLab/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/braingeneers/SpikeLab/actions/workflows/tests.yml?query=branch%3Amain) [![Black Formatting](https://github.com/braingeneers/SpikeLab/actions/workflows/black.yml/badge.svg)](https://github.com/braingeneers/SpikeLab/actions/workflows/black.yml)

SpikeLab is a Python library for loading, analyzing, visualizing, and exporting neuronal spike train data from multi-electrode array (MEA) electrophysiology experiments.

## What SpikeLab can do

- **Load data** from common neuroscience formats (HDF5, NWB, KiloSort/Phy, SpikeInterface)
- **Represent spike trains** as `SpikeData` objects with per-unit spike times in milliseconds
- **Compute firing rates** as `RateData` objects (instantaneous firing rates binned over time)
- **Slice around events** to create `SpikeSliceStack` or `RateSliceStack` objects for event-aligned analysis
- **Conduct analyses** at the single unit, pairwise and population level
- **Export data** to KiloSort, NWB, and other formats
- **Store and organize results** using the `AnalysisWorkspace` for multi-stage analysis projects
- **Access programmatically** via a built-in MCP server for tool-based workflows

## Installation

### Prerequisites

You need **Python 3.10 or later**. If you don't have Python installed, we recommend installing it via [Miniconda](https://docs.anaconda.com/miniconda/).

### Option 1: pip install (recommended)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/braingeneers/SpikeLab.git
cd SpikeLab
pip install -e .
```

This installs SpikeLab and its core dependencies (numpy, scipy, h5py, mcp).

### Option 2: conda environment

If you prefer a conda environment with all dependencies pre-configured:

```bash
git clone https://github.com/braingeneers/SpikeLab.git
cd SpikeLab
conda env create -f environment.yml
conda activate spikelab
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

Some features require additional packages that are not installed by default:

| Feature | Install command | What it enables |
|---|---|---|
| S3 cloud storage | `pip install -e ".[s3]"` | Upload/download data from Amazon S3 |
| GPLVM latent models | `pip install -e ".[gplvm]"` | Gaussian Process Latent Variable Model fitting |
| NWB file support | `pip install neo quantities` | Reading NWB (Neurodata Without Borders) files |
| UMAP dimensionality reduction | `pip install umap-learn` | UMAP projections of pairwise matrices |
| Development tools | `pip install -e ".[dev]"` | pytest, black, and other dev utilities |

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

When the CLI agent accesses the SpikeLab repository, it can find the **spikelab-analysis-implementer** skill in ./.agent/skills/spiklab-analysis-implementer which allows it to:

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
│       ├── workspace/          # Analysis workspace for storing intermediate results
│       │   ├── workspace.py        # AnalysisWorkspace class
│       │   └── hdf5_io.py          # HDF5 serialization for workspace objects
│       └── mcp_server/         # MCP protocol server for programmatic access
│           ├── server.py           # MCP server implementation
│           └── tools/              # MCP tool definitions
├── tests/              # Test suite (pytest)
├── .agent/             # Claude Code skill for AI-assisted analysis
├── environment.yml     # Conda environment specification
└── pyproject.toml      # Package configuration
```

## Running tests

```bash
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
