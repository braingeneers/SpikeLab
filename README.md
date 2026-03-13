# IntegratedAnalysisTools

[![SpikeData Tests](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml?query=branch%3Amain) [![Black Formatting](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml/badge.svg)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml)

A monorepo for a suite of analysis tools supporting automated closed-loop experimentation and data analysis in neuroscience and related fields.

## Overview

IntegratedAnalysisTools provides a unified framework for working with neuronal spike train data. The main components include:

- **SpikeData**: Core module for spike train data representation, manipulation, and analysis
- **RateData**: Core module for instantaneous firing rate data representation, manipulation, and analysis
- **SliceStack**: Slice Spike or Rate data and create a stack of slices for comparative analysis
- **Pairwise**: Tools for further analysis of nxn pairwise similarity matrices   
- **Data loaders**: Utilities to load various file formats (HDF5, NWB, KiloSort/Phy, SpikeInterface)
- **Data exporters**: Export SpikeData to common neuroscience formats

## Installation

For development:

```bash
git clone https://github.com/braingeneers/IntegratedAnalysisTools.git
cd IntegratedAnalysisTools
pip install -e ".[dev]"
```

Or with conda:

```bash
git clone https://github.com/braingeneers/IntegratedAnalysisTools.git
cd IntegratedAnalysisTools
conda env create -f environment.yml
conda activate integrated-analysis-tools
pip install -e ".[dev]"
```

## Quick Start

```python
from spikedata import SpikeData
from data_loaders import load_spikedata_from_nwb

# Load spike data from NWB file
sd = load_spikedata_from_nwb("recording.nwb")

# Compute firing rates
rates = sd.rates(bin_size=100.0)

# Get a spike raster
raster = sd.raster(bin_size=1.0)

# Export to different format
sd.to_kilosort("ks_output/", fs_Hz=30000.0)
```

## Using the MCP server in Cursor

The MCP (Model Context Protocol) server exposes IntegratedAnalysisTools as Cursor tools so you can load data, compute rates, plot heatmaps, and more from chat.

### 1. Set up the environment

From the **IntegratedAnalysisTools** directory, create the conda env and install the package in editable mode:

```bash
cd IntegratedAnalysisTools
conda env create -f environment.yml
conda activate integrated-analysis-tools
python -m pip install -e .
```

Use `python -m pip` (or the full path to the env's pip) so the install uses the conda env's Python, not the system one.

### 2. Add the server in Cursor

In Cursor, go to **Settings → MCP** and add a new server, or edit your MCP config file (e.g. `~/.cursor/mcp.json`). Add an entry like this, replacing the paths with your actual paths:

```json
"IntegratedAnalysisTools": {
  "command": "/path/to/your/anaconda3/envs/integrated-analysis-tools/bin/python",
  "args": ["-m", "mcp_server"],
  "cwd": "/path/to/your/IntegratedAnalysisTools",
  "type": "stdio"
}
```

Replace `command` with the path to your conda env's Python (run `which python` with the env activated). Replace `cwd` with the absolute path to your IntegratedAnalysisTools directory.

### 3. Reload and use

Reload Cursor (or turn the IntegratedAnalysisTools MCP off and on). The server should start and expose tools such as `create_workspace`, `load_from_pickle`, `load_from_hdf5`, `compute_resampled_isi`, `plot_rate_heatmap`, and others. You can then ask the AI to load data, compute rates, and plot heatmaps using those tools.

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/braingeneers/IntegratedAnalysisTools) for guidelines.
