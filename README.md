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



## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/braingeneers/IntegratedAnalysisTools) for guidelines.
