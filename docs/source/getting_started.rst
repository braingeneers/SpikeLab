Getting Started
===============

Installation
------------

Basic Installation
^^^^^^^^^^^^^^^^^^

Install the package using pip:

.. code-block:: bash

   pip install integrated-analysis-tools

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

For development, clone the repository and install with development dependencies:

.. code-block:: bash

   git clone https://github.com/braingeneers/IntegratedAnalysisTools.git
   cd IntegratedAnalysisTools
   pip install -e ".[dev]"

Documentation Build
^^^^^^^^^^^^^^^^^^^

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built documentation will be available in ``docs/build/html/index.html``.

Quick Start
-----------

Creating a SpikeData Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several ways to create a ``SpikeData`` object:

From spike times:

.. code-block:: python

   from spikedata import SpikeData
   import numpy as np

   # Each list element is a numpy array of spike times (in ms) for one neuron
   spike_trains = [
       np.array([10.5, 23.1, 45.8, 67.2]),
       np.array([12.3, 34.5, 56.7, 78.9]),
       np.array([15.6, 27.8, 39.1, 51.4])
   ]

   sd = SpikeData(spike_trains, length=100.0)  # 100 ms recording

From a raster matrix:

.. code-block:: python

   # Binary raster: neurons × time bins
   raster = np.random.rand(50, 1000) < 0.02
   sd = SpikeData.from_raster(raster, bin_size=1.0)

Loading Data
^^^^^^^^^^^^

Load from various file formats:

.. code-block:: python

   from data_loaders import (
       load_spikedata_from_hdf5,
       load_spikedata_from_nwb,
       load_spikedata_from_kilosort,
   )

   # From HDF5 (NWB-like ragged arrays)
   sd = load_spikedata_from_hdf5(
       "data.h5",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s"
   )

   # From NWB file
   sd = load_spikedata_from_nwb("recording.nwb")

   # From KiloSort/Phy output
   sd = load_spikedata_from_kilosort("path/to/kilosort/", fs_Hz=30000.0)

Basic Analysis
^^^^^^^^^^^^^^

Compute firing rates:

.. code-block:: python

   # Get binned spike counts
   spike_counts = sd.binned(bin_size=100.0)  # 100 ms bins

   # Get firing rates in Hz
   rates = sd.rates(bin_size=100.0)

   # Get mean firing rate per neuron
   mean_rates = sd.binned_meanrate(bin_size=100.0)

Get a spike raster:

.. code-block:: python

   # Dense binary raster
   raster = sd.raster(bin_size=1.0)  # 1 ms bins

   # Sparse raster (memory efficient for large recordings)
   sparse_raster = sd.sparse_raster(bin_size=1.0)

Analyze interspike intervals:

.. code-block:: python

   # Get ISIs for each neuron
   isis = sd.interspike_intervals()

   # Get resampled ISI distribution
   pooled_isis = sd.resampled_isi(num_samples=1000)

Subsetting and Time Slicing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Work with subsets of neurons or time windows:

.. code-block:: python

   # Get a subset of neurons
   subset = sd.subset([0, 1, 2, 5, 10])

   # Extract a time window
   window = sd.subtime(start=100.0, stop=500.0)  # 100-500 ms

   # Iterate over time frames
   for frame in sd.frames(frame_size=1000.0, overlap=100.0):
       # Analyze each frame
       rates = frame.rates(bin_size=100.0)

Exporting Data
^^^^^^^^^^^^^^

Export to various formats:

.. code-block:: python

   # Export to HDF5 (ragged array style)
   sd.to_hdf5("output.h5", style="ragged", spike_times_unit="s")

   # Export to NWB
   sd.to_nwb("output.nwb")

   # Export to KiloSort format
   sd.to_kilosort("ks_output/", fs_Hz=30000.0)

Next Steps
----------

* Learn more about the :doc:`spikedata` class and its methods
* Explore :doc:`data_loaders` for loading various file formats
* Check out :doc:`data_exporters` for exporting your data
* Browse the :doc:`api/index` for detailed API documentation

