SpikeData Class
===============

Overview
--------

The ``SpikeData`` class provides a unified, extensible interface for representing, manipulating, and analyzing neuronal spike train data. It is designed to support a wide range of neuroscience data analysis workflows, with a focus on clarity, performance, and interoperability.


Key Features
^^^^^^^^^^^^

* Represents a collection of spike trains, one per neuron, as lists of numpy arrays
* Supports loading from various formats (indices/times, rasters, events, Neo objects)
* Stores metadata, neuron attributes, and optional raw timeseries data
* Provides methods for binning, rate calculation, interspike interval analysis, subsetting, concatenation, time slicing, latency analysis, and spike time tiling coefficient (STTC) computation
* Supports channel-based aggregation and mapping when channel information is available in neuron attributes

Class Attributes
----------------


* STTC helpers (``_sttc_ta``, ``_sttc_na``) are now colocated with ``spike_time_tiling`` for transparency and maintainability.

----

.. toctree::
   :maxdepth: 2
   :caption: Contents

   class_methods/index
   standalone_utilities
   migration_tips

For detailed API documentation with docstrings, see the :doc:`api/index`.
