SpikeData Class
===============

Overview
--------

The ``SpikeData`` class provides a unified, extensible interface for representing, manipulating, and analyzing neuronal spike train data. It is designed to support a wide range of neuroscience data analysis workflows, with a focus on clarity, performance, and interoperability.

The 2025-09 refactor streamlines the API, focusing on core spike train operations and removing legacy or niche features.

Key Features
^^^^^^^^^^^^

* Represents a collection of spike trains, one per neuron, as lists of numpy arrays
* Supports loading from various formats (indices/times, rasters, events, Neo objects)
* Stores metadata, neuron attributes, and optional raw timeseries data
* Provides methods for binning, rate calculation, interspike interval analysis, subsetting, concatenation, time slicing, latency analysis, and spike time tiling coefficient (STTC) computation

Class Attributes
----------------


* STTC helpers (``_sttc_ta``, ``_sttc_na``) are now colocated with ``spike_time_tiling`` for transparency and maintainability.

----

.. toctree::
   :maxdepth: 2
   :caption: Contents

   spikedata_methods/index
   standalone_utilities
   migration_tips

For detailed API documentation with docstrings, see the :doc:`api/index`.


API Changes in 2025-09 Refactor
--------------------------------

Removed/Deprecated
^^^^^^^^^^^^^^^^^^

The following features were removed in the 2025-09 refactor:

* NEST/MuscleBeachTools integration (``NestIDNeuronAttributes``, ``from_nest``, ``from_mbt_neurons``)
* ISI analytics (``isi_skewness``, ``isi_log_histogram``, ``isi_threshold_cma``)
* Burst/avalanche/DCC analysis (``burstiness_index``, ``avalanches``, ``avalanche_duration_size``, ``deviation_from_criticality``, ``DCCResult``, ``_p_and_alpha``)
* Randomization utilities (``randomized``, ``randomize_raster``, ``randomize_raster_greedy``, ``randomize_raster_okun``, ``_okun_swap``, ``best_effort_sample``)
* Population/correlation/histogram utilities (``population_firing_rate``, ``fano_factors``, ``pearson``, ``cumulative_moving_average``, ``burst_detection``)

