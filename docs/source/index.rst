========
SpikeLab
========

Python library for spike train analysis of neural electrophysiology data.

SpikeLab provides a complete toolkit for loading, analyzing, and exporting
neuronal spike train data from multi-electrode array (MEA) electrophysiology
experiments.

Key features:

- **Spike and rate data structures** -- ``SpikeData`` and ``RateData`` classes
  for representing raw spike trains and instantaneous firing rates, with
  built-in analysis methods (ISI, firing rate, burst detection, and more).
- **Event-aligned slicing** -- Extract trial-aligned windows of spike or rate
  data around stimulus events, stored as ``SpikeSliceStack`` and
  ``RateSliceStack`` objects for easy averaging and visualization.
- **Pairwise comparison matrices** -- Compute unit-by-unit similarity or
  distance matrices with ``PairwiseCompMatrix`` and stack them across
  conditions with ``PairwiseCompMatrixStack``.
- **Flexible I/O** -- Load data from pickle, NWB, Neo, and custom formats;
  export to CSV, pickle, or HDF5 workspaces; optionally read from Amazon S3.
- **MCP server** -- Programmatic access to all core analysis via the Model
  Context Protocol, enabling integration with LLM-based analysis tools.

.. note::

   All spike times in SpikeLab are stored and processed in **milliseconds**
   unless explicitly stated otherwise.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started/index
   guides/index
   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
