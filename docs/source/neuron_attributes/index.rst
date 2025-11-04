Neuron Attributes
=================

Overview
--------

The ``NeuronAttributes`` class provides a DataFrame-based system for managing neuron-level metadata in SpikeData objects. This allows you to store and manipulate information about individual neurons such as cluster IDs, electrode positions, firing rates, quality metrics, and computed analysis results.

Key Features
^^^^^^^^^^^^

* **Flexible Storage**: Store any neuron-level data as columns in a DataFrame
* **Standard Columns**: Recommended column names for common attributes
* **Automatic Operations**: Attributes are preserved during subsetting, concatenation, and export
* **Computed Metrics**: Built-in methods for computing ISI statistics, latency analysis, and burst participation
* **Format Support**: Automatically loaded from and saved to HDF5, NWB, KiloSort formats

Quick Start
^^^^^^^^^^^

.. code-block:: python

   from spikedata import SpikeData
   
   # Create SpikeData with neuron attributes
   sd = SpikeData(
       trains,
       neuron_attributes={
           'unit_id': [1, 2, 3],
           'cluster_id': [10, 20, 30],
           'firing_rate_hz': [5.2, 8.1, 3.4]
       }
   )
   
   # Access attributes
   cluster_ids = sd.get_neuron_attribute('cluster_id')
   
   # Compute and store metrics
   sd.compute_firing_rates(unit='Hz')
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Get all attributes as DataFrame
   df = sd.neuron_attributes.to_dataframe()

Guide Sections
--------------

.. toctree::
   :maxdepth: 2

   creating_attributes
   working_with_attributes
   computing_metrics
   operations
   loading_and_exporting
   workflows
   advanced_usage

