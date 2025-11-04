Neuron Attributes
=================

The ``NeuronAttributes`` class provides a DataFrame-based system for managing neuron-level metadata and computed metrics in SpikeData objects.

Quick Start
-----------

.. code-block:: python

   from spikedata import SpikeData
   
   # Create with attributes
   sd = SpikeData(
       trains,
       neuron_attributes={
           'unit_id': [1, 2, 3],
           'cluster_id': [10, 20, 30]
       }
   )
   
   # Compute and store metrics
   sd.compute_firing_rates(unit='Hz')
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Access attributes
   df = sd.neuron_attributes.to_dataframe()

Detailed Guide
--------------

For comprehensive documentation, see:

.. toctree::
   :maxdepth: 2

   neuron_attributes/index


Common Tasks
------------

Creating Attributes
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # From dictionary
   sd = SpikeData(trains, neuron_attributes={'unit_id': [1, 2, 3]})
   
   # Add later
   sd.set_neuron_attribute('quality', ['good', 'good', 'mua'])

Computing Metrics
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Firing rates (auto-saved)
   sd.compute_firing_rates(unit='Hz')
   
   # ISI statistics (7 metrics)
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Latency to reference neuron
   lat_stats = sd.neuron_attributes.compute_latency_statistics(
       sd, reference_neuron=0, window_ms=100.0
   )
   
   # Burst participation
   burst_stats = sd.neuron_attributes.compute_burst_participation(
       sd, burst_edges=edges, min_spikes=5, backbone_threshold=0.6
   )

Accessing Attributes
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Single attribute
   cluster_ids = sd.get_neuron_attribute('cluster_id')
   
   # All attributes as DataFrame
   df = sd.neuron_attributes.to_dataframe()
   
   # Filter by attribute
   good_neurons = sd.subset(['good'], by='quality')

Standard Column Names
---------------------

**Core**: unit_id, cluster_id, electrode_id, channel, firing_rate_hz

**ISI Metrics**: mean_isi_ms, median_isi_ms, cv_isi, isi_skewness, burst_index, pause_ratio, refractory_violations

**Latency**: mean_latency_ms, median_latency_ms, latency_jitter_ms

**Burst**: burst_participation, is_backbone_unit

**Quality**: snr, amplitude, isolation_distance, isi_violations

**Spatial**: unit_x, unit_y, unit_z, electrode_x, electrode_y, electrode_z

**Waveform**: mean_waveform, waveform_channel, waveform_n_spikes

See :doc:`neuron_attributes/creating_attributes` for the complete list and details.

