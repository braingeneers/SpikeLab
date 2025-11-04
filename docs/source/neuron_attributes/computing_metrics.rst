Computing Analysis Metrics
==========================

The ``NeuronAttributes`` class provides methods to compute and automatically store common analysis metrics. All methods follow the ``auto_save=True`` pattern, which stores results in the neuron_attributes DataFrame for future use.

ISI Statistics
--------------

Compute interspike interval (ISI) statistics to characterize temporal firing patterns.

Method
^^^^^^

.. code-block:: python

   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd, auto_save=True)

Parameters
^^^^^^^^^^

* ``spikedata`` (SpikeData): The SpikeData object containing spike trains
* ``auto_save`` (bool): If True (default), automatically stores results in neuron_attributes

Returns
^^^^^^^

Dictionary with seven ISI metrics as numpy arrays:

* ``mean_isi_ms``: Mean ISI in milliseconds (average time between spikes)
* ``median_isi_ms``: Median ISI (robust to bursting)
* ``cv_isi``: Coefficient of variation (std/mean)
  
  - < 0.5: Regular firing
  - ~1.0: Poisson-like
  - > 1.0: Irregular/bursting

* ``isi_skewness``: Distribution skewness (positive indicates bursting tendency)
* ``burst_index``: Fraction of ISIs < 10ms (burstiness measure)
* ``pause_ratio``: Fraction of ISIs > 100ms (long pause frequency)
* ``refractory_violations``: Count of ISIs < 2ms (quality metric, should be ~0)

Example
^^^^^^^

.. code-block:: python

   # Compute and auto-save all ISI statistics
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Access specific attributes
   cv_values = sd.neuron_attributes.get_attribute('cv_isi')
   burst_index = sd.neuron_attributes.get_attribute('burst_index')
   
   # Find regular vs bursting neurons
   import numpy as np
   regular_neurons = np.where(cv_values < 0.5)[0]
   bursting_neurons = np.where(burst_index > 0.2)[0]
   
   # Quality filtering
   violations = sd.neuron_attributes.get_attribute('refractory_violations')
   good_neurons = np.where(violations == 0)[0]



Latency Statistics
------------------

Compute timing statistics relative to a reference neuron (e.g., a pacemaker or leader).

Method
^^^^^^

.. code-block:: python

   lat_stats = sd.neuron_attributes.compute_latency_statistics(
       sd,
       reference_neuron=0,
       window_ms=100.0,
       auto_save=True
   )

Parameters
^^^^^^^^^^

* ``spikedata`` (SpikeData): The SpikeData object
* ``reference_neuron`` (int): Index of reference neuron to compute latencies relative to
* ``window_ms`` (float): Maximum latency window in milliseconds. Default: 100.0
* ``auto_save`` (bool): If True (default), stores results in neuron_attributes

Returns
^^^^^^^

Dictionary with three latency metrics:

* ``mean_latency_ms``: Mean latency to reference neuron
  
  - Positive: fires after reference
  - Negative: fires before reference

* ``median_latency_ms``: Median latency (robust to outliers)
* ``latency_jitter_ms``: Standard deviation of latencies (low jitter = precise timing)

Example
^^^^^^^

.. code-block:: python

   # Compute latencies relative to neuron 0 (e.g., pacemaker)
   lat_stats = sd.neuron_attributes.compute_latency_statistics(
       sd,
       reference_neuron=0,
       window_ms=100.0
   )
   
   # Find neurons that consistently follow the reference
   import numpy as np
   mean_lat = sd.neuron_attributes.get_attribute('mean_latency_ms')
   jitter = sd.neuron_attributes.get_attribute('latency_jitter_ms')
   
   followers = np.where(
       (mean_lat > 0) &      # Fires after reference
       (jitter < 5)          # Low jitter (precise timing)
   )[0]
   
   # Find leaders (fire before reference)
   leaders = np.where(mean_lat < 0)[0]


Burst Participation
-------------------

Analyze which neurons participate in network bursts and identify "backbone units" that consistently drive bursting activity.

Method
^^^^^^

.. code-block:: python

   burst_stats = sd.neuron_attributes.compute_burst_participation(
       sd,
       burst_edges,
       min_spikes=5,
       backbone_threshold=0.5,
       auto_save=True
   )

Parameters
^^^^^^^^^^

* ``spikedata`` (SpikeData): The SpikeData object
* ``burst_edges`` (np.ndarray): Array of shape (n_bursts, 2) defining burst time windows in milliseconds. Each row is [start, end].
* ``min_spikes`` (int): Minimum number of spikes for a neuron to be "active" in a burst. Default: 5
* ``backbone_threshold`` (float): Fraction of bursts (0-1) a neuron must participate in to be classified as a backbone unit. Default: 0.5
* ``auto_save`` (bool): If True (default), stores results in neuron_attributes

Returns
^^^^^^^

Dictionary with:

* ``burst_participation``: Fraction of bursts each neuron participates in (0-1)
* ``is_backbone_unit``: Boolean array indicating backbone classification
* ``backbone_indices``: Indices of backbone neurons
* ``frac_per_burst``: Fraction of neurons active in each burst

Additionally, metadata is stored in ``spikedata.metadata['burst_analysis']`` with burst edges, parameters, and counts.

Example
^^^^^^^

.. code-block:: python

   import numpy as np
   
   # Define burst windows (in milliseconds)
   burst_edges = np.array([
       [100, 200],
       [500, 600],
       [1000, 1200]
   ])
   
   # Compute participation metrics
   burst_stats = sd.neuron_attributes.compute_burst_participation(
       sd,
       burst_edges=burst_edges,
       min_spikes=3,           # Minimum spikes to be "active" in a burst
       backbone_threshold=0.7   # Must participate in 70% of bursts
   )
   
   # Find backbone neurons
   backbone_neurons = burst_stats['backbone_indices']
   
   # Access from attributes
   participation = sd.neuron_attributes.get_attribute('burst_participation')
   is_backbone = sd.neuron_attributes.get_attribute('is_backbone_unit')
   
   # Access metadata
   n_bursts = sd.metadata['burst_analysis']['n_bursts']
   
   # Analyze backbone vs non-backbone
   backbone_participation = participation[is_backbone].mean()
   print(f"Backbone neurons participate in {backbone_participation*100:.1f}% of bursts")


Workflow Example
----------------

Combining multiple metrics in an analysis pipeline:

.. code-block:: python

   import numpy as np
   from spikedata import SpikeData
   
   # Load your data
   sd = ...  # Your SpikeData object
   
   # 1. Compute basic firing rates
   sd.compute_firing_rates(unit='Hz')
   
   # 2. Compute ISI statistics for quality control
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # 3. Filter to high-quality neurons
   violations = sd.neuron_attributes.get_attribute('refractory_violations')
   good_neurons = np.where(violations == 0)[0]
   sd_clean = sd.subset(good_neurons)
   
   # 4. Find potential pacemaker (lowest CV, high firing rate)
   cv_values = sd_clean.neuron_attributes.get_attribute('cv_isi')
   firing_rates = sd_clean.neuron_attributes.get_attribute('firing_rate_hz')
   pacemaker_idx = np.argmin(cv_values * (1.0 / firing_rates))
   
   # 5. Compute latencies relative to pacemaker
   lat_stats = sd_clean.neuron_attributes.compute_latency_statistics(
       sd_clean,
       reference_neuron=pacemaker_idx,
       window_ms=100.0
   )
   
   # 6. Detect bursts (external method)
   burst_edges = detect_network_bursts(sd_clean)  # Your burst detection
   
   # 7. Compute burst participation
   burst_stats = sd_clean.neuron_attributes.compute_burst_participation(
       sd_clean,
       burst_edges,
       backbone_threshold=0.6
   )
   
   # 8. Get all computed metrics as DataFrame
   attrs = sd_clean.neuron_attributes.to_dataframe()
   print(attrs.columns)
   
   # Now you have: firing_rate_hz, mean_isi_ms, median_isi_ms, cv_isi,
   # isi_skewness, burst_index, pause_ratio, refractory_violations,
   # mean_latency_ms, median_latency_ms, latency_jitter_ms,
   # burst_participation, is_backbone_unit

Auto-save Pattern
-----------------

All compute methods follow the same pattern:

.. code-block:: python

   # Default: auto_save=True (stores in neuron_attributes)
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Later, access from attributes
   cv = sd.neuron_attributes.get_attribute('cv_isi')
   
   # Or, get without saving
   isi_stats = sd.neuron_attributes.compute_isi_statistics(sd, auto_save=False)
   cv = isi_stats['cv_isi']  # Use directly without storing



