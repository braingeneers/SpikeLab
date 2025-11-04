Operations with Neuron Attributes
==================================

Neuron attributes are automatically handled during common SpikeData operations like subsetting and concatenation.

Subsetting
----------

When you subset neurons, the attributes are automatically subsetted to match:

By Indices
^^^^^^^^^^

.. code-block:: python

   # Subset by indices
   sd_subset = sd.subset([0, 2, 4])
   
   # The neuron_attributes are automatically filtered
   print(sd_subset.neuron_attributes.to_dataframe())
   # Only rows for neurons 0, 2, 4 are kept

By Attribute Value
^^^^^^^^^^^^^^^^^^

You can also subset based on attribute values:

.. code-block:: python

   # Subset by attribute value
   good_neurons = sd.subset(['good'], by='quality')
   
   # Or subset by multiple values
   high_quality = sd.subset(['good', 'excellent'], by='quality')
   
   # Filter by numerical attribute
   df = sd.neuron_attributes.to_dataframe()
   high_snr_indices = df[df['snr'] > 5.0].index.tolist()
   sd_filtered = sd.subset(high_snr_indices)

Example: Multi-step Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   
   # Get attributes as DataFrame for complex queries
   df = sd.neuron_attributes.to_dataframe()
   
   # Combine multiple criteria
   good_units = df[
       (df['refractory_violations'] == 0) &      # No violations
       (df['firing_rate_hz'] > 1.0) &            # Active
       (df['firing_rate_hz'] < 50.0) &           # Not too fast
       (df['snr'] > 3.0)                         # Good SNR
   ]
   
   # Get indices and subset
   good_indices = good_units.index.tolist()
   sd_clean = sd.subset(good_indices)

Concatenation
-------------

When concatenating SpikeData objects, attributes are combined appropriately.

Time Concatenation (Append)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When appending along time (same neurons, extended recording), neuron attributes should match:

.. code-block:: python

   # Time concatenation (append) - keeps same neurons
   sd_combined = sd1.append(sd2)
   
   # Neuron attributes from sd1 are preserved
   # sd2's neuron_attributes are ignored (same neurons expected)

Neuron Concatenation
^^^^^^^^^^^^^^^^^^^^^

When combining neurons from different SpikeData objects:

.. code-block:: python

   # Neuron concatenation - adds new neurons
   sd1.concatenate_spike_data(sd2)
   
   # Attributes from both are combined
   # Result has attributes for all neurons from sd1 and sd2

If attributes don't match between objects, missing columns are filled with NaN or appropriate defaults.

Time Slicing
------------

When extracting time windows, neuron attributes are preserved:

.. code-block:: python

   # Extract a time window
   sd_window = sd.subtime(start=100.0, stop=500.0)
   
   # Neuron attributes are copied to the new object
   # (same neurons, different time range)

This preserves all neuron metadata like cluster IDs, quality metrics, etc.

Iterating Over Frames
----------------------

When iterating over time frames, each frame inherits neuron attributes:

.. code-block:: python

   # Iterate over time frames
   for frame in sd.frames(frame_size=1000.0, overlap=100.0):
       # Each frame has the same neuron attributes
       cluster_ids = frame.get_neuron_attribute('cluster_id')
       
       # Compute frame-specific metrics
       rates = frame.rates(bin_size=100.0)

Common Patterns
---------------

Quality-Based Filtering
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compute quality metrics
   sd.neuron_attributes.compute_isi_statistics(sd)
   
   # Filter based on multiple quality criteria
   df = sd.neuron_attributes.to_dataframe()
   
   high_quality = df[
       (df['refractory_violations'] == 0) &
       (df['cv_isi'] < 2.0) &
       (df['isi_violations'] < 0.01)
   ]
   
   sd_clean = sd.subset(high_quality.index.tolist())

Spatial Region Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Filter to a spatial region
   df = sd.neuron_attributes.to_dataframe()
   
   # Neurons in specific X-Y region
   region = df[
       (df['unit_x'] >= 100) & (df['unit_x'] <= 200) &
       (df['unit_y'] >= 50) & (df['unit_y'] <= 150)
   ]
   
   sd_region = sd.subset(region.index.tolist())

Cluster-Based Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Analyze each cluster separately
   df = sd.neuron_attributes.to_dataframe()
   
   for cluster_id in df['cluster_id'].unique():
       # Get neurons in this cluster
       cluster_neurons = df[df['cluster_id'] == cluster_id].index.tolist()
       sd_cluster = sd.subset(cluster_neurons)
       
       # Analyze cluster
       print(f"Cluster {cluster_id}: {sd_cluster.N} neurons")
       rates = sd_cluster.binned_meanrate(bin_size=100.0)

