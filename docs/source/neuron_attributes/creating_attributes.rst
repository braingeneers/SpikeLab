Creating Neuron Attributes
==========================

There are several ways to create SpikeData objects with neuron attributes attached.

From a Dictionary
-----------------

The simplest approach is to pass a dictionary of column names to lists of values:

.. code-block:: python

   from spikedata import SpikeData
   
   # Create spike trains
   trains = [...]  # Your spike time arrays
   
   # Create with neuron attributes from dict
   sd = SpikeData(
       trains,
       neuron_attributes={
           'unit_id': [101, 102, 103],
           'cluster_id': [1, 1, 2],
           'electrode_id': [0, 1, 1],
           'firing_rate_hz': [5.2, 8.1, 3.4],
           'snr': [4.2, 6.8, 3.1]
       }
   )

All lists must have the same length (equal to the number of neurons).

From a DataFrame
----------------

You can also pass a pandas DataFrame directly:

.. code-block:: python

   import pandas as pd
   from spikedata import SpikeData
   
   # Create a DataFrame with neuron metadata
   attrs_df = pd.DataFrame({
       'unit_id': [101, 102, 103],
       'cluster_id': [1, 1, 2],
       'quality': ['good', 'good', 'mua']
   })
   
   sd = SpikeData(trains, neuron_attributes=attrs_df)

The DataFrame index doesn't matter; it will be reset to match neuron indices.

Using NeuronAttributes Directly
--------------------------------

For more control, create a ``NeuronAttributes`` object first:

.. code-block:: python

   from spikedata import SpikeData, NeuronAttributes
   
   attrs = NeuronAttributes.from_dict({
       'unit_id': [1, 2, 3],
       'cluster_id': [10, 20, 30]
   }, n_neurons=3)
   
   sd = SpikeData(trains, neuron_attributes=attrs)

Starting Empty
--------------

If you don't have attributes at creation time, you can add them later or start with minimal information:

.. code-block:: python

   from spikedata import SpikeData, NeuronAttributes
   
   # Create without attributes
   sd = SpikeData(trains)
   
   # Initialize empty NeuronAttributes
   sd.neuron_attributes = NeuronAttributes.from_dict(
       {'unit_id': list(range(sd.N))},
       n_neurons=sd.N
   )
   
   # Add attributes later
   sd.set_neuron_attribute('cluster_id', [1, 1, 2, 2, 3])

Standard Column Names
---------------------

While you can use any column names, the following standard names are recommended for consistency:

**Core Attributes:**

* ``unit_id``: Unique identifier for each neuron
* ``cluster_id``: Cluster assignment (multiple neurons can share a cluster_id)
* ``electrode_id``: Physical electrode identifier
* ``channel``: Recording channel number
* ``firing_rate_hz``: Mean firing rate in Hz

**Quality Metrics:**

* ``snr``: Signal-to-noise ratio
* ``amplitude``: Spike amplitude (μV or arbitrary units)
* ``isolation_distance``: Cluster isolation quality metric
* ``isi_violations``: Inter-spike interval violation rate
* ``refractory_violations``: Count of ISIs < 2ms

**Spatial Location:**

* ``unit_x``, ``unit_y``, ``unit_z``: Unit position coordinates (μm)
* ``electrode_x``, ``electrode_y``, ``electrode_z``: Electrode coordinates (μm)

**ISI-based Temporal Patterns:**

* ``mean_isi_ms``: Mean interspike interval in milliseconds
* ``median_isi_ms``: Median ISI (robust to bursting)
* ``cv_isi``: Coefficient of variation (< 0.5: regular, ~1.0: Poisson, > 1.0: irregular/bursting)
* ``isi_skewness``: Distribution skewness (positive = bursting tendency)
* ``burst_index``: Fraction of ISIs < 10ms
* ``pause_ratio``: Fraction of ISIs > 100ms

**Latency Statistics:**

* ``mean_latency_ms``: Mean latency relative to reference neuron
* ``median_latency_ms``: Median latency (robust to outliers)
* ``latency_jitter_ms``: Standard deviation of latencies

**Burst Participation:**

* ``burst_participation``: Fraction of network bursts neuron participates in (0-1)
* ``is_backbone_unit``: Boolean indicating backbone classification

**Waveform Attributes:**

* ``mean_waveform``: Average spike waveform
* ``waveform_channel``: Channel used for waveform extraction
* ``waveform_n_spikes``: Number of spikes averaged

**Note on Spatial Coordinates:**

Store spatial coordinates as separate x, y, z columns rather than as arrays or tuples. This makes filtering and querying much easier. Combine them using ``np.column_stack()`` when needed for computations.

.. code-block:: python

   # Good: Separate columns
   sd.neuron_attributes = {
       'unit_x': [100.0, 150.0, 200.0],
       'unit_y': [50.0, 75.0, 100.0],
       'unit_z': [0.0, 0.0, 50.0]
   }
   
   # Easy to filter
   df = sd.neuron_attributes.to_dataframe()
   high_neurons = df[df['unit_z'] > 25.0]
   
   # Easy to combine for computation
   import numpy as np
   positions = np.column_stack([df['unit_x'], df['unit_y'], df['unit_z']])

