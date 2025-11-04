Working with Attributes
=======================

Once you have a SpikeData object with neuron attributes, you can get and set attributes, compute firing rates, and work with spatial locations.

Setting Attributes
------------------

Add or update attributes using ``set_neuron_attribute()``:

.. code-block:: python

   # Set a new attribute
   sd.set_neuron_attribute('quality', ['good', 'good', 'mua'])
   
   # Set quality metrics
   sd.set_neuron_attribute('snr', [4.2, 5.1, 3.8])
   
   # Update existing attribute
   sd.set_neuron_attribute('cluster_id', [1, 2, 2])

The value must be a list or array with length equal to the number of neurons.

Getting Attributes
------------------

Retrieve attributes using ``get_neuron_attribute()``:

.. code-block:: python

   # Get a specific attribute as numpy array
   cluster_ids = sd.get_neuron_attribute('cluster_id')
   
   # Get multiple attributes
   snr_values = sd.get_neuron_attribute('snr')
   quality = sd.get_neuron_attribute('quality')
   
   # Access the full DataFrame
   df = sd.neuron_attributes.to_dataframe()
   print(df)

The DataFrame includes all attributes with neuron indices as the index.

Computing and Storing Firing Rates
-----------------------------------

The ``compute_firing_rates()`` method computes firing rates and automatically stores them in neuron_attributes:

.. code-block:: python

   # Compute firing rates and store in neuron_attributes
   rates = sd.compute_firing_rates(unit='Hz')
   
   # Access stored rates later
   rates = sd.get_neuron_attribute('firing_rate_hz')
   
   # Also works with per-second rates
   rates_per_s = sd.compute_firing_rates(unit='per_second')

The firing rate is stored as ``firing_rate_hz`` for consistency.

Working with Spatial Locations
-------------------------------

Store spatial coordinates as separate x, y, z columns for easy manipulation:

.. code-block:: python

   # Store as separate x, y, z coordinates (recommended)
   sd.set_neuron_attribute('unit_x', [100.5, 150.2, 200.8])
   sd.set_neuron_attribute('unit_y', [50.3, 75.1, 100.0])
   sd.set_neuron_attribute('unit_z', [0.0, 0.0, 50.0])
   
   # Access spatial data
   x_coords = sd.get_neuron_attribute('unit_x')
   
   # Combine coordinates for analysis
   import numpy as np
   df = sd.neuron_attributes.to_dataframe()
   positions = np.column_stack([df['unit_x'], df['unit_y'], df['unit_z']])
   # positions is now a (N, 3) array of coordinates
   
   # Same approach for electrode locations
   sd.set_neuron_attribute('electrode_x', [100.0, 150.0, 200.0])
   sd.set_neuron_attribute('electrode_y', [50.0, 75.0, 100.0])
   sd.set_neuron_attribute('electrode_z', [0.0, 0.0, 50.0])

Spatial Filtering Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Filter neurons within a spatial region
   df = sd.neuron_attributes.to_dataframe()
   
   # Find neurons in a specific X range (100-200 μm)
   in_region = (df['unit_x'] >= 100) & (df['unit_x'] <= 200)
   region_indices = df[in_region].index.tolist()
   sd_region = sd.subset(region_indices)
   
   # Find neurons near a specific electrode
   electrode_neurons = sd.subset([1], by='electrode_id')
   
   # Calculate distances from a reference point
   ref_point = np.array([150.0, 75.0, 10.0])
   positions = np.column_stack([
       df['unit_x'].values,
       df['unit_y'].values,
       df['unit_z'].values
   ])
   distances = np.linalg.norm(positions - ref_point, axis=1)
   
   # Store distances as an attribute
   sd.set_neuron_attribute('distance_from_ref', distances)
   
   # Filter to neurons within 100 μm of reference
   nearby = distances < 100.0
   nearby_indices = [i for i, keep in enumerate(nearby) if keep]
   sd_nearby = sd.subset(nearby_indices)

Checking for Attributes
------------------------

Check if attributes exist before accessing:

.. code-block:: python

   # Check if neuron_attributes exist
   if sd.neuron_attributes is not None:
       df = sd.neuron_attributes.to_dataframe()
       print(df.columns)  # See available attributes
   
   # Check if specific column exists
   if sd.neuron_attributes is not None:
       df = sd.neuron_attributes.to_dataframe()
       if 'cluster_id' in df.columns:
           cluster_ids = sd.get_neuron_attribute('cluster_id')

