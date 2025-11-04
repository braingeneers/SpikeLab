Burst Analysis
==============

get_frac_active(edges, MIN_SPIKES, backbone_threshold)
-------------------------------------------------------

Calculate the fraction of active neurons in defined time windows (bursts) and identify backbone units that consistently participate.

**Parameters:**

* ``edges`` (np.ndarray): Array of shape (n_bursts, 2) defining burst time windows in milliseconds. Each row is [start, end] time.
* ``MIN_SPIKES`` (int): Minimum number of spikes required for a neuron to be considered "active" in a burst.
* ``backbone_threshold`` (float): Fraction of bursts (0-1) a neuron must participate in to be classified as a backbone unit.

**Returns:**

* **tuple**: (frac_per_unit, frac_per_burst, backbone_units)
  
  - ``frac_per_unit`` (np.ndarray): Fraction of bursts each neuron participates in (length N).
  - ``frac_per_burst`` (np.ndarray): Fraction of neurons active in each burst (length n_bursts).
  - ``backbone_units`` (np.ndarray): Boolean array indicating backbone units (length N).

**Description:**

This method analyzes network burst participation by:

1. Counting how many spikes each neuron fires during each burst window
2. Marking neurons as "active" in a burst if they exceed MIN_SPIKES
3. Computing participation fractions for both neurons and bursts
4. Identifying "backbone units" that consistently participate across bursts

**Example:**

.. code-block:: python

   import numpy as np
   
   # Define burst time windows (in milliseconds)
   burst_edges = np.array([
       [100.0, 200.0],   # Burst 1: 100-200 ms
       [500.0, 650.0],   # Burst 2: 500-650 ms
       [1000.0, 1150.0]  # Burst 3: 1000-1150 ms
   ])
   
   # Calculate burst participation
   frac_per_unit, frac_per_burst, is_backbone = sd.get_frac_active(
       edges=burst_edges,
       MIN_SPIKES=5,           # Neuron needs ≥5 spikes to be "active"
       backbone_threshold=0.7   # Must participate in ≥70% of bursts
   )
   
   # Find backbone neurons
   backbone_indices = np.where(is_backbone)[0]
   print(f"Backbone neurons: {backbone_indices}")
   
   # Analyze participation rates
   print(f"Mean participation per neuron: {frac_per_unit.mean():.2f}")
   print(f"Neurons active per burst: {frac_per_burst * sd.N}")

**Use with NeuronAttributes:**

For automatic storage and caching, use the ``NeuronAttributes.compute_burst_participation()`` method instead:

.. code-block:: python

   # Automatically saves to neuron_attributes
   burst_stats = sd.neuron_attributes.compute_burst_participation(
       sd,
       burst_edges=burst_edges,
       min_spikes=5,
       backbone_threshold=0.7
   )
   
   # Access saved attributes
   participation = sd.neuron_attributes.get_attribute('burst_participation')
   is_backbone = sd.neuron_attributes.get_attribute('is_backbone_unit')

See :doc:`/neuron_attributes/computing_metrics` for details on the caching methods.

