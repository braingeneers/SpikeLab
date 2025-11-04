Data Selection and Manipulation
================================

subset(neuron_indices)
----------------------

Return a new SpikeData object containing only the specified neurons.

**Parameters:**

* ``neuron_indices`` (list or array): Indices of neurons to include.

**Returns:**

* ``SpikeData``: Subsetted object.

----

append(other)
-------------

Append spike trains from another SpikeData object.

**Parameters:**

* ``other`` (``SpikeData``): Another SpikeData instance.

**Returns:**

* ``None``

----

concatenate_spike_data(others)
-------------------------------

Concatenate multiple SpikeData objects along the neuron axis.

**Parameters:**

* ``others`` (list of ``SpikeData``): Objects to concatenate.

**Returns:**

* ``SpikeData``: Concatenated object.

