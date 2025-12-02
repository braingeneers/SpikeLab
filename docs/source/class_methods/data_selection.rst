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

----

neuron_to_channel_map(channel_attr=None)
-----------------------------------------

Return a mapping from neuron indices to channel indices.

Extracts channel information from ``neuron_attributes``. If ``channel_attr`` is not specified, attempts to find channel information using common attribute names: ``'channel'``, ``'channel_id'``, ``'channel_index'``, ``'ch'``, ``'channel_idx'``.

**Parameters:**

* ``channel_attr`` (str, optional): Name of the attribute in ``neuron_attributes`` that contains the channel index. If None, searches for common attribute names.

**Returns:**

* ``dict[int, int]``: Dictionary mapping neuron index (int) to channel index (int). If ``neuron_attributes`` is None or no channel information is found, returns an empty dict.

**Example:**

.. code-block:: python

   from dataclasses import dataclass
   
   @dataclass
   class NeuronAttrs:
       channel: int
   
   attrs = [NeuronAttrs(channel=i % 4) for i in range(10)]
   sd = SpikeData([[]] * 10, neuron_attributes=attrs, length=100.0)
   mapping = sd.neuron_to_channel_map()
   # mapping[0] -> 0 (neuron 0 maps to channel 0)
   # mapping[5] -> 1 (neuron 5 maps to channel 1)

