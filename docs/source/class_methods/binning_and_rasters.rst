Binning and Rasters
===================

binned(bin_size)
----------------

Return a binned spike count array for all neurons.

**Parameters:**

* ``bin_size`` (float): Bin width in ms.

**Returns:**

* ``np.ndarray``: 2D array (neurons × bins) of spike counts.

----

raster(bin_size=1.0)
--------------------

Return a dense binary raster (neurons × time bins) of spike events.

**Parameters:**

* ``bin_size`` (float, optional): Bin width in ms (default: 1.0).

**Returns:**

* ``np.ndarray``: Binary matrix of shape (neurons, bins).

----

sparse_raster(bin_size=1.0)
----------------------------

Return a sparse matrix representation of the spike raster.

**Parameters:**

* ``bin_size`` (float, optional): Bin width in ms (default: 1.0).

**Returns:**

* ``scipy.sparse.csr_matrix``: Sparse binary raster.

----

channel_raster(bin_size=20.0, channel_attr=None)
--------------------------------------------------

Create a raster aggregated by channel instead of neuron.

Returns a dense array where entry (c, j) is the total number of spikes from all neurons on channel c in bin j. Channels are determined from ``neuron_attributes`` using the same logic as ``neuron_to_channel_map()``.

**Parameters:**

* ``bin_size`` (float, optional): Bin size in milliseconds (same as ``raster()``). Default: 20.0.
* ``channel_attr`` (str, optional): Name of the attribute in ``neuron_attributes`` that contains the channel index. If None, searches for common attribute names. See ``neuron_to_channel_map()`` for details.

**Returns:**

* ``np.ndarray``: Dense array of shape (n_channels, n_bins) where n_channels is the number of unique channels found.

**Raises:**

* ``ValueError``: If ``neuron_attributes`` is None or no channel information can be found.

**Example:**

.. code-block:: python

   from dataclasses import dataclass
   
   @dataclass
   class NeuronAttrs:
       channel: int
   
   # Create 6 neurons: 0,1 on channel 0; 2,3 on channel 1; 4,5 on channel 2
   attrs = [NeuronAttrs(channel=i // 2) for i in range(6)]
   trains = [[10.0, 20.0], [15.0], [25.0], [30.0], [35.0], [40.0]]
   sd = SpikeData(trains, neuron_attributes=attrs, length=50.0)
   ch_raster = sd.channel_raster(bin_size=10.0)
   # ch_raster.shape -> (3, n_bins)  # 3 channels
   # ch_raster[0, :].sum() -> 3  # Channel 0 has 3 spikes total

