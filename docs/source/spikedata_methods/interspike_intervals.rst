Interspike Intervals
====================

interspike_intervals()
----------------------

Return a list of interspike intervals for each neuron.

**Returns:**

* ``List[np.ndarray]``: Each array contains ISIs for one neuron.

----

resampled_isi(num_samples=1000)
--------------------------------

Return a resampled distribution of interspike intervals for all neurons.

**Parameters:**

* ``num_samples`` (int): Number of samples to draw.

**Returns:**

* ``np.ndarray``: Pooled, resampled ISIs.

