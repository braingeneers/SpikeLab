Firing Rates
============

rates(bin_size=100.0)
---------------------

Compute firing rates for each neuron in specified bins.

**Parameters:**

* ``bin_size`` (float): Bin width in ms.

**Returns:**

* ``np.ndarray``: Firing rates (Hz) per neuron per bin.

----

binned_meanrate(bin_size=100.0)
--------------------------------

Compute the mean firing rate for each neuron over the entire recording.

**Parameters:**

* ``bin_size`` (float): Bin width in ms.

**Returns:**

* ``np.ndarray``: Mean firing rate (Hz) per neuron.

