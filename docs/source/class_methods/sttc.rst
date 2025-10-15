Spike Time Tiling Coefficient
==============================

spike_time_tiling(i, j, delt=20.0)
-----------------------------------

Compute the spike time tiling coefficient (STTC) between two neurons.

**Parameters:**

* ``i`` (int): Index of first neuron.
* ``j`` (int): Index of second neuron.
* ``delt`` (float): Window size (ms).

**Returns:**

* ``float``: STTC value.

----

spike_time_tilings(delt=20.0)
------------------------------

Compute the pairwise STTC matrix for all neuron pairs.

**Parameters:**

* ``delt`` (float): Window size (ms).

**Returns:**

* ``np.ndarray``: STTC matrix (neurons × neurons).

