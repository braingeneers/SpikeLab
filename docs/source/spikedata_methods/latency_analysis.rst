Latency Analysis
================

latencies(times, window_ms=100.0)
----------------------------------

Compute latencies from given times to nearest spikes in each train within a window.

**Parameters:**

* ``times`` (list or array): Reference times.
* ``window_ms`` (float): Maximum latency window (ms).

**Returns:**

* ``List[List[float]]``: Latencies per neuron per reference time.

----

latencies_to_index(i, window_ms=100.0)
---------------------------------------

Compute latencies from all spikes in neuron ``i`` to all other neurons.

**Parameters:**

* ``i`` (int): Index of reference neuron.
* ``window_ms`` (float): Maximum latency window (ms).

**Returns:**

* ``List[List[float]]``: Latencies per neuron.

