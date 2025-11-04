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

**Note:** This method performs the actual computation. For cached access with automatic memoization, use ``get_sttc_matrix()`` instead.

----

get_sttc_matrix(delt=20.0, use_cache=True)
-------------------------------------------

Compute or retrieve cached STTC matrix for all neuron pairs. This is a caching wrapper around ``spike_time_tilings()`` that stores results to avoid recomputation.

**Parameters:**

* ``delt`` (float): Window size (ms). Default: 20.0
* ``use_cache`` (bool): If True (default), return cached result if available for this delt value. If False, force recomputation.

**Returns:**

* ``np.ndarray``: STTC matrix (neurons × neurons).

**Performance:**

STTC computation is expensive (O(N²) complexity). Caching provides dramatic speedups
for cached calls.

**Example:**

.. code-block:: python

   # First call - computes and caches
   sttc = sd.get_sttc_matrix(delt=20.0)
   
   # Subsequent calls - instant retrieval
   sttc_again = sd.get_sttc_matrix(delt=20.0)  # Uses cache
   
   # Different delt values are cached separately
   sttc_40 = sd.get_sttc_matrix(delt=40.0)  # Computes new
   sttc_20 = sd.get_sttc_matrix(delt=20.0)  # Still cached
   
   # Force recomputation
   sttc_fresh = sd.get_sttc_matrix(delt=20.0, use_cache=False)

**Memory Management:**

Cache persists across operations. Clear it when:

- After subsetting (old cache no longer valid)
- To free memory when done with analysis
- Before saving to reduce file size

Use ``clear_sttc_cache()`` to manage cache.

----

clear_sttc_cache(delt=None)
----------------------------

Clear cached STTC matrices to free memory.

**Parameters:**

* ``delt`` (float | None): If specified, clear only the cache for this delt value. If None (default), clear all cached STTC matrices.

**Example:**

.. code-block:: python

   # Clear specific delt value
   sd.clear_sttc_cache(delt=20.0)
   
   # Clear all cached matrices
   sd.clear_sttc_cache()
   
   # Clear before subsetting
   sd_subset = sd.subset([0, 1, 2])
   sd_subset.clear_sttc_cache()  # Old cache invalid
   
   # Clear before saving to reduce file size
   sd.clear_sttc_cache()
   sd.to_hdf5('output.h5', style='ragged')

**Recommendations:**

1. Use ``get_sttc_matrix()`` instead of ``spike_time_tilings()`` for analysis workflows
2. Cache persists, so clear after subsetting or time slicing
3. Clear before exporting to reduce file size

