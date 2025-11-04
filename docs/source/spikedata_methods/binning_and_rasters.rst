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

