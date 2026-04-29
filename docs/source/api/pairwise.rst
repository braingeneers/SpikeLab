====================
Pairwise Comparisons
====================

The pairwise module provides data structures for unit-by-unit comparison
matrices. ``PairwiseCompMatrix`` holds a single ``(N, N)`` matrix (e.g.,
correlation or distance between all pairs of units), while
``PairwiseCompMatrixStack`` holds an ``(N, N, S)`` stack of such matrices
across multiple conditions or time windows.

PairwiseCompMatrix
------------------

.. autoclass:: spikelab.spikedata.pairwise.PairwiseCompMatrix
   :members:
   :show-inheritance:
   :exclude-members: matrix, labels, metadata

PairwiseCompMatrixStack
-----------------------

.. autoclass:: spikelab.spikedata.pairwise.PairwiseCompMatrixStack
   :members:
   :show-inheritance:
   :exclude-members: stack, labels, times, metadata
