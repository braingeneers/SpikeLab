=============
Spike Sorting
=============

The ``spikelab.spike_sorting`` sub-package provides a full spike-sorting
pipeline: loading raw recordings, running a sorter backend (Kilosort2,
Kilosort4, or RT-Sort), extracting waveforms, curating units, and compiling
results into :class:`~spikelab.SpikeData` objects.

See the :doc:`/guides/spike_sorting` guide for usage examples and environment
setup instructions.


Entry Points
------------

.. autofunction:: spikelab.spike_sorting.sort_recording

.. autofunction:: spikelab.spike_sorting.sort_multistream


Configuration
-------------

.. automodule:: spikelab.spike_sorting.config
   :members:
   :show-inheritance:


Backend Registry
----------------

.. automodule:: spikelab.spike_sorting.backends
   :members:
   :show-inheritance:

.. autoclass:: spikelab.spike_sorting.backends.base.SorterBackend
   :members:
   :show-inheritance:


Classified Exceptions
---------------------

When a sort fails, SpikeLab can classify the failure into one of three
categories so that callers can implement skip/retry/stop policies without
parsing generic error messages.

.. automodule:: spikelab.spike_sorting._exceptions
   :members:
   :show-inheritance:


Post-Failure Classifiers
------------------------

The classifier module inspects sorter logs and exception chains to produce
specific :class:`~spikelab.spike_sorting._exceptions.SpikeSortingClassifiedError`
subclasses from generic failures.

.. autofunction:: spikelab.spike_sorting._classifier.classify_ks2_failure

.. autofunction:: spikelab.spike_sorting._classifier.classify_ks4_failure
