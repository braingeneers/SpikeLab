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

.. autofunction:: spikelab.spike_sorting._classifier.classify_rt_sort_failure


Sort Run Reports
----------------

``sort_recording`` can return a structured per-run report via the
``out_report=`` keyword argument, capturing per-recording status, timings,
and any classified failure.

.. autoclass:: spikelab.spike_sorting.pipeline.SortRunReport
   :members:
   :show-inheritance:

.. autoclass:: spikelab.spike_sorting.pipeline.RecordingResult
   :members:
   :show-inheritance:

After a successful sort, the pipeline writes a human-readable
``sorting_report.md`` next to the results. The functions below let you
regenerate it manually or extract its components programmatically.

.. autofunction:: spikelab.spike_sorting.report.generate_sorting_report

.. autofunction:: spikelab.spike_sorting.report.parse_sorting_log

.. autofunction:: spikelab.spike_sorting.report.extract_unit_quality_stats


Resource Guards
---------------

The pipeline ships with a set of preflight checks and live watchdogs that
run automatically during a sort. Most users never need to touch these
directly — they are configured via :class:`~spikelab.spike_sorting.config.ExecutionConfig`
and surface as classified exceptions when triggered. The pieces below are
exposed for advanced users who want to run preflight checks standalone or
inspect watchdog state.

.. autofunction:: spikelab.spike_sorting.guards.run_preflight

.. autoclass:: spikelab.spike_sorting.guards.PreflightFinding
   :members:

.. autoclass:: spikelab.spike_sorting.guards.HostMemoryWatchdog
   :members:

.. autoclass:: spikelab.spike_sorting.guards.GpuMemoryWatchdog
   :members:

.. autoclass:: spikelab.spike_sorting.guards.DiskUsageWatchdog
   :members:

.. autoclass:: spikelab.spike_sorting.guards.LogInactivityWatchdog
   :members:

.. autoclass:: spikelab.spike_sorting.guards.IOStallWatchdog
   :members:

.. autoclass:: spikelab.spike_sorting.guards.DiskExhaustionReport
   :members:


Pipeline Canary
---------------

.. autofunction:: spikelab.spike_sorting.canary.run_canary
