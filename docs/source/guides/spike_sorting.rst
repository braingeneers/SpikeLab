==========================
Spike Sorting and Curation
==========================

SpikeLab includes a spike-sorting pipeline in the ``spikelab.spike_sorting``
sub-package. It supports three sorting algorithms — Kilosort2
(Pachitariu et al. 2016), Kilosort4 (Pachitariu et al. 2024), and RT-Sort
(Van der Molen et al. 2024) — behind a unified interface built on
SpikeInterface (Buccino et al. 2020). The pipeline returns curated
:class:`~spikelab.SpikeData` objects ready for downstream analysis.

SpikeLab also provides standalone curation methods that can be used on any
``SpikeData`` object, whether it came from the sorting pipeline or from an
external source.

**References:**

- Pachitariu, M., Steinmetz, N., Kadir, S., Carandini, M. & Harris, K. D.
  Kilosort: realtime spike-sorting for extracellular electrophysiology with
  hundreds of channels. *bioRxiv* (2016).
- Pachitariu, M., Sridhar, S., Pennington, J. & Stringer, C. Spike sorting
  with Kilosort4. *Nature Methods* 21, 914--921 (2024).
- Van der Molen, T., Lim, M., Bartram, J. et al. RT-Sort: An action potential
  propagation-based algorithm for real time spike detection and sorting with
  millisecond latencies. *PLoS ONE* 19, e0312438 (2024).
- Buccino, A. P., Hurwitz, C. L., Garcia, S. et al. SpikeInterface, a unified
  framework for spike sorting. *eLife* 9, e61834 (2020).

.. contents:: On this page
   :local:
   :depth: 2


Spike Sorting
-------------

Prerequisites
^^^^^^^^^^^^^

The sorting pipeline requires external dependencies that are **not** installed
with SpikeLab by default:

- **Kilosort2** requires MATLAB (R2019b+) and the `Kilosort2 repository
  <https://github.com/MouseLand/Kilosort2>`_. A Docker variant is available
  that removes the MATLAB requirement.
- **Kilosort4** is pure Python but requires ``torch`` and ``kilosort``.
  A Docker variant is also available.
- **RT-Sort** requires ``torch`` for its neural-network spike detection model.

For Maxwell Biosystems ``.h5`` files the HDF5 decompression plugin must also
be installed; follow the instructions printed by the loader if the plugin is
missing.

Basic usage
^^^^^^^^^^^

The main entry point is :func:`~spikelab.spike_sorting.sort_recording`, which
accepts a list of recording files and returns a list of
:class:`~spikelab.SpikeData` objects:

.. code-block:: python

   from spikelab.spike_sorting import sort_recording

   results = sort_recording(
       recording_files=["session1.raw.h5"],
       sorter="kilosort4",
   )

   sd = results[0]
   print(sd.N, "units")
   print(sd.length / 1000, "seconds")

The ``sorter`` parameter selects the algorithm: ``"kilosort2"``,
``"kilosort4"``, or ``"rt_sort"``.

Configuration and presets
^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline is configured via a :class:`~spikelab.spike_sorting.config.SortingPipelineConfig`
dataclass composed of sub-configs for recording I/O, sorting, curation,
waveform extraction, and execution. Pre-built presets provide sensible defaults:

.. code-block:: python

   from spikelab.spike_sorting.config import KILOSORT4

   results = sort_recording(
       recording_files=["session1.raw.h5"],
       config=KILOSORT4,
   )

To customise a preset, use the ``override`` method:

.. code-block:: python

   config = KILOSORT4.override(
       fr_min=0.1,             # stricter minimum firing rate (Hz)
       snr_min=6.0,            # stricter SNR threshold
       n_jobs=16,
       total_memory="32G",
   )
   results = sort_recording(["session1.raw.h5"], config=config)

Individual parameters can also be passed directly as keyword arguments to
``sort_recording``, which builds a config internally:

.. code-block:: python

   results = sort_recording(
       recording_files=["session1.raw.h5"],
       sorter="kilosort2",
       kilosort_path="/opt/Kilosort2",
       fr_min=0.1,
       n_jobs=8,
   )

Multi-stream recordings
^^^^^^^^^^^^^^^^^^^^^^^

For multi-stream files (e.g. Maxwell multi-well ``.raw.h5``), use
:func:`~spikelab.spike_sorting.sort_multistream`:

.. code-block:: python

   from spikelab.spike_sorting import sort_multistream

   stream_results = sort_multistream(
       recording="multiwell.raw.h5",
       stream_ids=["well000", "well001"],
       sorter="kilosort4",
   )

   for stream_id, sds in stream_results.items():
       print(f"{stream_id}: {sds[0].N} units")

This returns a dict mapping stream IDs to lists of ``SpikeData`` objects.

Reusing intermediate results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline caches intermediate files (binary recordings, sorting output,
waveforms). Control which stages are re-run with the ``recompute_*`` flags:

.. code-block:: python

   results = sort_recording(
       recording_files=["session1.raw.h5"],
       sorter="kilosort4",
       recompute_recording=False,   # reuse existing binary
       recompute_sorting=True,      # force re-sort
       reextract_waveforms=True,    # force waveform re-extraction
   )

See the :doc:`API reference </api/spike_sorting>` for the full configuration
options.


Stimulation Artifact Removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When sorting recordings with electrical stimulation, stim artifacts must be
removed before spike detection. SpikeLab provides
:func:`~spikelab.spike_sorting.stim_sorting.artifact_removal.remove_stim_artifacts`
for this purpose.

Two methods are available:

- **polynomial** (default) — fits a low-order polynomial to the
  post-saturation artifact tail and subtracts it, preserving neural spikes
  that ride on the artifact decay. Saturated samples are blanked (zeroed).
- **blank** — zeros the entire artifact window. Simpler but discards any
  spikes within the window.

Stim times logged by the hardware may not align exactly with the artifact in
the recording. Use
:func:`~spikelab.spike_sorting.stim_sorting.recentering.recenter_stim_times`
to find the true artifact onset:

.. code-block:: python

   from spikelab.spike_sorting.stim_sorting.recentering import recenter_stim_times
   from spikelab.spike_sorting.stim_sorting.artifact_removal import remove_stim_artifacts

   # Recenter stim times to the actual artifact onset
   corrected_times = recenter_stim_times(
       traces,                    # (channels, samples) raw voltage
       stim_times_ms,             # logged stim times in ms
       fs_Hz=20000,
       peak_mode="down_edge",     # alignment mode for biphasic pulses
   )

   # Remove artifacts
   cleaned, blanked_mask = remove_stim_artifacts(
       traces,
       corrected_times,
       fs_Hz=20000,
       method="polynomial",
       artifact_window_ms=10.0,   # max tail duration after desaturation
       poly_order=3,              # cubic polynomial (default)
   )

The ``peak_mode`` parameter controls how each artifact is aligned:

- ``"abs_max"`` — largest absolute voltage (monophasic pulses)
- ``"down_edge"`` — positive-to-negative zero-crossing (biphasic
  anodic-first)
- ``"up_edge"`` — negative-to-positive zero-crossing (biphasic
  cathodic-first)
- ``"pos_peak"`` / ``"neg_peak"`` — largest positive or most negative voltage

When multiple stim events occur in rapid succession (e.g. burst or
paired-pulse protocols), the module automatically detects whether the signal
has returned to baseline between events and extends the blanking region
across the entire burst if needed.

The polynomial detrend approach is related to SALPA, adapted for offline use:

- Wagenaar, D. A. & Potter, S. M. Real-time multi-channel stimulus artifact
  suppression by local curve fitting. *J Neurosci Methods* 120, 113--120 (2002).


Unit Curation
-------------

SpikeLab provides curation methods that filter units by quality metrics.
These work on any :class:`~spikelab.SpikeData` object — not just output from
the sorting pipeline.

Each curation method returns a tuple ``(sd_curated, result_dict)`` where
``result_dict`` contains:

- ``"metric"`` — ``np.ndarray (N,)`` with the per-unit metric for **all**
  original units.
- ``"passed"`` — ``np.ndarray (N,)`` boolean mask of units that passed.

Individual criteria
^^^^^^^^^^^^^^^^^^^

Apply a single quality criterion at a time:

.. code-block:: python

   # Remove units with fewer than 50 spikes
   sd_curated, res = sd.curate_by_min_spikes(min_spikes=50)

   # Remove units below 0.1 Hz firing rate
   sd_curated, res = sd.curate_by_firing_rate(min_rate_hz=0.1)

   # Remove units with > 1% ISI violations
   sd_curated, res = sd.curate_by_isi_violations(
       max_violation=1.0, threshold_ms=1.5,
   )

   # Remove units with low SNR
   sd_curated, res = sd.curate_by_snr(min_snr=5.0)

   # Remove units with inconsistent waveforms
   sd_curated, res = sd.curate_by_std_norm(max_std_norm=1.0)

SNR and waveform consistency (``curate_by_snr``, ``curate_by_std_norm``)
require that the ``SpikeData`` object has ``raw_data`` attached. If the
metrics have not been pre-computed, call
:meth:`~spikelab.SpikeData.compute_waveform_metrics` first:

.. code-block:: python

   sd, metrics = sd.compute_waveform_metrics()
   sd_curated, res = sd.curate_by_snr(min_snr=5.0)

Merge-based deduplication
^^^^^^^^^^^^^^^^^^^^^^^^^

When spike sorting produces duplicate units for the same neuron recorded on
adjacent electrodes, you can merge them using
:meth:`~spikelab.SpikeData.curate_by_merge_duplicates`. This finds spatially
nearby unit pairs, filters by ISI violation rate and waveform cosine
similarity, then greedily merges accepted pairs:

.. code-block:: python

   sd_merged, res = sd.curate_by_merge_duplicates(
       dist_um=24.8,            # max inter-electrode distance in um
       cosine_threshold=0.5,    # min waveform similarity
       max_isi_increase=0.04,   # reject merge if ISI violations rise too much
   )

   n_absorbed = (~res["passed"]).sum()
   print(f"Merged {n_absorbed} duplicate units")

This requires ``neuron_attributes`` with position and ``avg_waveform`` entries,
which are set automatically when the ``SpikeData`` comes from the sorting
pipeline.

Batch curation
^^^^^^^^^^^^^^

To apply multiple criteria in one call, use
:meth:`~spikelab.SpikeData.curate`. Only criteria with non-``None`` values
are applied:

.. code-block:: python

   sd_curated, results = sd.curate(
       min_spikes=50,
       min_rate_hz=0.1,
       isi_max=1.0,
       min_snr=5.0,
   )

   # results contains per-criterion entries
   for criterion, res in results.items():
       n_removed = (~res["passed"]).sum()
       print(f"{criterion}: removed {n_removed} units")

Curation history
^^^^^^^^^^^^^^^^

For reproducibility, build a serialisable record of what was removed and why:

.. code-block:: python

   history = sd.build_curation_history(
       sd_original=sd_raw,
       sd_curated=sd_curated,
       results=results,
   )

The returned dict is JSON-serialisable and can be stored in a workspace or
saved alongside the curated data.


Sorting Concatenated Recordings
--------------------------------

When a directory containing multiple recording files is passed to
``sort_recording``, the pipeline concatenates them into a single recording for
sorting. The returned ``SpikeData`` objects are automatically split back into
per-recording epochs. When a list of recording paths is passed instead, each
file is processed sequentially without concatenation.

If you need to re-split a concatenated ``SpikeData`` manually (e.g. after
loading a saved pickle that was not yet split), use
:meth:`~spikelab.SpikeData.split_epochs`. This requires ``rec_chunks_ms`` in
``metadata`` (set automatically by the sorting pipeline) and time-shifts each
epoch to start at t=0:

.. code-block:: python

   epoch_sds = sd.split_epochs()

   for i, epoch_sd in enumerate(epoch_sds):
       print(f"Epoch {i}: {epoch_sd.N} units, {epoch_sd.length:.0f} ms")


Handling Sort Failures
----------------------

When a sorting run fails, SpikeLab can classify the failure into one of three
categories so that batch scripts can implement skip/retry/stop policies without
parsing generic error messages.

The three categories are:

- **BiologicalSortFailure** -- the recording itself cannot be sorted (too
  silent, all channels bad, no waveforms to compute metrics on). Recommended
  policy: mark the target as not-sortable and move on.
- **EnvironmentSortFailure** -- the host environment or container runtime is
  misconfigured (Docker down, HDF5 plugin missing). Recommended policy: stop
  and fix the environment.
- **ResourceSortFailure** -- the job exhausted a machine resource (GPU out of
  memory). Recommended policy: retry with reduced parameters.

All three inherit from
:class:`~spikelab.spike_sorting._exceptions.SpikeSortingClassifiedError`, which
in turn inherits from ``RuntimeError``.

Each sorter has a dedicated classifier that inspects both sorter logs and
exception chains to identify known failure signatures:

- ``classify_ks2_failure`` — Kilosort2
- ``classify_ks4_failure`` — Kilosort4
- ``classify_rt_sort_failure`` — RT-Sort

.. code-block:: python

   from spikelab.spike_sorting._classifier import (
       classify_ks2_failure,
       classify_ks4_failure,
       classify_rt_sort_failure,
   )
   from spikelab.spike_sorting._exceptions import (
       BiologicalSortFailure,
       EnvironmentSortFailure,
       ResourceSortFailure,
   )

   # Pick the classifier matching your sorter
   classify = classify_ks4_failure  # or classify_ks2_failure, classify_rt_sort_failure

   try:
       results = sort_recording(["session1.raw.h5"], sorter="kilosort4")
   except Exception as exc:
       classified = classify(output_folder, exc)
       if classified is not None:
           if isinstance(classified, BiologicalSortFailure):
               print(f"Skipping (biology): {classified}")
           elif isinstance(classified, EnvironmentSortFailure):
               raise  # stop the batch
           elif isinstance(classified, ResourceSortFailure):
               print(f"Retry with smaller batch: {classified}")
       else:
           raise  # unrecognised failure

Specific exception classes carry diagnostic attributes. For example,
:class:`~spikelab.spike_sorting._exceptions.InsufficientActivityError` exposes
``threshold_crossings``, ``units_at_failure``, and ``nspks_at_failure`` parsed
from sorter logs. RT-Sort additionally raises
:class:`~spikelab.spike_sorting._exceptions.ModelLoadingError` when the
detection model cannot be loaded. See the
:doc:`API reference </api/spike_sorting>` for the full exception hierarchy.
