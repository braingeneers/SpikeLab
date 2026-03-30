============
Spike Sorting
============

SpikeLab ships a full Kilosort2 spike-sorting pipeline in the
``spikelab.spike_sorting`` sub-package.  The single public entry point,
:ref:`sort_with_kilosort2 <spikesorting-api>`, runs the sorter on one or more
raw recordings and returns the curated results as
:class:`~spikelab.spikedata.SpikeData` objects ready for downstream analysis.

.. contents:: On this page
   :local:
   :depth: 2


Prerequisites
-------------

The spike-sorting pipeline requires three external components that are **not**
installed automatically with SpikeLab:

1. **MATLAB** (R2019b or later recommended).
2. **Kilosort2** — clone from `MouseLand/Kilosort2
   <https://github.com/MouseLand/Kilosort2>`_ and note the path to the root
   directory (the folder that contains ``master_kilosort.m``).
3. **SpikeInterface** — ``pip install spikeinterface``.

For Maxwell Biosystems ``.h5`` files the HDF5 decompression plugin must also be
installed; follow the instructions printed by the loader if the plugin is
missing.


Basic usage
-----------

.. code-block:: python

   from spikelab.spike_sorting import sort_with_kilosort2

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/session1.raw.h5"],
       kilosort_path="/opt/Kilosort2",
   )

   sd = results[0]          # SpikeData for the first (and only) recording
   print(sd.N)              # number of curated units
   print(sd.length / 1000)  # recording duration in seconds

``sort_with_kilosort2`` accepts a **list** of recording paths and always
returns a **list** of :class:`~spikelab.spikedata.SpikeData` objects — one per
successfully sorted recording.  If a recording fails (e.g. MATLAB error) it is
silently skipped so the returned list may be shorter than *recording_files*.


Sorting multiple recordings
---------------------------

.. code-block:: python

   from pathlib import Path
   from spikelab.spike_sorting import sort_with_kilosort2

   recordings = [
       "/data/recordings/session1.raw.h5",
       "/data/recordings/session2.raw.h5",
   ]

   results = sort_with_kilosort2(
       recording_files=recordings,
       kilosort_path="/opt/Kilosort2",
       n_jobs=16,
       total_memory="32G",
   )

   for path, sd in zip(recordings, results):
       print(f"{Path(path).name}: {sd.N} units")


Maxwell multi-stream recordings
--------------------------------

Maxwell Biosystems ``.h5`` files can contain multiple recording streams
(e.g. ``"well000"``, ``"well001"``).  Pass ``stream_id`` to select the
desired stream:

.. code-block:: python

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/multiwell.raw.h5"],
       kilosort_path="/opt/Kilosort2",
       stream_id="well000",
   )


Controlling curation thresholds
---------------------------------

The pipeline applies two curation stages.  Stage 1 filters by firing rate,
ISI violations, SNR, and minimum spike count.  Stage 2 filters by waveform
consistency.  All thresholds can be overridden:

.. code-block:: python

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/session1.raw.h5"],
       kilosort_path="/opt/Kilosort2",
       # Stage 1
       fr_min=0.1,          # Hz — stricter than the default 0.05
       isi_viol_max=0.5,    # % — stricter than the default 1
       snr_min=6,
       spikes_min_first=50,
       # Stage 2
       spikes_min_second=100,
       std_norm_max=0.8,
   )

To disable a curation stage entirely:

.. code-block:: python

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/session1.raw.h5"],
       kilosort_path="/opt/Kilosort2",
       curate_first=False,
       curate_second=False,
   )


Kilosort2 parameters
--------------------

Pass a dict to ``kilosort_params`` to override any Kilosort2 configuration
value:

.. code-block:: python

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/session1.raw.h5"],
       kilosort_path="/opt/Kilosort2",
       kilosort_params={
           'detect_threshold': 5,
           'car': True,
           'keep_good_only': True,
       },
   )

See the :ref:`API reference <spikesorting-api>` for the full default parameter
dict.


Reusing intermediate results
-----------------------------

By default the pipeline rewrites the ``.dat`` binary on every run but skips
re-running Kilosort2 if ``spike_times.npy`` already exists in the output
folder.  Use the ``recompute_*`` flags to control which stages are re-run:

.. code-block:: python

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/session1.raw.h5"],
       kilosort_path="/opt/Kilosort2",
       recompute_recording=False,   # reuse existing .dat
       recompute_sorting=True,      # force re-run of Kilosort2
       reextract_waveforms=True,    # force waveform re-extraction
   )


Downstream analysis
-------------------

The returned :class:`~spikelab.spikedata.SpikeData` objects integrate
directly with all SpikeLab analysis methods:

.. code-block:: python

   from spikelab.spike_sorting import sort_with_kilosort2

   results = sort_with_kilosort2(
       recording_files=["/data/recordings/session1.raw.h5"],
       kilosort_path="/opt/Kilosort2",
   )
   sd = results[0]

   # Firing rates
   fr = sd.mean_firing_rate()

   # Burst detection
   bursts = sd.burst_detection()

   # Pairwise STTC correlation
   sttc = sd.spike_time_tiling_coefficient()

   # Save for later
   from spikelab.data_loaders.data_exporters import export_spikedata_to_hdf5
   export_spikedata_to_hdf5(sd, "session1_sorted.h5")
