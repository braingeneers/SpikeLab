============
Spike Sorting
============

The ``spikelab.spike_sorting`` sub-package wraps the Kilosort2 MATLAB spike
sorter, handling the full pipeline from raw recording files to a
:class:`~spikelab.spikedata.SpikeData` object: loading, bandpass filtering,
writing the ``.dat`` file for Kilosort2, running MATLAB, extracting waveforms,
and a two-stage quality curation.

.. note::

   ``spikelab.spike_sorting`` depends on a working **MATLAB** installation,
   the `Kilosort2 <https://github.com/MouseLand/Kilosort2>`_ MATLAB source
   tree, and `SpikeInterface <https://spikeinterface.readthedocs.io/>`_.
   These are not installed automatically with SpikeLab.  See the
   :doc:`/guides/spike_sorting` guide for environment setup instructions.

.. _spikesorting-api:

sort_with_kilosort2
-------------------

.. code-block:: python

   from spikelab.spike_sorting import sort_with_kilosort2

**Signature**

.. code-block:: text

   sort_with_kilosort2(
       recording_files,
       intermediate_folders=None,
       results_folders=None,
       compiled_results_folder=None,
       out_file="sort_with_kilosort2.out",
       kilosort_path=None,
       stream_id=None,
       kilosort_params={...},
       recompute_recording=True,
       recompute_sorting=False,
       reextract_waveforms=False,
       recurate_first=False,
       recurate_second=False,
       recompile_single_recording=False,
       recompile_all_recordings=False,
       delete_inter=True,
       save_script=False,
       n_jobs=8,
       total_memory="16G",
       use_parallel_processing_for_raw_conversion=True,
       first_n_mins=None,
       mea_y_max=None,
       gain_to_uv=None,
       offset_to_uv=None,
       rec_chunks=[],
       freq_min=300,
       freq_max=6000,
       waveforms_ms_before=2,
       waveforms_ms_after=2,
       pos_peak_thresh=2,
       max_waveforms_per_unit=300,
       curate_first=True,
       curate_second=True,
       fr_min=0.05,
       isi_viol_max=1,
       snr_min=5,
       spikes_min_first=30,
       spikes_min_second=50,
       std_norm_max=1,
       std_at_peak=True,
       std_over_window_ms_before=0.5,
       std_over_window_ms_after=1.5,
       save_electrodes=True,
       save_spike_times=True,
       save_dl_data=True,
       compile_single_recording=True,
       compile_to_mat=False,
       compile_to_npz=True,
       compile_waveforms=False,
       compile_all_recordings=False,
       compiled_waveforms_ms_before=2,
       compiled_waveforms_ms_after=2,
       scale_compiled_waveforms=True,
       create_figures=False,
       figures_dpi=None,
       figures_font_size=12,
       ...figure/plot kwargs...
   ) -> list[SpikeData]

**Description**

Run the full Kilosort2 spike-sorting pipeline on one or more recordings and
return the curated results as :class:`~spikelab.spikedata.SpikeData` objects.

For each recording the function:

1. Loads the recording via SpikeInterface (Maxwell ``.h5``, NWB ``.nwb``, or
   any pre-loaded ``BaseRecording`` object).
2. Scales traces to µV and applies a bandpass filter.
3. Writes a binary ``.dat`` file required by Kilosort2.
4. Invokes Kilosort2 via a MATLAB subprocess.
5. Extracts per-unit waveforms.
6. Applies a two-stage quality curation (firing rate, ISI violations, SNR,
   spike count, waveform consistency).
7. Optionally saves ``spike_times.npy`` / ``spike_clusters.npy`` and
   compiled ``.npz`` / ``.mat`` result files.

**Parameters (key subset)**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``recording_files``
     - *(required)*
     - List of file paths (``str`` or ``Path``) to raw recordings, **or**
       pre-loaded SpikeInterface ``BaseRecording`` objects.
   * - ``intermediate_folders``
     - ``None``
     - List of folders for intermediate files (one per recording). Auto-generated
       next to the recording if ``None``.
   * - ``results_folders``
     - ``None``
     - Output folders for per-recording compiled results. Defaults to
       ``<recording_dir>/sorted_kilosort2``.
   * - ``kilosort_path``
     - ``None``
     - Path to the Kilosort2 MATLAB source tree. Falls back to the
       ``KILOSORT_PATH`` environment variable.
   * - ``stream_id``
     - ``None``
     - Stream identifier forwarded to ``MaxwellRecordingExtractor`` when
       loading ``.h5`` files that contain multiple recording streams.
   * - ``kilosort_params``
     - *see below*
     - Dict of Kilosort2 configuration values (detect threshold, projection
       threshold, CAR, etc.).
   * - ``delete_inter``
     - ``True``
     - Delete intermediate files (binary ``.dat``, raw waveforms) after sorting.
   * - ``n_jobs``
     - ``8``
     - Number of parallel CPU threads.
   * - ``total_memory``
     - ``"16G"``
     - Memory budget for parallel processing.
   * - ``first_n_mins``
     - ``None``
     - Truncate each recording to the first *N* minutes before sorting.
   * - ``freq_min`` / ``freq_max``
     - ``300`` / ``6000``
     - Bandpass filter cutoffs in Hz.
   * - ``curate_first`` / ``curate_second``
     - ``True`` / ``True``
     - Enable the two curation stages.
   * - ``fr_min``
     - ``0.05``
     - Minimum firing rate (Hz) for first-stage curation.
   * - ``isi_viol_max``
     - ``1``
     - Maximum ISI violation percentage (%) for first-stage curation.
   * - ``snr_min``
     - ``5``
     - Minimum SNR for first-stage curation.

**Default kilosort_params**

.. code-block:: python

   {
       'detect_threshold': 6,
       'projection_threshold': [10, 4],
       'preclust_threshold': 8,
       'car': True,
       'minFR': 0.1,
       'minfr_goodchannels': 0.1,
       'freq_min': 150,
       'sigmaMask': 30,
       'nPCs': 3,
       'ntbuff': 64,
       'nfilt_factor': 4,
       'NT': None,
       'keep_good_only': False,
   }

**Returns**

``list[SpikeData]`` — one entry per successfully sorted recording, in the same
order as *recording_files*.  Recordings that fail (file not found, MATLAB
error, etc.) are skipped and do not appear in the list.

Each :class:`~spikelab.spikedata.SpikeData` object has:

- Spike trains in **milliseconds**.
- ``neuron_attributes``: list of ``{"unit_id": int}`` dicts, one per curated unit.
- ``metadata["source_file"]``: path to the original recording.
- ``metadata["source_format"]``: ``"Kilosort2"``.
- ``metadata["fs_Hz"]``: sampling frequency used during sorting.
