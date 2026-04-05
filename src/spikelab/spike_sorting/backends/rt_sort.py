"""RT-Sort sorter backend.

Implements the ``SorterBackend`` interface using RT-Sort
(van der Molen, Lim et al. 2024, PLOS ONE, DOI
10.1371/journal.pone.0312438).  RT-Sort is a deep-learning-based
detection and propagation-sequence sorting algorithm vendored in
``spikelab.spike_sorting.rt_sort`` under the MIT license.

Unlike the Kilosort backends, RT-Sort does not run via
``spikeinterface.sorters.run_sorter``.  Instead it runs its own
two-stage pipeline (sequence detection + offline spike assignment)
and returns a ``NumpySorting`` object that plugs into the same
downstream waveform extraction + SpikeData conversion path used by
the rest of the pipeline.

Requirements:
    pip install torch diptest scikit-learn spikeinterface
    # Torch should be installed with a CUDA wheel matching your GPU —
    # see https://pytorch.org/get-started/locally/
"""

from typing import Any

from .. import _globals
from ..config import SortingPipelineConfig
from .base import SorterBackend


class RTSortBackend(SorterBackend):
    """SorterBackend implementation for RT-Sort.

    RT-Sort trains a set of propagation sequences from a recording
    (stage 1, ``detect_sequences``) and then assigns spikes in the
    recording to those sequences (stage 2, ``RTSort.sort_offline``).
    Both stages share the same recording and intermediate cache.

    For Phase 2 stim-aware sorting (forthcoming), the trained ``RTSort``
    object is persisted to the output folder as ``rt_sort.pickle`` and
    can be reloaded to sort a separate stimulation recording using the
    same sequences.

    Waveform extraction uses the shared ``WaveformExtractor`` because
    RT-Sort's output is a standard SpikeInterface ``NumpySorting``.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
            Reads ``config.recording``, ``config.rt_sort``, and
            ``config.waveform`` / ``config.execution`` for the
            downstream stages.
    """

    def __init__(self, config: SortingPipelineConfig) -> None:
        super().__init__(config)
        self._check_dependencies()
        self._sync_globals()

    def _check_dependencies(self) -> None:
        """Raise a clear ImportError listing any missing RT-Sort deps."""
        missing = []
        for name, pkg in [
            ("torch", "torch"),
            ("diptest", "diptest"),
            ("h5py", "h5py"),
            ("sklearn", "scikit-learn"),
            ("spikeinterface", "spikeinterface"),
            ("tqdm", "tqdm"),
        ]:
            try:
                __import__(name)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                "RT-Sort backend requires the following packages "
                f"which are not installed: {', '.join(missing)}. "
                "For PyTorch, install a CUDA-matching wheel from "
                "https://pytorch.org/get-started/locally/"
            )

    def _sync_globals(self) -> None:
        """Set module-level globals in _globals.py from the config.

        The shared recording loader and pipeline stages still read
        globals for parameters they own.  RT-Sort-specific parameters
        live under ``config.rt_sort``.
        """
        cfg = self.config
        rec = cfg.recording
        rts = cfg.rt_sort
        wf = cfg.waveform
        cur = cfg.curation
        comp = cfg.compilation
        exe = cfg.execution

        # Recording
        _globals.STREAM_ID = rec.stream_id
        _globals.FIRST_N_MINS = rec.first_n_mins
        _globals.MEA_Y_MAX = rec.mea_y_max
        _globals.GAIN_TO_UV = rec.gain_to_uv
        _globals.OFFSET_TO_UV = rec.offset_to_uv
        _globals.REC_CHUNKS = list(rec.rec_chunks)
        _globals.REC_CHUNKS_S = list(rec.rec_chunks_s)
        _globals.START_TIME_S = rec.start_time_s
        _globals.END_TIME_S = rec.end_time_s
        _globals._REC_CHUNK_NAMES = []
        _globals.FREQ_MIN = rec.freq_min
        _globals.FREQ_MAX = rec.freq_max

        # RT-Sort
        _globals.RT_SORT_MODEL_PATH = rts.model_path
        _globals.RT_SORT_DEVICE = rts.device
        _globals.RT_SORT_NUM_PROCESSES = rts.num_processes
        _globals.RT_SORT_RECORDING_WINDOW_MS = rts.recording_window_ms
        _globals.RT_SORT_SAVE_PICKLE = rts.save_rt_sort_pickle
        _globals.RT_SORT_DELETE_INTER = rts.delete_inter
        _globals.RT_SORT_VERBOSE = rts.verbose
        # Merge the probe into params so the runner can read both from
        # a single dict-shaped global.
        merged_params = {"probe": rts.probe}
        if rts.params:
            merged_params.update(rts.params)
        _globals.RT_SORT_PARAMS = merged_params

        # Waveforms
        _globals.WAVEFORMS_MS_BEFORE = wf.ms_before
        _globals.WAVEFORMS_MS_AFTER = wf.ms_after
        _globals.POS_PEAK_THRESH = wf.pos_peak_thresh
        _globals.MAX_WAVEFORMS_PER_UNIT = wf.max_waveforms_per_unit
        _globals.COMPILED_WAVEFORMS_MS_BEFORE = wf.compiled_ms_before
        _globals.COMPILED_WAVEFORMS_MS_AFTER = wf.compiled_ms_after
        _globals.SCALE_COMPILED_WAVEFORMS = wf.scale_compiled_waveforms
        _globals.STD_AT_PEAK = wf.std_at_peak
        _globals.STD_OVER_WINDOW_MS_BEFORE = wf.std_over_window_ms_before
        _globals.STD_OVER_WINDOW_MS_AFTER = wf.std_over_window_ms_after

        # Curation
        _globals.CURATE_FIRST = cur.curate_first
        _globals.CURATE_SECOND = cur.curate_second
        _globals.CURATION_EPOCH = cur.curation_epoch
        _globals.FR_MIN = cur.fr_min
        _globals.ISI_VIOL_MAX = cur.isi_viol_max
        _globals.ISI_VIOLATION_METHOD = cur.isi_violation_method
        _globals.SNR_MIN = cur.snr_min
        _globals.SPIKES_MIN_FIRST = cur.spikes_min_first
        _globals.SPIKES_MIN_SECOND = cur.spikes_min_second
        _globals.STD_NORM_MAX = cur.std_norm_max

        # Compilation
        _globals.COMPILE_SINGLE_RECORDING = comp.compile_single_recording
        _globals.COMPILE_TO_MAT = comp.compile_to_mat
        _globals.COMPILE_TO_NPZ = comp.compile_to_npz
        _globals.COMPILE_WAVEFORMS = comp.compile_waveforms
        _globals.SAVE_ELECTRODES = comp.save_electrodes
        _globals.SAVE_SPIKE_TIMES = comp.save_spike_times
        _globals.SAVE_RAW_PKL = comp.save_raw_pkl
        _globals.SAVE_DL_DATA = comp.save_dl_data

        # Execution
        _globals.N_JOBS = exe.n_jobs
        _globals.TOTAL_MEMORY = exe.total_memory
        _globals.USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION = (
            exe.use_parallel_processing_for_raw_conversion
        )
        _globals.SAVE_SCRIPT = exe.save_script
        _globals.OUT_FILE = exe.out_file
        _globals.RECOMPUTE_RECORDING = exe.recompute_recording
        _globals.RECOMPUTE_SORTING = exe.recompute_sorting
        _globals.REEXTRACT_WAVEFORMS = exe.reextract_waveforms
        _globals.RECURATE_FIRST = exe.recurate_first
        _globals.RECURATE_SECOND = exe.recurate_second
        _globals.RECOMPILE_SINGLE_RECORDING = exe.recompile_single_recording

    def load_recording(self, rec_path: Any) -> Any:
        """Load and preprocess a recording via the shared loader.

        Uses the same Maxwell/NWB loader as the Kilosort backends.
        """
        from ..recording_io import load_recording as _load_recording

        recording = _load_recording(rec_path)

        self.rec_chunk_names = list(_globals._REC_CHUNK_NAMES or [])
        self.config.recording.rec_chunks = list(_globals.REC_CHUNKS or [])

        return recording

    def sort(
        self,
        recording: Any,
        rec_path: Any,
        recording_dat_path: Any,
        output_folder: Any,
    ) -> Any:
        """Run the RT-Sort offline pipeline via rt_sort_runner."""
        from ..rt_sort_runner import spike_sort

        return spike_sort(
            rec_cache=recording,
            rec_path=rec_path,
            recording_dat_path=recording_dat_path,
            output_folder=output_folder,
        )

    def extract_waveforms(
        self,
        recording: Any,
        sorting: Any,
        waveforms_folder: Any,
        curation_folder: Any,
        rec_path: Any = None,
    ) -> Any:
        """Extract waveforms via the shared custom WaveformExtractor.

        RT-Sort returns a standard SpikeInterface ``NumpySorting``, so
        the existing extraction pipeline used by the Kilosort backends
        works without modification.
        """
        from ..recording_io import extract_waveforms as _extract_waveforms

        return _extract_waveforms(
            recording_path=rec_path,
            recording=recording,
            sorting=sorting,
            root_folder=waveforms_folder,
            initial_folder=curation_folder,
            n_jobs=self.config.execution.n_jobs,
            total_memory=self.config.execution.total_memory,
            progress_bar=True,
        )
