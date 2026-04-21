"""Kilosort4 sorter backend.

Implements the ``SorterBackend`` interface using Kilosort4 (pure Python,
PyTorch-based) via SpikeInterface's ``run_sorter("kilosort4", ...)``.
Uses the same custom ``WaveformExtractor`` and per-spike centering as
the Kilosort2 backend.

Requirements:
    pip install kilosort
    # Plus PyTorch with CUDA — see https://pytorch.org/get-started/locally/
"""

from typing import Any

from .. import _globals
from ..config import SortingPipelineConfig
from .base import SorterBackend

DEFAULT_KILOSORT4_PARAMS = {
    "do_CAR": True,
    "invert_sign": False,
    "save_extra_vars": False,
    "save_preprocessed_copy": False,
    "torch_device": "auto",
    "bad_channels": None,
    "clear_cache": False,
    "do_correction": True,
    "skip_kilosort_preprocessing": False,
    "keep_good_only": False,
    "use_binary_file": True,
    "delete_recording_dat": True,
}
"""Default Kilosort4 parameters.  Originally tuned for Neuropixels probes.
Used by the backend and ``KILOSORT4_NEUROPIXELS`` preset config."""


class Kilosort4Backend(SorterBackend):
    """SorterBackend implementation for Kilosort4.

    Kilosort4 is a pure Python spike sorter (no MATLAB required).  It
    runs via SpikeInterface's ``run_sorter("kilosort4", ...)`` interface,
    which handles binary conversion, parameter passing, and result
    loading.

    Waveform extraction uses the same custom ``WaveformExtractor``
    with per-spike centering, since the output format
    (``spike_times.npy``, ``spike_clusters.npy``, ``templates.npy``) is
    compatible.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
    """

    def __init__(self, config: SortingPipelineConfig) -> None:
        super().__init__(config)
        self._sync_globals()

    def _sync_globals(self) -> None:
        """Set module-level globals in _globals.py from the config.

        The shared WaveformExtractor and recording loader still read
        globals. This bridges the config-based architecture with those
        functions.
        """
        cfg = self.config
        rec = cfg.recording
        sor = cfg.sorter
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

        # Sorter — KS4 params are passed directly to SI, but we store
        # them in the global for the KilosortSortingExtractor to read
        _globals.KILOSORT_PATH = sor.sorter_path
        _globals.KILOSORT_PARAMS = {
            **DEFAULT_KILOSORT4_PARAMS,
            **(sor.sorter_params or {}),
        }
        _globals.USE_DOCKER = sor.use_docker

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
        _globals.STREAMING_WAVEFORMS = wf.streaming
        _globals.SAVE_WAVEFORM_FILES = wf.save_waveform_files

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

        Uses the same Maxwell/NWB loader as the Kilosort2 backend.
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
        """Run Kilosort4 spike sorting via ks4_runner."""
        from ..ks4_runner import spike_sort

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
        """Extract waveforms via the custom WaveformExtractor.

        Uses the same extraction pipeline and per-spike centering as
        the Kilosort2 backend.
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
