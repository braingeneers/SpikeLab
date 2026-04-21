"""Kilosort2 sorter backend.

Implements the ``SorterBackend`` interface by delegating to functions
in ``ks2_runner`` and ``recording_io``. The underlying functions still
read module-level globals from ``_globals.py``, so this backend sets
those globals from the ``SortingPipelineConfig`` on construction.

This is a transitional design. In a future cleanup, the underlying
functions will be refactored to accept the config directly, and the
global-setting logic will be removed.
"""

from typing import Any

from .. import _globals
from ..config import SortingPipelineConfig
from .base import SorterBackend

DEFAULT_KILOSORT2_PARAMS = {
    "detect_threshold": 6,
    "projection_threshold": [10, 4],
    "preclust_threshold": 8,
    "car": True,
    "minFR": 0.1,
    "minfr_goodchannels": 0.1,
    "freq_min": 150,
    "sigmaMask": 30,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "NT": None,
    "keep_good_only": False,
}
"""Default Kilosort2 parameters."""


class Kilosort2Backend(SorterBackend):
    """SorterBackend implementation for Kilosort2.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
    """

    def __init__(self, config: SortingPipelineConfig) -> None:
        super().__init__(config)
        self._sync_globals()

    def _sync_globals(self) -> None:
        """Set module-level globals in _globals.py from the config.

        This bridges the config-based architecture with functions that
        still read globals. Will be removed once all functions accept
        config directly.
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
        # HDF5 plugin path is now set upstream in pipeline.sort_recording()
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

        # Sorter
        _globals.KILOSORT_PATH = sor.sorter_path
        _globals.KILOSORT_PARAMS = {
            **DEFAULT_KILOSORT2_PARAMS,
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
        """Load and preprocess a recording.

        Handles Maxwell ``.h5``, NWB, directories (concatenation),
        and pre-loaded BaseRecording objects.

        After loading, ``self.rec_chunk_names`` and
        ``self.config.recording.rec_chunks`` are updated if the
        recording was concatenated from multiple files.
        """
        from ..recording_io import load_recording as _load_recording

        recording = _load_recording(rec_path)

        # Capture concatenation state set by load_recording/concatenate_recordings
        self.rec_chunk_names = list(_globals._REC_CHUNK_NAMES or [])
        self.config.recording.rec_chunks = list(_globals.REC_CHUNKS or [])

        return recording

    def sort(
        self, recording: Any, rec_path: Any, recording_dat_path: Any, output_folder: Any
    ) -> Any:
        """Run Kilosort2 spike sorting.

        Delegates to ``ks2_runner.spike_sort`` which handles binary
        conversion, MATLAB/Docker execution, and result loading.
        """
        from ..ks2_runner import spike_sort

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

        Uses the legacy extraction pipeline with per-spike centering.
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
