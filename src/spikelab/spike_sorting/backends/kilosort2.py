"""Kilosort2 sorter backend.

Implements the ``SorterBackend`` interface by delegating to the existing
functions in ``spikelab.spike_sorting.kilosort2``.  The legacy functions
still read module-level globals, so this backend sets those globals from
the ``SortingPipelineConfig`` on construction.

This is a transitional design.  In a future cleanup, the underlying
functions will be refactored to accept the config directly, and the
global-setting logic will be removed.
"""

import os

from ..config import SortingPipelineConfig
from .base import SorterBackend


class Kilosort2Backend(SorterBackend):
    """SorterBackend implementation for Kilosort2.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
    """

    def __init__(self, config: SortingPipelineConfig):
        super().__init__(config)
        self._sync_globals()

    def _sync_globals(self):
        """Set module-level globals in kilosort2.py from the config.

        This bridges the config-based architecture with the legacy
        global-reading functions.  Will be removed once all functions
        accept config directly.
        """
        from .. import kilosort2 as ks2

        cfg = self.config
        rec = cfg.recording
        sor = cfg.sorter
        wf = cfg.waveform
        cur = cfg.curation
        comp = cfg.compilation
        exe = cfg.execution

        # Recording
        ks2.STREAM_ID = rec.stream_id
        if rec.hdf5_plugin_path is not None:
            os.environ["HDF5_PLUGIN_PATH"] = str(rec.hdf5_plugin_path)
        ks2.FIRST_N_MINS = rec.first_n_mins
        ks2.MEA_Y_MAX = rec.mea_y_max
        ks2.GAIN_TO_UV = rec.gain_to_uv
        ks2.OFFSET_TO_UV = rec.offset_to_uv
        ks2.REC_CHUNKS = list(rec.rec_chunks)
        ks2._REC_CHUNK_NAMES = []
        ks2.FREQ_MIN = rec.freq_min
        ks2.FREQ_MAX = rec.freq_max

        # Sorter
        ks2.KILOSORT_PATH = sor.sorter_path
        _default_ks2_params = {
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
        ks2.KILOSORT_PARAMS = {**_default_ks2_params, **(sor.sorter_params or {})}
        ks2.USE_DOCKER = sor.use_docker

        # Waveforms
        ks2.WAVEFORMS_MS_BEFORE = wf.ms_before
        ks2.WAVEFORMS_MS_AFTER = wf.ms_after
        ks2.POS_PEAK_THRESH = wf.pos_peak_thresh
        ks2.MAX_WAVEFORMS_PER_UNIT = wf.max_waveforms_per_unit
        ks2.COMPILED_WAVEFORMS_MS_BEFORE = wf.compiled_ms_before
        ks2.COMPILED_WAVEFORMS_MS_AFTER = wf.compiled_ms_after
        ks2.SCALE_COMPILED_WAVEFORMS = wf.scale_compiled_waveforms
        ks2.STD_AT_PEAK = wf.std_at_peak
        ks2.STD_OVER_WINDOW_MS_BEFORE = wf.std_over_window_ms_before
        ks2.STD_OVER_WINDOW_MS_AFTER = wf.std_over_window_ms_after

        # Curation
        ks2.CURATE_FIRST = cur.curate_first
        ks2.CURATE_SECOND = cur.curate_second
        ks2.CURATION_EPOCH = cur.curation_epoch
        ks2.FR_MIN = cur.fr_min
        ks2.ISI_VIOL_MAX = cur.isi_viol_max
        ks2.ISI_VIOLATION_METHOD = cur.isi_violation_method
        ks2.SNR_MIN = cur.snr_min
        ks2.SPIKES_MIN_FIRST = cur.spikes_min_first
        ks2.SPIKES_MIN_SECOND = cur.spikes_min_second
        ks2.STD_NORM_MAX = cur.std_norm_max

        # Compilation
        ks2.COMPILE_SINGLE_RECORDING = comp.compile_single_recording
        ks2.COMPILE_TO_MAT = comp.compile_to_mat
        ks2.COMPILE_TO_NPZ = comp.compile_to_npz
        ks2.COMPILE_WAVEFORMS = comp.compile_waveforms
        ks2.COMPILE_ALL_RECORDINGS = comp.compile_all_recordings
        ks2.SAVE_ELECTRODES = comp.save_electrodes
        ks2.SAVE_SPIKE_TIMES = comp.save_spike_times
        ks2.SAVE_RAW_PKL = comp.save_raw_pkl
        ks2.SAVE_DL_DATA = comp.save_dl_data

        # Execution
        ks2.N_JOBS = exe.n_jobs
        ks2.TOTAL_MEMORY = exe.total_memory
        ks2.USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION = (
            exe.use_parallel_processing_for_raw_conversion
        )
        ks2.SAVE_SCRIPT = exe.save_script
        ks2.OUT_FILE = exe.out_file
        ks2.RECOMPUTE_RECORDING = exe.recompute_recording
        ks2.RECOMPUTE_SORTING = exe.recompute_sorting
        ks2.REEXTRACT_WAVEFORMS = exe.reextract_waveforms
        ks2.RECURATE_FIRST = exe.recurate_first
        ks2.RECURATE_SECOND = exe.recurate_second
        ks2.RECOMPILE_SINGLE_RECORDING = exe.recompile_single_recording
        ks2.RECOMPILE_ALL_RECORDINGS = exe.recompile_all_recordings

    def load_recording(self, rec_path):
        """Load and preprocess a recording via the legacy loader.

        Handles Maxwell ``.h5``, NWB, directories (concatenation),
        and pre-loaded BaseRecording objects.

        After loading, ``self.rec_chunk_names`` and
        ``self.config.recording.rec_chunks`` are updated if the
        recording was concatenated from multiple files.
        """
        from .. import kilosort2 as ks2

        recording = ks2.load_recording(rec_path)

        # Capture concatenation state set by load_recording/concatenate_recordings
        self.rec_chunk_names = getattr(ks2, "_REC_CHUNK_NAMES", []) or []
        self.config.recording.rec_chunks = list(getattr(ks2, "REC_CHUNKS", []) or [])

        return recording

    def sort(self, recording, rec_path, recording_dat_path, output_folder):
        """Run Kilosort2 spike sorting.

        Delegates to the legacy ``spike_sort`` function which handles
        binary conversion, MATLAB/Docker execution, and result loading.
        """
        from .. import kilosort2 as ks2

        return ks2.spike_sort(
            rec_cache=recording,
            rec_path=rec_path,
            recording_dat_path=recording_dat_path,
            output_folder=output_folder,
        )

    def extract_waveforms(
        self, recording, sorting, waveforms_folder, curation_folder, rec_path=None
    ):
        """Extract waveforms via the custom WaveformExtractor.

        Uses the legacy extraction pipeline with per-spike centering.
        """
        from .. import kilosort2 as ks2

        return ks2.extract_waveforms(
            recording_path=rec_path,
            recording=recording,
            sorting=sorting,
            root_folder=waveforms_folder,
            initial_folder=curation_folder,
            n_jobs=self.config.execution.n_jobs,
            total_memory=self.config.execution.total_memory,
            progress_bar=True,
        )
