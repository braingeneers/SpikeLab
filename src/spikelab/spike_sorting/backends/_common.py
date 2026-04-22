"""Shared helpers for sorter backends.

Contains the common portion of ``_sync_globals()`` that is identical
across all backends.  Each backend calls ``_sync_globals_from_config``
with its own ``sorter_globals`` dict for the sorter-specific section.
"""

from typing import Dict

from .. import _globals
from ..config import SortingPipelineConfig


def _sync_globals_from_config(
    config: SortingPipelineConfig,
    sorter_globals: Dict[str, object],
) -> None:
    """Set module-level globals in ``_globals.py`` from a pipeline config.

    Handles all common sections (recording, waveform, curation,
    compilation, execution).  The caller supplies ``sorter_globals``
    — a dict of ``{global_name: value}`` pairs — for the
    sorter-specific section, which varies per backend.

    Parameters:
        config: Full pipeline configuration.
        sorter_globals: Mapping of ``_globals`` attribute names to
            values for the sorter-specific section (e.g.
            ``{"KILOSORT_PATH": ..., "KILOSORT_PARAMS": ..., ...}``).
    """
    rec = config.recording
    wf = config.waveform
    cur = config.curation
    comp = config.compilation
    exe = config.execution

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

    # Sorter-specific
    for attr, value in sorter_globals.items():
        setattr(_globals, attr, value)

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
