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

    Handles all common sections (recording, waveform, execution).
    The caller supplies ``sorter_globals`` — a dict of
    ``{global_name: value}`` pairs — for the sorter-specific section,
    which varies per backend.

    Only writes globals that production source code actually reads.
    Curation, compilation, figures, and dead-write execution fields
    are read directly from ``config`` by the modern call sites; their
    ``_globals`` entries were removed in the Phase 2.0 cleanup
    (see ``iat/_globals_audit.md``).

    Parameters:
        config: Full pipeline configuration.
        sorter_globals: Mapping of ``_globals`` attribute names to
            values for the sorter-specific section (e.g.
            ``{"KILOSORT_PATH": ..., "KILOSORT_PARAMS": ..., ...}``).
    """
    rec = config.recording
    wf = config.waveform
    exe = config.execution

    # Recording
    _globals.STREAM_ID = rec.stream_id
    _globals.FIRST_N_MINS = rec.first_n_mins
    _globals.MEA_Y_MAX = rec.mea_y_max
    _globals.GAIN_TO_UV = rec.gain_to_uv
    _globals.OFFSET_TO_UV = rec.offset_to_uv
    _globals.REC_CHUNKS = list(rec.rec_chunks)
    # Anything the user supplied here is by definition not
    # auto-populated; reset the flag so a stale True from a previous
    # sort or canary run does not silently allow a real
    # frame/time-slice combination through the loader's guard.
    _globals.REC_CHUNKS_FROM_CONCAT = False
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
    _globals.STREAMING_WAVEFORMS = wf.streaming
    _globals.SAVE_WAVEFORM_FILES = wf.save_waveform_files

    # Execution
    _globals.N_JOBS = exe.n_jobs
    _globals.TOTAL_MEMORY = exe.total_memory
    _globals.USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION = (
        exe.use_parallel_processing_for_raw_conversion
    )
    _globals.RECOMPUTE_SORTING = exe.recompute_sorting
    _globals.REEXTRACT_WAVEFORMS = exe.reextract_waveforms
