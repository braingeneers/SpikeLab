"""Centralized global state for the spike sorting pipeline.

All module-level globals that were previously scattered across
kilosort2.py are declared here. Backends set these via
``_sync_globals()`` and the various legacy functions read them
at call time.

This is a transitional design. In a future cleanup, functions
will accept a config object directly and this module will be removed.
See ``iat/TO_IMPLEMENT.md`` for the staged migration plan and
``iat/_globals_audit.md`` for the per-global inventory.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
STREAM_ID: Optional[str] = None
FIRST_N_MINS: Optional[float] = None
MEA_Y_MAX: Optional[int] = None
GAIN_TO_UV: Optional[float] = None
OFFSET_TO_UV: Optional[float] = None
REC_CHUNKS: List = []
REC_CHUNKS_S: List = []
# True when ``REC_CHUNKS`` was auto-populated by
# ``concatenate_recordings`` from the per-file frame boundaries of a
# directory input (rather than supplied by the user). The loader
# treats explicit time slicing (``start_time_s`` / ``end_time_s`` /
# ``rec_chunks_s``) as an override of these auto-populated chunks and
# only raises the "cannot combine frame- and time-based" error when
# the user *did* set ``rec_chunks`` themselves.
REC_CHUNKS_FROM_CONCAT: bool = False
START_TIME_S: Optional[float] = None
END_TIME_S: Optional[float] = None
_REC_CHUNK_NAMES: List[str] = []
FREQ_MIN: int = 300
FREQ_MAX: int = 6000

# ---------------------------------------------------------------------------
# Sorter
# ---------------------------------------------------------------------------
KILOSORT_PATH: Optional[str] = None
KILOSORT_PARAMS: Optional[Dict[str, Any]] = None
USE_DOCKER: bool = False

# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------
WAVEFORMS_MS_BEFORE: float = 2.0
WAVEFORMS_MS_AFTER: float = 2.0
POS_PEAK_THRESH: float = 2.0
MAX_WAVEFORMS_PER_UNIT: int = 300
STREAMING_WAVEFORMS: bool = True
SAVE_WAVEFORM_FILES: bool = True

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
N_JOBS: int = 8
TOTAL_MEMORY: str = "16G"
USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION: bool = True
RECOMPUTE_SORTING: bool = False
REEXTRACT_WAVEFORMS: bool = False

# ---------------------------------------------------------------------------
# RT-Sort (used by RTSortBackend and rt_sort_runner)
# ---------------------------------------------------------------------------
RT_SORT_MODEL_PATH: Optional[str] = None
RT_SORT_DEVICE: str = "cuda"
RT_SORT_NUM_PROCESSES: Optional[int] = None
RT_SORT_RECORDING_WINDOW_MS: Optional[Any] = None
RT_SORT_PARAMS: Optional[Dict[str, Any]] = None
RT_SORT_SAVE_PICKLE: bool = True
RT_SORT_DELETE_INTER: bool = False
RT_SORT_VERBOSE: bool = True
RT_SORT_DETECTION_WINDOW_S: Optional[float] = None
