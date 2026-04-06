"""Centralized global state for the spike sorting pipeline.

All module-level globals that were previously scattered across
kilosort2.py are declared here. Backends set these via
``_sync_globals()`` and the various legacy functions read them
at call time.

This is a transitional design. In a future cleanup, functions
will accept a config object directly and this module will be removed.
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
COMPILED_WAVEFORMS_MS_BEFORE: float = 2.0
COMPILED_WAVEFORMS_MS_AFTER: float = 2.0
SCALE_COMPILED_WAVEFORMS: bool = True
STD_AT_PEAK: bool = True
STD_OVER_WINDOW_MS_BEFORE: float = 0.5
STD_OVER_WINDOW_MS_AFTER: float = 1.5

# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------
CURATE_FIRST: bool = True
CURATE_SECOND: bool = True
CURATION_EPOCH: Optional[int] = None
FR_MIN: float = 0.05
ISI_VIOL_MAX: float = 1.0
ISI_VIOLATION_METHOD: str = "percent"
SNR_MIN: float = 5.0
SPIKES_MIN_FIRST: int = 30
SPIKES_MIN_SECOND: int = 50
STD_NORM_MAX: float = 1.0

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
COMPILE_SINGLE_RECORDING: bool = True
COMPILE_TO_MAT: bool = False
COMPILE_TO_NPZ: bool = True
COMPILE_WAVEFORMS: bool = False
SAVE_ELECTRODES: bool = True
SAVE_SPIKE_TIMES: bool = True
SAVE_RAW_PKL: bool = False
SAVE_DL_DATA: bool = False

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
N_JOBS: int = 8
TOTAL_MEMORY: str = "16G"
USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION: bool = True
SAVE_SCRIPT: bool = False
OUT_FILE: str = "sort_with_kilosort2.out"
RECOMPUTE_RECORDING: bool = False
RECOMPUTE_SORTING: bool = False
REEXTRACT_WAVEFORMS: bool = False
RECURATE_FIRST: bool = False
RECURATE_SECOND: bool = False
RECOMPILE_SINGLE_RECORDING: bool = False

# ---------------------------------------------------------------------------
# Paths (set per-run by sort_with_kilosort2 / sort_recording)
# ---------------------------------------------------------------------------
RECORDING_FILES: List = []
INTERMEDIATE_FOLDERS: List = []
RESULTS_FOLDERS: List = []
COMPILED_RESULTS_FOLDER: Optional[str] = None

# ---------------------------------------------------------------------------
# Figure settings (used by Compiler)
# ---------------------------------------------------------------------------
CREATE_FIGURES: bool = False
FIGURES_DPI: Optional[int] = None
FIGURES_FONT_SIZE: int = 12
BAR_X_LABEL: str = "Recording"
BAR_Y_LABEL: str = "Number of Units"
BAR_LABEL_ROTATION: int = 0
BAR_TOTAL_LABEL: str = "First Curation"
BAR_SELECTED_LABEL: str = "Selected Curation"
SCATTER_STD_MAX_UNITS_PER_RECORDING: Optional[int] = None
SCATTER_RECORDING_COLORS: Optional[List] = None
SCATTER_RECORDING_ALPHA: float = 1.0
SCATTER_X_LABEL: str = "Number of Spikes"
SCATTER_Y_LABEL: str = "avg. STD / amplitude"
SCATTER_X_MAX_BUFFER: int = 300
SCATTER_Y_MAX_BUFFER: float = 0.2
ALL_TEMPLATES_COLOR_CURATED: str = "#000000"
ALL_TEMPLATES_COLOR_FAILED: str = "#FF0000"
ALL_TEMPLATES_PER_COLUMN: int = 50
ALL_TEMPLATES_Y_SPACING: int = 50
ALL_TEMPLATES_Y_LIM_BUFFER: int = 10
ALL_TEMPLATES_WINDOW_MS_BEFORE_PEAK: float = 5.0
ALL_TEMPLATES_WINDOW_MS_AFTER_PEAK: float = 5.0
ALL_TEMPLATES_LINE_MS_BEFORE_PEAK: float = 1.0
ALL_TEMPLATES_LINE_MS_AFTER_PEAK: float = 4.0
ALL_TEMPLATES_X_LABEL: str = "Time Rel. to Peak (ms)"
