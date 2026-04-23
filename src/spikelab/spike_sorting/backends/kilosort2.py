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
from ._common import _sync_globals_from_config
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
        sor = self.config.sorter
        _sync_globals_from_config(
            self.config,
            sorter_globals={
                "KILOSORT_PATH": sor.sorter_path,
                "KILOSORT_PARAMS": {
                    **DEFAULT_KILOSORT2_PARAMS,
                    **(sor.sorter_params or {}),
                },
                "USE_DOCKER": sor.use_docker,
            },
        )

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
        rng: Any = None,
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
            rng=rng,
        )
