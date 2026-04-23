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
from ._common import _sync_globals_from_config
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
        sor = self.config.sorter
        _sync_globals_from_config(
            self.config,
            sorter_globals={
                "KILOSORT_PATH": sor.sorter_path,
                "KILOSORT_PARAMS": {
                    **DEFAULT_KILOSORT4_PARAMS,
                    **(sor.sorter_params or {}),
                },
                "USE_DOCKER": sor.use_docker,
            },
        )

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
        rng: Any = None,
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
            rng=rng,
        )
