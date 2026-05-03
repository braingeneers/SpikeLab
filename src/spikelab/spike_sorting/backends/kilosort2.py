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
        Computes the sorter inactivity tolerance from the recording
        duration; the local-MATLAB path consumes it via the
        ``inactivity_timeout_s`` argument, and the Docker path picks
        it up via the ContextVar published by
        :func:`set_active_inactivity_timeout_s` so
        ``patched_container_client`` can install a container-aware
        :class:`LogInactivityWatchdog`.
        """
        from ..guards import set_active_inactivity_timeout_s
        from ..ks2_runner import spike_sort

        inactivity_timeout_s = self._resolve_inactivity_timeout_s(recording)

        with set_active_inactivity_timeout_s(inactivity_timeout_s):
            return spike_sort(
                rec_cache=recording,
                rec_path=rec_path,
                recording_dat_path=recording_dat_path,
                output_folder=output_folder,
                inactivity_timeout_s=inactivity_timeout_s,
            )

    def scale_oom_params(self, factor: float) -> bool:
        """Halve (or scale) Kilosort2's ``NT`` to reduce GPU memory.

        ``NT`` is the per-batch sample count consumed by the KS2
        template-matching CUDA kernel; halving it roughly halves
        per-batch VRAM. The new value is rounded to a multiple of 32
        (KS2 kernel constraint, also enforced by ``format_params``).

        Parameters:
            factor (float): Multiplicative factor in ``(0, 1]``.

        Returns:
            scaled (bool): True when ``NT`` was reduced. False when
                the existing ``NT`` is too small to halve safely
                (below 1024) — at which point further reduction is
                unlikely to help.
        """
        if factor <= 0.0 or factor >= 1.0:
            return False

        params = dict(self.config.sorter.sorter_params or {})
        # ``format_params`` resolves NT=None to (64*1024 + ntbuff)
        # before the first sort. After that the value is concrete in
        # _globals.KILOSORT_PARAMS, but we want to mutate the
        # config's representation so subsequent sorts persist the
        # scaled value rather than reverting to the default.
        nt = params.get("NT")
        if nt is None:
            ntbuff = params.get("ntbuff", 64)
            nt = 64 * 1024 + int(ntbuff)
        new_nt = int(int(nt) * float(factor))
        # KS2 mex requires NT to be a multiple of 32; round down.
        new_nt = (new_nt // 32) * 32
        if new_nt < 1024:
            print(
                f"[oom retry] kilosort2: NT would drop to {new_nt} "
                "after scaling — refusing to scale further."
            )
            return False
        params["NT"] = new_nt
        self.config.sorter.sorter_params = params
        # Re-sync globals so the runner sees the scaled NT.
        self._sync_globals()
        print(
            f"[oom retry] kilosort2: scaled NT {nt} -> {new_nt} " f"(factor={factor})."
        )
        return True

    def snapshot_oom_params(self) -> dict:
        """Snapshot ``sorter_params`` (which carries ``NT``)."""
        params = self.config.sorter.sorter_params
        return {
            "sorter_params": dict(params) if params is not None else None,
        }

    def restore_oom_params(self, snapshot: dict) -> None:
        """Restore ``sorter_params`` from a snapshot and re-sync globals."""
        if not snapshot:
            return
        self.config.sorter.sorter_params = snapshot.get("sorter_params")
        self._sync_globals()

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
