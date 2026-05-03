"""Abstract base class for spike sorter backends.

Each backend implements the three-step pipeline: load recording, run
sorter, extract waveforms.  The pipeline module (``pipeline.py``)
calls these methods and handles everything downstream (SpikeData
conversion, curation, compilation, figures).

To add a new sorter:

1. Create a new module in ``backends/`` (e.g. ``kilosort4.py``).
2. Subclass ``SorterBackend`` and implement all three methods.
3. Register the backend in ``backends/__init__.py``.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..config import SortingPipelineConfig


class SorterBackend(ABC):
    """Interface that each spike sorter backend must implement.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
            Backends read their relevant sub-configs (``config.recording``,
            ``config.sorter``, ``config.waveform``, ``config.execution``).
    """

    def __init__(self, config: SortingPipelineConfig) -> None:
        self.config = config

    @abstractmethod
    def load_recording(self, rec_path: Any):
        """Load and preprocess a single recording.

        Handles format-specific loading (Maxwell ``.h5``, NWB, etc.),
        gain/offset scaling, and bandpass filtering.

        Parameters:
            rec_path: Path to a recording file, a directory of files
                to concatenate, or a pre-loaded BaseRecording object.

        Returns:
            recording: A SpikeInterface ``BaseRecording`` ready for
                sorting (scaled, filtered, single-segment).
        """

    @abstractmethod
    def sort(self, recording, rec_path, recording_dat_path, output_folder):
        """Run the spike sorter on a preprocessed recording.

        Parameters:
            recording: SpikeInterface ``BaseRecording`` from
                ``load_recording``.
            rec_path: Original recording file path (for binary
                conversion or metadata).
            recording_dat_path (Path): Path for the binary ``.dat``
                file (used by sorters that require pre-converted input).
            output_folder (Path): Directory for sorter output files.

        Returns:
            sorting: A SpikeInterface ``BaseSorting`` with detected
                units and spike trains.
        """

    @abstractmethod
    def extract_waveforms(
        self,
        recording,
        sorting,
        waveforms_folder,
        curation_folder,
        rec_path=None,
        rng=None,
    ):
        """Extract per-unit waveforms and compute templates.

        Parameters:
            recording: SpikeInterface ``BaseRecording``.
            sorting: SpikeInterface ``BaseSorting`` from ``sort``.
            waveforms_folder (Path): Root directory for waveform
                storage.
            curation_folder (Path): Directory for initial unit list
                and metadata.

        Returns:
            waveform_extractor: An object providing at minimum:

                - ``sorting`` — the sorting object (possibly with
                  centered spike times)
                - ``recording`` — the recording object
                - ``sampling_frequency`` — float
                - ``peak_ind`` — int (peak sample index in template)
                - ``chans_max_all`` — dict or array mapping unit_id
                  to max-amplitude channel index
                - ``use_pos_peak`` — dict or array mapping unit_id
                  to bool (polarity)
                - ``get_computed_template(unit_id, mode)`` — returns
                  ``(n_samples, n_channels)`` template array
                - ``ms_to_samples(ms)`` — time conversion
                - ``root_folder`` — Path to waveform files

              This can be the custom ``WaveformExtractor`` (Kilosort2
              backend) or a wrapper around SpikeInterface's
              ``WaveformExtractor`` (future backends).
        """

    def write_recording(self, recording: Any, dat_path: Any) -> None:
        """Convert a recording to the binary format needed by the sorter.

        Not all sorters need this (some read recordings directly via
        SpikeInterface).  The default implementation is a no-op.

        Parameters:
            recording: SpikeInterface ``BaseRecording``.
            dat_path (Path): Output binary file path.
        """
        pass

    def scale_oom_params(self, factor: float) -> bool:
        """Mutate ``self.config`` to halve (or scale) the OOM-bound knob.

        Each backend overrides this to adjust the parameter most
        directly responsible for GPU memory consumption — typically
        the per-batch sample count. The default implementation does
        nothing and reports failure so callers know retry-on-OOM is
        not supported for that backend.

        Parameters:
            factor (float): Multiplicative factor in ``(0, 1]`` to
                apply. ``0.5`` halves the parameter.

        Returns:
            scaled (bool): True when at least one parameter was
                changed; False when no scaling was applied. Callers
                should skip the retry when False is returned.
        """
        return False

    def snapshot_oom_params(self) -> dict:
        """Return a snapshot of OOM-bound config fields for restore.

        Used by the per-recording OOM-retry loop so a scale-down
        applied for one recording does not silently persist into the
        next. The returned dict is opaque — only
        :meth:`restore_oom_params` is expected to read it.

        Returns:
            snapshot (dict): Backend-specific snapshot. Default
                implementation returns an empty dict.
        """
        return {}

    def restore_oom_params(self, snapshot: dict) -> None:
        """Restore the OOM-bound config fields from a prior snapshot.

        Default implementation is a no-op. Backends that override
        :meth:`scale_oom_params` should also override this so the
        retry loop can reset the config between recordings.

        Parameters:
            snapshot (dict): Object returned by
                :meth:`snapshot_oom_params`.
        """
        return None

    def _make_in_process_inactivity_watchdog(self, recording: Any, *, sorter: str):
        """Build a no-popen :class:`LogInactivityWatchdog` for an in-process sort.

        Looks up the active per-recording log path (published by
        ``sort_recording``) and computes the inactivity tolerance from
        the recording's wall-clock duration. Returns a watchdog
        whose kill path is :func:`make_in_process_kill_callback` —
        i.e. ``_thread.interrupt_main`` followed by ``os._exit`` if
        Python is unresponsive.

        Parameters:
            recording: SpikeInterface ``BaseRecording`` for the
                inactivity-tolerance scaling.
            sorter (str): Short identifier used in diagnostics and
                in the resulting :class:`SorterTimeoutError`.

        Returns:
            watchdog (LogInactivityWatchdog or None): A configured
                watchdog (no-op when the active log path is missing
                or the timeout is disabled) ready to be used as a
                context manager. Returns ``None`` when no log path
                is currently active — the caller should skip
                wrapping in that case.
        """
        from ..guards import (
            LogInactivityWatchdog,
            get_active_log_path,
            make_in_process_kill_callback,
        )

        log_path = get_active_log_path()
        if log_path is None:
            return None

        inactivity_s = self._resolve_inactivity_timeout_s(recording)
        grace_s = float(
            getattr(self.config.execution, "sorter_inactivity_in_process_grace_s", 10.0)
        )
        return LogInactivityWatchdog(
            log_path=log_path,
            popen=None,
            inactivity_s=inactivity_s,
            sorter=sorter,
            kill_callback=make_in_process_kill_callback(
                interrupt_grace_s=grace_s,
                sorter=sorter,
            ),
        )

    def _resolve_inactivity_timeout_s(self, recording: Any) -> Optional[float]:
        """Compute the recording-aware sorter inactivity tolerance.

        Reads the watchdog knobs from ``self.config.execution`` and
        scales the tolerance with the recording's wall-clock duration
        so a long sort that takes several minutes between log writes
        is not falsely killed by a watchdog tuned for short test
        recordings.

        Parameters:
            recording: SpikeInterface ``BaseRecording`` providing
                ``get_num_samples()`` and ``get_sampling_frequency()``.

        Returns:
            timeout_s (float or None): Resolved tolerance in seconds,
                or ``None`` when the watchdog is disabled in the
                configuration. ``None`` is also returned when the
                recording duration cannot be determined; the caller
                must treat ``None`` as "do not start the watchdog".
        """
        exe = self.config.execution
        if not getattr(exe, "sorter_inactivity_timeout", False):
            return None
        try:
            n_samples = float(recording.get_num_samples())
            fs_hz = float(recording.get_sampling_frequency())
        except Exception:
            return None
        if fs_hz <= 0.0:
            return None
        duration_min = n_samples / fs_hz / 60.0

        from ..guards import compute_inactivity_timeout_s

        return compute_inactivity_timeout_s(
            recording_duration_min=duration_min,
            base_s=exe.sorter_inactivity_base_s,
            per_min_s=exe.sorter_inactivity_per_min_s,
            max_s=exe.sorter_inactivity_max_s,
        )
