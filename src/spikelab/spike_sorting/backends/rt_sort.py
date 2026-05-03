"""RT-Sort sorter backend.

Implements the ``SorterBackend`` interface using RT-Sort
(van der Molen, Lim et al. 2024, PLOS ONE, DOI
10.1371/journal.pone.0312438).  RT-Sort is a deep-learning-based
detection and propagation-sequence sorting algorithm vendored in
``spikelab.spike_sorting.rt_sort`` under the MIT license.

Unlike the Kilosort backends, RT-Sort does not run via
``spikeinterface.sorters.run_sorter``.  Instead it runs its own
two-stage pipeline (sequence detection + offline spike assignment)
and returns a ``NumpySorting`` object that plugs into the same
downstream waveform extraction + SpikeData conversion path used by
the rest of the pipeline.

Requirements:
    pip install torch diptest scikit-learn spikeinterface
    # Torch should be installed with a CUDA wheel matching your GPU —
    # see https://pytorch.org/get-started/locally/
"""

from pathlib import Path
from typing import Any

import numpy as np

from .. import _globals
from ..config import SortingPipelineConfig
from ._common import _sync_globals_from_config
from .base import SorterBackend


def _numpy_sorting_to_ks_extractor(sorting, recording, output_folder, root_elecs=None):
    """Convert a SpikeInterface NumpySorting to a KilosortSortingExtractor.

    Writes the Kilosort-format files that ``KilosortSortingExtractor``
    expects (``spike_times.npy``, ``spike_clusters.npy``,
    ``templates.npy``, ``channel_map.npy``, ``params.py``) to
    *output_folder*, then returns a ``KilosortSortingExtractor`` that
    reads them.

    Parameters:
        sorting: SpikeInterface NumpySorting.
        recording: SpikeInterface BaseRecording.
        output_folder: Path for Kilosort-format output files.
        root_elecs (list or None): Per-unit root electrode indices from
            RTSort._seq_root_elecs.  Used to set the peak channel in
            synthetic templates so that get_chans_max() returns the
            correct channel for each unit.
    """
    from ..sorting_extractor import KilosortSortingExtractor

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fs = recording.get_sampling_frequency()
    n_channels = recording.get_num_channels()
    unit_ids = sorting.get_unit_ids()

    # Build spike_times and spike_clusters arrays
    all_times = []
    all_clusters = []
    for uid in unit_ids:
        train = sorting.get_unit_spike_train(uid)
        all_times.append(train)
        all_clusters.append(np.full(len(train), uid, dtype=np.int64))

    if all_times:
        spike_times = np.concatenate(all_times)
        spike_clusters = np.concatenate(all_clusters)
        order = np.argsort(spike_times)
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]
    else:
        spike_times = np.array([], dtype=np.int64)
        spike_clusters = np.array([], dtype=np.int64)

    np.save(str(output_folder / "spike_times.npy"), spike_times)
    np.save(str(output_folder / "spike_clusters.npy"), spike_clusters)

    # Channel map: identity mapping (all channels)
    channel_map = np.arange(n_channels, dtype=np.int32)
    np.save(str(output_folder / "channel_map.npy"), channel_map)

    # Synthetic templates: (n_units, n_samples, n_channels).
    # The WaveformExtractor uses get_chans_max() on these templates to
    # determine the peak channel per unit.  RT-Sort knows each unit's
    # root electrode (_seq_root_elecs), so we place a negative peak
    # marker on the correct channel for each unit.  The actual waveform
    # templates are recomputed from raw data during extraction.
    n_template_samples = 82  # KS2 default template length
    max_uid = max(unit_ids) + 1 if len(unit_ids) else 0
    templates = np.zeros(
        (max_uid, n_template_samples, n_channels),
        dtype=np.float32,
    )
    mid = n_template_samples // 2
    for i, uid in enumerate(unit_ids):
        chan = 0
        if root_elecs is not None and i < len(root_elecs):
            re = root_elecs[i]
            chan = re if re < n_channels else 0
        templates[uid, mid, chan] = -1.0

    np.save(str(output_folder / "templates.npy"), templates)

    # params.py — minimal, only sample_rate is read by
    # KilosortSortingExtractor
    with open(output_folder / "params.py", "w") as f:
        f.write(f"sample_rate = {fs}\n")
        f.write(f"n_channels_dat = {n_channels}\n")
        f.write(f"dtype = 'float32'\n")
        f.write(f"hp_filtered = True\n")

    return KilosortSortingExtractor(
        folder_path=output_folder,
        keep_good_only=bool(
            _globals.KILOSORT_PARAMS and _globals.KILOSORT_PARAMS.get("keep_good_only")
        ),
        pos_peak_thresh=_globals.POS_PEAK_THRESH,
    )


class RTSortBackend(SorterBackend):
    """SorterBackend implementation for RT-Sort.

    RT-Sort trains a set of propagation sequences from a recording
    (stage 1, ``detect_sequences``) and then assigns spikes in the
    recording to those sequences (stage 2, ``RTSort.sort_offline``).
    Both stages share the same recording and intermediate cache.

    For Phase 2 stim-aware sorting (forthcoming), the trained ``RTSort``
    object is persisted to the output folder as ``rt_sort.pickle`` and
    can be reloaded to sort a separate stimulation recording using the
    same sequences.

    Waveform extraction uses the shared ``WaveformExtractor`` because
    RT-Sort's output is a standard SpikeInterface ``NumpySorting``.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
            Reads ``config.recording``, ``config.rt_sort``, and
            ``config.waveform`` / ``config.execution`` for the
            downstream stages.
    """

    def __init__(self, config: SortingPipelineConfig) -> None:
        super().__init__(config)
        self._check_dependencies()
        self._sync_globals()

    def _check_dependencies(self) -> None:
        """Raise a clear ImportError listing any missing RT-Sort deps."""
        missing = []
        for name, pkg in [
            ("torch", "torch"),
            ("diptest", "diptest"),
            ("h5py", "h5py"),
            ("sklearn", "scikit-learn"),
            ("spikeinterface", "spikeinterface"),
            ("tqdm", "tqdm"),
        ]:
            try:
                __import__(name)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                "RT-Sort backend requires the following packages "
                f"which are not installed: {', '.join(missing)}. "
                "For PyTorch, install a CUDA-matching wheel from "
                "https://pytorch.org/get-started/locally/"
            )

    def _sync_globals(self) -> None:
        """Set module-level globals in _globals.py from the config.

        The shared recording loader and pipeline stages still read
        globals for parameters they own.  RT-Sort-specific parameters
        live under ``config.rt_sort``.
        """
        rts = self.config.rt_sort

        # Merge the probe into params so the runner can read both from
        # a single dict-shaped global.
        merged_params = {"probe": rts.probe}
        if rts.params:
            merged_params.update(rts.params)

        _sync_globals_from_config(
            self.config,
            sorter_globals={
                "RT_SORT_MODEL_PATH": rts.model_path,
                "RT_SORT_DEVICE": rts.device,
                "RT_SORT_NUM_PROCESSES": rts.num_processes,
                "RT_SORT_RECORDING_WINDOW_MS": rts.recording_window_ms,
                "RT_SORT_SAVE_PICKLE": rts.save_rt_sort_pickle,
                "RT_SORT_DELETE_INTER": rts.delete_inter,
                "RT_SORT_VERBOSE": rts.verbose,
                "RT_SORT_DETECTION_WINDOW_S": rts.detection_window_s,
                "RT_SORT_PARAMS": merged_params,
            },
        )

    def load_recording(self, rec_path: Any) -> Any:
        """Load and preprocess a recording via the shared loader.

        Uses the same Maxwell/NWB loader as the Kilosort backends.
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
        """Run the RT-Sort offline pipeline via rt_sort_runner.

        RT-Sort returns a SpikeInterface ``NumpySorting``.  The
        downstream ``WaveformExtractor`` expects a
        ``KilosortSortingExtractor``, so we convert the result by
        writing Kilosort-format files (``spike_times.npy``,
        ``spike_clusters.npy``, ``templates.npy``, ``channel_map.npy``,
        ``params.py``) to the output folder and returning a
        ``KilosortSortingExtractor`` that reads them.

        RT-Sort runs in-process (no subprocess to terminate). The
        in-process inactivity watchdog watches the per-recording Tee
        log and falls back to ``os._exit`` if Python cannot recover
        cleanly from ``_thread.interrupt_main``.
        """
        from ..rt_sort_runner import spike_sort

        watchdog = self._make_in_process_inactivity_watchdog(
            recording, sorter="rt_sort"
        )

        def _do_sort():
            return spike_sort(
                rec_cache=recording,
                rec_path=rec_path,
                recording_dat_path=recording_dat_path,
                output_folder=output_folder,
            )

        if watchdog is None:
            result = _do_sort()
        else:
            try:
                with watchdog:
                    result = _do_sort()
            except KeyboardInterrupt:
                if watchdog.tripped():
                    return watchdog.make_error()
                raise
            if watchdog.tripped():
                return watchdog.make_error()

        if isinstance(result, BaseException):
            return result

        sorting, root_elecs = result

        return _numpy_sorting_to_ks_extractor(
            sorting,
            recording,
            output_folder,
            root_elecs=root_elecs,
        )

    def scale_oom_params(self, factor: float) -> bool:
        """Halve (or scale) RT-Sort's ``num_processes`` to reduce memory.

        RT-Sort does not expose a single per-batch sample-count knob
        comparable to KS2 ``NT`` or KS4 ``batch_size``. The closest
        memory-pressure lever is ``num_processes``, which controls
        the number of parallel inference workers; halving it cuts
        peak combined RSS roughly in half.

        Parameters:
            factor (float): Multiplicative factor in ``(0, 1]``.

        Returns:
            scaled (bool): True when ``num_processes`` was reduced.
                False when it is already ``1`` (or below) and cannot
                be further halved.

        Notes:
            - For GPU OOM specifically, this is a best-effort fallback;
              RT-Sort's deep-learning detection model is small (~740 KB
              of weights) and processes channels in fixed-size chunks,
              so GPU OOM is rare in practice. Host-RAM pressure is the
              more common failure mode and is already covered by the
              host-memory watchdog.
        """
        if factor <= 0.0 or factor >= 1.0:
            return False

        rt = self.config.rt_sort
        current = rt.num_processes
        if current is None:
            try:
                import os

                current = max(1, round(os.cpu_count() * 2 / 3))
            except Exception:
                current = 4
        if current <= 1:
            print(
                "[oom retry] rt_sort: num_processes already at 1 — "
                "no further scaling possible."
            )
            return False
        new_n = max(1, int(current * float(factor)))
        if new_n >= current:
            return False
        rt.num_processes = new_n
        self._sync_globals()
        print(
            f"[oom retry] rt_sort: scaled num_processes {current} -> "
            f"{new_n} (factor={factor})."
        )
        return True

    def snapshot_oom_params(self) -> dict:
        """Snapshot ``rt_sort.num_processes`` for per-recording restore."""
        return {"num_processes": self.config.rt_sort.num_processes}

    def restore_oom_params(self, snapshot: dict) -> None:
        """Restore ``rt_sort.num_processes`` from a snapshot."""
        if not snapshot:
            return
        self.config.rt_sort.num_processes = snapshot.get("num_processes")
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
        """Extract waveforms via the shared custom WaveformExtractor.

        RT-Sort returns a standard SpikeInterface ``NumpySorting``, so
        the existing extraction pipeline used by the Kilosort backends
        works without modification.
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
