"""Sorter-agnostic spike sorting pipeline orchestration.

This module contains the functions that run after a sorter backend
has produced its output: SpikeData conversion, curation, compilation,
and epoch splitting.  These functions are independent of which sorter
was used — they operate on SpikeData and the ``SortingPipelineConfig``.

The backend-specific steps (loading, sorting, waveform extraction) are
handled by the ``SorterBackend`` subclass passed to
``process_recording``.
"""

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import warnings
from pathlib import Path

import numpy as np

from .config import SortingPipelineConfig

from .sorting_utils import (
    Stopwatch,
    Tee,
    print_stage,
    create_folder,
    delete_folder,
    get_paths,
)

# Display names for the source_format metadata field.
_SORTER_DISPLAY_NAMES = {
    "kilosort2": "Kilosort2",
    "kilosort4": "Kilosort4",
    "rt_sort": "RT-Sort",
}

# ---------------------------------------------------------------------------
# SpikeData conversion
# ---------------------------------------------------------------------------


def _get_noise_levels(
    recording: Any,
    return_scaled: bool = True,
    num_chunks: int = 20,
    chunk_size: int = 10000,
    seed: int = 0,
) -> np.ndarray:
    """Estimate per-channel noise using MAD on random recording chunks.

    Parameters:
        recording: SpikeInterface BaseRecording.
        return_scaled (bool): Use scaled traces.
        num_chunks (int): Number of random chunks to sample.
        chunk_size (int): Samples per chunk.
        seed (int): Random seed.

    Returns:
        noise_levels (np.ndarray): Per-channel noise, shape ``(channels,)``.
    """
    length = recording.get_num_samples()
    rng = np.random.RandomState(seed=seed)
    starts = rng.randint(0, length - chunk_size, size=num_chunks)
    chunks = []
    for s in starts:
        chunks.append(
            recording.get_traces(
                start_frame=s,
                end_frame=s + chunk_size,
                return_scaled=return_scaled,
            )
        )
    data = np.concatenate(chunks, axis=0)
    med = np.median(data, axis=0, keepdims=True)
    return np.median(np.abs(data - med), axis=0) / 0.6745


def build_spikedata(
    w_e: Any,
    rec_path: Any,
    config: Any,
    rec_chunks: Optional[list] = None,
    rec_chunk_names: Optional[list] = None,
) -> Any:
    """Convert a waveform extractor to a SpikeData with rich neuron attributes.

    This is the bridge between any sorter backend's waveform extractor
    and the sorter-agnostic downstream pipeline (curation, compilation).

    Parameters:
        w_e: Waveform extractor object (custom or SpikeInterface).
            Must provide: ``sorting``, ``recording``,
            ``sampling_frequency``, ``chans_max_all``, ``use_pos_peak``,
            ``peak_ind``, ``get_computed_template(unit_id, mode)``,
            ``ms_to_samples(ms)``, ``root_folder``.
        rec_path (str or Path): Original recording file path.
        config (SortingPipelineConfig): Pipeline configuration.
        rec_chunks (list of (int, int) or None): Frame boundaries for
            concatenated recording epochs.
        rec_chunk_names (list of str or None): File names for each epoch.

    Returns:
        sd (SpikeData): Enriched SpikeData with per-unit attributes.
    """
    from spikelab.spikedata import SpikeData

    wf_cfg = config.waveform
    sorting = w_e.sorting
    fs_Hz = float(w_e.sampling_frequency)
    rec_locations = w_e.recording.get_channel_locations()
    channel_ids = w_e.recording.get_channel_ids()

    try:
        electrode_ids = w_e.recording.get_property("electrode")
    except Exception:
        electrode_ids = None
    if electrode_ids is None:
        electrode_ids = channel_ids

    noise_levels = _get_noise_levels(w_e.recording, getattr(w_e, "return_scaled", True))

    use_pos_peak = w_e.use_pos_peak

    nbefore_compiled = w_e.ms_to_samples(wf_cfg.compiled_ms_before)
    nafter_compiled = w_e.ms_to_samples(wf_cfg.compiled_ms_after) + 1

    has_epochs = rec_chunks is not None and len(rec_chunks) > 1

    trains = []
    neuron_attributes = []
    for uid in sorting.unit_ids:
        spike_samples = sorting.get_unit_spike_train(uid)
        spike_times_ms = np.sort(spike_samples.astype(float) / fs_Hz * 1000.0)
        trains.append(spike_times_ms)

        chan_max = int(w_e.chans_max_all[uid])
        x, y = rec_locations[chan_max]

        template_mean = w_e.get_computed_template(unit_id=uid, mode="average")
        template_std = w_e.get_computed_template(unit_id=uid, mode="std")
        peak_ind_full = w_e.peak_ind

        # When scale_compiled_waveforms is False, convert µV templates
        # back to raw ADC counts for users who want raw values.
        if not wf_cfg.scale_compiled_waveforms and getattr(w_e, "return_scaled", False):
            gain = w_e.recording.get_channel_gains()
            offset = w_e.recording.get_channel_offsets()
            template_mean = ((template_mean - offset) / gain).astype(
                w_e.recording.get_dtype()
            )
            template_std = ((template_std - offset) / gain).astype(
                w_e.recording.get_dtype()
            )

        template_windowed = template_mean[
            peak_ind_full - nbefore_compiled : peak_ind_full + nafter_compiled, :
        ]

        template_abs = np.abs(template_windowed)
        peak_inds = np.argmax(template_abs, axis=0)
        amplitudes = template_abs[peak_inds, range(peak_inds.size)]
        amplitude_max = float(amplitudes[chan_max])

        noise = float(noise_levels[chan_max]) if chan_max < len(noise_levels) else 1.0
        snr = float(amplitude_max / noise) if noise > 0 else 0.0

        peak_ind_buffer = peak_ind_full - nbefore_compiled
        if wf_cfg.std_at_peak:
            stds = template_std[peak_ind_buffer + peak_inds, range(peak_inds.size)]
        else:
            nb = w_e.ms_to_samples(wf_cfg.std_over_window_ms_before)
            na = w_e.ms_to_samples(wf_cfg.std_over_window_ms_after) + 1
            stds = np.mean(
                template_std[
                    peak_ind_buffer + peak_inds - nb : peak_ind_buffer + peak_inds + na,
                    range(peak_inds.size),
                ],
                axis=0,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            std_norms_all = np.where(amplitudes > 0, stds / amplitudes, np.inf)
        std_norm = float(std_norms_all[chan_max])

        spike_train_samples = spike_samples.copy()

        attrs = {
            "unit_id": int(uid),
            "channel": chan_max,
            "channel_id": channel_ids[chan_max],
            "x": float(x),
            "y": float(y),
            "electrode": electrode_ids[chan_max],
            "template": template_mean[:, chan_max].copy(),
            "template_full": template_mean.copy(),
            "template_windowed": template_windowed.copy(),
            "template_peak_ind": int(peak_ind_full),
            "amplitude": amplitude_max,
            "amplitudes": amplitudes.copy(),
            "peak_inds": peak_inds.copy(),
            "std_norms_all": std_norms_all.copy(),
            "has_pos_peak": bool(use_pos_peak[uid]),
            "snr": snr,
            "std_norm": std_norm,
            "spike_train_samples": spike_train_samples,
        }

        # Per-epoch templates
        if has_epochs:
            wfs, sampled_indices = w_e.get_waveforms(uid, with_index=True)
            all_spike_samples = sorting.get_unit_spike_train(uid)
            epoch_templates = []
            for start_frame, end_frame in rec_chunks:
                epoch_mask = np.array(
                    [
                        start_frame <= all_spike_samples[idx] < end_frame
                        for idx in sampled_indices
                    ]
                )
                if np.any(epoch_mask):
                    epoch_wfs = wfs[epoch_mask]
                    epoch_avg = np.mean(epoch_wfs, axis=0)
                    epoch_templates.append(epoch_avg[:, chan_max].copy())
                else:
                    epoch_templates.append(np.zeros_like(template_mean[:, chan_max]))
            attrs["epoch_templates"] = epoch_templates

        wf_file = w_e.root_folder / "waveforms" / f"waveforms_{uid}.npy"
        if wf_file.exists():
            attrs["_waveforms_path"] = str(wf_file)
            attrs["_waveforms_window"] = (
                int(peak_ind_full - nbefore_compiled),
                int(peak_ind_full + nafter_compiled),
            )

        neuron_attributes.append(attrs)

    metadata = {
        "source_file": str(rec_path),
        "source_format": _SORTER_DISPLAY_NAMES.get(
            config.sorter.sorter_name, config.sorter.sorter_name
        ),
        "fs_Hz": fs_Hz,
        "channel_locations": rec_locations.copy(),
        "n_samples": int(w_e.recording.get_num_samples()),
    }
    if has_epochs:
        metadata["rec_chunks_frames"] = list(rec_chunks)
        metadata["rec_chunks_ms"] = [
            (s / fs_Hz * 1000.0, e / fs_Hz * 1000.0) for s, e in rec_chunks
        ]
        metadata["rec_chunk_names"] = list(rec_chunk_names) if rec_chunk_names else None

    return SpikeData(trains, metadata=metadata, neuron_attributes=neuron_attributes)


# ---------------------------------------------------------------------------
# Curation wrapper
# ---------------------------------------------------------------------------


def curate_spikedata(
    sd: Any, curation_folder: Any, config: Any, recurate: bool = False
) -> Tuple[Any, dict]:
    """Curate a SpikeData with disk caching.

    Reads curation thresholds from *config* and applies them via
    ``sd.curate()``.  Results are cached to *curation_folder*.

    Parameters:
        sd (SpikeData): Uncurated SpikeData.
        curation_folder (str or Path): Cache directory.
        config (SortingPipelineConfig): Pipeline configuration.
        recurate (bool): Re-run curation even when cached.

    Returns:
        sd_curated (SpikeData): Curated SpikeData.
        history (dict): Serializable curation history.
    """
    from spikelab.spikedata.curation import build_curation_history

    cur = config.curation
    curate_kwargs = {}

    if cur.curate_first:
        if cur.fr_min is not None:
            curate_kwargs["min_rate_hz"] = cur.fr_min
        if cur.isi_viol_max is not None:
            curate_kwargs["isi_max"] = cur.isi_viol_max
            curate_kwargs["isi_threshold_ms"] = 1.5
            curate_kwargs["isi_method"] = cur.isi_violation_method
        if cur.snr_min is not None:
            curate_kwargs["min_snr"] = cur.snr_min
        if cur.spikes_min_first is not None:
            curate_kwargs["min_spikes"] = cur.spikes_min_first
    if cur.curate_second:
        if cur.spikes_min_second is not None:
            curate_kwargs["min_spikes"] = cur.spikes_min_second
        if cur.std_norm_max is not None:
            curate_kwargs["max_std_norm"] = cur.std_norm_max

    curation_folder = Path(curation_folder)
    unit_ids_path = curation_folder / "unit_ids.npy"
    history_path = curation_folder / "curation_history.json"

    # Check cache
    if not recurate and unit_ids_path.exists() and history_path.exists():
        cached_ids = set(int(x) for x in np.load(str(unit_ids_path)))
        passing = [
            i
            for i in range(sd.N)
            if sd.neuron_attributes is not None
            and int(sd.neuron_attributes[i].get("unit_id", i)) in cached_ids
        ]
        sd_curated = sd.subset(passing)
        with open(history_path, "r") as f:
            history = json.load(f)
        return sd_curated, history

    # Run curation
    sd_curated, results = sd.curate(**curate_kwargs)
    history = build_curation_history(sd, sd_curated, results, parameters=curate_kwargs)

    # Save to disk
    curation_folder.mkdir(parents=True, exist_ok=True)
    np.save(str(unit_ids_path), np.array(history["curated_final"]))
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    return sd_curated, history


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class Compiler:
    """Aggregates sorting results from one or more SpikeData objects for export.

    Reads unit metadata from ``neuron_attributes`` and writes combined
    ``.npz``, ``.mat``, and figure outputs.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        fig = config.figures
        comp = config.compilation
        cur = config.curation

        self.create_figures = fig.create_figures
        self.create_std_scatter_plot = (
            cur.curate_second
            and cur.spikes_min_second is not None
            and cur.std_norm_max is not None
        )
        self.compile_to_mat = comp.compile_to_mat
        self.compile_to_npz = comp.compile_to_npz
        self.save_electrodes = comp.save_electrodes
        self.recs_cache = []

    def add_recording(
        self, rec_name: str, sd: Any, curation_history: Optional[dict] = None
    ) -> None:
        """Queue a recording for compilation.

        Parameters:
            rec_name (str): Short name for the recording.
            sd (SpikeData): Curated SpikeData.
            curation_history (dict or None): Curation history dict.
        """
        self.recs_cache.append((rec_name, sd, curation_history))

    def save_results(self, folder: Any) -> None:
        """Compile and save results from all queued recordings.

        Parameters:
            folder (Path or str): Output directory.
        """
        try:
            from scipy.io import savemat
        except ImportError:
            savemat = None

        create_folder(folder)
        folder = Path(folder)

        cfg = self.config
        comp = cfg.compilation
        fig = cfg.figures

        all_units = []
        rec_metadata = {}
        bar_rec_names = []
        bar_n_total = []
        bar_n_selected = []
        scatter_n_spikes = {}
        scatter_std_norms = {}
        fig_fs_Hz = None

        for rec_name, sd, curation_history in self.recs_cache:
            print(f"Adding recording: {rec_name}")

            fs_Hz = sd.metadata.get("fs_Hz", 30000.0)
            rec_metadata[rec_name] = {
                "fs": fs_Hz,
                "locations": sd.metadata.get("channel_locations"),
                "n_samples": sd.metadata.get("n_samples", 0),
            }
            if fig_fs_Hz is None:
                fig_fs_Hz = fs_Hz

            for i in range(sd.N):
                attrs = sd.neuron_attributes[i] if sd.neuron_attributes else {}
                all_units.append((attrs, True, rec_name))

            if self.create_figures:
                curated_ids = set()
                if sd.neuron_attributes is not None:
                    for attrs in sd.neuron_attributes:
                        curated_ids.add(int(attrs.get("unit_id", -1)))
                n_total = len(curated_ids)
                if curation_history is not None:
                    n_total = len(curation_history.get("initial", curated_ids))
                bar_rec_names.append(rec_name)
                bar_n_total.append(n_total)
                bar_n_selected.append(sd.N)

                if self.create_std_scatter_plot and curation_history is not None:
                    scatter_n_spikes[rec_name] = curation_history.get(
                        "metrics", {}
                    ).get("spike_count", {})
                    scatter_std_norms[rec_name] = curation_history.get(
                        "metrics", {}
                    ).get("std_norm", {})

        # Sort by polarity then amplitude
        neg_units = [u for u in all_units if not u[0].get("has_pos_peak", False)]
        pos_units = [u for u in all_units if u[0].get("has_pos_peak", False)]
        neg_units.sort(key=lambda x: float(x[0].get("amplitude", 0)), reverse=True)
        pos_units.sort(key=lambda x: float(x[0].get("amplitude", 0)), reverse=True)

        compile_dict = None
        if self.compile_to_mat or self.compile_to_npz:
            if len(rec_metadata) == 1:
                rec = list(rec_metadata.keys())[0]
                meta = rec_metadata[rec]
                compile_dict = {
                    "units": [],
                    "locations": meta["locations"],
                    "fs": meta["fs"],
                }

        if comp.compile_waveforms:
            create_folder(folder / "negative_peaks")
            create_folder(folder / "positive_peaks")

        fig_templates = []
        fig_peak_indices = []
        fig_is_curated = []
        fig_has_pos_peak = []

        sorted_index = 0
        for group_label, units_group in [
            ("negative", neg_units),
            ("positive", pos_units),
        ]:
            has_pos = group_label == "positive"
            print(
                f"\nIterating through {len(units_group)} units with "
                f"{group_label} peaks"
            )
            for attrs, is_curated, rec_name in units_group:
                if is_curated:
                    if compile_dict is not None:
                        spike_train_samples = attrs.get("spike_train_samples")
                        if comp.save_dl_data:
                            unit_dict = {
                                "unit_id": attrs.get("unit_id"),
                                "spike_train": spike_train_samples,
                                "x_max": attrs.get("x"),
                                "y_max": attrs.get("y"),
                                "template": attrs.get("template_windowed"),
                                "sorted_index": sorted_index,
                                "max_channel_si": attrs.get("channel"),
                                "max_channel_id": attrs.get("channel_id"),
                                "peak_sign": group_label,
                                "peak_ind": attrs.get("peak_inds"),
                                "amplitudes": attrs.get("amplitudes"),
                                "std_norms": attrs.get("std_norms_all"),
                            }
                        else:
                            unit_dict = {
                                "unit_id": attrs.get("unit_id"),
                                "spike_train": spike_train_samples,
                                "x_max": attrs.get("x"),
                                "y_max": attrs.get("y"),
                                "template": attrs.get("template_windowed"),
                            }
                        if self.save_electrodes:
                            unit_dict["electrode"] = attrs.get("electrode")
                        compile_dict["units"].append(unit_dict)

                    if comp.compile_waveforms:
                        wf_path = attrs.get("_waveforms_path")
                        wf_window = attrs.get("_waveforms_window")
                        if wf_path is not None:
                            waveforms = np.load(wf_path, mmap_mode="r")
                            if wf_window is not None:
                                waveforms = waveforms[:, wf_window[0] : wf_window[1], :]
                            wf_folder = (
                                folder / "positive_peaks"
                                if has_pos
                                else folder / "negative_peaks"
                            )
                            np.save(
                                wf_folder / f"waveforms_{sorted_index}.npy",
                                np.array(waveforms),
                            )

                    sorted_index += 1

                if self.create_figures:
                    fig_templates.append(attrs.get("template", np.array([])))
                    fig_peak_indices.append(attrs.get("template_peak_ind", 0))
                    fig_is_curated.append(is_curated)
                    fig_has_pos_peak.append(has_pos)

        if compile_dict is not None:
            if self.compile_to_mat and savemat is not None:
                savemat(folder / "sorted.mat", compile_dict)
                print("Compiled results to .mat")
            if self.compile_to_npz:
                np.savez(folder / "sorted.npz", **compile_dict)
                print("Compiled results to .npz")

        if self.create_figures:
            from .figures import plot_curation_bar, plot_std_scatter, plot_templates

            figures_path = folder / "figures"
            print("\nSaving figures")
            create_folder(figures_path)

            plot_curation_bar(
                bar_rec_names,
                bar_n_total,
                bar_n_selected,
                total_label=fig.bar_total_label,
                selected_label=fig.bar_selected_label,
                x_label=fig.bar_x_label,
                y_label=fig.bar_y_label,
                label_rotation=fig.bar_label_rotation,
                save_path=str(figures_path / "curation_bar_plot.png"),
            )
            print("Curation bar plot has been saved")

            if self.create_std_scatter_plot and scatter_n_spikes:
                plot_std_scatter(
                    scatter_n_spikes,
                    scatter_std_norms,
                    spikes_thresh=cfg.curation.spikes_min_second,
                    std_thresh=cfg.curation.std_norm_max,
                    colors=fig.scatter_recording_colors[:],
                    alpha=fig.scatter_recording_alpha,
                    x_label=fig.scatter_x_label,
                    y_label=fig.scatter_y_label,
                    x_max_buffer=fig.scatter_x_max_buffer,
                    y_max_buffer=fig.scatter_y_max_buffer,
                    save_path=str(figures_path / "std_scatter_plot.png"),
                )
                print("Std scatter plot has been saved")

            if fig_templates and fig_fs_Hz is not None:
                plot_templates(
                    fig_templates,
                    fig_peak_indices,
                    fig_fs_Hz,
                    fig_is_curated,
                    fig_has_pos_peak,
                    templates_per_column=fig.templates_per_column,
                    y_spacing=fig.templates_y_spacing,
                    y_lim_buffer=fig.templates_y_lim_buffer,
                    color_curated=fig.templates_color_curated,
                    color_failed=fig.templates_color_failed,
                    window_ms_before=fig.templates_window_ms_before,
                    window_ms_after=fig.templates_window_ms_after,
                    line_ms_before=fig.templates_line_ms_before,
                    line_ms_after=fig.templates_line_ms_after,
                    x_label=fig.templates_x_label,
                    save_path=str(figures_path / "all_templates_plot.png"),
                )
                print("All templates plot has been saved")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def process_recording(
    backend,
    config,
    rec_name,
    rec_path,
    inter_path,
    results_path,
    rec_loaded=None,
    rec_chunks=None,
    rec_chunk_names=None,
    rng=None,
):
    """Run the full sorting pipeline on a single recording.

    Delegates loading, sorting, and waveform extraction to the
    *backend*, then handles SpikeData conversion, curation, and
    compilation using the *config*.

    Parameters:
        backend (SorterBackend): Sorter backend instance.
        config (SortingPipelineConfig): Pipeline configuration.
        rec_name (str): Short name for the recording.
        rec_path (str or Path): Path to the recording file.
        inter_path (str or Path): Root intermediate directory.
        results_path (str or Path): Root results directory.
        rec_loaded: Pre-loaded recording object, or None.
        rec_chunks (list of (int, int) or None): Epoch frame boundaries.
        rec_chunk_names (list of str or None): Epoch file names.
        rng (np.random.Generator or None): Random number generator for
            reproducible waveform sampling.  When ``None``, a new
            ``default_rng()`` is created.

    Returns:
        result (SpikeData or tuple or Exception): ``sd_curated`` on
            success, or ``(sd_raw, sd_curated)`` when
            ``config.compilation.save_raw_pkl`` is True.  Returns the
            caught exception if any stage failed.
    """
    exe = config.execution
    cur = config.curation
    comp = config.compilation

    create_folder(inter_path)
    # Acquire a per-recording lock so two ``sort_recording`` calls
    # against the same intermediate folder cannot corrupt each
    # other's binary artefacts. Stale locks from crashed sorts are
    # reclaimed automatically. A concurrent-sort failure returns
    # the exception as a sentinel — we never enter the sort body.
    from ._exceptions import ConcurrentSortError
    from .guards import acquire_sort_lock

    sort_lock_cm = acquire_sort_lock(Path(inter_path))
    try:
        sort_lock_cm.__enter__()
    except ConcurrentSortError as exc:
        print(f"Concurrent sort detected: {exc}")
        print("Moving on to next recording")
        return exc

    try:
        return _process_recording_body(
            backend=backend,
            config=config,
            rec_name=rec_name,
            rec_path=rec_path,
            inter_path=inter_path,
            results_path=results_path,
            rec_loaded=rec_loaded,
            rec_chunks=rec_chunks,
            rec_chunk_names=rec_chunk_names,
            rng=rng,
        )
    finally:
        sort_lock_cm.__exit__(None, None, None)


def _process_recording_body(
    backend,
    config,
    rec_name,
    rec_path,
    inter_path,
    results_path,
    rec_loaded=None,
    rec_chunks=None,
    rec_chunk_names=None,
    rng=None,
):
    """Inner body of process_recording — runs inside the sort lock."""
    exe = config.execution
    cur = config.curation
    comp = config.compilation

    with Tee(Path(inter_path) / exe.out_file, "a"):
        stopwatch = Stopwatch()

        (
            rec_path,
            inter_path,
            recording_dat_path,
            output_folder,
            waveforms_root_folder,
            curation_initial_folder,
            curation_first_folder,
            curation_second_folder,
            results_path,
        ) = get_paths(rec_path, inter_path, results_path, exe)

        # Load Recording
        try:
            recording_filtered = backend.load_recording(
                rec_path if rec_loaded is None else rec_loaded
            )
        except Exception as e:
            print(f"Could not open the recording file because of {e}")
            print("Moving on to next recording")
            return e

        # Everything past this point is wrapped so that a failure in
        # waveform extraction, build_spikedata, curation, figure
        # generation, or compile cannot kill the surrounding batch
        # loop. KeyboardInterrupt is caught specifically because the
        # host-memory watchdog uses ``_thread.interrupt_main`` to
        # abort the run; we re-raise it as a classified
        # ``HostMemoryWatchdogError`` so the caller can route it as
        # a resource failure.
        from .guards import get_active_watchdog

        try:
            # Spike sorting
            sorting = backend.sort(
                recording_filtered, rec_path, recording_dat_path, output_folder
            )
            if isinstance(sorting, BaseException):
                return sorting

            # Extract waveforms
            w_e_raw = backend.extract_waveforms(
                recording_filtered,
                sorting,
                waveforms_root_folder,
                curation_initial_folder,
                rec_path=rec_path,
                rng=rng,
            )

            # Convert to SpikeData
            sd = build_spikedata(
                w_e_raw,
                rec_path,
                config,
                rec_chunks=rec_chunks,
                rec_chunk_names=rec_chunk_names,
            )

            # Generate figures if create_figures is enabled.
            # Per-unit figures are generated before curation (while individual
            # waveforms are still on disk), then sorted into curated/failed
            # subdirs after curation completes.
            unit_figures_dir = Path(results_path) / "figures" / "units"
            _fig = {}
            figures_dir = Path(results_path) / "figures"
            _thresholds = {
                "fr_min": cur.fr_min,
                "isi_viol_max": cur.isi_viol_max,
                "snr_min": cur.snr_min,
                "spikes_min_second": cur.spikes_min_second,
                "std_norm_max": cur.std_norm_max,
            }

            if not config.figures.create_figures:
                print("Skipping figure generation (create_figures=False)")
            else:
                unit_figures_dir.mkdir(parents=True, exist_ok=True)
                figures_dir.mkdir(parents=True, exist_ok=True)

                _fmod = None
                try:
                    from scripts import generate_sorting_figures as _fmod
                except ImportError:
                    import importlib.util

                    _script = (
                        Path(__file__).parents[2]
                        / "scripts"
                        / "generate_sorting_figures.py"
                    )
                    if _script.exists():
                        _spec = importlib.util.spec_from_file_location(
                            "generate_sorting_figures", _script
                        )
                        _fmod = importlib.util.module_from_spec(_spec)
                        _spec.loader.exec_module(_fmod)

                if _fmod is not None:
                    for name in (
                        "generate_per_unit_figures",
                        "generate_quality_distributions",
                        "generate_builtin_figures",
                        "generate_raster_overview",
                    ):
                        _fig[name] = getattr(_fmod, name, None)

                if (
                    config.figures.create_unit_figures
                    and _fig.get("generate_per_unit_figures") is not None
                ):
                    print_stage("GENERATING PER-UNIT FIGURES")
                    _fig["generate_per_unit_figures"](
                        sd,
                        unit_figures_dir,
                        amp_thresh_uv=15.0,
                        w_e_raw=w_e_raw,
                    )
                elif not config.figures.create_unit_figures:
                    print("Skipping per-unit figures (create_unit_figures=False)")

                if _fig.get("generate_quality_distributions") is not None:
                    print_stage("GENERATING QUALITY DISTRIBUTIONS (ALL UNITS)")
                    _fig["generate_quality_distributions"](
                        sd,
                        is_pre_curation=True,
                        thresholds=_thresholds,
                        out_dir=figures_dir,
                    )

            # Curate
            has_epochs = bool(sd.metadata.get("rec_chunks_ms"))
            if cur.curation_epoch is not None and has_epochs:
                epoch_sds = sd.split_epochs()
                if cur.curation_epoch < 0 or cur.curation_epoch >= len(epoch_sds):
                    raise ValueError(
                        f"curation_epoch={cur.curation_epoch} is out of range "
                        f"(recording has {len(epoch_sds)} epochs, 0-indexed)."
                    )
                sd_for_curation = epoch_sds[cur.curation_epoch]
                print(
                    f"Curating based on epoch {cur.curation_epoch} "
                    f"({sd_for_curation.metadata.get('source_file', '')})"
                )
            else:
                sd_for_curation = sd

            sd_epoch_curated, curation_history = curate_spikedata(
                sd_for_curation,
                curation_folder=curation_first_folder,
                config=config,
                recurate=exe.recurate_first or exe.recurate_second,
            )

            # When curating on a single epoch, apply passing units to full SD
            if sd_for_curation is not sd:
                passing_ids = set()
                if sd_epoch_curated.neuron_attributes is not None:
                    for attrs in sd_epoch_curated.neuron_attributes:
                        uid = attrs.get("unit_id")
                        if uid is not None:
                            passing_ids.add(int(uid))
                passing_indices = [
                    i
                    for i in range(sd.N)
                    if sd.neuron_attributes is not None
                    and int(sd.neuron_attributes[i].get("unit_id", -1)) in passing_ids
                ]
                sd_curated = sd.subset(passing_indices)
            else:
                sd_curated = sd_epoch_curated

            n_before = sd.N
            n_after = sd_curated.N
            print(
                f"Curation: {n_before} -> {n_after} units "
                f"({n_before - n_after} removed)"
            )

            # Sort per-unit figures into curated/failed subdirectories
            if unit_figures_dir.exists() and any(unit_figures_dir.glob("unit_*.png")):
                curated_ids = set()
                if sd_curated.neuron_attributes is not None:
                    for attrs in sd_curated.neuron_attributes:
                        uid = attrs.get("unit_id")
                        if uid is not None:
                            curated_ids.add(int(uid))

                curated_dir = unit_figures_dir / "curated"
                failed_dir = unit_figures_dir / "failed"
                curated_dir.mkdir(exist_ok=True)
                failed_dir.mkdir(exist_ok=True)

                for png in unit_figures_dir.glob("unit_*.png"):
                    try:
                        uid = int(png.stem.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    dest = curated_dir if uid in curated_ids else failed_dir
                    shutil.move(str(png), str(dest / png.name))

                n_curated_figs = len(list(curated_dir.glob("*.png")))
                n_failed_figs = len(list(failed_dir.glob("*.png")))
                print(
                    f"Per-unit figures sorted: {n_curated_figs} curated, "
                    f"{n_failed_figs} failed"
                )

            # Generate remaining figures (need curated SpikeData)
            if _fig.get("generate_builtin_figures") is not None:
                print_stage("GENERATING QC FIGURES")
                _fig["generate_builtin_figures"](sd_curated, _thresholds, figures_dir)
            if _fig.get("generate_raster_overview") is not None:
                generate_raster_overview = _fig["generate_raster_overview"]
                generate_raster_overview(sd_curated, figures_dir)

            # Compile results
            compile_results(
                config,
                rec_name,
                rec_path,
                results_path,
                sd_curated,
                curation_history,
                rec_chunks,
            )

            print_stage("DONE WITH RECORDING")
            print(f"Recording: {rec_path}")
            stopwatch.log_time("Total")

            if comp.save_raw_pkl:
                return sd, sd_curated
            return sd_curated
        except KeyboardInterrupt:
            # The host-memory watchdog uses ``_thread.interrupt_main``
            # to abort the run. Re-raise as a classified resource
            # failure so the per-recording loop can route it the same
            # way as a GPU OOM.
            wd = get_active_watchdog()
            if wd is not None and wd.tripped():
                err = wd.make_error()
                print(f"Recording aborted by host-memory watchdog: {err}")
                return err
            raise
        except MemoryError as e:
            print(f"Recording aborted due to MemoryError: {e!r}")
            print("Moving on to next recording")
            return e
        except Exception as e:
            print(f"Recording failed in post-sort pipeline: {e!r}")
            print("Moving on to next recording")
            return e


def compile_results(
    config, rec_name, rec_path, results_path, sd, curation_history=None, rec_chunks=None
):
    """Compile and export sorting results for a single recording.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.
        rec_name (str): Short name for the recording.
        rec_path (str or Path): Original recording file path.
        results_path (Path): Output directory.
        sd (SpikeData): Curated SpikeData.
        curation_history (dict or None): Curation history dict.
        rec_chunks (list or None): Epoch frame boundaries.
    """
    comp = config.compilation
    exe = config.execution

    compile_stopwatch = Stopwatch("COMPILING RESULTS")
    print(f"For recording: {rec_path}")
    if comp.compile_single_recording:
        if (
            not (Path(results_path) / "parameters.json").exists()
            or exe.recompile_single_recording
        ):
            print(f"Saving to path: {results_path}")
            if rec_chunks is not None and len(rec_chunks) > 1:
                epoch_sds = sd.split_epochs()
                for c, sd_chunk in enumerate(epoch_sds):
                    print(f"Compiling chunk {c}")
                    compiler = Compiler(config)
                    compiler.add_recording(rec_name, sd_chunk, curation_history)
                    compiler.save_results(Path(results_path) / f"chunk{c}")
            else:
                compiler = Compiler(config)
                compiler.add_recording(rec_name, sd, curation_history)
                compiler.save_results(results_path)
                compile_stopwatch.log_time("Done compiling results.")
        else:
            print(
                "Skipping compiling results because 'recompile_single_recording' "
                "is set to False and already compiled"
            )
    else:
        print(
            "Skipping compiling results because 'compile_single_recording' "
            "is set to False"
        )


# ---------------------------------------------------------------------------
# Generic entry points
# ---------------------------------------------------------------------------


from contextlib import contextmanager


@contextmanager
def _bounded_host_memory(frac: float = 0.8):
    """Cap the calling process's heap allocations at ``frac`` of system RAM.

    Best-effort guard against OOM during local sorting (especially RT-Sort,
    which can exhaust host RAM on long recordings or high-unit-count
    populations). Uses ``RLIMIT_DATA`` rather than ``RLIMIT_AS`` so that
    file-backed mmap regions used for recording I/O are not capped — only
    anonymous heap allocations (numpy / torch tensors) are bounded, which
    is where the OOM actually originates.

    Behaviour by platform:
        - Linux (kernel 4.7+): caps anonymous heap (brk + anonymous mmap).
          This is the strict OOM guard intended.
        - macOS / other POSIX: caps the brk segment only; large mmap
          allocations are not capped (semantics are weaker).
        - Windows: no-op with a printed notice (``resource`` module
          unavailable). Host RAM is unprotected — rely on Docker's
          ``mem_limit`` for containerised sorters, or monitor RAM
          manually for local runs.

    The original soft limit is restored on context exit so the cap does
    not leak into longer-lived sessions (e.g. notebooks).

    Parameters:
        frac (float): Fraction of total physical RAM to cap heap at.
            Defaults to ``0.8``.
    """
    try:
        import resource
    except ImportError:
        print(
            "[host memory cap] Windows detected — RLIMIT_DATA unavailable. "
            "Local sorting is not protected from host OOM. "
            "Use Docker, or monitor RAM manually."
        )
        yield
        return

    from .sorting_utils import get_system_ram_bytes

    ram_bytes = get_system_ram_bytes()
    if ram_bytes is None:
        print("[host memory cap] Could not detect system RAM; cap not enforced.")
        yield
        return

    new_soft = int(ram_bytes * frac)
    soft_orig, hard_orig = resource.getrlimit(resource.RLIMIT_DATA)
    if hard_orig != resource.RLIM_INFINITY and new_soft > hard_orig:
        new_soft = hard_orig

    try:
        resource.setrlimit(resource.RLIMIT_DATA, (new_soft, hard_orig))
    except (ValueError, OSError) as exc:
        print(f"[host memory cap] Failed to set RLIMIT_DATA: {exc}; cap not enforced.")
        yield
        return

    try:
        yield
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (soft_orig, hard_orig))
        except (ValueError, OSError):
            pass


def _print_pipeline_banner(
    sorter: str,
    rec_path: Any,
    config: "SortingPipelineConfig",
    log_path: Path,
    recording: Any = None,
    docker_image_tag: Optional[str] = None,
    docker_image_digest: Optional[str] = None,
) -> None:
    """Print an environment + system + input banner at the start of a sort.

    Captured by the surrounding ``Tee`` and persisted to the
    ``sorting_*.log`` file alongside the run's stdout.

    Parameters:
        sorter (str): Sorter name (``"kilosort2"``, ``"kilosort4"``,
            ``"rt_sort"``).
        rec_path: Recording file path used for the run.
        config (SortingPipelineConfig): Pipeline configuration.
        log_path (Path): Path to the per-recording Tee log file.
        recording: Optional pre-loaded ``BaseRecording``. When
            provided and ``sorter == "rt_sort"``, the banner prints a
            projected on-disk size for RT-Sort's intermediate files.
        docker_image_tag (str or None): Resolved Docker image tag for
            this sort, when ``use_docker`` is set. Printed under the
            Environment section so the post-sort report can surface it.
        docker_image_digest (str or None): Local Docker image digest
            (``sha256:...``) for the resolved tag, when available.
            Recording the digest pins the actually-used image bits
            for audit / reproducibility.
    """
    import datetime as _dt
    import platform
    import socket
    import subprocess

    from .sorting_utils import get_system_ram_bytes, print_stage

    print_stage(f"SPIKE SORTING — {sorter.upper()}")
    print()
    print("-- Environment --")
    print(f"Started:        {_dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"Host:           {socket.gethostname()}")
    print(f"Platform:       {platform.platform()}")
    print(f"Python:         {sys.version.split()[0]}")

    try:
        import spikeinterface as _si

        print(f"SpikeInterface: {_si.__version__}")
    except ImportError:
        pass

    try:
        import spikelab as _sl

        version = getattr(_sl, "__version__", "unknown")
        print(f"SpikeLab:       {version}")
    except ImportError:
        pass

    print()
    print("-- System Resources --")
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        print(f"CPU cores:      {cpu_count}")

    ram_bytes = get_system_ram_bytes()
    if ram_bytes is not None:
        print(f"RAM total:      {ram_bytes / 1e9:.1f} GB")

    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_DATA)
        if soft == resource.RLIM_INFINITY:
            print("Heap cap:       (unlimited)")
        else:
            print(f"Heap cap:       {soft / 1e9:.1f} GB (RLIMIT_DATA)")
    except ImportError:
        print("Heap cap:       (Windows — not enforced)")

    try:
        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
        print(f"GPU:            {gpu_info}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("GPU:            (nvidia-smi unavailable)")

    if config.sorter.use_docker:
        if docker_image_tag:
            print(f"Docker image:   {docker_image_tag}")
        if docker_image_digest:
            print(f"Docker image digest: {docker_image_digest}")

    print()
    print("-- Run --")
    print(f"Sorter:         {sorter}")
    print(f"Use Docker:     {config.sorter.use_docker}")
    print(f"Recording:      {rec_path}")
    print(f"Log file:       {log_path}")

    # RT-Sort projects a sizeable on-disk footprint per recording
    # (scaled traces + model traces + model outputs). Print it here
    # so the user sees the requirement before the sort starts —
    # complements the ``low_disk_inter`` preflight check.
    if sorter.lower() == "rt_sort" and recording is not None:
        try:
            from .guards import estimate_rt_sort_intermediate_gb

            n_ch = int(recording.get_num_channels())
            n_smp = int(recording.get_num_samples())
            projected_gb = estimate_rt_sort_intermediate_gb(
                n_channels=n_ch, n_samples=n_smp
            )
            print(f"RT-Sort disk:   ~{projected_gb:.1f} GB intermediates projected")
        except Exception:
            pass

    print()


def _print_pipeline_summary(
    status: str,
    elapsed_s: float,
    error: Optional[BaseException] = None,
) -> None:
    """Print a closing summary banner with status, wall time, and resources."""
    import datetime as _dt
    import subprocess

    from .sorting_utils import get_system_ram_bytes, print_stage

    print()
    print_stage("SUMMARY")
    print()
    print(f"Status:         {status}")
    if error is not None:
        print(f"Error:          {type(error).__name__}: {error}")

    minutes, seconds = divmod(int(elapsed_s), 60)
    print(f"Wall time:      {minutes}m {seconds}s")

    ram_bytes = get_system_ram_bytes()
    if ram_bytes is not None:
        print(f"RAM total:      {ram_bytes / 1e9:.1f} GB")
    try:
        gpu_mem = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
        print(f"GPU memory:     {gpu_mem}")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    print(f"Finished:       {_dt.datetime.now().isoformat(timespec='seconds')}")


def sort_recording(
    recording_files,
    config=None,
    sorter="kilosort2",
    intermediate_folders=None,
    results_folders=None,
    *,
    out_report: Optional["SortRunReport"] = None,
    **kwargs,
):
    """Run spike sorting on one or more recordings using any registered backend.

    This is the primary entry point for the modular sorting pipeline.

    Parameters:
        recording_files (list): Paths to recording files or directories.
            Each entry is sorted independently. Directories have their
            contents concatenated before sorting and split back into
            per-file SpikeData afterward.
        config (SortingPipelineConfig or None): Pre-built configuration.
            When provided, ``**kwargs`` are applied as overrides via
            ``config.override()``.  When None, a fresh config is built
            from ``sorter`` + ``**kwargs``.  Preset configs are
            available in ``spikelab.spike_sorting.config`` (e.g.
            ``KILOSORT2``).
        sorter (str): Registered sorter backend name.  Only used when
            ``config`` is None.  Available: ``"kilosort2"``,
            ``"kilosort4"``.
        intermediate_folders (list or None): Intermediate result
            directories, one per recording.  Auto-generated if None.
        results_folders (list or None): Output directories, one per
            recording.  Auto-generated if None.
        out_report (SortRunReport or None): Optional report instance
            populated in-place with one :class:`RecordingResult` per
            input recording. The same information is always written
            per-recording to ``<results_folder>/recording_report.json``
            regardless of this argument; ``out_report`` only adds a
            programmatic accessor for the batch.
        **kwargs: Override individual config fields (e.g.
            ``snr_min=5.0``, ``use_docker=True``, ``fr_min=0.05``).
            See ``spikelab.spike_sorting.config`` for all available
            parameters, grouped by: ``RecordingConfig``,
            ``SorterConfig``, ``WaveformConfig``, ``CurationConfig``,
            ``CompilationConfig``, ``FigureConfig``,
            ``ExecutionConfig``.

    Returns:
        results (list[SpikeData]): One SpikeData per original recording
            file.  For directory inputs, the concatenated recording is
            split back into per-file SpikeData objects.

    Notes:
        - Pickle files (``sorted_spikedata_curated.pkl`` and optionally
          ``sorted_spikedata.pkl``) are saved to each results folder.
        - ``hdf5_plugin_path`` (passed via config or kwargs) sets
          ``os.environ['HDF5_PLUGIN_PATH']`` before any recording is
          loaded.  This is needed for Maxwell ``.h5`` files and
          applies to all backends.
    """
    import datetime

    from .backends import get_backend_class
    from .config import SortingPipelineConfig

    if config is not None:
        if kwargs:
            config = config.override(**kwargs)
        sorter = config.sorter.sorter_name
    else:
        config = SortingPipelineConfig.from_kwargs(**kwargs)

    # Set HDF5 plugin path before any recording is loaded (affects all backends)
    if config.recording.hdf5_plugin_path is not None:
        import os

        os.environ["HDF5_PLUGIN_PATH"] = str(config.recording.hdf5_plugin_path)

    backend_cls = get_backend_class(sorter)
    backend = backend_cls(config)

    # Auto-generate folder paths
    def _rec_to_path(rec):
        try:
            from spikeinterface.core import BaseRecording as _BR
        except ImportError:
            _BR = None
        if _BR is not None and isinstance(rec, _BR):
            kw = rec._kwargs
            backing = kw.get("file_path") or (kw.get("file_paths") or [None])[0]
            if backing is None:
                raise ValueError(
                    f"Cannot auto-generate intermediate_folders / "
                    f"results_folders for a {type(rec).__name__} without a "
                    f"backing file path.  Pass `intermediate_folders` and "
                    f"`results_folders` explicitly."
                )
            return Path(backing)
        return Path(rec)

    if intermediate_folders is None:
        cur_dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        intermediate_folders = [
            _rec_to_path(rec).parent / f"inter_{sorter}_{cur_dt}"
            for rec in recording_files
        ]
    if results_folders is None:
        results_folders = [
            _rec_to_path(rec).parent / f"sorted_{sorter}" for rec in recording_files
        ]
    # Validate
    if not (len(recording_files) == len(intermediate_folders) == len(results_folders)):
        raise ValueError(
            f"recording_files ({len(recording_files)}), "
            f"intermediate_folders ({len(intermediate_folders)}), and "
            f"results_folders ({len(results_folders)}) must all have "
            "the same length."
        )

    # Figure settings
    try:
        import matplotlib as mpl

        if config.figures.create_figures:
            if config.figures.dpi is not None:
                mpl.rcParams["figure.dpi"] = config.figures.dpi
            if config.figures.font_size is not None:
                mpl.rcParams["font.size"] = config.figures.font_size
    except ImportError:
        pass

    rng = np.random.default_rng(config.execution.random_seed)

    # Preflight checks: free disk, available RAM, GPU VRAM, HDF5 plugin path.
    # Findings are printed; ``preflight_strict`` flips warnings into hard
    # failures.
    if config.execution.preflight:
        from .guards import report_findings, run_preflight

        try:
            findings = run_preflight(
                config,
                recording_files,
                intermediate_folders,
                results_folders,
            )
            report_findings(findings, strict=config.execution.preflight_strict)
        except Exception as exc:
            print(f"Preflight aborted the run: {exc!r}")
            raise

    # Main loop — wrap in a host heap cap (Linux-only), a live host
    # memory watchdog (cross-platform), and a GPU memory watchdog
    # (when a GPU device is reachable) so local sorts cannot drag the
    # workstation into swap or fragment GPU VRAM into a hard failure.
    # The host-memory watchdog is the primary protection on Windows,
    # where ``_bounded_host_memory`` is a no-op.
    from contextlib import nullcontext

    from .guards import GpuMemoryWatchdog, HostMemoryWatchdog, resolve_active_device

    exe_cfg = config.execution
    if exe_cfg.host_ram_watchdog:
        watchdog_ctx = HostMemoryWatchdog(
            warn_pct=exe_cfg.host_ram_warn_pct,
            abort_pct=exe_cfg.host_ram_abort_pct,
            poll_interval_s=exe_cfg.host_ram_poll_interval_s,
        )
    else:
        watchdog_ctx = nullcontext()

    if getattr(exe_cfg, "gpu_watchdog", True):
        gpu_watchdog_ctx = GpuMemoryWatchdog(
            device_index=resolve_active_device(config),
            warn_pct=exe_cfg.gpu_warn_pct,
            abort_pct=exe_cfg.gpu_abort_pct,
            poll_interval_s=exe_cfg.gpu_poll_interval_s,
        )
    else:
        gpu_watchdog_ctx = nullcontext()

    # Per-batch report. Always populated; if the caller passed an
    # ``out_report`` we mirror entries into it as well.
    report = SortRunReport()

    # Windows Job Object kernel-enforced memory cap. Complements
    # the userspace ``HostMemoryWatchdog`` — the watchdog provides
    # warn-stage detection at 85% and graceful abort at 92%, the
    # Job Object is the kernel-level hard limit (configurable via
    # ``host_ram_abort_pct``). No-op on non-Windows or when
    # pywin32 is missing.
    from .guards import (
        cleanup_temp_files,
        prevent_system_sleep,
        windows_job_object_cap,
    )

    job_object_cap = windows_job_object_cap(
        frac=float(exe_cfg.host_ram_abort_pct) / 100.0
    )

    # Sweep stale sorter temp files at batch end (only on clean exit
    # — failures intentionally leave temp files for inspection).
    temp_cleanup_ctx = cleanup_temp_files(
        enabled=getattr(exe_cfg, "cleanup_temp_files", True)
    )

    # Prevent Windows from sleeping mid-sort (lid-close, modern
    # standby). No-op on non-Windows.
    if getattr(exe_cfg, "prevent_system_sleep", True):
        sleep_lock_ctx = prevent_system_sleep()
    else:
        sleep_lock_ctx = nullcontext()

    spikedata_results = []
    with (
        _bounded_host_memory(0.8),
        job_object_cap,
        sleep_lock_ctx,
        temp_cleanup_ctx,
        watchdog_ctx,
        gpu_watchdog_ctx,
    ):
        for rec_path, inter_path, res_path in zip(
            recording_files, intermediate_folders, results_folders
        ):
            try:
                from spikeinterface.core import BaseRecording
            except ImportError:
                BaseRecording = None

            rec_loaded = None
            if BaseRecording is not None and isinstance(rec_path, BaseRecording):
                rec_loaded = rec_path
                if "file_path" in rec_loaded._kwargs:
                    rec_path = rec_loaded._kwargs["file_path"]
                else:
                    rec_path = rec_loaded._kwargs["file_paths"][0]

            rec_name = str(rec_path).split("/")[-1].split("\\")[-1].split(".")[0]

            # Mirror stdout to a per-recording log file from start to finish.
            # The log captures the environment banner, every sorting stage, the
            # closing summary, and any exception traceback — making it the
            # canonical artefact for the post-sorting report.
            res_path_obj = Path(res_path)
            res_path_obj.mkdir(parents=True, exist_ok=True)
            log_ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            log_path = res_path_obj / f"sorting_{log_ts}.log"

            # Resolve the Docker image tag + local digest once per
            # recording. The digest is recorded in config_used.json
            # and printed in the banner so two sorts months apart can
            # be compared at the bit level rather than only by the
            # mutable image tag. ``ExecutionConfig
            # .docker_image_expected_digest`` is checked further down
            # for a warn-only mismatch finding.
            docker_image_tag: Optional[str] = None
            docker_image_digest: Optional[str] = None
            if config.sorter.use_docker:
                try:
                    from .docker_utils import (
                        get_docker_image,
                        get_local_image_digest,
                    )

                    docker_image_tag = get_docker_image(sorter)
                    docker_image_digest = get_local_image_digest(docker_image_tag)
                except Exception as exc:
                    print(f"[docker digest] resolution failed: {exc!r}")

            # Persist a JSON snapshot of the config so the
            # post-sorting Markdown report can list non-default
            # settings later without needing live config access.
            try:
                from .report import serialize_config_for_report

                snapshot = serialize_config_for_report(config)
                if docker_image_tag or docker_image_digest:
                    snapshot["_runtime"] = {
                        "docker_image": docker_image_tag,
                        "docker_image_digest": docker_image_digest,
                    }
                config_used_path = res_path_obj / "config_used.json"
                tmp = config_used_path.with_suffix(config_used_path.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2)
                os.replace(tmp, config_used_path)
            except Exception as exc:
                print(f"[config snapshot] failed: {exc!r}")

            sd_raw = None
            sd_curated = None
            result = None
            from .guards import set_active_log_path

            with Tee(log_path, file_mode="w"), set_active_log_path(log_path):
                _print_pipeline_banner(
                    sorter,
                    rec_path,
                    config,
                    log_path,
                    recording=rec_loaded,
                    docker_image_tag=docker_image_tag,
                    docker_image_digest=docker_image_digest,
                )

                # Warn (no fail) when the operator pinned an expected
                # digest and the live image doesn't match — the tag
                # was re-pushed at some point.
                expected_digest = getattr(exe_cfg, "docker_image_expected_digest", None)
                if (
                    config.sorter.use_docker
                    and expected_digest
                    and docker_image_digest
                    and expected_digest != docker_image_digest
                ):
                    print(
                        f"WARNING: Docker image digest mismatch — "
                        f"expected {expected_digest!r}, found "
                        f"{docker_image_digest!r}. The image tag has "
                        "been re-pushed to the registry since the "
                        "expected digest was recorded; runs are no "
                        "longer bit-identical."
                    )

                t_start = time.time()

                # Per-recording OOM-retry loop. Snapshot the
                # backend's memory-bound params first so a scale-down
                # applied during retries can be restored before the
                # next recording.
                from ._exceptions import (
                    DiskExhaustionError,
                    GpuMemoryWatchdogError,
                    GPUOutOfMemoryError,
                    HostMemoryWatchdogError,
                    SorterTimeoutError,
                )

                oom_max = max(0, int(exe_cfg.oom_retry_max))
                oom_factor = float(exe_cfg.oom_retry_factor)
                oom_snapshot = backend.snapshot_oom_params()
                attempt = 0

                # Per-recording disk-usage watchdog. Watches the
                # intermediate folder (where RT-Sort and KS2 write
                # the bulk of intermediates). Trip path: build a
                # DiskExhaustionReport, then call the in-process
                # kill callback. If interrupt_main is delivered
                # cleanly the surrounding ``except KeyboardInterrupt``
                # path converts the trip into a returnable exception;
                # otherwise the os._exit fallback fires.
                disk_wd = _make_disk_watchdog(
                    inter_path=inter_path,
                    config=config,
                    sorter=sorter,
                    rec_loaded=rec_loaded,
                )

                # Per-recording I/O stall watchdog (catches hung
                # network mounts or unresponsive storage that the
                # log-inactivity watchdog can miss when the sorter
                # keeps logging while waiting on kernel I/O).
                from contextlib import nullcontext

                from .guards import IOStallWatchdog

                if getattr(exe_cfg, "io_stall_watchdog", True):
                    io_stall_wd = IOStallWatchdog(
                        folder=Path(inter_path),
                        stall_s=exe_cfg.io_stall_s,
                        poll_interval_s=exe_cfg.io_stall_poll_interval_s,
                    )
                else:
                    io_stall_wd = nullcontext()

                # Enter the disk watchdog manually so we can keep the
                # surrounding try/finally structure intact and still
                # close the watchdog cleanly in the finally block.
                # The I/O stall watchdog runs alongside.
                disk_wd.__enter__()
                disk_wd_active = True
                io_stall_wd.__enter__()
                io_stall_wd_active = True
                try:
                    # Pipeline canary: short-window smoke test before the
                    # full sort. Catches MEX / preprocessing / Docker /
                    # model-loading failures in seconds rather than
                    # hours. Disabled by default; enabled by
                    # ExecutionConfig.canary_first_n_s > 0.
                    canary_window_s = float(
                        getattr(exe_cfg, "canary_first_n_s", 0.0) or 0.0
                    )
                    canary_result: Optional[BaseException] = None
                    if canary_window_s > 0:
                        rec_dur_s: Optional[float] = None
                        if rec_loaded is not None:
                            try:
                                n_smp = int(rec_loaded.get_num_samples())
                                fs_hz = float(rec_loaded.get_sampling_frequency())
                                if fs_hz > 0:
                                    rec_dur_s = n_smp / fs_hz
                            except Exception:
                                rec_dur_s = None
                        if rec_dur_s is not None and rec_dur_s < canary_window_s:
                            print(
                                f"[canary] skipping {rec_name}: recording "
                                f"({rec_dur_s:.1f} s) is shorter than the "
                                f"canary window ({canary_window_s:.1f} s)."
                            )
                        else:
                            from .canary import run_canary

                            canary_result = run_canary(
                                config,
                                rec_loaded,
                                rec_path,
                                inter_path,
                                sorter_name=sorter,
                                rec_name=rec_name,
                                rng=rng,
                            )

                    if canary_result is not None:
                        # Classified failure surfaced by the canary — skip
                        # the full sort and let the existing failure
                        # handling propagate the exception as the
                        # recording's result.
                        result = canary_result
                    else:
                        while True:
                            result = process_recording(
                                backend,
                                config,
                                rec_name,
                                rec_path,
                                inter_path,
                                res_path,
                                rec_loaded=rec_loaded,
                                rec_chunks=config.recording.rec_chunks or None,
                                rec_chunk_names=getattr(
                                    backend, "rec_chunk_names", None
                                ),
                                rng=rng,
                            )

                            if (
                                isinstance(result, GPUOutOfMemoryError)
                                and attempt < oom_max
                            ):
                                attempt += 1
                                scaled = backend.scale_oom_params(oom_factor)
                                if not scaled:
                                    print(
                                        "[oom retry] backend declined to scale "
                                        "memory-bound params; surrendering."
                                    )
                                    break
                                print(
                                    f"[oom retry] retrying recording "
                                    f"({attempt}/{oom_max}) after GPU OOM "
                                    f"with reduced batch."
                                )
                                # Reclaim GPU memory before the retry.
                                _free_gpu_and_python_memory()
                                continue
                            break

                    # If the disk watchdog tripped during process_recording
                    # (interrupt_main delivered cleanly into the backend
                    # which converted to a return value, or didn't fire
                    # at all), surface the trip as the recording's result.
                    if disk_wd.tripped() and not isinstance(
                        result, DiskExhaustionError
                    ):
                        result = disk_wd.make_error()

                    if isinstance(result, BaseException):
                        if isinstance(result, HostMemoryWatchdogError):
                            status = "ABORTED (host RAM watchdog)"
                        elif isinstance(result, GpuMemoryWatchdogError):
                            status = "ABORTED (GPU VRAM watchdog)"
                        elif isinstance(result, SorterTimeoutError):
                            status = "ABORTED (sorter inactivity timeout)"
                        elif isinstance(result, DiskExhaustionError):
                            status = "ABORTED (disk exhausted)"
                        elif isinstance(result, GPUOutOfMemoryError):
                            status = "OOM (GPU)"
                        elif isinstance(result, MemoryError):
                            status = "OOM (MemoryError)"
                        else:
                            status = "FAILED"
                        _print_pipeline_summary(
                            status, time.time() - t_start, error=result
                        )
                        continue

                    if config.compilation.save_raw_pkl:
                        sd_raw, sd_curated = result
                    else:
                        sd_curated = result

                    # Save pickle (atomic: write to .tmp + os.replace
                    # so an os._exit fired by the in-process inactivity
                    # watchdog mid-write cannot corrupt the result file).
                    res_path = Path(res_path)

                    if config.compilation.save_raw_pkl:
                        raw_pkl = res_path / "sorted_spikedata.pkl"
                        _atomic_write_pickle(sd_raw, raw_pkl)
                        print(f"Saved {sd_raw.N} raw units to {raw_pkl}")

                    curated_pkl = res_path / "sorted_spikedata_curated.pkl"
                    _atomic_write_pickle(sd_curated, curated_pkl)
                    print(f"Saved {sd_curated.N} curated units to {curated_pkl}")

                    # Epoch splitting
                    if sd_curated.metadata.get("rec_chunks_ms"):
                        epoch_sds = sd_curated.split_epochs()
                        spikedata_results.extend(epoch_sds)
                    else:
                        spikedata_results.append(sd_curated)

                    if config.execution.delete_inter:
                        import shutil as _shutil

                        _shutil.rmtree(inter_path)

                    _print_pipeline_summary("SUCCESS", time.time() - t_start)
                except KeyboardInterrupt:
                    # If the disk watchdog tripped (interrupt_main
                    # delivered, but no inner handler caught it),
                    # convert to a clean DiskExhaustionError result.
                    if disk_wd.tripped():
                        result = disk_wd.make_error()
                        _print_pipeline_summary(
                            "ABORTED (disk exhausted)",
                            time.time() - t_start,
                            error=result,
                        )
                    else:
                        raise
                finally:
                    # Close the per-recording watchdogs before
                    # reading their trip state for the report.
                    if disk_wd_active:
                        disk_wd.__exit__(None, None, None)
                        disk_wd_active = False
                    if io_stall_wd_active:
                        try:
                            io_stall_wd.__exit__(None, None, None)
                        except Exception:
                            pass
                        io_stall_wd_active = False
                    # If the disk watchdog tripped, persist the
                    # report JSON next to the recording report.
                    if disk_wd.tripped() and disk_wd.report() is not None:
                        _write_disk_exhaustion_report(disk_wd.report(), Path(res_path))
                    # Surface an I/O stall trip as the recording's
                    # result, mirroring the disk-watchdog logic.
                    if (
                        hasattr(io_stall_wd, "tripped")
                        and io_stall_wd.tripped()
                        and not isinstance(result, BaseException)
                    ):
                        result = io_stall_wd.make_error()
                    # Build the per-recording report entry from
                    # whatever ``result`` ended up being. Done before
                    # cleanup deletes the ``result`` reference.
                    rec_record = _make_recording_result(
                        rec_name=rec_name,
                        rec_path=rec_path,
                        results_folder=res_path,
                        result=result,
                        wall_time_s=time.time() - t_start,
                        retries_used=attempt,
                        log_path=log_path,
                    )
                    report.add(rec_record)
                    if out_report is not None:
                        out_report.add(rec_record)
                    _write_recording_report(rec_record, Path(res_path))

                    # Generate the human-readable Markdown sorting
                    # report (Stream 2). Best-effort: failures here
                    # don't block batch progress, but they DO gate
                    # the tee_log_policy — only a successful report
                    # write triggers Tee log delete/gzip below.
                    if getattr(exe_cfg, "generate_sorting_report", True):
                        from .report import (
                            apply_tee_log_policy,
                            generate_sorting_report as _gen_report,
                        )

                        report_path = None
                        try:
                            report_path = _gen_report(
                                results_folder=Path(res_path),
                                log_path=log_path,
                            )
                        except Exception as exc:
                            print(f"[sorting report] generation raised: {exc!r}")

                        # Apply tee_log_policy ONLY when the report
                        # write succeeded AND the recording was a
                        # success — failures always preserve the
                        # log so the traceback survives.
                        if report_path is not None and rec_record.status == "success":
                            apply_tee_log_policy(
                                log_path,
                                getattr(
                                    exe_cfg,
                                    "tee_log_policy",
                                    "delete_on_success",
                                ),
                            )

                    # Always-run cleanup: drop large local references
                    # (the disk pickles and the entries in
                    # ``spikedata_results`` retain the data; this only
                    # frees in-loop aliases) and reclaim GPU + Python
                    # memory before the next iteration. Also restore
                    # the backend's OOM-bound config so a retry-side
                    # scale-down does not persist into the next
                    # recording.
                    del sd_raw, sd_curated, result
                    _free_gpu_and_python_memory()
                    backend.restore_oom_params(oom_snapshot)

    if report.records:
        _print_batch_summary(report)

    return spikedata_results


def _free_gpu_and_python_memory() -> None:
    """Force a Python GC pass and empty the PyTorch CUDA cache.

    Called between recordings (and between OOM-retry attempts) so a
    leak — or simply a long-running batch's accumulated allocator
    fragmentation — does not inflate the baseline for the next
    iteration. Safe to call when ``torch`` is not installed: the
    CUDA-cache step is skipped silently.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass
class RecordingResult:
    """Outcome of sorting a single recording within a batch.

    Parameters:
        rec_name (str): Short recording identifier (the file's basename).
        rec_path (str): Original recording path as a string.
        results_folder (str): Per-recording results folder.
        status (str): One of ``"success"``, ``"failed"``,
            ``"oom_gpu"``, ``"oom_host_ram"``, ``"oom_memoryerror"``,
            ``"sorter_timeout"``, ``"disk_exhausted"``.
        wall_time_s (float): Wall-clock time spent on this recording
            (including OOM retries).
        n_curated_units (int or None): Number of curated units when
            successful, otherwise ``None``.
        error_class (str or None): ``type(exc).__name__`` on failure,
            otherwise ``None``.
        error_message (str or None): ``str(exc)`` on failure (first
            500 chars), otherwise ``None``.
        retries_used (int): OOM-retry attempts consumed.
        log_path (str or None): Path to the per-recording Tee log
            file (``sorting_<timestamp>.log``). Populated by
            ``sort_recording`` so the batch summary can point users
            at the log for failure diagnosis.
    """

    rec_name: str
    rec_path: str
    results_folder: str
    status: str
    wall_time_s: float
    n_curated_units: Optional[int] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    retries_used: int = 0
    log_path: Optional[str] = None
    peak_host_ram_pct: Optional[float] = None
    peak_gpu_used_pct: Optional[float] = None
    min_disk_free_gb: Optional[float] = None


@dataclass
class SortRunReport:
    """Per-batch summary of a :func:`sort_recording` invocation.

    Records a :class:`RecordingResult` for each input recording — both
    successes and failures — so callers can inspect the outcome
    programmatically without parsing the per-recording log files.

    The report is also serialised to disk:

    * Per-recording: ``<results_folder>/recording_report.json``
      (always written).
    * Per-batch: optional, see ``out_report`` parameter on
      :func:`sort_recording`.

    Attributes:
        records (list[RecordingResult]): Per-recording outcomes in
            the order they were processed. Use the convenience
            properties for filtered views.
    """

    records: List[RecordingResult] = field(default_factory=list)

    def add(self, record: RecordingResult) -> None:
        """Append a per-recording result.

        Parameters:
            record (RecordingResult): Outcome of one recording.
        """
        self.records.append(record)

    @property
    def succeeded(self) -> List[RecordingResult]:
        """All successful recordings, in run order."""
        return [r for r in self.records if r.status == "success"]

    @property
    def failed(self) -> List[RecordingResult]:
        """All non-successful recordings, in run order."""
        return [r for r in self.records if r.status != "success"]

    @property
    def all_succeeded(self) -> bool:
        """True if every recording in the batch succeeded."""
        return bool(self.records) and all(r.status == "success" for r in self.records)

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict representation."""
        return {
            "records": [r.__dict__.copy() for r in self.records],
            "n_total": len(self.records),
            "n_succeeded": len(self.succeeded),
            "n_failed": len(self.failed),
        }


def _classify_recording_status(result: Any) -> str:
    """Map an exception (or None) to a :class:`RecordingResult` status."""
    from ._exceptions import (
        ConcurrentSortError,
        DiskExhaustionError,
        GpuMemoryWatchdogError,
        GPUOutOfMemoryError,
        HostMemoryWatchdogError,
        IOStallError,
        SorterTimeoutError,
    )

    if not isinstance(result, BaseException):
        return "success"
    if isinstance(result, HostMemoryWatchdogError):
        return "oom_host_ram"
    if isinstance(result, GpuMemoryWatchdogError):
        return "oom_gpu"
    if isinstance(result, SorterTimeoutError):
        return "sorter_timeout"
    if isinstance(result, DiskExhaustionError):
        return "disk_exhausted"
    if isinstance(result, IOStallError):
        return "io_stall"
    if isinstance(result, ConcurrentSortError):
        return "concurrent_sort"
    if isinstance(result, GPUOutOfMemoryError):
        return "oom_gpu"
    if isinstance(result, MemoryError):
        return "oom_memoryerror"
    return "failed"


def _read_peaks_from_audit(folder: Path) -> Dict[str, Optional[float]]:
    """Compute per-recording peak resource values from the audit log.

    Reads ``<folder>/watchdog_events.jsonl`` (written by watchdog
    warn / abort events) and returns the worst observed value for
    each tracked metric. Returns ``None`` for metrics that never
    produced an event (i.e. the corresponding watchdog never
    crossed its warn threshold).
    """
    audit = Path(folder) / "watchdog_events.jsonl"
    peaks: Dict[str, Optional[float]] = {
        "peak_host_ram_pct": None,
        "peak_gpu_used_pct": None,
        "min_disk_free_gb": None,
    }
    if not audit.is_file():
        return peaks
    try:
        text = audit.read_text(encoding="utf-8")
    except OSError:
        return peaks
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        wd = entry.get("watchdog")
        if wd == "host_memory" and "used_pct" in entry:
            try:
                pct = float(entry["used_pct"])
            except (TypeError, ValueError):
                continue
            if peaks["peak_host_ram_pct"] is None or pct > peaks["peak_host_ram_pct"]:
                peaks["peak_host_ram_pct"] = pct
        elif wd == "gpu_memory" and "used_pct" in entry:
            try:
                pct = float(entry["used_pct"])
            except (TypeError, ValueError):
                continue
            if peaks["peak_gpu_used_pct"] is None or pct > peaks["peak_gpu_used_pct"]:
                peaks["peak_gpu_used_pct"] = pct
        elif wd == "disk" and "free_gb" in entry:
            try:
                gb = float(entry["free_gb"])
            except (TypeError, ValueError):
                continue
            if peaks["min_disk_free_gb"] is None or gb < peaks["min_disk_free_gb"]:
                peaks["min_disk_free_gb"] = gb
    return peaks


def _make_recording_result(
    *,
    rec_name: str,
    rec_path: Any,
    results_folder: Any,
    result: Any,
    wall_time_s: float,
    retries_used: int,
    log_path: Any = None,
) -> "RecordingResult":
    """Construct a :class:`RecordingResult` from a per-recording outcome.

    The ``result`` argument is whatever ``process_recording`` returned
    — either a ``SpikeData``, a ``(sd_raw, sd_curated)`` tuple when
    ``save_raw_pkl=True``, or a ``BaseException`` sentinel.
    """
    status = _classify_recording_status(result)
    n_units: Optional[int] = None
    err_class: Optional[str] = None
    err_msg: Optional[str] = None

    if isinstance(result, BaseException):
        err_class = type(result).__name__
        err_msg = str(result)[:500] if str(result) else None
    else:
        # Success — extract the curated unit count.
        sd_curated = result[1] if isinstance(result, tuple) else result
        try:
            n_units = int(sd_curated.N)
        except Exception:
            n_units = None

    peaks = _read_peaks_from_audit(Path(results_folder))
    return RecordingResult(
        rec_name=rec_name,
        rec_path=str(rec_path),
        results_folder=str(results_folder),
        status=status,
        wall_time_s=float(wall_time_s),
        n_curated_units=n_units,
        error_class=err_class,
        error_message=err_msg,
        retries_used=int(retries_used),
        log_path=str(log_path) if log_path is not None else None,
        peak_host_ram_pct=peaks.get("peak_host_ram_pct"),
        peak_gpu_used_pct=peaks.get("peak_gpu_used_pct"),
        min_disk_free_gb=peaks.get("min_disk_free_gb"),
    )


def _write_recording_report(record: "RecordingResult", results_folder: Path) -> None:
    """Write a per-recording report JSON to the results folder.

    Best-effort: failures here are logged but do not propagate so a
    bad disk does not break the surrounding batch.
    """
    target = Path(results_folder) / "recording_report.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so an os._exit during report serialisation
        # cannot corrupt the file the next batch may try to read.
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record.__dict__, f, indent=2, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp, target)
    except Exception as exc:
        print(f"[recording report] failed to write {target}: {exc!r}")


def _print_batch_summary(report: "SortRunReport") -> None:
    """Print a final summary table for a sort_recording batch.

    Surfaces enough detail per failed recording (status, error
    class+message, wall time, retry count, log file path, links to
    JSON reports) that the operator can diagnose without grepping
    run output. The log path is the most load-bearing piece — it's
    the only artefact that captures the full traceback for failures
    inside C extensions.
    """
    print_stage("BATCH SUMMARY")
    n = len(report.records)
    print(f"Total recordings:  {n}")
    print(f"Succeeded:         {len(report.succeeded)}")
    failed = report.failed
    print(f"Failed:            {len(failed)}")

    if report.succeeded:
        print()
        print("Successful recordings:")
        for rec in report.succeeded:
            units = rec.n_curated_units if rec.n_curated_units is not None else "?"
            print(
                f"  - {rec.rec_name}: {units} curated units, "
                f"{rec.wall_time_s:.1f}s wall time"
            )

    # Cross-recording resource trending — surfaces memory creep
    # and disk consumption across the batch when watchdog warn
    # thresholds were crossed.
    if any(
        rec.peak_host_ram_pct is not None
        or rec.peak_gpu_used_pct is not None
        or rec.min_disk_free_gb is not None
        for rec in report.records
    ):
        print()
        print("Resource trends (only recordings that hit a warn threshold):")
        print("| Recording | Peak host RAM % | Peak GPU % | Min disk free GB |")
        print("|---|---|---|---|")
        for rec in report.records:
            if (
                rec.peak_host_ram_pct is None
                and rec.peak_gpu_used_pct is None
                and rec.min_disk_free_gb is None
            ):
                continue
            ram = (
                f"{rec.peak_host_ram_pct:.1f}"
                if rec.peak_host_ram_pct is not None
                else "—"
            )
            gpu = (
                f"{rec.peak_gpu_used_pct:.1f}"
                if rec.peak_gpu_used_pct is not None
                else "—"
            )
            disk = (
                f"{rec.min_disk_free_gb:.2f}"
                if rec.min_disk_free_gb is not None
                else "—"
            )
            print(f"| {rec.rec_name} | {ram} | {gpu} | {disk} |")

    if failed:
        print()
        print("Failures:")
        for rec in failed:
            err_kind = rec.error_class or "?"
            err_msg = (
                (rec.error_message or "").splitlines()[0] if rec.error_message else ""
            )
            retries = (
                f", {rec.retries_used} retry/retries used" if rec.retries_used else ""
            )
            print(f"  - {rec.rec_name}  [{rec.status}]  {err_kind}: {err_msg}")
            print(f"      wall time: {rec.wall_time_s:.1f}s{retries}")
            if rec.log_path:
                print(f"      log: {rec.log_path}")
            results_folder = Path(rec.results_folder) if rec.results_folder else None
            if results_folder is not None:
                rec_report = results_folder / "recording_report.json"
                if rec_report.exists():
                    print(f"      report: {rec_report}")
                disk_report = results_folder / "disk_exhaustion_report.json"
                if disk_report.exists():
                    print(f"      disk report: {disk_report}")
                gpu_snap = results_folder / "gpu_snapshot_at_trip.txt"
                if gpu_snap.exists():
                    print(f"      gpu snapshot: {gpu_snap}")


def _make_disk_watchdog(
    *,
    inter_path: Any,
    config: Any,
    sorter: str,
    rec_loaded: Any = None,
):
    """Build a per-recording :class:`DiskUsageWatchdog`.

    Returns a configured (but not-yet-entered) watchdog that watches
    the recording's intermediate folder. Kill path is the in-process
    interrupt-then-os._exit callback so the trip works for both
    in-process backends (KS4 host, RT-Sort) and subprocess backends
    (KS2). For RT-Sort with a pre-loaded recording, the projected
    on-disk need is included in the report for the operator.

    Parameters:
        inter_path: Per-recording intermediate folder.
        config: Pipeline configuration (reads ``execution.disk_*``).
        sorter (str): Short sorter identifier for diagnostics.
        rec_loaded: Optional pre-loaded recording, used to project
            RT-Sort's intermediate disk footprint.

    Returns:
        watchdog (DiskUsageWatchdog): Disabled when
            ``execution.disk_watchdog`` is False.
    """
    from .guards import (
        DiskUsageWatchdog,
        estimate_rt_sort_intermediate_gb,
        make_in_process_kill_callback,
    )

    exe = config.execution
    if not getattr(exe, "disk_watchdog", True):
        return DiskUsageWatchdog(
            folder=Path(inter_path),
            warn_free_gb=exe.disk_warn_free_gb,
            abort_free_gb=exe.disk_abort_free_gb,
            poll_interval_s=exe.disk_poll_interval_s,
            sorter=sorter,
            popen=None,
            kill_callback=None,
        )

    projected_gb = None
    if sorter.lower() == "rt_sort" and rec_loaded is not None:
        try:
            n_ch = int(rec_loaded.get_num_channels())
            n_smp = int(rec_loaded.get_num_samples())
            projected_gb = estimate_rt_sort_intermediate_gb(
                n_channels=n_ch, n_samples=n_smp
            )
        except Exception:
            projected_gb = None

    grace_s = float(getattr(exe, "sorter_inactivity_in_process_grace_s", 10.0))
    return DiskUsageWatchdog(
        folder=Path(inter_path),
        warn_free_gb=exe.disk_warn_free_gb,
        abort_free_gb=exe.disk_abort_free_gb,
        poll_interval_s=exe.disk_poll_interval_s,
        sorter=sorter,
        projected_need_gb=projected_gb,
        kill_callback=make_in_process_kill_callback(
            interrupt_grace_s=grace_s,
            sorter=f"disk-watchdog/{sorter}",
        ),
    )


def _write_disk_exhaustion_report(report: Any, results_folder: Path) -> None:
    """Write a :class:`DiskExhaustionReport` next to the recording report.

    Best-effort: failures here are logged but do not propagate so a
    bad disk does not break the surrounding batch.
    """
    target = Path(results_folder) / "disk_exhaustion_report.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp, target)
        print(f"[disk watchdog] wrote disk-exhaustion report: {target}")
    except Exception as exc:
        print(f"[disk exhaustion report] failed to write {target}: {exc!r}")


def _atomic_write_pickle(
    obj: Any, path: Any, *, protocol: Optional[int] = None
) -> None:
    """Pickle ``obj`` to ``path`` atomically.

    Writes to ``<path>.tmp``, ``flush()`` + ``os.fsync()``, then
    :func:`os.replace` to the final path. ``os.replace`` is atomic
    on both POSIX and Windows: a reader will either see the previous
    contents or the new contents, never a half-written file.

    Important when the in-process inactivity watchdog is active —
    that watchdog can call ``os._exit`` mid-write, and a non-atomic
    write would leave the result file unreadable.

    Parameters:
        obj: Any picklable object.
        path: Destination path (will be coerced to ``Path``).
        protocol (int or None): Optional pickle protocol. ``None``
            uses the default for the running interpreter.
    """
    import pickle as _pkl

    final = Path(path)
    tmp = final.with_suffix(final.suffix + ".tmp")
    final.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp, "wb") as f:
        if protocol is None:
            _pkl.dump(obj, f)
        else:
            _pkl.dump(obj, f, protocol=protocol)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            # fsync can fail on certain Windows file systems and
            # raises AttributeError on some non-OS file objects
            # (e.g. test-time wrappers). The replace below is still
            # atomic; we just skip the durability hint.
            pass

    os.replace(tmp, final)


def sort_multistream(recording, stream_ids, config=None, sorter="kilosort2", **kwargs):
    """Sort a multi-stream recording across multiple stream IDs.

    Calls ``sort_recording`` once per stream ID, routing each stream
    to its own intermediate and results folders. Validates that the
    requested stream IDs exist in the recording file before sorting.

    Parameters:
        recording (str or Path): Path to a single multi-stream
            recording file (e.g. MaxTwo ``.raw.h5``) or a directory of
            such files.  When a directory is given, all files are
            concatenated per stream.
        stream_ids (list of str): Stream identifiers to sort, e.g.
            ``["well000", "well001", "well002"]``.
        config (SortingPipelineConfig or None): Pre-built configuration.
            When provided, ``**kwargs`` are applied as overrides.
        sorter (str): Registered sorter backend name (default
            ``"kilosort2"``).  Only used when ``config`` is None.
        **kwargs: Override individual config fields.  The following
            must not be provided:

            - ``intermediate_folders`` and ``results_folders`` are
              auto-generated per stream.
            - ``stream_id`` is set automatically per iteration.

    Returns:
        results (dict): ``{stream_id: list[SpikeData]}``.

    Notes:
        - Stream ID validation uses SpikeInterface's extractor for the
          recording format.  Currently supports Maxwell ``.h5`` files.
          For other formats, validation is skipped and invalid stream
          IDs will produce errors at loading time.
        - When *recording* is a directory of files, each file is
          concatenated per stream before sorting.  Channel count and
          sampling frequency must match across files (raises
          ``ValueError``); mismatched channel IDs or locations produce
          warnings.
    """
    import datetime

    if "stream_id" in kwargs:
        raise ValueError(
            "Do not pass 'stream_id' to sort_multistream — it is set "
            "automatically for each stream. Pass stream IDs via the "
            "'stream_ids' parameter instead."
        )
    if kwargs.get("intermediate_folders") is not None:
        raise ValueError(
            "'intermediate_folders' cannot be specified for "
            "sort_multistream — folders are auto-generated per stream."
        )
    if kwargs.get("results_folders") is not None:
        raise ValueError(
            "'results_folders' cannot be specified for "
            "sort_multistream — folders are auto-generated per stream."
        )

    recording = Path(recording)

    # Validate stream IDs against the recording file
    h5_files = []
    if recording.is_dir():
        try:
            from natsort import natsorted
        except ImportError:
            natsorted = sorted
        h5_files = [
            recording / name
            for name in natsorted(
                p.name for p in recording.iterdir() if p.name.endswith(".raw.h5")
            )
        ]
    elif str(recording).endswith(".h5"):
        h5_files = [recording]

    if h5_files:
        try:
            from spikeinterface.extractors import MaxwellRecordingExtractor

            _, available_ids = MaxwellRecordingExtractor.get_streams(str(h5_files[0]))
            missing = [sid for sid in stream_ids if sid not in available_ids]
            if missing:
                raise ValueError(
                    f"Stream ID(s) {missing} not found in "
                    f"{h5_files[0].name}. Available streams: {available_ids}"
                )
        except ImportError:
            pass  # SI not available — skip validation

    results = {}
    for sid in stream_ids:
        print_stage(f"SORTING STREAM: {sid}")

        if recording.is_dir():
            base = recording
        else:
            base = recording.parent

        cur_dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        inter = [str(base / f"inter_{sorter}_{sid}_{cur_dt}")]
        res = [str(base / f"sorted_{sorter}_{sid}")]

        stream_results = sort_recording(
            recording_files=[str(recording)],
            config=config,
            sorter=sorter,
            intermediate_folders=inter,
            results_folders=res,
            stream_id=sid,
            **kwargs,
        )
        results[sid] = stream_results

    return results
