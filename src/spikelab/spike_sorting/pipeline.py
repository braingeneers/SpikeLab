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


def sort_recording(
    recording_files,
    config=None,
    sorter="kilosort2",
    intermediate_folders=None,
    results_folders=None,
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
    if intermediate_folders is None:
        cur_dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        intermediate_folders = [
            Path(rec).parent / f"inter_{sorter}_{cur_dt}" for rec in recording_files
        ]
    if results_folders is None:
        results_folders = [
            Path(rec).parent / f"sorted_{sorter}" for rec in recording_files
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

    np.random.seed(1)

    # Main loop
    spikedata_results = []
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

        result = process_recording(
            backend,
            config,
            rec_name,
            rec_path,
            inter_path,
            res_path,
            rec_loaded=rec_loaded,
            rec_chunks=config.recording.rec_chunks or None,
            rec_chunk_names=getattr(backend, "rec_chunk_names", None),
        )

        if isinstance(result, BaseException):
            continue

        if config.compilation.save_raw_pkl:
            sd_raw, sd_curated = result
        else:
            sd_curated = result

        # Save pickle
        import pickle as _pkl

        res_path = Path(res_path)

        if config.compilation.save_raw_pkl:
            raw_pkl = res_path / "sorted_spikedata.pkl"
            with open(raw_pkl, "wb") as f:
                _pkl.dump(sd_raw, f)
            print(f"Saved {sd_raw.N} raw units to {raw_pkl}")

        curated_pkl = res_path / "sorted_spikedata_curated.pkl"
        with open(curated_pkl, "wb") as f:
            _pkl.dump(sd_curated, f)
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

    return spikedata_results


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
