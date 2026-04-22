"""Recording I/O, legacy waveform extraction wrapper, and legacy pipeline functions (process_recording, compile_results, Compiler). Used by the legacy sort_with_kilosort2 entry point — new code should use pipeline.py."""

import datetime
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    natsorted = None

try:
    from scipy.io import savemat
except ImportError:  # pragma: no cover
    savemat = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import spikeinterface.core.segmentutils as si_segmentutils
    from spikeinterface.core import BaseRecording
    from spikeinterface.extractors.extractor_classes import (
        MaxwellRecordingExtractor,
        NwbRecordingExtractor,
    )
    from spikeinterface.preprocessing import bandpass_filter
    from spikeinterface.preprocessing.preprocessing_classes import ScaleRecording

    _SI_AVAILABLE = True
except ImportError:  # pragma: no cover
    si_segmentutils = None
    BaseRecording = None
    MaxwellRecordingExtractor = None
    NwbRecordingExtractor = None
    bandpass_filter = None
    ScaleRecording = None
    _SI_AVAILABLE = False

from . import _globals
from .ks2_runner import spike_sort
from .sorting_extractor import KilosortSortingExtractor
from .sorting_utils import (
    Stopwatch,
    Tee,
    create_folder,
    delete_folder,
    get_paths,
    print_stage,
)
from .waveform_extractor import WaveformExtractor


# Upstream `neo.rawio.maxwellrawio.auto_install_maxwell_hdf5_compression_plugin`
# treats `HDF5_PLUGIN_PATH` as a single directory. HDF5 actually defines it as
# an os.pathsep-separated list (like `PATH`), so when the env var holds multiple
# entries (e.g. `/home/mxwbio/MaxLab/so/:/home/sharf-lab/MaxLab/so`) upstream
# tries to `Path(...).mkdir()` on the compound string and fails. This wrapper
# patches the helper at SpikeLab import time so the fix survives any `neo`
# reinstall/upgrade.
def _patch_neo_maxwell_hdf5_plugin_path_handling() -> None:
    try:
        import platform
        from pathlib import Path
        from urllib.request import urlopen

        import neo.rawio.maxwellrawio as _mwrawio
    except ImportError:
        return

    def auto_install_maxwell_hdf5_compression_plugin(
        hdf5_plugin_path=None, force_download=True
    ):
        if hdf5_plugin_path is None:
            env_value = os.getenv("HDF5_PLUGIN_PATH", None)
            if env_value is not None:
                # HDF5_PLUGIN_PATH follows PATH-style semantics: a list of
                # directories separated by os.pathsep (':' on Linux/macOS,
                # ';' on Windows). Scan each component for an existing
                # libcompression library before downloading.
                for component in env_value.split(os.pathsep):
                    component = component.strip()
                    if not component:
                        continue
                    candidate_dir = Path(component)
                    if platform.system() == "Linux":
                        candidate = candidate_dir / "libcompression.so"
                    elif platform.system() == "Darwin":
                        candidate = candidate_dir / "libcompression.dylib"
                    elif platform.system() == "Windows":
                        candidate = candidate_dir / "compression.dll"
                    else:
                        candidate = None
                    if candidate is not None and candidate.is_file():
                        hdf5_plugin_path = candidate_dir
                        break
                if hdf5_plugin_path is None:
                    # No existing plugin found in any component; fall back to
                    # the first non-empty component as the install target.
                    for component in env_value.split(os.pathsep):
                        component = component.strip()
                        if component:
                            hdf5_plugin_path = Path(component)
                            break
            if hdf5_plugin_path is None:
                hdf5_plugin_path = Path.home() / "hdf5_plugin_path_maxwell"
                os.environ["HDF5_PLUGIN_PATH"] = str(hdf5_plugin_path)
        hdf5_plugin_path = Path(hdf5_plugin_path)
        hdf5_plugin_path.mkdir(exist_ok=True)

        if platform.system() == "Linux":
            remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FLinux%2Flibcompression.so&dl=1"
            local_lib = hdf5_plugin_path / "libcompression.so"
        elif platform.system() == "Darwin":
            if platform.machine() == "arm64":
                remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FMacOS%2FMac_arm64%2Flibcompression.dylib&dl=1"
            else:
                remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FMacOS%2FMac_x86_64%2Flibcompression.dylib&dl=1"
            local_lib = hdf5_plugin_path / "libcompression.dylib"
        elif platform.system() == "Windows":
            remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FWindows%2Fcompression.dll&dl=1"
            local_lib = hdf5_plugin_path / "compression.dll"

        if not force_download and local_lib.is_file():
            print(
                f"The h5 compression library for Maxwell is already located in {local_lib}!"
            )
            return

        dist = urlopen(remote_lib)
        with open(local_lib, "wb") as f:
            f.write(dist.read())

    setattr(
        _mwrawio,
        "auto_install_maxwell_hdf5_compression_plugin",
        auto_install_maxwell_hdf5_compression_plugin,
    )


_patch_neo_maxwell_hdf5_plugin_path_handling()


class Compiler:
    """Aggregates sorting results from one or more SpikeData objects for export.

    Reads unit metadata from ``neuron_attributes`` and writes combined
    ``.npz``, ``.mat``, and figure outputs.
    """

    def __init__(self):
        self.create_figures = _globals.CREATE_FIGURES
        self.create_std_scatter_plot = (
            _globals.CURATE_SECOND
            and _globals.SPIKES_MIN_SECOND is not None
            and _globals.STD_NORM_MAX is not None
        )
        self.compile_to_mat = _globals.COMPILE_TO_MAT
        self.compile_to_npz = _globals.COMPILE_TO_NPZ
        self.save_electrodes = _globals.SAVE_ELECTRODES

        self.recs_cache = []

    def add_recording(
        self, rec_name: str, sd: Any, curation_history: Optional[dict] = None
    ) -> None:
        """Queue a recording for compilation.

        Parameters:
            rec_name (str): Short name for the recording.
            sd (SpikeData): Curated SpikeData with enriched
                ``neuron_attributes``.
            curation_history (dict or None): Curation history dict
                from ``build_curation_history``.
        """
        self.recs_cache.append((rec_name, sd, curation_history))

    def save_results(self, folder: Union[str, Path]) -> None:
        """Compile and save results from all queued recordings.

        Parameters:
            folder (Path or str): Output directory.
        """
        create_folder(folder)
        folder = Path(folder)

        # ------------------------------------------------------------------
        # Collect all units from all recordings
        # ------------------------------------------------------------------
        all_units = []  # list of (attrs_dict, is_curated, rec_name)
        rec_metadata = {}  # rec_name -> {fs, locations, n_samples}

        # Figure data
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

            # All units are curated (sd is already curated)
            curated_ids = set()
            if sd.neuron_attributes is not None:
                for attrs in sd.neuron_attributes:
                    curated_ids.add(int(attrs.get("unit_id", -1)))

            for i in range(sd.N):
                attrs = sd.neuron_attributes[i] if sd.neuron_attributes else {}
                all_units.append((attrs, True, rec_name))

            # Figure data
            if self.create_figures:
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

        # ------------------------------------------------------------------
        # Sort units by amplitude within polarity groups
        # ------------------------------------------------------------------
        neg_units = [
            (a, c, r) for a, c, r in all_units if not a.get("has_pos_peak", False)
        ]
        pos_units = [(a, c, r) for a, c, r in all_units if a.get("has_pos_peak", False)]

        # Sort by amplitude descending
        neg_units.sort(key=lambda x: float(x[0].get("amplitude", 0)), reverse=True)
        pos_units.sort(key=lambda x: float(x[0].get("amplitude", 0)), reverse=True)

        # ------------------------------------------------------------------
        # Build compile_dict and save waveforms/figures
        # ------------------------------------------------------------------
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

        if _globals.COMPILE_WAVEFORMS:
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
            for attrs, is_curated, rec_name in tqdm(units_group):
                if is_curated:
                    if compile_dict is not None:
                        spike_train_samples = attrs.get("spike_train_samples")
                        if _globals.SAVE_DL_DATA:
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

                    if _globals.COMPILE_WAVEFORMS:
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
            if self.compile_to_mat:
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
                total_label=_globals.BAR_TOTAL_LABEL,
                selected_label=_globals.BAR_SELECTED_LABEL,
                x_label=_globals.BAR_X_LABEL,
                y_label=_globals.BAR_Y_LABEL,
                label_rotation=_globals.BAR_LABEL_ROTATION,
                save_path=str(figures_path / "curation_bar_plot.png"),
            )
            print("Curation bar plot has been saved")

            if self.create_std_scatter_plot and scatter_n_spikes:
                plot_std_scatter(
                    scatter_n_spikes,
                    scatter_std_norms,
                    spikes_thresh=_globals.SPIKES_MIN_SECOND,
                    std_thresh=_globals.STD_NORM_MAX,
                    colors=_globals.SCATTER_RECORDING_COLORS[:],
                    alpha=_globals.SCATTER_RECORDING_ALPHA,
                    x_label=_globals.SCATTER_X_LABEL,
                    y_label=_globals.SCATTER_Y_LABEL,
                    x_max_buffer=_globals.SCATTER_X_MAX_BUFFER,
                    y_max_buffer=_globals.SCATTER_Y_MAX_BUFFER,
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
                    templates_per_column=_globals.ALL_TEMPLATES_PER_COLUMN,
                    y_spacing=_globals.ALL_TEMPLATES_Y_SPACING,
                    y_lim_buffer=_globals.ALL_TEMPLATES_Y_LIM_BUFFER,
                    color_curated=_globals.ALL_TEMPLATES_COLOR_CURATED,
                    color_failed=_globals.ALL_TEMPLATES_COLOR_FAILED,
                    window_ms_before=_globals.ALL_TEMPLATES_WINDOW_MS_BEFORE_PEAK,
                    window_ms_after=_globals.ALL_TEMPLATES_WINDOW_MS_AFTER_PEAK,
                    line_ms_before=_globals.ALL_TEMPLATES_LINE_MS_BEFORE_PEAK,
                    line_ms_after=_globals.ALL_TEMPLATES_LINE_MS_AFTER_PEAK,
                    x_label=_globals.ALL_TEMPLATES_X_LABEL,
                    save_path=str(figures_path / "all_templates_plot.png"),
                )
                print("All templates plot has been saved")


# create_folder and delete_folder are imported from sorting_utils.


def _time_chunks_to_frames(
    start_time_s: Optional[float],
    end_time_s: Optional[float],
    rec_chunks_s: List[Tuple[float, float]],
    fs: float,
    total_duration_s: float,
) -> List[Tuple[int, int]]:
    """Convert time-based slicing parameters to frame tuples.

    Combines ``start_time_s``/``end_time_s`` (single range) and
    ``rec_chunks_s`` (multiple ranges) into a single list of
    ``(start_frame, end_frame)`` tuples in samples.

    Parameters:
        start_time_s: Start time in seconds, or ``None``.
        end_time_s: End time in seconds, or ``None``.
        rec_chunks_s: List of ``(start_s, end_s)`` ranges in seconds.
        fs: Sampling frequency in Hz.
        total_duration_s: Full recording duration in seconds (used to
            clip ``end_time_s`` if it exceeds the recording).

    Returns:
        List of ``(start_frame, end_frame)`` tuples. Empty list when
        no time-based parameters are provided.

    Raises:
        ValueError: If a time range is invalid (negative start or
            start >= end).
    """
    chunks: List[Tuple[int, int]] = []

    if start_time_s is not None or end_time_s is not None:
        start_s = start_time_s if start_time_s is not None else 0.0
        end_s = end_time_s if end_time_s is not None else total_duration_s
        if end_s > total_duration_s:
            print(
                f"'end_time_s' ({end_s}) exceeds recording duration "
                f"({total_duration_s:.2f}s); clipping to the end."
            )
            end_s = total_duration_s
        if start_s < 0 or start_s >= end_s:
            raise ValueError(
                f"Invalid time range: start_time_s={start_s}, "
                f"end_time_s={end_s}. Must satisfy 0 <= start < end."
            )
        chunks.append((int(round(start_s * fs)), int(round(end_s * fs))))

    for start_s, end_s in rec_chunks_s:
        if start_s < 0 or start_s >= end_s:
            raise ValueError(
                f"Invalid chunk in rec_chunks_s: ({start_s}, {end_s}). "
                f"Must satisfy 0 <= start < end."
            )
        if end_s > total_duration_s:
            print(
                f"'rec_chunks_s' entry ({start_s}, {end_s}) exceeds "
                f"recording duration ({total_duration_s:.2f}s); clipping."
            )
            end_s = total_duration_s
        chunks.append((int(round(start_s * fs)), int(round(end_s * fs))))

    return chunks


def load_recording(rec_path: Any) -> BaseRecording:
    """Load a recording, apply optional truncation and coordinate transforms.

    Loads a single recording file via ``load_single_recording``, or all
    recordings in a directory via ``concatenate_recordings``. Then applies
    the module-level configuration: truncation to ``FIRST_N_MINS``, frame
    chunking via ``REC_CHUNKS``, y-coordinate flipping via ``MEA_Y_MAX``,
    and custom gain/offset scaling.

    Parameters:
        rec_path (str, Path, or BaseRecording): Path to a recording file,
            a directory containing ``.raw.h5`` / ``.nwb`` files to
            concatenate, or a pre-loaded ``BaseRecording``.

    Returns:
        rec (BaseRecording): The loaded and optionally transformed
            SpikeInterface recording object.
    """
    print_stage("LOADING RECORDING")
    print(f"Recording path: {rec_path}")
    stopwatch = Stopwatch()
    rec_path = Path(rec_path)
    if rec_path.is_dir():
        rec = concatenate_recordings(rec_path)
    else:
        rec = load_single_recording(rec_path)

    print(f"Recording has {rec.get_num_channels()} channels")

    # Convert time-based slicing parameters (seconds) to frame tuples.
    time_chunks = _time_chunks_to_frames(
        start_time_s=_globals.START_TIME_S,
        end_time_s=_globals.END_TIME_S,
        rec_chunks_s=_globals.REC_CHUNKS_S,
        fs=rec.get_sampling_frequency(),
        total_duration_s=rec.get_total_duration(),
    )
    if time_chunks:
        if len(_globals.REC_CHUNKS) > 0:
            raise ValueError(
                "Cannot combine frame-based 'rec_chunks' with time-based "
                "'start_time_s'/'end_time_s'/'rec_chunks_s'. Use one or the "
                "other."
            )
        _globals.REC_CHUNKS = time_chunks

    if _globals.FIRST_N_MINS is not None:
        end_frame = _globals.FIRST_N_MINS * 60 * rec.get_sampling_frequency()
        if end_frame > rec.get_num_samples():
            print(
                f"'first_n_mins' is set to {_globals.FIRST_N_MINS}, but recording is only {rec.get_total_duration() / 60:.2f} min long"
            )
            print(
                f"Using entire duration of recording: {rec.get_total_duration() / 60:.2f}min"
            )
        else:
            print(f"Only analyzing the first {_globals.FIRST_N_MINS} min of recording")
            rec = rec.frame_slice(start_frame=0, end_frame=end_frame)
    else:
        print(
            f"Using entire duration of recording: {rec.get_total_duration() / 60:.2f}min"
        )

    if len(_globals.REC_CHUNKS) > 0:
        print(f"Using {len(_globals.REC_CHUNKS)} chunks of the recording")
        rec_chunks = []
        for c, (start_frame, end_frame) in enumerate(_globals.REC_CHUNKS):
            print(f"Chunk {c}: {start_frame} to {end_frame} frame")
            chunk = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
            rec_chunks.append(chunk)
        rec = si_segmentutils.concatenate_recordings(rec_chunks)
    else:
        print(f"Using entire recording")

    if _globals.MEA_Y_MAX is not None:
        print(
            f"Flipping y-coordinates of channel locations. MEA height: {_globals.MEA_Y_MAX}"
        )
        probes_all = []
        for probe in rec.get_probes():
            y_cords = probe._contact_positions[:, 1]

            if _globals.MEA_Y_MAX is None:
                y_cords_flipped = y_cords
            elif _globals.MEA_Y_MAX == -1:
                y_cords_flipped = max(y_cords) - y_cords
            else:
                y_cords_flipped = _globals.MEA_Y_MAX - y_cords

            probe._contact_positions[np.arange(y_cords_flipped.size), 1] = (
                y_cords_flipped
            )
            probes_all.append(probe)
        rec = rec.set_probes(probes_all)

    stopwatch.log_time("Done loading recording.")

    return rec


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


def _waveform_extractor_to_spikedata(
    w_e: Any, rec_path: Any, rec_chunks: Optional[list] = None
) -> Any:
    """Convert a WaveformExtractor to a SpikeData with rich neuron attributes.

    Extracts spike trains, full waveform templates, channel locations,
    SNR, normalized STD, polarity, and all per-unit metadata needed by
    the Compiler.  The resulting SpikeData does **not** carry
    ``raw_data`` (to avoid duplicating large voltage traces).

    When *rec_chunks* is provided (list of ``(start_frame, end_frame)``
    tuples from concatenated recordings), per-epoch average waveform
    templates are computed and stored as ``epoch_templates``.

    Parameters
    ----------
    w_e : WaveformExtractor
        Waveform extractor (curated or uncurated).
    rec_path : str or Path
        Original recording file path, stored as source metadata.
    rec_chunks : list of (int, int) or None
        Frame boundaries for each concatenated recording epoch.
        When None or empty, ``epoch_templates`` is not stored.

    Returns
    -------
    SpikeData
        Spike trains in milliseconds with per-unit attributes:
        ``unit_id``, ``channel``, ``channel_id``, ``x``, ``y``,
        ``electrode``, ``template``, ``template_full``,
        ``template_peak_ind``, ``amplitude``, ``amplitudes``,
        ``peak_inds``, ``std_norms_all``, ``has_pos_peak``,
        ``snr``, ``std_norm``, and optionally ``epoch_templates``.
    """
    from spikelab.spikedata import SpikeData

    sorting = w_e.sorting
    fs_Hz = float(w_e.sampling_frequency)
    rec_locations = w_e.recording.get_channel_locations()
    channel_ids = w_e.recording.get_channel_ids()

    # Electrode IDs (optional)
    try:
        electrode_ids = w_e.recording.get_property("electrode")
    except Exception:
        electrode_ids = None
    if electrode_ids is None:
        electrode_ids = channel_ids

    # Noise levels for SNR
    noise_levels = _get_noise_levels(w_e.recording, getattr(w_e, "return_scaled", True))

    # Polarity flags
    use_pos_peak = w_e.use_pos_peak

    # Template windowing for compile_dict
    nbefore_compiled = w_e.ms_to_samples(_globals.COMPILED_WAVEFORMS_MS_BEFORE)
    nafter_compiled = w_e.ms_to_samples(_globals.COMPILED_WAVEFORMS_MS_AFTER) + 1

    has_epochs = rec_chunks is not None and len(rec_chunks) > 1

    trains = []
    neuron_attributes = []
    for uid in sorting.unit_ids:
        spike_samples = sorting.get_unit_spike_train(uid)
        spike_times_ms = np.sort(spike_samples.astype(float) / fs_Hz * 1000.0)
        trains.append(spike_times_ms)

        # Channel with largest amplitude
        chan_max = int(w_e.chans_max_all[uid])
        x, y = rec_locations[chan_max]

        # Full template (all channels)
        template_mean = w_e.get_computed_template(unit_id=uid, mode="average")
        template_std = w_e.get_computed_template(unit_id=uid, mode="std")
        peak_ind_full = w_e.peak_ind

        # When SCALE_COMPILED_WAVEFORMS is False, convert µV templates
        # back to raw ADC counts.  Waveforms are now extracted as µV by
        # default (return_scaled=True), so this inverts the scaling.
        if not _globals.SCALE_COMPILED_WAVEFORMS and w_e.return_scaled:
            gain = w_e.recording.get_channel_gains()
            offset = w_e.recording.get_channel_offsets()
            template_mean = ((template_mean - offset) / gain).astype(
                w_e.recording.get_dtype()
            )
            template_std = ((template_std - offset) / gain).astype(
                w_e.recording.get_dtype()
            )

        # Windowed template (for compile_dict)
        template_windowed = template_mean[
            peak_ind_full - nbefore_compiled : peak_ind_full + nafter_compiled, :
        ]

        # Per-channel amplitudes and peak indices (from windowed template)
        template_abs = np.abs(template_windowed)
        peak_inds = np.argmax(template_abs, axis=0)
        amplitudes = template_abs[peak_inds, range(peak_inds.size)]
        amplitude_max = float(amplitudes[chan_max])

        # SNR on max channel
        noise = float(noise_levels[chan_max]) if chan_max < len(noise_levels) else 1.0
        snr = float(amplitude_max / noise) if noise > 0 else 0.0

        # Normalized STD per channel
        peak_ind_buffer = peak_ind_full - nbefore_compiled
        if _globals.STD_AT_PEAK:
            stds = template_std[peak_ind_buffer + peak_inds, range(peak_inds.size)]
        else:
            nb = w_e.ms_to_samples(_globals.STD_OVER_WINDOW_MS_BEFORE)
            na = w_e.ms_to_samples(_globals.STD_OVER_WINDOW_MS_AFTER) + 1
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

        # Spike train in samples (for compilation)
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

        # Waveforms path (for COMPILE_WAVEFORMS — loaded on demand)
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
        "source_format": "Kilosort2",  # Legacy path; only used by old non-modular pipeline
        "fs_Hz": fs_Hz,
        "channel_locations": rec_locations.copy(),
        "n_samples": int(w_e.recording.get_num_samples()),
    }
    if has_epochs:
        metadata["rec_chunks_frames"] = list(rec_chunks)
        metadata["rec_chunks_ms"] = [
            (s / fs_Hz * 1000.0, e / fs_Hz * 1000.0) for s, e in rec_chunks
        ]
        metadata["rec_chunk_names"] = (
            list(_globals._REC_CHUNK_NAMES) if _globals._REC_CHUNK_NAMES else None
        )

    return SpikeData(trains, metadata=metadata, neuron_attributes=neuron_attributes)


def _curate_spikedata(
    sd: Any,
    curation_folder: Union[str, Path],
    recurate: bool = False,
    **curate_kwargs: Any,
) -> Tuple[Any, dict]:
    """Curate a SpikeData with disk caching for the sorting pipeline.

    If cached results exist and *recurate* is False, loads the cached
    unit IDs and returns a subset of *sd*.  Otherwise runs
    ``sd.curate()``, saves the results (``unit_ids.npy`` and
    ``curation_history.json``) to *curation_folder*, and returns the
    curated SpikeData.

    Parameters
    ----------
    sd : SpikeData
        Uncurated (or partially curated) SpikeData.
    curation_folder : str or Path
        Directory for cached curation artefacts.
    recurate : bool
        If True, re-run curation even when cached results exist.
    **curate_kwargs
        Keyword arguments forwarded to ``sd.curate()`` (e.g.
        ``min_spikes``, ``min_rate_hz``, ``min_snr``, etc.).

    Returns
    -------
    sd_curated : SpikeData
        SpikeData containing only units that passed all criteria.
    history : dict
        Serializable curation history dict.
    """
    import json
    from spikelab.spikedata import SpikeData
    from spikelab.spikedata.curation import build_curation_history

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


def load_single_recording(rec_path: Any) -> BaseRecording:
    """Load one recording file and return a scaled, bandpass-filtered recording.

    Supports Maxwell ``.h5`` files, NWB ``.nwb`` files, and pre-loaded
    SpikeInterface ``BaseRecording`` objects. The recording is scaled to
    µV (using ``GAIN_TO_UV`` / ``OFFSET_TO_UV`` or the recording's own
    gains) and bandpass-filtered between ``FREQ_MIN`` and ``FREQ_MAX``.

    Parameters:
        rec_path (str, Path, or BaseRecording): Path to a ``.h5`` or
            ``.nwb`` file, or an already-loaded ``BaseRecording``.

    Returns:
        rec (BaseRecording): Scaled and bandpass-filtered recording.
    """
    if isinstance(rec_path, BaseRecording):
        rec = rec_path
    elif str(rec_path).endswith(".h5"):
        maxwell_kwargs = {}
        if _globals.STREAM_ID is not None:
            maxwell_kwargs["stream_id"] = _globals.STREAM_ID
        rec = MaxwellRecordingExtractor(rec_path, **maxwell_kwargs)
        test_file = h5py.File(rec_path)
        if "sig" not in test_file:  # Test if hdf5_plugin_path is needed
            try:
                test_file["/data_store/data0000/groups/routed/raw"][0, 0]
            except OSError as exception:
                test_file.close()
                print("*" * 10)
                print("""This MaxWell Biosystems file format is based on HDF5.
The internal compression requires a custom plugin.
Please visit this page and install the missing decompression libraries:
https://share.mxwbio.com/d/4742248b2e674a85be97/

Setup options (choose one):
    1. Pass hdf5_plugin_path='/path/to/plugin/' to sort_with_kilosort2().
    2. Set os.environ['HDF5_PLUGIN_PATH'] BEFORE importing this module.
    3. Follow the Maxwell instructions at the link above.
""")
                print("*" * 10)
                raise (exception)
        test_file.close()
        # Reconcile declared vs. routed channels. MaxOne recordings report
        # 1024 readout channels but get_traces() returns the full 1024-wide
        # array regardless of routing; slicing by the extractor's own
        # channel_ids forces the width to match get_num_channels(). No-op
        # when all channels are routed (MaxTwo).
        rec = rec.select_channels(rec.get_channel_ids())
    elif str(rec_path).endswith(".nwb"):
        rec = NwbRecordingExtractor(rec_path)
    else:
        raise ValueError(
            f"Recording {rec_path} is not in .h5 or .nwb format.\n"
            f"Load it with SpikeInterface and pass the BaseRecording object "
            f"instead of the file path. See "
            f"https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html"
        )

    if rec.get_num_segments() != 1:
        raise ValueError(
            f"Recording has {rec.get_num_segments()} segments — expected 1. "
            "Divide the recording into separate single-segment recordings."
        )

    if _globals.GAIN_TO_UV is not None:
        gain = _globals.GAIN_TO_UV
    elif rec.get_channel_gains() is not None:
        gain = rec.get_channel_gains()
    else:
        print("Recording does not have channel gains to uV")
        gain = 1.0

    if _globals.OFFSET_TO_UV is not None:
        offset = _globals.OFFSET_TO_UV
    elif rec.get_channel_offsets() is not None:
        offset = rec.get_channel_offsets()
    else:
        print("Recording does not have channel offsets to uV")
        offset = 0.0

    print(
        f"Scaling recording to uV with gain {np.median(np.array(gain))} and offset {np.median(np.array(offset))}"
    )
    print(f"Converting recording dtype from {rec.get_dtype()} to float32")

    rec = ScaleRecording(rec, gain=gain, offset=offset, dtype="float32")

    rec = bandpass_filter(rec, freq_min=_globals.FREQ_MIN, freq_max=_globals.FREQ_MAX)

    return rec


def concatenate_recordings(rec_path: Path) -> BaseRecording:
    """Load and concatenate all recordings in a directory.

    Scans *rec_path* for ``.raw.h5`` and ``.nwb`` files, loads each via
    ``load_single_recording``, and concatenates them into a single
    multi-segment recording. Updates the global ``REC_CHUNKS`` with the
    frame boundaries of each constituent recording.

    Parameters:
        rec_path (Path): Directory containing recording files.

    Returns:
        rec (BaseRecording): The concatenated recording.

    Notes:
        Before concatenation, all recordings are validated for
        compatibility:

        - **Channel count** and **sampling frequency** must match
          across all files — a ``ValueError`` is raised otherwise.
        - **Channel IDs** and **channel locations** are compared
          against the first file.  Mismatches produce a warning but
          do not block concatenation, since the user may intentionally
          combine recordings with different routing configurations.
          However, differing electrode layouts will likely produce
          unreliable sorting results.
    """
    print("Concatenating recordings")
    recordings = []

    new_rec_chunks = []
    start_frame = 0

    recording_names = natsorted(
        [
            p.name
            for p in rec_path.iterdir()
            if p.name.endswith(".raw.h5") or p.name.endswith(".nwb")
        ]
    )
    for rec_name in recording_names:
        rec_file = [p for p in rec_path.iterdir() if p.name == rec_name][0]
        rec = load_single_recording(rec_file)
        recordings.append(rec)
        print(
            f"{rec_name}: DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
            f"NUM. CHANNELS: {rec.get_num_channels()}"
        )

        end_frame = start_frame + rec.get_total_samples()
        new_rec_chunks.append((start_frame, end_frame))
        start_frame = end_frame

    # Validate compatibility before concatenation
    if len(recordings) > 1:
        ref = recordings[0]
        ref_name = recording_names[0]
        ref_n_ch = ref.get_num_channels()
        ref_fs = ref.get_sampling_frequency()
        ref_ids = list(ref.get_channel_ids())
        ref_locs = ref.get_channel_locations()

        for i, (rec_i, name_i) in enumerate(
            zip(recordings[1:], recording_names[1:]), start=1
        ):
            # Hard error: channel count or sampling frequency mismatch
            n_ch = rec_i.get_num_channels()
            if n_ch != ref_n_ch:
                raise ValueError(
                    f"Cannot concatenate: {name_i} has {n_ch} channels "
                    f"but {ref_name} has {ref_n_ch}."
                )
            fs = rec_i.get_sampling_frequency()
            if fs != ref_fs:
                raise ValueError(
                    f"Cannot concatenate: {name_i} has sampling frequency "
                    f"{fs} Hz but {ref_name} has {ref_fs} Hz."
                )

            # Warning: channel IDs differ
            ids_i = list(rec_i.get_channel_ids())
            if ids_i != ref_ids:
                warnings.warn(
                    f"{name_i} has different channel IDs than {ref_name}. "
                    "Concatenation will proceed but results may be unreliable "
                    "if the electrode configurations differ.",
                    stacklevel=2,
                )

            # Warning: channel locations differ
            locs_i = rec_i.get_channel_locations()
            if not np.array_equal(ref_locs, locs_i):
                warnings.warn(
                    f"{name_i} has different channel locations than "
                    f"{ref_name}. This likely means different electrode "
                    "configurations — concatenation will proceed but "
                    "sorting results may be unreliable.",
                    stacklevel=2,
                )

    if len(recordings) == 1:
        rec = recordings[0]
    else:
        rec = si_segmentutils.concatenate_recordings(recordings)
        if len(_globals.REC_CHUNKS) == 0:
            _globals.REC_CHUNKS = new_rec_chunks

    print(f"Done concatenating {len(recordings)} recordings")
    print(f"Total duration: {rec.get_total_duration()}s")

    # Store file names globally so _waveform_extractor_to_spikedata can
    # include them in metadata for downstream epoch splitting.
    _globals._REC_CHUNK_NAMES = recording_names

    return rec


def extract_waveforms(
    recording_path: Any,
    recording: BaseRecording,
    sorting: Any,
    root_folder: Path,
    initial_folder: Path,
    **job_kwargs: Any,
) -> Any:
    """
    Extracts waveform on paired Recording-Sorting objects.
    Waveforms are persistent on disk and cached in memory.

    Parameters
    ----------
    recording_path: Path
        The path of the raw recording
    recording: Recording
        The recording object
    sorting: Sorting
        The sorting object
    root_folder: Path
        The root folder of waveforms
    initial_folder: Path
        Folder representing units before curation

    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object that represents the waveforms
    """

    print_stage("EXTRACTING WAVEFORMS")
    stopwatch = Stopwatch()

    if (
        not _globals.REEXTRACT_WAVEFORMS and (root_folder / "waveforms").is_dir()
    ):  # Load saved waveform extractor
        print("Loading waveforms from folder")
        we = WaveformExtractor.load_from_folder(
            recording, sorting, root_folder, initial_folder
        )
        stopwatch.log_time("Done extracting waveforms.")
    else:  # Create new waveform extractor
        we = WaveformExtractor.create_initial(
            recording_path, recording, sorting, root_folder, initial_folder
        )
        if _globals.STREAMING_WAVEFORMS:
            # Streaming path: per-unit waveforms + templates in one pass.
            # Bounded peak RAM (one unit's buffer at a time); avoids the
            # 39 GB pre-allocated per-unit memmap pile that the parallel
            # path creates for high-unit-count sorts on dense MEAs.
            print("Streaming waveform extraction (per-unit, low RAM)")
            we.run_extract_waveforms_streaming()
            stopwatch.log_time("Done extracting waveforms (streaming).")
            # Templates already populated by the streaming pass.
        else:
            we.run_extract_waveforms(**job_kwargs)
            stopwatch.log_time("Done extracting waveforms.")
            we.compute_templates(
                modes=("average", "std"), n_jobs=job_kwargs.get("n_jobs", 1)
            )
    return we


def process_recording(
    rec_name: str,
    rec_path: Any,
    inter_path: Any,
    results_path: Any,
    rec_loaded: Any = None,
) -> Any:
    """Run the full sorting pipeline on a single recording.

    Orchestrates path setup, recording loading, spike sorting, waveform
    extraction, SpikeData-based curation, result compilation, and
    optional trace saving for downstream models.

    Parameters:
        rec_name (str): Short name for the recording (used in logging
            and result filenames).
        rec_path (str or Path): Path to the recording file.
        inter_path (str or Path): Root intermediate directory.
        results_path (str or Path): Root results directory.
        rec_loaded (BaseRecording or None): Pre-loaded recording object.
            When provided, used instead of loading from *rec_path*.

    Returns:
        result (tuple or Exception): ``(sd_raw, sd_curated)`` on success
            when ``SAVE_RAW_PKL`` is True, otherwise just ``sd_curated``.
            Returns the caught exception if any stage failed.
    """
    create_folder(inter_path)
    with Tee(Path(inter_path) / _globals.OUT_FILE, "a"):
        stopwatch = Stopwatch()

        # Get Paths
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
        ) = get_paths(rec_path, inter_path, results_path)

        # Save a copy of the script
        if _globals.SAVE_SCRIPT:
            print_stage("SAVING SCRIPT")
            copy_script(inter_path)

        # Load Recording
        try:
            recording_filtered = load_recording(
                rec_path if rec_loaded is None else rec_loaded
            )
        except Exception as e:
            print(f"Could not open the recording file because of {e}")
            print("Moving on to next recording")
            return e

        # Spike sorting
        sorting = spike_sort(
            rec_cache=recording_filtered,
            rec_path=rec_path,
            recording_dat_path=recording_dat_path,
            output_folder=output_folder,
        )
        if isinstance(sorting, BaseException):  # Could not sort recording
            return sorting

        # Extract waveforms
        w_e_raw = extract_waveforms(
            rec_path,
            recording_filtered,
            sorting,
            waveforms_root_folder,
            curation_initial_folder,
            n_jobs=_globals.N_JOBS,
            total_memory=_globals.TOTAL_MEMORY,
            progress_bar=True,
        )

        # Convert to SpikeData with enriched neuron_attributes
        # (SNR, std_norm, channel locations, templates)
        sd = _waveform_extractor_to_spikedata(
            w_e_raw, rec_path, rec_chunks=_globals.REC_CHUNKS or None
        )

        # Curate via SpikeData methods with disk caching
        curate_kwargs = {}
        if _globals.CURATE_FIRST:
            if _globals.FR_MIN is not None:
                curate_kwargs["min_rate_hz"] = _globals.FR_MIN
            if _globals.ISI_VIOL_MAX is not None:
                curate_kwargs["isi_max"] = _globals.ISI_VIOL_MAX
                curate_kwargs["isi_threshold_ms"] = 1.5
                curate_kwargs["isi_method"] = _globals.ISI_VIOLATION_METHOD
            if _globals.SNR_MIN is not None:
                curate_kwargs["min_snr"] = _globals.SNR_MIN
            if _globals.SPIKES_MIN_FIRST is not None:
                curate_kwargs["min_spikes"] = _globals.SPIKES_MIN_FIRST
        if _globals.CURATE_SECOND:
            # Use the stricter spike count if second-stage is enabled
            if _globals.SPIKES_MIN_SECOND is not None:
                curate_kwargs["min_spikes"] = _globals.SPIKES_MIN_SECOND
            if _globals.STD_NORM_MAX is not None:
                curate_kwargs["max_std_norm"] = _globals.STD_NORM_MAX

        # Determine which SpikeData to curate on: the full concatenated
        # one (default) or a single epoch's data.
        has_epochs = bool(sd.metadata.get("rec_chunks_ms"))
        if _globals.CURATION_EPOCH is not None and has_epochs:
            epoch_sds = sd.split_epochs()
            if _globals.CURATION_EPOCH < 0 or _globals.CURATION_EPOCH >= len(epoch_sds):
                raise ValueError(
                    f"curation_epoch={_globals.CURATION_EPOCH} is out of range "
                    f"(recording has {len(epoch_sds)} epochs, 0-indexed)."
                )
            sd_for_curation = epoch_sds[_globals.CURATION_EPOCH]
            print(
                f"Curating based on epoch {_globals.CURATION_EPOCH} "
                f"({sd_for_curation.metadata.get('source_file', '')})"
            )
        else:
            sd_for_curation = sd

        sd_epoch_curated, curation_history = _curate_spikedata(
            sd_for_curation,
            curation_folder=curation_first_folder,
            recurate=_globals.RECURATE_FIRST or _globals.RECURATE_SECOND,
            **curate_kwargs,
        )

        # When curating on a single epoch, apply the passing unit IDs
        # back to the full concatenated SpikeData.
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

        # Compile results using SpikeData
        compile_results(rec_name, rec_path, results_path, sd_curated, curation_history)

        # Save scaled traces for training detection model
        if _globals.SAVE_DL_DATA:
            from .trace_io import save_traces

            save_stopwatch = Stopwatch("SAVING TRACES FOR DETECTION MODEL")
            save_traces(rec_path if rec_loaded is None else rec_loaded, results_path)
            save_stopwatch.log_time()

        print_stage(f"DONE WITH RECORDING")
        print(f"Recording: {rec_path}")
        stopwatch.log_time("Total")

        if _globals.SAVE_RAW_PKL:
            return sd, sd_curated
        return sd_curated


def copy_script(path: Path) -> None:
    """Save a timestamped copy of this module to the given directory.

    Parameters:
        path (Path): Destination directory.
    """
    copied_script_name = (
        time.strftime("%y%m%d_%H%M%S") + "_" + os.path.basename(__file__)
    )
    copied_path = (path / copied_script_name).absolute()
    shutil.copyfile(__file__, copied_path)
    print(f"Saved a copy of script to {copied_path}")


def compile_results(
    rec_name: str,
    rec_path: Any,
    results_path: Any,
    sd: Any,
    curation_history: Optional[dict] = None,
) -> None:
    """Compile and export sorting results for a single recording.

    Saves spike times, electrode information, and optionally ``.npz`` /
    ``.mat`` files via a ``Compiler`` instance. When the recording was
    built from multiple chunks (``REC_CHUNKS``), each chunk is compiled
    separately into its own sub-folder using ``split_epochs``.

    Parameters:
        rec_name (str): Short name for the recording.
        rec_path (str or Path): Original recording file path.
        results_path (Path): Output directory for compiled results.
        sd (SpikeData): Curated SpikeData with enriched neuron_attributes.
        curation_history (dict or None): Curation history dict.
    """
    compile_stopwatch = Stopwatch("COMPILING RESULTS")
    print(f"For recording: {rec_path}")
    if _globals.COMPILE_SINGLE_RECORDING:
        if (
            not (results_path / "parameters.json").exists()
            or _globals.RECOMPILE_SINGLE_RECORDING
        ):
            print(f"Saving to path: {results_path}")
            if len(_globals.REC_CHUNKS) > 1:
                epoch_sds = sd.split_epochs()
                for c, sd_chunk in enumerate(epoch_sds):
                    print(f"Compiling chunk {c}")
                    compiler = Compiler()
                    compiler.add_recording(rec_name, sd_chunk, curation_history)
                    compiler.save_results(results_path / f"chunk{c}")
            else:
                compiler = Compiler()
                compiler.add_recording(rec_name, sd, curation_history)
                compiler.save_results(results_path)
                compile_stopwatch.log_time("Done compiling results.")
        else:
            print(
                "Skipping compiling results because 'recompile_single_recording' is set to False and already compiled"
            )
    else:
        print(
            f"Skipping compiling results because 'compile_single_recording' is set to False"
        )
