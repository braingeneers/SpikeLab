"""
Lightweight loaders that convert common neurophysiology formats into
`spikedata.SpikeData` objects.

Supported inputs (best-effort, optional deps):
    - HDF5 (generic): spike times, (indices,times), or raster matrices
    - NWB: reads Units table spike_times (via pynwb if available, else h5py)
    - KiloSort/Phy outputs: spike_times.npy + spike_clusters.npy (+ optional TSV)
    - SpikeInterface: from a SortingExtractor

Times are converted to milliseconds to match `SpikeData` conventions.
These helpers avoid hard dependencies: optional libraries are imported lazily.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Union

import os
import warnings

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

from spikedata import SpikeData

__all__ = [
    "load_spikedata_from_hdf5",
    "load_spikedata_from_hdf5_raw_thresholded",
    "load_spikedata_from_nwb",
    "load_spikedata_from_kilosort",
    "load_spikedata_from_spikeinterface",
    "load_spikedata_from_spikeinterface_recording",
]


def _ensure_h5py():
    """Ensure the optional h5py dependency is available."""
    if h5py is None:
        raise ImportError("h5py is required for HDF5/NWB loaders. `pip install h5py`.")


def _to_ms(values: np.ndarray, unit: str, fs_Hz: Optional[float]) -> np.ndarray:
    """Convert a vector of times to milliseconds."""
    if unit == "ms":
        return values.astype(float)
    if unit == "s":
        return values.astype(float) * 1e3
    if unit == "samples":
        if not fs_Hz or fs_Hz <= 0:
            raise ValueError("fs_Hz must be provided and > 0 when unit='samples'")
        return values.astype(float) / fs_Hz * 1e3
    raise ValueError(f"Unknown time unit '{unit}' (expected 's','ms','samples')")


def _trains_from_flat_index(
    flat_times: np.ndarray,
    end_indices: np.ndarray,
    *,
    unit: str,
    fs_Hz: Optional[float],
) -> List[np.ndarray]:
    """Split a flat time array into per-unit trains using end indices and convert to ms."""
    trains: List[np.ndarray] = []
    start = 0
    for stop in end_indices:
        segment = flat_times[start:stop]
        trains.append(_to_ms(segment, unit, fs_Hz))
        start = stop
    return trains


def _read_raw_arrays(
    f,
    raw_dataset: Optional[str],
    raw_time_dataset: Optional[str],
    raw_time_unit: str,
    fs_Hz: Optional[float],
) -> tuple[Optional[np.ndarray], Optional[Union[np.ndarray, float]]]:
    """Read optional raw arrays and convert the time vector to milliseconds."""
    raw_data = None
    raw_time: Optional[Union[np.ndarray, float]] = None
    if raw_dataset is not None:
        raw_data = np.asarray(f[raw_dataset])
        if raw_time_dataset is not None:
            raw_time_vals = np.asarray(f[raw_time_dataset])
            if raw_time_unit == "s":
                raw_time = raw_time_vals * 1e3
            elif raw_time_unit == "ms":
                raw_time = raw_time_vals
            elif raw_time_unit == "samples":
                if not fs_Hz:
                    raise ValueError(
                        "fs_Hz must be provided for raw_time_unit='samples'"
                    )
                raw_time = raw_time_vals / float(fs_Hz) * 1e3
            else:
                raise ValueError("raw_time_unit must be one of 's','ms','samples'")
    return raw_data, raw_time


def _maybe_with_raw(
    sd: SpikeData,
    raw_data: Optional[np.ndarray],
    raw_time: Optional[Union[np.ndarray, float]],
) -> SpikeData:
    """Return SpikeData with raw fields attached if provided, else original."""
    if raw_data is not None and raw_time is not None:
        return _build_spikedata(
            sd.train,
            length_ms=sd.length,
            metadata=sd.metadata,
            raw_data=raw_data,
            raw_time=raw_time,
        )
    return sd


def _build_spikedata(
    trains_ms: List[np.ndarray],
    *,
    length_ms: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
    raw_data: Optional[np.ndarray] = None,
    raw_time: Optional[Union[np.ndarray, float]] = None,
    neuron_attributes: Optional[List[dict]] = None,
) -> SpikeData:
    """Internal helper to construct a SpikeData with sensible defaults. Infers `length_ms` from the last spike if not provided."""
    if length_ms is None:
        last = [t[-1] for t in trains_ms if len(t) > 0]
        length_ms = float(max(last)) if last else 0.0
    return SpikeData(
        trains_ms,
        length=length_ms,
        metadata=dict(metadata) if metadata else {},
        raw_data=raw_data,
        raw_time=raw_time,
        neuron_attributes=neuron_attributes,
    )


# ----------------------------
# HDF5
# ----------------------------


def load_spikedata_from_hdf5(
    filepath: str,
    *,
    raster_dataset: Optional[str] = None,
    raster_bin_size_ms: Optional[float] = None,
    spike_times_dataset: Optional[str] = None,
    spike_times_index_dataset: Optional[str] = None,
    spike_times_unit: str = "s",
    fs_Hz: Optional[float] = None,
    group_per_unit: Optional[str] = None,
    group_time_unit: str = "s",
    idces_dataset: Optional[str] = None,
    times_dataset: Optional[str] = None,
    times_unit: str = "s",
    raw_dataset: Optional[str] = None,
    raw_time_dataset: Optional[str] = None,
    raw_time_unit: str = "s",
    length_ms: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> SpikeData:
    """
    Load spike trains from a generic HDF5 file using one of four supported input styles.

    Input Styles:

    1. Raster Matrix
        Use when the HDF5 file contains a 2D array representing spike counts or a binary raster.
        Parameters:
            raster_dataset (str): Path to the dataset containing the raster/counts matrix (shape: units × time).
            raster_bin_size_ms (float): Bin width in milliseconds.
        The matrix is interpreted as (units × time bins), where each entry is the spike count (or 0/1 for binary).
        Example: `raster_dataset="/spikes/raster", raster_bin_size_ms=1.0`

    2. Ragged Arrays (NWB-style)
        Use when spike times for all units are concatenated into a single array, with an index array marking the end of each unit's spike times.
        Parameters:
            spike_times_dataset (str): Path to the flat array of spike times.
            spike_times_index_dataset (str): Path to the array of indices (end positions for each unit).
            spike_times_unit (str): Unit of the spike times ('s', 'ms', or 'samples').
            fs_Hz (float, optional): Required if unit is 'samples'.
        Example: `spike_times_dataset="/units/spike_times", spike_times_index_dataset="/units/spike_times_index"`

    3. Group-per-Unit
        Use when each unit's spike times are stored as a separate dataset within a group.
        Parameters:
            group_per_unit (str): Path to the group containing one dataset per unit.
            group_time_unit (str): Unit of the spike times ('s', 'ms', or 'samples').
            fs_Hz (float, optional): Required if unit is 'samples'.
        Example: `group_per_unit="/spikes/unit_times"`

    4. Paired Arrays (Indices and Times)
        Use when there are two parallel arrays: one for unit indices and one for spike times.
        Parameters:
            idces_dataset (str): Path to the array of unit indices (int).
            times_dataset (str): Path to the array of spike times.
            times_unit (str): Unit of the spike times ('s', 'ms', or 'samples').
            fs_Hz (float, optional): Required if unit is 'samples'.
        Example: `idces_dataset="/spikes/unit_ids", times_dataset="/spikes/times"`

    Optional Raw Data:
        You may also attach raw analog data and its timebase by specifying:
            raw_dataset (str): Path to the raw data array.
            raw_time_dataset (str): Path to the time vector for the raw data.
            raw_time_unit (str): Unit of the raw time vector ('s', 'ms', or 'samples').
            fs_Hz (float, required if 'samples'): Sampling frequency for conversion.


    Returns: sd (SpikeData): The loaded spike train data.

    Raises: ValueError: If not exactly one input style is specified, or if required arguments are missing.
    """
    _ensure_h5py()

    # Validate exactly one style is provided
    provided = [
        raster_dataset is not None,
        spike_times_dataset is not None and spike_times_index_dataset is not None,
        group_per_unit is not None,
        idces_dataset is not None and times_dataset is not None,
    ]
    if sum(provided) != 1:
        raise ValueError("Specify exactly one HDF5 input style")

    # Accumulate metadata and preserve file path provenance
    meta = dict(metadata or {})
    meta.setdefault("source_file", os.path.abspath(filepath))

    with h5py.File(filepath, "r") as f:  # type: ignore
        # Optionally read raw arrays and a time vector
        raw_data, raw_time = _read_raw_arrays(
            f,
            raw_dataset,
            raw_time_dataset,
            raw_time_unit,
            fs_Hz,
        )

        if raster_dataset is not None:
            # Style (1): counts/raster matrix -> SpikeData via from_raster
            if raster_bin_size_ms is None:
                raise ValueError("raster_bin_size_ms is required for raster_dataset")
            raster = np.asarray(f[raster_dataset])
            if raster.ndim != 2:
                raise ValueError("raster_dataset must be 2D (units, time)")
            total_time = raster.shape[1] * raster_bin_size_ms
            if total_time > 0:
                # subtract the smallest representable spacing so the length is
                # slightly less than the exact bin-aligned value and avoids
                # triggering the extra empty bin in `SpikeData.raster`.
                length_ms = max(total_time - np.spacing(total_time), 0.0)
            else:
                length_ms = 0.0
            sd = SpikeData.from_raster(raster, raster_bin_size_ms, length=length_ms)
            sd.metadata.update(meta)
            return _maybe_with_raw(sd, raw_data, raw_time)

        if spike_times_dataset is not None and spike_times_index_dataset is not None:
            # Style (2): flat ragged spike_times + spike_times_index
            flat = np.asarray(f[spike_times_dataset])
            index = np.asarray(f[spike_times_index_dataset])
            trains = _trains_from_flat_index(
                flat, index, unit=spike_times_unit, fs_Hz=fs_Hz
            )
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        if group_per_unit is not None:
            # Style (3): each child dataset is a unit's spike times
            grp = f[group_per_unit]
            keys = sorted(list(grp.keys()))
            trains = [_to_ms(np.asarray(grp[k]), group_time_unit, fs_Hz) for k in keys]
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        # Style (4): paired indices and times arrays
        idces = np.asarray(f[idces_dataset])  # type: ignore
        times = _to_ms(np.asarray(f[times_dataset]), times_unit, fs_Hz)  # type: ignore
        N = int(idces.max()) + 1 if idces.size else 0
        sd = SpikeData.from_idces_times(idces, times, N=N, length=length_ms)
        sd.metadata.update(meta)
        return _maybe_with_raw(sd, raw_data, raw_time)


def load_spikedata_from_hdf5_raw_thresholded(
    filepath: str,
    dataset: str,
    *,
    fs_Hz: float,
    threshold_sigma: float = 5.0,
    filter: Union[dict, bool] = True,
    hysteresis: bool = True,
    direction: str = "both",
) -> SpikeData:
    """
    Threshold-and-detect spikes from an HDF5 dataset of raw traces.

    Parameters:
        filepath (str): Path to HDF5 file.
        dataset (str): HDF5 dataset path containing raw traces shaped (channels, time).
        fs_Hz (float): Sampling frequency in Hz.
        threshold_sigma (float): Threshold in units of per-channel standard deviation.
        filter (dict | bool): If True, apply default Butterworth bandpass; if dict, pass to filter; if False, no filtering.
        hysteresis (bool): Use rising-edge detection if True.
        direction (str): 'both' | 'up' | 'down'.
    Returns: sd (SpikeData): The detected spike train data.
    """
    _ensure_h5py()
    with h5py.File(filepath, "r") as f:  # type: ignore
        data = np.asarray(f[dataset])
    return SpikeData.from_thresholding(
        data,
        fs_Hz=fs_Hz,
        threshold_sigma=threshold_sigma,
        filter=filter,
        hysteresis=hysteresis,
        direction=direction,  # type: ignore[arg-type]
    )


# ----------------------------
# NWB (units table)
# ----------------------------


def load_spikedata_from_nwb(
    filepath: str,
    *,
    prefer_pynwb: bool = True,
    length_ms: Optional[float] = None,
) -> SpikeData:
    """
    Load spike trains from an NWB file's Units table.

    Parameters:
        filepath (str): Path to the NWB file.
        prefer_pynwb (bool): If True, try pynwb first; if False, try h5py.
        length_ms (float, optional): Recording duration in milliseconds.

    Returns: sd (SpikeData): The loaded spike train data.
    """
    trains: List[np.ndarray] = []
    neuron_attributes: List[dict] = []
    meta = {"source_file": os.path.abspath(filepath), "format": "NWB"}

    if prefer_pynwb:
        try:
            from pynwb import NWBHDF5IO  # type: ignore

            with NWBHDF5IO(filepath, "r") as io:
                nwb = io.read()
                if getattr(nwb, "units", None) is None:
                    raise ValueError("NWB file has no Units table")
                df = nwb.units.to_dataframe()
                for row in df.itertuples():
                    stimes = np.asarray(row.spike_times, dtype=float)
                    trains.append(stimes * 1e3)
                    attr = {"unit_id": row.Index}
                    for col in ("electrodes", "electrode_group", "channel", "ch"):
                        if col in df.columns:
                            val = getattr(row, col, None)
                            if val is not None:
                                attr["channel"] = (
                                    int(val[0])
                                    if hasattr(val, "__len__")
                                    and not isinstance(val, str)
                                    else val
                                )
                                break
                    neuron_attributes.append(attr)
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                neuron_attributes=neuron_attributes,
            )
        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"Falling back to h5py for NWB reading ({type(e).__name__}: {e})"
            )

    _ensure_h5py()
    with h5py.File(filepath, "r") as f:  # type: ignore
        if "units" not in f:
            raise ValueError("NWB file missing '/units' group")
        unit_grp = f["units"]
        st_key = "spike_times"
        idx_key = "spike_times_index"
        if st_key not in unit_grp or idx_key not in unit_grp:
            candidates = [k for k in unit_grp.keys() if k.endswith("spike_times")]
            idx_candidates = [
                k for k in unit_grp.keys() if k.endswith("spike_times_index")
            ]
            if not candidates or not idx_candidates:
                raise ValueError("Could not find spike_times datasets in NWB file")
            st_key = candidates[0]
            idx_key = idx_candidates[0]

        flat = np.asarray(unit_grp[st_key])
        index = np.asarray(unit_grp[idx_key])
        trains.extend(
            _trains_from_flat_index(flat.astype(float), index, unit="s", fs_Hz=None)
        )

        unit_ids = (
            np.asarray(unit_grp["id"]) if "id" in unit_grp else range(len(trains))
        )

        electrode_indices = None
        if "electrodes" in unit_grp and "electrodes_index" in unit_grp:
            elec_flat = np.asarray(unit_grp["electrodes"])
            elec_idx = np.asarray(unit_grp["electrodes_index"])
            electrode_indices = []
            start = 0
            for stop in elec_idx:
                electrode_indices.append(elec_flat[start:stop])
                start = stop

        for i, uid in enumerate(unit_ids):
            attr = {"unit_id": int(uid)}
            if electrode_indices and len(electrode_indices[i]) > 0:
                attr["channel"] = int(electrode_indices[i][0])
            neuron_attributes.append(attr)

    return _build_spikedata(
        trains, length_ms=length_ms, metadata=meta, neuron_attributes=neuron_attributes
    )


# ----------------------------
# SpikeInterface
# ----------------------------


def load_spikedata_from_spikeinterface(
    sorting,
    *,
    sampling_frequency: Optional[float] = None,
    unit_ids: Optional[Sequence[Union[int, str]]] = None,
    segment_index: int = 0,
) -> SpikeData:
    """Convert a SpikeInterface SortingExtractor-like object to SpikeData.

    Parameters:
        sorting (object): Exposes get_unit_ids(), get_sampling_frequency(), get_unit_spike_train(...).
        sampling_frequency (float, optional): Optional override for sampling frequency (Hz).
        unit_ids (sequence, optional): Optional subset of unit IDs to include.
        segment_index (int): Segment index for multi-segment sortings.

    Returns: sd (SpikeData): The converted spike train data.
    """
    try:
        get_unit_ids = sorting.get_unit_ids  # type: ignore[attr-defined]
        get_sf = sorting.get_sampling_frequency  # type: ignore[attr-defined]
        get_train = sorting.get_unit_spike_train  # type: ignore[attr-defined]
    except Exception as e:
        raise TypeError(
            "`sorting` must be a SpikeInterface SortingExtractor-like object"
        ) from e

    fs = sampling_frequency or float(get_sf())
    if not fs or fs <= 0:
        raise ValueError("A positive sampling_frequency (Hz) is required")

    ids = list(unit_ids) if unit_ids is not None else list(get_unit_ids())
    trains: List[np.ndarray] = []
    neuron_attributes: List[dict] = []

    channel_prop = None
    if hasattr(sorting, "get_property"):
        for prop_name in ("channel", "ch", "peak_channel"):
            try:
                channel_prop = sorting.get_property(prop_name)
                break
            except Exception:
                pass

    for i, uid in enumerate(ids):
        st = np.asarray(get_train(unit_id=uid, segment_index=segment_index))
        trains.append(_to_ms(st.astype(float), "samples", fs))
        attr = {"unit_id": uid}
        if channel_prop is not None:
            attr["channel"] = int(channel_prop[i])
        neuron_attributes.append(attr)

    meta = {"source_format": "SpikeInterface", "unit_ids": ids, "fs_Hz": fs}
    return _build_spikedata(trains, metadata=meta, neuron_attributes=neuron_attributes)


# ----------------------------
# KiloSort / Phy
# ----------------------------


def load_spikedata_from_kilosort(
    folder: str,
    *,
    fs_Hz: float,
    spike_times_file: str = "spike_times.npy",
    spike_clusters_file: str = "spike_clusters.npy",
    cluster_info_tsv: Optional[str] = None,
    time_unit: str = "samples",
    include_noise: bool = False,
    length_ms: Optional[float] = None,
    channel_map_file: str = "channel_map.npy",
) -> SpikeData:
    """
    Load KiloSort/Phy outputs into SpikeData.

    Parameters:
        folder (str): Path to the KiloSort/Phy output directory.
        fs_Hz (float): Sampling frequency in Hz.
        spike_times_file (str): Path to the spike_times.npy file.
        spike_clusters_file (str): Path to the spike_clusters.npy file.
        cluster_info_tsv (str, optional): Path to the cluster info TSV file.
        time_unit (str): Unit of the spike times ('samples', 's', or 'ms').
        include_noise (bool): If True, include noise clusters.
        length_ms (float, optional): Recording duration in milliseconds.
        channel_map_file (str): Path to channel_map.npy for channel info.
    Returns: sd (SpikeData): The loaded spike train data.

    Note:
    - This loader does not extract or include waveform data; only spike times and cluster assignments are loaded.
    - Reads spike_times.npy (samples) and spike_clusters.npy; groups times per cluster
    and converts to ms using fs_Hz.
    """
    st_path = os.path.join(folder, spike_times_file)
    sc_path = os.path.join(folder, spike_clusters_file)
    spike_times = np.load(st_path)
    spike_clusters = np.load(sc_path)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times and spike_clusters length mismatch")

    channel_map: Optional[np.ndarray] = None
    cm_path = os.path.join(folder, channel_map_file)
    if os.path.exists(cm_path):
        try:
            channel_map = np.load(cm_path).flatten()
        except Exception as e:
            warnings.warn(f"Failed loading channel_map: {e}")

    keep_clusters: Optional[set] = None
    if cluster_info_tsv is not None:
        tsv_path = os.path.join(folder, cluster_info_tsv)
        if os.path.exists(tsv_path):
            try:
                import pandas as pd

                df = pd.read_csv(tsv_path, sep="\t")
                label_col = (
                    "group"
                    if "group" in df.columns
                    else ("KSLabel" if "KSLabel" in df.columns else None)
                )
                id_col = (
                    "cluster_id"
                    if "cluster_id" in df.columns
                    else ("id" if "id" in df.columns else None)
                )
                if id_col is None or label_col is None:
                    warnings.warn(
                        "Could not find id/label columns in cluster TSV; keeping all clusters"
                    )
                else:
                    if include_noise:
                        keep_clusters = set(df[id_col].astype(int).tolist())
                    else:
                        mask = (
                            df[label_col]
                            .astype(str)
                            .str.lower()
                            .isin(["good", "mua", "mua good"])
                        )  # permissive
                        keep_clusters = set(df.loc[mask, id_col].astype(int).tolist())
            except Exception as e:
                warnings.warn(
                    f"Failed parsing cluster info TSV: {e}; keeping all clusters"
                )

    trains: List[np.ndarray] = []
    metadata_units: List[int] = []
    neuron_attributes: List[dict] = []
    for clu in np.unique(spike_clusters):
        if keep_clusters is not None and int(clu) not in keep_clusters:
            continue
        times = spike_times[spike_clusters == clu]
        times_ms = _to_ms(times.astype(float), time_unit, fs_Hz)
        trains.append(np.sort(times_ms))
        metadata_units.append(int(clu))

        attr: dict = {"unit_id": int(clu)}
        if channel_map is not None and int(clu) < len(channel_map):
            attr["channel"] = int(channel_map[int(clu)])
        neuron_attributes.append(attr)

    meta = {
        "source_folder": os.path.abspath(folder),
        "source_format": "KiloSort",
        "cluster_ids": metadata_units,
        "fs_Hz": fs_Hz,
    }
    return _build_spikedata(
        trains, length_ms=length_ms, metadata=meta, neuron_attributes=neuron_attributes
    )


# ----------------------------
# SpikeInterface BaseRecording -> SpikeData via thresholding
# ----------------------------


def load_spikedata_from_spikeinterface_recording(
    recording,
    *,
    segment_index: int = 0,
    threshold_sigma: float = 5.0,
    filter: Union[dict, bool] = False,
    hysteresis: bool = True,
    direction: str = "both",
) -> SpikeData:
    """
    Convert a SpikeInterface BaseRecording-like object into SpikeData.

    Parameters:
        recording (object): Exposes get_traces(segment_index=..., ...), get_sampling_frequency(), get_num_channels().
        segment_index (int): Segment index for multi-segment recordings.
        threshold_sigma (float): Threshold in units of per-channel standard deviation.
        filter (dict | bool): If True, apply default Butterworth bandpass; if dict, pass to filter; if False, no filtering.
        hysteresis (bool): Use rising-edge detection if True.
        direction (str): 'both' | 'up' | 'down'.

    Returns: sd (SpikeData): The converted spike train data.
    """
    # Resolve sampling frequency
    if hasattr(recording, "get_sampling_frequency"):
        fs = float(recording.get_sampling_frequency())
    else:
        fs = float(getattr(recording, "sampling_frequency"))
    if not fs or fs <= 0:
        raise ValueError("A positive sampling_frequency (Hz) is required on recording")

    # Retrieve traces (2D array) and coerce to numpy
    traces = recording.get_traces(segment_index=segment_index)
    data = np.asarray(traces)

    # Ensure orientation is (channels, time) via robust heuristic:
    # choose the smaller dimension as channels (typical: channels << time).
    if data.ndim != 2:
        raise ValueError("recording.get_traces() must return a 2D array")
    data_ct = data if data.shape[0] <= data.shape[1] else data.T

    # Delegate detection to SpikeData convenience constructor
    return SpikeData.from_thresholding(
        data_ct,
        fs_Hz=fs,
        threshold_sigma=threshold_sigma,
        filter=filter,
        hysteresis=hysteresis,
        direction=direction,  # type: ignore[arg-type]
    )
