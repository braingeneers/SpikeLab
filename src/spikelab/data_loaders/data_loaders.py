"""
Lightweight loaders that convert common neurophysiology formats into
`spikedata.SpikeData` objects.

Supported inputs (best-effort, optional deps):
    - HDF5 (generic): spike times, (indices,times), or raster matrices
    - NWB: reads Units table spike_times (via pynwb if available, else h5py)
    - KiloSort/Phy outputs: spike_times.npy + spike_clusters.npy (+ optional TSV)
    - SpikeInterface: from a SortingExtractor
    - IBL (International Brain Laboratory): via ONE API + brainwidemap

Times are converted to milliseconds to match `SpikeData` conventions.
These helpers avoid hard dependencies: optional libraries are imported lazily.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Union

import os
import warnings

import numpy as np

import pickle

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

from ..spikedata import SpikeData

__all__ = [
    "load_spikedata_from_hdf5",
    "load_spikedata_from_hdf5_raw_thresholded",
    "load_spikedata_from_nwb",
    "load_spikedata_from_kilosort",
    "load_spikedata_from_spikeinterface",
    "load_spikedata_from_spikeinterface_recording",
    "load_spikedata_from_pickle",
    "load_spikedata_from_ibl",
    "query_ibl_probes",
    "load_spikedata_from_spikelab_sorted_npz",
]

from ..spikedata.utils import ensure_h5py, to_ms


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
        trains.append(to_ms(segment, unit, fs_Hz))
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
            start_time=sd.start_time,
            metadata=sd.metadata,
            raw_data=raw_data,
            raw_time=raw_time,
            neuron_attributes=sd.neuron_attributes,
        )
    if (raw_data is None) != (raw_time is None):
        present = "raw_data" if raw_data is not None else "raw_time"
        missing = "raw_time" if raw_data is not None else "raw_data"
        warnings.warn(
            f"{present} was provided but {missing} is None — "
            f"raw data will not be attached to the SpikeData.",
            UserWarning,
        )
    return sd


def _build_spikedata(
    trains_ms: List[np.ndarray],
    *,
    length_ms: Optional[float] = None,
    start_time: float = 0.0,
    metadata: Optional[Mapping[str, object]] = None,
    raw_data: Optional[np.ndarray] = None,
    raw_time: Optional[Union[np.ndarray, float]] = None,
    neuron_attributes: Optional[List[dict]] = None,
) -> SpikeData:
    """Internal helper to construct a SpikeData with sensible defaults. Infers `length_ms` from the last spike if not provided."""
    if length_ms is None:
        last = [t[-1] for t in trains_ms if len(t) > 0]
        length_ms = float(max(last)) - start_time if last else 0.0
    return SpikeData(
        trains_ms,
        length=length_ms,
        start_time=start_time,
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
    """Load spike trains from a generic HDF5 file using one of four supported input styles.

    Exactly one input style must be specified. The four styles are: raster
    matrix, ragged arrays, group-per-unit, and paired arrays.

    Parameters:
        filepath (str): Path to the HDF5 file.
        raster_dataset (str | None): Dataset path for a 2D raster/counts matrix
            (units x time). Activates raster style.
        raster_bin_size_ms (float | None): Bin width in milliseconds. Required
            for raster style.
        spike_times_dataset (str | None): Dataset path for flat concatenated
            spike times. Activates ragged style (requires
            spike_times_index_dataset).
        spike_times_index_dataset (str | None): Dataset path for cumulative
            end-of-unit indices into the flat spike times array.
        spike_times_unit (str): Time unit for ragged spike times
            ('s', 'ms', or 'samples').
        fs_Hz (float | None): Sampling frequency in Hz. Required when any
            time unit is 'samples'.
        group_per_unit (str | None): HDF5 group path containing one dataset
            per unit. Activates group-per-unit style.
        group_time_unit (str): Time unit for group-per-unit datasets
            ('s', 'ms', or 'samples').
        idces_dataset (str | None): Dataset path for unit index array.
            Activates paired-arrays style (requires times_dataset).
        times_dataset (str | None): Dataset path for spike times array
            (paired with idces_dataset).
        times_unit (str): Time unit for paired spike times
            ('s', 'ms', or 'samples').
        raw_dataset (str | None): Dataset path for optional raw analog data.
        raw_time_dataset (str | None): Dataset path for the raw data time
            vector.
        raw_time_unit (str): Time unit for the raw time vector
            ('s', 'ms', or 'samples').
        length_ms (float | None): Recording duration in milliseconds. If not
            provided, inferred from the latest spike time.
        metadata (Mapping | None): Additional metadata to attach to the
            resulting SpikeData.

    Returns:
        sd (SpikeData): The loaded spike train data.

    Raises:
        ValueError: If not exactly one input style is specified, or if
            required arguments are missing.
    """
    ensure_h5py()

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
        # Read start_time if stored (backward compatible default 0.0)
        file_start_time = float(f.attrs.get("start_time", 0.0))

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
            sd = SpikeData.from_raster(
                raster, raster_bin_size_ms, length=length_ms, start_time=file_start_time
            )
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
                start_time=file_start_time,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        if group_per_unit is not None:
            # Style (3): each child dataset is a unit's spike times
            grp = f[group_per_unit]
            keys = sorted(list(grp.keys()))
            trains = [to_ms(np.asarray(grp[k]), group_time_unit, fs_Hz) for k in keys]
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                start_time=file_start_time,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        # Style (4): paired indices and times arrays
        idces = np.asarray(f[idces_dataset])  # type: ignore
        times = to_ms(np.asarray(f[times_dataset]), times_unit, fs_Hz)  # type: ignore
        N = int(idces.max()) + 1 if idces.size else 0
        sd = SpikeData.from_idces_times(
            idces, times, N=N, length=length_ms, start_time=file_start_time
        )
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
    """Threshold-and-detect spikes from an HDF5 dataset of raw traces.

    Parameters:
        filepath (str): Path to HDF5 file.
        dataset (str): HDF5 dataset path containing raw traces shaped
            (channels, time).
        fs_Hz (float): Sampling frequency in Hz.
        threshold_sigma (float): Threshold in units of per-channel standard deviation.
        filter (dict | bool): If True, apply default Butterworth bandpass;
            if dict, pass to filter; if False, no filtering.
        hysteresis (bool): Use rising-edge detection if True.
        direction (str): 'both' | 'up' | 'down'.

    Returns:
        sd (SpikeData): The detected spike train data.
    """
    ensure_h5py()
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
    """Load spike trains from an NWB file's Units table.

    Parameters:
        filepath (str): Path to the NWB file.
        prefer_pynwb (bool): If True, try pynwb first; if False, try h5py.
        length_ms (float | None): Recording duration in milliseconds.

    Returns:
        sd (SpikeData): The loaded spike train data.
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

                electrode_positions: Optional[dict] = None
                if getattr(nwb, "electrodes", None) is not None:
                    elec_df = nwb.electrodes.to_dataframe()
                    electrode_positions = {}
                    for elec_row in elec_df.itertuples():
                        pos = []
                        for coord in ("x", "y", "z"):
                            if coord in elec_df.columns:
                                val = getattr(elec_row, coord, None)
                                if val is not None and not np.isnan(val):
                                    pos.append(float(val))
                        if pos:
                            electrode_positions[elec_row.Index] = pos

                for row in df.itertuples():
                    stimes = np.asarray(row.spike_times, dtype=float)
                    trains.append(stimes * 1e3)
                    attr = {"unit_id": row.Index}
                    electrode_id = None
                    for col in ("electrodes", "electrode_group", "channel", "ch"):
                        if col in df.columns:
                            val = getattr(row, col, None)
                            if val is not None:
                                if (
                                    hasattr(val, "__len__")
                                    and not isinstance(val, str)
                                    and len(val) > 0
                                ):
                                    channel_val = val[0]
                                else:
                                    channel_val = val
                                try:
                                    attr["electrode"] = int(channel_val)
                                    electrode_id = int(channel_val)
                                except (TypeError, ValueError):
                                    attr["electrode"] = channel_val
                                    electrode_id = channel_val
                                break
                    if electrode_positions and electrode_id in electrode_positions:
                        attr["location"] = electrode_positions[electrode_id]
                    neuron_attributes.append(attr)
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                neuron_attributes=neuron_attributes,
            )
        except ImportError:  # pragma: no cover
            pass  # pynwb not installed — fall back to h5py
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
        ) as e:  # pragma: no cover
            warnings.warn(
                f"pynwb failed to load NWB file ({type(e).__name__}: {e}); "
                f"falling back to h5py. If this is unexpected, check the file "
                f"format or report a bug.",
                stacklevel=2,
            )
            trains.clear()
            neuron_attributes.clear()

    ensure_h5py()
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
            if len(elec_idx) > 0 and elec_idx[-1] > len(elec_flat):
                warnings.warn(
                    "NWB electrodes_index exceeds electrodes array length; "
                    "electrode data may be truncated.",
                    UserWarning,
                )
            electrode_indices = []
            start = 0
            for stop in elec_idx:
                electrode_indices.append(elec_flat[start:stop])
                start = stop

        electrode_positions: Optional[dict] = None
        elec_table_path = "general/extracellular_ephys/electrodes"
        if elec_table_path in f:
            elec_grp = f[elec_table_path]
            electrode_positions = {}
            x_arr = np.asarray(elec_grp["x"]) if "x" in elec_grp else None
            y_arr = np.asarray(elec_grp["y"]) if "y" in elec_grp else None
            z_arr = np.asarray(elec_grp["z"]) if "z" in elec_grp else None
            elec_ids = (
                np.asarray(elec_grp["id"])
                if "id" in elec_grp
                else np.arange(len(x_arr) if x_arr is not None else 0)
            )
            for idx, eid in enumerate(elec_ids):
                pos = []
                if x_arr is not None and idx < len(x_arr):
                    pos.append(float(x_arr[idx]))
                if y_arr is not None and idx < len(y_arr):
                    pos.append(float(y_arr[idx]))
                if z_arr is not None and idx < len(z_arr):
                    pos.append(float(z_arr[idx]))
                if pos:
                    electrode_positions[int(eid)] = pos

        for i, uid in enumerate(unit_ids):
            attr = {"unit_id": int(uid)}
            electrode_id = None
            if (
                electrode_indices
                and i < len(electrode_indices)
                and len(electrode_indices[i]) > 0
            ):
                electrode_id = int(electrode_indices[i][0])
                attr["electrode"] = electrode_id
            if electrode_positions and electrode_id in electrode_positions:
                attr["location"] = electrode_positions[electrode_id]
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
        sorting (object): Exposes get_unit_ids(),
            get_sampling_frequency(), get_unit_spike_train(...).
        sampling_frequency (float | None): Optional override for sampling
            frequency (Hz).
        unit_ids (Sequence | None): Optional subset of unit IDs to include.
        segment_index (int): Segment index for multi-segment sortings.

    Returns:
        sd (SpikeData): The converted spike train data.
    """
    try:
        get_unit_ids = sorting.get_unit_ids  # type: ignore[attr-defined]
        get_sf = sorting.get_sampling_frequency  # type: ignore[attr-defined]
        get_train = sorting.get_unit_spike_train  # type: ignore[attr-defined]
    except AttributeError as e:
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
    location_prop = None
    if hasattr(sorting, "get_property"):
        for prop_name in ("channel", "ch", "peak_channel", "electrode"):
            try:
                channel_prop = sorting.get_property(prop_name)
            except (AttributeError, KeyError):
                continue
            if channel_prop is not None:
                break
        for prop_name in ("location", "unit_location", "position"):
            try:
                location_prop = sorting.get_property(prop_name)
            except (AttributeError, KeyError):
                continue
            if location_prop is not None:
                break

    for i, uid in enumerate(ids):
        st = np.asarray(get_train(unit_id=uid, segment_index=segment_index))
        trains.append(to_ms(st.astype(float), "samples", fs))
        attr = {"unit_id": uid}
        if channel_prop is not None and i < len(channel_prop):
            attr["electrode"] = int(channel_prop[i])
        if location_prop is not None and i < len(location_prop):
            loc = location_prop[i]
            if loc is not None:
                attr["location"] = list(loc) if hasattr(loc, "__iter__") else [loc]
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
    channel_positions_file: str = "channel_positions.npy",
) -> SpikeData:
    """Load KiloSort/Phy outputs into SpikeData.

    Parameters:
        folder (str): Path to the KiloSort/Phy output directory.
        fs_Hz (float): Sampling frequency in Hz.
        spike_times_file (str): Path to the spike_times.npy file.
        spike_clusters_file (str): Path to the spike_clusters.npy file.
        cluster_info_tsv (str | None): Path to the cluster info TSV file.
        time_unit (str): Unit of the spike times ('samples', 's', or 'ms').
        include_noise (bool): If True, include noise clusters.
        length_ms (float | None): Recording duration in milliseconds.
        channel_map_file (str): Filename of the channel map file relative
            to folder. Expected format: 1D numpy array mapping cluster
            indices to channel numbers.
        channel_positions_file (str): Filename of the channel positions
            file relative to folder. Expected format: 2D numpy array of
            shape (channels, 3) containing channel positions.

    Returns:
        sd (SpikeData): The loaded spike train data.

    Notes:
        - This loader does not extract or include waveform data; only
          spike times and cluster assignments are loaded.
        - Reads spike_times.npy (samples) and spike_clusters.npy; groups
          times per cluster and converts to ms using fs_Hz.
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
        except (IOError, ValueError) as e:
            warnings.warn(f"Failed loading channel_map: {e}")

    channel_positions: Optional[np.ndarray] = None
    cp_path = os.path.join(folder, channel_positions_file)
    if os.path.exists(cp_path):
        try:
            channel_positions = np.load(cp_path)
        except (IOError, ValueError) as e:
            warnings.warn(f"Failed loading channel_positions: {e}")

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
            except ImportError:
                warnings.warn(
                    "pandas is required to parse cluster info TSV. "
                    "Install with: pip install spikelab[io]. "
                    "Keeping all clusters."
                )
            except (IOError, ValueError, KeyError) as e:
                warnings.warn(
                    f"Failed parsing cluster info TSV: {e}; keeping all clusters"
                )

    trains: List[np.ndarray] = []
    metadata_units: List[int] = []
    neuron_attributes: List[dict] = []
    unit_idx = 0
    for clu in np.unique(spike_clusters):
        if keep_clusters is not None and int(clu) not in keep_clusters:
            continue
        times = spike_times[spike_clusters == clu]
        times_ms = to_ms(times.astype(float), time_unit, fs_Hz)
        trains.append(np.sort(times_ms))
        metadata_units.append(int(clu))

        attr: dict = {"unit_id": int(clu)}
        channel_idx = None
        int_clu = int(clu)
        # channel_map is indexed by cluster ID — only correct when cluster
        # IDs are sequential integers starting from 0.
        if channel_map is not None and int_clu < len(channel_map):
            channel_idx = int(channel_map[int_clu])
            attr["electrode"] = channel_idx

        if channel_positions is not None:
            if channel_idx is not None and channel_idx < len(channel_positions):
                attr["location"] = list(channel_positions[channel_idx])
            elif unit_idx < len(channel_positions):
                # Fallback: use unit index when channel map lookup fails
                attr["location"] = list(channel_positions[unit_idx])
        neuron_attributes.append(attr)
        unit_idx += 1

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
# SpikeLab sorted .npz -> SpikeData
# ----------------------------


def load_spikedata_from_spikelab_sorted_npz(
    filepath: str,
    *,
    length_ms: Optional[float] = None,
) -> SpikeData:
    """Load a SpikeLab compiled sorting result (``.npz``) into SpikeData.

    These ``.npz`` files are produced by :func:`sort_with_kilosort2`'s
    ``compile_results`` step and contain per-unit spike trains, electrode
    locations, waveform templates, and quality metrics.

    Parameters:
        filepath (str): Path to the ``.npz`` file.
        length_ms (float | None): Recording duration in milliseconds.
            Inferred from the latest spike time when *None*.

    Returns:
        sd (SpikeData): The loaded spike train data with neuron attributes
            (unit_id, location, electrode, template, amplitudes, etc.).
    """
    data = np.load(filepath, allow_pickle=True)

    units = data["units"]
    fs_Hz = float(data["fs"])
    locations = data.get("locations", None)

    trains: List[np.ndarray] = []
    neuron_attributes: List[dict] = []

    for unit in units:
        spike_samples = unit["spike_train"]
        spike_times_ms = np.sort(spike_samples.astype(float) / fs_Hz * 1000.0)
        trains.append(spike_times_ms)

        attr: dict = {"unit_id": int(unit["unit_id"])}
        if "x_max" in unit and "y_max" in unit:
            attr["location"] = [float(unit["x_max"]), float(unit["y_max"])]
        if "electrode" in unit:
            attr["electrode"] = int(unit["electrode"])
        if "template" in unit:
            attr["template"] = np.asarray(unit["template"])
        if "amplitudes" in unit:
            attr["amplitudes"] = np.asarray(unit["amplitudes"])
        if "std_norms" in unit:
            attr["std_norms"] = np.asarray(unit["std_norms"])
        if "peak_sign" in unit:
            attr["peak_sign"] = str(unit["peak_sign"])
        if "max_channel_id" in unit:
            attr["max_channel_id"] = str(unit["max_channel_id"])
        neuron_attributes.append(attr)

    meta = {
        "source_file": os.path.abspath(filepath),
        "source_format": "SpikeLab_npz",
        "fs_Hz": fs_Hz,
    }
    if locations is not None:
        meta["channel_locations"] = locations

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
    """Convert a SpikeInterface BaseRecording-like object into SpikeData.

    Parameters:
        recording (object): Exposes get_traces(segment_index=...),
            get_sampling_frequency(), get_num_channels().
        segment_index (int): Segment index for multi-segment recordings.
        threshold_sigma (float): Threshold in units of per-channel standard deviation.
        filter (dict | bool): If True, apply default Butterworth bandpass;
            if dict, pass to filter; if False, no filtering.
        hysteresis (bool): Use rising-edge detection if True.
        direction (str): 'both' | 'up' | 'down'.

    Returns:
        sd (SpikeData): The converted spike train data.
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
    if data.shape[0] == data.shape[1]:
        warnings.warn(
            f"Ambiguous data orientation: shape is {data.shape} (square). "
            "Assuming (channels, time). Pass data with an explicit orientation "
            "if this is incorrect.",
            UserWarning,
        )
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


# ----------------------------
# Pickle
# ----------------------------


def load_spikedata_from_pickle(
    filepath: str,
    *,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> SpikeData:
    """Load a SpikeData object from a pickle file.

    Warning:
        Only load pickle files from trusted sources. Pickle
        deserialization can execute arbitrary code and should never be
        used with untrusted data.

    Parameters:
        filepath (str): Path to the pickle file, or an S3 URL
            (s3://bucket/key).
        aws_access_key_id (str | None): AWS access key ID for S3
            downloads.
        aws_secret_access_key (str | None): AWS secret access key for
            S3 downloads.
        aws_session_token (str | None): AWS session token for temporary
            credentials.
        region_name (str | None): AWS region name for S3 access.

    Returns:
        sd (SpikeData): The deserialized SpikeData object.
    """
    from .s3_utils import ensure_local_file

    local_path, is_temp = ensure_local_file(
        filepath,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    try:
        with open(local_path, "rb") as f:
            obj = pickle.load(f)
    finally:
        if is_temp:
            try:
                os.remove(local_path)
            except OSError:
                pass

    if not isinstance(obj, SpikeData):
        raise ValueError(
            f"Pickle file does not contain a SpikeData object (found {type(obj).__name__})"
        )

    return obj


# ----------------------------
# IBL (International Brain Laboratory)
# ----------------------------

#: IBL public server URL used for ONE authentication.
_IBL_BASE_URL = "https://openalyx.internationalbrainlab.org"

#: Collections searched in order when loading spikes. The probe-specific
#: collection is prepended at runtime based on the PID suffix.
_IBL_FALLBACK_COLLECTIONS = [
    "alf/probe00/pykilosort",
    "alf/probe01/pykilosort",
    "alf",
]


def load_spikedata_from_ibl(
    eid: str,
    pid: str,
    *,
    length_ms: Optional[float] = None,
) -> SpikeData:
    """Load spike trains for a single IBL probe into SpikeData.

    Authenticates against the public IBL server automatically. Only
    units labelled as good (``label == 1``) in the Brain-Wide Map unit
    table are included. Trial event times are stored in
    ``SpikeData.metadata`` as individual numpy arrays, all in
    milliseconds.

    Parameters:
        eid (str): IBL experiment ID (UUID string).
        pid (str): IBL probe ID (UUID string).
        length_ms (float | None): Recording duration in milliseconds.
            If not provided, the maximum spike time across all units is used.

    Returns:
        sd (SpikeData): Loaded spike train data.
            ``neuron_attributes`` contains
            ``{"region": <Beryl atlas region>}`` per unit.
            ``metadata`` contains ``eid``, ``pid``, ``n_trials``,
            ``trial_start_times``, ``trial_end_times``,
            ``stim_on_times``, ``stim_off_times``, ``go_cue_times``,
            ``response_times``, ``feedback_times``,
            ``first_movement_times``, ``choice``, ``feedback_type``,
            ``contrast_left``, ``contrast_right``, and
            ``probability_left``. All time arrays are in milliseconds.

    Notes:
        - Requires ``one-api`` and ``brainwidemap`` packages (optional dependencies).
        - Spike times are converted from seconds (IBL convention) to milliseconds.
        - Trial times are converted from seconds to milliseconds.
        - Probe collection is inferred from the PID suffix; falls back through
          ``alf/probe00/pykilosort``, ``alf/probe01/pykilosort``, and ``alf``.
    """
    try:
        from one.api import ONE  # type: ignore
    except ImportError as e:
        raise ImportError(
            "one-api is required for load_spikedata_from_ibl. "
            "Install with: pip install one-api"
        ) from e

    try:
        from brainwidemap import bwm_units  # type: ignore
    except ImportError as e:
        raise ImportError(
            "brainwidemap is required for load_spikedata_from_ibl. "
            "Install with: pip install brainwidemap"
        ) from e

    # Authenticate against the public IBL server.
    ONE.setup(base_url=_IBL_BASE_URL, silent=True)
    one = ONE(password="international")

    # Retrieve good units for this probe from the Brain-Wide Map table.
    unit_df = bwm_units(one)
    good_units = unit_df[(unit_df["pid"] == pid) & (unit_df["label"] == 1)]

    # Build the ordered list of collections to try, with the probe-specific
    # one first when the PID suffix hints at the probe number. This is a
    # best-effort heuristic for ordering (PIDs are UUIDs, so the last two
    # hex chars can coincidentally match "00"/"01"). All candidates are
    # tried regardless, so correctness is not affected — only the order.
    collections = []
    if pid.endswith("00") or pid.endswith("01"):
        collections.append(f"alf/probe{pid[-2:]}/pykilosort")
    collections.extend(_IBL_FALLBACK_COLLECTIONS)
    # Deduplicate while preserving order.
    seen: set = set()
    ordered_collections: List[str] = []
    for c in collections:
        if c not in seen:
            seen.add(c)
            ordered_collections.append(c)

    # Load spikes from the first available collection.
    spikes = None
    for collection in ordered_collections:
        try:
            spikes = one.load_object(eid, "spikes", collection=collection)
            break
        except (ValueError, KeyError, FileNotFoundError):
            continue

    # Build per-unit spike trains (seconds → milliseconds).
    spike_trains: List[np.ndarray] = []
    neuron_attributes: List[dict] = []
    for _, unit in good_units.iterrows():
        if spikes is None:
            spike_trains.append(np.array([], dtype=float))
        else:
            mask = spikes["clusters"] == unit["cluster_id"]
            spike_trains.append(spikes["times"][mask] * 1_000.0)
        neuron_attributes.append({"region": unit["Beryl"]})

    # Infer session length from the largest spike time if not provided.
    if length_ms is None:
        max_t = max((t.max() for t in spike_trains if len(t) > 0), default=0.0)
        length_ms = float(max_t) if max_t > 0 else 10_000.0

    # Load trials and extract relevant fields as numpy arrays (seconds → ms).
    trials = one.load_object(eid, "trials")
    trials_df = trials.to_df()
    n_trials = len(trials_df)

    def _to_ms_array(col: str) -> np.ndarray:
        """Extract a trials column and convert seconds to milliseconds."""
        return trials_df[col].to_numpy(dtype=float) * 1_000.0

    def _to_array(col: str) -> np.ndarray:
        """Extract a trials column as a plain numpy array (no unit conversion)."""
        return trials_df[col].to_numpy(dtype=float)

    metadata: dict = {
        "eid": eid,
        "pid": pid,
        "n_trials": n_trials,
        "trial_start_times": _to_ms_array("intervals_0"),
        "trial_end_times": _to_ms_array("intervals_1"),
        "stim_on_times": _to_ms_array("stimOn_times"),
        "stim_off_times": _to_ms_array("stimOff_times"),
        "go_cue_times": _to_ms_array("goCue_times"),
        "response_times": _to_ms_array("response_times"),
        "feedback_times": _to_ms_array("feedback_times"),
        "first_movement_times": _to_ms_array("firstMovement_times"),
        "choice": _to_array("choice"),
        "feedback_type": _to_array("feedbackType"),
        "contrast_left": _to_array("contrastLeft"),
        "contrast_right": _to_array("contrastRight"),
        "probability_left": _to_array("probabilityLeft"),
    }

    return _build_spikedata(
        spike_trains,
        length_ms=length_ms,
        metadata=metadata,
        neuron_attributes=neuron_attributes,
    )


def query_ibl_probes(
    target_regions: Optional[List[str]] = None,
    *,
    min_units: int = 0,
    min_fraction_in_target: float = 0.0,
) -> "tuple[list[tuple[str, str]], pd.DataFrame]":
    """Search the IBL Brain-Wide Map database for probes matching given criteria.

    Authenticates against the public IBL server automatically. Filters
    probes by brain region and unit count. Returns matching (eid, pid)
    pairs alongside a per-probe statistics DataFrame.

    Parameters:
        target_regions (list[str] | None): Beryl atlas region names to
            filter by (e.g. ``["MOs", "MOp"]``). If None, no region
            filter is applied.
        min_units (int): Minimum number of good units required per probe.
            Default ``0`` (no minimum).
        min_fraction_in_target (float): Minimum fraction (0–1) of good units
            that must fall within ``target_regions``. Ignored when
            ``target_regions`` is ``None``. Default ``0.0``.

    Returns:
        probes (list[tuple[str, str]]): List of ``(eid, pid)`` pairs for
            probes that pass all filters, sorted by descending good unit count.
        stats (pd.DataFrame): One row per matching probe with columns:
            ``eid``, ``pid``, ``n_good_units``, and (when ``target_regions``
            is not ``None``) ``n_in_target`` and ``fraction_in_target``.

    Notes:
        - Requires ``one-api`` and ``brainwidemap`` packages (optional
          dependencies).
        - ``bwm_units()`` fetches the full Brain-Wide Map unit table from the
          IBL server; this may take several seconds on first call.
    """
    try:
        from one.api import ONE  # type: ignore
    except ImportError as e:
        raise ImportError(
            "one-api is required for query_ibl_probes. "
            "Install with: pip install one-api"
        ) from e

    try:
        from brainwidemap import bwm_units  # type: ignore
    except ImportError as e:
        raise ImportError(
            "brainwidemap is required for query_ibl_probes. "
            "Install with: pip install brainwidemap"
        ) from e

    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pandas is required for query_ibl_probes. "
            "Install with: pip install spikelab[io]"
        ) from e

    # Authenticate against the public IBL server.
    ONE.setup(base_url=_IBL_BASE_URL, silent=True)
    one = ONE(password="international")

    # Fetch all good units from the Brain-Wide Map table.
    unit_df = bwm_units(one)
    good_units = unit_df[unit_df["label"] == 1].copy()

    # Build per-probe aggregation.
    agg = good_units.groupby(["eid", "pid"], as_index=False).agg(
        n_good_units=("cluster_id", "count"),
    )

    # Compute region-based columns when target_regions is provided.
    if target_regions is not None:
        in_target = good_units["Beryl"].isin(target_regions)
        region_counts = (
            good_units[in_target]
            .groupby(["eid", "pid"], as_index=False)
            .agg(n_in_target=("cluster_id", "count"))
        )
        agg = agg.merge(region_counts, on=["eid", "pid"], how="left")
        agg["n_in_target"] = agg["n_in_target"].fillna(0).astype(int)
        agg["fraction_in_target"] = np.where(
            agg["n_good_units"] > 0,
            agg["n_in_target"] / agg["n_good_units"],
            0.0,
        )

    # Apply unit-count filter.
    mask = agg["n_good_units"] >= min_units

    # Apply region fraction filter.
    if target_regions is not None:
        mask = mask & (agg["fraction_in_target"] >= min_fraction_in_target)

    stats = (
        agg[mask].sort_values("n_good_units", ascending=False).reset_index(drop=True)
    )

    probes = list(zip(stats["eid"].tolist(), stats["pid"].tolist()))
    return probes, stats
