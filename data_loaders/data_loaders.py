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
]


def _ensure_h5py():
    if h5py is None:
        raise ImportError("h5py is required for HDF5/NWB loaders. `pip install h5py`.")


def _to_ms(values: np.ndarray, unit: str, fs_Hz: Optional[float]) -> np.ndarray:
    if unit == "ms":
        return values.astype(float)
    if unit == "s":
        return values.astype(float) * 1e3
    if unit == "samples":
        if not fs_Hz or fs_Hz <= 0:
            raise ValueError("fs_Hz must be provided and > 0 when unit='samples'")
        return values.astype(float) / fs_Hz * 1e3
    raise ValueError(f"Unknown time unit '{unit}' (expected 's','ms','samples')")


def _build_spikedata(
    trains_ms: List[np.ndarray],
    *,
    length_ms: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
    raw_data: Optional[np.ndarray] = None,
    raw_time: Optional[Union[np.ndarray, float]] = None,
) -> SpikeData:
    if length_ms is None:
        last = [t[-1] for t in trains_ms if len(t) > 0]
        length_ms = float(max(last)) if last else 0.0
    return SpikeData(
        trains_ms,
        length=length_ms,
        metadata=dict(metadata) if metadata else {},
        raw_data=raw_data,
        raw_time=raw_time,
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
    _ensure_h5py()

    provided = [
        raster_dataset is not None,
        spike_times_dataset is not None and spike_times_index_dataset is not None,
        group_per_unit is not None,
        idces_dataset is not None and times_dataset is not None,
    ]
    if sum(provided) != 1:
        raise ValueError("Specify exactly one HDF5 input style")

    meta = dict(metadata or {})
    meta.setdefault("source_file", os.path.abspath(filepath))

    with h5py.File(filepath, "r") as f:  # type: ignore
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

        if raster_dataset is not None:
            if raster_bin_size_ms is None:
                raise ValueError("raster_bin_size_ms is required for raster_dataset")
            raster = np.asarray(f[raster_dataset])
            if raster.ndim != 2:
                raise ValueError("raster_dataset must be 2D (units, time)")
            sd = SpikeData.from_raster(raster, raster_bin_size_ms)
            sd.metadata.update(meta)
            if raw_data is not None and raw_time is not None:
                return _build_spikedata(
                    sd.train,
                    length_ms=sd.length,
                    metadata=sd.metadata,
                    raw_data=raw_data,
                    raw_time=raw_time,
                )
            return sd

        if spike_times_dataset is not None and spike_times_index_dataset is not None:
            flat = np.asarray(f[spike_times_dataset])
            index = np.asarray(f[spike_times_index_dataset])
            trains: List[np.ndarray] = []
            start = 0
            for stop in index:
                seg = flat[start:stop]
                trains.append(_to_ms(seg, spike_times_unit, fs_Hz))
                start = stop
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        if group_per_unit is not None:
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

        idces = np.asarray(f[idces_dataset])  # type: ignore
        times = _to_ms(np.asarray(f[times_dataset]), times_unit, fs_Hz)  # type: ignore
        N = int(idces.max()) + 1 if idces.size else 0
        sd = SpikeData.from_idces_times(idces, times, N=N, length=length_ms)
        sd.metadata.update(meta)
        if raw_data is not None and raw_time is not None:
            return _build_spikedata(
                sd.train,
                length_ms=sd.length,
                metadata=sd.metadata,
                raw_data=raw_data,
                raw_time=raw_time,
            )
        return sd


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
    trains: List[np.ndarray] = []
    meta = {"source_file": os.path.abspath(filepath), "format": "NWB"}

    if prefer_pynwb:
        try:
            from pynwb import NWBHDF5IO  # type: ignore

            with NWBHDF5IO(filepath, "r") as io:
                nwb = io.read()
                if getattr(nwb, "units", None) is None:
                    raise ValueError("NWB file has no Units table")
                for row in nwb.units.to_dataframe().itertuples():  # type: ignore
                    stimes = np.asarray(row.spike_times, dtype=float)
                    trains.append(stimes * 1e3)
            return _build_spikedata(trains, length_ms=length_ms, metadata=meta)
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
        start = 0
        for stop in index:
            seg = flat[start:stop]
            trains.append(seg.astype(float) * 1e3)
            start = stop
    return _build_spikedata(trains, length_ms=length_ms, metadata=meta)


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
    for uid in ids:
        st = np.asarray(get_train(unit_id=uid, segment_index=segment_index))
        trains.append(_to_ms(st.astype(float), "samples", fs))

    meta = {"source_format": "SpikeInterface", "unit_ids": ids, "fs_Hz": fs}
    return _build_spikedata(trains, metadata=meta)


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
) -> SpikeData:
    st_path = os.path.join(folder, spike_times_file)
    sc_path = os.path.join(folder, spike_clusters_file)
    spike_times = np.load(st_path)
    spike_clusters = np.load(sc_path)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times and spike_clusters length mismatch")

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
    for clu in np.unique(spike_clusters):
        if keep_clusters is not None and int(clu) not in keep_clusters:
            continue
        times = spike_times[spike_clusters == clu]
        times_ms = _to_ms(times.astype(float), time_unit, fs_Hz)
        trains.append(np.sort(times_ms))
        metadata_units.append(int(clu))

    meta = {
        "source_folder": os.path.abspath(folder),
        "source_format": "KiloSort",
        "cluster_ids": metadata_units,
        "fs_Hz": fs_Hz,
    }
    return _build_spikedata(trains, length_ms=length_ms, metadata=meta)
