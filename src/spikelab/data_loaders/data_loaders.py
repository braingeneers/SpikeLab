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

from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import json
import os
import re
import warnings

import numpy as np

import pickle

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

try:
    import pandas as pd  # noqa: F401  # used in type annotations only
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

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
    "load_spikedata_from_dandi",
    "list_dandi_assets",
    "load_recording_from_dandi",
    "load_spikedata_from_spikelab_sorted_npz",
]

from ..spikedata.utils import ensure_h5py, to_ms


def _natural_sort_key(s: str):
    """Sort key that orders embedded digit runs numerically.

    `sorted(["1", "10", "2"], key=_natural_sort_key)` returns
    `["1", "2", "10"]` instead of the lexicographic `["1", "10", "2"]`.

    Notes:
        Returns a list of ``(kind, value)`` tuples so the comparison is
        type-stable on Python 3 — mixing bare ``int`` and ``str`` tokens
        in the same list would raise ``TypeError`` when two keys
        compare a numeric token against a string token (e.g.
        ``"unit_5"`` vs ``"5_unit"``). The ``kind`` prefix (``0`` for
        numeric, ``1`` for string) puts every numeric token strictly
        less than every string token at the same position.
    """
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", s)]


def _trains_from_flat_index(
    flat_times: np.ndarray,
    end_indices: np.ndarray,
    *,
    unit: str,
    fs_Hz: Optional[float],
    n_units: Optional[int] = None,
) -> List[np.ndarray]:
    """Split a flat time array into per-unit trains using end indices and convert to ms.

    Two index conventions are accepted:

    * **Cumulative-end (length N)**: ``[c0, c0+c1, ..., total]`` —
      the convention used by SpikeLab's HDF5 ragged exporter and by
      the NWB spec. Iterated with an implicit ``start = 0`` for
      unit 0.
    * **Leading-zero cumulative (length N+1)**: ``[0, c0, c0+c1,
      ..., total]`` — common in NWB ``spike_times_index`` files in
      the wild that don't strictly follow the NWB spec.

    Disambiguation rules:

    * **With ``n_units``** (preferred — used by the NWB loader): the
      length is checked against both candidates. Length ``n_units``
      → cumulative-end. Length ``n_units + 1`` with leading 0 →
      leading-zero. A mismatch with both raises ``ValueError``.
    * **Without ``n_units``**: a heuristic auto-detect runs. The
      leading-zero variant is selected when ``len(end_indices) >= 2``,
      ``end_indices[0] == 0``, **and** ``end_indices[-1] > 0`` (i.e.
      there is at least one non-empty unit). A bare ``[0]`` or an
      all-zero array stays cumulative-end so existing
      all-empty-trains fixtures continue to round-trip correctly.
      Callers that can supply ``n_units`` should — the heuristic
      cannot disambiguate ``[0, 5]`` (two units, first empty vs one
      unit with five spikes).

    Parameters:
        flat_times (np.ndarray): Concatenated spike times.
        end_indices (np.ndarray): Cumulative end indices, in either
            of the two supported conventions.
        unit (str): Time unit of ``flat_times`` (``"ms"``, ``"s"``,
            or ``"samples"``).
        fs_Hz (float or None): Sample rate, required when
            ``unit == "samples"``.
        n_units (int or None): Known number of units, used to
            disambiguate the index convention. ``None`` triggers
            the heuristic auto-detect described above.

    Returns:
        trains (list of np.ndarray): Per-unit spike-time arrays in
            milliseconds.
    """
    segments = _split_by_index(
        flat_times, end_indices, n_units=n_units, name="spike_times_index"
    )
    return [to_ms(seg, unit, fs_Hz) for seg in segments]


def _split_by_index(
    flat: np.ndarray,
    end_indices: np.ndarray,
    *,
    n_units: Optional[int] = None,
    name: str = "index",
) -> List[np.ndarray]:
    """Split a flat array into per-unit chunks via cumulative end indices.

    Handles both NWB conventions: cumulative-end (length N) and
    leading-zero (length N+1). Disambiguation matches the rules in
    :func:`_trains_from_flat_index`: when ``n_units`` is provided, length
    is checked against both candidates; otherwise a heuristic
    auto-detect runs.

    Parameters:
        flat (np.ndarray): The concatenated array to split.
        end_indices (np.ndarray): Cumulative end indices.
        n_units (int or None): Known number of units, used to
            disambiguate the index convention. ``None`` triggers the
            heuristic auto-detect.
        name (str): Display name for error messages (e.g.
            ``"spike_times_index"`` or ``"electrodes_index"``).

    Returns:
        chunks (list of np.ndarray): Per-unit slices of ``flat``. No
            type conversion is applied.
    """
    end_indices = np.asarray(end_indices)
    # Early return for n_units=0 — an all-empty NWB file or a sorting
    # with no surviving units. The disambiguation logic below has a
    # convoluted branch for the n_units=0 + leading-zero ``[0]``
    # variant; bypassing it makes the intent obvious and avoids the
    # n_units=0 + len(end_indices)=1 special case entirely.
    if n_units == 0:
        return []
    if len(end_indices) > 0:
        # Reject float / non-integer dtype upfront with a friendly error;
        # numpy slicing on float indices raises a confusing TypeError mid-loop.
        if not np.issubdtype(end_indices.dtype, np.integer):
            raise ValueError(
                f"{name} must be an integer array, got dtype "
                f"{end_indices.dtype}. HDF5 datasets stored as float should "
                "be cast (e.g. `np.asarray(f[idx_key]).astype(np.int64)`)."
            )
        if end_indices[0] < 0:
            raise ValueError(
                f"{name} entries must be non-negative; got "
                f"{end_indices[0]} at position 0. Cumulative-end indices "
                "represent counts and cannot be negative."
            )
        if not np.all(np.diff(end_indices) >= 0):
            raise ValueError(f"{name} must be monotonically non-decreasing")
        if end_indices[-1] > len(flat):
            raise ValueError(
                f"{name} final value ({end_indices[-1]}) exceeds "
                f"flat array length ({len(flat)})"
            )

    if n_units is not None:
        if len(end_indices) == n_units + 1 and (
            len(end_indices) == 0 or end_indices[0] == 0
        ):
            # NWB leading-zero convention: strip the leading 0 to fall
            # through to cumulative-end iteration.
            end_indices = end_indices[1:]
        elif len(end_indices) != n_units:
            raise ValueError(
                f"{name} length {len(end_indices)} does not match "
                f"n_units={n_units} for either cumulative-end (length N) or "
                f"leading-zero (length N+1) convention."
            )
    elif len(end_indices) >= 2 and end_indices[0] == 0 and end_indices[-1] > 0:
        # Heuristic auto-detect: leading 0 followed by at least one
        # non-zero entry is unambiguously the leading-zero variant
        # (cumulative-end with c0=0 produces the same prefix only
        # when the array is entirely zero, which is excluded here).
        end_indices = end_indices[1:]

    chunks: List[np.ndarray] = []
    start = 0
    for stop in end_indices:
        chunks.append(flat[start:stop])
        start = stop
    return chunks


def _read_raw_arrays(
    f,
    raw_dataset: Optional[str],
    raw_time_dataset: Optional[str],
    raw_time_unit: str,
    fs_Hz: Optional[float],
) -> tuple[Optional[np.ndarray], Optional[Union[np.ndarray, float]]]:
    """Read optional raw arrays and convert the time vector to milliseconds.

    Raises:
        ValueError: If ``raw_data.shape[-1]`` does not equal
            ``raw_time.shape[0]``. The trailing axis of ``raw_data`` is
            the time axis by convention; a mismatch with the time vector
            length means the two arrays are not aligned and the resulting
            ``SpikeData`` would carry silently corrupt raw signal.
    """
    raw_data = None
    raw_time: Optional[Union[np.ndarray, float]] = None
    if raw_dataset is not None:
        raw_data = np.asarray(f[raw_dataset])
        if raw_time_dataset is not None:
            raw_time_vals = np.asarray(f[raw_time_dataset])
            # Reject shape mismatch at the loader boundary. Without this
            # the SpikeData constructor accepts the mis-aligned arrays
            # (its own suffix-shape check tolerates extra axes) and the
            # silent corruption only surfaces when downstream code indexes
            # into the wrong sample positions.
            if raw_data.shape[-1] != raw_time_vals.shape[0]:
                raise ValueError(
                    f"raw_data trailing axis length ({raw_data.shape[-1]}) "
                    f"does not match raw_time length ({raw_time_vals.shape[0]}). "
                    f"raw_data.shape={raw_data.shape}, "
                    f"raw_time.shape={raw_time_vals.shape}. The trailing axis "
                    "of raw_data is the time axis by convention."
                )
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
        if last:
            # Add one ULP at the magnitude of the latest spike so the
            # constructor's strict ``t[-1] > start_time + length`` check
            # passes even when unit-conversion round-trips (samples → s
            # → ms in the loaders) drift the loaded spike value by a
            # ULP above the inferred end. ``np.spacing(x)`` returns the
            # gap between ``x`` and the next float; at typical recording
            # scales (~1e5 ms) that's ~1.5e-11 ms — far below any
            # measurable precision but enough to keep the inequality
            # strict.
            #
            # Edge case at very short recordings: when ``max_last`` is
            # near zero (e.g. ``max_last=0.001 ms`` and ``start_time=0``),
            # the ULP scales down with it (~1e-307 ms at the smallest
            # float64 magnitudes). The subtraction ``max_last -
            # start_time`` is still exact in float arithmetic, so the
            # tiny ULP addition does not change behaviour — the comment
            # describes the dominant case (large recordings), not the
            # short-recording edge.
            max_last = float(max(last))
            length_ms = max_last - start_time + np.spacing(max_last)
        else:
            length_ms = 0.0
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
        # Read the persisted length so recordings with trailing silence
        # beyond the last spike round-trip accurately. The caller's
        # explicit ``length_ms`` parameter still takes precedence; the
        # file attr is the second-best source. The raster style derives
        # its own length from the matrix shape and is handled separately
        # below — the file attr is not applied to it.
        file_length_ms = float(f.attrs["length_ms"]) if "length_ms" in f.attrs else None
        non_raster_length_ms = length_ms if length_ms is not None else file_length_ms

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
                computed_length_ms = max(total_time - np.spacing(total_time), 0.0)
            else:
                computed_length_ms = 0.0
            # Warn when the user supplied an explicit length_ms that
            # differs from the shape-derived value — the raster style
            # always derives length from the matrix, so an explicit
            # length_ms is silently ignored.
            if length_ms is not None and length_ms != computed_length_ms:
                warnings.warn(
                    f"length_ms={length_ms} ignored for raster style; "
                    f"length is derived from raster.shape[1] * raster_bin_size_ms "
                    f"= {computed_length_ms}.",
                    UserWarning,
                    stacklevel=2,
                )
            length_ms = computed_length_ms
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
                length_ms=non_raster_length_ms,
                start_time=file_start_time,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        if group_per_unit is not None:
            # Style (3): each child dataset is a unit's spike times.
            # Sort numerically (so "10" sorts after "9", not after "1") so
            # round-trip with the matching exporter preserves unit identity
            # at N>=10.
            grp = f[group_per_unit]
            keys = sorted(grp.keys(), key=_natural_sort_key)
            trains = [to_ms(np.asarray(grp[k]), group_time_unit, fs_Hz) for k in keys]
            return _build_spikedata(
                trains,
                length_ms=non_raster_length_ms,
                start_time=file_start_time,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
            )

        # Style (4): paired indices and times arrays
        idces = np.asarray(f[idces_dataset])  # type: ignore
        times = to_ms(np.asarray(f[times_dataset]), times_unit, fs_Hz)  # type: ignore
        if len(idces) != len(times):
            raise ValueError(
                f"idces_dataset and times_dataset must have equal length; "
                f"got len(idces)={len(idces)}, len(times)={len(times)}."
            )
        N = int(idces.max()) + 1 if idces.size else 0
        # Surface sparse cluster_id padding so the operator can tell a
        # legitimate "unit N had no spikes" case apart from "unit N
        # was dropped by Phy curation and the loader filled in an
        # empty train". Without this, a curated Phy export that
        # dropped clusters 2..46 silently produced 45 empty units in
        # the middle and downstream consumers had no signal.
        unique_ids = np.unique(idces) if idces.size else np.array([])
        if idces.size and len(unique_ids) != N:
            missing = sorted(set(range(N)) - set(int(u) for u in unique_ids))
            preview = missing[:5]
            more = "..." if len(missing) > 5 else ""
            warnings.warn(
                f"paired-style HDF5 has sparse cluster_ids: max+1={N} "
                f"but only {len(unique_ids)} distinct ids present. "
                f"The loader will create {len(missing)} empty unit(s) "
                f"to pad (cluster_ids {preview}{more}). If this is a "
                "Phy-curated export, the empty units may not match the "
                "user's mental model — consider compacting the cluster "
                "ids upstream.",
                UserWarning,
                stacklevel=2,
            )
        sd = SpikeData.from_idces_times(
            idces,
            times,
            N=N,
            length=non_raster_length_ms,
            start_time=file_start_time,
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
    # Validate ``fs_Hz`` at the loader boundary so the user sees a clear
    # error mentioning the loader parameter, rather than a cryptic
    # ZeroDivisionError / ValueError raised deep inside
    # ``SpikeData.from_thresholding``.
    if not (isinstance(fs_Hz, (int, float)) and fs_Hz > 0):
        raise ValueError(
            f"fs_Hz must be a positive finite number, got {fs_Hz!r}. "
            "Set the sampling frequency in Hz when calling "
            "load_spikedata_from_hdf5_raw_thresholded(...)."
        )
    ensure_h5py()
    with h5py.File(filepath, "r") as f:  # type: ignore
        data = np.asarray(f[dataset])
        # Honour the same file-level ``length_ms``/``start_time`` attrs
        # that ``load_spikedata_from_hdf5`` reads. Without these, the
        # two loaders for the same on-disk file format had asymmetric
        # round-trip semantics for trailing silence and event-centered
        # start_time: the raster path preserved them, the thresholded
        # path silently inferred from data shape.
        file_length = float(f.attrs["length_ms"]) if "length_ms" in f.attrs else None
        file_start_time = (
            float(f.attrs["start_time"]) if "start_time" in f.attrs else None
        )
    return SpikeData.from_thresholding(
        data,
        fs_Hz=fs_Hz,
        threshold_sigma=threshold_sigma,
        filter=filter,
        hysteresis=hysteresis,
        direction=direction,  # type: ignore[arg-type]
        length=file_length,
        start_time=file_start_time,
    )


# ----------------------------
# NWB (units table)
# ----------------------------


def _nwb_str_or_none(value) -> Optional[str]:
    """Coerce ``value`` to a non-empty stripped string, else ``None``.

    Module-private helper for the NWB loader's file-level metadata
    population — used to clean up Subject / session text fields that
    may be ``None``, empty, or whitespace.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _nwb_safe_get(table, column: str, row_idx: int, *, cast=None):
    """Pull one cell out of a pynwb ``DynamicTable`` column.

    pynwb's ``__getitem__`` semantics are quirky (region references,
    slicing) and a malformed column shouldn't crash the whole
    extraction. Broad ``Exception`` catches are intentional — callers
    receive ``None`` for any cell pynwb can't decode.
    """
    try:
        col = table[column]
    except Exception:
        return None
    try:
        value = col[row_idx]
    except Exception:
        return None
    if cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _nwb_collect_file_metadata(nwbfile) -> dict:
    """Build the file-level metadata dict folded into ``SpikeData.metadata``.

    Reads subject, session, devices, and per-channel electrode info
    (locations + 3D coords + group names). Returns a flat dict with
    string keys — caller merges it into ``meta`` alongside the existing
    ``source_file`` / ``format`` keys.

    Keys populated (all optional — absent when the file doesn't carry
    the field):

      * ``identifier``: NWB file identifier (UUID-shaped string).
      * ``session_description``, ``session_start_time`` (ISO string).
      * ``subject_id``, ``species``, ``sex``, ``age``, ``date_of_birth``
        (ISO string).
      * ``device_names``: list of device names (probes, rigs), sorted.
      * ``sampling_rate_hz``, ``duration_seconds``: from the first
        ElectricalSeries acquisition object.
      * ``unit_count``: number of rows in ``nwbfile.units``.
      * ``electrodes_by_channel``: ``{channel_id (int): {"location":
        str|None, "group_name": str|None, "x"/"y"/"z": float|None}}``.
        The free-text ``location`` field is what downstream ontology
        resolvers consume (cross-references to UBERON / Allen CCF).
    """
    meta: Dict[str, object] = {}

    identifier = getattr(nwbfile, "identifier", None)
    if identifier is not None and str(identifier).strip():
        meta["identifier"] = str(identifier).strip()

    sess_desc = _nwb_str_or_none(getattr(nwbfile, "session_description", None))
    if sess_desc is not None:
        meta["session_description"] = sess_desc

    sess_start = getattr(nwbfile, "session_start_time", None)
    if isinstance(sess_start, datetime):
        meta["session_start_time"] = sess_start.isoformat()

    subject = getattr(nwbfile, "subject", None)
    if subject is not None:
        for field_name, attr in (
            ("subject_id", "subject_id"),
            ("species", "species"),
            ("sex", "sex"),
            ("age", "age"),
        ):
            val = _nwb_str_or_none(getattr(subject, attr, None))
            if val is not None:
                meta[field_name] = val
        dob = getattr(subject, "date_of_birth", None)
        if isinstance(dob, datetime):
            meta["date_of_birth"] = dob.isoformat()

    nwb_devices = getattr(nwbfile, "devices", None)
    if nwb_devices:
        try:
            meta["device_names"] = sorted(str(name) for name in nwb_devices.keys())
        except Exception:
            pass

    sample_rate, duration = _read_nwb_first_acquisition_timing(nwbfile)
    if sample_rate is not None:
        meta["sampling_rate_hz"] = sample_rate
    if duration is not None:
        meta["duration_seconds"] = duration

    nwb_units = getattr(nwbfile, "units", None)
    if nwb_units is not None:
        try:
            meta["unit_count"] = int(len(nwb_units))
        except (TypeError, ValueError):
            meta["unit_count"] = 0

    electrodes_by_channel = _read_nwb_electrodes(nwbfile)
    if electrodes_by_channel:
        meta["electrodes_by_channel"] = electrodes_by_channel

    return meta


def _read_nwb_electrodes(nwbfile) -> dict:
    """Read per-channel electrode metadata from ``nwbfile.electrodes``.

    Returns ``{channel_id (int): {"location": str|None, "group_name":
    str|None, "x"/"y"/"z": float|None}}``. The ``location`` value is
    the source dataset's free-text region label (e.g. ``"VISp"``,
    ``"CA1"``, ``"primary visual cortex"``) — kept verbatim; the
    consumer (e.g. gateway ingestion) resolves to a canonical ontology
    ID.
    """
    table = getattr(nwbfile, "electrodes", None)
    if table is None:
        return {}
    try:
        n_rows = int(len(table))
    except (TypeError, ValueError):
        return {}
    cols = set(getattr(table, "colnames", ()) or ())
    out: Dict[int, dict] = {}
    for i in range(n_rows):
        entry: Dict[str, object] = {}
        if "location" in cols:
            loc = _nwb_str_or_none(_nwb_safe_get(table, "location", i))
            if loc is not None:
                entry["location"] = loc
        if "group_name" in cols:
            gn = _nwb_str_or_none(_nwb_safe_get(table, "group_name", i))
            if gn is not None:
                entry["group_name"] = gn
        for coord in ("x", "y", "z"):
            if coord in cols:
                v = _nwb_safe_get(table, coord, i, cast=float)
                if v is not None:
                    entry[coord] = v
        # Even an empty entry is recorded — the consumer needs the full
        # set of channel ids to build a layout description.
        out[i] = entry
    return out


def _resolve_electrode_ref(val):
    """Resolve a units-table electrode-reference cell to a scalar id.

    Cell values can come back as:
      * a pandas DataFrame slice (the dereferenced electrodes-table rows
        when the column is a ``DynamicTableRegion`` — pynwb default).
        Take the first row's index.
      * a list / ndarray of integer indices. Take the first element.
      * a bare scalar (rare — used by some non-standard NWB writers).
        Use as-is.

    Returns the resolved scalar, or the original ``val`` if no
    interpretation fits (caller's ``int()`` coercion will then fail
    cleanly).
    """
    # pandas DataFrame: dereferenced electrode rows from a DynamicTableRegion.
    # Prefer the underlying ``Index`` (the electrode-table row ids) over
    # value-indexing the columns. ``__contains__`` discriminates a
    # DataFrame from a numpy array (both have ``index`` only when truly
    # pandas — numpy doesn't expose ``index``).
    if hasattr(val, "index") and not isinstance(val, (str, bytes)):
        try:
            n = len(val.index)
        except Exception:
            n = 0
        if n > 0:
            try:
                return val.index[0]
            except Exception:
                pass
    if (
        hasattr(val, "__len__")
        and not isinstance(val, str)
        and not hasattr(val, "index")  # already handled above
    ):
        try:
            n = len(val)
        except Exception:
            n = 0
        if n > 0:
            try:
                return val[0]
            except Exception:
                return val
    return val


def _read_nwb_first_acquisition_timing(
    nwbfile,
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(sampling_rate_hz, duration_seconds)`` from the first
    ``ElectricalSeries``-shaped acquisition object, or ``(None, None)``.

    NWB stores the rate per acquisition object; we record the first as
    a representative value. Multi-rate recordings are flagged for the
    consumer by repeating the call against later acquisitions if
    needed.
    """
    acq = getattr(nwbfile, "acquisition", None)
    if not acq:
        return (None, None)
    for _name, obj in acq.items():
        rate = getattr(obj, "rate", None)
        data = getattr(obj, "data", None)
        if rate is None or data is None:
            continue
        try:
            rate_f = float(rate)
        except (TypeError, ValueError):
            continue
        n_samples: Optional[int] = None
        shape = getattr(data, "shape", None)
        if shape is not None:
            try:
                n_samples = int(shape[0])
            except (TypeError, ValueError, IndexError):
                n_samples = None
        if rate_f > 0 and n_samples is not None:
            return (rate_f, n_samples / rate_f)
        if rate_f > 0:
            return (rate_f, None)
    return (None, None)


def load_spikedata_from_nwb(
    filepath: str,
    *,
    prefer_pynwb: bool = True,
    length_ms: Optional[float] = None,
    start_time_ms: Optional[float] = None,
    allow_no_units: bool = False,
) -> SpikeData:
    """Load spike trains from an NWB file's Units table.

    Parameters:
        filepath (str): Path to the NWB file.
        prefer_pynwb (bool): If True, try pynwb first; if False, try h5py.
        length_ms (float | None): Recording duration in milliseconds.
            When ``None``, reads from the file-level ``length_ms``
            attribute (written by ``export_spikedata_to_nwb``); falls
            back to inferring from the latest spike time if the
            attribute is absent.
        start_time_ms (float | None): Recording start time in
            milliseconds. When ``None``, reads from the file-level
            ``start_time`` attribute (written by
            ``export_spikedata_to_nwb``); falls back to 0.0 if the
            attribute is absent. Mirrors the ``length_ms`` ladder.
        allow_no_units (bool): When ``True``, files without a Units table
            return a ``SpikeData`` with ``N=0`` and empty trains rather
            than raising ``ValueError``. The file-level metadata in
            ``sd.metadata`` is still populated. Useful for metadata-
            only callers (e.g. ingestion pipelines that need to gate on
            "is this sorted?" without crashing on unsorted inputs).
            Only honored on the pynwb path; the h5py fallback still
            requires the ``/units`` group.

    Returns:
        sd (SpikeData): The loaded spike train data. Under the pynwb
            path, ``sd.metadata`` is populated with file-level NWB
            metadata in addition to the usual ``source_file`` /
            ``format``: ``identifier``, ``session_description``,
            ``session_start_time`` (ISO string), ``subject_id``,
            ``species``, ``sex``, ``age``, ``date_of_birth`` (ISO),
            ``device_names`` (sorted list), ``sampling_rate_hz``,
            ``duration_seconds``, ``unit_count``, and
            ``electrodes_by_channel`` (``{channel_id: {"location",
            "group_name", "x", "y", "z"}}``). Each entry in
            ``sd.neuron_attributes`` gains a ``location_label`` key
            (textual region from the electrodes table) and
            ``group_name`` key alongside the existing ``location`` 3D
            coordinate list. The h5py fallback path doesn't populate
            these extra fields — ``length_ms`` / ``start_time`` remain
            the only file-level attrs it carries.
    """
    trains: List[np.ndarray] = []
    neuron_attributes: List[dict] = []
    meta: Dict[str, object] = {
        "source_file": os.path.abspath(filepath),
        "format": "NWB",
    }

    # Read file-level attributes via h5py up-front so both the pynwb
    # and h5py paths benefit. Caller overrides take precedence; missing
    # attrs fall back to None/0 (the SpikeData defaults).
    file_length_ms: Optional[float] = None
    file_start_time_ms: float = 0.0
    if length_ms is None or start_time_ms is None:
        try:
            import h5py as _h5  # type: ignore

            with _h5.File(filepath, "r") as _attrs_f:
                if "length_ms" in _attrs_f.attrs:
                    file_length_ms = float(_attrs_f.attrs["length_ms"])
                if "start_time" in _attrs_f.attrs:
                    file_start_time_ms = float(_attrs_f.attrs["start_time"])
        except Exception:
            # Attribute read is best-effort; if h5py can't open the file
            # (corrupt, unsupported plugin, etc.) the loader proper will
            # raise the real error below.
            pass
    if length_ms is None:
        length_ms = file_length_ms
    if start_time_ms is None:
        start_time_ms = file_start_time_ms

    if prefer_pynwb:
        try:
            from pynwb import NWBHDF5IO  # type: ignore

            with NWBHDF5IO(filepath, "r") as io:
                nwb = io.read()
                has_units = getattr(nwb, "units", None) is not None
                if not has_units and not allow_no_units:
                    raise ValueError("NWB file has no Units table")

                # File-level metadata: subject, session, devices, per-
                # channel electrode info (location text + 3D coords +
                # group name), sampling rate, unit count. Merged into
                # ``meta`` so downstream consumers (analysis code,
                # gateway ingestion, etc.) have everything they need
                # without re-opening the file. Done BEFORE the units
                # branch so metadata-only callers (``allow_no_units=True``)
                # also get fully-populated metadata.
                meta.update(_nwb_collect_file_metadata(nwb))

                electrodes_by_channel: dict = meta.get(
                    "electrodes_by_channel", {}
                )  # type: ignore[assignment]

                if not has_units:
                    # Metadata-only return for unsorted files. Caller
                    # detects via ``sd.N == 0`` or
                    # ``sd.metadata.get("unit_count", 0) == 0``.
                    return _build_spikedata(
                        trains,
                        length_ms=length_ms,
                        start_time=start_time_ms if start_time_ms is not None else 0.0,
                        metadata=meta,
                        neuron_attributes=neuron_attributes,
                    )

                df = nwb.units.to_dataframe()

                for row in df.itertuples():
                    stimes = np.asarray(row.spike_times, dtype=float)
                    trains.append(stimes * 1e3)
                    attr = {"unit_id": row.Index}
                    electrode_id = None
                    for col in ("electrodes", "electrode_group", "channel", "ch"):
                        if col in df.columns:
                            val = getattr(row, col, None)
                            if val is not None:
                                channel_val = _resolve_electrode_ref(val)
                                try:
                                    attr["electrode"] = int(channel_val)
                                    electrode_id = int(channel_val)
                                except (TypeError, ValueError):
                                    attr["electrode"] = channel_val
                                    electrode_id = channel_val
                                break
                    # Backwards-compatible enrichment: ``location`` stays
                    # the 3D coordinate list (existing analysis code
                    # depends on that shape); ``location_label`` is the
                    # new textual region name (e.g. ``"VISp"``,
                    # ``"CA1"``) consumed by ontology resolvers.
                    if electrodes_by_channel and electrode_id in electrodes_by_channel:
                        ec = electrodes_by_channel[electrode_id]
                        pos = [ec[c] for c in ("x", "y", "z") if c in ec]
                        if pos:
                            attr["location"] = pos
                        if "location" in ec:
                            attr["location_label"] = ec["location"]
                        if "group_name" in ec:
                            attr["group_name"] = ec["group_name"]
                    neuron_attributes.append(attr)
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                start_time=start_time_ms if start_time_ms is not None else 0.0,
                metadata=meta,
                neuron_attributes=neuron_attributes,
            )
        except ImportError:  # pragma: no cover
            pass  # pynwb not installed — fall back to h5py
        except Exception as e:
            # Broad catch is intentional: pynwb raises a wide variety of
            # exception types depending on schema/runtime issue (TypeError,
            # ValueError, KeyError, AttributeError, RuntimeError on schema
            # mismatch, OSError on HDF5-plugin issues, etc.). Any pynwb
            # failure should fall back to the h5py loader rather than
            # propagating to the caller. The warning preserves the original
            # exception type+message for diagnosis.
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
        # ``export_spikedata_to_nwb`` writes ``length_ms`` as a file-
        # level attribute so the loader can recover the exact recording
        # duration on reload — NWB's spec has no canonical place for
        # this, and inferring length from the max spike time silently
        # drops trailing silence. Honor the attr when the caller did
        # not supply an explicit override; the caller's argument still
        # takes precedence for backward-compatibility.
        if length_ms is None and "length_ms" in f.attrs:
            length_ms = float(f.attrs["length_ms"])
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
        # Read the unit-id table first so we know N upfront. NWB
        # files in the wild use either the spec-compliant
        # cumulative-end (length N) or a leading-zero (length N+1)
        # convention; the unit count disambiguates them.
        n_units = int(np.asarray(unit_grp["id"]).shape[0]) if "id" in unit_grp else None
        trains.extend(
            _trains_from_flat_index(
                flat.astype(float),
                index,
                unit="s",
                fs_Hz=None,
                n_units=n_units,
            )
        )

        unit_ids = (
            np.asarray(unit_grp["id"]) if "id" in unit_grp else range(len(trains))
        )

        electrode_indices = None
        if "electrodes" in unit_grp and "electrodes_index" in unit_grp:
            elec_flat = np.asarray(unit_grp["electrodes"])
            elec_idx = np.asarray(unit_grp["electrodes_index"])
            if len(elec_idx) > 0 and elec_idx[-1] > len(elec_flat):
                # Quantify the truncation so the user can tell whether
                # one entry got clipped or half the table is missing.
                # The previous "may be truncated" wording was too vague
                # to act on.
                lost = int(elec_idx[-1] - len(elec_flat))
                warnings.warn(
                    "NWB electrodes_index final value "
                    f"({int(elec_idx[-1])}) exceeds electrodes array length "
                    f"({len(elec_flat)}); the trailing {lost} index entries "
                    "will be silently truncated. Inspect the NWB file's "
                    "/units/electrodes_index and /units/electrodes datasets.",
                    UserWarning,
                )
                # Clip the index so _split_by_index's strict
                # overflow check doesn't promote the warning to an
                # error. We've already told the caller about the
                # truncation; preserve the pre-refactor lenient
                # contract for this specific NWB-in-the-wild quirk.
                elec_idx = np.minimum(elec_idx, len(elec_flat))
            # Use the shared splitter so the leading-zero (length N+1)
            # NWB convention is honoured the same way it is for
            # spike_times_index. The previous inline ``for stop in
            # elec_idx`` loop silently misaligned per-unit electrodes
            # by one when the file used the leading-zero convention.
            electrode_indices = _split_by_index(
                elec_flat,
                elec_idx,
                n_units=n_units,
                name="electrodes_index",
            )

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
            # NWB stores unit IDs in the units/id dataset; the spec
            # permits any numeric dtype and some files in the wild use
            # string IDs (e.g. UUIDs from automated pipelines). The
            # pynwb branch above accepts the raw Index value verbatim;
            # the h5py path used to ``int(uid)`` unconditionally and
            # crash mid-loop on non-numeric IDs. Fall back to the
            # original value with a warning to preserve the loader's
            # forward progress and keep the two paths symmetric.
            try:
                attr = {"unit_id": int(uid)}
            except (TypeError, ValueError):
                warnings.warn(
                    f"NWB unit id {uid!r} is not coercible to int; "
                    "storing the raw value on neuron_attributes['unit_id'].",
                    UserWarning,
                    stacklevel=2,
                )
                attr = {"unit_id": uid}
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
        trains,
        length_ms=length_ms,
        start_time=start_time_ms if start_time_ms is not None else 0.0,
        metadata=meta,
        neuron_attributes=neuron_attributes,
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
            When provided, the order of the returned SpikeData's units
            follows the caller's order (after presence validation).
        segment_index (int): Segment index for multi-segment sortings.

    Returns:
        sd (SpikeData): The converted spike train data.

    Notes:
        - When ``unit_ids is None``, the resulting unit order follows
          ``sorting.get_unit_ids()`` order, which is backend-dependent
          (KiloSort returns sequential IDs; some SpikeInterface
          variants reorder by sort metric). Two SpikeData objects
          built from different backends may therefore index the same
          physical unit at different positions. Pass an explicit
          ``unit_ids`` sequence when the unit ordering matters across
          backends.
        - ``neuron_attributes[i]["unit_id"]`` records the original
          backend ID, providing a stable mapping from position to
          source ID irrespective of the order convention.
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

    available_ids = list(get_unit_ids())
    if unit_ids is not None:
        # Pre-validate unit_ids at the loader boundary so the user sees a
        # clear error mentioning the loader parameter, rather than a
        # backend-specific exception raised mid-loop by
        # ``sorting.get_unit_spike_train(missing_id)``.
        requested = list(unit_ids)
        missing = [uid for uid in requested if uid not in available_ids]
        if missing:
            raise ValueError(
                f"unit_ids contains IDs not present in the sorting: {missing!r}. "
                f"Available unit IDs: {available_ids!r}."
            )
        ids = requested
    else:
        ids = available_ids
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
    # Pre-loop validation. ``to_ms`` raises ValueError for unknown
    # ``time_unit`` values, but only when it is reached mid-loop after
    # spike_times.npy / spike_clusters.npy / channel_map.npy I/O has
    # already completed. Surface the typo here so a typo'd time_unit
    # surfaces before any disk reads.
    if time_unit not in ("samples", "s", "ms"):
        raise ValueError(
            f"time_unit={time_unit!r} is not one of "
            "('samples', 's', 'ms'); pass a valid unit."
        )
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

    # Per-cluster physical-channel mapping. Built by one of:
    #   (1) cluster_info.tsv ``ch`` column — canonical Phy answer, set
    #       below if the TSV provides it.
    #   (2) spike_templates.npy + templates.npy — Phy/phylib's
    #       template-amplitude fallback, set further below if the
    #       intermediate kilosort files are present.
    #   (3) channel_map[cluster_id] — legacy fallback used per-cluster
    #       inside the main loop when neither (1) nor (2) yields an
    #       entry for the cluster.
    #
    # Phy's merge/split renumbers ``spike_clusters`` non-sequentially
    # but leaves ``spike_templates`` invariant, so the templates-based
    # path survives curation. The legacy fallback only happens to give
    # correct results when cluster IDs are sequential 0..N-1 AND each
    # cluster's dominant template lives at the matching ordinal
    # channel position — i.e. fresh, uncurated kilosort output.
    cluster_id_to_channel: Optional[Dict[int, int]] = None

    keep_clusters: Optional[set] = None
    if cluster_info_tsv is not None:
        tsv_path = os.path.join(folder, cluster_info_tsv)
        if not os.path.exists(tsv_path):
            warnings.warn(
                f"cluster_info_tsv path does not exist: {tsv_path}. "
                "Falling back to keeping all clusters; pass cluster_info_tsv=None "
                "to silence this warning, or check the path for typos.",
                UserWarning,
                stacklevel=2,
            )
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
                # Extract Phy's canonical post-curation channel mapping
                # from the ``ch`` column when present. ``cluster_info.tsv``
                # is written by ``phy save`` and survives merge/split
                # because Phy recomputes the dominant channel per
                # cluster from current waveforms. This bypasses the
                # buggy ``channel_map[cluster_id]`` lookup entirely.
                if id_col is not None and "ch" in df.columns:
                    try:
                        cluster_id_to_channel = dict(
                            zip(
                                df[id_col].astype(int).tolist(),
                                df["ch"].astype(int).tolist(),
                            )
                        )
                    except (ValueError, TypeError) as exc:
                        warnings.warn(
                            f"Failed parsing 'ch' column from cluster TSV "
                            f"({exc!r}); falling back to templates / "
                            "channel_map for cluster→channel mapping.",
                            UserWarning,
                            stacklevel=2,
                        )
            except ImportError:
                warnings.warn(
                    "pandas is required to parse cluster info TSV. "
                    "Install with: pip install spikelab[io]. "
                    "Keeping all clusters."
                )
            except pd.errors.EmptyDataError as e:
                raise ValueError(
                    f"Cluster info TSV at {tsv_path!r} is empty (0 rows). "
                    f"Provide a TSV with at least a header row, or omit "
                    f"cluster_info_tsv to skip cluster filtering."
                ) from e
            except (IOError, ValueError, KeyError) as e:
                warnings.warn(
                    f"Failed parsing cluster info TSV: {e}; keeping all clusters"
                )

    # Templates-based fallback for cluster→channel when TSV is absent
    # or lacks the ``ch`` column. Loads ``spike_templates.npy`` (per-spike
    # template ID — invariant under Phy curation) and ``templates.npy``
    # (per-template waveform). For each unique cluster:
    #   1. find its dominant template via mode of ``spike_templates``
    #      over the cluster's spikes;
    #   2. find that template's peak channel via argmax of the
    #      max-absolute-amplitude per channel position;
    #   3. translate channel position → physical channel ID via
    #      ``channel_map``.
    # When either intermediate file is missing or channel_map is
    # unavailable, the fallback is skipped silently — the per-cluster
    # loop below then falls through to the legacy
    # ``channel_map[cluster_id]`` path.
    if cluster_id_to_channel is None:
        st_tpl_path = os.path.join(folder, "spike_templates.npy")
        tpl_path = os.path.join(folder, "templates.npy")
        if (
            os.path.exists(st_tpl_path)
            and os.path.exists(tpl_path)
            and channel_map is not None
        ):
            try:
                spike_templates_arr = np.load(st_tpl_path).flatten()
                templates_arr = np.load(tpl_path)
                if (
                    templates_arr.ndim == 3
                    and spike_templates_arr.shape[0] == spike_clusters.shape[0]
                ):
                    # Per-template peak channel position (argmax of
                    # max |amp| across time). Shape: (n_templates,).
                    amplitudes = np.abs(templates_arr).max(axis=1)
                    template_peak_pos = amplitudes.argmax(axis=1)
                    cluster_id_to_channel = {}
                    for clu in np.unique(spike_clusters):
                        mask = spike_clusters == clu
                        if not mask.any():
                            continue
                        tpls = spike_templates_arr[mask]
                        unique_tpl, counts = np.unique(tpls, return_counts=True)
                        dominant_template = int(unique_tpl[counts.argmax()])
                        if 0 <= dominant_template < len(template_peak_pos):
                            pos = int(template_peak_pos[dominant_template])
                            if 0 <= pos < len(channel_map):
                                cluster_id_to_channel[int(clu)] = int(channel_map[pos])
                    if not cluster_id_to_channel:
                        # No cluster resolved successfully — discard
                        # the empty dict so the per-cluster loop below
                        # falls through to the legacy path.
                        cluster_id_to_channel = None
                else:
                    warnings.warn(
                        f"Templates fallback skipped: templates.npy shape "
                        f"{templates_arr.shape} is not 3-D, or "
                        f"spike_templates length {spike_templates_arr.shape[0]} "
                        f"doesn't match spike_clusters length "
                        f"{spike_clusters.shape[0]}.",
                        UserWarning,
                        stacklevel=2,
                    )
            except (IOError, ValueError) as exc:
                warnings.warn(
                    f"Failed loading spike_templates.npy / templates.npy "
                    f"for cluster→channel fallback: {exc!r}. Falling back "
                    "to channel_map[cluster_id] lookup.",
                    UserWarning,
                    stacklevel=2,
                )

    trains: List[np.ndarray] = []
    metadata_units: List[int] = []
    neuron_attributes: List[dict] = []
    unique_clusters = np.unique(spike_clusters)
    # Only warn about non-sequential cluster IDs when neither the TSV
    # ``ch`` map nor the templates fallback resolved a cluster→channel
    # mapping. With either of those in place the legacy
    # ``channel_map[cluster_id]`` path is bypassed and the misalignment
    # bug no longer applies.
    if (
        cluster_id_to_channel is None
        and channel_map is not None
        and len(unique_clusters) > 0
    ):
        expected_sequential = np.arange(len(unique_clusters))
        if not np.array_equal(unique_clusters, expected_sequential):
            warnings.warn(
                f"Cluster IDs are not sequential (0..{len(unique_clusters)-1}): "
                f"channel_map lookup uses cluster ID as array index, which "
                f"may assign incorrect electrode/location metadata after "
                f"Phy curation. Provide cluster_info_tsv with a 'ch' column "
                f"or ensure spike_templates.npy + templates.npy are in the "
                f"folder so the loader can use the correct mapping.",
                UserWarning,
            )
    unit_idx = 0
    for clu in unique_clusters:
        if keep_clusters is not None and int(clu) not in keep_clusters:
            continue
        times = spike_times[spike_clusters == clu]
        times_ms = to_ms(times.astype(float), time_unit, fs_Hz)
        trains.append(np.sort(times_ms))
        metadata_units.append(int(clu))

        attr: dict = {"unit_id": int(clu)}
        channel_idx = None
        int_clu = int(clu)
        # Resolve cluster → physical channel by priority:
        #   1. ``cluster_id_to_channel`` from TSV ``ch`` or templates
        #      fallback — both produce physical channel IDs and both
        #      survive Phy curation.
        #   2. Legacy ``channel_map[cluster_id]`` lookup — only correct
        #      for fresh uncurated kilosort output. Kept as last
        #      resort because removing it would break loaders for
        #      users who don't provide cluster_info.tsv and whose
        #      kilosort folders lack spike_templates.npy / templates.npy.
        if cluster_id_to_channel is not None and int_clu in cluster_id_to_channel:
            channel_idx = cluster_id_to_channel[int_clu]
            attr["electrode"] = channel_idx
        elif channel_map is not None and int_clu < len(channel_map):
            channel_idx = int(channel_map[int_clu])
            attr["electrode"] = channel_idx
        elif channel_map is not None:
            # Out-of-range cluster ID — channel_map lookup is skipped
            # and the unit ends up without an electrode/location. The
            # upstream "non-sequential cluster IDs" warning fires once
            # per loader call; this per-cluster warning surfaces
            # *which* clusters lost their metadata so users debugging
            # missing locations can pinpoint the offending units.
            warnings.warn(
                f"Cluster {int_clu} exceeds channel_map length "
                f"({len(channel_map)}); skipping electrode/location "
                "assignment for this unit.",
                UserWarning,
                stacklevel=2,
            )

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

    available_keys = list(data.files)
    for required_key in ("units", "fs"):
        if required_key not in available_keys:
            raise KeyError(
                f"NPZ file {filepath!r} is missing required key {required_key!r}. "
                f"Available keys: {available_keys}. Verify the file was saved by "
                "SpikeLab's sorter export pipeline."
            )
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
    allow_remote: bool = False,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> SpikeData:
    """Load a SpikeData object from a pickle file.

    Warning:
        Only load pickle files from trusted sources. Pickle
        deserialization can execute arbitrary code and should never be
        used with untrusted data. The file is deserialized before type
        checking — malicious payloads execute regardless of the
        subsequent isinstance check. Remote (S3) loads require
        ``allow_remote=True`` so the caller has to opt in.

    Parameters:
        filepath (str): Path to the pickle file, or an S3 URL
            (s3://bucket/key). Remote URLs require ``allow_remote=True``.
        allow_remote (bool): When ``False`` (default), S3 URLs are
            rejected with a ``ValueError``. Pass ``True`` to opt in to
            loading a pickle from a remote bucket; a ``UserWarning``
            is also emitted at the call site so the risk surfaces in
            batch-job logs.
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
    from .s3_utils import ensure_local_file, is_s3_url

    if is_s3_url(filepath):
        # Pickle's arbitrary-code-execution risk is amplified when the
        # source is a remote bucket: a malicious upload (or a workspace
        # JSON file rewritten by a hostile agent in batch jobs) would
        # execute attacker code before the isinstance(SpikeData) check
        # below can reject it. Force callers to opt in, and surface a
        # warning in logs so the risk is visible at runtime rather
        # than buried in this docstring.
        if not allow_remote:
            raise ValueError(
                f"Refusing to load pickle from remote URL {filepath!r}: "
                "pickle.load executes arbitrary code from the source. "
                "Pass allow_remote=True to confirm you trust the bucket."
            )
        warnings.warn(
            f"Loading pickle from remote URL {filepath!r}; pickle.load "
            "will execute arbitrary code embedded in the file. Trust "
            "the bucket and its credentials before continuing.",
            UserWarning,
            stacklevel=2,
        )

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
            f"Pickle file does not contain a SpikeData object (found {type(obj).__name__}). "
            "Use load_from_pickle for the generic loader that accepts any spikelab data type."
        )

    return obj


def load_from_pickle(
    filepath: str,
    *,
    allow_remote: bool = False,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
):
    """Load any spikelab data object from a pickle file.

    Companion to ``load_spikedata_from_pickle``: accepts any of the six
    types that ``export_to_pickle`` supports (``SpikeData``, ``RateData``,
    ``PairwiseCompMatrix``, ``PairwiseCompMatrixStack``, ``RateSliceStack``,
    ``SpikeSliceStack``).

    Warning:
        Only load pickle files from trusted sources. Pickle
        deserialization can execute arbitrary code; the type check below
        runs after deserialisation completes.

    Parameters:
        filepath (str): Path to the pickle file, or an S3 URL.
        allow_remote (bool): Opt-in flag for S3 URLs (default False).
        aws_access_key_id (str | None): AWS access key ID for S3 downloads.
        aws_secret_access_key (str | None): AWS secret access key.
        aws_session_token (str | None): AWS session token.
        region_name (str | None): AWS region name.

    Returns:
        obj: The deserialized spikelab data object.
    """
    from ..spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
    from ..spikedata.ratedata import RateData
    from ..spikedata.rateslicestack import RateSliceStack
    from ..spikedata.spikeslicestack import SpikeSliceStack
    from .s3_utils import ensure_local_file, is_s3_url

    _SUPPORTED = (
        SpikeData,
        RateData,
        PairwiseCompMatrix,
        PairwiseCompMatrixStack,
        RateSliceStack,
        SpikeSliceStack,
    )

    if is_s3_url(filepath):
        if not allow_remote:
            raise ValueError(
                f"Refusing to load pickle from remote URL {filepath!r}: "
                "pickle.load executes arbitrary code from the source. "
                "Pass allow_remote=True to confirm you trust the bucket."
            )
        warnings.warn(
            f"Loading pickle from remote URL {filepath!r}; pickle.load "
            "will execute arbitrary code embedded in the file. Trust "
            "the bucket and its credentials before continuing.",
            UserWarning,
            stacklevel=2,
        )

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

    if not isinstance(obj, _SUPPORTED):
        supported_names = ", ".join(t.__name__ for t in _SUPPORTED)
        raise ValueError(
            f"Pickle file does not contain a spikelab data object "
            f"({supported_names}); found {type(obj).__name__}"
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
    collection: Optional[str] = None,
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
        collection (str | None): If provided, skip the heuristic
            collection search and load spikes directly from this
            collection (e.g. ``"alf/probe00/pykilosort"``). Saves 3-4
            network round-trips per call when the caller already knows
            the canonical collection (e.g. in batch workflows that
            resolve the collection once and reuse it). ``None``
            (default) falls back to the PID-suffix heuristic + fallback
            chain.

    Returns:
        sd (SpikeData): Loaded spike train data.
            ``neuron_attributes`` carries the Beryl region per unit plus,
            when the Brain-Wide Map table provides them, Allen acronym
            and atlas_id, the Cosmos parcellation parent, and per-unit
            QC fields (``firing_rate``, ``presence_ratio``,
            ``amp_median``, ``contamination``, ``drift``,
            ``noise_cutoff``, ``cluster_id``).
            ``metadata`` carries:
              * Existing trial fields: ``eid``, ``pid``, ``n_trials``,
                ``trial_start_times``, ``trial_end_times``,
                ``stim_on_times``, ``stim_off_times``, ``go_cue_times``,
                ``response_times``, ``feedback_times``,
                ``first_movement_times``, ``choice``, ``feedback_type``,
                ``contrast_left``, ``contrast_right``, ``probability_left``.
              * File-level identification: ``identifier`` (= eid),
                ``format`` (``"IBL"``), ``unit_count``,
                ``sampling_rate_hz`` (when present on the spikes
                object), ``duration_seconds``.
              * Session metadata (best-effort, one extra REST call):
                ``session_start_time``, ``session_end_time``, ``lab``,
                ``task_protocol``, ``project``, ``session_number``,
                ``procedures``, ``qc``.
              * Subject metadata (same REST chain):
                ``subject_id``, ``species``, ``sex``, ``date_of_birth``,
                ``age_weeks``, ``strain``, ``genotype``,
                ``responsible_user``.
              * Probe insertion (best-effort REST call):
                ``probe_name``, ``probe_model``, and ``insertion_*``
                coordinates for any of {x, y, z, theta, phi, depth}
                present.
              * ``electrodes_by_channel`` (best-effort
                ``one.load_object("channels")`` call): per-channel
                ``location`` (Allen acronym), ``atlas_id`` (Allen
                Structure ID), ``x``/``y``/``z`` (ML/AP/DV in mm),
                ``local_x``/``local_y`` (probe-relative, μm),
                ``raw_index``. Key shape matches the NWB loader's
                ``electrodes_by_channel`` for cross-format consumers.
            All time arrays are in milliseconds.

    Notes:
        - Requires ``one-api`` and ``brainwidemap`` packages (optional dependencies).
        - Spike times are converted from seconds (IBL convention) to milliseconds.
        - Trial times are converted from seconds to milliseconds.
        - When ``collection`` is ``None``, the probe collection is
          inferred from the PID suffix; falls back through
          ``alf/probe00/pykilosort``, ``alf/probe01/pykilosort``, and
          ``alf``.
        - Session, subject, insertion, and channels lookups are
          best-effort. A failure on any of them yields an absent
          metadata field rather than raising — the spike-train load
          succeeds as long as the units table is reachable.
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

    # Build the ordered list of collections to try. When the caller
    # supplied an explicit ``collection``, short-circuit the heuristic
    # search — saves 3-4 network round-trips per call in batch
    # workflows that already know the canonical collection.
    if collection is not None:
        ordered_collections: List[str] = [collection]
    else:
        # With the probe-specific collection first when the PID suffix
        # hints at the probe number. This is a best-effort heuristic
        # for ordering (PIDs are UUIDs, so the last two hex chars can
        # coincidentally match "00"/"01"). All candidates are tried
        # regardless, so correctness is not affected — only the order.
        collections = []
        if pid.endswith("00") or pid.endswith("01"):
            collections.append(f"alf/probe{pid[-2:]}/pykilosort")
        collections.extend(_IBL_FALLBACK_COLLECTIONS)
        # Deduplicate while preserving order.
        seen: set = set()
        ordered_collections = []
        for c in collections:
            if c not in seen:
                seen.add(c)
                ordered_collections.append(c)

    # Load spikes from the first available collection.
    spikes = None
    for candidate in ordered_collections:
        try:
            spikes = one.load_object(eid, "spikes", collection=candidate)
            break
        except (ValueError, KeyError, FileNotFoundError):
            continue

    # When every collection fallback failed and the Brain-Wide Map lists
    # good units for this probe, the loader is about to return a
    # SpikeData full of empty trains — a result that looks valid to
    # downstream code (correct N, correct neuron_attributes, correct
    # trial metadata) but contains zero actual spikes. Surface the
    # silent failure with a loud UserWarning so it shows up in batch
    # logs without crashing the calling script.
    if spikes is None and len(good_units) > 0:
        warnings.warn(
            f"load_spikedata_from_ibl: failed to load spikes for "
            f"eid={eid!r}, pid={pid!r} from any of the candidate "
            f"collections: {ordered_collections}. The Brain-Wide Map "
            f"lists {len(good_units)} good unit(s) for this probe — "
            f"the returned SpikeData will have {len(good_units)} units "
            f"but zero spikes. Verify the eid/pid pair on the IBL "
            f"server, or check ``sd.train`` for the all-empty "
            f"signature before using.",
            UserWarning,
            stacklevel=2,
        )

    # Build per-unit spike trains (seconds → milliseconds).
    spike_trains: List[np.ndarray] = []
    neuron_attributes: List[dict] = []
    # Per-unit QC + Allen-precise region fields, copied from the
    # Brain-Wide Map unit table when present. Only columns that exist
    # are copied so future bwm_units schema changes don't break the
    # loader.
    _IBL_PER_UNIT_COPY = (
        ("acronym", "acronym"),  # Allen acronym
        ("atlas_id", "atlas_id"),  # Allen Structure ID
        ("Cosmos", "cosmos"),  # Cosmos parcellation parent
        ("firing_rate", "firing_rate"),
        ("presence_ratio", "presence_ratio"),
        ("amp_median", "amp_median"),
        ("contamination", "contamination"),
        ("drift", "drift"),
        ("noise_cutoff", "noise_cutoff"),
        ("cluster_id", "cluster_id"),
    )
    for _, unit in good_units.iterrows():
        if spikes is None:
            spike_trains.append(np.array([], dtype=float))
        else:
            mask = spikes["clusters"] == unit["cluster_id"]
            spike_trains.append(spikes["times"][mask] * 1_000.0)
        attr: dict = {"region": unit["Beryl"]}  # Beryl region (existing)
        for src_col, dst_key in _IBL_PER_UNIT_COPY:
            if src_col in good_units.columns:
                try:
                    val = unit[src_col]
                except KeyError:
                    continue
                # Skip NaN / pandas-NA without depending on pandas being
                # imported here (the IBL function already requires it
                # transitively, but be defensive).
                if val is None:
                    continue
                if hasattr(val, "item"):
                    try:
                        val = val.item()
                    except (TypeError, ValueError):
                        pass
                # Filter NaN floats (pandas often emits these for
                # missing values in float columns).
                if isinstance(val, float) and val != val:  # NaN check
                    continue
                attr[dst_key] = val
        neuron_attributes.append(attr)

    # Infer session length from the largest spike time if not provided.
    if length_ms is None:
        max_t = max((t.max() for t in spike_trains if len(t) > 0), default=0.0)
        if max_t > 0:
            length_ms = float(max_t)
        else:
            # All surviving units returned zero spikes for this probe.
            # The previous fabricated 10 000 ms default silently produced
            # a SpikeData whose downstream rate normalisation
            # (``rates()``) and binning (``raster(bin_size_ms)``) were
            # based on the magic duration. Refuse instead and force the
            # caller to supply ``length_ms`` explicitly, so the time
            # axis used downstream is provenance-traceable.
            raise ValueError(
                f"IBL probe {pid!r} returned zero spikes for every "
                "surviving unit, so the session length cannot be "
                "inferred from spike times. Pass an explicit "
                "``length_ms`` argument to load_spikedata_from_ibl "
                "(typically the session duration from the IBL "
                "trials table)."
            )

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

    # File-level identification + counts (Tier 2 — free, from already-
    # loaded data). ``identifier`` parallels NWB so downstream consumers
    # can read the same key across loaders.
    metadata["identifier"] = eid
    metadata["format"] = "IBL"
    metadata["unit_count"] = int(len(good_units))
    if length_ms is not None:
        metadata["duration_seconds"] = float(length_ms) / 1_000.0
    if spikes is not None:
        sr = getattr(spikes, "sampling_rate", None)
        if sr is not None:
            try:
                metadata["sampling_rate_hz"] = float(sr)
            except (TypeError, ValueError):
                pass

    # Session + subject + probe-insertion metadata (Tier 1 — two extra
    # cached REST queries). Wrapped in best-effort helper so a failed
    # Alyx request doesn't crash the spike-train load.
    metadata.update(_ibl_collect_session_metadata(one, eid, pid))

    # Per-channel Allen acronym + atlas_id + 3D coords (Tier 3 — one
    # extra cached ONE.load_object call). Empty dict when the channels
    # object isn't available for any of the probe collections.
    electrodes = _ibl_collect_channels(one, eid, ordered_collections)
    if electrodes:
        metadata["electrodes_by_channel"] = electrodes

    return _build_spikedata(
        spike_trains,
        length_ms=length_ms,
        metadata=metadata,
        neuron_attributes=neuron_attributes,
    )


def _ibl_collect_session_metadata(one, eid: str, pid: str) -> Dict[str, object]:
    """Best-effort gather of IBL session + subject + probe-insertion
    metadata via the Alyx REST API.

    Every individual query is wrapped — a failure (network blip,
    permissions, schema drift) yields an absent field rather than
    crashing the spike-train load. Returns a dict containing whichever
    fields could be resolved.

    Keys populated (all optional):
      * Session: ``session_start_time``, ``session_end_time``, ``lab``,
        ``task_protocol``, ``project``, ``session_number``, ``procedures``,
        ``qc``.
      * Subject (one REST call away from session): ``subject_id``,
        ``species`` (typically ``"Mus musculus"`` for IBL),
        ``sex``, ``date_of_birth``, ``age_weeks``, ``strain``,
        ``genotype``, ``responsible_user``.
      * Probe insertion: ``probe_name``, ``probe_model``, and
        ``insertion_*`` for any of {x, y, z, theta, phi, depth}
        present in the insertion's free-form json.
    """
    md: Dict[str, object] = {}

    # Session
    sess = None
    try:
        sess = one.alyx.rest("sessions", "read", id=eid)
    except Exception:
        sess = None
    if isinstance(sess, dict):
        for src_key, dst_key in (
            ("start_time", "session_start_time"),
            ("end_time", "session_end_time"),
            ("lab", "lab"),
            ("task_protocol", "task_protocol"),
            ("project", "project"),
            ("number", "session_number"),
            ("procedures", "procedures"),
            ("qc", "qc"),
        ):
            val = sess.get(src_key)
            if val is not None:
                md[dst_key] = val

        # Subject — Alyx returns the subject as a nickname string on
        # session records. A second REST query fetches the full subject
        # detail.
        subj = sess.get("subject")
        if isinstance(subj, str) and subj:
            md["subject_id"] = subj
            subj_data = None
            try:
                subj_data = one.alyx.rest("subjects", "read", id=subj)
            except Exception:
                subj_data = None
            if isinstance(subj_data, dict):
                for src_key, dst_key in (
                    ("species", "species"),
                    ("sex", "sex"),
                    ("birth_date", "date_of_birth"),
                    ("age_weeks", "age_weeks"),
                    ("strain", "strain"),
                    ("genotype", "genotype"),
                    ("responsible_user", "responsible_user"),
                ):
                    val = subj_data.get(src_key)
                    if val is not None:
                        md[dst_key] = val

    # Probe insertion
    ins = None
    try:
        ins = one.alyx.rest("insertions", "read", id=pid)
    except Exception:
        ins = None
    if isinstance(ins, dict):
        if ins.get("name"):
            md["probe_name"] = ins["name"]
        if ins.get("model"):
            md["probe_model"] = ins["model"]
        ij = ins.get("json")
        if isinstance(ij, dict):
            for key in ("x", "y", "z", "theta", "phi", "depth"):
                v = ij.get(key)
                if v is not None:
                    md[f"insertion_{key}"] = v

    return md


def _ibl_collect_channels(
    one, eid: str, ordered_collections: List[str]
) -> Dict[int, dict]:
    """Best-effort load of the IBL ``channels`` object and projection
    onto a NWB-style ``electrodes_by_channel`` dict.

    Returns ``{channel_id (int): {"location" (Allen acronym),
    "atlas_id" (Allen Structure ID), "x"/"y"/"z" (ML/AP/DV in mm),
    "local_x"/"local_y" (probe-relative, micrometres), "raw_index"}}``.
    Empty dict when none of the probe collections yield a channels
    object.

    Key names match the NWB loader's ``electrodes_by_channel`` shape so
    downstream consumers (ingestion mappers, analysis pipelines) read
    the same fields across formats. ``location`` carries the Allen
    acronym for IBL — analogous to the textual region NWB writes in the
    electrodes table's ``location`` column.
    """
    channels = None
    for candidate in ordered_collections:
        try:
            channels = one.load_object(eid, "channels", collection=candidate)
            break
        except (ValueError, KeyError, FileNotFoundError):
            continue
    if channels is None:
        return {}

    acronyms = getattr(channels, "acronym", None)
    atlas_ids = getattr(channels, "atlas_id", None)
    mlapdv = getattr(channels, "mlapdv", None)
    local_coords = getattr(channels, "localCoordinates", None)
    raw_inds = getattr(channels, "rawInd", None)

    # Pick the largest length across the per-attribute arrays as the
    # channel count; this is robust to any one attribute being absent.
    n = 0
    for arr in (acronyms, atlas_ids, raw_inds):
        try:
            n = max(n, len(arr))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    if mlapdv is not None:
        try:
            n = max(n, int(mlapdv.shape[0]))
        except (TypeError, ValueError, IndexError, AttributeError):
            pass
    if local_coords is not None:
        try:
            n = max(n, int(local_coords.shape[0]))
        except (TypeError, ValueError, IndexError, AttributeError):
            pass
    if n == 0:
        return {}

    out: Dict[int, dict] = {}
    for i in range(n):
        entry: Dict[str, object] = {}
        if acronyms is not None:
            try:
                entry["location"] = str(acronyms[i])
            except (IndexError, TypeError, ValueError):
                pass
        if atlas_ids is not None:
            try:
                entry["atlas_id"] = int(atlas_ids[i])
            except (IndexError, TypeError, ValueError):
                pass
        if mlapdv is not None:
            try:
                entry["x"] = float(mlapdv[i, 0])  # ML
                entry["y"] = float(mlapdv[i, 1])  # AP
                entry["z"] = float(mlapdv[i, 2])  # DV
            except (IndexError, TypeError, ValueError, AttributeError):
                pass
        if local_coords is not None:
            try:
                entry["local_x"] = float(local_coords[i, 0])
                entry["local_y"] = float(local_coords[i, 1])
            except (IndexError, TypeError, ValueError, AttributeError):
                pass
        if raw_inds is not None:
            try:
                entry["raw_index"] = int(raw_inds[i])
            except (IndexError, TypeError, ValueError):
                pass
        out[i] = entry
    return out


# ----------------------------
# DANDI Archive
# ----------------------------

#: DANDI Archive REST API base URL. Public dandisets are readable without
#: auth; embargoed dandisets need ``DANDI_API_TOKEN`` in the env.
_DANDI_API_BASE = "https://api.dandiarchive.org/api"

#: Default page size used by :func:`list_dandi_assets`.
_DANDI_DEFAULT_PAGE_SIZE = 100


def _dandi_json_get(
    url: str,
    *,
    api_token: Optional[str] = None,
    timeout_seconds: float = 30.0,
) -> dict:
    """GET ``url`` and decode JSON. Adds optional bearer auth.

    Used by both the asset listing and asset detail endpoints. Caller
    is responsible for retries — DANDI's API is generally reliable
    enough that a single attempt is fine for most workflows.
    """
    import urllib.request as _ur

    req = _ur.Request(url)
    req.add_header("Accept", "application/json")
    if api_token:
        req.add_header("Authorization", f"token {api_token}")
    with _ur.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _dandi_download_asset(
    download_url: str,
    dest: str,
    *,
    api_token: Optional[str] = None,
    timeout_seconds: float = 600.0,
) -> str:
    """Stream a DANDI asset to ``dest``.

    Uses ``urlopen`` + a 1-MiB read loop so the per-request timeout
    applies to the whole download, not just the header exchange. NWB
    assets on DANDI range from MB to multi-GB; the default 10-minute
    timeout covers ~50 GB at 100 Mbps.
    """
    import urllib.request as _ur

    req = _ur.Request(download_url)
    if api_token:
        req.add_header("Authorization", f"token {api_token}")
    with _ur.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
    return dest


def list_dandi_assets(
    dandiset_id: str,
    *,
    version: str = "draft",
    path_glob: Optional[str] = None,
    api_token: Optional[str] = None,
    api_base: str = _DANDI_API_BASE,
    page_size: int = _DANDI_DEFAULT_PAGE_SIZE,
    request_timeout_seconds: float = 30.0,
):
    """Yield assets in a DANDI dandiset version.

    Parameters:
        dandiset_id (str): Six-digit DANDI identifier (e.g. ``"000006"``).
            Leading zeros matter.
        version (str): Dandiset version. ``"draft"`` (default) is the
            in-progress version; published versions are tagged like
            ``"0.231012.0"``.
        path_glob (str | None): Optional ``glob`` pattern (e.g.
            ``"*.nwb"``) the API filters on server-side. Cheaper than
            client-side filtering when most assets aren't of interest.
        api_token (str | None): Personal access token. Required for
            embargoed dandisets. Defaults to the ``DANDI_API_TOKEN``
            env var when not supplied; public dandisets work without one.
        api_base (str): API root. Override for staging or self-hosted DANDI.
        page_size (int): Per-page result count. Default 100 — the
            iterator pages internally, so this only affects request
            granularity.
        request_timeout_seconds (float): Per-request timeout.

    Yields:
        dict: One asset per yielded value, with keys:
            ``asset_id`` (str, UUID), ``path`` (str, dandiset-relative),
            ``size`` (int, bytes), ``download_url`` (str),
            ``dandiset_id`` (str), ``version`` (str).

    Notes:
        Pagination is handled transparently — caller iterates without
        worrying about ``next_page``. Large dandisets can have
        thousands of assets, so consumers should consume the iterator
        lazily rather than materialising the full list.
    """
    import urllib.parse as _urlparse

    if api_token is None:
        api_token = os.environ.get("DANDI_API_TOKEN")

    params = {"page_size": str(page_size)}
    if path_glob:
        params["glob"] = path_glob
    url = (
        f"{api_base.rstrip('/')}/dandisets/{dandiset_id}/versions/"
        f"{version}/assets/?{_urlparse.urlencode(params)}"
    )
    while url:
        payload = _dandi_json_get(
            url, api_token=api_token, timeout_seconds=request_timeout_seconds
        )
        for entry in payload.get("results") or ():
            asset_id = str(entry.get("asset_id") or entry.get("identifier") or "")
            path = str(entry.get("path") or "")
            size_raw = entry.get("size") or 0
            try:
                size = int(size_raw)
            except (TypeError, ValueError):
                size = 0
            content_urls = entry.get("contentUrl") or entry.get("contentUrls") or ()
            if isinstance(content_urls, str):
                content_urls = [content_urls]
            download_url = (
                entry.get("download_url")
                or (content_urls[0] if content_urls else "")
                or f"{api_base.rstrip('/')}/assets/{asset_id}/download/"
            )
            yield {
                "asset_id": asset_id,
                "path": path,
                "size": size,
                "download_url": str(download_url),
                "dandiset_id": dandiset_id,
                "version": version,
            }
        url = payload.get("next") or ""


def load_spikedata_from_dandi(
    asset_id: str,
    *,
    dandiset_id: Optional[str] = None,
    version: str = "draft",
    download_dir: Optional[str] = None,
    api_token: Optional[str] = None,
    api_base: str = _DANDI_API_BASE,
    request_timeout_seconds: float = 30.0,
    download_timeout_seconds: float = 600.0,
    allow_no_units: bool = False,
    length_ms: Optional[float] = None,
    start_time_ms: Optional[float] = None,
) -> SpikeData:
    """Download one DANDI NWB asset and load it as a :class:`SpikeData`.

    Resolves the asset's download URL via DANDI's asset-detail endpoint
    (or accepts a direct URL when ``asset_id`` looks like one), streams
    the bytes to disk, then delegates to :func:`load_spikedata_from_nwb`
    for the actual NWB parsing. DANDI provenance fields are added to
    ``SpikeData.metadata``.

    Parameters:
        asset_id (str): Either a DANDI asset UUID (the
            ``asset_id`` field from :func:`list_dandi_assets`) or a
            fully-qualified asset download URL.
        dandiset_id (str | None): Owning dandiset id (e.g. ``"000006"``).
            Used to build the ``source_reference`` provenance string;
            optional when only the asset_id is known.
        version (str): Dandiset version. Recorded on metadata.
        download_dir (str | None): Directory the downloaded file lives
            in. When ``None``, a :class:`tempfile.TemporaryDirectory`
            is used and the file is deleted after the load. When a
            path is supplied, the directory is created if needed and
            the file is kept — caller manages cleanup. The file path
            is then recorded on ``SpikeData.metadata`` as
            ``downloaded_path``.
        api_token (str | None): Personal access token. Required for
            embargoed dandisets. Defaults to the ``DANDI_API_TOKEN``
            env var when not supplied.
        api_base (str): API root override.
        request_timeout_seconds (float): Per-API-call timeout (asset
            detail lookup, etc.).
        download_timeout_seconds (float): Per-download timeout. Default
            10 min covers ~50 GB at 100 Mbps; raise for very large
            assets on slow links.
        allow_no_units (bool): Passed through to
            :func:`load_spikedata_from_nwb`. ``True`` lets metadata-
            only callers load files without a Units table.
        length_ms (float | None): Passed through.
        start_time_ms (float | None): Passed through.

    Returns:
        sd (SpikeData): Loaded spike data. ``sd.metadata`` carries all
            the keys :func:`load_spikedata_from_nwb` populates, plus
            DANDI-specific fields: ``dandi_asset_id``,
            ``dandi_dandiset_id`` (when ``dandiset_id`` is provided),
            ``dandi_version``, ``source_reference`` (DANDI URL),
            ``downloaded_path`` (only when ``download_dir`` is supplied).

    Raises:
        urllib.error.URLError: On network / HTTP failure.
        ValueError / ImportError: From the delegated NWB load.

    Notes:
        Public dandisets work without authentication. Embargoed
        dandisets need a personal access token (Account → My Tokens on
        dandiarchive.org). Streaming download keeps memory bounded
        independent of asset size; on-disk space proportional to the
        file is required.

        DANDI also hosts raw recordings (NWB files with an
        ``ElectricalSeries`` acquisition but no ``Units`` table). This
        loader does NOT materialise the raw voltage traces — the
        function name signals "spike data only". For metadata triage
        on raw assets, pass ``allow_no_units=True``: the returned
        SpikeData has ``N=0`` but ``metadata`` is fully populated
        (subject, session, ``electrodes_by_channel``,
        ``sampling_rate_hz``, ``duration_seconds``, etc.). Loading the
        raw ElectricalSeries as a SpikeInterface ``BaseRecording`` is
        a separate operation; pair this loader's metadata triage with
        SpikeInterface's NWB reader on the ``downloaded_path`` for
        that case.
    """
    import tempfile as _tempfile

    if api_token is None:
        api_token = os.environ.get("DANDI_API_TOKEN")

    # Resolve download URL. Caller may supply either an asset id (we
    # look up the detail endpoint to get the URL) or a full URL — the
    # latter shape lets batch workflows reuse already-listed assets
    # without re-querying.
    if asset_id.startswith("http://") or asset_id.startswith("https://"):
        download_url = asset_id
        asset_path_hint = ""
        resolved_asset_id = asset_id.rsplit("/", 1)[-1] or asset_id
    else:
        detail_url = f"{api_base.rstrip('/')}/assets/{asset_id}/"
        try:
            detail = _dandi_json_get(
                detail_url,
                api_token=api_token,
                timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:
            raise ValueError(
                f"DANDI asset detail fetch failed for {asset_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        content_urls = detail.get("contentUrl") or detail.get("contentUrls") or ()
        if isinstance(content_urls, str):
            content_urls = [content_urls]
        download_url = str(
            detail.get("download_url")
            or (content_urls[0] if content_urls else "")
            or f"{api_base.rstrip('/')}/assets/{asset_id}/download/"
        )
        asset_path_hint = str(detail.get("path") or "")
        resolved_asset_id = asset_id

    # Manage the download directory's lifecycle. Caller-supplied dirs
    # are kept around after the load; tempdirs are cleaned up.
    cleanup = None
    if download_dir is None:
        cleanup = _tempfile.TemporaryDirectory(prefix="dandi-asset-")
        download_root = cleanup.name
    else:
        os.makedirs(download_dir, exist_ok=True)
        download_root = download_dir

    try:
        # Pick a non-traversal filename. DANDI paths are
        # forward-slash-separated and well-behaved, but defensive
        # ``os.path.basename`` strips any directory components and any
        # ``..`` segments that might appear.
        rel = os.path.basename(asset_path_hint) or f"{resolved_asset_id}.nwb"
        dest = os.path.join(download_root, rel)
        _dandi_download_asset(
            download_url,
            dest,
            api_token=api_token,
            timeout_seconds=download_timeout_seconds,
        )

        sd = load_spikedata_from_nwb(
            dest,
            prefer_pynwb=True,
            length_ms=length_ms,
            start_time_ms=start_time_ms,
            allow_no_units=allow_no_units,
        )

        # Stamp DANDI provenance onto SpikeData.metadata. ``metadata``
        # is always a dict at this point — load_spikedata_from_nwb
        # initialises it with at least source_file + format keys.
        if sd.metadata is not None:
            sd.metadata["dandi_asset_id"] = resolved_asset_id
            if dandiset_id:
                sd.metadata["dandi_dandiset_id"] = dandiset_id
            sd.metadata["dandi_version"] = version
            if dandiset_id:
                sd.metadata["source_reference"] = (
                    f"dandi://dandiarchive.org/dandisets/{dandiset_id}"
                    f"/versions/{version}/assets/{resolved_asset_id}"
                )
            else:
                sd.metadata["source_reference"] = (
                    f"dandi://dandiarchive.org/assets/{resolved_asset_id}"
                )
            if download_dir is not None:
                # Caller owns the directory; record where the file
                # ended up so the gateway / analysis caller can find it
                # for content-hashing or further processing.
                sd.metadata["downloaded_path"] = dest
        return sd
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def load_recording_from_dandi(
    asset_id: str,
    zarr_dest: str,
    *,
    dandiset_id: Optional[str] = None,
    version: str = "draft",
    electrical_series_path: Optional[str] = None,
    overwrite: bool = False,
    download_dir: Optional[str] = None,
    keep_nwb: bool = False,
    api_token: Optional[str] = None,
    api_base: str = _DANDI_API_BASE,
    request_timeout_seconds: float = 30.0,
    download_timeout_seconds: float = 600.0,
    save_kwargs: Optional[dict] = None,
) -> dict:
    """Download a DANDI NWB asset and convert its raw ElectricalSeries
    to SpikeInterface Zarr format.

    Complementary to :func:`load_spikedata_from_dandi`: that one is for
    pre-sorted Units tables; this one is for the raw voltage traces
    (the ``ElectricalSeries`` acquisition objects DANDI hosts but that
    no analysis tooling pre-processes for you). Output is a Zarr
    directory that any consumer of SpikeInterface — e.g. a spike
    sorter run later — can re-open via
    :func:`spikeinterface.core.read_zarr_recording`.

    Parameters:
        asset_id (str): DANDI asset UUID or a fully-qualified asset URL.
            Same shapes accepted by :func:`load_spikedata_from_dandi`.
        zarr_dest (str): Target Zarr directory. Created if absent.
        dandiset_id (str | None): Owning dandiset id, used for the
            ``source_reference`` provenance string.
        version (str): Dandiset version. Recorded on metadata.
        electrical_series_path (str | None): When the NWB file has
            multiple ElectricalSeries objects, the HDMF location of the
            one to convert (e.g. ``"acquisition/ElectricalSeriesAP"``).
            ``None`` (default) lets SpikeInterface auto-pick — works
            when there's exactly one.
        overwrite (bool): When ``True``, an existing ``zarr_dest`` is
            removed first. When ``False`` (default), an existing target
            raises.
        download_dir (str | None): Directory the downloaded ``.nwb``
            lives in. ``None`` uses a tempdir. The NWB is removed after
            the Zarr write unless ``keep_nwb=True``.
        keep_nwb (bool): When ``True``, the downloaded ``.nwb`` is left
            on disk after the Zarr is written. Useful for callers that
            want to content-hash the original bytes (e.g. gateway
            ingestion). The downloaded path is included in the return
            dict under ``downloaded_nwb_path``.
        api_token (str | None): DANDI personal access token. Defaults
            to ``DANDI_API_TOKEN`` env var.
        api_base (str): API root override.
        request_timeout_seconds (float): Per-API-call timeout.
        download_timeout_seconds (float): Per-download timeout.
        save_kwargs (dict | None): Forwarded to
            :meth:`BaseRecording.save` — e.g. ``{"n_jobs": 4,
            "chunk_duration_s": 1.0}``. Defaults to ``{}``.

    Returns:
        dict: Conversion outcome with the following keys:
            * ``zarr_path``: Absolute path to the Zarr directory.
            * ``recording_metadata_path``: JSON sidecar (DANDI provenance
              + NWB file-level metadata + recording shape).
            * ``downloaded_nwb_path``: Present only when ``keep_nwb`` is
              True. Path to the source NWB file.
            * ``dandi_asset_id``, ``dandi_dandiset_id`` (when supplied),
              ``dandi_version``, ``source_reference``: provenance.
            * ``sampling_rate_hz``, ``n_channels``, ``n_samples``,
              ``duration_seconds``: recording shape, surfaced from the
              SpikeInterface extractor for callers that don't want to
              re-open the Zarr just to check.
            * Subject + session fields merged from the NWB metadata
              (when present): ``identifier``, ``subject_id``,
              ``species``, ``sex``, ``session_start_time``, etc.

    Raises:
        ImportError: If ``spikeinterface`` (or its NWB extractor) isn't
            installed.
        ValueError: If asset detail fetch fails or the NWB file has no
            ElectricalSeries the extractor can resolve.
        FileExistsError: If ``zarr_dest`` exists and ``overwrite=False``.

    Notes:
        Streaming download keeps memory bounded. Zarr writes are
        proportional to the recording size — a 1-hour, 384-channel,
        30 kHz Neuropixels session is ~80 GB raw → ~20–40 GB with the
        default LZ4 compressor. Plan disk accordingly.

        SpikeInterface's NWB extractor decides chunking + dtype from
        the source ElectricalSeries. Pass ``save_kwargs={"n_jobs": N}``
        to parallelise the chunk write for large recordings.

        For DANDI assets that ARE pre-sorted (Units table present), use
        :func:`load_spikedata_from_dandi` instead — that path stops at
        the spike trains rather than rewriting the voltage traces.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    try:
        from spikeinterface.extractors import NwbRecordingExtractor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "load_recording_from_dandi requires spikeinterface; install via "
            "the spikelab[io] extra (which pulls spikeinterface + zarr) "
            "or `pip install spikeinterface`."
        ) from exc

    if api_token is None:
        api_token = os.environ.get("DANDI_API_TOKEN")

    if os.path.exists(zarr_dest):
        if not overwrite:
            raise FileExistsError(
                f"zarr_dest {zarr_dest!r} already exists; pass "
                "overwrite=True to replace it."
            )
        _shutil.rmtree(zarr_dest)
    os.makedirs(os.path.dirname(os.path.abspath(zarr_dest)) or ".", exist_ok=True)

    # Resolve download URL (same shape as load_spikedata_from_dandi).
    if asset_id.startswith("http://") or asset_id.startswith("https://"):
        download_url = asset_id
        asset_path_hint = ""
        resolved_asset_id = asset_id.rsplit("/", 1)[-1] or asset_id
    else:
        detail_url = f"{api_base.rstrip('/')}/assets/{asset_id}/"
        try:
            detail = _dandi_json_get(
                detail_url,
                api_token=api_token,
                timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:
            raise ValueError(
                f"DANDI asset detail fetch failed for {asset_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        content_urls = detail.get("contentUrl") or detail.get("contentUrls") or ()
        if isinstance(content_urls, str):
            content_urls = [content_urls]
        download_url = str(
            detail.get("download_url")
            or (content_urls[0] if content_urls else "")
            or f"{api_base.rstrip('/')}/assets/{asset_id}/download/"
        )
        asset_path_hint = str(detail.get("path") or "")
        resolved_asset_id = asset_id

    # Manage the download dir.
    cleanup = None
    if download_dir is None:
        cleanup = _tempfile.TemporaryDirectory(prefix="dandi-recording-")
        download_root = cleanup.name
    else:
        os.makedirs(download_dir, exist_ok=True)
        download_root = download_dir

    nwb_path = None
    try:
        rel = os.path.basename(asset_path_hint) or f"{resolved_asset_id}.nwb"
        nwb_path = os.path.join(download_root, rel)
        _dandi_download_asset(
            download_url,
            nwb_path,
            api_token=api_token,
            timeout_seconds=download_timeout_seconds,
        )

        # File-level metadata via pynwb. Reuses the same helper as the
        # NWB spike-train loader so metadata shape is uniform across
        # both paths.
        nwb_metadata: dict = {}
        try:
            from pynwb import NWBHDF5IO  # type: ignore

            with NWBHDF5IO(nwb_path, mode="r", load_namespaces=True) as io:
                _nwb = io.read()
                nwb_metadata = _nwb_collect_file_metadata(_nwb)
        except Exception as exc:
            warnings.warn(
                f"load_recording_from_dandi: NWB metadata read failed "
                f"({type(exc).__name__}: {exc}); proceeding with empty "
                "metadata sidecar.",
                stacklevel=2,
            )

        # SpikeInterface extractor + Zarr write.
        extractor_kwargs: Dict[str, object] = {}
        if electrical_series_path is not None:
            extractor_kwargs["electrical_series_path"] = electrical_series_path
        recording = NwbRecordingExtractor(nwb_path, **extractor_kwargs)

        sample_rate = float(recording.get_sampling_frequency())
        n_channels = int(recording.get_num_channels())
        n_samples = int(recording.get_num_frames())
        duration_seconds = n_samples / sample_rate if sample_rate > 0 else None

        sk = dict(save_kwargs or {})
        recording.save(
            folder=zarr_dest,
            format="zarr",
            overwrite=False,  # we already enforced the policy above
            **sk,
        )
        # Release the extractor's file handle BEFORE we try to delete
        # the source NWB. SpikeInterface's NwbRecordingExtractor holds
        # an h5py handle via pynwb that doesn't auto-release on save;
        # leaving it open makes ``os.remove(nwb_path)`` fail on
        # Windows (file-in-use), and the subsequent tempdir cleanup
        # misinterprets that error and tries to rmtree the .nwb as a
        # directory. ``del`` + ``gc.collect`` forces finalisation.
        del recording
        import gc as _gc

        _gc.collect()

        # Drop a JSON sidecar so consumers reading the Zarr later have
        # provenance + file-level metadata without re-opening the
        # source NWB.
        sidecar: Dict[str, object] = {
            "format": "spikeinterface_zarr",
            "dandi_asset_id": resolved_asset_id,
            "dandi_version": version,
            "sampling_rate_hz": sample_rate,
            "n_channels": n_channels,
            "n_samples": n_samples,
        }
        if dandiset_id:
            sidecar["dandi_dandiset_id"] = dandiset_id
            sidecar["source_reference"] = (
                f"dandi://dandiarchive.org/dandisets/{dandiset_id}"
                f"/versions/{version}/assets/{resolved_asset_id}"
            )
        else:
            sidecar["source_reference"] = (
                f"dandi://dandiarchive.org/assets/{resolved_asset_id}"
            )
        if duration_seconds is not None:
            sidecar["duration_seconds"] = duration_seconds
        # Merge file-level NWB metadata last so it overlays the
        # computed sidecar fields (the NWB unit_count + duration are
        # the same conceptually as the SI-derived ones; keep the SI
        # values authoritative for the recording-shape fields).
        for k, v in nwb_metadata.items():
            sidecar.setdefault(k, v)

        sidecar_path = os.path.join(zarr_dest, "recording_metadata.json")
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2, default=str)

        # Build the return dict (a flat copy of sidecar plus paths).
        result: Dict[str, object] = dict(sidecar)
        result["zarr_path"] = os.path.abspath(zarr_dest)
        result["recording_metadata_path"] = os.path.abspath(sidecar_path)
        if keep_nwb:
            result["downloaded_nwb_path"] = os.path.abspath(nwb_path)
        return result
    finally:
        if not keep_nwb and nwb_path is not None and os.path.isfile(nwb_path):
            try:
                os.remove(nwb_path)
            except OSError:
                pass
        if cleanup is not None:
            cleanup.cleanup()


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
