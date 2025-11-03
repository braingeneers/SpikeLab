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
import pandas as pd

import zipfile
import os

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore


try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore

from spikedata import SpikeData, NeuronAttributes

__all__ = [
    "load_spikedata_from_hdf5",
    "load_spikedata_from_hdf5_raw_thresholded",
    "load_spikedata_from_nwb",
    "load_spikedata_from_kilosort",
    "load_spikedata_from_spikeinterface",
    "load_spikedata_from_spikeinterface_recording",
    "load_spikedata_from_acqm",
    "download_s3_to_local",
    "load_spikedata_from_s3",
]


def _ensure_h5py():
    """Ensure the optional h5py dependency is available.

    Raises
    ------
    ImportError
        If h5py is not installed and an HDF5/NWB loader is invoked.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5/NWB loaders. `pip install h5py`.")


def _ensure_boto3():
    """Ensure the optional boto3 dependency is available.

    Raises
    ------
    ImportError
        If boto3 is not installed and an S3 loader is invoked.
    """
    if boto3 is None:
        raise ImportError("boto3 is required for S3 loaders. `pip install boto3`.")

def _to_ms(values: np.ndarray, unit: str, fs_Hz: Optional[float]) -> np.ndarray:
    """Convert a vector of times to milliseconds.

    Parameters
    ----------
    values : np.ndarray
        Time values.
    unit : str
        's' (seconds), 'ms' (milliseconds), or 'samples'.
    fs_Hz : float | None
        Sampling frequency (Hz). Required when unit == 'samples'.
    """
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


def _load_neuron_attributes_from_hdf5(
    f,  # h5py.File-like
    n_neurons: int,
) -> Optional[NeuronAttributes]:
    """Load neuron attributes from HDF5 file if /neuron_attributes group exists."""
    if 'neuron_attributes' not in f:
        return None

    try:
        attr_group = f['neuron_attributes']
        attr_data = {}

        # Read each dataset in the group as a column
        for key in attr_group.keys():
            data = np.asarray(attr_group[key])
            if len(data) == n_neurons:
                attr_data[key] = data

        if attr_data:
            return NeuronAttributes.from_dict(attr_data, n_neurons=n_neurons)
    except Exception as e:
        warnings.warn(f"Failed to load neuron_attributes from HDF5: {e}")

    return None


def _read_raw_arrays(
    f,  # h5py.File-like
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
    neuron_attributes: Optional[NeuronAttributes] = None,
) -> SpikeData:
    """Internal helper to construct a SpikeData with sensible defaults.

    - Infers `length_ms` from the last spike if not provided.
    - Copies metadata and attaches optional raw arrays.
    - Accepts optional neuron_attributes.
    """
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
# S3
# ----------------------------

def download_s3_to_local(
    src: str,
    dst: str,
    *,
    endpoint_url: str = 'https://s3-west.nrp-nautilus.io',
    **s3_client_kwargs
) -> None:
    """Download a file from S3 to local filesystem.
    
    Parameters
    ----------
    src : str
        S3 URI in the format 's3://bucket/key/path'.
    dst : str
        Local destination file path.
    endpoint_url : str, optional
        S3 endpoint URL. Defaults to Nautilus endpoint.
    **s3_client_kwargs
        Additional keyword arguments passed to boto3.client().
        
    Raises
    ------
    RuntimeError
        If src doesn't start with 's3://'.
    ImportError
        If boto3 is not installed.
        
    Examples
    --------
    >>> download_s3_to_local('s3://my-bucket/data.h5', '/tmp/data.h5')
    >>>
    """
    _ensure_boto3()
    
    if not src.startswith('s3://'):
        raise RuntimeError(f'Input filepath must start with s3://! Got: {src}')
    
    print(f'Downloading {src} to {dst} with boto3 version: {boto3.__version__}')
    
    # Parse S3 URI
    bucket, key = src[len('s3://'):].split('/', maxsplit=1)
    
    # Create S3 client
    s3_client = boto3.client('s3', endpoint_url=endpoint_url, **s3_client_kwargs)
    
    # Download file
    s3_client.download_file(bucket, key, dst)
    print(f'Successfully downloaded to {dst}')


def _resolve_s3_path(
    filepath: str,
    cache_dir: Optional[str] = None,
    endpoint_url: str = 'https://s3-west.nrp-nautilus.io',
) -> str:
    """Resolve an S3 path by downloading to cache if needed.
    
    Parameters
    ----------
    filepath : str
        Either a local path or an S3 URI (s3://...).
    cache_dir : str, optional
        Directory to cache downloaded S3 files. If None, uses a temp directory.
    endpoint_url : str
        S3 endpoint URL.
        
    Returns
    -------
    str
        Local file path (either original if local, or cached if from S3).
    """
    if not filepath.startswith('s3://'):
        return filepath
    
    import tempfile
    
    # Determine cache location
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix='spikedata_s3_')
    else:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Create local filename from S3 key
    basename = os.path.basename(filepath)
    local_path = os.path.join(cache_dir, basename)
    
    # Download if not already cached
    if not os.path.exists(local_path):
        download_s3_to_local(filepath, local_path, endpoint_url=endpoint_url)

    # If the file is a zip file, unzip it
    if local_path.endswith('.zip'):
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        local_path = os.path.join(cache_dir, os.path.basename(local_path).replace('.zip', ''))

    else:
        print(f'Using cached file: {local_path}')
    
    return local_path
    


# ----------------------------
# HDF5
# ----------------------------


def load_spikedata_from_hdf5(
    filepath: str,
    *,
    cache_dir: Optional[str] = None,
    s3_endpoint_url: str = 'https://s3-west.nrp-nautilus.io',
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

    This function provides a flexible interface for loading spike train data from HDF5 files
    that may be organized in different ways. Exactly one of the following four input styles
    must be specified via the corresponding arguments:

    **Input Styles:**

    1. **Raster Matrix**
        - Use when the HDF5 file contains a 2D array representing spike counts or a binary raster.
        - Arguments:
            - `raster_dataset` (str): Path to the dataset containing the raster/counts matrix (shape: units × time).
            - `raster_bin_size_ms` (float): Bin width in milliseconds.
        - The matrix is interpreted as (units × time bins), where each entry is the spike count (or 0/1 for binary).
        - Example: `raster_dataset="/spikes/raster", raster_bin_size_ms=1.0`

    2. **Ragged Arrays (NWB-style)**
        - Use when spike times for all units are concatenated into a single array, with an index array marking the end of each unit's spike times.
        - Arguments:
            - `spike_times_dataset` (str): Path to the flat array of spike times.
            - `spike_times_index_dataset` (str): Path to the array of indices (end positions for each unit).
            - `spike_times_unit` (str): Unit of the spike times ('s', 'ms', or 'samples').
            - `fs_Hz` (float, optional): Required if unit is 'samples'.
        - Example: `spike_times_dataset="/units/spike_times", spike_times_index_dataset="/units/spike_times_index"`

    3. **Group-per-Unit**
        - Use when each unit's spike times are stored as a separate dataset within a group.
        - Arguments:
            - `group_per_unit` (str): Path to the group containing one dataset per unit.
            - `group_time_unit` (str): Unit of the spike times ('s', 'ms', or 'samples').
            - `fs_Hz` (float, optional): Required if unit is 'samples'.
        - Example: `group_per_unit="/spikes/unit_times"`

    4. **Paired Arrays (Indices and Times)**
        - Use when there are two parallel arrays: one for unit indices and one for spike times.
        - Arguments:
            - `idces_dataset` (str): Path to the array of unit indices (int).
            - `times_dataset` (str): Path to the array of spike times.
            - `times_unit` (str): Unit of the spike times ('s', 'ms', or 'samples').
            - `fs_Hz` (float, optional): Required if unit is 'samples'.
        - Example: `idces_dataset="/spikes/unit_ids", times_dataset="/spikes/times"`

    **Optional Raw Data:**
        - You may also attach raw analog data and its timebase by specifying:
            - `raw_dataset` (str): Path to the raw data array.
            - `raw_time_dataset` (str): Path to the time vector for the raw data.
            - `raw_time_unit` (str): Unit of the raw time vector ('s', 'ms', or 'samples').
            - `fs_Hz` (float, required if 'samples'): Sampling frequency for conversion.

    **Parameters**
    ----------
    filepath : str
        Path to the HDF5 file.
    length_ms : float, optional
        Recording duration in milliseconds (inferred if not provided).
    metadata : dict, optional
        Additional metadata to attach to the SpikeData object.

    Returns
    -------
    SpikeData
        The loaded spike train data.

    Raises
    ------
    ValueError
        If not exactly one input style is specified, or if required arguments are missing.

    Examples
    --------
    # Load from a raster matrix
    sd = load_spikedata_from_hdf5("file.h5", raster_dataset="raster", raster_bin_size_ms=1.0)

    # Load from ragged arrays (NWB-style)
    sd = load_spikedata_from_hdf5("file.h5", spike_times_dataset="spike_times", spike_times_index_dataset="spike_times_index")

    # Load from group-per-unit
    sd = load_spikedata_from_hdf5("file.h5", group_per_unit="unit_group")

    # Load from paired arrays
    sd = load_spikedata_from_hdf5("file.h5", idces_dataset="unit_ids", times_dataset="spike_times")

    # Direct S3 support in loaders (if you modify existing functions)
    sd = load_spikedata_from_hdf5('s3://my-bucket/recording.h5',
                                  raster_dataset='spikes',
                                  raster_bin_size_ms=1.0)
    """
    # Resolve S3 path if needed
    filepath = _resolve_s3_path(filepath, cache_dir, s3_endpoint_url)
    
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
            n_neurons = raster.shape[0]
            neuron_attrs = _load_neuron_attributes_from_hdf5(f, n_neurons)
            sd = SpikeData.from_raster(raster, raster_bin_size_ms, neuron_attributes=neuron_attrs)
            sd.metadata.update(meta)
            return _maybe_with_raw(sd, raw_data, raw_time)

        if spike_times_dataset is not None and spike_times_index_dataset is not None:
            # Style (2): flat ragged spike_times + spike_times_index
            flat = np.asarray(f[spike_times_dataset])
            index = np.asarray(f[spike_times_index_dataset])
            trains = _trains_from_flat_index(
                flat, index, unit=spike_times_unit, fs_Hz=fs_Hz
            )
            neuron_attrs = _load_neuron_attributes_from_hdf5(f, len(trains))
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
                neuron_attributes=neuron_attrs,
            )

        if group_per_unit is not None:
            # Style (3): each child dataset is a unit's spike times
            grp = f[group_per_unit]
            keys = sorted(list(grp.keys()))
            trains = [_to_ms(np.asarray(grp[k]), group_time_unit, fs_Hz) for k in keys]
            neuron_attrs = _load_neuron_attributes_from_hdf5(f, len(trains))
            return _build_spikedata(
                trains,
                length_ms=length_ms,
                metadata=meta,
                raw_data=raw_data,
                raw_time=raw_time,
                neuron_attributes=neuron_attrs,
            )

        # Style (4): paired indices and times arrays
        idces = np.asarray(f[idces_dataset])  # type: ignore
        times = _to_ms(np.asarray(f[times_dataset]), times_unit, fs_Hz)  # type: ignore
        N = int(idces.max()) + 1 if idces.size else 0
        neuron_attrs = _load_neuron_attributes_from_hdf5(f, N)
        sd = SpikeData.from_idces_times(idces, times, N=N, length=length_ms, neuron_attributes=neuron_attrs)
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

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    dataset : str
        HDF5 dataset path containing raw traces shaped (channels, time).
    fs_Hz : float
        Sampling frequency in Hz.
    threshold_sigma : float
        Threshold in units of per-channel standard deviation.
    filter : dict | bool
        If True, apply default Butterworth bandpass; if dict, pass to filter; if False, no filtering.
    hysteresis : bool
        Use rising-edge detection if True.
    direction : str
        'both' | 'up' | 'down'.
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
    """Load spike trains from an NWB file's Units table.

    Prefers pynwb; falls back to h5py reading VectorData/VectorIndex at
    '/units/spike_times' and '/units/spike_times_index'. Times are in seconds
    and converted to milliseconds.

    Extracts all available columns from the Units table into neuron_attributes.
    """
    trains: List[np.ndarray] = []
    neuron_attrs_df: Optional[pd.DataFrame] = None
    meta = {"source_file": os.path.abspath(filepath), "format": "NWB"}

    if prefer_pynwb:
        try:
            from pynwb import NWBHDF5IO  # type: ignore

            with NWBHDF5IO(filepath, "r") as io:
                nwb = io.read()
                if getattr(nwb, "units", None) is None:
                    raise ValueError("NWB file has no Units table")
                units_df = nwb.units.to_dataframe()  # type: ignore
                for row in units_df.itertuples():
                    stimes = np.asarray(row.spike_times, dtype=float)
                    trains.append(stimes * 1e3)

                # Extract neuron attributes (excluding spike_times column)
                if len(units_df.columns) > 1:
                    neuron_attrs_df = units_df.drop(columns=['spike_times'], errors='ignore').copy()
                    # Reset index to 0-based integer
                    neuron_attrs_df.reset_index(drop=True, inplace=True)

            neuron_attrs = None
            if neuron_attrs_df is not None and not neuron_attrs_df.empty:
                neuron_attrs = NeuronAttributes.from_dataframe(neuron_attrs_df, n_neurons=len(trains))

            return _build_spikedata(trains, length_ms=length_ms, metadata=meta, neuron_attributes=neuron_attrs)
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

        # Extract other attributes from the units group (h5py fallback)
        attr_data = {}
        for key in unit_grp.keys():
            if key not in [st_key, idx_key] and not key.endswith('_index'):
                try:
                    data = np.asarray(unit_grp[key])
                    # Check if it's a VectorData with matching length
                    if len(data) == len(trains) or (key + '_index' in unit_grp.keys()):
                        # If has an index, it's ragged - skip for now
                        if key + '_index' not in unit_grp.keys():
                            attr_data[key] = data
                except Exception:
                    pass  # Skip columns that can't be read as simple arrays

        neuron_attrs = None
        if attr_data:
            neuron_attrs = NeuronAttributes.from_dict(attr_data, n_neurons=len(trains))

    return _build_spikedata(trains, length_ms=length_ms, metadata=meta, neuron_attributes=neuron_attrs)


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

    Extracts all available unit properties into neuron_attributes.

    Parameters
    ----------
    sorting : object
        Exposes get_unit_ids(), get_sampling_frequency(), get_unit_spike_train(...),
        and optionally get_property_keys() and get_property().
    sampling_frequency : float | None
        Optional override for sampling frequency (Hz).
    unit_ids : sequence | None
        Optional subset of unit IDs to include.
    segment_index : int
        Segment index for multi-segment sortings.
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
    for uid in ids:
        st = np.asarray(get_train(unit_id=uid, segment_index=segment_index))
        trains.append(_to_ms(st.astype(float), "samples", fs))

    # Extract neuron attributes from SpikeInterface properties
    neuron_attrs = None
    try:
        get_property_keys = sorting.get_property_keys  # type: ignore[attr-defined]
        get_property = sorting.get_property  # type: ignore[attr-defined]

        property_keys = get_property_keys()
        if property_keys:
            attr_data = {}
            for prop_key in property_keys:
                try:
                    # Get all property values at once (SpikeInterface API returns array for all units)
                    prop_values = get_property(prop_key, ids)
                    attr_data[prop_key] = list(prop_values)
                except Exception:
                    pass  # Skip properties that can't be extracted

            # Add unit_id column
            attr_data['unit_id'] = ids

            if attr_data:
                neuron_attrs = NeuronAttributes.from_dict(attr_data, n_neurons=len(trains))
    except (AttributeError, Exception):
        # SortingExtractor doesn't have properties or extraction failed
        pass

    meta = {"source_format": "SpikeInterface", "unit_ids": ids, "fs_Hz": fs}
    return _build_spikedata(trains, metadata=meta, neuron_attributes=neuron_attrs)


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
    """
    # misses critical information about waveform data - load if it same in file and put in spikedata

    Load KiloSort/Phy outputs into SpikeData.

    Reads spike_times.npy (samples) and spike_clusters.npy; groups times per cluster
    and converts to ms using fs_Hz. If a TSV is provided, optionally filter to
    "good"/"mua" unless include_noise=True. Stores cluster IDs in metadata and
    extracts attributes from TSV into neuron_attributes.
    """
    st_path = os.path.join(folder, spike_times_file)
    sc_path = os.path.join(folder, spike_clusters_file)
    spike_times = np.load(st_path)
    spike_clusters = np.load(sc_path)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times and spike_clusters length mismatch")

    keep_clusters: Optional[set] = None
    cluster_info_df: Optional[pd.DataFrame] = None

    # Try to load cluster info TSV/CSV
    if cluster_info_tsv is not None:
        tsv_path = os.path.join(folder, cluster_info_tsv)
        if os.path.exists(tsv_path):
            try:
                cluster_info_df = pd.read_csv(tsv_path, sep="\t")
                label_col = (
                    "group"
                    if "group" in cluster_info_df.columns
                    else ("KSLabel" if "KSLabel" in cluster_info_df.columns else None)
                )
                id_col = (
                    "cluster_id"
                    if "cluster_id" in cluster_info_df.columns
                    else ("id" if "id" in cluster_info_df.columns else None)
                )
                if id_col is None or label_col is None:
                    warnings.warn(
                        "Could not find id/label columns in cluster TSV; keeping all clusters"
                    )
                else:
                    if include_noise:
                        keep_clusters = set(cluster_info_df[id_col].astype(int).tolist())
                    else:
                        mask = (
                            cluster_info_df[label_col]
                            .astype(str)
                            .str.lower()
                            .isin(["good", "mua", "mua good"])
                        )  # permissive
                        keep_clusters = set(cluster_info_df.loc[mask, id_col].astype(int).tolist())
            except Exception as e:
                warnings.warn(
                    f"Failed parsing cluster info TSV: {e}; keeping all clusters"
                )
                cluster_info_df = None
    else:
        # Try to find cluster_info.tsv or cluster_group.tsv automatically
        for possible_name in ['cluster_info.tsv', 'cluster_group.tsv', 'cluster_KSLabel.tsv']:
            possible_path = os.path.join(folder, possible_name)
            if os.path.exists(possible_path):
                try:
                    cluster_info_df = pd.read_csv(possible_path, sep="\t")
                    break
                except Exception:
                    pass

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

    # Build neuron_attributes from cluster_info_df
    neuron_attrs = None
    if cluster_info_df is not None:
        # Determine ID column
        id_col = None
        for candidate in ['cluster_id', 'id']:
            if candidate in cluster_info_df.columns:
                id_col = candidate
                break

        if id_col is not None:
            # Filter and reorder to match our trains
            cluster_info_df = cluster_info_df.set_index(id_col)
            attrs_list = []
            for clu_id in metadata_units:
                if clu_id in cluster_info_df.index:
                    attrs_list.append(cluster_info_df.loc[clu_id].to_dict())
                else:
                    # Missing data - use NaN
                    attrs_list.append({col: np.nan for col in cluster_info_df.columns})

            if attrs_list:
                attrs_df = pd.DataFrame(attrs_list)
                # Add cluster_id as a column (standard name)
                attrs_df.insert(0, 'cluster_id', metadata_units)
                neuron_attrs = NeuronAttributes.from_dataframe(attrs_df, n_neurons=len(trains))

    return _build_spikedata(trains, length_ms=length_ms, metadata=meta, neuron_attributes=neuron_attrs)


# ----------------------------
# ACQM (NPZ format)
# ----------------------------


def load_spikedata_from_acqm(
    filepath: str,
    *,
    cache_dir: Optional[str] = None,
    s3_endpoint_url: str = 'https://s3-west.nrp-nautilus.io',
    length_ms: Optional[float] = None,
) -> SpikeData:
    """Load spike trains from an ACQM file (.acqm or .acqm.zip stored as NPZ).

    ACQM files are numpy compressed archives containing spike train data with the following structure:
    - train: dict mapping unit IDs to spike time arrays (in samples)
    - neuron_data: dict mapping unit IDs to metadata dicts with per-neuron attributes
    - fs: sampling frequency in Hz
    - config: optional recording configuration dict
    - redundant_pairs: optional array of redundant unit pairs

    Spike times are converted from samples to milliseconds using the stored sampling frequency.
    Neuron metadata is extracted into neuron_attributes.

    Parameters
    ----------
    filepath : str
        Path to the ACQM file (local or S3 URI starting with 's3://').
        Can be .acqm.zip (handled as npz) or regular .npz file.
    cache_dir : str, optional
        If filepath is an S3 URI, directory to cache downloaded files.
    s3_endpoint_url : str
        S3 endpoint URL (only used if filepath is S3 URI).
    length_ms : float, optional
        Recording duration in milliseconds (inferred if not provided).

    Returns
    -------
    SpikeData
        The loaded spike train data with neuron attributes.

    Raises
    ------
    ValueError
        If the file is missing required fields (train, fs) or has invalid data.
    ImportError
        If boto3 is required but not installed (for S3 URIs).

    Examples
    --------
    # Load from local file
    >>> sd = load_spikedata_from_acqm("recording_acqm.zip")

    # Load from S3
    >>> sd = load_spikedata_from_acqm("s3://my-bucket/data/recording.acqm.zip")

    # Access neuron attributes
    >>> print(sd.neuron_attributes.df['channel'])
    """
    # Resolve S3 path if needed
    filepath = _resolve_s3_path(filepath, cache_dir, s3_endpoint_url)

    # Load the npz file
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load ACQM file '{filepath}': {e}") from e

    # Validate required fields
    if 'train' not in data:
        raise ValueError("ACQM file missing required 'train' field")
    if 'fs' not in data:
        raise ValueError("ACQM file missing required 'fs' (sampling frequency) field")

    # Extract sampling frequency
    fs_Hz = float(data['fs'])
    if fs_Hz <= 0:
        raise ValueError(f"Invalid sampling frequency: {fs_Hz} Hz")

    # Extract train dict and convert to list of spike trains
    train_dict = data['train'].item()
    if not isinstance(train_dict, dict):
        raise ValueError("'train' field must be a dictionary mapping unit IDs to spike times")

    # Sort unit IDs for consistent ordering
    unit_ids = sorted(train_dict.keys())
    trains: List[np.ndarray] = []
    for uid in unit_ids:
        spike_times_samples = np.asarray(train_dict[uid], dtype=float)
        spike_times_ms = _to_ms(spike_times_samples, "samples", fs_Hz)
        trains.append(np.sort(spike_times_ms))  # Ensure sorted

    # Extract neuron attributes if available
    neuron_attrs = None
    if 'neuron_data' in data:
        try:
            neuron_data_dict = data['neuron_data'].item()
            if isinstance(neuron_data_dict, dict) and neuron_data_dict:
                # Build a dataframe from neuron_data
                attrs_list = []
                for uid in unit_ids:
                    if uid in neuron_data_dict:
                        neuron_info = neuron_data_dict[uid]
                        if isinstance(neuron_info, dict):
                            # Flatten nested structures (keep only scalar/array values)
                            flat_info = {}
                            for key, value in neuron_info.items():
                                # Skip large arrays like waveforms, neighbor_templates
                                if key in ['waveforms', 'neighbor_templates', 'amplitudes']:
                                    continue
                                # Convert arrays to strings/tuples for scalar storage
                                if isinstance(value, np.ndarray):
                                    if value.size <= 3:  # Small arrays like position
                                        flat_info[key] = tuple(value.tolist())
                                    elif key in ['neighbor_channels', 'neighbor_positions']:
                                        # Store as comma-separated string for readability
                                        flat_info[key] = str(value.tolist())
                                    else:
                                        # Skip large arrays
                                        continue
                                else:
                                    flat_info[key] = value
                            attrs_list.append(flat_info)
                        else:
                            attrs_list.append({})
                    else:
                        attrs_list.append({})

                if attrs_list:
                    attrs_df = pd.DataFrame(attrs_list)
                    # Ensure unit_id column exists
                    if 'unit_id' not in attrs_df.columns:
                        attrs_df.insert(0, 'unit_id', unit_ids)
                    neuron_attrs = NeuronAttributes.from_dataframe(attrs_df, n_neurons=len(trains))
        except Exception as e:
            warnings.warn(f"Failed to load neuron_data from ACQM: {e}")

    # Build metadata
    meta = {
        "source_file": os.path.abspath(filepath),
        "source_format": "ACQM",
        "unit_ids": unit_ids,
        "fs_Hz": fs_Hz,
    }

    # Add config if present
    if 'config' in data:
        try:
            config = data['config'].item()
            if isinstance(config, dict):
                meta['config'] = config
        except Exception:
            pass

    # Add redundant_pairs if present
    if 'redundant_pairs' in data:
        try:
            redundant_pairs = np.asarray(data['redundant_pairs'])
            if redundant_pairs.size > 0:
                meta['redundant_pairs'] = redundant_pairs.tolist()
        except Exception:
            pass

    return _build_spikedata(
        trains,
        length_ms=length_ms,
        metadata=meta,
        neuron_attributes=neuron_attrs,
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

    Spikes are detected by thresholding the raw traces. The orientation of
    the returned trace matrix is inferred (smaller dimension assumed channels).

    Expected `recording` interface (duck-typed):
      - get_traces(segment_index=..., ...) -> ndarray (channels,time) or (time,channels)
      - sampling_frequency attribute or get_sampling_frequency() method
      - get_num_channels() is optional and not strictly required
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
