"""
Data exporters that mirror data_loaders, writing SpikeData to common formats.

Provided exporters:
- HDF5 generic, with four styles:
  1) raster (units x time) with a specified bin size in ms
  2) ragged arrays (flat spike_times + spike_times_index)
  3) group-per-unit (one dataset per unit)
  4) paired arrays (idces + times)
- NWB Units table (spike_times/spike_times_index) via h5py
- KiloSort/Phy (spike_times.npy + spike_clusters.npy)

All exporters accept SpikeData times in milliseconds and convert to the
target time units as needed.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import os
import warnings

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore

import pickle

if TYPE_CHECKING:  # avoid runtime circular import
    from ..spikedata import SpikeData  # noqa: F401

from ..spikedata.utils import TimeUnit, ensure_h5py, times_from_ms


def export_spikedata_to_hdf5(
    sd: "SpikeData",
    filepath: str,
    *,
    style: Literal["raster", "ragged", "group", "paired"] = "ragged",
    # raster
    raster_dataset: str = "raster",
    raster_bin_size_ms: Optional[float] = None,
    # ragged
    spike_times_dataset: str = "spike_times",
    spike_times_index_dataset: str = "spike_times_index",
    spike_times_unit: TimeUnit = "s",
    fs_Hz: Optional[float] = None,
    # group-per-unit
    group_per_unit: str = "units",
    group_time_unit: TimeUnit = "s",
    # paired arrays
    idces_dataset: str = "idces",
    times_dataset: str = "times",
    times_unit: TimeUnit = "ms",
    # optional raw arrays (written if present and destinations provided)
    raw_dataset: Optional[str] = None,
    raw_time_dataset: Optional[str] = None,
    raw_time_unit: TimeUnit = "ms",
) -> None:
    """Export a SpikeData to a generic HDF5 file using a chosen style.

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        filepath (str): Path where the HDF5 file will be created
            (overwrites existing).
        style (Literal["raster", "ragged", "group", "paired"]): Export
            format style. "raster": 2D binary/count matrix (units x time
            bins). "ragged": flat concatenated spike times with cumulative
            indices. "group": one HDF5 group containing one dataset per
            unit. "paired": two parallel arrays of unit indices and spike
            times.
        raster_dataset (str): HDF5 dataset name for the raster matrix.
        raster_bin_size_ms (float | None): Bin size in milliseconds for
            rasterization. Required for raster style.
        spike_times_dataset (str): Dataset name for concatenated spike
            times.
        spike_times_index_dataset (str): Dataset name for cumulative
            spike count indices.
        spike_times_unit (TimeUnit): Time unit for spike times
            ('ms', 's', 'samples').
        fs_Hz (float | None): Sampling frequency in Hz. Required when
            any unit is 'samples'.
        group_per_unit (str): HDF5 group name containing per-unit
            datasets.
        group_time_unit (TimeUnit): Time unit for individual unit
            datasets.
        idces_dataset (str): Dataset name for unit indices array.
        times_dataset (str): Dataset name for spike times array.
        times_unit (TimeUnit): Time unit for spike times.
        raw_dataset (str | None): Dataset name for raw analog data
            (if present in sd).
        raw_time_dataset (str | None): Dataset name for raw data time
            vector.
        raw_time_unit (TimeUnit): Time unit for raw data timestamps.

    Raises:
        ImportError: If h5py is not available.
        ValueError: If style is invalid, required parameters are missing,
            or fs_Hz is needed but not provided for 'samples' unit.

    Notes:
        - Spike times are automatically converted from milliseconds to
          the requested unit.
        - The function creates or overwrites the target HDF5 file.
        - Raw data is only written if both raw_dataset and
          raw_time_dataset are provided and the SpikeData contains
          raw_data and raw_time attributes.
        - For raster style, the bin size is stored as an attribute for
          provenance.
        - Parameters mirror the corresponding loader function to ease
          round-tripping.
        - ``neuron_attributes`` and ``metadata`` are not persisted in the
          generic HDF5 format. Use ``AnalysisWorkspace.save`` (workspace
          HDF5) or ``export_spikedata_to_pickle`` for full-fidelity
          round-trips.
    """
    ensure_h5py()

    style = style.lower()  # normalize
    valid_styles = {"raster", "ragged", "group", "paired"}
    if style not in valid_styles:
        raise ValueError(
            f"Unknown style '{style}' (choose one of {sorted(valid_styles)})"
        )

    # Create or overwrite the HDF5 file
    with h5py.File(filepath, "w") as f:  # type: ignore
        # Store start_time for event-centered data round-trips
        f.attrs["start_time"] = float(sd.start_time)

        # Optionally write raw arrays if destinations are provided and data exist
        if (
            raw_dataset
            and raw_time_dataset
            and getattr(sd, "raw_data", None) is not None
            and sd.raw_data.size > 0
        ):
            f.create_dataset(raw_dataset, data=np.asarray(sd.raw_data))
            # Export raw_time converted to the requested unit
            raw_time = np.asarray(sd.raw_time)
            if raw_time_unit == "ms":
                raw_time_out = raw_time
            elif raw_time_unit == "s":
                raw_time_out = raw_time / 1e3
            elif raw_time_unit == "samples":
                if not fs_Hz or fs_Hz <= 0:
                    raise ValueError(
                        "fs_Hz must be provided for raw_time_unit='samples'"
                    )
                raw_time_out = np.rint(raw_time * (fs_Hz / 1e3)).astype(int)
            else:
                raise ValueError("raw_time_unit must be one of 's','ms','samples'")
            f.create_dataset(raw_time_dataset, data=raw_time_out)

        if style == "raster":
            if raster_bin_size_ms is None or raster_bin_size_ms <= 0:
                raise ValueError(
                    "raster_bin_size_ms must be provided and > 0 for raster style"
                )
            raster = sd.raster(raster_bin_size_ms)
            f.create_dataset(raster_dataset, data=np.asarray(raster))
            # Store bin size as an attribute for provenance (readers can ignore)
            f[raster_dataset].attrs["bin_size_ms"] = float(raster_bin_size_ms)
            return  # file-level attr (start_time) already written above

        if style == "ragged":
            # Flatten all trains and write cumulative end indices
            counts = [len(t) for t in sd.train]
            flat_ms = np.concatenate(sd.train) if sum(counts) else np.array([], float)
            flat = times_from_ms(flat_ms, spike_times_unit, fs_Hz)
            index = np.cumsum(counts, dtype=int)
            f.create_dataset(spike_times_dataset, data=flat)
            f.create_dataset(spike_times_index_dataset, data=index)
            return

        if style == "group":
            grp = f.create_group(group_per_unit)
            for i, tms in enumerate(sd.train):
                grp.create_dataset(
                    str(i), data=times_from_ms(np.asarray(tms), group_time_unit, fs_Hz)
                )
            return

        # paired
        idces: list[int] = []
        times_ms: list[float] = []
        for unit_index, tms in enumerate(sd.train):
            if len(tms) == 0:
                continue
            idces.extend([unit_index] * len(tms))
            times_ms.extend(tms.tolist())
        idces_arr = np.array(idces, dtype=int)
        times_arr = times_from_ms(np.array(times_ms, dtype=float), times_unit, fs_Hz)
        f.create_dataset(idces_dataset, data=idces_arr)
        f.create_dataset(times_dataset, data=times_arr)


def export_spikedata_to_nwb(
    sd: "SpikeData",
    filepath: str,
    *,
    spike_times_dataset: str = "spike_times",
    spike_times_index_dataset: str = "spike_times_index",
    group: str = "units",
) -> None:
    """Export SpikeData to a minimal NWB-like file using h5py.

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        filepath (str): Path where the NWB file will be created
            (overwrites existing).
        spike_times_dataset (str): Name of the dataset containing
            concatenated spike times. Default is "spike_times" per NWB
            convention.
        spike_times_index_dataset (str): Name of the dataset containing
            cumulative indices. Default is "spike_times_index" per NWB
            convention.
        group (str): Name of the HDF5 group to contain the datasets.
            Default is "units" per NWB convention.

    Raises:
        ImportError: If h5py is not available.

    Notes:
        - Spike times are automatically converted from milliseconds to
          seconds.
        - The output file structure follows NWB conventions but is
          minimal (does not include full NWB metadata or schema
          validation).
        - Empty units (no spikes) are handled correctly in the index
          array.
        - This is compatible with the load_spikedata_from_nwb function
          when prefer_pynwb=False.
    """
    ensure_h5py()
    if sd.start_time != 0:
        warnings.warn(
            f"Exporting event-centered SpikeData (start_time={sd.start_time}) "
            "to NWB. The NWB format does not store start_time, so spike times "
            "are written as-is. On reload, start_time will default to 0.",
            UserWarning,
        )
    counts = [len(t) for t in sd.train]
    flat_ms = np.concatenate(sd.train) if sum(counts) else np.array([], float)
    flat_s = times_from_ms(flat_ms, "s", fs_Hz=None)
    index = np.cumsum(counts, dtype=int)
    with h5py.File(filepath, "w") as f:  # type: ignore
        g = f.create_group(group)
        g.create_dataset(spike_times_dataset, data=flat_s)
        g.create_dataset(spike_times_index_dataset, data=index)
        g.create_dataset("id", data=np.arange(sd.N, dtype=int))

        electrodes = sd.electrodes
        if electrodes is not None:
            g.create_dataset("electrodes", data=electrodes)
            g.create_dataset("electrodes_index", data=np.arange(1, sd.N + 1, dtype=int))

        unit_locations = sd.unit_locations
        if unit_locations is not None:
            elec_grp = f.create_group("general/extracellular_ephys/electrodes")
            locations = unit_locations

            # Build electrodes table IDs to be consistent with units/electrodes.
            if electrodes is not None:
                elec_ids = np.asarray(sd.electrodes, dtype=int)
                # Unique electrode IDs and representative indices into unit_locations
                unique_ids, first_indices = np.unique(elec_ids, return_index=True)
                # Sort by electrode ID for a stable, ordered table
                sort_idx = np.argsort(unique_ids)
                unique_ids = unique_ids[sort_idx]
                first_indices = first_indices[sort_idx]

                elec_grp.create_dataset("id", data=unique_ids)
                elec_locations = locations[first_indices]
            else:
                # Fallback: no explicit electrode IDs; use 0..N-1 as before
                elec_grp.create_dataset("id", data=np.arange(sd.N, dtype=int))
                elec_locations = locations

            elec_grp.create_dataset("x", data=elec_locations[:, 0])
            if elec_locations.shape[1] > 1:
                elec_grp.create_dataset("y", data=elec_locations[:, 1])
            if elec_locations.shape[1] > 2:
                elec_grp.create_dataset("z", data=elec_locations[:, 2])


def export_spikedata_to_kilosort(
    sd: "SpikeData",
    folder: str,
    *,
    fs_Hz: float,
    spike_times_file: str = "spike_times.npy",
    spike_clusters_file: str = "spike_clusters.npy",
    time_unit: TimeUnit = "samples",
    cluster_ids: Optional[Sequence[int]] = None,
) -> Tuple[str, str]:
    """Export SpikeData to a KiloSort/Phy-like folder.

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        folder (str): Directory path where the .npy files will be
            created. Created if it doesn't exist.
        fs_Hz (float): Sampling frequency in Hz. Required for time unit
            conversion, especially when time_unit='samples'.
        spike_times_file (str): Filename for the spike times array.
            Default is "spike_times.npy".
        spike_clusters_file (str): Filename for the spike clusters array.
            Default is "spike_clusters.npy".
        time_unit (TimeUnit): Time unit for output spike times.
            'samples': integer sample indices (default, KiloSort
            standard). 'ms': milliseconds (float). 's': seconds (float).
        cluster_ids (Sequence[int] | None): Custom cluster IDs for each
            unit. If None, uses sequential integers 0, 1, 2, ... Length
            must match sd.N.

    Returns:
        paths (tuple[str, str]): Paths to the created spike_times.npy
            and spike_clusters.npy files.

    Notes:
        - The output arrays have the same length (one entry per spike
          across all units).
        - Spike times are sorted by unit order, not chronologically.
        - Empty units (no spikes) don't contribute entries to the output
          arrays.
        - The 'samples' time unit produces integer arrays suitable for
          KiloSort/Phy.
        - Cluster IDs can be arbitrary integers and don't need to be
          sequential.
    """
    if not fs_Hz or fs_Hz <= 0:
        raise ValueError("A positive fs_Hz is required for KiloSort export")
    if sd.start_time != 0:
        warnings.warn(
            f"Exporting event-centered SpikeData (start_time={sd.start_time}) "
            "to KiloSort. The format does not store start_time, so spike times "
            "are written as-is. On reload, start_time will default to 0.",
            UserWarning,
        )
    os.makedirs(folder, exist_ok=True)

    # Build flat arrays
    idces: list[int] = []
    times_ms: list[float] = []
    for unit_index, tms in enumerate(sd.train):
        if len(tms) == 0:
            continue
        idces.extend([unit_index] * len(tms))
        times_ms.extend(tms.tolist())

    # Map units -> cluster ids
    if cluster_ids is None:
        cluster_ids = list(range(sd.N))
    if len(cluster_ids) != sd.N:
        raise ValueError("cluster_ids length must match sd.N")
    clusters = np.array([int(cluster_ids[i]) for i in idces], dtype=int)

    # Convert times
    if time_unit == "samples":
        times_out = times_from_ms(np.array(times_ms, dtype=float), "samples", fs_Hz)
    elif time_unit == "ms":
        times_out = np.array(times_ms, dtype=float)
    elif time_unit == "s":
        times_out = np.array(times_ms, dtype=float) / 1e3
    else:
        raise ValueError("time_unit must be one of 'samples','ms','s'")

    # KiloSort expects numpy arrays saved to .npy
    spike_times_path = os.path.join(folder, spike_times_file)
    spike_clusters_path = os.path.join(folder, spike_clusters_file)
    np.save(spike_times_path, times_out)
    np.save(spike_clusters_path, clusters)

    if sd.electrodes is not None:
        np.save(os.path.join(folder, "channel_map.npy"), sd.electrodes)

    return spike_times_path, spike_clusters_path


def export_spikedata_to_pickle(
    sd: "SpikeData",
    filepath: str,
    *,
    protocol: Optional[int] = None,
    s3_upload: bool = False,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> str:
    """Export a SpikeData object to a pickle file.

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        filepath (str): Path where the pickle file will be created
            (overwrites existing). If s3_upload=True, this should be an
            S3 URL (s3://bucket/key).
        protocol (int | None): Pickle protocol version. If None, uses
            the highest protocol available. Lower protocols (e.g., 2, 3)
            may be needed for compatibility with older Python versions.
        s3_upload (bool): If True, upload to S3 URL specified in
            filepath.
        aws_access_key_id (str | None): AWS access key ID for S3
            uploads.
        aws_secret_access_key (str | None): AWS secret access key for
            S3 uploads.
        aws_session_token (str | None): AWS session token for temporary
            credentials.
        region_name (str | None): AWS region name for S3 access.

    Returns:
        path (str): Path to the created pickle file (local path or S3
            URL).
    """
    import tempfile

    from .s3_utils import is_s3_url, upload_to_s3 as _upload_to_s3

    if s3_upload:
        if not is_s3_url(filepath):
            raise ValueError(
                f"filepath must be an S3 URL when s3_upload=True (got '{filepath}')"
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            temp_path = tmp.name
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(sd, f, protocol=protocol)
            _upload_to_s3(
                temp_path,
                filepath,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            return filepath
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Best-effort cleanup: ignore failures when removing temporary file.

    else:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(sd, f, protocol=protocol)
        return filepath


__all__ = [
    "export_spikedata_to_hdf5",
    "export_spikedata_to_nwb",
    "export_spikedata_to_kilosort",
    "export_spikedata_to_pickle",
]
