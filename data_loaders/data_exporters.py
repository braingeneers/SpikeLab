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
import pickle

import zipfile


import boto3

import numpy as np

try:  # optional dependency for file formats based on HDF5
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

if TYPE_CHECKING:  # avoid runtime circular import
    from spikedata import SpikeData  # noqa: F401

TimeUnit = Literal["ms", "s", "samples"]

s3_client = boto3.client("s3", endpoint_url="https://s3-west.nrp-nautilus.io")
braingeneers_bucket = "braingeneers"


def _ensure_h5py():
    """Ensure h5py is available for HDF5-based exporters.

    Raises:
        ImportError: If h5py is not installed.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5/NWB exporters. `pip install h5py`."
        )


def _times_from_ms(
    times_ms: np.ndarray, unit: TimeUnit, fs_Hz: Optional[float]
) -> np.ndarray:
    """Convert times from milliseconds to the requested unit.

    This helper function converts spike times from the internal millisecond
    representation to the target time unit for export. The conversion depends
    on the target unit:
    - 'ms': No conversion (identity)
    - 's': Divide by 1000 to convert milliseconds to seconds
    - 'samples': Multiply by sampling frequency and convert to integer sample indices

    Parameters:
        times_ms (np.ndarray): Array of spike times in milliseconds.
        unit (TimeUnit): Target time unit ('ms', 's', or 'samples').
        fs_Hz (Optional[float]): Sampling frequency in Hz. Required when unit='samples'.

    Returns:
        np.ndarray: Converted times in the requested unit. For 'samples', returns
                   integer array; otherwise returns float array.

    Raises:
        ValueError: If unit is 'samples' but fs_Hz is not provided or <= 0,
                   or if unit is not one of the valid options.

    Examples:
        >>> times_ms = np.array([100.0, 200.0, 300.0])
        >>> _times_from_ms(times_ms, 's', None)
        array([0.1, 0.2, 0.3])
        >>> _times_from_ms(times_ms, 'samples', 1000.0)
        array([100, 200, 300])
    """
    if unit == "ms":
        return times_ms.astype(float)
    if unit == "s":
        return times_ms.astype(float) / 1e3
    if unit == "samples":
        if not fs_Hz or fs_Hz <= 0:
            raise ValueError("fs_Hz must be provided and > 0 when unit='samples'")
        # Use round-to-nearest to produce integer samples
        return np.rint(times_ms.astype(float) * (fs_Hz / 1e3)).astype(int)
    raise ValueError(f"Unknown time unit '{unit}' (expected 's','ms','samples')")


def _save_to_s3(file_path: str) -> None:
    """Save HDF5 file to S3 bucket."""
    if not file_path.startswith(f"s3://{braingeneers_bucket}/ephys/"):
        raise ValueError(
            f"URI is unexpected and non-canonical ({file_path})!  Skipping upload to s3."
        )
    # Get the key from the filepath
    key = file_path.replace(f"s3://{braingeneers_bucket}/ephys/", "")
    # Save the file to S3
    s3_client.upload_file(file_path, braingeneers_bucket, key)
    print(
        f"Saved {file_path} to S3 bucket {braingeneers_bucket} as {key} with boto3 version: {boto3.__version__}"
    )


def _save_neuron_attributes_to_hdf5(
    f,  # h5py.File-like
    sd: "SpikeData",
) -> None:
    """Save neuron attributes to HDF5 file in /neuron_attributes group.

    Handles different data types appropriately:
    - Numeric/string columns: saved directly as datasets
    - Object dtype containing arrays: saved as variable-length datasets
    - Other object types: converted to strings
    """
    if sd.neuron_attributes is None:
        return

    try:
        attr_group = f.create_group("neuron_attributes")
        df = sd.neuron_attributes.to_dataframe()

        # Save each column as a dataset
        for col in df.columns:
            data = df[col].values

            # Handle object dtype columns specially
            if data.dtype == object:
                # Check if it's an array of arrays (like waveforms)
                if len(data) > 0 and isinstance(data[0], np.ndarray):
                    # Create variable-length dataset for arrays
                    try:
                        dt = h5py.vlen_dtype(np.dtype("float64"))
                        dset = attr_group.create_dataset(col, (len(data),), dtype=dt)
                        for i, arr in enumerate(data):
                            if arr is not None and isinstance(arr, np.ndarray):
                                dset[i] = arr.astype(np.float64).flatten()
                            else:
                                dset[i] = np.array([], dtype=np.float64)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to save column '{col}' as variable-length array: {e}"
                        )
                        continue
                else:
                    # Convert other object types to strings
                    try:
                        str_data = np.array([str(x) for x in data], dtype="S")
                        attr_group.create_dataset(col, data=str_data)
                    except Exception as e:
                        warnings.warn(f"Failed to save column '{col}' as strings: {e}")
                        continue
            else:
                # Regular numeric or fixed-length string data
                try:
                    attr_group.create_dataset(col, data=data)
                except Exception as e:
                    warnings.warn(f"Failed to save column '{col}': {e}")
                    continue
    except Exception as e:
        warnings.warn(f"Failed to save neuron_attributes to HDF5: {e}")


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

    This function provides four different export styles to accommodate various
    data formats and analysis workflows. The spike times from SpikeData (stored
    internally in milliseconds) are converted to the requested time units for
    the output file.

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        filepath (str): Path where the HDF5 file will be created (overwrites existing).
        style (Literal["raster", "ragged", "group", "paired"]): Export format style.
            - "raster": 2D binary/count matrix (units × time bins)
            - "ragged": Flat concatenated spike times with cumulative indices
            - "group": One HDF5 group containing one dataset per unit
            - "paired": Two parallel arrays of unit indices and spike times

    Raster Style Parameters:
        raster_dataset (str): HDF5 dataset name for the raster matrix.
        raster_bin_size_ms (Optional[float]): Bin size in milliseconds for rasterization.
                                            Required for raster style.

    Ragged Arrays Style Parameters:
        spike_times_dataset (str): Dataset name for concatenated spike times.
        spike_times_index_dataset (str): Dataset name for cumulative spike count indices.
        spike_times_unit (TimeUnit): Time unit for spike times ('ms', 's', 'samples').
        fs_Hz (Optional[float]): Sampling frequency in Hz. Required when any unit is 'samples'.

    Group-per-Unit Style Parameters:
        group_per_unit (str): HDF5 group name containing per-unit datasets.
        group_time_unit (TimeUnit): Time unit for individual unit datasets.

    Paired Arrays Style Parameters:
        idces_dataset (str): Dataset name for unit indices array.
        times_dataset (str): Dataset name for spike times array.
        times_unit (TimeUnit): Time unit for spike times.

    Optional Raw Data Parameters:
        raw_dataset (Optional[str]): Dataset name for raw analog data (if present in sd).
        raw_time_dataset (Optional[str]): Dataset name for raw data time vector.
        raw_time_unit (TimeUnit): Time unit for raw data timestamps.

    Raises:
        ImportError: If h5py is not available.
        ValueError: If style is invalid, required parameters are missing, or
                   fs_Hz is needed but not provided for 'samples' unit.

    Notes:
        - The function creates or overwrites the target HDF5 file.
        - Raw data is only written if both raw_dataset and raw_time_dataset are
          provided and the SpikeData contains raw_data and raw_time attributes.
        - For raster style, the bin size is stored as an attribute for provenance.
        - Parameters mirror the corresponding loader function to ease round-tripping.

    Examples:
        >>> # Export as ragged arrays in seconds
        >>> export_spikedata_to_hdf5(sd, "output.h5", style="ragged",
        ...                          spike_times_unit="s")

        >>> # Export as raster with 1ms bins
        >>> export_spikedata_to_hdf5(sd, "output.h5", style="raster",
        ...                          raster_bin_size_ms=1.0)
    """
    _ensure_h5py()

    style = style.lower()  # normalize
    valid_styles = {"raster", "ragged", "group", "paired"}
    if style not in valid_styles:
        raise ValueError(
            f"Unknown style '{style}' (choose one of {sorted(valid_styles)})"
        )

    # Create or overwrite the HDF5 file
    with h5py.File(filepath, "w") as f:  # type: ignore
        # Optionally write raw arrays if destinations are provided and data exist
        if (
            raw_dataset
            and raw_time_dataset
            and getattr(sd, "raw_data", None) is not None
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

        # Save neuron_attributes if present
        _save_neuron_attributes_to_hdf5(f, sd)

        if style == "raster":
            if raster_bin_size_ms is None or raster_bin_size_ms <= 0:
                raise ValueError(
                    "raster_bin_size_ms must be provided and > 0 for raster style"
                )
            raster = sd.raster(raster_bin_size_ms)
            f.create_dataset(raster_dataset, data=np.asarray(raster))
            # Store bin size as an attribute for provenance (readers can ignore)
            try:
                f[raster_dataset].attrs["bin_size_ms"] = float(raster_bin_size_ms)
            except Exception:
                pass
            return

        if style == "ragged":
            # Flatten all trains and write cumulative end indices
            counts = [len(t) for t in sd.train]
            flat_ms = np.concatenate(sd.train) if sum(counts) else np.array([], float)
            flat = _times_from_ms(flat_ms, spike_times_unit, fs_Hz)
            index = np.cumsum(counts, dtype=int)
            f.create_dataset(spike_times_dataset, data=flat)
            f.create_dataset(spike_times_index_dataset, data=index)
            return

        if style == "group":
            grp = f.create_group(group_per_unit)
            for i, tms in enumerate(sd.train):
                grp.create_dataset(
                    str(i), data=_times_from_ms(np.asarray(tms), group_time_unit, fs_Hz)
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
        times_arr = _times_from_ms(np.array(times_ms, dtype=float), times_unit, fs_Hz)
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

    Creates a minimal NWB-compatible HDF5 file containing spike times in the
    standard ragged array format used by the NWB specification. The output
    uses seconds as the time unit, which is the NWB standard for spike times.
    This format is sufficient for round-tripping with the NWB loader when
    prefer_pynwb=False.

    The function creates an HDF5 file with the structure:
    /{group}/{spike_times_dataset} - concatenated spike times in seconds
    /{group}/{spike_times_index_dataset} - cumulative spike counts per unit

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        filepath (str): Path where the NWB file will be created (overwrites existing).
        spike_times_dataset (str): Name of the dataset containing concatenated spike times.
                                  Default is "spike_times" per NWB convention.
        spike_times_index_dataset (str): Name of the dataset containing cumulative indices.
                                        Default is "spike_times_index" per NWB convention.
        group (str): Name of the HDF5 group to contain the datasets.
                    Default is "units" per NWB convention.

    Raises:
        ImportError: If h5py is not available.

    Notes:
        - Spike times are automatically converted from milliseconds to seconds.
        - The output file structure follows NWB conventions but is minimal
          (does not include full NWB metadata or schema validation).
        - Empty units (no spikes) are handled correctly in the index array.
        - This is compatible with the load_spikedata_from_nwb function when
          prefer_pynwb=False.

    Examples:
        >>> # Export to standard NWB format
        >>> export_spikedata_to_nwb(sd, "experiment.nwb")

        >>> # Export with custom dataset names
        >>> export_spikedata_to_nwb(sd, "data.nwb",
        ...                         spike_times_dataset="my_spike_times",
        ...                         group="my_units")
    """
    _ensure_h5py()
    counts = [len(t) for t in sd.train]
    flat_ms = np.concatenate(sd.train) if sum(counts) else np.array([], float)
    flat_s = _times_from_ms(flat_ms, "s", fs_Hz=None)
    index = np.cumsum(counts, dtype=int)
    with h5py.File(filepath, "w") as f:  # type: ignore
        g = f.create_group(group)
        g.create_dataset(spike_times_dataset, data=flat_s)
        g.create_dataset(spike_times_index_dataset, data=index)

        # Save neuron_attributes as additional columns in the units group
        if sd.neuron_attributes is not None:
            try:
                df = sd.neuron_attributes.to_dataframe()
                for col in df.columns:
                    if col != "spike_times":  # Avoid collision with spike_times dataset
                        data = df[col].values

                        # Handle object dtype columns specially for NWB
                        if data.dtype == object:
                            # Check if it's an array of arrays (like waveforms)
                            if len(data) > 0 and isinstance(data[0], np.ndarray):
                                # Create variable-length dataset for arrays
                                try:
                                    dt = h5py.vlen_dtype(np.dtype("float64"))
                                    dset = g.create_dataset(col, (len(data),), dtype=dt)
                                    for i, arr in enumerate(data):
                                        if arr is not None and isinstance(
                                            arr, np.ndarray
                                        ):
                                            dset[i] = arr.astype(np.float64).flatten()
                                        else:
                                            dset[i] = np.array([], dtype=np.float64)
                                except Exception as e:
                                    warnings.warn(
                                        f"Failed to save NWB column '{col}': {e}"
                                    )
                                    continue
                            else:
                                # Convert other object types to strings
                                try:
                                    str_data = np.array(
                                        [str(x) for x in data], dtype="S"
                                    )
                                    g.create_dataset(col, data=str_data)
                                except Exception as e:
                                    warnings.warn(
                                        f"Failed to save NWB column '{col}' as strings: {e}"
                                    )
                                    continue
                        else:
                            # Regular numeric data
                            try:
                                g.create_dataset(col, data=data)
                            except Exception as e:
                                warnings.warn(f"Failed to save NWB column '{col}': {e}")
                                continue
            except Exception as e:
                warnings.warn(f"Failed to save neuron_attributes to NWB: {e}")


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

    Creates the standard KiloSort output format consisting of two numpy arrays:
    spike_times.npy and spike_clusters.npy. Each spike event is represented
    by its timestamp and the cluster (unit) ID it belongs to. This format
    is compatible with Phy for manual curation and other spike sorting tools.

    The function flattens all spike trains into two parallel arrays where
    each spike gets a timestamp and cluster assignment. Units are mapped to
    cluster IDs either using the provided cluster_ids sequence or by default
    using sequential integers (0, 1, 2, ...).

    Parameters:
        sd (SpikeData): The SpikeData object to export.
        folder (str): Directory path where the .npy files will be created.
                     Created if it doesn't exist.
        fs_Hz (float): Sampling frequency in Hz. Required for time unit conversion,
                      especially when time_unit='samples'.
        spike_times_file (str): Filename for the spike times array.
                               Default is "spike_times.npy".
        spike_clusters_file (str): Filename for the spike clusters array.
                                  Default is "spike_clusters.npy".
        time_unit (TimeUnit): Time unit for output spike times.
                             - 'samples': Integer sample indices (default, KiloSort standard)
                             - 'ms': Milliseconds (float)
                             - 's': Seconds (float)
        cluster_ids (Optional[Sequence[int]]): Custom cluster IDs for each unit.
                                              If None, uses sequential integers 0, 1, 2, ...
                                              Length must match sd.N.

    Returns:
        Tuple[str, str]: Paths to the created spike_times.npy and spike_clusters.npy files.

    Raises:
        ValueError: If fs_Hz is not positive, or if cluster_ids length doesn't match sd.N,
                   or if time_unit is invalid.
        OSError: If the folder cannot be created.

    Notes:
        - The output arrays have the same length (one entry per spike across all units).
        - Spike times are sorted by unit order, not chronologically.
        - Empty units (no spikes) don't contribute entries to the output arrays.
        - The 'samples' time unit produces integer arrays suitable for KiloSort/Phy.
        - Cluster IDs can be arbitrary integers and don't need to be sequential.

    Examples:
        >>> # Export with default sample-based timing
        >>> paths = export_spikedata_to_kilosort(sd, "kilosort_output", fs_Hz=30000)
        >>> print(f"Created {paths[0]} and {paths[1]}")

        >>> # Export with custom cluster IDs and millisecond timing
        >>> export_spikedata_to_kilosort(sd, "output", fs_Hz=30000,
        ...                              time_unit="ms",
        ...                              cluster_ids=[10, 20, 30])
    """
    if not fs_Hz or fs_Hz <= 0:
        raise ValueError("A positive fs_Hz is required for KiloSort export")
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
        times_out = _times_from_ms(np.array(times_ms, dtype=float), "samples", fs_Hz)
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

    # Save neuron_attributes as cluster_info.tsv if available
    if sd.neuron_attributes is not None:
        try:
            import pandas as pd

            df = sd.neuron_attributes.to_dataframe()
            # Use cluster_ids for the cluster_id column if it exists, otherwise add it
            if "cluster_id" not in df.columns:
                df.insert(0, "cluster_id", list(cluster_ids))

            cluster_info_path = os.path.join(folder, "cluster_info.tsv")
            df.to_csv(cluster_info_path, sep="\t", index=False)
        except Exception as e:
            warnings.warn(f"Failed to save cluster_info.tsv: {e}")

    return spike_times_path, spike_clusters_path


def export_pickle_to_s3(sd: "SpikeData", file_path: str) -> None:
    """Export a SpikeData object to a pickle file and save it to S3.
    Parameters
    ----------
    sd : SpikeData
        The SpikeData object to export.
    file_path : str
        Path to the output pickle file. Must be a valid S3 URI.

    Notes
    -----
    - The file is saved to S3 in a zip file with the pickle file inside.

    Examples
    --------
    >>> from data_loaders.data_exporters import export_pickle_to_s3
    >>> export_pickle_to_s3(sd, 's3://my-bucket/data/recording.pkl')
    """

    # Create a temporary pickle file name
    pickle_name = os.path.basename(file_path)
    if not pickle_name.endswith(".pkl"):
        pickle_name = pickle_name + ".pkl"

    # Add .zip extension if not present
    zip_path = file_path if file_path.endswith(".zip") else file_path + ".zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Write pickle data directly to zip
        with zipf.open(pickle_name, "w") as f:
            pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)

    _save_to_s3(zip_path)


def export_spikedata_to_pickle(sd: "SpikeData", file_path: str) -> None:
    """Export a SpikeData object to a pickle file.
    Parameters
    ----------
    sd : SpikeData
        The SpikeData object to export.
    file_path : str
        Path to the output pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = [
    "export_spikedata_to_hdf5",
    "export_spikedata_to_nwb",
    "export_spikedata_to_kilosort",
    "export_pickle_to_s3",
    "export_spikedata_to_pickle",
]
