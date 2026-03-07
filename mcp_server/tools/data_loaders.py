"""
MCP tools for loading spike data from various formats.

Supports HDF5, NWB, KiloSort, and SpikeInterface formats.
Handles both local files and S3 URLs.
"""

import os
import tempfile
from typing import Any, Dict, Optional

from data_loaders.data_loaders import (
    load_spikedata_from_hdf5,
    load_spikedata_from_hdf5_raw_thresholded,
    load_spikedata_from_kilosort,
    load_spikedata_from_nwb,
    load_spikedata_from_pickle,
    load_spikedata_from_spikeinterface,
    load_spikedata_from_spikeinterface_recording,
)

from mcp_server.s3_adapter import ensure_local, is_s3_url
from mcp_server.sessions import get_session_manager


async def load_from_hdf5(
    file_path: str,
    style: str = "ragged",
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
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load spike data from an HDF5 file.

    Supports four input styles:
    - 'raster': 2D raster matrix (requires raster_dataset and raster_bin_size_ms)
    - 'ragged': Flat spike times with index array (requires spike_times_dataset and spike_times_index_dataset)
    - 'group': Group per unit (requires group_per_unit)
    - 'paired': Paired indices and times arrays (requires idces_dataset and times_dataset)

    Args:
        file_path: Local file path or S3 URL to HDF5 file
        style: Input style ('raster', 'ragged', 'group', or 'paired')
        raster_dataset: Dataset path for raster matrix (style='raster')
        raster_bin_size_ms: Bin size in ms for raster (style='raster')
        spike_times_dataset: Dataset path for flat spike times (style='ragged')
        spike_times_index_dataset: Dataset path for spike times index (style='ragged')
        spike_times_unit: Time unit for spike times ('s', 'ms', 'samples')
        fs_Hz: Sampling frequency in Hz (required for 'samples' unit)
        group_per_unit: Group path containing per-unit datasets (style='group')
        group_time_unit: Time unit for group datasets
        idces_dataset: Dataset path for unit indices (style='paired')
        times_dataset: Dataset path for spike times (style='paired')
        times_unit: Time unit for paired times
        raw_dataset: Optional raw data dataset path
        raw_time_dataset: Optional raw time dataset path
        raw_time_unit: Time unit for raw data
        length_ms: Optional recording length in ms
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'session_id' and 'info' (num_neurons, length_ms, metadata)
    """
    local_path, is_temp = ensure_local(
        file_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    try:
        # Build kwargs based on style
        kwargs = {
            "spike_times_unit": spike_times_unit,
            "fs_Hz": fs_Hz,
            "group_time_unit": group_time_unit,
            "times_unit": times_unit,
            "raw_time_unit": raw_time_unit,
            "length_ms": length_ms,
        }

        if style == "raster":
            if raster_dataset is None or raster_bin_size_ms is None:
                raise ValueError(
                    "raster_dataset and raster_bin_size_ms required for raster style"
                )
            kwargs["raster_dataset"] = raster_dataset
            kwargs["raster_bin_size_ms"] = raster_bin_size_ms
        elif style == "ragged":
            if spike_times_dataset is None or spike_times_index_dataset is None:
                raise ValueError(
                    "spike_times_dataset and spike_times_index_dataset required for ragged style"
                )
            kwargs["spike_times_dataset"] = spike_times_dataset
            kwargs["spike_times_index_dataset"] = spike_times_index_dataset
        elif style == "group":
            if group_per_unit is None:
                raise ValueError("group_per_unit required for group style")
            kwargs["group_per_unit"] = group_per_unit
        elif style == "paired":
            if idces_dataset is None or times_dataset is None:
                raise ValueError(
                    "idces_dataset and times_dataset required for paired style"
                )
            kwargs["idces_dataset"] = idces_dataset
            kwargs["times_dataset"] = times_dataset
        else:
            raise ValueError(f"Unknown style: {style}")

        if raw_dataset:
            kwargs["raw_dataset"] = raw_dataset
        if raw_time_dataset:
            kwargs["raw_time_dataset"] = raw_time_dataset

        spikedata = load_spikedata_from_hdf5(local_path, **kwargs)

        session_manager = get_session_manager()
        session_id = session_manager.create_session(spikedata)

        return {
            "session_id": session_id,
            "info": {
                "num_neurons": spikedata.N,
                "length_ms": spikedata.length,
                "metadata": spikedata.metadata,
            },
        }
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except Exception:
                pass


async def load_from_nwb(
    file_path: str,
    prefer_pynwb: bool = True,
    length_ms: Optional[float] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load spike data from an NWB file.

    Args:
        file_path: Local file path or S3 URL to NWB file
        prefer_pynwb: Prefer pynwb library over h5py for reading
        length_ms: Optional recording length in ms
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'session_id' and 'info' (num_neurons, length_ms, metadata)
    """
    local_path, is_temp = ensure_local(
        file_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    try:
        spikedata = load_spikedata_from_nwb(
            local_path, prefer_pynwb=prefer_pynwb, length_ms=length_ms
        )

        session_manager = get_session_manager()
        session_id = session_manager.create_session(spikedata)

        return {
            "session_id": session_id,
            "info": {
                "num_neurons": spikedata.N,
                "length_ms": spikedata.length,
                "metadata": spikedata.metadata,
            },
        }
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except Exception:
                pass


async def load_from_kilosort(
    folder_path: str,
    fs_Hz: float,
    spike_times_file: str = "spike_times.npy",
    spike_clusters_file: str = "spike_clusters.npy",
    cluster_info_tsv: Optional[str] = None,
    time_unit: str = "samples",
    include_noise: bool = False,
    length_ms: Optional[float] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load spike data from KiloSort/Phy output folder.

    Args:
        folder_path: Local folder path or S3 URL prefix to KiloSort output folder
        fs_Hz: Sampling frequency in Hz
        spike_times_file: Filename for spike_times.npy
        spike_clusters_file: Filename for spike_clusters.npy
        cluster_info_tsv: Optional path to cluster_info.tsv
        time_unit: Time unit in input files ('samples', 'ms', 's')
        include_noise: Include noise clusters if cluster_info.tsv is provided
        length_ms: Optional recording length in ms
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'session_id' and 'info' (num_neurons, length_ms, metadata)
    """
    # For S3, we need to handle folder paths differently
    # For now, assume local folder or handle S3 folder as a prefix
    if is_s3_url(folder_path):
        # For S3 folders, we'd need to download the specific files
        # This is a simplified version - in practice you might want more sophisticated handling
        raise NotImplementedError(
            "S3 folder paths for KiloSort not yet fully supported"
        )
    else:
        local_folder = folder_path

    if not os.path.isdir(local_folder):
        raise ValueError(f"Folder not found: {local_folder}")

    spikedata = load_spikedata_from_kilosort(
        local_folder,
        fs_Hz=fs_Hz,
        spike_times_file=spike_times_file,
        spike_clusters_file=spike_clusters_file,
        cluster_info_tsv=cluster_info_tsv,
        time_unit=time_unit,
        include_noise=include_noise,
        length_ms=length_ms,
    )

    session_manager = get_session_manager()
    session_id = session_manager.create_session(spikedata)

    return {
        "session_id": session_id,
        "info": {
            "num_neurons": spikedata.N,
            "length_ms": spikedata.length,
            "metadata": spikedata.metadata,
        },
    }


async def load_from_pickle(
    file_path: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load spike data from a pickle file.

    WARNING: Only load pickle files from trusted sources. Pickle deserialization
    can execute arbitrary code.

    Args:
        file_path: Local file path or S3 URL to pickle file
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'session_id' and 'info' (num_neurons, length_ms, metadata)
    """
    spikedata = load_spikedata_from_pickle(
        file_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    session_manager = get_session_manager()
    session_id = session_manager.create_session(spikedata)

    return {
        "session_id": session_id,
        "info": {
            "num_neurons": spikedata.N,
            "length_ms": spikedata.length,
            "metadata": spikedata.metadata,
        },
    }


async def load_from_hdf5_thresholded(
    file_path: str,
    dataset: str,
    fs_Hz: float,
    threshold_sigma: float = 5.0,
    filter: bool = True,
    hysteresis: bool = True,
    direction: str = "both",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and threshold raw data from an HDF5 file.

    Args:
        file_path: Local file path or S3 URL to HDF5 file
        dataset: HDF5 dataset path containing raw traces (channels, time)
        fs_Hz: Sampling frequency in Hz
        threshold_sigma: Threshold in units of per-channel standard deviation
        filter: Apply Butterworth bandpass filter (300Hz-6kHz default)
        hysteresis: Use rising-edge detection
        direction: Threshold direction ('both', 'up', 'down')
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'session_id' and 'info' (num_neurons, length_ms, metadata)
    """
    local_path, is_temp = ensure_local(
        file_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    try:
        spikedata = load_spikedata_from_hdf5_raw_thresholded(
            local_path,
            dataset,
            fs_Hz=fs_Hz,
            threshold_sigma=threshold_sigma,
            filter=filter,
            hysteresis=hysteresis,
            direction=direction,
        )

        session_manager = get_session_manager()
        session_id = session_manager.create_session(spikedata)

        return {
            "session_id": session_id,
            "info": {
                "num_neurons": spikedata.N,
                "length_ms": spikedata.length,
                "metadata": spikedata.metadata,
            },
        }
    finally:
        if is_temp:
            try:
                os.unlink(local_path)
            except Exception:
                pass
