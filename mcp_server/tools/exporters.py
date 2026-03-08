"""
MCP tools for exporting spike data to various formats.

Supports HDF5, NWB, and KiloSort export formats.
Handles both local files and S3 uploads.
"""

import os
import tempfile
from typing import Any, Dict, List, Literal, Optional

from data_loaders.data_exporters import (
    export_spikedata_to_hdf5,
    export_spikedata_to_kilosort,
    export_spikedata_to_nwb,
    export_spikedata_to_pickle,
)

from mcp_server.s3_adapter import is_s3_url, upload as upload_to_s3
from workspace.workspace import get_workspace_manager

_SPIKEDATA_KEY = "spikedata"


def _get_spikedata(workspace_id: str, namespace: str):
    """Load SpikeData from (namespace, 'spikedata') in the workspace."""
    from spikedata.spikedata import SpikeData

    ws = get_workspace_manager().get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")
    sd = ws.get(namespace, _SPIKEDATA_KEY)
    if sd is None or not isinstance(sd, SpikeData):
        raise ValueError(
            f"No SpikeData found at ({namespace!r}, {_SPIKEDATA_KEY!r}). "
            "Load a recording first using one of: "
            "load_from_hdf5, load_from_nwb, load_from_kilosort, "
            "load_from_pickle, load_from_hdf5_thresholded."
        )
    return sd


async def export_to_hdf5(
    workspace_id: str,
    namespace: str,
    file_path: str,
    style: Literal["raster", "ragged", "group", "paired"] = "ragged",
    raster_dataset: str = "raster",
    raster_bin_size_ms: Optional[float] = None,
    spike_times_dataset: str = "spike_times",
    spike_times_index_dataset: str = "spike_times_index",
    spike_times_unit: Literal["ms", "s", "samples"] = "s",
    fs_Hz: Optional[float] = None,
    group_per_unit: str = "units",
    group_time_unit: Literal["ms", "s", "samples"] = "s",
    idces_dataset: str = "idces",
    times_dataset: str = "times",
    times_unit: Literal["ms", "s", "samples"] = "ms",
    raw_dataset: Optional[str] = None,
    raw_time_dataset: Optional[str] = None,
    raw_time_unit: Literal["ms", "s", "samples"] = "ms",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export spike data to an HDF5 file.

    Supports four export styles:
    - 'raster': 2D raster matrix (requires raster_bin_size_ms)
    - 'ragged': Flat spike times with index array (NWB-like)
    - 'group': Group per unit
    - 'paired': Paired indices and times arrays

    Args:
        workspace_id: Workspace ID containing the SpikeData
        namespace: Namespace within the workspace
        file_path: Local file path or S3 URL for output
        style: Export style ('raster', 'ragged', 'group', 'paired')
        raster_dataset: Dataset name for raster (style='raster')
        raster_bin_size_ms: Bin size in ms for raster (style='raster')
        spike_times_dataset: Dataset name for spike times (style='ragged')
        spike_times_index_dataset: Dataset name for spike times index (style='ragged')
        spike_times_unit: Time unit for spike times
        fs_Hz: Sampling frequency in Hz (required for 'samples' unit)
        group_per_unit: Group name for per-unit datasets (style='group')
        group_time_unit: Time unit for group datasets
        idces_dataset: Dataset name for unit indices (style='paired')
        times_dataset: Dataset name for spike times (style='paired')
        times_unit: Time unit for paired times
        raw_dataset: Optional raw data dataset name
        raw_time_dataset: Optional raw time dataset name
        raw_time_unit: Time unit for raw data
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'file_path' (output path) and 'style'
    """
    spikedata = _get_spikedata(workspace_id, namespace)

    # Determine if output is S3 or local
    is_s3 = is_s3_url(file_path)
    if is_s3:
        # Create temporary local file, then upload
        suffix = ".h5"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        local_path = temp_file.name
        temp_file.close()
    else:
        local_path = file_path
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(local_path) if os.path.dirname(local_path) else ".",
            exist_ok=True,
        )

    # Build export kwargs
    kwargs = {
        "style": style,
        "raster_dataset": raster_dataset,
        "raster_bin_size_ms": raster_bin_size_ms,
        "spike_times_dataset": spike_times_dataset,
        "spike_times_index_dataset": spike_times_index_dataset,
        "spike_times_unit": spike_times_unit,
        "fs_Hz": fs_Hz,
        "group_per_unit": group_per_unit,
        "group_time_unit": group_time_unit,
        "idces_dataset": idces_dataset,
        "times_dataset": times_dataset,
        "times_unit": times_unit,
        "raw_dataset": raw_dataset,
        "raw_time_dataset": raw_time_dataset,
        "raw_time_unit": raw_time_unit,
    }

    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        export_spikedata_to_hdf5(spikedata, local_path, **kwargs)

        # Upload to S3 if needed
        if is_s3:
            upload_to_s3(
                local_path,
                file_path,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            # Clean up temp file
            try:
                os.unlink(local_path)
            except Exception:
                pass
            output_path = file_path
        else:
            output_path = local_path

        return {
            "file_path": output_path,
            "style": style,
        }
    except Exception as e:
        # Clean up temp file on error
        if is_s3:
            try:
                os.unlink(local_path)
            except Exception:
                pass
        raise


async def export_to_nwb(
    workspace_id: str,
    namespace: str,
    file_path: str,
    spike_times_dataset: str = "spike_times",
    spike_times_index_dataset: str = "spike_times_index",
    group: str = "units",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export spike data to an NWB file.

    Args:
        workspace_id: Workspace ID containing the SpikeData
        namespace: Namespace within the workspace
        file_path: Local file path or S3 URL for output
        spike_times_dataset: Dataset name for spike times
        spike_times_index_dataset: Dataset name for spike times index
        group: Group name for units
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'file_path' (output path)
    """
    spikedata = _get_spikedata(workspace_id, namespace)

    is_s3 = is_s3_url(file_path)
    if is_s3:
        suffix = ".nwb"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        local_path = temp_file.name
        temp_file.close()
    else:
        local_path = file_path
        os.makedirs(
            os.path.dirname(local_path) if os.path.dirname(local_path) else ".",
            exist_ok=True,
        )

    try:
        export_spikedata_to_nwb(
            spikedata,
            local_path,
            spike_times_dataset=spike_times_dataset,
            spike_times_index_dataset=spike_times_index_dataset,
            group=group,
        )

        if is_s3:
            upload_to_s3(
                local_path,
                file_path,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            try:
                os.unlink(local_path)
            except Exception:
                pass
            output_path = file_path
        else:
            output_path = local_path

        return {
            "file_path": output_path,
        }
    except Exception as e:
        if is_s3:
            try:
                os.unlink(local_path)
            except Exception:
                pass
        raise


async def export_to_kilosort(
    workspace_id: str,
    namespace: str,
    folder_path: str,
    fs_Hz: float,
    spike_times_file: str = "spike_times.npy",
    spike_clusters_file: str = "spike_clusters.npy",
    time_unit: Literal["samples", "ms", "s"] = "samples",
    cluster_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Export spike data to a KiloSort/Phy folder.

    Args:
        workspace_id: Workspace ID containing the SpikeData
        namespace: Namespace within the workspace
        folder_path: Local folder path for output
        fs_Hz: Sampling frequency in Hz
        spike_times_file: Filename for spike_times.npy
        spike_clusters_file: Filename for spike_clusters.npy
        time_unit: Time unit for output ('samples', 'ms', 's')
        cluster_ids: Optional list of cluster IDs (must match num neurons)

    Returns:
        Dictionary with 'folder_path' and 'files' (list of created files)
    """
    spikedata = _get_spikedata(workspace_id, namespace)

    is_s3 = is_s3_url(folder_path)
    if is_s3:
        raise NotImplementedError(
            "S3 folder paths for KiloSort export not yet fully supported"
        )
    else:
        local_folder = folder_path
        os.makedirs(local_folder, exist_ok=True)

    spike_times_path, spike_clusters_path = export_spikedata_to_kilosort(
        spikedata,
        local_folder,
        fs_Hz=fs_Hz,
        spike_times_file=spike_times_file,
        spike_clusters_file=spike_clusters_file,
        time_unit=time_unit,
        cluster_ids=cluster_ids,
    )

    return {
        "folder_path": local_folder,
        "files": [spike_times_path, spike_clusters_path],
    }


async def export_to_pickle(
    workspace_id: str,
    namespace: str,
    file_path: str,
    protocol: Optional[int] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export spike data to a pickle file.

    Args:
        workspace_id: Workspace ID containing the SpikeData
        namespace: Namespace within the workspace
        file_path: Local file path or S3 URL for output
        protocol: Pickle protocol version (None uses highest available)
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'file_path' (output path)
    """
    spikedata = _get_spikedata(workspace_id, namespace)

    result_path = export_spikedata_to_pickle(
        spikedata,
        file_path,
        protocol=protocol,
        upload_to_s3=is_s3_url(file_path),
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    return {"file_path": result_path}
