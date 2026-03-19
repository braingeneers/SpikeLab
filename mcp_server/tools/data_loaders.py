"""
MCP tools for loading spike data from various formats.

Supports HDF5, NWB, KiloSort, and SpikeInterface formats.
Handles both local files and S3 URLs.
"""

import os
from typing import Any, Dict, Optional

from ...data_loaders.data_loaders import (
    load_spikedata_from_hdf5,
    load_spikedata_from_hdf5_raw_thresholded,
    load_spikedata_from_ibl,
    load_spikedata_from_kilosort,
    load_spikedata_from_nwb,
    load_spikedata_from_pickle,
    query_ibl_probes as _query_ibl_probes,
)

from ...data_loaders.s3_utils import ensure_local_file, is_s3_url
from ...workspace.workspace import get_workspace_manager

# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _resolve_workspace(workspace_id: str, name: Optional[str] = None):
    """
    Get or create a workspace.

    Returns (workspace, workspace_id). Creates a new workspace when
    workspace_id is empty; retrieves an existing one otherwise.
    """
    wm = get_workspace_manager()
    if workspace_id:
        ws = wm.get_workspace(workspace_id)
        if ws is None:
            raise ValueError(f"Workspace not found: {workspace_id}")
        return ws, workspace_id
    new_id = wm.create_workspace(name=name)
    return wm.get_workspace(new_id), new_id


def _namespace_from_path(path: str, namespace: str) -> str:
    """Return namespace, or derive from file/folder basename if empty."""
    if namespace:
        return namespace
    stem = os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
    return stem or "recording"


def _unique_namespace(ws, namespace: str) -> str:
    """
    Return namespace, appending _1, _2, ... until unique within ws.

    If the namespace does not yet exist in ws, it is returned unchanged.
    """
    existing = set(ws.list_keys().keys())
    if namespace not in existing:
        return namespace
    i = 1
    while f"{namespace}_{i}" in existing:
        i += 1
    return f"{namespace}_{i}"


# ---------------------------------------------------------------------------
# Loader tool wrappers
# ---------------------------------------------------------------------------


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
    workspace_id: str = "",
    namespace: str = "",
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
        workspace_id: Workspace to store the SpikeData in; creates a new one if empty
        namespace: Recording namespace within the workspace; derived from file name if empty
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'workspace_id', 'namespace', 'workspace_key', and 'info'
    """
    local_path, is_temp = ensure_local_file(
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

        ns_derived = _namespace_from_path(file_path, namespace)
        ws, resolved_wid = _resolve_workspace(workspace_id, name=ns_derived)
        ns_final = _unique_namespace(ws, ns_derived)
        ws.store(ns_final, "spikedata", spikedata)

        return {
            "workspace_id": resolved_wid,
            "namespace": ns_final,
            "workspace_key": "spikedata",
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
    workspace_id: str = "",
    namespace: str = "",
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
        workspace_id: Workspace to store the SpikeData in; creates a new one if empty
        namespace: Recording namespace within the workspace; derived from file name if empty
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'workspace_id', 'namespace', 'workspace_key', and 'info'
    """
    local_path, is_temp = ensure_local_file(
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

        ns_derived = _namespace_from_path(file_path, namespace)
        ws, resolved_wid = _resolve_workspace(workspace_id, name=ns_derived)
        ns_final = _unique_namespace(ws, ns_derived)
        ws.store(ns_final, "spikedata", spikedata)

        return {
            "workspace_id": resolved_wid,
            "namespace": ns_final,
            "workspace_key": "spikedata",
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
    workspace_id: str = "",
    namespace: str = "",
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
        workspace_id: Workspace to store the SpikeData in; creates a new one if empty
        namespace: Recording namespace within the workspace; derived from folder name if empty
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'workspace_id', 'namespace', 'workspace_key', and 'info'
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

    ns_derived = _namespace_from_path(folder_path, namespace)
    ws, resolved_wid = _resolve_workspace(workspace_id, name=ns_derived)
    ns_final = _unique_namespace(ws, ns_derived)
    ws.store(ns_final, "spikedata", spikedata)

    return {
        "workspace_id": resolved_wid,
        "namespace": ns_final,
        "workspace_key": "spikedata",
        "info": {
            "num_neurons": spikedata.N,
            "length_ms": spikedata.length,
            "metadata": spikedata.metadata,
        },
    }


async def load_from_pickle(
    file_path: str,
    workspace_id: str = "",
    namespace: str = "",
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
        workspace_id: Workspace to store the SpikeData in; creates a new one if empty
        namespace: Recording namespace within the workspace; derived from file name if empty
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'workspace_id', 'namespace', 'workspace_key', and 'info'
    """
    spikedata = load_spikedata_from_pickle(
        file_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )

    ns_derived = _namespace_from_path(file_path, namespace)
    ws, resolved_wid = _resolve_workspace(workspace_id, name=ns_derived)
    ns_final = _unique_namespace(ws, ns_derived)
    ws.store(ns_final, "spikedata", spikedata)

    return {
        "workspace_id": resolved_wid,
        "namespace": ns_final,
        "workspace_key": "spikedata",
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
    workspace_id: str = "",
    namespace: str = "",
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
        workspace_id: Workspace to store the SpikeData in; creates a new one if empty
        namespace: Recording namespace within the workspace; derived from file name if empty
        aws_access_key_id: Optional AWS access key for S3
        aws_secret_access_key: Optional AWS secret key for S3
        aws_session_token: Optional AWS session token for S3
        region_name: Optional AWS region name

    Returns:
        Dictionary with 'workspace_id', 'namespace', 'workspace_key', and 'info'
    """
    local_path, is_temp = ensure_local_file(
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

        ns_derived = _namespace_from_path(file_path, namespace)
        ws, resolved_wid = _resolve_workspace(workspace_id, name=ns_derived)
        ns_final = _unique_namespace(ws, ns_derived)
        ws.store(ns_final, "spikedata", spikedata)

        return {
            "workspace_id": resolved_wid,
            "namespace": ns_final,
            "workspace_key": "spikedata",
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


async def load_from_ibl(
    eid: str,
    pid: str,
    length_ms: Optional[float] = None,
    workspace_id: str = "",
    namespace: str = "",
) -> Dict[str, Any]:
    """
    Load spike data for a single IBL probe into the workspace.

    Authenticates against the public IBL server automatically. Only units
    with label==1 in the Brain-Wide Map table are included. Trial event times
    are stored in SpikeData.metadata as individual numpy arrays in milliseconds.
    Stores SpikeData at (namespace, 'spikedata').

    Args:
        eid: IBL experiment ID (UUID string).
        pid: IBL probe ID (UUID string).
        length_ms: Optional recording duration in ms; inferred from max spike time if absent.
        workspace_id: Workspace to store the SpikeData in; creates a new one if empty.
        namespace: Recording namespace; derived from the eid if empty.

    Returns:
        Dictionary with workspace_id, namespace, workspace_key, and info.
    """
    spikedata = load_spikedata_from_ibl(eid, pid, length_ms=length_ms)

    ns_derived = namespace or eid[:8]
    ws, resolved_wid = _resolve_workspace(workspace_id, name=ns_derived)
    ns_final = _unique_namespace(ws, ns_derived)
    ws.store(ns_final, "spikedata", spikedata)

    return {
        "workspace_id": resolved_wid,
        "namespace": ns_final,
        "workspace_key": "spikedata",
        "info": {
            "num_neurons": spikedata.N,
            "length_ms": spikedata.length,
            "metadata": {
                k: v
                for k, v in spikedata.metadata.items()
                if not hasattr(v, "__len__") or isinstance(v, str)
            },
        },
    }


async def query_ibl_probes(
    target_regions: Optional[list] = None,
    min_units: int = 0,
    min_fraction_in_target: float = 0.0,
    labs: Optional[list] = None,
    subjects: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Search the IBL Brain-Wide Map database for probes matching given criteria.

    Returns matching (eid, pid) pairs and per-probe statistics inline.
    Does not store anything in the workspace.

    Args:
        target_regions: Beryl atlas region names to filter by (e.g. ["MOs", "MOp"]).
            If None, no region filter is applied.
        min_units: Minimum number of good units required per probe.
        min_fraction_in_target: Minimum fraction of good units in target_regions.
            Ignored when target_regions is None.
        labs: Restrict to probes from these lab names. If None, no filter applied.
        subjects: Restrict to probes from these subject names. If None, no filter applied.

    Returns:
        Dictionary with probes list and stats list of dicts.
    """
    probes, stats_df = _query_ibl_probes(
        target_regions,
        min_units=min_units,
        min_fraction_in_target=min_fraction_in_target,
        labs=labs,
        subjects=subjects,
    )
    stats = stats_df.to_dict(orient="records")
    return {
        "probes": probes,
        "n_probes": len(probes),
        "stats": stats,
    }
