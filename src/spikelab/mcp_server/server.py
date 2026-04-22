"""
Main MCP server implementation for spike data analysis.

Registers all tools and handles transport (stdio or SSE).
"""

import argparse
import asyncio
import json
import sys
import traceback
from typing import Any

try:
    from mcp.server import Server
    from mcp import types
    from mcp.server.stdio import stdio_server
except ImportError as e:
    raise ImportError(
        "The MCP server requires the 'mcp' package. "
        "Install with: pip install spikelab[mcp]"
    ) from e

from .tools import analysis, data_loaders, exporters

# Create the MCP server instance
server = Server("spikelab")

# Shared workspace parameter schema properties used in multiple tools.
_WS_PROPS = {
    "workspace_id": {
        "type": "string",
        "description": "Workspace ID",
    },
    "namespace": {
        "type": "string",
        "description": "Recording namespace within the workspace",
    },
}


@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    """List all available tools."""
    tools = []

    # -----------------------------------------------------------------------
    # Data loader tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="load_from_hdf5_raster",
                description="Load spike data from an HDF5 raster matrix. Stores SpikeData at (namespace, 'spikedata').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local path or S3 URL",
                        },
                        "raster_dataset": {
                            "type": "string",
                            "description": "Dataset path for raster matrix",
                        },
                        "raster_bin_size_ms": {
                            "type": "number",
                            "description": "Bin size in ms",
                        },
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "length_ms": {"type": "number"},
                        "workspace_id": {"type": "string", "default": ""},
                        "namespace": {"type": "string", "default": ""},
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path", "raster_dataset", "raster_bin_size_ms"],
                },
            ),
            types.Tool(
                name="load_from_hdf5_ragged",
                description="Load spike data from HDF5 ragged spike times + index. Stores SpikeData at (namespace, 'spikedata').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local path or S3 URL",
                        },
                        "spike_times_dataset": {
                            "type": "string",
                            "default": "spike_times",
                        },
                        "spike_times_index_dataset": {
                            "type": "string",
                            "default": "spike_times_index",
                        },
                        "spike_times_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "fs_Hz": {"type": "number"},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "length_ms": {"type": "number"},
                        "workspace_id": {"type": "string", "default": ""},
                        "namespace": {"type": "string", "default": ""},
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
            types.Tool(
                name="load_from_hdf5_group",
                description="Load spike data from HDF5 group-per-unit structure. Stores SpikeData at (namespace, 'spikedata').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local path or S3 URL",
                        },
                        "group_per_unit": {"type": "string", "default": "units"},
                        "group_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "fs_Hz": {"type": "number"},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "length_ms": {"type": "number"},
                        "workspace_id": {"type": "string", "default": ""},
                        "namespace": {"type": "string", "default": ""},
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
            types.Tool(
                name="load_from_hdf5_paired",
                description="Load spike data from HDF5 paired indices + times arrays. Stores SpikeData at (namespace, 'spikedata').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local path or S3 URL",
                        },
                        "idces_dataset": {"type": "string", "default": "idces"},
                        "times_dataset": {"type": "string", "default": "times"},
                        "times_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "ms",
                        },
                        "fs_Hz": {"type": "number"},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "length_ms": {"type": "number"},
                        "workspace_id": {"type": "string", "default": ""},
                        "namespace": {"type": "string", "default": ""},
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
            types.Tool(
                name="load_from_nwb",
                description=(
                    "Load spike data from an NWB file. Accepts local file paths or "
                    "S3 URLs. Stores SpikeData at (namespace, 'spikedata') in the "
                    "workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
                        },
                        "prefer_pynwb": {"type": "boolean", "default": True},
                        "length_ms": {
                            "type": "number",
                            "description": "Optional recording length in ms",
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace ID to store the SpikeData in. If empty, a new workspace is created.",
                            "default": "",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Recording namespace within the workspace. If empty, derived from the file name.",
                            "default": "",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
            types.Tool(
                name="load_from_kilosort",
                description=(
                    "Load spike data from KiloSort/Phy output folder. Stores "
                    "SpikeData at (namespace, 'spikedata') in the workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_path": {
                            "type": "string",
                            "description": "Local folder path",
                        },
                        "fs_Hz": {
                            "type": "number",
                            "description": "Sampling frequency in Hz",
                        },
                        "spike_times_file": {
                            "type": "string",
                            "default": "spike_times.npy",
                        },
                        "spike_clusters_file": {
                            "type": "string",
                            "default": "spike_clusters.npy",
                        },
                        "cluster_info_tsv": {
                            "type": "string",
                            "description": "Optional cluster_info.tsv path",
                        },
                        "time_unit": {
                            "type": "string",
                            "enum": ["samples", "ms", "s"],
                            "default": "samples",
                        },
                        "include_noise": {"type": "boolean", "default": False},
                        "length_ms": {"type": "number"},
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace ID to store the SpikeData in. If empty, a new workspace is created.",
                            "default": "",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Recording namespace within the workspace. If empty, derived from the folder name.",
                            "default": "",
                        },
                    },
                    "required": ["folder_path", "fs_Hz"],
                },
            ),
            types.Tool(
                name="load_from_pickle",
                description=(
                    "Load spike data from a pickle file. Accepts local file paths or "
                    "S3 URLs. WARNING: only load from trusted sources. Stores SpikeData "
                    "at (namespace, 'spikedata') in the workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace ID to store the SpikeData in. If empty, a new workspace is created.",
                            "default": "",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Recording namespace within the workspace. If empty, derived from the file name.",
                            "default": "",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
            types.Tool(
                name="load_from_hdf5_thresholded",
                description=(
                    "Load and threshold raw data from an HDF5 file. Stores SpikeData "
                    "at (namespace, 'spikedata') in the workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "dataset": {
                            "type": "string",
                            "description": "HDF5 dataset path",
                        },
                        "fs_Hz": {"type": "number"},
                        "threshold_sigma": {"type": "number", "default": 5.0},
                        "filter": {"type": "boolean", "default": True},
                        "hysteresis": {"type": "boolean", "default": True},
                        "direction": {
                            "type": "string",
                            "enum": ["both", "up", "down"],
                            "default": "both",
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace ID to store the SpikeData in. If empty, a new workspace is created.",
                            "default": "",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Recording namespace within the workspace. If empty, derived from the file name.",
                            "default": "",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["file_path", "dataset", "fs_Hz"],
                },
            ),
            types.Tool(
                name="load_from_ibl",
                description=(
                    "Load spike data for a single IBL probe from the public IBL server. "
                    "Authenticates automatically. Only good units (label==1) are included. "
                    "Trial event times are stored in metadata as numpy arrays (ms). "
                    "Stores SpikeData at (namespace, 'spikedata') in the workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "eid": {
                            "type": "string",
                            "description": "IBL experiment ID (UUID string)",
                        },
                        "pid": {
                            "type": "string",
                            "description": "IBL probe ID (UUID string)",
                        },
                        "length_ms": {
                            "type": "number",
                            "description": "Recording duration in ms. Inferred from max spike time if not provided.",
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace ID to store the SpikeData in. If empty, a new workspace is created.",
                            "default": "",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Recording namespace within the workspace. If empty, derived from the eid.",
                            "default": "",
                        },
                    },
                    "required": ["eid", "pid"],
                },
            ),
            types.Tool(
                name="load_from_spikelab_sorted_npz",
                description=(
                    "Load spike data from a SpikeLab compiled sorting .npz file. "
                    "These files are produced by the spike sorting pipeline and "
                    "contain per-unit spike trains, electrode locations, waveform "
                    "templates, and quality metrics. "
                    "Stores SpikeData at (namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local path to the .npz file",
                        },
                        "length_ms": {
                            "type": "number",
                            "description": (
                                "Recording duration in ms. Inferred from "
                                "max spike time if not provided."
                            ),
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": (
                                "Workspace ID to store the SpikeData in. "
                                "If empty, a new workspace is created."
                            ),
                            "default": "",
                        },
                        "namespace": {
                            "type": "string",
                            "description": (
                                "Recording namespace within the workspace. "
                                "If empty, derived from the file name."
                            ),
                            "default": "",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            types.Tool(
                name="query_ibl_probes",
                description=(
                    "Search the IBL Brain-Wide Map database for probes matching given "
                    "criteria. Returns (eid, pid) pairs and per-probe statistics inline. "
                    "Does not store anything in the workspace. Requires one-api and "
                    "brainwidemap packages."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_regions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Beryl atlas region names to filter by (e.g. ['MOs', 'MOp']). If omitted, no region filter is applied.",
                        },
                        "min_units": {
                            "type": "integer",
                            "default": 0,
                            "description": "Minimum number of good units required per probe.",
                        },
                        "min_fraction_in_target": {
                            "type": "number",
                            "default": 0.0,
                            "description": "Minimum fraction (0-1) of good units in target_regions. Ignored when target_regions is not provided.",
                        },
                    },
                    "required": [],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Basic analysis tools — SpikeData → ndarray stored in workspace
    # All require workspace_id, namespace (SpikeData at 'spikedata'), and key.
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="compute_rates",
                description=(
                    "Calculate the mean firing rate of each neuron. Loads SpikeData "
                    "from (namespace, 'spikedata') and stores a (U,) rate array at "
                    "(namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["Hz", "kHz"],
                            "default": "kHz",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_binned",
                description=(
                    "Get binned spike counts. Stores a (U, T_bins) array at "
                    "(namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 40.0,
                            "description": "Bin size in milliseconds",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_binned_meanrate",
                description=(
                    "Calculate the mean firing rate across the population in each time "
                    "bin. Stores a (T_bins,) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 40.0,
                            "description": "Bin size in milliseconds",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["Hz", "kHz"],
                            "default": "kHz",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_raster",
                description=(
                    "Generate a dense spike raster matrix of spike counts per bin. Stores a (U, T_bins) array "
                    "at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in milliseconds",
                        },
                        "time_offset": {
                            "type": "number",
                            "default": 0.0,
                            "description": "Additional offset in ms added to spike times before binning",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_channel_raster",
                description=(
                    "Generate a channel-aggregated raster matrix. Stores a (C, T_bins) "
                    "array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in milliseconds",
                        },
                        "channel_attr": {
                            "type": "string",
                            "description": "Channel attribute name",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_interspike_intervals",
                description=(
                    "Calculate interspike intervals for each neuron. Stores a "
                    "NaN-padded (U, max_isi_count) array at (namespace, key). "
                    "Prerequisite: load_from_* to create SpikeData."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_resampled_isi",
                description=(
                    "Compute instantaneous firing rates via the resampled ISI method "
                    "and store the result as a RateData object at (namespace, key). "
                    "Prerequisite: load_from_* to create SpikeData."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the RateData object",
                        },
                        "times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of time points in ms at which to evaluate instantaneous firing rates",
                        },
                        "sigma_ms": {"type": "number", "default": 10.0},
                    },
                    "required": ["workspace_id", "namespace", "key", "times"],
                },
            ),
            types.Tool(
                name="compute_spike_time_tiling",
                description=(
                    "Calculate spike time tiling coefficient (STTC) between two "
                    "neurons. Stores a length-1 array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "neuron_i": {"type": "integer"},
                        "neuron_j": {"type": "integer"},
                        "delt": {"type": "number", "default": 20.0},
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "neuron_i",
                        "neuron_j",
                    ],
                },
            ),
            types.Tool(
                name="compute_spike_time_tilings",
                description=(
                    "Compute the full STTC matrix for all neuron pairs. Stores a "
                    "(U, U) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "delt": {"type": "number", "default": 20.0},
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="threshold_spike_time_tilings",
                description=(
                    "Compute the full STTC matrix and apply a binary threshold. "
                    "Stores a binary (U, U) connectivity matrix at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Values with absolute value > threshold become 1, else 0",
                        },
                        "delt": {
                            "type": "number",
                            "default": 20.0,
                            "description": "Time window in ms for STTC computation",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "threshold"],
                },
            ),
            types.Tool(
                name="compute_latencies",
                description=(
                    "Compute latencies from reference times to spikes in each neuron. "
                    "Stores a NaN-padded (U, max_latency_count) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of reference times in ms",
                        },
                        "window_ms": {"type": "number", "default": 100.0},
                    },
                    "required": ["workspace_id", "namespace", "key", "times"],
                },
            ),
            types.Tool(
                name="compute_latencies_to_index",
                description=(
                    "Compute latencies from a specific neuron to all other neurons. "
                    "Stores a NaN-padded (U, max_latency_count) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "neuron_index": {"type": "integer"},
                        "window_ms": {"type": "number", "default": 100.0},
                    },
                    "required": ["workspace_id", "namespace", "key", "neuron_index"],
                },
            ),
            types.Tool(
                name="get_pop_rate",
                description=(
                    "Compute the smoothed population firing rate using square then "
                    "Gaussian convolution. Stores a (T,) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "square_width": {
                            "type": "integer",
                            "default": 20,
                            "description": "Width of square smoothing window in bins",
                        },
                        "gauss_sigma": {
                            "type": "integer",
                            "default": 100,
                            "description": "Sigma of Gaussian smoothing window in bins",
                        },
                        "raster_bin_size_ms": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Raster bin size in ms",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_spike_trig_pop_rate",
                description=(
                    "Compute spike-triggered population rate (stPR) for each neuron. "
                    "Stores stPR (U, T) at key, lags (T,) at key_lags, and coupling "
                    "stats (3, U) at key_coupling (row 0=zero-lag, 1=max, 2=delays)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output key for stPR_filtered (U, T)",
                        },
                        "key_lags": {
                            "type": "string",
                            "description": "Output key for lags time axis (T,)",
                        },
                        "key_coupling": {
                            "type": "string",
                            "description": "Output key for coupling stats (3, U): row 0=zero_lag, row 1=max, row 2=delays",
                        },
                        "window_ms": {
                            "type": "integer",
                            "default": 80,
                            "description": "Half-width of lag window in ms",
                        },
                        "cutoff_hz": {
                            "type": "number",
                            "default": 20,
                            "description": "Low-pass filter cutoff in Hz",
                        },
                        "fs": {
                            "type": "number",
                            "default": 1000,
                            "description": "Sampling rate in Hz for filter design",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1,
                            "description": "Spike raster bin size in ms",
                        },
                        "cut_outer": {
                            "type": "integer",
                            "default": 10,
                            "description": "Number of outer lag bins to ignore when computing peak coupling",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "key_lags",
                        "key_coupling",
                    ],
                },
            ),
            types.Tool(
                name="get_bursts",
                description=(
                    "Detect bursts from the population firing rate using thresholded "
                    "peak finding. Stores burst times at key_tburst, burst edges at "
                    "key_edges (B, 2), and peak amplitudes at key_amp."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key_tburst": {
                            "type": "string",
                            "description": "Output key for burst peak times (B,)",
                        },
                        "key_edges": {
                            "type": "string",
                            "description": "Output key for burst edges (B, 2) — required by get_frac_active",
                        },
                        "key_amp": {
                            "type": "string",
                            "description": "Output key for burst peak amplitudes (B,)",
                        },
                        "thr_burst": {
                            "type": "number",
                            "description": "RMS multiplier for burst peak threshold",
                        },
                        "min_burst_diff": {
                            "type": "integer",
                            "description": "Minimum number of bins between burst peaks",
                        },
                        "burst_edge_mult_thresh": {
                            "type": "number",
                            "description": "Multiplier for burst edge detection threshold",
                        },
                        "square_width": {"type": "integer", "default": 20},
                        "gauss_sigma": {"type": "integer", "default": 100},
                        "acc_square_width": {"type": "integer", "default": 8},
                        "acc_gauss_sigma": {"type": "integer", "default": 8},
                        "raster_bin_size_ms": {"type": "number", "default": 1.0},
                        "peak_to_trough": {"type": "boolean", "default": True},
                        "pop_rms_override": {
                            "type": "number",
                            "description": "Override baseline RMS for cross-dataset normalization",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key_tburst",
                        "key_edges",
                        "key_amp",
                        "thr_burst",
                        "min_burst_diff",
                        "burst_edge_mult_thresh",
                    ],
                },
            ),
            types.Tool(
                name="burst_sensitivity",
                description=(
                    "Sweep burst detection parameters (thr_burst × min_burst_diff) "
                    "and store a 2-D matrix of detected burst counts at key."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output key for burst counts matrix (len(thr_values), len(dist_values))",
                        },
                        "thr_values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "1-D array of thr_burst values to sweep",
                        },
                        "dist_values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "1-D array of min_burst_diff values (in bins) to sweep",
                        },
                        "burst_edge_mult_thresh": {
                            "type": "number",
                            "description": "Multiplier for burst edge detection threshold (held constant)",
                        },
                        "square_width": {"type": "integer", "default": 20},
                        "gauss_sigma": {"type": "integer", "default": 100},
                        "acc_square_width": {"type": "integer", "default": 8},
                        "acc_gauss_sigma": {"type": "integer", "default": 8},
                        "raster_bin_size_ms": {"type": "number", "default": 1.0},
                        "peak_to_trough": {"type": "boolean", "default": True},
                        "pop_rms_override": {
                            "type": "number",
                            "description": "Override baseline RMS for cross-dataset normalization",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "thr_values",
                        "dist_values",
                        "burst_edge_mult_thresh",
                    ],
                },
            ),
            types.Tool(
                name="get_frac_active",
                description=(
                    "Calculate fraction of active neurons in bursts. Loads burst edges "
                    "from edges_key (output of get_bursts). Stores per-unit fraction at "
                    "key_frac_unit, per-burst fraction at key_frac_burst, and backbone "
                    "unit indices at key_backbone."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "edges_key": {
                            "type": "string",
                            "description": "Workspace key of the burst edges (B, 2) array — use key_edges from get_bursts",
                        },
                        "key_frac_unit": {
                            "type": "string",
                            "description": "Output key for per-unit fraction active (U,)",
                        },
                        "key_frac_burst": {
                            "type": "string",
                            "description": "Output key for per-burst fraction active (B,)",
                        },
                        "key_backbone": {
                            "type": "string",
                            "description": "Output key for backbone unit indices",
                        },
                        "min_spikes": {"type": "integer"},
                        "backbone_threshold": {
                            "type": "number",
                            "description": "Threshold between 0-1",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "edges_key",
                        "key_frac_unit",
                        "key_frac_burst",
                        "key_backbone",
                        "min_spikes",
                        "backbone_threshold",
                    ],
                },
            ),
            types.Tool(
                name="get_frac_spikes_in_burst",
                description=(
                    "Calculate fraction of each unit's spikes that fall inside burst "
                    "windows. Loads burst edges from edges_key (output of get_bursts). "
                    "Stores per-unit fraction at key."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "edges_key": {
                            "type": "string",
                            "description": "Workspace key of the burst edges (B, 2) array — use key_edges from get_bursts",
                        },
                        "key": {
                            "type": "string",
                            "description": "Output key for per-unit fraction of spikes in bursts (U,)",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "edges_key",
                        "key",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Metadata query tools — load SpikeData from workspace, return inline
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="get_data_info",
                description=(
                    "Get information about the SpikeData stored at "
                    "(namespace, 'spikedata'): num_neurons, length_ms, start_time, metadata."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {**_WS_PROPS},
                    "required": ["workspace_id", "namespace"],
                },
            ),
            types.Tool(
                name="list_neurons",
                description=(
                    "List available neurons with their attributes from the SpikeData "
                    "at (namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {**_WS_PROPS},
                    "required": ["workspace_id", "namespace"],
                },
            ),
            types.Tool(
                name="get_neuron_attribute",
                description=(
                    "Get the value of a neuron attribute across all units from the "
                    "SpikeData at (namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Attribute name to retrieve",
                        },
                        "default": {
                            "description": "Value to return for units missing the attribute (default: null)",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="set_neuron_attribute",
                description=(
                    "Set a neuron attribute on the SpikeData at "
                    "(namespace, 'spikedata'). Modifies and re-stores the object."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Attribute name to set",
                        },
                        "values": {
                            "description": "Single value (applied to all) or list matching neuron_indices length",
                        },
                        "neuron_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Neuron indices to update. If null, updates all.",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "values"],
                },
            ),
            types.Tool(
                name="get_neuron_to_channel_map",
                description=(
                    "Get the mapping from neuron indices to channel indices from the "
                    "SpikeData at (namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "channel_attr": {
                            "type": "string",
                            "description": "Attribute name containing the channel index. If null, auto-detects.",
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # SpikeData transform tools — output stored as SpikeData in workspace
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="subtime",
                description=(
                    "Extract a time window from SpikeData. Loads from "
                    "(namespace, 'spikedata') and stores the result at "
                    "(out_namespace, 'spikedata'). If out_namespace is empty, "
                    "overwrites in place."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "start": {"type": "number", "description": "Start time in ms"},
                        "end": {"type": "number", "description": "End time in ms"},
                        "out_namespace": {
                            "type": "string",
                            "description": "Namespace to store result. If empty, overwrites the input namespace.",
                            "default": "",
                        },
                    },
                    "required": ["workspace_id", "namespace", "start", "end"],
                },
            ),
            types.Tool(
                name="subset",
                description=(
                    "Select specific neurons from SpikeData. Loads from "
                    "(namespace, 'spikedata') and stores the result at "
                    "(out_namespace, 'spikedata'). If out_namespace is empty, "
                    "overwrites in place."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "units": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of unit indices (or attribute values if 'by' is set)",
                        },
                        "by": {
                            "type": "string",
                            "description": "Attribute name to select by",
                        },
                        "out_namespace": {
                            "type": "string",
                            "description": "Namespace to store result. If empty, overwrites the input namespace.",
                            "default": "",
                        },
                    },
                    "required": ["workspace_id", "namespace", "units"],
                },
            ),
            types.Tool(
                name="append_session",
                description=(
                    "Append a second SpikeData recording in time after the first. "
                    "Loads from (namespace_a, 'spikedata') and (namespace_b, 'spikedata'), "
                    "stores result at (out_namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace_a": {
                            "type": "string",
                            "description": "Namespace of the first recording",
                        },
                        "namespace_b": {
                            "type": "string",
                            "description": "Namespace of the recording to append",
                        },
                        "out_namespace": {
                            "type": "string",
                            "description": "Namespace to store result. If empty, overwrites namespace_a.",
                            "default": "",
                        },
                        "offset": {
                            "type": "number",
                            "description": "Gap in ms between recordings (default: 0.0)",
                            "default": 0.0,
                        },
                    },
                    "required": ["workspace_id", "namespace_a", "namespace_b"],
                },
            ),
            types.Tool(
                name="concatenate_units",
                description=(
                    "Add all units from a second SpikeData into the first (both must "
                    "have the same length). Modifies and re-stores (namespace_a, 'spikedata') "
                    "in place."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace_a": {
                            "type": "string",
                            "description": "Namespace to add units into (modified in place)",
                        },
                        "namespace_b": {
                            "type": "string",
                            "description": "Namespace whose units are added",
                        },
                    },
                    "required": ["workspace_id", "namespace_a", "namespace_b"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Curation tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="curate_spikedata",
                description=(
                    "Apply quality-control curation filters to SpikeData. "
                    "Stores the curated SpikeData at (out_namespace, 'spikedata') "
                    "and returns curation history inline. Only criteria with "
                    "non-null thresholds are applied."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "out_namespace": {
                            "type": "string",
                            "description": (
                                "Namespace for the curated SpikeData. "
                                "Defaults to '<namespace>_curated'."
                            ),
                        },
                        "min_spikes": {
                            "type": "integer",
                            "description": "Minimum spike count per unit",
                        },
                        "min_rate_hz": {
                            "type": "number",
                            "description": "Minimum firing rate in Hz",
                        },
                        "isi_max": {
                            "type": "number",
                            "description": "Maximum ISI violation metric",
                        },
                        "isi_threshold_ms": {
                            "type": "number",
                            "description": "Refractory period threshold in ms for ISI check",
                            "default": 1.5,
                        },
                        "isi_method": {
                            "type": "string",
                            "description": "'percent' or 'hill' for ISI violation method",
                            "default": "percent",
                        },
                        "min_snr": {
                            "type": "number",
                            "description": (
                                "Minimum SNR. Requires precomputed 'snr' in "
                                "neuron_attributes or raw_data on the SpikeData."
                            ),
                        },
                        "max_std_norm": {
                            "type": "number",
                            "description": (
                                "Maximum normalized waveform STD. Requires "
                                "precomputed 'std_norm' in neuron_attributes "
                                "or raw_data on the SpikeData."
                            ),
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
            types.Tool(
                name="curate_merge_duplicates",
                description=(
                    "Merge duplicate units by spatial proximity and waveform "
                    "similarity. Finds nearby unit pairs, filters by ISI "
                    "violations and cosine similarity, then greedily merges "
                    "accepted pairs. Requires neuron_attributes with position "
                    "and avg_waveform entries. Stores merged SpikeData at "
                    "(out_namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "out_namespace": {
                            "type": "string",
                            "description": (
                                "Namespace for the merged SpikeData. "
                                "Defaults to '<namespace>_merged'."
                            ),
                        },
                        "dist_um": {
                            "type": "number",
                            "default": 24.8,
                            "description": (
                                "Maximum inter-electrode distance in µm "
                                "to consider a pair as candidate duplicates."
                            ),
                        },
                        "max_violation_rate": {
                            "type": "number",
                            "default": 0.04,
                            "description": (
                                "Maximum ISI violation rate (fraction) for "
                                "a unit to participate in a merge."
                            ),
                        },
                        "isi_threshold_ms": {
                            "type": "number",
                            "default": 1.5,
                            "description": "Refractory period threshold in ms.",
                        },
                        "cosine_threshold": {
                            "type": "number",
                            "default": 0.5,
                            "description": "Minimum cosine similarity to merge a pair.",
                        },
                        "max_lag": {
                            "type": "integer",
                            "default": 10,
                            "description": (
                                "Maximum lag in samples for cosine similarity alignment."
                            ),
                        },
                        "delta_ms": {
                            "type": "number",
                            "default": 0.4,
                            "description": (
                                "Spike deduplication window in ms when merging trains."
                            ),
                        },
                        "max_isi_increase": {
                            "type": "number",
                            "default": 0.04,
                            "description": (
                                "Maximum allowable absolute increase in ISI "
                                "violation fraction after merging."
                            ),
                        },
                        "verbose": {
                            "type": "boolean",
                            "default": False,
                            "description": "Print per-pair merge decisions.",
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # RateData-based analysis tools — load RateData from workspace
    # Prerequisite: compute_resampled_isi
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="compute_pairwise_fr_corr",
                description=(
                    "Compute the pairwise unit-to-unit firing rate correlation and lag "
                    "matrices. Loads RateData from (namespace, rate_key). Stores (U, U) "
                    "correlation matrix at key_corr and lag matrix at key_lag. "
                    "Prerequisite: compute_resampled_isi."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "rate_key": {
                            "type": "string",
                            "description": "Workspace key of the RateData object (from compute_resampled_isi)",
                        },
                        "key_corr": {
                            "type": "string",
                            "description": "Output key for the (U, U) correlation matrix",
                        },
                        "key_lag": {
                            "type": "string",
                            "description": "Output key for the (U, U) lag matrix",
                        },
                        "max_lag": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum lag in time bins for cross-correlation",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "rate_key",
                        "key_corr",
                        "key_lag",
                    ],
                },
            ),
            types.Tool(
                name="compute_pairwise_ccg",
                description=(
                    "Compute pairwise cross-correlogram matrices from binned binary "
                    "spike arrays. Stores PairwiseCompMatrix for correlation at key_corr "
                    "and lag at key_lag. Prerequisite: any load_from_* tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key_corr": {
                            "type": "string",
                            "description": "Output key for the (U, U) correlation PairwiseCompMatrix",
                        },
                        "key_lag": {
                            "type": "string",
                            "description": "Output key for the (U, U) lag PairwiseCompMatrix",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in milliseconds for the binary raster (default: 1.0)",
                        },
                        "max_lag": {
                            "type": "number",
                            "default": 350,
                            "description": "Maximum lag in milliseconds (default: 350)",
                        },
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cross_correlation",
                            "description": "Comparison function: 'cross_correlation' (default) or 'cosine_similarity'",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key_corr",
                        "key_lag",
                    ],
                },
            ),
            types.Tool(
                name="compute_pairwise_latencies",
                description=(
                    "Compute pairwise nearest-spike latency distributions between all "
                    "unit pairs. Stores PairwiseCompMatrix for mean latency at key_mean "
                    "and std latency at key_std. Prerequisite: any load_from_* tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key_mean": {
                            "type": "string",
                            "description": "Output key for the (U, U) mean latency PairwiseCompMatrix",
                        },
                        "key_std": {
                            "type": "string",
                            "description": "Output key for the (U, U) std latency PairwiseCompMatrix",
                        },
                        "window_ms": {
                            "type": "number",
                            "description": "Maximum absolute latency in ms to include (default: no filtering)",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key_mean",
                        "key_std",
                    ],
                },
            ),
            types.Tool(
                name="compute_rate_manifold",
                description=(
                    "Project instantaneous firing rates into a low-dimensional manifold "
                    "using PCA or UMAP. Loads RateData from (namespace, rate_key) and "
                    "stores a (T, n_components) embedding at (namespace, key). "
                    "Prerequisite: compute_resampled_isi."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "rate_key": {
                            "type": "string",
                            "description": "Workspace key of the RateData object (from compute_resampled_isi)",
                        },
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the embedding",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["PCA", "UMAP"],
                            "default": "PCA",
                        },
                        "n_components": {"type": "integer", "default": 2},
                        "n_neighbors": {"type": "integer", "description": "UMAP only"},
                        "min_dist": {"type": "number", "description": "UMAP only"},
                        "metric": {"type": "string", "description": "UMAP only"},
                        "random_state": {"type": "integer"},
                        "store_pca_details": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, store explained variance and PC components to workspace",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "rate_key",
                        "key",
                    ],
                },
            ),
            types.Tool(
                name="frames_rate_data",
                description=(
                    "Split a RateData firing rate trace into fixed-length frames and "
                    "store the resulting RateSliceStack. Loads RateData from "
                    "(namespace, rate_key). Prerequisite: compute_resampled_isi."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "rate_key": {
                            "type": "string",
                            "description": "Workspace key of the RateData object (from compute_resampled_isi)",
                        },
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the RateSliceStack",
                        },
                        "length": {
                            "type": "number",
                            "description": "Frame length in ms",
                        },
                        "overlap": {
                            "type": "number",
                            "default": 0.0,
                            "description": "Overlap between consecutive frames in ms",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "rate_key",
                        "key",
                        "length",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # SpikeData → slice stack creation tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="create_rate_slice_stack",
                description=(
                    "Build event-aligned firing rate slices from SpikeData and store "
                    "the RateSliceStack at (namespace, key). Compatible with all "
                    "compute_rate_slice_* tools."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the RateSliceStack",
                        },
                        "times_start_to_end": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "description": "List of [start, end] time windows in ms",
                        },
                        "sigma_ms": {
                            "type": "number",
                            "default": 10.0,
                            "description": "Gaussian smoothing sigma in ms",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "times_start_to_end",
                    ],
                },
            ),
            types.Tool(
                name="frames_spike_data",
                description=(
                    "Split a SpikeData recording into fixed-length frames and store "
                    "the resulting SpikeSliceStack at (namespace, key). Partial windows "
                    "at the end are excluded. Use spike_slice_to_raster to convert to "
                    "a raster count stack."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the SpikeSliceStack",
                        },
                        "length": {
                            "type": "number",
                            "description": "Frame length in ms",
                        },
                        "overlap": {
                            "type": "number",
                            "default": 0.0,
                            "description": "Overlap between consecutive frames in ms",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "length"],
                },
            ),
            types.Tool(
                name="create_spike_slice_stack",
                description=(
                    "Build event-aligned spike slices from SpikeData and store the "
                    "SpikeSliceStack at (namespace, key). Use spike_slice_to_raster to "
                    "convert to a raster count stack."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the SpikeSliceStack",
                        },
                        "times_start_to_end": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "description": "List of [start, end] time windows in ms",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "times_start_to_end",
                    ],
                },
            ),
            types.Tool(
                name="spike_slice_to_raster",
                description=(
                    "Convert a SpikeSliceStack stored in the workspace to a (U, T, S) "
                    "spike count raster ndarray. Loads SpikeSliceStack from "
                    "(namespace, stack_key) and stores the ndarray at (namespace, key). "
                    "Prerequisite: frames_spike_data or create_spike_slice_stack."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the SpikeSliceStack",
                        },
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the (U, T, S) ndarray",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in ms for the raster",
                        },
                    },
                    "required": ["workspace_id", "namespace", "stack_key", "key"],
                },
            ),
            types.Tool(
                name="align_to_events",
                description=(
                    "Create an event-aligned slice stack from SpikeData and store it "
                    "in the workspace. Events can be a list of times in ms or a string "
                    "key into SpikeData.metadata. kind='spike' stores a SpikeSliceStack; "
                    "kind='rate' stores a RateSliceStack. Out-of-bounds events are "
                    "dropped with a warning. Prerequisite: any load_from_* tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the slice stack",
                        },
                        "events": {
                            "description": "List of event times in ms, or a string metadata key (e.g. 'stim_on_times')",
                        },
                        "pre_ms": {
                            "type": "number",
                            "description": "Window duration before each event in ms",
                        },
                        "post_ms": {
                            "type": "number",
                            "description": "Window duration after each event in ms",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["spike", "rate"],
                            "default": "spike",
                            "description": "'spike' → SpikeSliceStack; 'rate' → RateSliceStack",
                        },
                        "bin_size_ms": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in ms for RateSliceStack (ignored for kind='spike')",
                        },
                        "sigma_ms": {
                            "type": "number",
                            "default": 10.0,
                            "description": "Gaussian smoothing sigma in ms for RateSliceStack (ignored for kind='spike')",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "events",
                        "pre_ms",
                        "post_ms",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # RateSliceStack analysis tools — load from workspace
    # Prerequisite: create_rate_slice_stack or frames_rate_data
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="compute_rate_slice_unit_corr",
                description=(
                    "Compute slice-to-slice unit correlation across event-aligned firing "
                    "rate slices. Loads RateSliceStack from (namespace, stack_key) and "
                    "stores the PairwiseCompMatrixStack (S, S, U) at (namespace, out_key). "
                    "Prerequisite: create_rate_slice_stack or frames_rate_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored RateSliceStack",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output workspace key for the PairwiseCompMatrixStack",
                        },
                        "min_rate_threshold": {"type": "number", "default": 0.1},
                        "min_frac": {"type": "number", "default": 0.3},
                        "max_lag": {"type": "integer", "default": 10},
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cross_correlation",
                        },
                        "frac_active_key": {
                            "type": "string",
                            "description": (
                                "Optional workspace key of a (U,) frac_active array "
                                "to override rate-based activity filtering. "
                                "Produced by compute_frac_active or get_frac_active."
                            ),
                        },
                    },
                    "required": ["workspace_id", "namespace", "stack_key", "out_key"],
                },
            ),
            types.Tool(
                name="compute_rate_slice_time_corr",
                description=(
                    "Compute slice-to-slice time-bin correlation across event-aligned "
                    "firing rate slices. Loads RateSliceStack from (namespace, stack_key) "
                    "and stores the PairwiseCompMatrixStack (S, S, T) at (namespace, out_key). "
                    "Prerequisite: create_rate_slice_stack or frames_rate_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored RateSliceStack",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output workspace key for the PairwiseCompMatrixStack",
                        },
                        "max_lag": {"type": "integer", "default": 0},
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cosine_similarity",
                        },
                    },
                    "required": ["workspace_id", "namespace", "stack_key", "out_key"],
                },
            ),
            types.Tool(
                name="compute_unit_to_unit_slice_corr",
                description=(
                    "Compute unit-to-unit correlation and lag across event-aligned firing "
                    "rate slices. Loads RateSliceStack from (namespace, stack_key). "
                    "Stores correlation PairwiseCompMatrixStack at out_key_corr and lag "
                    "stack at out_key_lag. "
                    "Prerequisite: create_rate_slice_stack or frames_rate_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored RateSliceStack",
                        },
                        "out_key_corr": {
                            "type": "string",
                            "description": "Output key for the correlation PairwiseCompMatrixStack",
                        },
                        "out_key_lag": {
                            "type": "string",
                            "description": "Output key for the lag PairwiseCompMatrixStack",
                        },
                        "max_lag": {"type": "integer", "default": 10},
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cross_correlation",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key_corr",
                        "out_key_lag",
                    ],
                },
            ),
            types.Tool(
                name="compute_rate_slice_unit_order",
                description=(
                    "Order units by their peak firing time from a RateSliceStack stored "
                    "in the workspace. Returns unit ordering inline, split into "
                    "highly_active and low_active groups based on min_frac_active. "
                    "Prerequisite: create_rate_slice_stack or frames_rate_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored RateSliceStack",
                        },
                        "agg_func": {
                            "type": "string",
                            "default": "median",
                            "description": "Aggregation function across slices ('median' or 'mean')",
                        },
                        "min_rate_threshold": {"type": "number", "default": 0.1},
                        "min_frac_active": {
                            "type": "number",
                            "default": 0.0,
                            "description": (
                                "Minimum fraction of slices a unit must be active in "
                                "to be placed in the highly_active group. "
                                "Default 0.0 puts all units in highly_active."
                            ),
                        },
                        "frac_active_key": {
                            "type": "string",
                            "description": (
                                "Optional workspace key of a (U,) frac_active array "
                                "to override rate-based activity filtering. "
                                "Produced by compute_frac_active or get_frac_active."
                            ),
                        },
                    },
                    "required": ["workspace_id", "namespace", "stack_key"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Pairwise matrix conditioning tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="remove_by_condition",
                description=(
                    "Remove entries from a PairwiseCompMatrix or PairwiseCompMatrixStack "
                    "where a condition matrix satisfies a comparison. Stores the masked "
                    "result at (namespace, out_key). Supports broadcasting a single "
                    "PairwiseCompMatrix condition across all slices of a target stack."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "target_key": {
                            "type": "string",
                            "description": (
                                "Workspace key of the target PairwiseCompMatrix or "
                                "PairwiseCompMatrixStack to mask"
                            ),
                        },
                        "condition_key": {
                            "type": "string",
                            "description": (
                                "Workspace key of the condition PairwiseCompMatrix or "
                                "PairwiseCompMatrixStack to evaluate"
                            ),
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output workspace key for the masked result",
                        },
                        "op": {
                            "type": "string",
                            "enum": [
                                "lt",
                                "le",
                                "gt",
                                "ge",
                                "eq",
                                "ne",
                                "abs_lt",
                                "abs_le",
                                "abs_gt",
                                "abs_ge",
                            ],
                            "description": (
                                "Comparison operator applied to the condition matrix. "
                                "Entries where the comparison is True are replaced by fill. "
                                "abs_ variants compare |condition| against threshold."
                            ),
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value for the comparison",
                        },
                        "fill": {
                            "type": "number",
                            "description": "Replacement value for removed entries (default: NaN)",
                        },
                        "condition_namespace": {
                            "type": "string",
                            "description": (
                                "Namespace for the condition key, if different from "
                                "the target namespace. Defaults to same namespace."
                            ),
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "target_key",
                        "condition_key",
                        "out_key",
                        "op",
                        "threshold",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # SpikeSliceStack analysis tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="spike_unit_to_unit_comparison",
                description=(
                    "Compute pairwise unit-to-unit similarity within each slice of a "
                    "SpikeSliceStack using STTC or CCG. Stores PairwiseCompMatrixStack "
                    "(U, U, S) at (namespace, out_key_corr) and optionally lag stack at "
                    "(namespace, out_key_lag). Returns average per slice inline. "
                    "Prerequisite: create_spike_slice_stack or frames_spike_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored SpikeSliceStack",
                        },
                        "out_key_corr": {
                            "type": "string",
                            "description": "Output key for correlation PairwiseCompMatrixStack",
                        },
                        "out_key_lag": {
                            "type": "string",
                            "description": (
                                "Output key for lag PairwiseCompMatrixStack "
                                "(only stored when metric is 'ccg')"
                            ),
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["ccg", "sttc"],
                            "default": "ccg",
                            "description": "'ccg' for cross-correlogram or 'sttc' for spike time tiling coefficient",
                        },
                        "delt": {
                            "type": "number",
                            "default": 20.0,
                            "description": "STTC time window in ms (only used for sttc)",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in ms for CCG raster (only used for ccg)",
                        },
                        "max_lag": {
                            "type": "number",
                            "default": 350,
                            "description": "Max lag in ms for CCG (only used for ccg)",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key_corr",
                        "out_key_lag",
                    ],
                },
            ),
            types.Tool(
                name="spike_slice_to_slice_unit_comparison",
                description=(
                    "Compute slice-to-slice similarity for each unit in a "
                    "SpikeSliceStack using STTC or CCG. Stores PairwiseCompMatrixStack "
                    "(S, S, U) at (namespace, out_key_corr) and optionally lag stack at "
                    "(namespace, out_key_lag). Returns average per unit inline. "
                    "Prerequisite: create_spike_slice_stack or frames_spike_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored SpikeSliceStack",
                        },
                        "out_key_corr": {
                            "type": "string",
                            "description": "Output key for correlation PairwiseCompMatrixStack",
                        },
                        "out_key_lag": {
                            "type": "string",
                            "description": (
                                "Output key for lag PairwiseCompMatrixStack "
                                "(only stored when metric is 'ccg')"
                            ),
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["ccg", "sttc"],
                            "default": "ccg",
                            "description": "'ccg' for cross-correlogram or 'sttc' for spike time tiling coefficient",
                        },
                        "delt": {
                            "type": "number",
                            "default": 20.0,
                            "description": "STTC time window in ms (only used for sttc)",
                        },
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in ms for CCG raster (only used for ccg)",
                        },
                        "max_lag": {
                            "type": "number",
                            "default": 350,
                            "description": "Max lag in ms for CCG (only used for ccg)",
                        },
                        "min_spikes": {
                            "type": "integer",
                            "default": 2,
                            "description": "Minimum spikes in a slice for a unit to be valid",
                        },
                        "min_frac": {
                            "type": "number",
                            "default": 0.3,
                            "description": "Max fraction of invalid slices before unit average is NaN",
                        },
                        "frac_active_key": {
                            "type": "string",
                            "description": (
                                "Optional workspace key of a (U,) frac_active array "
                                "to override internal activity filtering. "
                                "Produced by compute_frac_active or get_frac_active."
                            ),
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key_corr",
                        "out_key_lag",
                    ],
                },
            ),
            types.Tool(
                name="compute_frac_active",
                description=(
                    "Compute the fraction of slices each unit is active in from a "
                    "SpikeSliceStack. Stores a (U,) ndarray at (namespace, out_key). "
                    "The result can be passed as frac_active_key to other tools. "
                    "Prerequisite: create_spike_slice_stack or frames_spike_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored SpikeSliceStack",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output workspace key for the (U,) frac_active array",
                        },
                        "min_spikes": {
                            "type": "integer",
                            "default": 2,
                            "description": (
                                "Minimum spikes for a unit to count as active in a slice"
                            ),
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key",
                    ],
                },
            ),
            types.Tool(
                name="spike_order_units_across_slices",
                description=(
                    "Order units by their typical spike timing across slices of a "
                    "SpikeSliceStack. Returns unit ordering inline, split into "
                    "highly_active and low_active groups. Supports median, mean, "
                    "or first-spike timing. "
                    "Prerequisite: create_spike_slice_stack or frames_spike_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored SpikeSliceStack",
                        },
                        "agg_func": {
                            "type": "string",
                            "default": "median",
                            "description": "Aggregation across slices: 'median' or 'mean'",
                        },
                        "timing": {
                            "type": "string",
                            "enum": ["median", "mean", "first"],
                            "default": "median",
                            "description": (
                                "Which spike time to extract per unit per slice: "
                                "'median' (default), 'mean', or 'first' (onset latency)"
                            ),
                        },
                        "min_spikes": {
                            "type": "integer",
                            "default": 2,
                            "description": (
                                "Minimum spikes for a unit to count as active in a slice"
                            ),
                        },
                        "min_frac_active": {
                            "type": "number",
                            "default": 0.0,
                            "description": (
                                "Minimum fraction of slices a unit must be active in "
                                "to be placed in the highly_active group. "
                                "0.0 puts all units in highly_active."
                            ),
                        },
                        "frac_active_key": {
                            "type": "string",
                            "description": (
                                "Optional workspace key of a (U,) frac_active array "
                                "to override internal activity calculation. "
                                "Produced by compute_frac_active or get_frac_active."
                            ),
                        },
                    },
                    "required": ["workspace_id", "namespace", "stack_key"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Unit timing and rank-order correlation tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="get_unit_timing_per_slice_spike",
                description=(
                    "Compute a representative spike time for each unit in each slice "
                    "of a SpikeSliceStack. Stores a (U, S) ndarray at (namespace, out_key). "
                    "Result can be passed to rank_order_correlation_spike or "
                    "spike_order_units_across_slices via timing_key. "
                    "Prerequisite: create_spike_slice_stack or frames_spike_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored SpikeSliceStack",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the (U, S) timing matrix",
                        },
                        "timing": {
                            "type": "string",
                            "enum": ["median", "mean", "first"],
                            "default": "median",
                            "description": "Spike time to extract: 'median', 'mean', or 'first'",
                        },
                        "min_spikes": {
                            "type": "integer",
                            "default": 2,
                            "description": "Minimum spikes for a unit to be active in a slice",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key",
                    ],
                },
            ),
            types.Tool(
                name="get_unit_timing_per_slice_rate",
                description=(
                    "Compute the peak firing rate time bin for each unit in each slice "
                    "of a RateSliceStack. Stores a (U, S) ndarray at (namespace, out_key). "
                    "Result can be passed to rank_order_correlation_rate or "
                    "compute_rate_slice_unit_order via timing_key. "
                    "Prerequisite: create_rate_slice_stack or frames_rate_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored RateSliceStack",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the (U, S) timing matrix",
                        },
                        "min_rate_threshold": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Minimum peak firing rate for a unit to be active",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key",
                    ],
                },
            ),
            types.Tool(
                name="rank_order_correlation_spike",
                description=(
                    "Compute Spearman rank-order correlation of unit timing between all "
                    "slice pairs of a SpikeSliceStack. Stores correlation PairwiseCompMatrix "
                    "(S, S) at out_key_corr and overlap PairwiseCompMatrix (S, S) at "
                    "out_key_overlap. Supports shuffle-based z-scoring. "
                    "Prerequisite: create_spike_slice_stack or frames_spike_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored SpikeSliceStack",
                        },
                        "out_key_corr": {
                            "type": "string",
                            "description": "Output key for correlation PairwiseCompMatrix (S, S)",
                        },
                        "out_key_overlap": {
                            "type": "string",
                            "description": "Output key for overlap fraction PairwiseCompMatrix (S, S)",
                        },
                        "timing_key": {
                            "type": "string",
                            "description": (
                                "Optional workspace key of a pre-computed (U, S) timing "
                                "matrix from get_unit_timing_per_slice_spike"
                            ),
                        },
                        "timing": {
                            "type": "string",
                            "enum": ["median", "mean", "first"],
                            "default": "median",
                            "description": "Spike time mode (only used when timing_key is not provided)",
                        },
                        "min_spikes": {
                            "type": "integer",
                            "default": 2,
                            "description": "Minimum spikes for activity (only used when timing_key is not provided)",
                        },
                        "min_overlap": {
                            "type": "integer",
                            "default": 3,
                            "description": "Minimum units active in both slices",
                        },
                        "min_overlap_frac": {
                            "type": "number",
                            "description": (
                                "Minimum fraction of total units active in both slices. "
                                "Effective threshold = max(min_overlap, ceil(frac * U))."
                            ),
                        },
                        "n_shuffles": {
                            "type": "integer",
                            "default": 100,
                            "description": "Shuffle iterations for z-scoring. 0 = raw Spearman.",
                        },
                        "seed": {
                            "type": "integer",
                            "default": 1,
                            "description": "Random seed for shuffle reproducibility",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key_corr",
                        "out_key_overlap",
                    ],
                },
            ),
            types.Tool(
                name="rank_order_correlation_rate",
                description=(
                    "Compute Spearman rank-order correlation of unit timing between all "
                    "slice pairs of a RateSliceStack. Stores correlation PairwiseCompMatrix "
                    "(S, S) at out_key_corr and overlap PairwiseCompMatrix (S, S) at "
                    "out_key_overlap. Supports shuffle-based z-scoring. "
                    "Prerequisite: create_rate_slice_stack or frames_rate_data."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "stack_key": {
                            "type": "string",
                            "description": "Workspace key of the stored RateSliceStack",
                        },
                        "out_key_corr": {
                            "type": "string",
                            "description": "Output key for correlation PairwiseCompMatrix (S, S)",
                        },
                        "out_key_overlap": {
                            "type": "string",
                            "description": "Output key for overlap fraction PairwiseCompMatrix (S, S)",
                        },
                        "timing_key": {
                            "type": "string",
                            "description": (
                                "Optional workspace key of a pre-computed (U, S) timing "
                                "matrix from get_unit_timing_per_slice_rate"
                            ),
                        },
                        "min_rate_threshold": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Minimum peak firing rate (only used when timing_key is not provided)",
                        },
                        "min_overlap": {
                            "type": "integer",
                            "default": 3,
                            "description": "Minimum units active in both slices",
                        },
                        "min_overlap_frac": {
                            "type": "number",
                            "description": (
                                "Minimum fraction of total units active in both slices. "
                                "Effective threshold = max(min_overlap, ceil(frac * U))."
                            ),
                        },
                        "n_shuffles": {
                            "type": "integer",
                            "default": 100,
                            "description": "Shuffle iterations for z-scoring. 0 = raw Spearman.",
                        },
                        "seed": {
                            "type": "integer",
                            "default": 1,
                            "description": "Random seed for shuffle reproducibility",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "stack_key",
                        "out_key_corr",
                        "out_key_overlap",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Other workspace-based tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="get_idces_times",
                description=(
                    "Get all spike events as parallel unit-index and time arrays. "
                    "Stores a (2, n_spikes) float64 array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the (2, n_spikes) array",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="get_waveform_traces",
                description=(
                    "Extract raw voltage waveforms around spike times for a single unit. "
                    "Stores the (channels, samples, spikes) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key for the waveform array",
                        },
                        "unit": {
                            "type": "integer",
                            "description": "Unit index to extract waveforms for",
                        },
                        "ms_before": {"type": "number", "default": 1.0},
                        "ms_after": {"type": "number", "default": 2.0},
                        "bandpass_low_hz": {"type": "number"},
                        "bandpass_high_hz": {"type": "number"},
                        "filter_order": {"type": "integer", "default": 3},
                    },
                    "required": ["workspace_id", "namespace", "key", "unit"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Dimensionality reduction pipeline (workspace-native)
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="extract_lower_triangle_features",
                description=(
                    "Extract lower-triangle features from a PairwiseCompMatrixStack "
                    "(or (N, N, S) ndarray) stored in the workspace, producing a "
                    "(S, F) feature matrix stored at (namespace, out_key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the source PairwiseCompMatrixStack or (N, N, S) array",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the (S, F) feature matrix",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="pca_on_lower_triangle",
                description=(
                    "Extract lower-triangle features from a PairwiseCompMatrixStack "
                    "(or (N, N, S) ndarray) and reduce via PCA, storing a "
                    "(S, n_components) embedding at (namespace, out_key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the source PairwiseCompMatrixStack or (N, N, S) array",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the (S, n_components) embedding",
                        },
                        "n_components": {"type": "integer", "default": 2},
                        "store_pca_details": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, store explained variance and PC components to workspace",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="pca_on_workspace_item",
                description=(
                    "Apply PCA dimensionality reduction to a 2D ndarray stored in the "
                    "workspace, storing a (rows, n_components) embedding at "
                    "(namespace, out_key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the source 2D array",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the embedding",
                        },
                        "n_components": {"type": "integer", "default": 2},
                        "store_pca_details": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, store explained variance and PC components to workspace",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="umap_reduction",
                description=(
                    "Apply UMAP dimensionality reduction to a 2D ndarray stored in the "
                    "workspace, storing a (samples, n_components) embedding at "
                    "(namespace, out_key). Requires umap-learn."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the source 2D array",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the embedding",
                        },
                        "n_components": {"type": "integer", "default": 2},
                        "n_neighbors": {"type": "integer", "default": 15},
                        "min_dist": {"type": "number", "default": 0.1},
                        "metric": {"type": "string", "default": "euclidean"},
                        "random_state": {"type": "integer"},
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="umap_graph_communities",
                description=(
                    "Apply UMAP and Louvain community detection to a 2D ndarray stored "
                    "in the workspace; stores the embedding at (namespace, out_key) and "
                    "returns community labels inline. Requires umap-learn, networkx, "
                    "python-louvain."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the source 2D array",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the embedding",
                        },
                        "n_components": {"type": "integer", "default": 2},
                        "resolution": {"type": "number", "default": 1.0},
                        "n_neighbors": {"type": "integer", "default": 15},
                        "min_dist": {"type": "number", "default": 0.1},
                        "metric": {"type": "string", "default": "euclidean"},
                        "random_state": {"type": "integer"},
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # GPLVM tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="fit_gplvm",
                description=(
                    "Fit a Gaussian Process Latent Variable Model (GPLVM) to binned "
                    "spike counts. Stores the decode_res dict at (namespace, key), "
                    "reorder_indices at (namespace, key_reorder), and binned_spike_counts "
                    "at (namespace, key_binned). Returns log marginal likelihoods inline. "
                    "Requires poor_man_gplvm and jax."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output key for the decode_res dict",
                        },
                        "key_reorder": {
                            "type": "string",
                            "description": "Output key for reorder_indices (U,)",
                        },
                        "key_binned": {
                            "type": "string",
                            "description": "Output key for binned_spike_counts (T, U)",
                        },
                        "bin_size_ms": {
                            "type": "number",
                            "description": "Bin width in milliseconds",
                            "default": 50.0,
                        },
                        "movement_variance": {
                            "type": "number",
                            "description": "Movement variance hyperparameter",
                            "default": 1.0,
                        },
                        "tuning_lengthscale": {
                            "type": "number",
                            "description": "Tuning curve lengthscale hyperparameter",
                            "default": 10.0,
                        },
                        "n_latent_bin": {
                            "type": "integer",
                            "description": "Number of latent bins",
                            "default": 100,
                        },
                        "n_iter": {
                            "type": "integer",
                            "description": "Number of EM iterations",
                            "default": 20,
                        },
                        "n_time_per_chunk": {
                            "type": "integer",
                            "description": "Time bins per chunk (controls memory)",
                            "default": 10000,
                        },
                        "random_seed": {
                            "type": "integer",
                            "description": "Random seed for JAX PRNG",
                            "default": 3,
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "key_reorder",
                        "key_binned",
                    ],
                },
            ),
            types.Tool(
                name="compute_gplvm_state_entropy",
                description=(
                    "Compute Shannon entropy of the latent state distribution at each "
                    "time bin from a GPLVM decode_res dict. Stores ndarray (T,) at "
                    "(namespace, out_key). Requires fit_gplvm first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the decode_res dict from fit_gplvm",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the entropy array",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="compute_gplvm_continuity_prob",
                description=(
                    "Extract the continuity (non-jump) probability time series from a "
                    "GPLVM decode_res dict. Stores ndarray (T,) at (namespace, out_key). "
                    "Requires fit_gplvm first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the decode_res dict from fit_gplvm",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the continuity probability array",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="compute_gplvm_avg_state_prob",
                description=(
                    "Compute the average probability of each latent state across all "
                    "time bins from a GPLVM decode_res dict. Stores ndarray (K,) at "
                    "(namespace, out_key). Requires fit_gplvm first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Workspace key of the decode_res dict from fit_gplvm",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the average state probability array",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="compute_gplvm_consecutive_durations",
                description=(
                    "Compute lengths of consecutive runs above or below a threshold in "
                    "a 1-D signal stored in the workspace (e.g. continuity probability "
                    "from compute_gplvm_continuity_prob). Stores durations ndarray at "
                    "(namespace, out_key). Returns count and summary statistics inline."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": (
                                "Workspace key of the 1-D signal array "
                                "(e.g. from compute_gplvm_continuity_prob)"
                            ),
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key for the durations array",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value for the condition",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["above", "below"],
                            "description": "'above' for >= threshold; 'below' for < threshold",
                            "default": "above",
                        },
                        "min_dur": {
                            "type": "integer",
                            "description": "Minimum run length to keep",
                            "default": 1,
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "out_key",
                        "threshold",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Workspace management tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="create_workspace",
                description=(
                    "Create a new named workspace for storing analysis results. "
                    "Supports in-memory (default, fast) and disk-backed (lazy, low RAM) modes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional human-readable label for the workspace",
                        },
                        "lazy": {
                            "type": "boolean",
                            "description": (
                                "If true, use a disk-backed workspace: each item is "
                                "serialised to a temporary HDF5 file on store() and "
                                "deserialised on get(), so only index metadata is kept "
                                "in RAM. Useful when working with large recordings on "
                                "memory-constrained machines. Requires h5py. "
                                "Default: false (fully in-memory)."
                            ),
                            "default": False,
                        },
                    },
                },
            ),
            types.Tool(
                name="delete_workspace",
                description="Delete a workspace and all its contents.",
                inputSchema={
                    "type": "object",
                    "properties": {"workspace_id": {"type": "string"}},
                    "required": ["workspace_id"],
                },
            ),
            types.Tool(
                name="list_workspaces",
                description="List all registered workspaces with summary information.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="describe_workspace",
                description=(
                    "Return the full index of a workspace as a nested dict of "
                    "namespace → key → summary."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"workspace_id": {"type": "string"}},
                    "required": ["workspace_id"],
                },
            ),
            types.Tool(
                name="workspace_get_info",
                description="Return the summary metadata for a single item stored in the workspace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "key": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="rename_workspace_item",
                description="Rename a key within a workspace namespace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "old_key": {"type": "string"},
                        "new_key": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "old_key", "new_key"],
                },
            ),
            types.Tool(
                name="add_workspace_note",
                description="Add or replace a free-text note attached to a stored workspace item.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "key": {"type": "string"},
                        "note": {
                            "type": "string",
                            "description": "Note text to attach",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "note"],
                },
            ),
            types.Tool(
                name="delete_workspace_item",
                description="Delete a single item or an entire namespace from a workspace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "key": {
                            "type": "string",
                            "description": "Key to delete. If omitted, the entire namespace is deleted.",
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
            types.Tool(
                name="save_workspace",
                description=(
                    "Save a workspace to disk as {path}.h5 (data) and {path}.json (index)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "path": {
                            "type": "string",
                            "description": "Base file path without extension",
                        },
                    },
                    "required": ["workspace_id", "path"],
                },
            ),
            types.Tool(
                name="load_workspace",
                description=(
                    "Load a full workspace from disk, reconstructing all stored objects, "
                    "and register it in the workspace manager."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Base file path without extension",
                        },
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="load_workspace_item",
                description=(
                    "Load a single item from a saved workspace file into an existing "
                    "in-memory workspace without loading the full workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Base file path without extension",
                        },
                        "namespace": {"type": "string"},
                        "key": {"type": "string"},
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the in-memory workspace to load the item into",
                        },
                    },
                    "required": ["path", "namespace", "key", "workspace_id"],
                },
            ),
            types.Tool(
                name="merge_workspace",
                description=(
                    "Merge all items from a saved workspace file into an existing "
                    "in-memory workspace. Use this to combine results from parallel "
                    "agents that each saved to separate workspace files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the target workspace to merge into",
                        },
                        "path": {
                            "type": "string",
                            "description": "Base file path (without extension) of the saved workspace to merge from",
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "If true, overwrite existing keys; if false (default), skip duplicates",
                            "default": False,
                        },
                    },
                    "required": ["workspace_id", "path"],
                },
            ),
            types.Tool(
                name="fetch_workspace_item",
                description=(
                    "Retrieve a workspace item inline. Returns full data for "
                    "small types (ndarray, PairwiseCompMatrix, dict) and a "
                    "type-specific summary for large types (SpikeData, "
                    "RateData, slice stacks)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "key": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Shuffling and stack builders
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="spike_shuffle",
                description=(
                    "Create a degree-preserving shuffled copy of SpikeData. "
                    "Preserves per-unit spike counts and per-bin population rates. "
                    "Stores the shuffled SpikeData at (out_namespace, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "out_namespace": {
                            "type": "string",
                            "description": "Namespace for the shuffled copy. Defaults to '<namespace>_shuffled'.",
                        },
                        "swap_per_spike": {
                            "type": "integer",
                            "description": "Number of swap attempts per spike",
                            "default": 5,
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility",
                        },
                        "bin_size": {
                            "type": "integer",
                            "description": "Raster bin size for binarization",
                            "default": 1,
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
            types.Tool(
                name="spike_shuffle_stack",
                description=(
                    "Generate multiple degree-preserving shuffles as a "
                    "SpikeSliceStack for null distributions. "
                    "Stores the stack at (namespace, out_key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "out_key": {
                            "type": "string",
                            "description": "Key for the output SpikeSliceStack",
                        },
                        "n_shuffles": {
                            "type": "integer",
                            "description": "Number of shuffled copies to generate",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed; each shuffle uses seed+i",
                        },
                        "swap_per_spike": {
                            "type": "integer",
                            "description": "Swap attempts per spike",
                            "default": 5,
                        },
                        "bin_size": {
                            "type": "integer",
                            "description": "Raster bin size for binarization",
                            "default": 1,
                        },
                    },
                    "required": ["workspace_id", "namespace", "out_key", "n_shuffles"],
                },
            ),
            types.Tool(
                name="subset_stack",
                description=(
                    "Generate random unit subsets as a SpikeSliceStack for "
                    "sensitivity analysis. Stores at (namespace, out_key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "out_key": {
                            "type": "string",
                            "description": "Key for the output SpikeSliceStack",
                        },
                        "n_subsets": {
                            "type": "integer",
                            "description": "Number of random subsets",
                        },
                        "units_per_subset": {
                            "type": "integer",
                            "description": "Number of units per subset",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "out_key",
                        "n_subsets",
                        "units_per_subset",
                    ],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Waveform metrics
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="compute_waveform_metrics",
                description=(
                    "Compute SNR and normalized STD from raw waveforms. "
                    "Stores 'snr' and 'std_norm' in neuron_attributes. "
                    "Requires non-empty raw_data on the SpikeData."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "ms_before": {
                            "type": "number",
                            "description": "Waveform extraction window before spike (ms)",
                            "default": 1.0,
                        },
                        "ms_after": {
                            "type": "number",
                            "description": "Waveform extraction window after spike (ms)",
                            "default": 2.0,
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Epoch splitting
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="split_epochs",
                description=(
                    "Split a concatenated SpikeData into per-epoch objects "
                    "using metadata['rec_chunks_ms']. Each epoch is stored at "
                    "(<prefix>_epoch_<i>, 'spikedata')."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "out_namespace_prefix": {
                            "type": "string",
                            "description": "Prefix for epoch namespaces. Defaults to source namespace.",
                        },
                    },
                    "required": ["workspace_id", "namespace"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # RateData selection tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="ratedata_subset",
                description=(
                    "Select units from a stored RateData and store the result."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the RateData in the workspace",
                        },
                        "units": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of unit indices (or attribute values if 'by' is set)",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key (overwrite).",
                        },
                        "by": {
                            "type": "string",
                            "description": "Key in neuron_attributes to match against 'units'",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "units"],
                },
            ),
            types.Tool(
                name="ratedata_subtime",
                description=(
                    "Select a time window from a stored RateData. "
                    "Original time values are preserved (no shift to 0)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the RateData in the workspace",
                        },
                        "start": {
                            "type": "number",
                            "description": "Start time in ms (inclusive). null = no left clip.",
                        },
                        "end": {
                            "type": "number",
                            "description": "End time in ms (exclusive). null = no right clip.",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key (overwrite).",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # RateSliceStack selection tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="rate_slice_subset",
                description="Select units from a RateSliceStack and store the result.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the RateSliceStack in the workspace",
                        },
                        "units": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of unit indices",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key.",
                        },
                        "by": {
                            "type": "string",
                            "description": "Key in neuron_attributes to match against 'units'",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "units"],
                },
            ),
            types.Tool(
                name="rate_slice_subtime",
                description=(
                    "Trim the time axis of a RateSliceStack by bin index. "
                    "start_idx inclusive, end_idx exclusive. Supports negative indexing."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the RateSliceStack",
                        },
                        "start_idx": {
                            "type": "integer",
                            "description": "Start bin index (inclusive)",
                        },
                        "end_idx": {
                            "type": "integer",
                            "description": "End bin index (exclusive)",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key.",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "key",
                        "start_idx",
                        "end_idx",
                    ],
                },
            ),
            types.Tool(
                name="rate_slice_subslice",
                description="Select slices from a RateSliceStack by index.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the RateSliceStack",
                        },
                        "slices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of slice indices to keep",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key.",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "slices"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # PairwiseCompMatrixStack manipulation tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="pcm_stack_subslice",
                description="Select slices from a PairwiseCompMatrixStack by index.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the PairwiseCompMatrixStack",
                        },
                        "indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of slice indices to keep",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key.",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "indices"],
                },
            ),
            types.Tool(
                name="pcm_stack_mean",
                description=(
                    "Average a PairwiseCompMatrixStack across slices, "
                    "producing a single PairwiseCompMatrix."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the PairwiseCompMatrixStack",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Key for the averaged PairwiseCompMatrix",
                        },
                        "ignore_nan": {
                            "type": "boolean",
                            "description": "Use nanmean (default true)",
                            "default": True,
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "out_key"],
                },
            ),
            types.Tool(
                name="pcm_stack_threshold",
                description=(
                    "Apply a binary threshold to a PairwiseCompMatrixStack. "
                    "Values become 1 where |v| > threshold, else 0."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the PairwiseCompMatrixStack",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Output key. Defaults to input key.",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key", "threshold"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Shuffle statistics and slice analysis utilities
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="shuffle_z_score",
                description=(
                    "Z-score an observed value against a shuffle null distribution. "
                    "Both must be ndarrays in the workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "observed_key": {
                            "type": "string",
                            "description": "Key of the observed value (ndarray)",
                        },
                        "shuffle_key": {
                            "type": "string",
                            "description": "Key of the shuffle distribution (ndarray)",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Key for the z-scored result",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "observed_key",
                        "shuffle_key",
                        "out_key",
                    ],
                },
            ),
            types.Tool(
                name="shuffle_percentile",
                description=(
                    "Compute percentile rank of observed value within a "
                    "shuffle distribution. Non-parametric alternative to z-score."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "observed_key": {
                            "type": "string",
                            "description": "Key of the observed value (ndarray)",
                        },
                        "shuffle_key": {
                            "type": "string",
                            "description": "Key of the shuffle distribution (ndarray)",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Key for the percentile result",
                        },
                    },
                    "required": [
                        "workspace_id",
                        "namespace",
                        "observed_key",
                        "shuffle_key",
                        "out_key",
                    ],
                },
            ),
            types.Tool(
                name="slice_trend",
                description=(
                    "Fit a linear trend to a metric computed across ordered slices. "
                    "Returns slope and p-value inline (no workspace write)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the 1-D values (ndarray) in the workspace",
                        },
                        "times_key": {
                            "type": "string",
                            "description": (
                                "Key of the slice midpoints (ndarray) in the workspace. "
                                "If omitted, integer indices are used."
                            ),
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="slice_stability",
                description=(
                    "Compute coefficient of variation (std / |mean|) of a metric "
                    "across slices. Returns CV inline."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Key of the values (ndarray) in the workspace",
                        },
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="pairwise_tests",
                description=(
                    "Run pairwise statistical tests across groups with "
                    "multiple-comparison correction. Each key should point "
                    "to a 1-D ndarray in the workspace. Requires scipy."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Workspace keys of the group arrays",
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels for each group. Defaults to keys.",
                        },
                        "out_key": {
                            "type": "string",
                            "description": "Key to store the p-value matrix (optional)",
                        },
                        "test": {
                            "type": "string",
                            "enum": ["welch_t", "student_t", "mann_whitney"],
                            "description": "Statistical test to use",
                            "default": "welch_t",
                        },
                        "correction": {
                            "type": "string",
                            "enum": ["bonferroni"],
                            "description": "Multiple-comparison correction. null for none.",
                            "default": "bonferroni",
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Significance level",
                            "default": 0.05,
                        },
                    },
                    "required": ["workspace_id", "namespace", "keys"],
                },
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Export tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="export_to_hdf5_raster",
                description="Export spike data to HDF5 as a raster matrix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {"type": "string"},
                        "raster_dataset": {"type": "string", "default": "raster"},
                        "raster_bin_size_ms": {"type": "number", "default": 1.0},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "ms",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_hdf5_ragged",
                description="Export spike data to HDF5 as ragged spike times + index (NWB-like).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {"type": "string"},
                        "spike_times_dataset": {
                            "type": "string",
                            "default": "spike_times",
                        },
                        "spike_times_index_dataset": {
                            "type": "string",
                            "default": "spike_times_index",
                        },
                        "spike_times_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "s",
                        },
                        "fs_Hz": {"type": "number"},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "ms",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_hdf5_group",
                description="Export spike data to HDF5 as group-per-unit structure.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {"type": "string"},
                        "group_per_unit": {"type": "string", "default": "units"},
                        "group_time_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "s",
                        },
                        "fs_Hz": {"type": "number"},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "ms",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_hdf5_paired",
                description="Export spike data to HDF5 as paired indices + times arrays.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {"type": "string"},
                        "idces_dataset": {"type": "string", "default": "idces"},
                        "times_dataset": {"type": "string", "default": "times"},
                        "times_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "ms",
                        },
                        "fs_Hz": {"type": "number"},
                        "raw_dataset": {"type": "string"},
                        "raw_time_dataset": {"type": "string"},
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "ms",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_nwb",
                description="Export spike data to an NWB file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {"type": "string"},
                        "spike_times_dataset": {
                            "type": "string",
                            "default": "spike_times",
                        },
                        "spike_times_index_dataset": {
                            "type": "string",
                            "default": "spike_times_index",
                        },
                        "group": {"type": "string", "default": "units"},
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_pickle",
                description=(
                    "Export a workspace item to a pickle file. When key is "
                    "omitted, exports SpikeData at (namespace, 'spikedata'). "
                    "When key is provided, exports any supported type: "
                    "SpikeData, RateData, PairwiseCompMatrix, "
                    "PairwiseCompMatrixStack, RateSliceStack, SpikeSliceStack. "
                    "Accepts local file paths or S3 URLs."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Workspace key of the item to export. "
                                "Defaults to 'spikedata'."
                            ),
                        },
                        "protocol": {"type": "integer"},
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["workspace_id", "namespace", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_kilosort",
                description="Export spike data to a KiloSort/Phy folder.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "folder_path": {"type": "string"},
                        "fs_Hz": {"type": "number"},
                        "spike_times_file": {
                            "type": "string",
                            "default": "spike_times.npy",
                        },
                        "spike_clusters_file": {
                            "type": "string",
                            "default": "spike_clusters.npy",
                        },
                        "time_unit": {
                            "type": "string",
                            "enum": ["samples", "ms", "s"],
                            "default": "samples",
                        },
                        "cluster_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["workspace_id", "namespace", "folder_path", "fs_Hz"],
                },
            ),
        ]
    )

    return tools


_TOOL_DISPATCH: dict[str, Any] = {
    # Data loader tools
    "load_from_hdf5_raster": data_loaders.load_from_hdf5_raster,
    "load_from_hdf5_ragged": data_loaders.load_from_hdf5_ragged,
    "load_from_hdf5_group": data_loaders.load_from_hdf5_group,
    "load_from_hdf5_paired": data_loaders.load_from_hdf5_paired,
    "load_from_nwb": data_loaders.load_from_nwb,
    "load_from_kilosort": data_loaders.load_from_kilosort,
    "load_from_hdf5_thresholded": data_loaders.load_from_hdf5_thresholded,
    "load_from_pickle": data_loaders.load_from_pickle,
    "load_from_ibl": data_loaders.load_from_ibl,
    "load_from_spikelab_sorted_npz": data_loaders.load_from_spikelab_sorted_npz,
    "query_ibl_probes": data_loaders.query_ibl_probes,
    # Basic analysis tools
    "compute_rates": analysis.compute_rates,
    "compute_binned": analysis.compute_binned,
    "compute_binned_meanrate": analysis.compute_binned_meanrate,
    "compute_raster": analysis.compute_raster,
    "compute_channel_raster": analysis.compute_channel_raster,
    "compute_interspike_intervals": analysis.compute_interspike_intervals,
    "compute_resampled_isi": analysis.compute_resampled_isi,
    "compute_spike_time_tiling": analysis.compute_spike_time_tiling,
    "compute_spike_time_tilings": analysis.compute_spike_time_tilings,
    "threshold_spike_time_tilings": analysis.threshold_spike_time_tilings,
    "compute_latencies": analysis.compute_latencies,
    "compute_latencies_to_index": analysis.compute_latencies_to_index,
    "get_pop_rate": analysis.get_pop_rate,
    "compute_spike_trig_pop_rate": analysis.compute_spike_trig_pop_rate,
    "get_bursts": analysis.get_bursts,
    "burst_sensitivity": analysis.burst_sensitivity,
    "get_frac_active": analysis.get_frac_active,
    "get_frac_spikes_in_burst": analysis.get_frac_spikes_in_burst,
    # Metadata query tools
    "get_data_info": analysis.get_data_info,
    "list_neurons": analysis.list_neurons,
    "get_neuron_attribute": analysis.get_neuron_attribute,
    "set_neuron_attribute": analysis.set_neuron_attribute,
    "get_neuron_to_channel_map": analysis.get_neuron_to_channel_map,
    # SpikeData transform tools
    "subtime": analysis.subtime,
    "subset": analysis.subset,
    "append_session": analysis.append_session,
    "concatenate_units": analysis.concatenate_units,
    # Curation tools
    "curate_spikedata": analysis.curate_spikedata,
    "curate_merge_duplicates": analysis.curate_merge_duplicates,
    # RateData-based analysis tools
    "compute_pairwise_fr_corr": analysis.compute_pairwise_fr_corr,
    "compute_pairwise_ccg": analysis.compute_pairwise_ccg,
    "compute_pairwise_latencies": analysis.compute_pairwise_latencies,
    "compute_rate_manifold": analysis.compute_rate_manifold,
    "frames_rate_data": analysis.frames_rate_data,
    # Slice stack creation tools
    "create_rate_slice_stack": analysis.create_rate_slice_stack,
    "frames_spike_data": analysis.frames_spike_data,
    "create_spike_slice_stack": analysis.create_spike_slice_stack,
    "spike_slice_to_raster": analysis.spike_slice_to_raster,
    "align_to_events": analysis.align_to_events,
    # RateSliceStack analysis tools
    "compute_rate_slice_unit_corr": analysis.compute_rate_slice_unit_corr,
    "compute_rate_slice_time_corr": analysis.compute_rate_slice_time_corr,
    "compute_unit_to_unit_slice_corr": analysis.compute_unit_to_unit_slice_corr,
    "compute_rate_slice_unit_order": analysis.compute_rate_slice_unit_order,
    # Pairwise matrix conditioning tools
    "remove_by_condition": analysis.remove_by_condition,
    # SpikeSliceStack analysis tools
    "spike_unit_to_unit_comparison": analysis.spike_unit_to_unit_comparison,
    "spike_slice_to_slice_unit_comparison": analysis.spike_slice_to_slice_unit_comparison,
    "compute_frac_active": analysis.compute_frac_active,
    "spike_order_units_across_slices": analysis.spike_order_units_across_slices,
    # Unit timing and rank-order correlation tools
    "get_unit_timing_per_slice_spike": analysis.get_unit_timing_per_slice_spike,
    "get_unit_timing_per_slice_rate": analysis.get_unit_timing_per_slice_rate,
    "rank_order_correlation_spike": analysis.rank_order_correlation_spike,
    "rank_order_correlation_rate": analysis.rank_order_correlation_rate,
    # Other workspace-based tools
    "get_idces_times": analysis.get_idces_times,
    "get_waveform_traces": analysis.get_waveform_traces,
    # Dimensionality reduction pipeline
    "extract_lower_triangle_features": analysis.extract_lower_triangle_features,
    "pca_on_lower_triangle": analysis.pca_on_lower_triangle,
    "pca_on_workspace_item": analysis.pca_on_workspace_item,
    "umap_reduction": analysis.umap_reduction,
    "umap_graph_communities": analysis.umap_graph_communities,
    # GPLVM tools
    "fit_gplvm": analysis.fit_gplvm,
    "compute_gplvm_state_entropy": analysis.compute_gplvm_state_entropy,
    "compute_gplvm_continuity_prob": analysis.compute_gplvm_continuity_prob,
    "compute_gplvm_avg_state_prob": analysis.compute_gplvm_avg_state_prob,
    "compute_gplvm_consecutive_durations": analysis.compute_gplvm_consecutive_durations,
    # Workspace management tools
    "create_workspace": analysis.create_workspace,
    "delete_workspace": analysis.delete_workspace,
    "list_workspaces": analysis.list_workspaces,
    "describe_workspace": analysis.describe_workspace,
    "workspace_get_info": analysis.workspace_get_info,
    "rename_workspace_item": analysis.rename_workspace_item,
    "add_workspace_note": analysis.add_workspace_note,
    "delete_workspace_item": analysis.delete_workspace_item,
    "save_workspace": analysis.save_workspace,
    "load_workspace": analysis.load_workspace,
    "load_workspace_item": analysis.load_workspace_item,
    "merge_workspace": analysis.merge_workspace,
    "fetch_workspace_item": analysis.fetch_workspace_item,
    # Shuffling and stack builders
    "spike_shuffle": analysis.spike_shuffle,
    "spike_shuffle_stack": analysis.spike_shuffle_stack,
    "subset_stack": analysis.subset_stack,
    # Waveform metrics
    "compute_waveform_metrics": analysis.compute_waveform_metrics,
    # Epoch splitting
    "split_epochs": analysis.split_epochs,
    # RateData selection tools
    "ratedata_subset": analysis.ratedata_subset,
    "ratedata_subtime": analysis.ratedata_subtime,
    # RateSliceStack selection tools
    "rate_slice_subset": analysis.rate_slice_subset,
    "rate_slice_subtime": analysis.rate_slice_subtime,
    "rate_slice_subslice": analysis.rate_slice_subslice,
    # PairwiseCompMatrixStack manipulation tools
    "pcm_stack_subslice": analysis.pcm_stack_subslice,
    "pcm_stack_mean": analysis.pcm_stack_mean,
    "pcm_stack_threshold": analysis.pcm_stack_threshold,
    # Shuffle statistics and slice analysis
    "shuffle_z_score": analysis.shuffle_z_score,
    "shuffle_percentile": analysis.shuffle_percentile,
    "slice_trend": analysis.slice_trend,
    "slice_stability": analysis.slice_stability,
    "pairwise_tests": analysis.pairwise_tests,
    # Export tools
    "export_to_hdf5_raster": exporters.export_to_hdf5_raster,
    "export_to_hdf5_ragged": exporters.export_to_hdf5_ragged,
    "export_to_hdf5_group": exporters.export_to_hdf5_group,
    "export_to_hdf5_paired": exporters.export_to_hdf5_paired,
    "export_to_nwb": exporters.export_to_nwb,
    "export_to_kilosort": exporters.export_to_kilosort,
    "export_to_pickle": exporters.export_to_pickle,
}

# Tools that take no arguments (called without **arguments)
_NO_ARGS_TOOLS = {"list_workspaces"}


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        handler = _TOOL_DISPATCH.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")

        if name in _NO_ARGS_TOOLS:
            result = await handler()
        else:
            result = await handler(**arguments)

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        error_result = {"error": str(e), "type": type(e).__name__}
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transport selection."""
    parser = argparse.ArgumentParser(description="SpikeLab MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    return parser.parse_args()


async def main():
    """Run the MCP server with the selected transport."""
    args = _parse_args()

    if args.transport == "sse":
        try:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Mount
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "SSE transport requires starlette and uvicorn. "
                "Install them with: pip install starlette uvicorn"
            ) from exc

        sse = SseServerTransport("/messages/")

        async def handle_sse(scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )

        app = Starlette(
            routes=[
                Mount("/sse", app=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        print(
            f"MCP Server running at http://{args.host}:{args.port}/sse",
            file=sys.stderr,
            flush=True,
        )
        config = uvicorn.Config(app, host=args.host, port=args.port)
        uv_server = uvicorn.Server(config)
        await uv_server.serve()
    else:
        print(
            "MCP Server running at stdio://spikelab",
            file=sys.stderr,
            flush=True,
        )
        async with stdio_server() as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
