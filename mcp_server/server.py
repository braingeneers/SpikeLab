"""
Main MCP server implementation for spike data analysis.

Registers all tools and handles stdio transport.
"""

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server
from mcp import types
from mcp.server.stdio import stdio_server

from mcp_server.tools import analysis, data_loaders, exporters

# Create the MCP server instance
server = Server("integrated-analysis-tools")

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
                name="load_from_hdf5",
                description=(
                    "Load spike data from an HDF5 file. Supports raster, ragged, "
                    "group-per-unit, and paired array styles. Accepts local file paths "
                    "or S3 URLs. Stores SpikeData at (namespace, 'spikedata') in the "
                    "workspace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["raster", "ragged", "group", "paired"],
                            "description": "Input style",
                            "default": "ragged",
                        },
                        "raster_dataset": {
                            "type": "string",
                            "description": "Dataset path for raster (style='raster')",
                        },
                        "raster_bin_size_ms": {
                            "type": "number",
                            "description": "Bin size in ms (style='raster')",
                        },
                        "spike_times_dataset": {
                            "type": "string",
                            "description": "Dataset path for spike times (style='ragged')",
                        },
                        "spike_times_index_dataset": {
                            "type": "string",
                            "description": "Dataset path for spike times index (style='ragged')",
                        },
                        "spike_times_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "fs_Hz": {
                            "type": "number",
                            "description": "Sampling frequency in Hz",
                        },
                        "group_per_unit": {
                            "type": "string",
                            "description": "Group path (style='group')",
                        },
                        "group_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "idces_dataset": {
                            "type": "string",
                            "description": "Dataset path for unit indices (style='paired')",
                        },
                        "times_dataset": {
                            "type": "string",
                            "description": "Dataset path for spike times (style='paired')",
                        },
                        "times_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
                        "raw_dataset": {
                            "type": "string",
                            "description": "Optional raw data dataset",
                        },
                        "raw_time_dataset": {
                            "type": "string",
                            "description": "Optional raw time dataset",
                        },
                        "raw_time_unit": {
                            "type": "string",
                            "enum": ["s", "ms", "samples"],
                            "default": "s",
                        },
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
                        "aws_access_key_id": {
                            "type": "string",
                            "description": "Optional AWS access key",
                        },
                        "aws_secret_access_key": {
                            "type": "string",
                            "description": "Optional AWS secret key",
                        },
                        "aws_session_token": {
                            "type": "string",
                            "description": "Optional AWS session token",
                        },
                        "region_name": {
                            "type": "string",
                            "description": "Optional AWS region",
                        },
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
                        "bin_size": {"type": "number", "default": 40.0},
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
                        "bin_size": {"type": "number", "default": 40.0},
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
                    "Generate a dense spike raster matrix. Stores a (U, T_bins) array "
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
                        "bin_size": {"type": "number", "default": 20.0},
                    },
                    "required": ["workspace_id", "namespace", "key"],
                },
            ),
            types.Tool(
                name="compute_sparse_raster",
                description=(
                    "Generate a sparse spike raster matrix (densified). Stores a "
                    "(U, T_bins) array at (namespace, key)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        **_WS_PROPS,
                        "key": {
                            "type": "string",
                            "description": "Output workspace key",
                        },
                        "bin_size": {"type": "number", "default": 20.0},
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
                        "bin_size": {"type": "number", "default": 20.0},
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
                        "acc_square_width": {"type": "integer", "default": 5},
                        "acc_gauss_sigma": {"type": "integer", "default": 5},
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
                    "(namespace, 'spikedata'): num_neurons, length_ms, metadata."
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
                    "at the end are excluded. Use spike_slice_to_sparse to convert to "
                    "a binary raster stack."
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
                    "SpikeSliceStack at (namespace, key). Use spike_slice_to_sparse to "
                    "convert to a binary raster stack."
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
                name="spike_slice_to_sparse",
                description=(
                    "Convert a SpikeSliceStack stored in the workspace to a (U, T, S) "
                    "binary sparse raster ndarray. Loads SpikeSliceStack from "
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
                            "description": "Bin size in ms for the sparse raster",
                        },
                    },
                    "required": ["workspace_id", "namespace", "stack_key", "key"],
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
                    "stores the PairwiseCompMatrixStack (U, U, S) at (namespace, out_key). "
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
                    },
                    "required": ["workspace_id", "namespace", "stack_key", "out_key"],
                },
            ),
            types.Tool(
                name="compute_rate_slice_time_corr",
                description=(
                    "Compute slice-to-slice time-bin correlation across event-aligned "
                    "firing rate slices. Loads RateSliceStack from (namespace, stack_key) "
                    "and stores the PairwiseCompMatrixStack (T, T, S) at (namespace, out_key). "
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
                    "in the workspace. Returns unit ordering inline. "
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
                    },
                    "required": ["workspace_id", "namespace", "stack_key"],
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
                name="fetch_workspace_item",
                description=(
                    "Retrieve the data of a workspace item as a nested list. "
                    "Supported types: ndarray and PairwiseCompMatrixStack."
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
    # Export tools
    # -----------------------------------------------------------------------
    tools.extend(
        [
            types.Tool(
                name="export_to_hdf5",
                description="Export spike data to an HDF5 file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["raster", "ragged", "group", "paired"],
                            "default": "ragged",
                        },
                        "raster_dataset": {"type": "string"},
                        "raster_bin_size_ms": {"type": "number"},
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
                        "group_per_unit": {"type": "string", "default": "units"},
                        "group_time_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "s",
                        },
                        "idces_dataset": {"type": "string", "default": "idces"},
                        "times_dataset": {"type": "string", "default": "times"},
                        "times_unit": {
                            "type": "string",
                            "enum": ["ms", "s", "samples"],
                            "default": "ms",
                        },
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
                description="Export spike data to a pickle file. Accepts local file paths or S3 URLs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
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


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        # Data loader tools
        if name == "load_from_hdf5":
            result = await data_loaders.load_from_hdf5(**arguments)
        elif name == "load_from_nwb":
            result = await data_loaders.load_from_nwb(**arguments)
        elif name == "load_from_kilosort":
            result = await data_loaders.load_from_kilosort(**arguments)
        elif name == "load_from_hdf5_thresholded":
            result = await data_loaders.load_from_hdf5_thresholded(**arguments)
        elif name == "load_from_pickle":
            result = await data_loaders.load_from_pickle(**arguments)

        # Basic analysis tools
        elif name == "compute_rates":
            result = await analysis.compute_rates(**arguments)
        elif name == "compute_binned":
            result = await analysis.compute_binned(**arguments)
        elif name == "compute_binned_meanrate":
            result = await analysis.compute_binned_meanrate(**arguments)
        elif name == "compute_raster":
            result = await analysis.compute_raster(**arguments)
        elif name == "compute_sparse_raster":
            result = await analysis.compute_sparse_raster(**arguments)
        elif name == "compute_channel_raster":
            result = await analysis.compute_channel_raster(**arguments)
        elif name == "compute_interspike_intervals":
            result = await analysis.compute_interspike_intervals(**arguments)
        elif name == "compute_resampled_isi":
            result = await analysis.compute_resampled_isi(**arguments)
        elif name == "compute_spike_time_tiling":
            result = await analysis.compute_spike_time_tiling(**arguments)
        elif name == "compute_spike_time_tilings":
            result = await analysis.compute_spike_time_tilings(**arguments)
        elif name == "threshold_spike_time_tilings":
            result = await analysis.threshold_spike_time_tilings(**arguments)
        elif name == "compute_latencies":
            result = await analysis.compute_latencies(**arguments)
        elif name == "compute_latencies_to_index":
            result = await analysis.compute_latencies_to_index(**arguments)
        elif name == "get_pop_rate":
            result = await analysis.get_pop_rate(**arguments)
        elif name == "compute_spike_trig_pop_rate":
            result = await analysis.compute_spike_trig_pop_rate(**arguments)
        elif name == "get_bursts":
            result = await analysis.get_bursts(**arguments)
        elif name == "get_frac_active":
            result = await analysis.get_frac_active(**arguments)

        # Metadata query tools
        elif name == "get_data_info":
            result = await analysis.get_data_info(**arguments)
        elif name == "list_neurons":
            result = await analysis.list_neurons(**arguments)
        elif name == "get_neuron_attribute":
            result = await analysis.get_neuron_attribute(**arguments)
        elif name == "set_neuron_attribute":
            result = await analysis.set_neuron_attribute(**arguments)
        elif name == "get_neuron_to_channel_map":
            result = await analysis.get_neuron_to_channel_map(**arguments)

        # SpikeData transform tools
        elif name == "subtime":
            result = await analysis.subtime(**arguments)
        elif name == "subset":
            result = await analysis.subset(**arguments)
        elif name == "append_session":
            result = await analysis.append_session(**arguments)
        elif name == "concatenate_units":
            result = await analysis.concatenate_units(**arguments)

        # RateData-based analysis tools
        elif name == "compute_pairwise_fr_corr":
            result = await analysis.compute_pairwise_fr_corr(**arguments)
        elif name == "compute_rate_manifold":
            result = await analysis.compute_rate_manifold(**arguments)
        elif name == "frames_rate_data":
            result = await analysis.frames_rate_data(**arguments)

        # Slice stack creation tools
        elif name == "create_rate_slice_stack":
            result = await analysis.create_rate_slice_stack(**arguments)
        elif name == "frames_spike_data":
            result = await analysis.frames_spike_data(**arguments)
        elif name == "create_spike_slice_stack":
            result = await analysis.create_spike_slice_stack(**arguments)
        elif name == "spike_slice_to_sparse":
            result = await analysis.spike_slice_to_sparse(**arguments)

        # RateSliceStack analysis tools
        elif name == "compute_rate_slice_unit_corr":
            result = await analysis.compute_rate_slice_unit_corr(**arguments)
        elif name == "compute_rate_slice_time_corr":
            result = await analysis.compute_rate_slice_time_corr(**arguments)
        elif name == "compute_unit_to_unit_slice_corr":
            result = await analysis.compute_unit_to_unit_slice_corr(**arguments)
        elif name == "compute_rate_slice_unit_order":
            result = await analysis.compute_rate_slice_unit_order(**arguments)

        # Other workspace-based tools
        elif name == "get_idces_times":
            result = await analysis.get_idces_times(**arguments)
        elif name == "get_waveform_traces":
            result = await analysis.get_waveform_traces(**arguments)

        # Dimensionality reduction pipeline
        elif name == "extract_lower_triangle_features":
            result = await analysis.extract_lower_triangle_features(**arguments)
        elif name == "pca_on_lower_triangle":
            result = await analysis.pca_on_lower_triangle(**arguments)
        elif name == "pca_on_workspace_item":
            result = await analysis.pca_on_workspace_item(**arguments)
        elif name == "umap_reduction":
            result = await analysis.umap_reduction(**arguments)
        elif name == "umap_graph_communities":
            result = await analysis.umap_graph_communities(**arguments)

        # Workspace management tools
        elif name == "create_workspace":
            result = await analysis.create_workspace(**arguments)
        elif name == "delete_workspace":
            result = await analysis.delete_workspace(**arguments)
        elif name == "list_workspaces":
            result = await analysis.list_workspaces(**arguments)
        elif name == "describe_workspace":
            result = await analysis.describe_workspace(**arguments)
        elif name == "workspace_get_info":
            result = await analysis.workspace_get_info(**arguments)
        elif name == "rename_workspace_item":
            result = await analysis.rename_workspace_item(**arguments)
        elif name == "add_workspace_note":
            result = await analysis.add_workspace_note(**arguments)
        elif name == "delete_workspace_item":
            result = await analysis.delete_workspace_item(**arguments)
        elif name == "save_workspace":
            result = await analysis.save_workspace(**arguments)
        elif name == "load_workspace":
            result = await analysis.load_workspace(**arguments)
        elif name == "load_workspace_item":
            result = await analysis.load_workspace_item(**arguments)
        elif name == "fetch_workspace_item":
            result = await analysis.fetch_workspace_item(**arguments)

        # Export tools
        elif name == "export_to_hdf5":
            result = await exporters.export_to_hdf5(**arguments)
        elif name == "export_to_nwb":
            result = await exporters.export_to_nwb(**arguments)
        elif name == "export_to_kilosort":
            result = await exporters.export_to_kilosort(**arguments)
        elif name == "export_to_pickle":
            result = await exporters.export_to_pickle(**arguments)

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {"error": str(e), "type": type(e).__name__}
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
