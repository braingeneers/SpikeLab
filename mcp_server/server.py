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

from .tools import analysis, data_loaders, exporters

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
                        "labs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Restrict to probes from these lab names. If omitted, no lab filter is applied.",
                        },
                        "subjects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Restrict to probes from these subject names. If omitted, no subject filter is applied.",
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
                            "description": "Bin size in ms for the sparse raster",
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
        elif name == "load_from_ibl":
            result = await data_loaders.load_from_ibl(**arguments)
        elif name == "query_ibl_probes":
            result = await data_loaders.query_ibl_probes(**arguments)

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
        elif name == "burst_sensitivity":
            result = await analysis.burst_sensitivity(**arguments)
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
        elif name == "compute_pairwise_ccg":
            result = await analysis.compute_pairwise_ccg(**arguments)
        elif name == "compute_pairwise_latencies":
            result = await analysis.compute_pairwise_latencies(**arguments)
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
        elif name == "spike_slice_to_raster":
            result = await analysis.spike_slice_to_raster(**arguments)
        elif name == "align_to_events":
            result = await analysis.align_to_events(**arguments)

        # RateSliceStack analysis tools
        elif name == "compute_rate_slice_unit_corr":
            result = await analysis.compute_rate_slice_unit_corr(**arguments)
        elif name == "compute_rate_slice_time_corr":
            result = await analysis.compute_rate_slice_time_corr(**arguments)
        elif name == "compute_unit_to_unit_slice_corr":
            result = await analysis.compute_unit_to_unit_slice_corr(**arguments)
        elif name == "compute_rate_slice_unit_order":
            result = await analysis.compute_rate_slice_unit_order(**arguments)

        # Pairwise matrix conditioning tools
        elif name == "remove_by_condition":
            result = await analysis.remove_by_condition(**arguments)

        # SpikeSliceStack analysis tools
        elif name == "spike_unit_to_unit_comparison":
            result = await analysis.spike_unit_to_unit_comparison(**arguments)
        elif name == "spike_slice_to_slice_unit_comparison":
            result = await analysis.spike_slice_to_slice_unit_comparison(**arguments)
        elif name == "compute_frac_active":
            result = await analysis.compute_frac_active(**arguments)
        elif name == "spike_order_units_across_slices":
            result = await analysis.spike_order_units_across_slices(**arguments)

        # Unit timing and rank-order correlation tools
        elif name == "get_unit_timing_per_slice_spike":
            result = await analysis.get_unit_timing_per_slice_spike(**arguments)
        elif name == "get_unit_timing_per_slice_rate":
            result = await analysis.get_unit_timing_per_slice_rate(**arguments)
        elif name == "rank_order_correlation_spike":
            result = await analysis.rank_order_correlation_spike(**arguments)
        elif name == "rank_order_correlation_rate":
            result = await analysis.rank_order_correlation_rate(**arguments)

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

        # GPLVM tools
        elif name == "fit_gplvm":
            result = await analysis.fit_gplvm(**arguments)
        elif name == "compute_gplvm_state_entropy":
            result = await analysis.compute_gplvm_state_entropy(**arguments)
        elif name == "compute_gplvm_continuity_prob":
            result = await analysis.compute_gplvm_continuity_prob(**arguments)
        elif name == "compute_gplvm_avg_state_prob":
            result = await analysis.compute_gplvm_avg_state_prob(**arguments)
        elif name == "compute_gplvm_consecutive_durations":
            result = await analysis.compute_gplvm_consecutive_durations(**arguments)

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
        elif name == "merge_workspace":
            result = await analysis.merge_workspace(**arguments)
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
