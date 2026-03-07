"""
Main MCP server implementation for spike data analysis.

Registers all tools and handles stdio transport.
"""

import asyncio
import sys
from typing import Any

from mcp.server import Server
from mcp import types
from mcp.server.stdio import stdio_server

from mcp_server.tools import analysis, data_loaders, exporters

# Create the MCP server instance
server = Server("integrated-analysis-tools")


# Register data loader tools
@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    """List all available tools."""
    tools = []

    # Data loader tools
    tools.extend(
        [
            types.Tool(
                name="load_from_hdf5",
                description="Load spike data from an HDF5 file. Supports raster, ragged, group-per-unit, and paired array styles. Accepts local file paths or S3 URLs.",
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
                description="Load spike data from an NWB file. Accepts local file paths or S3 URLs.",
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
                description="Load spike data from KiloSort/Phy output folder.",
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
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["folder_path", "fs_Hz"],
                },
            ),
            types.Tool(
                name="load_from_pickle",
                description="Load spike data from a pickle file. Accepts local file paths or S3 URLs. WARNING: only load from trusted sources.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
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
                description="Load and threshold raw data from an HDF5 file.",
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

    # Analysis tools
    tools.extend(
        [
            types.Tool(
                name="compute_rates",
                description="Calculate the mean firing rate of each neuron.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "unit": {
                            "type": "string",
                            "enum": ["Hz", "kHz"],
                            "default": "kHz",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_binned",
                description="Get binned spike counts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "bin_size": {"type": "number", "default": 40.0},
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_binned_meanrate",
                description="Calculate the mean firing rate across the population in each time bin.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "bin_size": {"type": "number", "default": 40.0},
                        "unit": {
                            "type": "string",
                            "enum": ["Hz", "kHz"],
                            "default": "kHz",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_raster",
                description="Generate a dense spike raster matrix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "bin_size": {"type": "number", "default": 20.0},
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_sparse_raster",
                description="Generate a sparse spike raster matrix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "bin_size": {"type": "number", "default": 20.0},
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_channel_raster",
                description="Generate a channel-aggregated raster matrix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "bin_size": {"type": "number", "default": 20.0},
                        "channel_attr": {
                            "type": "string",
                            "description": "Channel attribute name",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_interspike_intervals",
                description="Calculate interspike intervals for each neuron.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_resampled_isi",
                description="Calculate firing rate at specific times using resampled ISI method.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of times in ms",
                        },
                        "sigma_ms": {"type": "number", "default": 10.0},
                    },
                    "required": ["session_id", "times"],
                },
            ),
            types.Tool(
                name="subtime",
                description="Extract a time window from the spike data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "start": {"type": "number", "description": "Start time in ms"},
                        "end": {"type": "number", "description": "End time in ms"},
                        "create_new_session": {"type": "boolean", "default": False},
                    },
                    "required": ["session_id", "start", "end"],
                },
            ),
            types.Tool(
                name="subset",
                description="Select specific neurons from the spike data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "units": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of unit indices",
                        },
                        "by": {
                            "type": "string",
                            "description": "Attribute name to select by",
                        },
                        "create_new_session": {"type": "boolean", "default": False},
                    },
                    "required": ["session_id", "units"],
                },
            ),
            types.Tool(
                name="compute_spike_time_tiling",
                description="Calculate spike time tiling coefficient between two neurons.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "neuron_i": {"type": "integer"},
                        "neuron_j": {"type": "integer"},
                        "delt": {"type": "number", "default": 20.0},
                    },
                    "required": ["session_id", "neuron_i", "neuron_j"],
                },
            ),
            types.Tool(
                name="compute_spike_time_tilings",
                description="Compute the full spike time tiling coefficient matrix for all neuron pairs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "delt": {"type": "number", "default": 20.0},
                    },
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_latencies",
                description="Compute latencies from reference times to spikes in each neuron.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of reference times in ms",
                        },
                        "window_ms": {"type": "number", "default": 100.0},
                    },
                    "required": ["session_id", "times"],
                },
            ),
            types.Tool(
                name="compute_latencies_to_index",
                description="Compute latencies from a specific neuron to all other neurons.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "neuron_index": {"type": "integer"},
                        "window_ms": {"type": "number", "default": 100.0},
                    },
                    "required": ["session_id", "neuron_index"],
                },
            ),
            types.Tool(
                name="get_frac_active",
                description="Calculate fraction of active neurons/units in bursts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "edges": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "List of [start, end] pairs for each burst",
                        },
                        "min_spikes": {"type": "integer"},
                        "backbone_threshold": {
                            "type": "number",
                            "description": "Threshold between 0-1",
                        },
                    },
                    "required": [
                        "session_id",
                        "edges",
                        "min_spikes",
                        "backbone_threshold",
                    ],
                },
            ),
            types.Tool(
                name="get_data_info",
                description="Get information about the spike data in a session.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="list_neurons",
                description="List available neurons with their attributes.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="get_pop_rate",
                description="Compute the smoothed population firing rate using square then Gaussian convolution.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="compute_spike_trig_pop_rate",
                description="Compute spike-triggered population rate (stPR) for each neuron.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                    "required": ["session_id"],
                },
            ),
            types.Tool(
                name="get_bursts",
                description="Detect bursts from the population firing rate using thresholded peak finding.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "square_width": {
                            "type": "integer",
                            "default": 20,
                            "description": "Square smoothing window width in bins",
                        },
                        "gauss_sigma": {
                            "type": "integer",
                            "default": 100,
                            "description": "Gaussian smoothing sigma in bins",
                        },
                        "acc_square_width": {
                            "type": "integer",
                            "default": 5,
                            "description": "Square window width for accurate pop rate in bins",
                        },
                        "acc_gauss_sigma": {
                            "type": "integer",
                            "default": 5,
                            "description": "Gaussian sigma for accurate pop rate in bins",
                        },
                        "raster_bin_size_ms": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Raster bin size in ms",
                        },
                        "peak_to_trough": {
                            "type": "boolean",
                            "default": True,
                            "description": "Use trough-to-trough baseline (True) or zero baseline (False)",
                        },
                        "pop_rms_override": {
                            "type": "number",
                            "description": "Override baseline RMS for cross-dataset normalization",
                        },
                    },
                    "required": [
                        "session_id",
                        "thr_burst",
                        "min_burst_diff",
                        "burst_edge_mult_thresh",
                    ],
                },
            ),
            types.Tool(
                name="threshold_spike_time_tilings",
                description="Compute the full STTC matrix and apply a binary threshold, returning a binary connectivity matrix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                    "required": ["session_id", "threshold"],
                },
            ),
            types.Tool(
                name="compute_pairwise_fr_corr",
                description="Compute the pairwise unit-to-unit firing rate correlation matrix from instantaneous firing rates.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of time points in ms at which to evaluate instantaneous firing rates",
                        },
                        "sigma_ms": {
                            "type": "number",
                            "default": 10.0,
                            "description": "Gaussian smoothing sigma in ms for ISI resampling",
                        },
                        "max_lag": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum lag in time bins to consider for cross-correlation",
                        },
                    },
                    "required": ["session_id", "times"],
                },
            ),
            types.Tool(
                name="compute_rate_manifold",
                description="Project instantaneous firing rates into a low-dimensional manifold using PCA or UMAP.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "times": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of time points in ms at which to evaluate instantaneous firing rates",
                        },
                        "sigma_ms": {
                            "type": "number",
                            "default": 10.0,
                            "description": "Gaussian smoothing sigma in ms for ISI resampling",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["PCA", "UMAP"],
                            "default": "PCA",
                            "description": "Dimensionality reduction method",
                        },
                        "n_components": {
                            "type": "integer",
                            "default": 2,
                            "description": "Number of output dimensions",
                        },
                        "n_neighbors": {
                            "type": "integer",
                            "description": "UMAP: number of neighbors (optional)",
                        },
                        "min_dist": {
                            "type": "number",
                            "description": "UMAP: minimum distance (optional)",
                        },
                        "metric": {
                            "type": "string",
                            "description": "UMAP: distance metric (optional)",
                        },
                        "random_state": {
                            "type": "integer",
                            "description": "UMAP: random seed (optional)",
                        },
                    },
                    "required": ["session_id", "times"],
                },
            ),
            types.Tool(
                name="compute_rate_slice_unit_corr",
                description="Compute slice-to-slice unit correlation across event-aligned firing rate slices.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "min_rate_threshold": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Minimum mean firing rate threshold (kHz) to include a unit",
                        },
                        "min_frac": {
                            "type": "number",
                            "default": 0.3,
                            "description": "Minimum fraction of slices where a unit must exceed the rate threshold",
                        },
                        "max_lag": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum lag in time bins for cross-correlation",
                        },
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cross_correlation",
                            "description": "Similarity function to use",
                        },
                    },
                    "required": ["session_id", "times_start_to_end"],
                },
            ),
            types.Tool(
                name="compute_rate_slice_time_corr",
                description="Compute slice-to-slice time-bin correlation across event-aligned firing rate slices.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "max_lag": {
                            "type": "integer",
                            "default": 0,
                            "description": "Maximum lag in time bins for comparison",
                        },
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cosine_similarity",
                            "description": "Similarity function to use",
                        },
                    },
                    "required": ["session_id", "times_start_to_end"],
                },
            ),
            types.Tool(
                name="compute_unit_to_unit_slice_corr",
                description="Compute unit-to-unit correlation averaged across event-aligned firing rate slices.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "max_lag": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum lag in time bins for cross-correlation",
                        },
                        "compare_func": {
                            "type": "string",
                            "enum": ["cross_correlation", "cosine_similarity"],
                            "default": "cross_correlation",
                            "description": "Similarity function to use",
                        },
                    },
                    "required": ["session_id", "times_start_to_end"],
                },
            ),
            types.Tool(
                name="compute_rate_slice_unit_order",
                description="Order units by their peak firing time across event-aligned firing rate slices.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "agg_func": {
                            "type": "string",
                            "default": "median",
                            "description": "Aggregation function across slices ('median' or 'mean')",
                        },
                        "min_rate_threshold": {
                            "type": "number",
                            "default": 0.1,
                            "description": "Minimum mean firing rate threshold (kHz) to include a unit",
                        },
                    },
                    "required": ["session_id", "times_start_to_end"],
                },
            ),
            types.Tool(
                name="compute_spike_slice_sparse_matrices",
                description="Build event-aligned spike slices and return a (U, T, S) binary sparse raster stack.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "bin_size": {
                            "type": "number",
                            "default": 1.0,
                            "description": "Bin size in ms for the sparse raster",
                        },
                    },
                    "required": ["session_id", "times_start_to_end"],
                },
            ),
        ]
    )

    # Export tools
    tools.extend(
        [
            types.Tool(
                name="export_to_hdf5",
                description="Export spike data to an HDF5 file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                    "required": ["session_id", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_nwb",
                description="Export spike data to an NWB file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                    "required": ["session_id", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_pickle",
                description="Export spike data to a pickle file. Accepts local file paths or S3 URLs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "file_path": {
                            "type": "string",
                            "description": "Local file path or S3 URL",
                        },
                        "protocol": {
                            "type": "integer",
                            "description": "Pickle protocol version (None uses highest available)",
                        },
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["session_id", "file_path"],
                },
            ),
            types.Tool(
                name="export_to_kilosort",
                description="Export spike data to a KiloSort/Phy folder.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                        "aws_access_key_id": {"type": "string"},
                        "aws_secret_access_key": {"type": "string"},
                        "aws_session_token": {"type": "string"},
                        "region_name": {"type": "string"},
                    },
                    "required": ["session_id", "folder_path", "fs_Hz"],
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

        # Analysis tools
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
        elif name == "subtime":
            result = await analysis.subtime(**arguments)
        elif name == "subset":
            result = await analysis.subset(**arguments)
        elif name == "compute_spike_time_tiling":
            result = await analysis.compute_spike_time_tiling(**arguments)
        elif name == "compute_spike_time_tilings":
            result = await analysis.compute_spike_time_tilings(**arguments)
        elif name == "compute_latencies":
            result = await analysis.compute_latencies(**arguments)
        elif name == "compute_latencies_to_index":
            result = await analysis.compute_latencies_to_index(**arguments)
        elif name == "get_frac_active":
            result = await analysis.get_frac_active(**arguments)
        elif name == "get_data_info":
            result = await analysis.get_data_info(**arguments)
        elif name == "list_neurons":
            result = await analysis.list_neurons(**arguments)
        elif name == "get_pop_rate":
            result = await analysis.get_pop_rate(**arguments)
        elif name == "compute_spike_trig_pop_rate":
            result = await analysis.compute_spike_trig_pop_rate(**arguments)
        elif name == "get_bursts":
            result = await analysis.get_bursts(**arguments)
        elif name == "threshold_spike_time_tilings":
            result = await analysis.threshold_spike_time_tilings(**arguments)
        elif name == "compute_pairwise_fr_corr":
            result = await analysis.compute_pairwise_fr_corr(**arguments)
        elif name == "compute_rate_manifold":
            result = await analysis.compute_rate_manifold(**arguments)
        elif name == "compute_rate_slice_unit_corr":
            result = await analysis.compute_rate_slice_unit_corr(**arguments)
        elif name == "compute_rate_slice_time_corr":
            result = await analysis.compute_rate_slice_time_corr(**arguments)
        elif name == "compute_unit_to_unit_slice_corr":
            result = await analysis.compute_unit_to_unit_slice_corr(**arguments)
        elif name == "compute_rate_slice_unit_order":
            result = await analysis.compute_rate_slice_unit_order(**arguments)
        elif name == "compute_spike_slice_sparse_matrices":
            result = await analysis.compute_spike_slice_sparse_matrices(**arguments)

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

        import json

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        import json

        error_result = {"error": str(e), "type": type(e).__name__}
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
