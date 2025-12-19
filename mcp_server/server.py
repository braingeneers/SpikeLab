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

        # Export tools
        elif name == "export_to_hdf5":
            result = await exporters.export_to_hdf5(**arguments)
        elif name == "export_to_nwb":
            result = await exporters.export_to_nwb(**arguments)
        elif name == "export_to_kilosort":
            result = await exporters.export_to_kilosort(**arguments)

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
