# MCP Server for IntegratedAnalysisTools

A Model Context Protocol (MCP) server that exposes comprehensive spike data analysis capabilities through a chat interface. This server allows users to load, analyze, and export neuronal spike train data using the IntegratedAnalysisTools library.

## Overview

The MCP server provides programmatic access to all analysis methods in the IntegratedAnalysisTools repository, enabling interactive data analysis workflows. It supports:

- **Data Loading**: Load spike data from HDF5, NWB, KiloSort/Phy, and SpikeInterface formats
- **S3 Integration**: Seamlessly handle data files stored in Amazon S3
- **Comprehensive Analysis**: Access all SpikeData analysis methods (rates, rasters, STTC, latencies, etc.)
- **Data Export**: Export results to HDF5, NWB, and KiloSort formats
- **Session Management**: Maintain multiple concurrent analysis sessions

## Server Structure

```
mcp_server/
├── __init__.py              # Package initialization
├── __main__.py              # CLI entry point
├── server.py                # Main MCP server implementation
├── s3_utils.py              # S3 file download/upload utilities
├── sessions.py              # Session management for SpikeData objects
├── tools/                   # MCP tool implementations
│   ├── __init__.py
│   ├── data_loaders.py      # Data loading tools
│   ├── analysis.py          # Analysis tools
│   └── exporters.py         # Data export tools
└── README.md                # This file
```

### Core Components

#### `server.py`
Main MCP server that:
- Registers all available tools
- Handles stdio transport for MCP communication
- Routes tool calls to appropriate handlers
- Manages error handling and response formatting

#### `s3_utils.py`
Utilities for S3 integration:
- `is_s3_url()`: Detect S3 URLs (s3:// or https://s3...)
- `parse_s3_url()`: Extract bucket and key from S3 URLs
- `download_from_s3()`: Download files from S3 to temporary local files
- `ensure_local_file()`: Handle both local paths and S3 URLs transparently

#### `sessions.py`
Session management:
- `SessionManager`: Manages in-memory storage of SpikeData objects
- Supports multiple concurrent sessions
- Automatic session expiration and cleanup
- Session information retrieval

#### `tools/data_loaders.py`
Data loading tools:
- `load_from_hdf5`: Load from HDF5 files (4 styles: raster, ragged, group, paired)
- `load_from_nwb`: Load from NWB files
- `load_from_kilosort`: Load from KiloSort/Phy outputs
- `load_from_hdf5_thresholded`: Load and threshold raw HDF5 data

#### `tools/analysis.py`
Comprehensive analysis tools:
- **Basic Analysis**: `compute_rates`, `compute_binned`, `compute_binned_meanrate`
- **Raster Generation**: `compute_raster`, `compute_sparse_raster`, `compute_channel_raster`
- **Time-based Analysis**: `compute_interspike_intervals`, `compute_resampled_isi`, `subtime`, `subset`
- **Correlation Analysis**: `compute_spike_time_tiling`, `compute_spike_time_tilings`
- **Latency Analysis**: `compute_latencies`, `compute_latencies_to_index`
- **Burst Analysis**: `get_frac_active`
- **Data Inspection**: `get_data_info`, `list_neurons`

#### `tools/exporters.py`
Data export tools:
- `export_to_hdf5`: Export to HDF5 (4 styles)
- `export_to_nwb`: Export to NWB format
- `export_to_kilosort`: Export to KiloSort/Phy format

## Installation

1. Install dependencies:
```bash
pip install -e ".[dev]"  # Includes mcp and boto3
```

2. Ensure AWS credentials are configured (for S3 access):
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
# Or use AWS IAM roles if running on EC2/Lambda
```

## Usage

### Running the Server

Start the MCP server using stdio transport:

```bash
python -m mcp_server
```

The server communicates via stdin/stdout using the MCP protocol.

### Example Workflow

1. **Load Data**:
   - Use `load_from_nwb` with an S3 URL: `s3://bucket/data.nwb`
   - Receive a `session_id` in response

2. **Analyze Data**:
   - Use `compute_rates` with the `session_id`
   - Use `compute_raster` to generate spike rasters
   - Use `compute_spike_time_tilings` for correlation analysis

3. **Export Results**:
   - Use `export_to_hdf5` with the `session_id` and output path
   - Results can be saved locally or uploaded to S3

### Tool Parameters

All tools that operate on SpikeData accept a `session_id` parameter. Loader tools return a `session_id` that should be used for subsequent analysis operations.

S3 URLs are automatically detected and handled. Files are downloaded to temporary locations, processed, and cleaned up automatically.

## Contribution Guidelines

### Adding New Tools

1. **Implement the Tool Function**:
   - Add async function in appropriate `tools/` module
   - Follow existing patterns for error handling and result formatting
   - Convert NumPy arrays to lists for JSON serialization

2. **Register the Tool**:
   - Add tool definition in `server.py` `list_tools()` function
   - Include comprehensive description and input schema
   - Add handler in `call_tool()` function

3. **Update Documentation**:
   - Add tool description to this README
   - Include example usage if applicable

### Code Style

- Follow existing code patterns and conventions
- Use type hints for function parameters and return values
- Include docstrings for all public functions
- Handle errors gracefully with informative messages

### Testing

- Test tools with both local files and S3 URLs
- Verify session management works correctly
- Test error handling for invalid inputs
- Ensure NumPy arrays are properly serialized

### Pull Request Process

1. Create a feature branch from `main`
2. Implement changes following contribution guidelines
3. Add tests if applicable
4. Update documentation
5. Submit pull request with clear description

## Future Enhancements

### GitHub Actions Integration

We plan to implement GitHub Actions workflows that:

1. **Automated Testing**: Run tests on all pull requests
2. **Agent Launch**: Automatically launch an analysis agent when functionality is changed or added via PR
3. **Documentation Updates**: Automatically update API documentation
4. **Version Management**: Handle version bumps and releases

The agent launch workflow will:
- Trigger on PRs that modify tool implementations
- Run comprehensive tests on new/changed functionality
- Generate analysis reports
- Provide feedback in PR comments

### Planned Features

- Support for additional data formats
- Batch processing capabilities
- Result caching for expensive operations
- WebSocket transport option
- Authentication and authorization
- Rate limiting and resource management

## Troubleshooting

### S3 Access Issues

- Verify AWS credentials are set correctly
- Check bucket permissions and IAM policies
- Ensure S3 URLs are correctly formatted

### Session Not Found Errors

- Sessions expire after 1 hour by default
- Check that you're using the correct `session_id`
- Create a new session if the old one expired

### Memory Issues

- Large datasets may require significant memory
- Consider using `subtime` or `subset` to work with smaller portions
- Sparse rasters are more memory-efficient than dense rasters

## License

See the main repository LICENSE file for details.

## Support

For issues, questions, or contributions, please use the main repository's issue tracker.

