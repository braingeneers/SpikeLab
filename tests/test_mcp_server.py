"""
Comprehensive tests for MCP server functionality.

Tests cover:
- S3 utilities
- Session management
- Data loaders
- Analysis tools
- Export tools
- Server integration
"""

import json
import os
import pathlib
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Check for MCP dependencies - import in stages
MCP_IMPORT_ERROR = None
MCP_AVAILABLE = False
MCP_SERVER_AVAILABLE = False

# Basic imports (no mcp dependency)
try:
    from mcp_server.s3_utils import (
        download_from_s3,
        ensure_local_file,
        is_s3_url,
        parse_s3_url,
    )
    from mcp_server.sessions import SessionManager, get_session_manager
    from spikedata import SpikeData
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_IMPORT_ERROR = str(e)

# Server imports (requires mcp package)
if MCP_AVAILABLE:
    try:
        from mcp_server.server import server
        from mcp_server.tools import analysis, data_loaders, exporters
        MCP_SERVER_AVAILABLE = True
    except ImportError as e:
        MCP_IMPORT_ERROR = str(e)


# ============================================================================
# Test Markers
# ============================================================================

# Skip infrastructure tests if basic imports fail
pytestmark_infra = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason=f"MCP server not available: {MCP_IMPORT_ERROR or 'dependencies not installed'}",
)

# Skip server/tool tests if mcp package is missing
pytestmark_server = pytest.mark.skipif(
    not MCP_SERVER_AVAILABLE,
    reason=f"MCP package not installed. Install with: pip install mcp",
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_spikedata():
    """Create a sample SpikeData object for testing."""
    train = [
        [10.0, 20.0, 30.0, 40.0],  # Neuron 0: 4 spikes
        [15.0, 25.0, 35.0],  # Neuron 1: 3 spikes
        [5.0, 45.0],  # Neuron 2: 2 spikes
    ]
    return SpikeData(train, length=50.0, metadata={"test": "data"})


@pytest.fixture
def session_id(sample_spikedata):
    """Create a session with sample data."""
    session_manager = get_session_manager()
    session_manager._sessions.clear()  # Reset for test isolation
    return session_manager.create_session(sample_spikedata)


@pytest.fixture(autouse=True)
def reset_session_manager():
    """Reset session manager before each test."""
    session_manager = get_session_manager()
    session_manager._sessions.clear()
    yield
    session_manager._sessions.clear()


# ============================================================================
# S3 Utilities Tests
# ============================================================================

# Skip all tests if MCP dependencies are not available
pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason=f"MCP dependencies not available: {MCP_IMPORT_ERROR or 'mcp package not installed'}",
)


# Skip infrastructure tests if basic imports fail
pytestmark_infra = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason=f"MCP server not available: {MCP_IMPORT_ERROR or 'dependencies not installed'}",
)

# Skip server/tool tests if mcp package is missing
pytestmark_server = pytest.mark.skipif(
    not MCP_SERVER_AVAILABLE,
    reason=f"MCP package not installed. Install with: pip install mcp",
)


class TestS3Utils:
    """Test S3 utility functions."""

    @pytestmark_infra
    def test_is_s3_url(self):
        """Test S3 URL detection."""
        assert is_s3_url("s3://bucket/key") is True
        assert is_s3_url("https://s3.amazonaws.com/bucket/key") is True
        assert is_s3_url("/local/path") is False
        assert is_s3_url("file.h5") is False

    @pytestmark_infra
    def test_parse_s3_url(self):
        """Test S3 URL parsing."""
        bucket, key = parse_s3_url("s3://bucket/key")
        assert bucket == "bucket"
        assert key == "key"

        bucket, key = parse_s3_url("s3://my-bucket/path/to/file.h5")
        assert bucket == "my-bucket"
        assert key == "path/to/file.h5"

    @pytestmark_infra
    @patch("mcp_server.s3_utils.boto3")
    def test_download_from_s3(self, mock_boto3):
        """Test S3 download."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        def mock_download(bucket, key, local_path):
            with open(local_path, "wb") as f:
                f.write(b"test data")

        mock_client.download_file.side_effect = mock_download

        result = download_from_s3("s3://bucket/key.h5")
        assert os.path.exists(result)
        os.unlink(result)

    @pytestmark_infra
    def test_ensure_local_file_local(self):
        """Test local file handling."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"data")
            tmp_path = tmp.name

        try:
            local_path, is_temp = ensure_local_file(tmp_path)
            assert local_path == tmp_path
            assert is_temp is False
        finally:
            os.unlink(tmp_path)


# ============================================================================
# Session Management Tests
# ============================================================================

class TestSessionManagement:
    """Test session management functionality."""

    @pytestmark_infra
    def test_create_and_get_session(self, sample_spikedata):
        """Test creating and retrieving sessions."""
        session_manager = SessionManager()
        session_id = session_manager.create_session(sample_spikedata)
        
        retrieved = session_manager.get_session(session_id)
        assert retrieved is not None
        assert retrieved.N == sample_spikedata.N

    @pytestmark_infra
    def test_session_expiration(self, sample_spikedata):
        """Test session expiration."""
        session_manager = SessionManager()
        session_id = session_manager.create_session(sample_spikedata, ttl_seconds=0.1)
        
        assert session_manager.get_session(session_id) is not None
        time.sleep(0.2)
        assert session_manager.get_session(session_id) is None

    @pytestmark_infra
    def test_update_and_delete_session(self, sample_spikedata):
        """Test updating and deleting sessions."""
        session_manager = SessionManager()
        session_id = session_manager.create_session(sample_spikedata)
        
        new_sd = SpikeData([[1.0, 2.0]], length=10.0)
        assert session_manager.update_session(session_id, new_sd) is True
        assert session_manager.delete_session(session_id) is True
        assert session_manager.get_session(session_id) is None


# ============================================================================
# Data Loader Tests
# ============================================================================

class TestDataLoaders:
    """Test data loading tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_load_from_nwb(self):
        """Test loading from NWB file."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nwb") as tmp:
            f = h5py.File(tmp.name, "w")
            units = f.create_group("units")
            spike_times = np.array([10.0, 20.0, 30.0]) / 1000.0
            spike_times_index = np.array([1, 2, 3])
            units.create_dataset("spike_times", data=spike_times)
            units.create_dataset("spike_times_index", data=spike_times_index)
            f.close()

            result = await data_loaders.load_from_nwb(tmp.name)
            assert "session_id" in result
            assert result["info"]["num_neurons"] == 3
            os.unlink(tmp.name)

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("mcp_server.tools.data_loaders.ensure_local_file")
    async def test_load_from_hdf5_s3(self, mock_ensure):
        """Test loading HDF5 from S3."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            f = h5py.File(tmp.name, "w")
            spike_times = np.array([10.0, 20.0]) / 1000.0
            spike_times_index = np.array([1, 2])
            f.create_dataset("spike_times", data=spike_times)
            f.create_dataset("spike_times_index", data=spike_times_index)
            f.close()

            mock_ensure.return_value = (tmp.name, True)
            result = await data_loaders.load_from_hdf5(
                "s3://bucket/data.h5",
                style="ragged",
                spike_times_dataset="spike_times",
                spike_times_index_dataset="spike_times_index",
                spike_times_unit="s",
            )
            assert "session_id" in result
            # File may have been cleaned up by ensure_local_file if is_temp=True
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


# ============================================================================
# Analysis Tools Tests
# ============================================================================

class TestAnalysisTools:
    """Test analysis tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rates(self, session_id):
        """Test computing firing rates."""
        result = await analysis.compute_rates(session_id, unit="kHz")
        assert "rates" in result
        assert "unit" in result
        assert result["unit"] == "kHz"
        assert len(result["rates"]) == 3

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_raster(self, session_id):
        """Test computing raster."""
        result = await analysis.compute_raster(session_id, bin_size=5.0)
        assert "raster" in result
        assert "bin_size_ms" in result
        assert result["bin_size_ms"] == 5.0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_spike_time_tiling(self, session_id):
        """Test computing STTC."""
        result = await analysis.compute_spike_time_tiling(
            session_id, neuron_i=0, neuron_j=1, delt=10.0
        )
        assert "sttc" in result
        assert isinstance(result["sttc"], (int, float))

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_subtime(self, session_id):
        """Test time window extraction."""
        result = await analysis.subtime(session_id, start=10.0, end=30.0)
        assert "session_id" in result
        assert result["info"]["start_ms"] == 10.0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_data_info(self, session_id, sample_spikedata):
        """Test getting data information."""
        result = await analysis.get_data_info(session_id)
        assert result["num_neurons"] == sample_spikedata.N
        assert result["length_ms"] == sample_spikedata.length

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ValueError, match="Session not found"):
            await analysis.compute_rates("invalid-session-id")


# ============================================================================
# Export Tools Tests
# ============================================================================

class TestExportTools:
    """Test export tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_hdf5(self, session_id):
        """Test exporting to HDF5."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp_path = tmp.name

        try:
            result = await exporters.export_to_hdf5(
                session_id, tmp_path, style="ragged", spike_times_unit="s"
            )
            assert "file_path" in result
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_nwb(self, session_id):
        """Test exporting to NWB."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nwb") as tmp:
            tmp_path = tmp.name

        try:
            result = await exporters.export_to_nwb(session_id, tmp_path)
            assert "file_path" in result
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_kilosort(self, session_id):
        """Test exporting to KiloSort."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await exporters.export_to_kilosort(
                session_id, tmpdir, fs_Hz=1000.0
            )
            assert "folder_path" in result
            assert len(result["files"]) == 2


# ============================================================================
# Server Integration Tests
# ============================================================================

class TestServerIntegration:
    """Test server integration and tool registration."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that tools are registered."""
        # Access the registered handler through the server's request handlers
        from mcp_server.server import _list_tools
        tools = await _list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

        tool_names = [tool.name for tool in tools]
        assert "load_from_nwb" in tool_names
        assert "compute_rates" in tool_names
        assert "export_to_hdf5" in tool_names

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_tool_schemas(self):
        """Test tool schemas are valid."""
        from mcp_server.server import _list_tools
        tools = await _list_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            assert tool.inputSchema["type"] == "object"

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("mcp_server.server.analysis.compute_rates")
    async def test_call_tool(self, mock_compute):
        """Test calling a tool through the server."""
        from mcp_server.server import _call_tool
        mock_compute.return_value = {
            "rates": [0.1, 0.2, 0.3],
            "unit": "kHz",
            "num_neurons": 3,
        }

        result = await _call_tool(
            "compute_rates", {"session_id": "test-session", "unit": "kHz"}
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "rates" in data
        mock_compute.assert_called_once()

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test error handling for unknown tool."""
        from mcp_server.server import _call_tool
        result = await _call_tool("unknown_tool", {})
        data = json.loads(result[0].text)
        assert "error" in data
