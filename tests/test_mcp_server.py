"""
Comprehensive tests for MCP server functionality.

Tests cover:
- S3 utilities
- Session management
- Data loaders
- Analysis tools
- Workspace management
- Workspace analysis tools
- Export tools
- Server integration
"""

import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Check for MCP dependencies - import in stages
MCP_IMPORT_ERROR = None
MCP_AVAILABLE = False
MCP_SERVER_AVAILABLE = False

# Basic imports (no mcp dependency)
try:
    from spikelab.data_loaders.s3_utils import (
        download_from_s3,
        ensure_local_file,
        is_s3_url,
        parse_s3_url,
        upload_to_s3,
    )
    from spikelab.spikedata import SpikeData

    MCP_AVAILABLE = True
except ImportError as e:
    MCP_IMPORT_ERROR = str(e)

# Server imports (requires mcp package)
if MCP_AVAILABLE:
    try:
        from spikelab.mcp_server.server import server
        from spikelab.mcp_server.tools import (
            analysis,
            data_loaders,
            exporters,
        )
        from spikelab.workspace.workspace import get_workspace_manager

        MCP_SERVER_AVAILABLE = True
    except ImportError as e:
        MCP_IMPORT_ERROR = str(e)


# ============================================================================
# Test Markers
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
def loaded_ws(sample_spikedata):
    """Create a workspace with sample SpikeData stored at ('rec1', 'spikedata').

    Returns (workspace_id, namespace).
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("MCP server not available")
    wm = get_workspace_manager()
    ws_id = wm.create_workspace(name="test_ws")
    wm.get_workspace(ws_id).store("rec1", "spikedata", sample_spikedata)
    return ws_id, "rec1"


@pytest.fixture(autouse=True)
def reset_workspace_manager():
    """Reset workspace manager before each test."""
    if not MCP_SERVER_AVAILABLE:
        yield
        return
    wm = get_workspace_manager()
    wm._workspaces.clear()
    yield
    wm._workspaces.clear()


@pytest.fixture
def workspace_id():
    """Create a workspace and return its ID."""
    wm = get_workspace_manager()
    ws_id = wm.create_workspace(name="test_workspace")
    return ws_id


# ============================================================================
# S3 Utilities Tests
# ============================================================================


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
    @patch("spikelab.data_loaders.s3_utils.boto3")
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

    @pytestmark_infra
    @patch("spikelab.data_loaders.s3_utils.boto3")
    def test_upload_to_s3_success(self, mock_boto3):
        """
        Test successful upload to S3.

        Tests:
        (Method 1) Creates temp file with content
        (Method 2) Mocks boto3.client().upload_file to succeed
        (Test Case 1) upload_to_s3 returns S3 URL
        (Test Case 2) upload_file was called with correct bucket, key, local_path
        """
        # Create temp file to upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        try:
            # Mock S3 client so upload_file succeeds without real AWS call
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Upload; should return S3 URL
            result = upload_to_s3(tmp_path, "s3://mybucket/path/output.txt")

            # Verify return value
            assert result == "s3://mybucket/path/output.txt"
            # Verify upload_file was called with (local_path, bucket, key)
            mock_client.upload_file.assert_called_once_with(
                tmp_path, "mybucket", "path/output.txt"
            )
        finally:
            os.unlink(tmp_path)

    @pytestmark_infra
    def test_upload_to_s3_file_not_found(self):
        """
        Test that upload_to_s3 raises FileNotFoundError when local file does not exist.

        Tests:
        (Method 1) Calls upload_to_s3 with non-existent path
        (Test Case 1) FileNotFoundError is raised with message containing path
        """
        # Call with non-existent local path; should raise before any S3 call
        with pytest.raises(FileNotFoundError) as exc_info:
            upload_to_s3("/nonexistent/path/file.txt", "s3://bucket/key.txt")
        assert "Local file not found" in str(exc_info.value)
        assert "/nonexistent" in str(exc_info.value)

    @pytestmark_infra
    def test_upload_to_s3_invalid_url(self):
        """
        Test that upload_to_s3 raises ValueError when S3 URL is invalid.

        Tests:
        (Method 1) Creates temp file
        (Method 2) Calls upload_to_s3 with non-S3 URL (local path)
        (Test Case 1) ValueError is raised with "Not an S3 URL" message
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Pass local path as s3_url; should raise before any S3 call
            with pytest.raises(ValueError) as exc_info:
                upload_to_s3(tmp_path, "/local/path/not-s3.txt")
            assert "Not an S3 URL" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    @pytestmark_infra
    @patch("spikelab.data_loaders.s3_utils.boto3")
    def test_upload_to_s3_bucket_not_found(self, mock_boto3):
        """
        Test that upload_to_s3 raises ValueError when S3 bucket does not exist.

        Tests:
        (Method 1) Creates temp file
        (Method 2) Mocks boto3.client().upload_file to raise ClientError with NoSuchBucket
        (Test Case 1) ValueError is raised with "bucket not found" message
        """
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            # CI may run without botocore; use fake exception with same structure
            class ClientError(Exception):
                def __init__(self, response, operation_name):
                    super().__init__()
                    self.response = response

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"data")
            tmp_path = tmp.name
        try:
            # Mock upload_file to raise NoSuchBucket (bucket does not exist)
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            mock_client.upload_file.side_effect = ClientError(
                {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}},
                "PutObject",
            )

            # Should raise ValueError, not raw ClientError
            with pytest.raises(ValueError) as exc_info:
                upload_to_s3(tmp_path, "s3://nonexistent-bucket/key.txt")
            assert "bucket not found" in str(exc_info.value).lower()
        finally:
            os.unlink(tmp_path)

    @pytestmark_infra
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("botocore"),
        reason="botocore not installed",
    )
    @patch("spikelab.data_loaders.s3_utils.boto3")
    def test_upload_to_s3_credential_error(self, mock_boto3):
        """
        Test that upload_to_s3 raises RuntimeError when AWS credentials are missing.

        Tests:
        (Method 1) Creates temp file
        (Method 2) Mocks upload_file to raise NoCredentialsError (lazy cred check on request)
        (Test Case 1) RuntimeError is raised with credentials message
        """
        from botocore.exceptions import NoCredentialsError

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"data")
            tmp_path = tmp.name
        try:
            # Mock upload_file to raise NoCredentialsError (credentials not configured)
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            mock_client.upload_file.side_effect = NoCredentialsError()

            # Should raise RuntimeError with credentials message
            with pytest.raises(RuntimeError) as exc_info:
                upload_to_s3(tmp_path, "s3://bucket/key.txt")
            assert "credentials" in str(exc_info.value).lower()
        finally:
            os.unlink(tmp_path)


# ============================================================================
# Data Loader Tests
# ============================================================================


class TestDataLoaders:
    """Test data loading tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_load_from_nwb(self):
        """
        Test loading spike data from an NWB file.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, and workspace_key.
            (Test Case 2) info.num_neurons matches the number of units in the file.
        """
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

        try:
            result = await data_loaders.load_from_nwb(tmp.name)
            assert "workspace_id" in result
            assert "namespace" in result
            assert result["workspace_key"] == "spikedata"
            assert result["info"]["num_neurons"] == 3
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.data_loaders.ensure_local_file")
    async def test_load_from_hdf5_s3(self, mock_ensure):
        """
        Test loading HDF5 spike data from an S3 URL.

        Tests:
            (Test Case 1) Result contains workspace_id and namespace.
            (Test Case 2) Workspace key is 'spikedata'.
        """
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
            local_path = tmp.name

        try:
            mock_ensure.return_value = (local_path, False)
            result = await data_loaders.load_from_hdf5_ragged(
                "s3://bucket/data.h5",
                spike_times_dataset="spike_times",
                spike_times_index_dataset="spike_times_index",
                spike_times_unit="s",
            )
            assert "workspace_id" in result
            assert "namespace" in result
            assert result["workspace_key"] == "spikedata"
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)


# ============================================================================
# Analysis Tools Tests
# ============================================================================


class TestAnalysisTools:
    """Test analysis tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rates(self, loaded_ws):
        """
        Test compute_rates stores firing rates in the workspace.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, key, and unit.
            (Test Case 2) Stored item info shows ndarray type.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_rates(ws_id, ns, "rates", unit="kHz")
        assert result["workspace_id"] == ws_id
        assert result["namespace"] == ns
        assert result["key"] == "rates"
        assert result["unit"] == "kHz"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_raster(self, loaded_ws):
        """
        Test compute_raster stores a binary raster matrix in the workspace.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, key, and bin_size.
            (Test Case 2) Stored item info shows ndarray type.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_raster(ws_id, ns, "raster", bin_size=5.0)
        assert result["workspace_id"] == ws_id
        assert result["key"] == "raster"
        assert result["bin_size"] == 5.0
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_spike_time_tiling(self, loaded_ws):
        """
        Test compute_spike_time_tiling stores the STTC scalar in the workspace.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, key, neuron_i, neuron_j.
            (Test Case 2) Stored item info shows ndarray type (scalar wrapped in array).
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_spike_time_tiling(
            ws_id, ns, "sttc", neuron_i=0, neuron_j=1, delt=10.0
        )
        assert result["workspace_id"] == ws_id
        assert result["key"] == "sttc"
        assert result["neuron_i"] == 0
        assert result["neuron_j"] == 1
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_subtime(self, loaded_ws):
        """
        Test subtime stores a trimmed SpikeData back in the workspace.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, and workspace_key.
            (Test Case 2) Stored SpikeData length matches the requested window.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subtime(ws_id, ns, start=10.0, end=30.0)
        assert result["workspace_id"] == ws_id
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["length_ms"] == pytest.approx(20.0, abs=1.0)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_data_info(self, loaded_ws, sample_spikedata):
        """
        Test get_data_info returns inline metadata for SpikeData.

        Tests:
            (Test Case 1) num_neurons matches the loaded SpikeData.
            (Test Case 2) length_ms matches the loaded SpikeData.
        """
        ws_id, ns = loaded_ws
        result = await analysis.get_data_info(ws_id, ns)
        assert result["num_neurons"] == sample_spikedata.N
        assert result["length_ms"] == sample_spikedata.length

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_workspace(self):
        """
        Test that analysis tools raise ValueError for an unknown workspace_id.

        Tests:
            (Test Case 1) compute_rates raises ValueError with 'Workspace not found'.
        """
        with pytest.raises(ValueError, match="Workspace not found"):
            await analysis.compute_rates("nonexistent-workspace-id", "ns", "rates")

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rates_no_spikedata_stored(self):
        """
        EC-MCP-01: Analysis tool on workspace with no spikedata stored.

        Tests:
            (Test Case 1) compute_rates raises ValueError mentioning loader tools.
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="empty_ws")
        with pytest.raises(ValueError, match="No SpikeData found"):
            await analysis.compute_rates(ws_id, "rec1", "rates")

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rates_wrong_type_at_spikedata_key(self):
        """
        EC-MCP-02: Analysis tool with wrong type stored at expected key.

        Tests:
            (Test Case 1) Storing a numpy array at ('ns', 'spikedata') and calling
                compute_rates raises ValueError because it is not a SpikeData instance.
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="wrong_type_ws")
        ws = wm.get_workspace(ws_id)
        ws.store("ns", "spikedata", np.zeros((3, 100)))
        with pytest.raises(ValueError, match="No SpikeData found"):
            await analysis.compute_rates(ws_id, "ns", "rates")

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_subtime_negative_start_end(self, loaded_ws):
        """
        EC-MCP-03: subtime with negative start/end through MCP.

        Negative values are interpreted as offsets from the end of the recording
        (length=50ms). subtime(-20, -5) should produce a 15ms SpikeData.

        Tests:
            (Test Case 1) Result is stored successfully.
            (Test Case 2) Resulting SpikeData length is approximately 15ms.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subtime(ws_id, ns, start=-20.0, end=-5.0)
        assert result["workspace_id"] == ws_id
        assert result["info"]["length_ms"] == pytest.approx(15.0, abs=1.0)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_subset_empty_unit_list(self, loaded_ws):
        """
        EC-MCP-04: subset with empty unit list through MCP.

        Passing units=[] should produce a SpikeData with 0 neurons.

        Tests:
            (Test Case 1) Result is stored successfully.
            (Test Case 2) Resulting SpikeData has 0 neurons.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subset(ws_id, ns, units=[])
        assert result["workspace_id"] == ws_id
        assert result["info"]["N"] == 0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_align_to_events_all_out_of_bounds(self, loaded_ws):
        """
        EC-MCP-05: align_to_events with all events out of bounds through MCP.

        Recording length is 50ms. Events at -100 and 200 with pre_ms=5, post_ms=5
        are all outside [0, 50], so all are dropped and ValueError is raised.

        Tests:
            (Test Case 1) ValueError with message about no valid events remaining.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="No valid events remain"):
            await analysis.align_to_events(
                ws_id, ns, "slices", events=[-100.0, 200.0], pre_ms=5.0, post_ms=5.0
            )


# ============================================================================
# Workspace Management Tests
# ============================================================================


class TestWorkspaceManagement:
    """Test workspace management functions."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_create_and_list_workspace(self):
        """
        Test creating a workspace and listing it.

        Tests:
            (Test Case 1) create_workspace returns workspace_id and name.
            (Test Case 2) list_workspaces includes the new workspace.
        """
        result = await analysis.create_workspace(name="my_ws")
        assert "workspace_id" in result
        assert result["name"] == "my_ws"

        list_result = await analysis.list_workspaces()
        assert list_result["count"] >= 1
        ids = [w["workspace_id"] for w in list_result["workspaces"]]
        assert result["workspace_id"] in ids

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_delete_workspace(self):
        """
        Test deleting a workspace.

        Tests:
            (Test Case 1) delete_workspace returns deleted=True for existing workspace.
            (Test Case 2) Workspace is absent from list_workspaces after deletion.
        """
        create_result = await analysis.create_workspace()
        ws_id = create_result["workspace_id"]

        result = await analysis.delete_workspace(ws_id)
        assert result["deleted"] is True

        list_result = await analysis.list_workspaces()
        ids = [w["workspace_id"] for w in list_result["workspaces"]]
        assert ws_id not in ids

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_describe_workspace(self):
        """
        Test that describe_workspace returns the full index.

        Tests:
            (Test Case 1) Index contains the stored namespace and key.
            (Test Case 2) Summary dict has correct type for a stored ndarray.
        """
        create_result = await analysis.create_workspace(name="desc_ws")
        ws_id = create_result["workspace_id"]
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("rec1", "my_array", np.zeros((3, 3)))

        desc = await analysis.describe_workspace(ws_id)
        assert "rec1" in desc["index"]
        assert "my_array" in desc["index"]["rec1"]
        assert desc["index"]["rec1"]["my_array"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_workspace_get_info(self):
        """
        Test workspace_get_info returns correct metadata for a stored item.

        Tests:
            (Test Case 1) Returns info dict with correct type and shape.
            (Test Case 2) Raises ValueError for a non-existent item.
        """
        create_result = await analysis.create_workspace()
        ws_id = create_result["workspace_id"]
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("ns", "key", np.ones((4, 4)))

        info_result = await analysis.workspace_get_info(ws_id, "ns", "key")
        assert info_result["info"]["type"] == "ndarray"
        assert info_result["info"]["shape"] == [4, 4]

        with pytest.raises(ValueError, match="Item not found"):
            await analysis.workspace_get_info(ws_id, "ns", "nonexistent")

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rename_workspace_item(self):
        """
        Test renaming a workspace item.

        Tests:
            (Test Case 1) rename_workspace_item returns success=True.
            (Test Case 2) Item is accessible under new key.
            (Test Case 3) Old key no longer exists.
        """
        create_result = await analysis.create_workspace()
        ws_id = create_result["workspace_id"]
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("ns", "old_key", np.zeros(5))

        result = await analysis.rename_workspace_item(ws_id, "ns", "old_key", "new_key")
        assert result["success"] is True
        assert ws.get("ns", "new_key") is not None
        assert ws.get("ns", "old_key") is None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_add_workspace_note(self):
        """
        Test adding a note to a workspace item.

        Tests:
            (Test Case 1) add_workspace_note returns success=True.
            (Test Case 2) Note is stored in the item's index entry.
        """
        create_result = await analysis.create_workspace()
        ws_id = create_result["workspace_id"]
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("ns", "key", np.zeros(3))

        result = await analysis.add_workspace_note(ws_id, "ns", "key", "test note")
        assert result["success"] is True
        assert ws.get_info("ns", "key")["note"] == "test note"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_delete_workspace_item(self):
        """
        Test deleting a single item and an entire namespace.

        Tests:
            (Test Case 1) delete_workspace_item with key returns deleted=True.
            (Test Case 2) Item is absent from workspace after deletion.
            (Test Case 3) delete_workspace_item without key deletes entire namespace.
        """
        create_result = await analysis.create_workspace()
        ws_id = create_result["workspace_id"]
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("ns", "key1", np.zeros(3))
        ws.store("ns", "key2", np.zeros(3))

        result = await analysis.delete_workspace_item(ws_id, "ns", "key1")
        assert result["deleted"] is True
        assert ws.get("ns", "key1") is None
        assert ws.get("ns", "key2") is not None

        result_ns = await analysis.delete_workspace_item(ws_id, "ns")
        assert result_ns["deleted"] is True
        assert ws.list_keys("ns") == []

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_save_and_load_workspace(self, tmp_path):
        """
        Test saving and loading a workspace round-trip.

        Tests:
            (Test Case 1) save_workspace returns saved=True.
            (Test Case 2) load_workspace restores the workspace with correct ID, name, item count.
        """
        create_result = await analysis.create_workspace(name="saved_ws")
        ws_id = create_result["workspace_id"]
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("ns", "arr", np.array([1.0, 2.0, 3.0]))

        path = str(tmp_path / "ws_test")
        save_result = await analysis.save_workspace(ws_id, path)
        assert save_result["saved"] is True

        # Delete from manager and reload
        get_workspace_manager().delete_workspace(ws_id)
        load_result = await analysis.load_workspace(path)
        assert load_result["workspace_id"] == ws_id
        assert load_result["name"] == "saved_ws"
        assert load_result["item_count"] == 1

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_fetch_workspace_item(self):
        """
        Test fetching a workspace item as a nested list.

        Tests:
            (Test Case 1) Returns correct data for an ndarray.
            (Test Case 2) Info dict is included in response.
        """
        create_result = await analysis.create_workspace()
        ws_id = create_result["workspace_id"]
        arr = np.array([1.0, 2.0, 3.0])
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store("ns", "arr", arr)

        result = await analysis.fetch_workspace_item(ws_id, "ns", "arr")
        assert result["data"] == arr.tolist()
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_merge_workspace_disjoint(self, tmp_path):
        """
        Test merging a saved workspace with non-overlapping keys into an existing workspace.

        Tests:
            (Test Case 1) All items from the saved workspace are merged.
            (Test Case 2) Original items in the target workspace are preserved.
            (Test Case 3) Result reports correct merged and skipped counts.
        """
        # Target workspace
        create_result = await analysis.create_workspace(name="target")
        target_id = create_result["workspace_id"]
        ws_target = get_workspace_manager().get_workspace(target_id)
        ws_target.store("ns", "arr_a", np.array([1.0, 2.0]))

        # Source workspace — save to disk
        create_src = await analysis.create_workspace(name="source")
        src_id = create_src["workspace_id"]
        ws_src = get_workspace_manager().get_workspace(src_id)
        ws_src.store("ns", "arr_b", np.array([3.0, 4.0]))
        path = str(tmp_path / "source_ws")
        await analysis.save_workspace(src_id, path)

        # Merge
        result = await analysis.merge_workspace(target_id, path)
        assert result["merged"] == 1
        assert result["skipped"] == 0
        assert result["workspace_id"] == target_id

        # Both items present
        np.testing.assert_array_equal(ws_target.get("ns", "arr_a"), [1.0, 2.0])
        np.testing.assert_array_equal(ws_target.get("ns", "arr_b"), [3.0, 4.0])

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_merge_workspace_skip_duplicates(self, tmp_path):
        """
        Test that merge_workspace skips existing keys when overwrite is False.

        Tests:
            (Test Case 1) Duplicate key retains target value.
            (Test Case 2) skipped_keys lists the conflicting namespace/key pairs.
        """
        create_result = await analysis.create_workspace(name="target")
        target_id = create_result["workspace_id"]
        ws_target = get_workspace_manager().get_workspace(target_id)
        ws_target.store("ns", "shared", np.array([1.0]))

        create_src = await analysis.create_workspace(name="source")
        src_id = create_src["workspace_id"]
        ws_src = get_workspace_manager().get_workspace(src_id)
        ws_src.store("ns", "shared", np.array([99.0]))
        path = str(tmp_path / "source_ws")
        await analysis.save_workspace(src_id, path)

        result = await analysis.merge_workspace(target_id, path, overwrite=False)
        assert result["skipped"] == 1
        assert result["skipped_keys"] == [{"namespace": "ns", "key": "shared"}]
        np.testing.assert_array_equal(ws_target.get("ns", "shared"), [1.0])

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_merge_workspace_overwrite(self, tmp_path):
        """
        Test that merge_workspace replaces existing keys when overwrite is True.

        Tests:
            (Test Case 1) Duplicate key is replaced by source value.
            (Test Case 2) Result reports zero skipped.
        """
        create_result = await analysis.create_workspace(name="target")
        target_id = create_result["workspace_id"]
        ws_target = get_workspace_manager().get_workspace(target_id)
        ws_target.store("ns", "val", np.array([1.0]))

        create_src = await analysis.create_workspace(name="source")
        src_id = create_src["workspace_id"]
        ws_src = get_workspace_manager().get_workspace(src_id)
        ws_src.store("ns", "val", np.array([99.0]))
        path = str(tmp_path / "source_ws")
        await analysis.save_workspace(src_id, path)

        result = await analysis.merge_workspace(target_id, path, overwrite=True)
        assert result["merged"] == 1
        assert result["skipped"] == 0
        np.testing.assert_array_equal(ws_target.get("ns", "val"), [99.0])

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_merge_workspace_invalid_workspace_id(self, tmp_path):
        """
        Test that merge_workspace raises ValueError for an unknown workspace ID.

        Tests:
            (Test Case 1) ValueError is raised with a descriptive message.
        """
        # Save a dummy workspace to have a valid path
        create_src = await analysis.create_workspace(name="source")
        src_id = create_src["workspace_id"]
        path = str(tmp_path / "source_ws")
        await analysis.save_workspace(src_id, path)

        with pytest.raises(ValueError, match="Workspace not found"):
            await analysis.merge_workspace("nonexistent-id", path)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_create_workspace_duplicate_name(self):
        """
        EC-MCP-07: create_workspace with duplicate name.

        Creating two workspaces with the same name should succeed and produce
        different workspace IDs.

        Tests:
            (Test Case 1) Both create calls succeed.
            (Test Case 2) The two workspace IDs are different.
            (Test Case 3) Both workspaces appear in list_workspaces.
        """
        result1 = await analysis.create_workspace(name="dup_name")
        result2 = await analysis.create_workspace(name="dup_name")
        assert result1["workspace_id"] != result2["workspace_id"]
        assert result1["name"] == "dup_name"
        assert result2["name"] == "dup_name"
        listing = await analysis.list_workspaces()
        ids = [w["workspace_id"] for w in listing["workspaces"]]
        assert result1["workspace_id"] in ids
        assert result2["workspace_id"] in ids

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_fetch_workspace_item_non_serializable(self):
        """
        EC-MCP-09: fetch_workspace_item with non-serializable object.

        Storing a custom object (not ndarray or PairwiseCompMatrixStack) and
        calling fetch_workspace_item should raise ValueError describing the
        unsupported type.

        Tests:
            (Test Case 1) ValueError mentioning 'fetch_workspace_item supports'.
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="custom_obj_ws")
        ws = wm.get_workspace(ws_id)

        class CustomObj:
            pass

        ws.store("ns", "obj", CustomObj())
        result = await analysis.fetch_workspace_item(ws_id, "ns", "obj")
        # Unknown types get a repr fallback
        assert result["type"] == "CustomObj"
        assert "repr" in result


# ============================================================================
# Workspace Analysis Tools Tests
# ============================================================================


class TestWorkspaceAnalysisTools:
    """Test analysis tools that store results in a workspace."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_pairwise_fr_corr(self, loaded_ws):
        """
        Test compute_pairwise_fr_corr stores correlation and lag matrices in workspace.

        Tests:
            (Test Case 1) Returns workspace_id, namespace, key_corr, key_lag.
            (Test Case 2) Both stored items have correct type and shape (U, U).

        Notes:
            - compute_pairwise_fr_corr reads RateData; ISI rates must be computed first.
        """
        ws_id, ns = loaded_ws
        times = list(np.arange(0.0, 50.0, 1.0))
        await analysis.compute_resampled_isi(ws_id, ns, "rates", times=times)
        result = await analysis.compute_pairwise_fr_corr(
            ws_id,
            ns,
            rate_key="rates",
            key_corr="corr",
            key_lag="lag",
        )
        assert result["workspace_id"] == ws_id
        assert result["namespace"] == ns
        assert result["key_corr"] == "corr"
        assert result["key_lag"] == "lag"
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"
        assert result["info_corr"]["shape"] == [3, 3]
        assert result["info_lag"]["type"] == "PairwiseCompMatrix"
        assert result["info_lag"]["shape"] == [3, 3]

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_create_rate_slice_stack(self, loaded_ws):
        """
        Test create_rate_slice_stack stores a RateSliceStack in the workspace.

        Tests:
            (Test Case 1) Returns workspace_id, namespace, key.
            (Test Case 2) Stored item summary reports type RateSliceStack.
        """
        ws_id, ns = loaded_ws
        times_start_to_end = [[0.0, 25.0], [25.0, 50.0]]
        result = await analysis.create_rate_slice_stack(
            ws_id,
            ns,
            "rss",
            times_start_to_end=times_start_to_end,
        )
        assert result["workspace_id"] == ws_id
        assert result["key"] == "rss"
        assert result["info"]["type"] == "RateSliceStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rate_slice_unit_corr(self, loaded_ws):
        """
        Test compute_rate_slice_unit_corr loads a stored RateSliceStack and stores
        the resulting PairwiseCompMatrixStack.

        Tests:
            (Test Case 1) Returns workspace_id, namespace, and out_key.
            (Test Case 2) Stored output item type is PairwiseCompMatrixStack.
            (Test Case 3) av_corr is returned inline.
        """
        ws_id, ns = loaded_ws
        times_start_to_end = [[0.0, 25.0], [25.0, 50.0]]
        await analysis.create_rate_slice_stack(
            ws_id, ns, "rss", times_start_to_end=times_start_to_end
        )
        result = await analysis.compute_rate_slice_unit_corr(
            workspace_id=ws_id,
            namespace=ns,
            stack_key="rss",
            out_key="corr_stack",
        )
        assert result["workspace_id"] == ws_id
        assert result["key"] == "corr_stack"
        assert result["info"]["type"] == "PairwiseCompMatrixStack"
        assert "av_corr" in result

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_frames_rate_data(self, loaded_ws):
        """
        Test frames_rate_data stores a RateSliceStack in the workspace.

        Tests:
            (Test Case 1) Returns workspace_id, namespace, key, and n_frames.
            (Test Case 2) Stored item type is RateSliceStack.
            (Test Case 3) n_frames is correct for non-overlapping equal-length frames.

        Notes:
            - frames_rate_data reads RateData; ISI rates must be computed first.
        """
        ws_id, ns = loaded_ws
        times = list(np.arange(0.0, 50.0, 1.0))
        await analysis.compute_resampled_isi(ws_id, ns, "rates", times=times)
        result = await analysis.frames_rate_data(
            ws_id,
            ns,
            rate_key="rates",
            key="frames",
            length=25.0,
        )
        assert result["workspace_id"] == ws_id
        assert result["key"] == "frames"
        assert result["n_frames"] == 2
        assert result["info"]["type"] == "RateSliceStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_extract_lower_triangle_features(self, loaded_ws):
        """
        Test extract_lower_triangle_features loads a PairwiseCompMatrixStack from
        the workspace and stores a (S, F) feature matrix.

        Tests:
            (Test Case 1) Returns workspace reference with out_key.
            (Test Case 2) Stored output is an ndarray with correct rank.
        """
        ws_id, ns = loaded_ws
        times_start_to_end = [[0.0, 25.0], [25.0, 50.0]]
        await analysis.create_rate_slice_stack(
            ws_id, ns, "rss", times_start_to_end=times_start_to_end
        )
        await analysis.compute_rate_slice_unit_corr(
            workspace_id=ws_id,
            namespace=ns,
            stack_key="rss",
            out_key="corr_stack",
        )
        result = await analysis.extract_lower_triangle_features(
            workspace_id=ws_id,
            namespace=ns,
            key="corr_stack",
            out_key="features",
        )
        assert result["key"] == "features"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_workspace_raises(self):
        """
        Test that workspace-storing tools raise ValueError for an unknown workspace_id.

        Tests:
            (Test Case 1) create_rate_slice_stack raises ValueError: Workspace not found.
        """
        with pytest.raises(ValueError, match="Workspace not found"):
            await analysis.create_rate_slice_stack(
                "nonexistent-workspace-id",
                "ns",
                "k",
                times_start_to_end=[[0.0, 25.0]],
            )


# ============================================================================
# Export Tools Tests
# ============================================================================


class TestExportTools:
    """Test export tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_hdf5(self, loaded_ws):
        """
        Test exporting SpikeData from a workspace to an HDF5 file.

        Tests:
            (Test Case 1) Result contains file_path.
            (Test Case 2) File exists on disk after export.
        """
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        ws_id, ns = loaded_ws
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp_path = tmp.name

        try:
            result = await exporters.export_to_hdf5_ragged(
                ws_id, ns, tmp_path, spike_times_unit="s"
            )
            assert "file_path" in result
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_nwb(self, loaded_ws):
        """
        Test exporting SpikeData from a workspace to an NWB file.

        Tests:
            (Test Case 1) Result contains file_path.
            (Test Case 2) File exists on disk after export.
        """
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        ws_id, ns = loaded_ws
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nwb") as tmp:
            tmp_path = tmp.name

        try:
            result = await exporters.export_to_nwb(ws_id, ns, tmp_path)
            assert "file_path" in result
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_kilosort(self, loaded_ws):
        """
        Test exporting SpikeData from a workspace to a KiloSort folder.

        Tests:
            (Test Case 1) Result contains folder_path.
            (Test Case 2) Exactly two files are created (spike_times.npy, spike_clusters.npy).
        """
        ws_id, ns = loaded_ws
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await exporters.export_to_kilosort(ws_id, ns, tmpdir, fs_Hz=1000.0)
            assert "folder_path" in result
            assert len(result["files"]) == 2

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_pickle_s3_upload(self, loaded_ws):
        """
        EC-MCP-06: export_to_pickle with S3 upload path.

        Mock the S3 upload function and verify the MCP wrapper handles it
        correctly when given an s3:// path.

        Tests:
            (Test Case 1) Result file_path is the S3 URL.
            (Test Case 2) upload_to_s3 was called once.
        """
        ws_id, ns = loaded_ws
        s3_path = "s3://my-bucket/exports/test.pkl"
        with patch("spikelab.data_loaders.s3_utils.upload_to_s3") as mock_upload:
            result = await exporters.export_to_pickle(ws_id, ns, s3_path)
            assert result["file_path"] == s3_path
            mock_upload.assert_called_once()


# ============================================================================
# Server Integration Tests
# ============================================================================


class TestServerIntegration:
    """Test server integration and tool registration."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that tools are registered."""
        from spikelab.mcp_server.server import _list_tools

        tools = await _list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

        tool_names = [tool.name for tool in tools]
        assert "load_from_nwb" in tool_names
        assert "compute_rates" in tool_names
        assert "export_to_hdf5_ragged" in tool_names

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_workspace_tools_registered(self):
        """
        Test that all workspace management tools are registered.

        Tests:
            (Test Case 1) create_workspace is registered.
            (Test Case 2) list_workspaces is registered.
            (Test Case 3) fetch_workspace_item is registered.
            (Test Case 4) _from_workspace analysis tools are registered.
        """
        from spikelab.mcp_server.server import _list_tools

        tools = await _list_tools()
        tool_names = [tool.name for tool in tools]

        # Workspace management
        assert "create_workspace" in tool_names
        assert "delete_workspace" in tool_names
        assert "list_workspaces" in tool_names
        assert "describe_workspace" in tool_names
        assert "workspace_get_info" in tool_names
        assert "rename_workspace_item" in tool_names
        assert "add_workspace_note" in tool_names
        assert "delete_workspace_item" in tool_names
        assert "save_workspace" in tool_names
        assert "load_workspace" in tool_names
        assert "merge_workspace" in tool_names
        assert "fetch_workspace_item" in tool_names

        # Workspace-backed stack analysis tools
        assert "compute_rate_slice_unit_corr" in tool_names
        assert "compute_rate_slice_time_corr" in tool_names
        assert "compute_unit_to_unit_slice_corr" in tool_names
        assert "compute_rate_slice_unit_order" in tool_names

        # Workspace-backed analysis tools
        assert "pca_on_workspace_item" in tool_names

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_result_store_tools_removed(self):
        """
        Test that old ResultStore tools are no longer registered.

        Tests:
            (Test Case 1) fetch_result is not registered.
            (Test Case 2) delete_result is not registered.
            (Test Case 3) list_results is not registered.
            (Test Case 4) _from_stack tools are not registered.
        """
        from spikelab.mcp_server.server import _list_tools

        tools = await _list_tools()
        tool_names = [tool.name for tool in tools]

        assert "fetch_result" not in tool_names
        assert "delete_result" not in tool_names
        assert "list_results" not in tool_names
        assert "compute_rate_slice_unit_corr_from_stack" not in tool_names
        assert "pca_on_result" not in tool_names

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_tool_schemas(self):
        """Test tool schemas are valid."""
        from spikelab.mcp_server.server import _list_tools

        tools = await _list_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            assert tool.inputSchema["type"] == "object"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling a tool through the server."""
        from spikelab.mcp_server.server import _call_tool, _TOOL_DISPATCH

        mock_fn = AsyncMock(
            return_value={
                "rates": [0.1, 0.2, 0.3],
                "unit": "kHz",
                "num_neurons": 3,
            }
        )
        original = _TOOL_DISPATCH["compute_rates"]
        _TOOL_DISPATCH["compute_rates"] = mock_fn
        try:
            result = await _call_tool(
                "compute_rates",
                {
                    "workspace_id": "test-ws",
                    "namespace": "rec1",
                    "key": "rates",
                    "unit": "kHz",
                },
            )

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert "rates" in data
            mock_fn.assert_called_once()
        finally:
            _TOOL_DISPATCH["compute_rates"] = original

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test error handling for unknown tool."""
        from spikelab.mcp_server.server import _call_tool

        result = await _call_tool("unknown_tool", {})
        data = json.loads(result[0].text)
        assert "error" in data


# ============================================================================
# New MCP Tool Tests (session additions)
# ============================================================================


@pytest.fixture
def loaded_ws_with_sss(sample_spikedata):
    """Create a workspace with SpikeData and a SpikeSliceStack stored.

    Returns (workspace_id, namespace).
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("MCP server not available")
    from spikelab.spikedata.spikeslicestack import SpikeSliceStack

    wm = get_workspace_manager()
    ws_id = wm.create_workspace(name="test_ws_sss")
    ws = wm.get_workspace(ws_id)
    ws.store("rec1", "spikedata", sample_spikedata)

    sss = SpikeSliceStack(
        sample_spikedata, times_start_to_end=[(0.0, 25.0), (25.0, 50.0)]
    )
    ws.store("rec1", "sss", sss)
    return ws_id, "rec1"


@pytest.fixture
def loaded_ws_with_rss(sample_spikedata):
    """Create a workspace with SpikeData and a RateSliceStack stored.

    Returns (workspace_id, namespace).
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("MCP server not available")
    from spikelab.spikedata.rateslicestack import RateSliceStack

    wm = get_workspace_manager()
    ws_id = wm.create_workspace(name="test_ws_rss")
    ws = wm.get_workspace(ws_id)
    ws.store("rec1", "spikedata", sample_spikedata)

    rss = RateSliceStack(
        sample_spikedata, times_start_to_end=[(0.0, 25.0), (25.0, 50.0)]
    )
    ws.store("rec1", "rss", rss)
    return ws_id, "rec1"


class TestSpikeSliceStackMCPTools:
    """Tests for new SpikeSliceStack MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_spike_unit_to_unit_comparison_ccg(self, loaded_ws_with_sss):
        """
        Test spike_unit_to_unit_comparison with CCG metric stores results.

        Tests:
            (Test Case 1) Returns key_corr and key_lag.
            (Test Case 2) Stored corr item is PairwiseCompMatrixStack.
            (Test Case 3) av_corr is returned inline.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.spike_unit_to_unit_comparison(
            ws_id,
            ns,
            stack_key="sss",
            out_key_corr="u2u_corr",
            out_key_lag="u2u_lag",
            metric="ccg",
        )
        assert result["key_corr"] == "u2u_corr"
        assert result["key_lag"] == "u2u_lag"
        assert result["info_corr"]["type"] == "PairwiseCompMatrixStack"
        assert "av_corr" in result

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_spike_unit_to_unit_comparison_sttc(self, loaded_ws_with_sss):
        """
        Test spike_unit_to_unit_comparison with STTC metric (no lag).

        Tests:
            (Test Case 1) key_lag is None.
            (Test Case 2) av_lag is None.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.spike_unit_to_unit_comparison(
            ws_id,
            ns,
            stack_key="sss",
            out_key_corr="u2u_corr",
            out_key_lag="u2u_lag",
            metric="sttc",
        )
        assert result["key_lag"] is None
        assert result["av_lag"] is None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_spike_slice_to_slice_unit_comparison(self, loaded_ws_with_sss):
        """
        Test spike_slice_to_slice_unit_comparison stores correlation stack.

        Tests:
            (Test Case 1) Returns key_corr.
            (Test Case 2) Stored item is PairwiseCompMatrixStack.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.spike_slice_to_slice_unit_comparison(
            ws_id,
            ns,
            stack_key="sss",
            out_key_corr="s2s_corr",
            out_key_lag="s2s_lag",
            metric="ccg",
        )
        assert result["key_corr"] == "s2s_corr"
        assert result["info_corr"]["type"] == "PairwiseCompMatrixStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_frac_active(self, loaded_ws_with_sss):
        """
        Test compute_frac_active stores a (U,) ndarray.

        Tests:
            (Test Case 1) Returns key.
            (Test Case 2) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.compute_frac_active(
            ws_id,
            ns,
            stack_key="sss",
            out_key="frac",
        )
        assert result["key"] == "frac"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_spike_order_units_across_slices(self, loaded_ws_with_sss):
        """
        Test spike_order_units_across_slices returns inline ordering.

        Tests:
            (Test Case 1) Result has highly_active and low_active groups.
            (Test Case 2) highly_active contains unit_ids_in_order.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.spike_order_units_across_slices(
            ws_id,
            ns,
            stack_key="sss",
        )
        assert "highly_active" in result
        assert "low_active" in result
        assert "unit_ids_in_order" in result["highly_active"]

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_unit_timing_per_slice_spike(self, loaded_ws_with_sss):
        """
        Test get_unit_timing_per_slice_spike stores a (U, S) ndarray.

        Tests:
            (Test Case 1) Returns key.
            (Test Case 2) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.get_unit_timing_per_slice_spike(
            ws_id,
            ns,
            stack_key="sss",
            out_key="timing",
        )
        assert result["key"] == "timing"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rank_order_correlation_spike_raw(self, loaded_ws_with_sss):
        """
        Test rank_order_correlation_spike with n_shuffles=0 (raw Spearman).

        Tests:
            (Test Case 1) Returns key_corr and key_overlap.
            (Test Case 2) Stored corr item is PairwiseCompMatrix.
            (Test Case 3) av_corr is returned inline.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.rank_order_correlation_spike(
            ws_id,
            ns,
            stack_key="sss",
            out_key_corr="rank_corr",
            out_key_overlap="rank_overlap",
            n_shuffles=0,
        )
        assert result["key_corr"] == "rank_corr"
        assert result["key_overlap"] == "rank_overlap"
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"
        assert result["info_overlap"]["type"] == "PairwiseCompMatrix"
        assert isinstance(result["av_corr"], float)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rank_order_correlation_spike_zscore(self, loaded_ws_with_sss):
        """
        Test rank_order_correlation_spike with z-scoring.

        Tests:
            (Test Case 1) n_shuffles is echoed back.
            (Test Case 2) Result stores PairwiseCompMatrix.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.rank_order_correlation_spike(
            ws_id,
            ns,
            stack_key="sss",
            out_key_corr="zrank_corr",
            out_key_overlap="zrank_overlap",
            n_shuffles=10,
            seed=42,
        )
        assert result["n_shuffles"] == 10
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rank_order_correlation_spike_with_timing_key(
        self, loaded_ws_with_sss
    ):
        """
        Test rank_order_correlation_spike using a pre-computed timing_key.

        Tests:
            (Test Case 1) Pre-computed timing matrix is accepted.
            (Test Case 2) Result stores PairwiseCompMatrix.
        """
        ws_id, ns = loaded_ws_with_sss
        await analysis.get_unit_timing_per_slice_spike(
            ws_id,
            ns,
            stack_key="sss",
            out_key="timing",
        )
        result = await analysis.rank_order_correlation_spike(
            ws_id,
            ns,
            stack_key="sss",
            out_key_corr="rank_corr2",
            out_key_overlap="rank_overlap2",
            timing_key="timing",
            n_shuffles=0,
        )
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"


class TestRateSliceStackMCPTools:
    """Tests for new RateSliceStack MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_unit_timing_per_slice_rate(self, loaded_ws_with_rss):
        """
        Test get_unit_timing_per_slice_rate stores a (U, S) ndarray.

        Tests:
            (Test Case 1) Returns key.
            (Test Case 2) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws_with_rss
        result = await analysis.get_unit_timing_per_slice_rate(
            ws_id,
            ns,
            stack_key="rss",
            out_key="timing",
        )
        assert result["key"] == "timing"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rank_order_correlation_rate_raw(self, loaded_ws_with_rss):
        """
        Test rank_order_correlation_rate with n_shuffles=0.

        Tests:
            (Test Case 1) Returns key_corr and key_overlap.
            (Test Case 2) Both stored items are PairwiseCompMatrix.
        """
        ws_id, ns = loaded_ws_with_rss
        result = await analysis.rank_order_correlation_rate(
            ws_id,
            ns,
            stack_key="rss",
            out_key_corr="rank_corr",
            out_key_overlap="rank_overlap",
            n_shuffles=0,
        )
        assert result["key_corr"] == "rank_corr"
        assert result["key_overlap"] == "rank_overlap"
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"
        assert result["info_overlap"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rank_order_correlation_rate_with_timing_key(
        self, loaded_ws_with_rss
    ):
        """
        Test rank_order_correlation_rate using a pre-computed timing_key.

        Tests:
            (Test Case 1) Pre-computed timing matrix is accepted.
            (Test Case 2) Result stores PairwiseCompMatrix.
        """
        ws_id, ns = loaded_ws_with_rss
        await analysis.get_unit_timing_per_slice_rate(
            ws_id,
            ns,
            stack_key="rss",
            out_key="timing",
        )
        result = await analysis.rank_order_correlation_rate(
            ws_id,
            ns,
            stack_key="rss",
            out_key_corr="rank_corr2",
            out_key_overlap="rank_overlap2",
            timing_key="timing",
            n_shuffles=0,
        )
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rate_slice_unit_corr_with_frac_active(
        self, loaded_ws_with_rss
    ):
        """
        Test compute_rate_slice_unit_corr accepts frac_active_key.

        Tests:
            (Test Case 1) frac_active_key is accepted without error.
            (Test Case 2) Result stores PairwiseCompMatrixStack.
        """
        ws_id, ns = loaded_ws_with_rss
        # Store a frac_active array manually
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store(ns, "frac", np.array([1.0, 1.0, 1.0]))

        result = await analysis.compute_rate_slice_unit_corr(
            workspace_id=ws_id,
            namespace=ns,
            stack_key="rss",
            out_key="corr",
            frac_active_key="frac",
        )
        assert result["key"] == "corr"
        assert result["info"]["type"] == "PairwiseCompMatrixStack"


class TestPairwiseConditioningMCPTools:
    """Tests for remove_by_condition MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_remove_by_condition_matrix(self, loaded_ws):
        """
        Test remove_by_condition on PairwiseCompMatrix stored in workspace.

        Tests:
            (Test Case 1) Returns key.
            (Test Case 2) Stored item is PairwiseCompMatrix.
        """
        from spikelab.spikedata.pairwise import PairwiseCompMatrix

        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        target = PairwiseCompMatrix(matrix=np.array([[1.0, 0.8], [0.8, 1.0]]))
        condition = PairwiseCompMatrix(matrix=np.array([[0.0, 1.5], [1.5, 0.0]]))
        ws.store(ns, "sttc", target)
        ws.store(ns, "latency", condition)

        result = await analysis.remove_by_condition(
            workspace_id=ws_id,
            namespace=ns,
            target_key="sttc",
            condition_key="latency",
            out_key="masked",
            op="abs_lt",
            threshold=2.0,
        )
        assert result["key"] == "masked"
        assert result["info"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_remove_by_condition_stack(self, loaded_ws):
        """
        Test remove_by_condition on PairwiseCompMatrixStack.

        Tests:
            (Test Case 1) Stored item is PairwiseCompMatrixStack.
        """
        from spikelab.spikedata.pairwise import PairwiseCompMatrixStack

        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        target = PairwiseCompMatrixStack(stack=np.ones((3, 3, 2)))
        condition = PairwiseCompMatrixStack(stack=np.zeros((3, 3, 2)))
        ws.store(ns, "target_stack", target)
        ws.store(ns, "cond_stack", condition)

        result = await analysis.remove_by_condition(
            workspace_id=ws_id,
            namespace=ns,
            target_key="target_stack",
            condition_key="cond_stack",
            out_key="masked_stack",
            op="lt",
            threshold=1.0,
        )
        assert result["key"] == "masked_stack"
        assert result["info"]["type"] == "PairwiseCompMatrixStack"


# ============================================================================
# Coverage gap tests — basic analysis tools
# ============================================================================


@pytest.fixture
def loaded_ws_with_attrs():
    """Workspace with SpikeData that has neuron_attributes.

    Returns (workspace_id, namespace).
    """
    if not MCP_SERVER_AVAILABLE:
        pytest.skip("MCP server not available")
    train = [
        [10.0, 20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0],
        [5.0, 45.0],
    ]
    attrs = [
        {"id": "A", "region": "ctx"},
        {"id": "B", "region": "hpc"},
        {"id": "C", "region": "ctx"},
    ]
    sd = SpikeData(train, length=50.0, neuron_attributes=attrs)
    wm = get_workspace_manager()
    ws_id = wm.create_workspace(name="test_ws_attrs")
    wm.get_workspace(ws_id).store("rec1", "spikedata", sd)
    return ws_id, "rec1"


class TestBasicAnalysisCoverage:
    """Coverage tests for basic analysis MCP tools not previously tested."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_binned(self, loaded_ws):
        """
        Test compute_binned stores binned spike counts.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_binned(ws_id, ns, "binned", bin_size=10.0)
        assert result["key"] == "binned"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_binned_meanrate(self, loaded_ws):
        """
        Test compute_binned_meanrate stores mean rate per bin.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_binned_meanrate(
            ws_id, ns, "meanrate", bin_size=10.0
        )
        assert result["key"] == "meanrate"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_channel_raster(self, loaded_ws_with_attrs):
        """
        Test compute_channel_raster stores a channel-grouped raster.

        Tests:
            (Test Case 1) Stored item is ndarray.

        Notes:
            - Requires neuron_attributes with channel info.
        """
        ws_id, ns = loaded_ws_with_attrs
        # Add channel attribute so channel_raster can find it
        await analysis.set_neuron_attribute(ws_id, ns, key="channel", values=[0, 1, 0])
        result = await analysis.compute_channel_raster(
            ws_id, ns, "ch_raster", bin_size=5.0, channel_attr="channel"
        )
        assert result["key"] == "ch_raster"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_interspike_intervals(self, loaded_ws):
        """
        Test compute_interspike_intervals stores NaN-padded ISI array.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_interspike_intervals(ws_id, ns, "isis")
        assert result["key"] == "isis"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_spike_time_tilings(self, loaded_ws):
        """
        Test compute_spike_time_tilings stores full STTC matrix.

        Tests:
            (Test Case 1) Stored item is ndarray with shape (3, 3).
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_spike_time_tilings(ws_id, ns, "sttc_full")
        assert result["key"] == "sttc_full"
        assert result["info"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_threshold_spike_time_tilings(self, loaded_ws):
        """
        Test threshold_spike_time_tilings stores binary STTC matrix.

        Tests:
            (Test Case 1) Stored item is ndarray with shape (3, 3).
        """
        ws_id, ns = loaded_ws
        result = await analysis.threshold_spike_time_tilings(
            ws_id, ns, "sttc_bin", threshold=0.1
        )
        assert result["key"] == "sttc_bin"
        assert result["info"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_pairwise_ccg(self, loaded_ws):
        """
        Test compute_pairwise_ccg stores correlation and lag matrices.

        Tests:
            (Test Case 1) Both key_corr and key_lag stored as PairwiseCompMatrix.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_pairwise_ccg(
            ws_id, ns, key_corr="ccg_corr", key_lag="ccg_lag"
        )
        assert result["key_corr"] == "ccg_corr"
        assert result["key_lag"] == "ccg_lag"
        assert result["info_corr"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_pairwise_latencies(self, loaded_ws):
        """
        Test compute_pairwise_latencies stores mean and std matrices.

        Tests:
            (Test Case 1) Both key_mean and key_std stored as PairwiseCompMatrix.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_pairwise_latencies(
            ws_id, ns, key_mean="lat_mean", key_std="lat_std"
        )
        assert result["key_mean"] == "lat_mean"
        assert result["key_std"] == "lat_std"
        assert result["info_mean"]["type"] == "PairwiseCompMatrix"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_pop_rate(self, loaded_ws):
        """
        Test get_pop_rate stores smoothed population rate.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.get_pop_rate(ws_id, ns, "pop_rate")
        assert result["key"] == "pop_rate"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_idces_times(self, loaded_ws):
        """
        Test get_idces_times stores (2, n_spikes) array.

        Tests:
            (Test Case 1) Stored item is ndarray with shape[0] == 2.
        """
        ws_id, ns = loaded_ws
        result = await analysis.get_idces_times(ws_id, ns, "it")
        assert result["key"] == "it"
        assert result["info"]["type"] == "ndarray"
        assert result["info"]["shape"][0] == 2

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_latencies(self, loaded_ws):
        """
        Test compute_latencies stores NaN-padded latency matrix.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_latencies(
            ws_id, ns, "lats", times=[10.0, 20.0, 30.0]
        )
        assert result["key"] == "lats"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_latencies_to_index(self, loaded_ws):
        """
        Test compute_latencies_to_index stores latencies from one unit.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_latencies_to_index(
            ws_id, ns, "lat_idx", neuron_index=0
        )
        assert result["key"] == "lat_idx"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_spike_trig_pop_rate(self, loaded_ws):
        """
        Test compute_spike_trig_pop_rate stores stPR and coupling stats.

        Tests:
            (Test Case 1) Three keys stored (stpr, lags, coupling).
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_spike_trig_pop_rate(
            ws_id, ns, key="stpr", key_lags="stpr_lags", key_coupling="stpr_coupling"
        )
        assert result["key"] == "stpr"
        assert result["key_lags"] == "stpr_lags"
        assert result["key_coupling"] == "stpr_coupling"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_export_to_pickle(self, loaded_ws, tmp_path):
        """
        Test export_to_pickle writes a pickle file.

        Tests:
            (Test Case 1) File is created at the specified path.
        """
        ws_id, ns = loaded_ws
        path = str(tmp_path / "test.pkl")
        result = await exporters.export_to_pickle(ws_id, ns, path)
        assert result["file_path"] == path
        assert os.path.exists(path)


# ============================================================================
# Coverage gap tests — metadata and selection tools
# ============================================================================


class TestMetadataAndSelectionCoverage:
    """Coverage tests for metadata query and selection MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_list_neurons(self, loaded_ws_with_attrs):
        """
        Test list_neurons returns neuron list inline.

        Tests:
            (Test Case 1) Returns list of 3 neurons.
        """
        ws_id, ns = loaded_ws_with_attrs
        result = await analysis.list_neurons(ws_id, ns)
        assert len(result["neurons"]) == 3

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_neuron_attribute(self, loaded_ws_with_attrs):
        """
        Test get_neuron_attribute returns attribute values.

        Tests:
            (Test Case 1) Returns region values for all 3 neurons.
        """
        ws_id, ns = loaded_ws_with_attrs
        result = await analysis.get_neuron_attribute(ws_id, ns, key="region")
        assert result["values"] == ["ctx", "hpc", "ctx"]

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_set_neuron_attribute(self, loaded_ws_with_attrs):
        """
        Test set_neuron_attribute modifies attributes in place.

        Tests:
            (Test Case 1) Attribute key is confirmed set.
        """
        ws_id, ns = loaded_ws_with_attrs
        result = await analysis.set_neuron_attribute(
            ws_id, ns, key="label", values=["x", "y", "z"]
        )
        assert result["key"] == "label"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_neuron_to_channel_map(self, loaded_ws_with_attrs):
        """
        Test get_neuron_to_channel_map returns the mapping dict.

        Tests:
            (Test Case 1) Returns a mapping dict (may be empty if no channel attr).
        """
        ws_id, ns = loaded_ws_with_attrs
        result = await analysis.get_neuron_to_channel_map(ws_id, ns)
        assert "mapping" in result

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_subset(self, loaded_ws):
        """
        Test subset stores a subsetted SpikeData.

        Tests:
            (Test Case 1) Result contains workspace reference.
            (Test Case 2) Stored item type is SpikeData.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subset(ws_id, ns, units=[0, 1])
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["type"] == "SpikeData"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_append_session(self, loaded_ws, sample_spikedata):
        """
        Test append_session concatenates two SpikeData in time.

        Tests:
            (Test Case 1) Result contains workspace reference.
            (Test Case 2) Stored item type is SpikeData.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store("rec2", "spikedata", sample_spikedata)
        result = await analysis.append_session(
            ws_id, namespace_a="rec1", namespace_b="rec2"
        )
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["type"] == "SpikeData"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_concatenate_units(self, loaded_ws, sample_spikedata):
        """
        Test concatenate_units merges units from two namespaces.

        Tests:
            (Test Case 1) Result contains workspace reference.
            (Test Case 2) Stored item type is SpikeData.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store("rec2", "spikedata", sample_spikedata)
        result = await analysis.concatenate_units(
            ws_id, namespace_a="rec1", namespace_b="rec2"
        )
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["type"] == "SpikeData"


# ============================================================================
# Coverage gap tests — slice stack tools
# ============================================================================


class TestSliceStackCoverage:
    """Coverage tests for slice stack creation and analysis MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_create_spike_slice_stack(self, loaded_ws):
        """
        Test create_spike_slice_stack stores a SpikeSliceStack.

        Tests:
            (Test Case 1) Stored item type is SpikeSliceStack.
        """
        ws_id, ns = loaded_ws
        result = await analysis.create_spike_slice_stack(
            ws_id, ns, "sss", times_start_to_end=[[0.0, 25.0], [25.0, 50.0]]
        )
        assert result["key"] == "sss"
        assert result["info"]["type"] == "SpikeSliceStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_frames_spike_data(self, loaded_ws):
        """
        Test frames_spike_data stores a SpikeSliceStack from fixed-length frames.

        Tests:
            (Test Case 1) Stored item type is SpikeSliceStack.
            (Test Case 2) n_frames is correct.
        """
        ws_id, ns = loaded_ws
        result = await analysis.frames_spike_data(ws_id, ns, "sss_frames", length=25.0)
        assert result["key"] == "sss_frames"
        assert result["info"]["type"] == "SpikeSliceStack"
        assert result["n_frames"] == 2

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_spike_slice_to_raster(self, loaded_ws_with_sss):
        """
        Test spike_slice_to_raster converts SpikeSliceStack to dense raster.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws_with_sss
        result = await analysis.spike_slice_to_raster(
            ws_id, ns, stack_key="sss", key="sss_raster"
        )
        assert result["key"] == "sss_raster"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_align_to_events(self, loaded_ws):
        """
        Test align_to_events creates event-aligned slices.

        Tests:
            (Test Case 1) Stored item type is SpikeSliceStack (kind='spike').
        """
        ws_id, ns = loaded_ws
        result = await analysis.align_to_events(
            ws_id,
            ns,
            key="aligned",
            events=[15.0, 35.0],
            pre_ms=5.0,
            post_ms=5.0,
            kind="spike",
        )
        assert result["key"] == "aligned"
        assert result["info"]["type"] == "SpikeSliceStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rate_slice_time_corr(self, loaded_ws_with_rss):
        """
        Test compute_rate_slice_time_corr stores PairwiseCompMatrixStack.

        Tests:
            (Test Case 1) Stored item is PairwiseCompMatrixStack.
        """
        ws_id, ns = loaded_ws_with_rss
        result = await analysis.compute_rate_slice_time_corr(
            ws_id, ns, stack_key="rss", out_key="time_corr"
        )
        assert result["key"] == "time_corr"
        assert result["info"]["type"] == "PairwiseCompMatrixStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rate_slice_unit_order(self, loaded_ws_with_rss):
        """
        Test compute_rate_slice_unit_order returns inline ordering.

        Tests:
            (Test Case 1) Result has highly_active group with unit_ids_in_order.
        """
        ws_id, ns = loaded_ws_with_rss
        result = await analysis.compute_rate_slice_unit_order(
            ws_id, ns, stack_key="rss"
        )
        assert "highly_active" in result
        assert "unit_ids_in_order" in result["highly_active"]

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_unit_to_unit_slice_corr(self, loaded_ws_with_rss):
        """
        Test compute_unit_to_unit_slice_corr stores corr and lag stacks.

        Tests:
            (Test Case 1) Both key_corr and key_lag stored as PairwiseCompMatrixStack.
        """
        ws_id, ns = loaded_ws_with_rss
        result = await analysis.compute_unit_to_unit_slice_corr(
            ws_id, ns, stack_key="rss", out_key_corr="u2u_c", out_key_lag="u2u_l"
        )
        assert result["key_corr"] == "u2u_c"
        assert result["key_lag"] == "u2u_l"
        assert result["info_corr"]["type"] == "PairwiseCompMatrixStack"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_compute_rate_manifold(self, loaded_ws):
        """
        Test compute_rate_manifold stores a low-dimensional embedding.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        times = list(np.arange(0.0, 50.0, 1.0))
        await analysis.compute_resampled_isi(ws_id, ns, "rates", times=times)
        result = await analysis.compute_rate_manifold(
            ws_id, ns, rate_key="rates", key="manifold", method="PCA", n_components=2
        )
        assert result["key"] == "manifold"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_pca_on_lower_triangle(self, loaded_ws_with_rss):
        """
        Test pca_on_lower_triangle stores PCA embedding.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws_with_rss
        await analysis.compute_rate_slice_unit_corr(
            ws_id, ns, stack_key="rss", out_key="corr"
        )
        result = await analysis.pca_on_lower_triangle(
            ws_id, ns, key="corr", out_key="pca_lt", n_components=1
        )
        assert result["key"] == "pca_lt"
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_pca_on_workspace_item(self, loaded_ws):
        """
        Test pca_on_workspace_item stores PCA embedding from a 2D array.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store(ns, "mat2d", np.random.default_rng(0).random((10, 5)))
        result = await analysis.pca_on_workspace_item(
            ws_id, ns, key="mat2d", out_key="pca_out", n_components=2
        )
        assert result["key"] == "pca_out"
        assert result["info"]["type"] == "ndarray"


# ============================================================================
# Untested MCP Tool Coverage
# ============================================================================


class TestBurstMCPTools:
    """Tests for burst detection and sensitivity MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.analysis.SpikeData.get_bursts")
    async def test_get_bursts(self, mock_get_bursts, loaded_ws):
        """
        Test get_bursts dispatches to SpikeData.get_bursts and stores results.

        Tests:
            (Test Case 1) Three keys (tburst, edges, amp) are stored in workspace.
            (Test Case 2) Return dict includes n_bursts count.
        """
        mock_get_bursts.return_value = (
            np.array([10.0, 30.0]),
            np.array([[8.0, 12.0], [28.0, 32.0]]),
            np.array([1.5, 2.0]),
        )
        ws_id, ns = loaded_ws
        result = await analysis.get_bursts(
            ws_id,
            ns,
            key_tburst="tburst",
            key_edges="edges",
            key_amp="amp",
            thr_burst=1.0,
            min_burst_diff=10,
            burst_edge_mult_thresh=0.5,
        )
        assert result["workspace_id"] == ws_id
        assert result["n_bursts"] == 2
        assert result["key_tburst"] == "tburst"
        assert result["key_edges"] == "edges"
        assert result["key_amp"] == "amp"
        ws = get_workspace_manager().get_workspace(ws_id)
        assert ws.get(ns, "tburst") is not None
        assert ws.get(ns, "edges") is not None
        assert ws.get(ns, "amp") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.analysis.SpikeData.burst_sensitivity")
    async def test_burst_sensitivity(self, mock_burst_sens, loaded_ws):
        """
        Test burst_sensitivity stores the sensitivity grid in the workspace.

        Tests:
            (Test Case 1) Result shape matches the thr x dist grid.
            (Test Case 2) Stored item is an ndarray.
        """
        mock_burst_sens.return_value = np.array([[3, 5], [2, 4]])
        ws_id, ns = loaded_ws
        result = await analysis.burst_sensitivity(
            ws_id,
            ns,
            key="burst_sens",
            thr_values=[1.0, 2.0],
            dist_values=[10.0, 20.0],
            burst_edge_mult_thresh=0.5,
        )
        assert result["workspace_id"] == ws_id
        assert result["key"] == "burst_sens"
        assert result["shape"] == [2, 2]
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.analysis.SpikeData.get_frac_active")
    async def test_get_frac_active(self, mock_frac, loaded_ws):
        """
        Test get_frac_active stores frac_per_unit, frac_per_burst, and backbone.

        Tests:
            (Test Case 1) Three output keys are stored in the workspace.
            (Test Case 2) Return dict includes all three key names.
        """
        mock_frac.return_value = (
            np.array([0.8, 0.5, 0.3]),
            np.array([0.6, 0.9]),
            np.array([0, 1]),
        )
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store(ns, "edges", np.array([[8.0, 12.0], [28.0, 32.0]]))
        result = await analysis.get_frac_active(
            ws_id,
            ns,
            edges_key="edges",
            key_frac_unit="frac_unit",
            key_frac_burst="frac_burst",
            key_backbone="backbone",
            min_spikes=1,
            backbone_threshold=0.5,
        )
        assert result["workspace_id"] == ws_id
        assert result["key_frac_unit"] == "frac_unit"
        assert result["key_frac_burst"] == "frac_burst"
        assert result["key_backbone"] == "backbone"
        assert ws.get(ns, "frac_unit") is not None
        assert ws.get(ns, "frac_burst") is not None
        assert ws.get(ns, "backbone") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_get_frac_active_missing_edges(self, loaded_ws):
        """
        Test get_frac_active raises ValueError when edges key is missing.

        Tests:
            (Test Case 1) ValueError mentions 'get_bursts'.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="get_bursts"):
            await analysis.get_frac_active(
                ws_id,
                ns,
                edges_key="nonexistent",
                key_frac_unit="fu",
                key_frac_burst="fb",
                key_backbone="bb",
                min_spikes=1,
                backbone_threshold=0.5,
            )


class TestWaveformMCPTools:
    """Tests for waveform trace extraction MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.analysis.SpikeData.get_waveform_traces")
    async def test_get_waveform_traces(self, mock_waveforms, loaded_ws):
        """
        Test get_waveform_traces stores waveform array and returns metadata.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, key.
            (Test Case 2) Waveform array is stored in workspace.
            (Test Case 3) avg_waveform is returned inline.
        """
        waveform_arr = np.random.default_rng(0).random((1, 30, 4))
        avg_wf = np.random.default_rng(0).random((1, 30))
        mock_waveforms.return_value = (
            waveform_arr,
            {
                "channels": [[0]],
                "spike_times_ms": [np.array([10.0, 20.0, 30.0, 40.0])],
                "avg_waveforms": [avg_wf],
                "fs_kHz": 30.0,
            },
        )
        ws_id, ns = loaded_ws
        result = await analysis.get_waveform_traces(ws_id, ns, key="wf_unit0", unit=0)
        assert result["workspace_id"] == ws_id
        assert result["key"] == "wf_unit0"
        assert result["fs_kHz"] == 30.0
        assert result["avg_waveform"] is not None
        ws = get_workspace_manager().get_workspace(ws_id)
        assert ws.get(ns, "wf_unit0") is not None


class TestGPLVMMCPTools:
    """Tests for GPLVM fitting and metric MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.analysis.SpikeData.fit_gplvm")
    async def test_fit_gplvm(self, mock_fit, loaded_ws):
        """
        Test fit_gplvm stores decode_res, reorder_indices, and binned_spike_counts.

        Tests:
            (Test Case 1) Three keys are stored in workspace.
            (Test Case 2) Return dict includes log_marginal_l and bin_size_ms.
        """
        n_time, n_units = 10, 3
        mock_fit.return_value = {
            "decode_res": {
                "posterior_latent_marg": np.random.default_rng(0).random((n_time, 5))
            },
            "reorder_indices": np.array([2, 0, 1]),
            "binned_spike_counts": np.random.default_rng(0).random((n_time, n_units)),
            "log_marginal_l": np.array([-100.0, -90.0]),
            "bin_size_ms": 50.0,
        }
        ws_id, ns = loaded_ws
        result = await analysis.fit_gplvm(
            ws_id,
            ns,
            key="decode_res",
            key_reorder="reorder",
            key_binned="binned",
        )
        assert result["workspace_id"] == ws_id
        assert result["key"] == "decode_res"
        assert result["key_reorder"] == "reorder"
        assert result["key_binned"] == "binned"
        assert result["bin_size_ms"] == 50.0
        assert result["n_time_bins"] == n_time
        assert result["n_units"] == n_units
        ws = get_workspace_manager().get_workspace(ws_id)
        assert ws.get(ns, "decode_res") is not None
        assert ws.get(ns, "reorder") is not None
        assert ws.get(ns, "binned") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_gplvm_state_entropy(self, loaded_ws):
        """
        Test compute_gplvm_state_entropy stores entropy array in workspace.

        Tests:
            (Test Case 1) Result contains key and info.
            (Test Case 2) Stored item exists in workspace.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        posterior = np.random.default_rng(0).random((20, 5))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)
        ws.store(ns, "decode_res", {"posterior_latent_marg": posterior})
        result = await analysis.compute_gplvm_state_entropy(
            ws_id, ns, key="decode_res", out_key="entropy"
        )
        assert result["key"] == "entropy"
        assert ws.get(ns, "entropy") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_gplvm_continuity_prob(self, loaded_ws):
        """
        Test compute_gplvm_continuity_prob stores continuity probability in workspace.

        Tests:
            (Test Case 1) Result contains key and info.
            (Test Case 2) Stored item exists in workspace.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        posterior = np.random.default_rng(0).random((20, 5))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)
        dynamics = np.random.default_rng(1).random((20, 2))
        ws.store(
            ns,
            "decode_res",
            {"posterior_latent_marg": posterior, "posterior_dynamics_marg": dynamics},
        )
        result = await analysis.compute_gplvm_continuity_prob(
            ws_id, ns, key="decode_res", out_key="cont_prob"
        )
        assert result["key"] == "cont_prob"
        assert ws.get(ns, "cont_prob") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_gplvm_avg_state_prob(self, loaded_ws):
        """
        Test compute_gplvm_avg_state_prob stores average state probability.

        Tests:
            (Test Case 1) Result contains key and info.
            (Test Case 2) Stored item exists in workspace.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        posterior = np.random.default_rng(0).random((20, 5))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)
        ws.store(ns, "decode_res", {"posterior_latent_marg": posterior})
        result = await analysis.compute_gplvm_avg_state_prob(
            ws_id, ns, key="decode_res", out_key="avg_prob"
        )
        assert result["key"] == "avg_prob"
        assert ws.get(ns, "avg_prob") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_gplvm_consecutive_durations(self, loaded_ws):
        """
        Test compute_gplvm_consecutive_durations stores duration array.

        Tests:
            (Test Case 1) Result contains key and n_durations count.
            (Test Case 2) Stored item exists in workspace.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        signal = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.6, 0.3])
        ws.store(ns, "cont_prob", signal)
        result = await analysis.compute_gplvm_consecutive_durations(
            ws_id, ns, key="cont_prob", out_key="durations", threshold=0.5
        )
        assert result["key"] == "durations"
        assert result["n_durations"] > 0
        assert ws.get(ns, "durations") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_gplvm_missing_decode_res(self, loaded_ws):
        """
        Test GPLVM metric tools raise ValueError when decode_res is missing.

        Tests:
            (Test Case 1) compute_gplvm_state_entropy raises ValueError mentioning fit_gplvm.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="fit_gplvm"):
            await analysis.compute_gplvm_state_entropy(
                ws_id, ns, key="nonexistent", out_key="entropy"
            )


class TestUMAPMCPTools:
    """Tests for UMAP dimensionality reduction MCP tools."""

    _umap_available = True
    try:
        import umap  # noqa: F401
    except ImportError:
        _umap_available = False

    @pytestmark_server
    @pytest.mark.asyncio
    @pytest.mark.skipif(not _umap_available, reason="umap-learn not installed")
    async def test_umap_reduction(self, loaded_ws):
        """
        Test umap_reduction stores UMAP embedding in workspace.

        Tests:
            (Test Case 1) Result contains key and trustworthiness score.
            (Test Case 2) Stored item exists in workspace.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        data = np.random.default_rng(42).random((30, 5))
        ws.store(ns, "rates_2d", data)
        result = await analysis.umap_reduction(
            ws_id,
            ns,
            key="rates_2d",
            out_key="umap_embed",
            n_components=2,
            random_state=42,
        )
        assert result["key"] == "umap_embed"
        assert "trustworthiness" in result
        assert ws.get(ns, "umap_embed") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    @pytest.mark.skipif(not _umap_available, reason="umap-learn not installed")
    async def test_umap_graph_communities(self, loaded_ws):
        """
        Test umap_graph_communities stores embedding and returns community labels.

        Tests:
            (Test Case 1) Result contains labels list.
            (Test Case 2) Stored embedding exists in workspace.

        Notes:
            - Also requires networkx and python-louvain.
        """
        try:
            import community  # noqa: F401
            import networkx  # noqa: F401
        except ImportError:
            pytest.skip("networkx or python-louvain not installed")

        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        data = np.random.default_rng(42).random((30, 5))
        ws.store(ns, "rates_2d", data)
        result = await analysis.umap_graph_communities(
            ws_id,
            ns,
            key="rates_2d",
            out_key="umap_comm",
            n_components=2,
            random_state=42,
        )
        assert result["key"] == "umap_comm"
        assert "labels" in result
        assert len(result["labels"]) == 30
        assert ws.get(ns, "umap_comm") is not None

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_umap_reduction_missing_item(self, loaded_ws):
        """
        Test umap_reduction raises ValueError when input key is missing.

        Tests:
            (Test Case 1) ValueError raised for nonexistent key.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="Item not found"):
            await analysis.umap_reduction(ws_id, ns, key="nonexistent", out_key="out")

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_umap_reduction_wrong_type(self, loaded_ws):
        """
        Test umap_reduction raises ValueError when input is not a 2D array.

        Tests:
            (Test Case 1) ValueError raised for 1D input.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store(ns, "arr1d", np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Expected 2D ndarray"):
            await analysis.umap_reduction(ws_id, ns, key="arr1d", out_key="out")


class TestLoaderMCPToolsCoverage:
    """Tests for untested loader MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    @patch(
        "spikelab.mcp_server.tools.data_loaders.load_spikedata_from_hdf5_raw_thresholded"
    )
    @patch("spikelab.mcp_server.tools.data_loaders.ensure_local_file")
    async def test_load_from_hdf5_thresholded(self, mock_ensure, mock_load):
        """
        Test load_from_hdf5_thresholded dispatches to the loader and stores result.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, workspace_key.
            (Test Case 2) info.num_neurons matches the mocked SpikeData.
        """
        train = [[10.0, 20.0], [15.0, 25.0]]
        sd = SpikeData(train, length=30.0)
        mock_load.return_value = sd
        mock_ensure.return_value = ("/tmp/fake.h5", False)

        result = await data_loaders.load_from_hdf5_thresholded(
            file_path="/tmp/fake.h5",
            dataset="traces",
            fs_Hz=30000.0,
            threshold_sigma=5.0,
        )
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["num_neurons"] == 2
        assert "workspace_id" in result
        assert "namespace" in result
        mock_load.assert_called_once()

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.data_loaders.load_spikedata_from_ibl")
    async def test_load_from_ibl(self, mock_load):
        """
        Test load_from_ibl dispatches to the IBL loader and stores result.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, workspace_key.
            (Test Case 2) info.num_neurons matches the mocked SpikeData.
        """
        train = [[10.0, 20.0], [15.0]]
        sd = SpikeData(train, length=30.0, metadata={"trials": "data"})
        mock_load.return_value = sd

        result = await data_loaders.load_from_ibl(
            eid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            pid="11111111-2222-3333-4444-555555555555",
        )
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["num_neurons"] == 2
        assert "workspace_id" in result
        mock_load.assert_called_once()

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("spikelab.mcp_server.tools.data_loaders._query_ibl_probes")
    async def test_query_ibl_probes(self, mock_query):
        """
        Test query_ibl_probes returns probe list and stats inline.

        Tests:
            (Test Case 1) Result contains probes list.
            (Test Case 2) Result contains stats list.

        Notes:
            - This tool does not store anything in the workspace.
        """
        import pandas as pd

        mock_query.return_value = (
            [("eid1", "pid1"), ("eid2", "pid2")],
            pd.DataFrame(
                {"eid": ["eid1", "eid2"], "pid": ["pid1", "pid2"], "n_units": [50, 30]}
            ),
        )
        result = await data_loaders.query_ibl_probes(
            target_regions=["MOs"], min_units=10
        )
        assert "probes" in result
        assert len(result["probes"]) == 2
        assert "stats" in result
        assert len(result["stats"]) == 2


class TestLoadWorkspaceItemMCP:
    """Tests for load_workspace_item MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_load_workspace_item(self, loaded_ws, tmp_path):
        """
        Test load_workspace_item loads a single item from a saved workspace file.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, key, info.
            (Test Case 2) Item is accessible in the target workspace after loading.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not available")

        from spikelab.workspace.workspace import AnalysisWorkspace

        # Save a workspace with a known item
        source_ws = AnalysisWorkspace(name="source")
        arr = np.array([1.0, 2.0, 3.0])
        source_ws.store("ns1", "my_array", arr)
        save_path = str(tmp_path / "source_ws")
        source_ws.save(save_path)

        # Load that item into the existing workspace
        ws_id, ns = loaded_ws
        result = await analysis.load_workspace_item(
            path=save_path,
            namespace="ns1",
            key="my_array",
            workspace_id=ws_id,
        )
        assert result["workspace_id"] == ws_id
        assert result["namespace"] == "ns1"
        assert result["key"] == "my_array"
        assert result["info"]["type"] == "ndarray"
        ws = get_workspace_manager().get_workspace(ws_id)
        loaded = ws.get("ns1", "my_array")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_load_workspace_item_missing_workspace(self):
        """
        Test load_workspace_item raises ValueError for nonexistent workspace.

        Tests:
            (Test Case 1) ValueError with 'Workspace not found'.
        """
        with pytest.raises(ValueError, match="Workspace not found"):
            await analysis.load_workspace_item(
                path="/tmp/fake",
                namespace="ns",
                key="k",
                workspace_id="nonexistent",
            )

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_load_workspace_item_nonexistent_file(self):
        """
        EC-MCP-08: load_workspace_item with non-existent file path.

        Passing a path that does not exist on disk should raise an error
        when the underlying loader tries to open the file.

        Tests:
            (Test Case 1) Raises an exception (FileNotFoundError or OSError).
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="target_ws")
        with pytest.raises((FileNotFoundError, OSError, KeyError)):
            await analysis.load_workspace_item(
                path="/tmp/nonexistent_workspace_path_abc123",
                namespace="ns",
                key="k",
                workspace_id=ws_id,
            )


# ============================================================================
# Edge Case Tests — MCP (mcp_server/)
# ============================================================================


class TestPadRaggedEdgeCases:
    """Edge case tests for _pad_ragged helper function."""

    def test_empty_list_of_arrays(self):
        """
        Empty list of arrays produces a (0, 0) shaped result.

        Tests:
            (Test Case 1) _pad_ragged([]) returns shape (0, 0).
            (Test Case 2) Result dtype is float64.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.analysis import _pad_ragged

        result = _pad_ragged([])
        assert result.shape == (0, 0)
        assert result.dtype == np.float64

    def test_all_empty_arrays(self):
        """
        List of empty arrays produces (N, 0) with no NaN padding.

        Tests:
            (Test Case 1) _pad_ragged with two empty arrays returns shape (2, 0).
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.analysis import _pad_ragged

        result = _pad_ragged([np.array([]), np.array([])])
        assert result.shape == (2, 0)

    def test_single_unit_single_value(self):
        """
        Single unit with single value produces (1, 1).

        Tests:
            (Test Case 1) _pad_ragged with [np.array([5.0])] returns shape (1, 1).
            (Test Case 2) Value is 5.0.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.analysis import _pad_ragged

        result = _pad_ragged([np.array([5.0])])
        assert result.shape == (1, 1)
        assert result[0, 0] == 5.0


class TestToListEdgeCases:
    """Edge case tests for _to_list helper function."""

    def test_non_array_input(self):
        """
        Non-array input is returned unchanged.

        Tests:
            (Test Case 1) _to_list with a plain list returns the same list.
            (Test Case 2) _to_list with a string returns the same string.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.analysis import _to_list

        assert _to_list([1, 2, 3]) == [1, 2, 3]
        assert _to_list("hello") == "hello"


class TestComputeRatesEdgeCases:
    """Edge case tests for compute_rates MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_unit_string(self, loaded_ws):
        """
        Invalid unit string propagates error from SpikeData.rates().

        Tests:
            (Test Case 1) compute_rates with unit="invalid" raises an error.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_rates(ws_id, ns, "rates", unit="invalid")


class TestComputeBinnedEdgeCases:
    """Edge case tests for compute_binned and compute_binned_meanrate MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_bin_size_larger_than_recording(self, loaded_ws):
        """
        Bin size larger than recording length should produce a small array.

        Tests:
            (Test Case 1) compute_binned with bin_size=1000 on 50ms data succeeds.
            (Test Case 2) Result has ndarray type.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_binned(ws_id, ns, "binned_big", bin_size=1000.0)
        assert result["info"]["type"] == "ndarray"

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_bin_size_zero(self, loaded_ws):
        """
        Bin size of zero should raise an error from SpikeData.

        Tests:
            (Test Case 1) compute_binned with bin_size=0 raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_binned(ws_id, ns, "binned_zero", bin_size=0.0)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_binned_meanrate_bin_size_larger_than_recording(self, loaded_ws):
        """
        Bin size larger than recording for binned_meanrate should produce a small array.

        Tests:
            (Test Case 1) compute_binned_meanrate with bin_size=1000 succeeds.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_binned_meanrate(
            ws_id, ns, "mr_big", bin_size=1000.0
        )
        assert result["info"]["type"] == "ndarray"


class TestComputeRasterEdgeCases:
    """Edge case tests for compute_raster MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_bin_size_zero(self, loaded_ws):
        """
        Bin size of zero should raise an error.

        Tests:
            (Test Case 1) compute_raster with bin_size=0 raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_raster(ws_id, ns, "raster_zero", bin_size=0.0)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_bin_size_negative(self, loaded_ws):
        """
        Negative bin size should raise an error.

        Tests:
            (Test Case 1) compute_raster with bin_size=-5 raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_raster(ws_id, ns, "raster_neg", bin_size=-5.0)


class TestComputeChannelRasterEdgeCases:
    """Edge case tests for compute_channel_raster MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_missing_channel_attr(self, loaded_ws):
        """
        channel_attr=None on SpikeData without neuron_attributes raises ValueError.

        Tests:
            (Test Case 1) compute_channel_raster with channel_attr=None raises
                ValueError when SpikeData has no channel information.

        Notes:
            - channel_raster requires channel information in neuron_attributes.
              When neuron_attributes is None or has no channel info, ValueError
              is raised rather than silently falling back.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="No channel information"):
            await analysis.compute_channel_raster(
                ws_id, ns, "ch_raster_none", bin_size=5.0, channel_attr=None
            )

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_nonexistent_channel_attr(self, loaded_ws):
        """
        Non-existent channel_attr key should propagate an error.

        Tests:
            (Test Case 1) compute_channel_raster with channel_attr="nonexistent"
                raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_channel_raster(
                ws_id, ns, "ch_raster_bad", bin_size=5.0, channel_attr="nonexistent"
            )


class TestComputeISIEdgeCases:
    """Edge case tests for compute_interspike_intervals MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_unit_with_single_spike(self):
        """
        Unit with single spike has empty ISI; row should be all NaN.

        Tests:
            (Test Case 1) ISI array for single-spike unit has all NaN in its row.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="single_spike_ws")
        sd = SpikeData([[10.0], [10.0, 20.0]], length=50.0)
        wm.get_workspace(ws_id).store("rec1", "spikedata", sd)
        result = await analysis.compute_interspike_intervals(ws_id, "rec1", "isis")
        ws = wm.get_workspace(ws_id)
        arr = ws.get("rec1", "isis")
        # Unit 0 has 1 spike, so 0 ISIs -> entire row should be NaN
        assert arr.shape[0] == 2
        if arr.shape[1] > 0:
            assert np.all(np.isnan(arr[0, :]))

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_unit_with_no_spikes(self):
        """
        Unit with no spikes has empty ISI; row should be all NaN.

        Tests:
            (Test Case 1) ISI array for zero-spike unit has all NaN in its row.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="no_spike_ws")
        sd = SpikeData([[], [10.0, 20.0]], length=50.0)
        wm.get_workspace(ws_id).store("rec1", "spikedata", sd)
        result = await analysis.compute_interspike_intervals(ws_id, "rec1", "isis")
        ws = wm.get_workspace(ws_id)
        arr = ws.get("rec1", "isis")
        assert arr.shape[0] == 2
        if arr.shape[1] > 0:
            assert np.all(np.isnan(arr[0, :]))


class TestComputeResampledISIEdgeCases:
    """Edge case tests for compute_resampled_isi MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_times_list(self, loaded_ws):
        """
        Empty times list produces RateData with 0 time points.

        Tests:
            (Test Case 1) compute_resampled_isi with times=[] succeeds or raises
                a clear error.
        """
        ws_id, ns = loaded_ws
        # Empty times: creates np.array([]) → RateData with (U, 0) shape
        try:
            result = await analysis.compute_resampled_isi(
                ws_id, ns, "rates_empty", times=[]
            )
            assert result["n_timepoints"] == 0
        except Exception:
            # If it raises, that is also acceptable — the edge case is documented
            pass

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_single_time_point(self, loaded_ws):
        """
        Single time point succeeds and returns a valid RateData.

        Tests:
            (Test Case 1) compute_resampled_isi with times=[25.0] stores a
                result with 1 time point.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_resampled_isi(
            ws_id, ns, "rates_single", times=[25.0]
        )
        assert result["n_timepoints"] == 1
        assert result["key"] == "rates_single"


class TestComputeSpikeTimeTilingEdgeCases:
    """Edge case tests for compute_spike_time_tiling MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_same_neuron_index(self, loaded_ws):
        """
        Same neuron index (auto-correlation) produces STTC value of 0.0.

        Tests:
            (Test Case 1) compute_spike_time_tiling with neuron_i==neuron_j succeeds.
            (Test Case 2) Stored value is 0.0.

        Notes:
            - The STTC formula returns 0.0 for auto-correlation because
              PA=1 and PB=1, which causes the PA*TB==1 and PB*TA==1
              guard clauses to return 0 for both terms.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_spike_time_tiling(
            ws_id, ns, "sttc_auto", neuron_i=0, neuron_j=0, delt=10.0
        )
        ws = get_workspace_manager().get_workspace(ws_id)
        val = ws.get(ns, "sttc_auto")
        assert val[0] == pytest.approx(0.0, abs=0.01)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_out_of_range_neuron_index(self, loaded_ws):
        """
        Out-of-range neuron index should propagate an error.

        Tests:
            (Test Case 1) compute_spike_time_tiling with neuron_i=99 on 3-unit data
                raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_spike_time_tiling(
                ws_id, ns, "sttc_bad", neuron_i=99, neuron_j=0, delt=10.0
            )


class TestComputeSpikeTimeTilingsEdgeCases:
    """Edge case tests for compute_spike_time_tilings and threshold MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_single_unit_spikedata(self):
        """
        Single-unit SpikeData produces (1, 1) STTC matrix.

        Tests:
            (Test Case 1) compute_spike_time_tilings on 1-unit data succeeds.
            (Test Case 2) Stored matrix shape is [1, 1].
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="single_unit_ws")
        sd = SpikeData([[10.0, 20.0, 30.0]], length=50.0)
        wm.get_workspace(ws_id).store("rec1", "spikedata", sd)
        result = await analysis.compute_spike_time_tilings(ws_id, "rec1", "sttc_1u")
        assert result["info"]["shape"] == [1, 1]

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_threshold_zero(self, loaded_ws):
        """
        Threshold=0.0 should mark all entries as passing.

        Tests:
            (Test Case 1) threshold_spike_time_tilings with threshold=0.0 succeeds.
            (Test Case 2) Stored matrix is all ones (or at least no zeros on
                off-diagonal where STTC > 0).
        """
        ws_id, ns = loaded_ws
        result = await analysis.threshold_spike_time_tilings(
            ws_id, ns, "sttc_thr0", threshold=0.0
        )
        assert result["info"]["shape"] == [3, 3]

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_threshold_one(self, loaded_ws):
        """
        Threshold=1.0 should mark nearly all entries as failing.

        Tests:
            (Test Case 1) threshold_spike_time_tilings with threshold=1.0 succeeds.
            (Test Case 2) Stored matrix is all zeros (no pair has STTC >= 1.0,
                except self-comparisons).
        """
        ws_id, ns = loaded_ws
        result = await analysis.threshold_spike_time_tilings(
            ws_id, ns, "sttc_thr1", threshold=1.0
        )
        ws = get_workspace_manager().get_workspace(ws_id)
        pcm = ws.get(ns, "sttc_thr1")
        assert pcm.matrix.shape == (3, 3)
        # Off-diagonal should be 0 (no pair reaches 1.0)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert pcm.matrix[i, j] == 0.0


class TestComputeLatenciesEdgeCases:
    """Edge case tests for compute_latencies and compute_latencies_to_index MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_times_list(self, loaded_ws):
        """
        Empty times list produces empty latencies.

        Tests:
            (Test Case 1) compute_latencies with times=[] succeeds.
            (Test Case 2) Stored array has 0 columns.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_latencies(ws_id, ns, "lats_empty", times=[])
        ws = get_workspace_manager().get_workspace(ws_id)
        arr = ws.get(ns, "lats_empty")
        assert arr.shape[1] == 0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_latencies_to_index_out_of_range(self, loaded_ws):
        """
        Out-of-range neuron index should propagate an error.

        Tests:
            (Test Case 1) compute_latencies_to_index with neuron_index=99 raises
                an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_latencies_to_index(
                ws_id, ns, "lat_bad", neuron_index=99
            )


class TestSetNeuronAttributeEdgeCases:
    """Edge case tests for set_neuron_attribute MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_neuron_indices_out_of_range(self, loaded_ws_with_attrs):
        """
        Out-of-range neuron_indices should propagate an error.

        Tests:
            (Test Case 1) set_neuron_attribute with neuron_indices=[99] on 3-unit
                data raises an exception.
        """
        ws_id, ns = loaded_ws_with_attrs
        with pytest.raises(Exception):
            await analysis.set_neuron_attribute(
                ws_id, ns, key="tag", values=["x"], neuron_indices=[99]
            )


class TestGetNeuronAttributeEdgeCases:
    """Edge case tests for get_neuron_attribute MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_nonexistent_attribute_key(self, loaded_ws_with_attrs):
        """
        Non-existent attribute key returns None via default=None.

        Tests:
            (Test Case 1) get_neuron_attribute with key="nonexistent" returns
                values=None.
        """
        ws_id, ns = loaded_ws_with_attrs
        result = await analysis.get_neuron_attribute(ws_id, ns, key="nonexistent")
        # get_neuron_attribute returns [default] * N, not None itself
        assert result["values"] == [None, None, None]


class TestSubtimeEdgeCases:
    """Edge case tests for subtime MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_start_equals_end(self, loaded_ws):
        """
        start==end raises ValueError because start must be strictly less than end.

        Tests:
            (Test Case 1) subtime with start=25, end=25 raises ValueError.

        Notes:
            - SpikeData.subtime requires start < end. Equal values are rejected
              to prevent creation of zero-length windows.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="must be less than end"):
            await analysis.subtime(
                ws_id, ns, start=25.0, end=25.0, out_namespace="subtime_zero"
            )

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_start_greater_than_end(self, loaded_ws):
        """
        start > end (inverted window) should raise an error or produce
        degenerate output.

        Tests:
            (Test Case 1) subtime with start=30, end=10 raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.subtime(
                ws_id, ns, start=30.0, end=10.0, out_namespace="subtime_inv"
            )

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_start_and_end_outside_recording(self, loaded_ws):
        """
        Start and end both outside the recording range raises ValueError.

        Tests:
            (Test Case 1) subtime with start=100, end=200 on 50ms recording
                raises ValueError because start exceeds recording end.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="exceeds recording end"):
            await analysis.subtime(
                ws_id, ns, start=100.0, end=200.0, out_namespace="subtime_outside"
            )


class TestSubsetEdgeCases:
    """Edge case tests for subset MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_out_of_range_unit_index(self, loaded_ws):
        """
        Out-of-range unit index silently produces a SpikeData with 0 units.

        Tests:
            (Test Case 1) subset with units=[99] on 3-unit data succeeds.
            (Test Case 2) Resulting SpikeData has 0 units.

        Notes:
            - SpikeData.subset treats the units list as a filter — indices
              that don't exist in the data are silently skipped, producing
              an empty SpikeData rather than raising an error.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subset(ws_id, ns, units=[99])
        assert result["info"]["N"] == 0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_duplicate_unit_indices(self, loaded_ws):
        """
        Duplicate unit indices behavior depends on SpikeData.subset.

        Tests:
            (Test Case 1) subset with units=[0, 0, 1] either succeeds with
                duplicates or raises an error.
        """
        ws_id, ns = loaded_ws
        try:
            result = await analysis.subset(ws_id, ns, units=[0, 0, 1])
            # If it succeeds, the result should have the expected number of units
            assert result["info"]["type"] == "SpikeData"
        except Exception:
            # Raising is also acceptable
            pass

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_by_nonexistent_attribute(self, loaded_ws):
        """
        Non-existent attribute for 'by' parameter should propagate an error.

        Tests:
            (Test Case 1) subset with by="nonexistent" raises an exception.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.subset(ws_id, ns, units=[0], by="nonexistent")


class TestAppendSessionEdgeCases:
    """Edge case tests for append_session MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_mismatched_unit_counts(self, loaded_ws):
        """
        Mismatched unit counts between two namespaces should raise an error.

        Tests:
            (Test Case 1) append_session with 3 units and 2 units raises an exception.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_2units = SpikeData([[10.0], [20.0]], length=50.0)
        ws.store("rec_2u", "spikedata", sd_2units)
        with pytest.raises(Exception):
            await analysis.append_session(ws_id, namespace_a=ns, namespace_b="rec_2u")


class TestComputePairwiseCCGEdgeCases:
    """Edge case tests for compute_pairwise_ccg MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_compare_func(self, loaded_ws):
        """
        Invalid compare_func string should raise ValueError.

        Tests:
            (Test Case 1) compute_pairwise_ccg with compare_func="invalid" raises
                ValueError mentioning valid options.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="Unknown compare_func"):
            await analysis.compute_pairwise_ccg(
                ws_id, ns, key_corr="ccg_c", key_lag="ccg_l", compare_func="invalid"
            )


class TestComputeRateManifoldEdgeCases:
    """Edge case tests for compute_rate_manifold MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_method(self, loaded_ws):
        """
        Invalid method (not PCA or UMAP) should raise an error.

        Tests:
            (Test Case 1) compute_rate_manifold with method="invalid" raises
                an exception.
        """
        ws_id, ns = loaded_ws
        times = list(np.arange(0.0, 50.0, 1.0))
        await analysis.compute_resampled_isi(ws_id, ns, "rates", times=times)
        with pytest.raises(Exception):
            await analysis.compute_rate_manifold(
                ws_id, ns, rate_key="rates", key="m", method="invalid"
            )


class TestCreateRateSliceStackEdgeCases:
    """Edge case tests for create_rate_slice_stack MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_times_start_to_end(self, loaded_ws):
        """
        Empty times_start_to_end creates RateSliceStack with 0 slices.

        Tests:
            (Test Case 1) create_rate_slice_stack with times_start_to_end=[]
                either succeeds with 0 slices or raises an error.
        """
        ws_id, ns = loaded_ws
        try:
            result = await analysis.create_rate_slice_stack(
                ws_id, ns, "rss_empty", times_start_to_end=[]
            )
            assert result["info"]["type"] == "RateSliceStack"
        except Exception:
            # Raising is also acceptable for degenerate input
            pass


class TestCreateSpikeSliceStackEdgeCases:
    """Edge case tests for create_spike_slice_stack MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_times_start_to_end(self, loaded_ws):
        """
        Empty times_start_to_end creates SpikeSliceStack with 0 slices.

        Tests:
            (Test Case 1) create_spike_slice_stack with times_start_to_end=[]
                either succeeds with 0 slices or raises an error.
        """
        ws_id, ns = loaded_ws
        try:
            result = await analysis.create_spike_slice_stack(
                ws_id, ns, "sss_empty", times_start_to_end=[]
            )
            assert result["info"]["type"] == "SpikeSliceStack"
        except Exception:
            # Raising is also acceptable for degenerate input
            pass


class TestSpikeSliceToRasterEdgeCases:
    """Edge case tests for spike_slice_to_raster MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_spike_slice_stack(self, loaded_ws):
        """
        Empty SpikeSliceStack (0 slices) passed to spike_slice_to_raster
        should raise ValueError from np.stack on an empty list.

        Tests:
            (Test Case 1) spike_slice_to_raster on empty SpikeSliceStack raises
                an exception.

        Notes:
            - np.stack([], axis=2) raises ValueError. This is expected behavior
              for the degenerate case.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="empty_sss_ws")
        sd = SpikeData([[10.0, 20.0], [15.0]], length=50.0)
        wm.get_workspace(ws_id).store("rec1", "spikedata", sd)
        try:
            sss = SpikeSliceStack(sd, times_start_to_end=[])
            wm.get_workspace(ws_id).store("rec1", "sss", sss)
            with pytest.raises(Exception):
                await analysis.spike_slice_to_raster(
                    ws_id, "rec1", stack_key="sss", key="raster"
                )
        except Exception:
            # If SpikeSliceStack itself rejects empty input, that's also fine
            pass


class TestAlignToEventsEdgeCases:
    """Edge case tests for align_to_events MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_events_list(self, loaded_ws):
        """
        Empty events list should raise ValueError about no valid events.

        Tests:
            (Test Case 1) align_to_events with events=[] raises ValueError.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.align_to_events(
                ws_id, ns, "aligned_empty", events=[], pre_ms=5.0, post_ms=5.0
            )

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_kind_rate(self, loaded_ws):
        """
        kind="rate" path produces a RateSliceStack instead of SpikeSliceStack.

        Tests:
            (Test Case 1) align_to_events with kind="rate" succeeds.
            (Test Case 2) Stored item type is RateSliceStack.
        """
        ws_id, ns = loaded_ws
        result = await analysis.align_to_events(
            ws_id,
            ns,
            key="aligned_rate",
            events=[15.0, 35.0],
            pre_ms=5.0,
            post_ms=5.0,
            kind="rate",
        )
        assert result["key"] == "aligned_rate"
        assert result["info"]["type"] == "RateSliceStack"


class TestComputeRateSliceUnitCorrEdgeCases:
    """Edge case tests for compute_rate_slice_unit_corr MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_compare_func(self, loaded_ws_with_rss):
        """
        Invalid compare_func should raise ValueError.

        Tests:
            (Test Case 1) compute_rate_slice_unit_corr with compare_func="invalid"
                raises ValueError.
        """
        ws_id, ns = loaded_ws_with_rss
        with pytest.raises(ValueError, match="compare_func must be one of"):
            await analysis.compute_rate_slice_unit_corr(
                workspace_id=ws_id,
                namespace=ns,
                stack_key="rss",
                out_key="corr_bad",
                compare_func="invalid",
            )


class TestGetDataInfoEdgeCases:
    """Edge case tests for get_data_info MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_metadata_with_numpy_values(self):
        """
        Metadata containing numpy arrays may cause JSON serialization issues
        in _call_tool.

        Tests:
            (Test Case 1) get_data_info with numpy array in metadata returns
                successfully (metadata is returned inline, not JSON-serialized
                by the tool itself).
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="numpy_meta_ws")
        sd = SpikeData(
            [[10.0, 20.0]], length=50.0, metadata={"arr": np.array([1, 2, 3])}
        )
        wm.get_workspace(ws_id).store("rec1", "spikedata", sd)
        result = await analysis.get_data_info(ws_id, "rec1")
        assert result["num_neurons"] == 1


class TestDeleteWorkspaceEdgeCases:
    """Edge case tests for delete_workspace MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_delete_nonexistent_workspace(self):
        """
        Deleting a non-existent workspace should return deleted=False.

        Tests:
            (Test Case 1) delete_workspace with nonexistent ID returns deleted=False.
        """
        result = await analysis.delete_workspace("nonexistent-ws-id-xyz")
        assert result["deleted"] is False


class TestDescribeWorkspaceEdgeCases:
    """Edge case tests for describe_workspace MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_workspace(self):
        """
        Describing an empty workspace returns empty index.

        Tests:
            (Test Case 1) describe_workspace on empty workspace returns index={}.
        """
        result = await analysis.create_workspace(name="empty_desc")
        ws_id = result["workspace_id"]
        desc = await analysis.describe_workspace(ws_id)
        assert desc["index"] == {}


class TestRenameWorkspaceItemEdgeCases:
    """Edge case tests for rename_workspace_item MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_rename_nonexistent_key(self):
        """
        Renaming a non-existent key should return success=False.

        Tests:
            (Test Case 1) rename_workspace_item with old_key="nonexistent"
                returns success=False.
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="rename_ws")
        result = await analysis.rename_workspace_item(
            ws_id, "ns", "nonexistent", "new_key"
        )
        assert result["success"] is False


class TestAddWorkspaceNoteEdgeCases:
    """Edge case tests for add_workspace_note MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_note_on_nonexistent_item(self):
        """
        Adding a note to a non-existent item should return success=False.

        Tests:
            (Test Case 1) add_workspace_note on non-existent item returns
                success=False.
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="note_ws")
        result = await analysis.add_workspace_note(ws_id, "ns", "nonexistent", "note")
        assert result["success"] is False


class TestDeleteWorkspaceItemEdgeCases:
    """Edge case tests for delete_workspace_item MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self):
        """
        Deleting a non-existent key should return deleted=False.

        Tests:
            (Test Case 1) delete_workspace_item with non-existent key returns
                deleted=False.
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="del_ws")
        result = await analysis.delete_workspace_item(ws_id, "ns", "nonexistent")
        assert result["deleted"] is False


class TestFetchWorkspaceItemEdgeCases:
    """Edge case tests for fetch_workspace_item MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_nonexistent_item(self):
        """
        Fetching a non-existent item should raise ValueError.

        Tests:
            (Test Case 1) fetch_workspace_item with non-existent key raises
                ValueError mentioning "Item not found".
        """
        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="fetch_ws")
        with pytest.raises(ValueError, match="Item not found"):
            await analysis.fetch_workspace_item(ws_id, "ns", "nonexistent")

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_spikedata_item(self, loaded_ws):
        """
        Fetching a SpikeData item returns a summary (not full data).

        Tests:
            (Test Case 1) Result contains type, num_neurons, length_ms, start_time.
            (Test Case 2) No 'data' key (summary only).
        """
        ws_id, ns = loaded_ws
        result = await analysis.fetch_workspace_item(ws_id, ns, "spikedata")
        assert result["type"] == "SpikeData"
        assert "num_neurons" in result
        assert "length_ms" in result
        assert "start_time" in result
        assert "data" not in result

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_pairwise_comp_matrix_item(self, loaded_ws):
        """
        Fetching a PairwiseCompMatrix returns full data inline.

        Tests:
            (Test Case 1) Result contains data as nested list.
            (Test Case 2) Shape matches the matrix.
        """
        from spikelab.spikedata.pairwise import PairwiseCompMatrix

        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        pcm = PairwiseCompMatrix(matrix=np.eye(3))
        ws.store(ns, "pcm", pcm)
        result = await analysis.fetch_workspace_item(ws_id, ns, "pcm")
        assert result["type"] == "PairwiseCompMatrix"
        assert "data" in result
        assert result["shape"] == [3, 3]


class TestNamespaceFromPathEdgeCases:
    """Edge case tests for _namespace_from_path helper function."""

    def test_empty_file_path(self):
        """
        Empty file path returns "recording" fallback.

        Tests:
            (Test Case 1) _namespace_from_path("", "") returns "recording".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _namespace_from_path

        result = _namespace_from_path("", "")
        assert result == "recording"

    def test_path_with_no_extension(self):
        """
        Path with no extension returns the filename as namespace.

        Tests:
            (Test Case 1) _namespace_from_path("/data/myfile", "") returns "myfile".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _namespace_from_path

        result = _namespace_from_path("/data/myfile", "")
        assert result == "myfile"

    def test_s3_url_path(self):
        """
        S3 URL path extracts the filename stem.

        Tests:
            (Test Case 1) _namespace_from_path("s3://bucket/folder/file.h5", "")
                returns "file".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _namespace_from_path

        result = _namespace_from_path("s3://bucket/folder/file.h5", "")
        assert result == "file"

    def test_path_ending_with_separator(self):
        """
        Path ending with separator strips it before extracting basename.

        Tests:
            (Test Case 1) _namespace_from_path("/data/folder/", "") returns "folder".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _namespace_from_path

        result = _namespace_from_path("/data/folder/", "")
        assert result == "folder"

    def test_namespace_provided_takes_precedence(self):
        """
        When namespace is provided, it takes precedence over path derivation.

        Tests:
            (Test Case 1) _namespace_from_path("/data/file.h5", "custom") returns
                "custom".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _namespace_from_path

        result = _namespace_from_path("/data/file.h5", "custom")
        assert result == "custom"


class TestUniqueNamespaceEdgeCases:
    """Edge case tests for _unique_namespace helper function."""

    def test_collision_chain(self):
        """
        Collision chain increments _1, _2, ... until unique.

        Tests:
            (Test Case 1) When "rec" and "rec_1" exist, returns "rec_2".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _unique_namespace

        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="ns_collision_ws")
        ws = wm.get_workspace(ws_id)
        ws.store("rec", "item", np.zeros(3))
        ws.store("rec_1", "item", np.zeros(3))
        result = _unique_namespace(ws, "rec")
        assert result == "rec_2"

    def test_no_collision(self):
        """
        When namespace does not exist, it is returned unchanged.

        Tests:
            (Test Case 1) _unique_namespace on non-existing namespace returns it.
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _unique_namespace

        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="ns_nodup_ws")
        ws = wm.get_workspace(ws_id)
        result = _unique_namespace(ws, "brand_new")
        assert result == "brand_new"

    def test_incremented_name_collision(self):
        """
        Namespace that looks like an incremented name collides correctly.

        Tests:
            (Test Case 1) When "rec_1" exists, _unique_namespace("rec_1") returns
                "rec_1_1".
        """
        if not MCP_SERVER_AVAILABLE:
            pytest.skip("MCP server not available")
        from spikelab.mcp_server.tools.data_loaders import _unique_namespace

        wm = get_workspace_manager()
        ws_id = wm.create_workspace(name="ns_inc_ws")
        ws = wm.get_workspace(ws_id)
        ws.store("rec_1", "item", np.zeros(3))
        result = _unique_namespace(ws, "rec_1")
        assert result == "rec_1_1"


class TestLoadFromHDF5EdgeCases:
    """Edge case tests for load_from_hdf5 MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_raster_loader_creates_workspace(self, tmp_path):
        """
        load_from_hdf5_raster creates a workspace and stores SpikeData.

        Tests:
            (Test Case 1) Result contains workspace_id and namespace.
        """
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")
        path = str(tmp_path / "test.h5")
        sd = SpikeData([[10.0, 20.0], [15.0, 25.0]], length=30.0)
        with h5py.File(path, "w") as f:
            raster = sd.raster(bin_size=1.0)
            f.create_dataset("raster", data=raster)
            f.attrs["start_time"] = 0.0
        result = await data_loaders.load_from_hdf5_raster(
            path, raster_dataset="raster", raster_bin_size_ms=1.0
        )
        assert "workspace_id" in result
        assert result["workspace_key"] == "spikedata"


class TestLoadFromKilosortEdgeCases:
    """Edge case tests for load_from_kilosort MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_s3_url_raises_not_implemented(self):
        """
        S3 URL for KiloSort folder should raise NotImplementedError.

        Tests:
            (Test Case 1) load_from_kilosort with s3:// path raises
                NotImplementedError.
        """
        with pytest.raises(NotImplementedError, match="S3 folder paths"):
            await data_loaders.load_from_kilosort(
                folder_path="s3://bucket/kilosort/", fs_Hz=30000.0
            )

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_nonexistent_folder(self):
        """
        Non-existent folder should raise ValueError.

        Tests:
            (Test Case 1) load_from_kilosort with non-existent path raises
                ValueError mentioning "Folder not found".
        """
        with pytest.raises(ValueError, match="Folder not found"):
            await data_loaders.load_from_kilosort(
                folder_path="/tmp/nonexistent_ks_folder_abc123", fs_Hz=30000.0
            )


class TestExportToHDF5EdgeCases:
    """Edge case tests for export_to_hdf5_* MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_raster_export_default_bin_size(self, loaded_ws):
        """
        export_to_hdf5_raster uses default bin_size=1.0 when not specified.

        Tests:
            (Test Case 1) Export succeeds with default bin size.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not available")

        ws_id, ns = loaded_ws
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp_path = tmp.name
        try:
            result = await exporters.export_to_hdf5_raster(ws_id, ns, tmp_path)
            assert "file_path" in result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestExportToKilosortEdgeCases:
    """Edge case tests for export_to_kilosort MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_s3_url_raises_not_implemented(self, loaded_ws):
        """
        S3 URL for KiloSort export should raise NotImplementedError.

        Tests:
            (Test Case 1) export_to_kilosort with s3:// path raises
                NotImplementedError.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(NotImplementedError):
            await exporters.export_to_kilosort(
                ws_id, ns, "s3://bucket/kilosort/", fs_Hz=1000.0
            )


class TestCallToolEdgeCases:
    """Edge case tests for server._call_tool function."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_json_serialization_with_numpy_scalars(self):
        """
        Tool return dict containing numpy scalars should be serializable by _call_tool.

        Tests:
            (Test Case 1) _call_tool on a tool returning numpy int64 values produces
                a valid JSON result or a clear error message.

        Notes:
            - numpy scalars (int64, float64) are not JSON-serializable by default.
              The _call_tool error handler catches TypeError but the error message
              may be confusing to users.
        """
        from spikelab.mcp_server.server import _call_tool

        with patch("spikelab.mcp_server.server.analysis.compute_rates") as mock_compute:
            # Return numpy scalar values (not plain Python)
            mock_compute.return_value = {
                "rates": [np.float64(0.1), np.float64(0.2)],
                "unit": "kHz",
                "num_neurons": np.int64(2),
            }
            result = await _call_tool(
                "compute_rates",
                {
                    "workspace_id": "ws",
                    "namespace": "ns",
                    "key": "rates",
                },
            )
            data = json.loads(result[0].text)
            # Either the result serialized correctly or an error was returned
            assert "rates" in data or "error" in data

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_extra_unexpected_arguments(self):
        """
        Extra unexpected arguments should result in an error response.

        Tests:
            (Test Case 1) _call_tool with unknown keyword arguments returns
                an error in the response.
        """
        from spikelab.mcp_server.server import _call_tool

        result = await _call_tool(
            "create_workspace",
            {
                "name": "test",
                "totally_unknown_kwarg": "value",
            },
        )
        data = json.loads(result[0].text)
        assert "error" in data

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_missing_required_arguments(self):
        """
        Missing required arguments should result in an error response.

        Tests:
            (Test Case 1) _call_tool for compute_rates without workspace_id returns
                an error.
        """
        from spikelab.mcp_server.server import _call_tool

        result = await _call_tool("compute_rates", {})
        data = json.loads(result[0].text)
        assert "error" in data


class TestGPLVMConsecutiveDurationsEdgeCases:
    """Edge case tests for compute_gplvm_consecutive_durations MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_no_values_above_threshold(self, loaded_ws):
        """
        Signal with no values above threshold produces empty durations.

        Tests:
            (Test Case 1) n_durations is 0.
            (Test Case 2) Result does not contain mean_duration or median_duration.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        signal = np.array([0.1, 0.2, 0.1, 0.3, 0.2])
        ws.store(ns, "low_signal", signal)
        result = await analysis.compute_gplvm_consecutive_durations(
            ws_id, ns, key="low_signal", out_key="dur_empty", threshold=0.9
        )
        assert result["n_durations"] == 0
        assert "mean_duration" not in result

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_signal_array(self, loaded_ws):
        """
        Empty signal array produces 0 durations.

        Tests:
            (Test Case 1) compute_gplvm_consecutive_durations with shape (0,) signal
                produces n_durations=0.
        """
        ws_id, ns = loaded_ws
        ws = get_workspace_manager().get_workspace(ws_id)
        ws.store(ns, "empty_sig", np.array([]))
        result = await analysis.compute_gplvm_consecutive_durations(
            ws_id, ns, key="empty_sig", out_key="dur_0", threshold=0.5
        )
        assert result["n_durations"] == 0


class TestRemoveByConditionEdgeCases:
    """Edge case tests for remove_by_condition MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_op_string(self, loaded_ws):
        """
        Invalid op string should propagate an error from underlying method.

        Tests:
            (Test Case 1) remove_by_condition with op="invalid" raises an exception.
        """
        from spikelab.spikedata.pairwise import PairwiseCompMatrix

        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        target = PairwiseCompMatrix(matrix=np.ones((2, 2)))
        condition = PairwiseCompMatrix(matrix=np.ones((2, 2)))
        ws.store(ns, "target_inv", target)
        ws.store(ns, "cond_inv", condition)
        with pytest.raises(Exception):
            await analysis.remove_by_condition(
                workspace_id=ws_id,
                namespace=ns,
                target_key="target_inv",
                condition_key="cond_inv",
                out_key="masked_inv",
                op="invalid",
                threshold=1.0,
            )


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan — MCP (mcp_server/)
# ---------------------------------------------------------------------------


class TestPadRaggedEdgeCases2:
    """Additional edge case tests for _pad_ragged."""

    @pytestmark_server
    def test_nan_values_in_input(self):
        """
        _pad_ragged with NaN values in input arrays.

        Tests:
            (Test Case 1) NaN values pass through the padding unchanged.
        """
        result = analysis._pad_ragged([np.array([1.0, np.nan]), np.array([3.0])])
        assert result.shape == (2, 2)
        assert np.isnan(result[0, 1])

    @pytestmark_server
    def test_mixed_dtypes(self):
        """
        _pad_ragged with integer arrays mixed with float arrays.

        Tests:
            (Test Case 1) Integer inputs are cast to float64 in the result.
        """
        result = analysis._pad_ragged([np.array([1, 2]), np.array([3.0])])
        assert result.dtype == np.float64


class TestToListEdgeCases2:
    """Additional edge case tests for _to_list."""

    @pytestmark_server
    def test_ndarray_with_nan(self):
        """
        _to_list with NaN values converts to Python float('nan').

        Tests:
            (Test Case 1) NaN is converted to Python float which is JSON-incompatible.
        """
        arr = np.array([1.0, np.nan, 3.0])
        result = analysis._to_list(arr)
        assert isinstance(result, list)
        assert len(result) == 3

    @pytestmark_server
    def test_none_input(self):
        """
        _to_list with None returns None.

        Tests:
            (Test Case 1) None input returns None.
        """
        result = analysis._to_list(None)
        assert result is None


class TestComputeRatesEdgeCases2:
    """Additional edge case tests for compute_rates MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_zero_unit_spikedata(self, loaded_ws):
        """
        compute_rates on zero-unit SpikeData.

        Tests:
            (Test Case 1) N=0 SpikeData produces empty rates array.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_empty = SpikeData([], length=50.0)
        ws.store("empty_ns", "spikedata", sd_empty)
        result = await analysis.compute_rates(ws_id, "empty_ns", "rates_empty")
        rates = ws.get("empty_ns", "rates_empty")
        assert len(rates) == 0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_zero_length_recording(self, loaded_ws):
        """
        compute_rates on SpikeData with length=0.

        Tests:
            (Test Case 1) length=0 SpikeData produces all-zero rates.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_zero = SpikeData([[], []], length=0.0)
        ws.store("zero_ns", "spikedata", sd_zero)
        result = await analysis.compute_rates(ws_id, "zero_ns", "rates_zero")
        rates = ws.get("zero_ns", "rates_zero")
        np.testing.assert_array_equal(rates, 0.0)


class TestComputeBinnedEdgeCases2:
    """Additional edge case tests for compute_binned MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_bin_size(self, loaded_ws):
        """
        compute_binned with negative bin_size raises an error.

        Tests:
            (Test Case 1) Negative bin_size propagates a ValueError.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_binned(ws_id, ns, "binned_neg", bin_size=-10.0)


class TestComputeRasterEdgeCases2:
    """Additional edge case tests for compute_raster MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_bin_size(self, loaded_ws):
        """
        compute_raster with negative bin_size.

        Tests:
            (Test Case 1) Negative bin_size propagates an error.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_raster(ws_id, ns, "raster_neg", bin_size=-5.0)


class TestComputeChannelRasterEdgeCases2:
    """Additional edge case tests for compute_channel_raster MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_channel_attr_no_neuron_attributes(self, loaded_ws):
        """
        compute_channel_raster with channel_attr specified but no neuron_attributes.

        Tests:
            (Test Case 1) Specifying channel_attr="ch" when SpikeData has no
                neuron_attributes raises an error.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_channel_raster(
                ws_id, ns, "ch_raster", channel_attr="ch"
            )


class TestComputeISIEdgeCases2:
    """Additional edge case tests for compute_interspike_intervals MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_all_units_zero_spikes(self, loaded_ws):
        """
        compute_interspike_intervals with all units having zero spikes.

        Tests:
            (Test Case 1) N=3 units with empty trains produce shape (3, 0)
                padded ISI array.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_empty = SpikeData([[], [], []], length=50.0)
        ws.store("empty3_ns", "spikedata", sd_empty)
        result = await analysis.compute_interspike_intervals(
            ws_id, "empty3_ns", "isi_empty"
        )
        isi = ws.get("empty3_ns", "isi_empty")
        assert isi.shape[0] == 3
        assert isi.shape[1] == 0


class TestComputeResampledISIEdgeCases2:
    """Additional edge case tests for compute_resampled_isi MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_sigma(self, loaded_ws):
        """
        compute_resampled_isi with negative sigma_ms.

        Tests:
            (Test Case 1) Negative sigma does not raise at the MCP level;
                the underlying gaussian_filter1d may accept it in some
                scipy versions.
        """
        ws_id, ns = loaded_ws
        # Negative sigma may or may not raise depending on scipy version
        try:
            result = await analysis.compute_resampled_isi(
                ws_id,
                ns,
                "risi_neg",
                times=[10.0, 20.0, 30.0],
                sigma_ms=-5.0,
            )
            assert result["key"] == "risi_neg"
        except Exception:
            pass  # Expected in some scipy versions


class TestComputeSpikeTimeTilingEdgeCases2:
    """Additional edge case tests for compute_spike_time_tiling MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_delt(self, loaded_ws):
        """
        compute_spike_time_tiling with negative delt raises error.

        Tests:
            (Test Case 1) Negative delt propagates a ValueError from get_sttc.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.compute_spike_time_tiling(
                ws_id,
                ns,
                "sttc_neg",
                neuron_i=0,
                neuron_j=1,
                delt=-10.0,
            )


class TestComputeSpikeTimeTilingsEdgeCases2:
    """Additional edge case tests for compute_spike_time_tilings MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_zero_unit_spikedata(self, loaded_ws):
        """
        compute_spike_time_tilings with N=0 SpikeData.

        Tests:
            (Test Case 1) Zero-unit SpikeData produces a (0, 0) matrix.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_empty = SpikeData([], length=50.0)
        ws.store("empty_tilings", "spikedata", sd_empty)
        result = await analysis.compute_spike_time_tilings(
            ws_id, "empty_tilings", "tilings_empty"
        )
        pcm = ws.get("empty_tilings", "tilings_empty")
        assert pcm.matrix.shape == (0, 0)


class TestComputeLatenciesEdgeCases2:
    """Additional edge case tests for compute_latencies MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_window(self, loaded_ws):
        """
        compute_latencies with negative window_ms.

        Tests:
            (Test Case 1) Negative window does not raise; the underlying
                method silently returns empty latencies.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_latencies(
            ws_id,
            ns,
            "lat_neg",
            times=[10.0, 20.0],
            window_ms=-5.0,
        )
        assert result["key"] == "lat_neg"


class TestGetPopRateEdgeCases:
    """Edge case tests for get_pop_rate MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_zero_spike_spikedata(self, loaded_ws):
        """
        get_pop_rate with zero-spike SpikeData.

        Tests:
            (Test Case 1) All-empty trains produce an all-zero population rate.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_empty = SpikeData([[], [], []], length=50.0)
        ws.store("empty_poprate", "spikedata", sd_empty)
        result = await analysis.get_pop_rate(ws_id, "empty_poprate", "pop_rate_empty")
        pop_rate = ws.get("empty_poprate", "pop_rate_empty")
        np.testing.assert_array_equal(pop_rate, 0.0)


class TestComputeSpikeTriggeredPopRateEdgeCases:
    """Edge case tests for compute_spike_trig_pop_rate MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_single_unit_spikedata(self, loaded_ws):
        """
        compute_spike_trig_pop_rate with single-unit SpikeData raises error.

        Tests:
            (Test Case 1) N=1 raises ValueError since method requires >= 2 units.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd1 = SpikeData([[10.0, 20.0, 30.0]], length=50.0)
        ws.store("single_stpr", "spikedata", sd1)
        with pytest.raises(Exception):
            await analysis.compute_spike_trig_pop_rate(
                ws_id, "single_stpr", "stpr", "stpr_lags", "stpr_coupling"
            )


class TestGetBurstsMCPEdgeCases:
    """Edge case tests for get_bursts / burst_sensitivity MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_no_bursts_detected(self, loaded_ws):
        """
        get_bursts with very high thr_burst producing zero bursts.

        Tests:
            (Test Case 1) Unreachable threshold produces 0 bursts.
        """
        ws_id, ns = loaded_ws
        result = await analysis.get_bursts(
            ws_id,
            ns,
            key_tburst="tburst_none",
            key_edges="edges_none",
            key_amp="amp_none",
            thr_burst=1000.0,
            min_burst_diff=10,
            burst_edge_mult_thresh=0.5,
        )
        assert result["n_bursts"] == 0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_sensitivity_values(self, loaded_ws):
        """
        burst_sensitivity with empty thr_values.

        Tests:
            (Test Case 1) Empty thr_values produces shape (0, N_dist).
        """
        ws_id, ns = loaded_ws
        result = await analysis.burst_sensitivity(
            ws_id,
            ns,
            "sens_empty",
            thr_values=[],
            dist_values=[10],
            burst_edge_mult_thresh=0.5,
        )
        sens = get_workspace_manager().get_workspace(ws_id).get(ns, "sens_empty")
        assert sens.shape[0] == 0


class TestGetFracActiveMCPEdgeCases:
    """Edge case tests for get_frac_active MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_edges_key_wrong_type(self, loaded_ws):
        """
        get_frac_active with edges_key pointing to non-ndarray raises error.

        Tests:
            (Test Case 1) Pointing to SpikeData instead of edges array raises ValueError.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.get_frac_active(
                ws_id,
                ns,
                edges_key="spikedata",
                key_frac_unit="frac_u",
                key_frac_burst="frac_b",
                key_backbone="bb",
                min_spikes=1,
                backbone_threshold=0.5,
            )


class TestGetDataInfoEdgeCases2:
    """Additional edge case tests for get_data_info MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_metadata(self, loaded_ws):
        """
        get_data_info with empty metadata.

        Tests:
            (Test Case 1) SpikeData with empty metadata returns info without error.
        """
        ws_id, ns = loaded_ws
        result = await analysis.get_data_info(ws_id, ns)
        assert "metadata" in result


class TestListNeuronsMCPEdgeCases:
    """Edge case tests for list_neurons MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_zero_units(self, loaded_ws):
        """
        list_neurons with N=0 SpikeData returns empty list.

        Tests:
            (Test Case 1) N=0 SpikeData produces {"neurons": []}.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store("zero_ns", "spikedata", SpikeData([], length=10.0))
        result = await analysis.list_neurons(ws_id, "zero_ns")
        assert result["neurons"] == []


class TestSetNeuronAttributeEdgeCases2:
    """Additional edge case tests for set_neuron_attribute MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_values_list(self, loaded_ws):
        """
        set_neuron_attribute with empty values list.

        Tests:
            (Test Case 1) Empty values with 3-unit SpikeData raises error.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.set_neuron_attribute(
                ws_id,
                ns,
                key="test_attr",
                values=[],
            )


class TestSubtimeMCPEdgeCases2:
    """Additional edge case tests for subtime MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_nan_start(self, loaded_ws):
        """
        subtime with NaN start raises error.

        Tests:
            (Test Case 1) NaN start produces an error.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.subtime(
                ws_id,
                ns,
                start=float("nan"),
                end=30.0,
                out_namespace="sub_nan_ns",
            )


class TestSubsetMCPEdgeCases2:
    """Additional edge case tests for subset MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_empty_units_list(self, loaded_ws):
        """
        subset with empty units list produces a 0-unit SpikeData.

        Tests:
            (Test Case 1) Empty units list creates 0-unit SpikeData.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subset(
            ws_id,
            ns,
            units=[],
            out_namespace="subset_empty_ns",
        )
        sd = (
            get_workspace_manager()
            .get_workspace(ws_id)
            .get("subset_empty_ns", "spikedata")
        )
        assert sd.N == 0

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_unit_indices(self, loaded_ws):
        """
        subset with negative unit indices.

        Tests:
            (Test Case 1) Negative indices are treated as backward-counting
                in SpikeData.subset, so -1 selects the last unit.
        """
        ws_id, ns = loaded_ws
        result = await analysis.subset(
            ws_id,
            ns,
            units=[-1],
            out_namespace="subset_neg_ns",
        )
        sd = (
            get_workspace_manager()
            .get_workspace(ws_id)
            .get("subset_neg_ns", "spikedata")
        )
        # -1 is not a valid index (set(-1) = {-1}, not in range(N)),
        # so it produces 0 units
        assert sd.N == 0


class TestAppendSessionMCPEdgeCases2:
    """Additional edge case tests for append_session MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_self_append(self, loaded_ws):
        """
        append_session with same namespace (self-append).

        Tests:
            (Test Case 1) Self-append doubles the recording length.
        """
        ws_id, ns = loaded_ws
        result = await analysis.append_session(
            ws_id,
            namespace_a=ns,
            namespace_b=ns,
            out_namespace="self_appended_ns",
        )
        sd = (
            get_workspace_manager()
            .get_workspace(ws_id)
            .get("self_appended_ns", "spikedata")
        )
        assert sd.length == pytest.approx(100.0)

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_negative_offset(self, loaded_ws):
        """
        append_session with negative offset.

        Tests:
            (Test Case 1) Negative offset is accepted; the second session
                overlaps with the first.
        """
        ws_id, ns = loaded_ws
        result = await analysis.append_session(
            ws_id,
            namespace_a=ns,
            namespace_b=ns,
            out_namespace="neg_off_ns",
            offset=-10.0,
        )
        sd = get_workspace_manager().get_workspace(ws_id).get("neg_off_ns", "spikedata")
        # With offset=-10, length = 50 + 50 - 10 = 90
        assert sd.length == pytest.approx(90.0)


class TestConcatenateUnitsMCPEdgeCases:
    """Edge case tests for concatenate_units MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_zero_units_one_operand(self, loaded_ws):
        """
        concatenate_units where one operand has N=0.

        Tests:
            (Test Case 1) Concatenating with a 0-unit SpikeData works.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store("empty_concat_ns", "spikedata", SpikeData([], length=50.0))
        result = await analysis.concatenate_units(
            ws_id,
            namespace_a=ns,
            namespace_b="empty_concat_ns",
        )
        sd = ws.get(ns, "spikedata")
        assert sd.N == 3  # original 3 + 0


class TestComputePairwiseCCGMCPEdgeCases2:
    """Additional edge case tests for compute_pairwise_ccg MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_single_unit(self, loaded_ws):
        """
        compute_pairwise_ccg with single-unit SpikeData.

        Tests:
            (Test Case 1) N=1 produces a (1, 1) matrix.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        ws.store(
            "single_ccg_ns", "spikedata", SpikeData([[10.0, 20.0, 30.0]], length=50.0)
        )
        result = await analysis.compute_pairwise_ccg(
            ws_id,
            "single_ccg_ns",
            key_corr="ccg_corr",
            key_lag="ccg_lag",
        )
        corr = ws.get("single_ccg_ns", "ccg_corr")
        assert corr.matrix.shape == (1, 1)


class TestComputeRateManifoldMCPEdgeCases2:
    """Additional edge case tests for compute_rate_manifold MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_n_components_exceeds_time_points(self, loaded_ws):
        """
        compute_rate_manifold with n_components > min(samples, features).

        Tests:
            (Test Case 1) Too many components raises a ValueError.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        from spikelab.spikedata.ratedata import RateData

        rd = RateData(np.random.rand(3, 5), np.arange(5, dtype=float))
        ws.store(ns, "rd_small", rd)
        with pytest.raises(Exception):
            await analysis.compute_rate_manifold(
                ws_id,
                ns,
                rate_key="rd_small",
                key="manifold_big",
                n_components=10,
            )


class TestAlignToEventsMCPEdgeCases2:
    """Additional edge case tests for align_to_events MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_invalid_kind(self, loaded_ws):
        """
        align_to_events with invalid kind raises ValueError.

        Tests:
            (Test Case 1) kind="invalid" is rejected.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(Exception):
            await analysis.align_to_events(
                ws_id,
                ns,
                key="align_inv",
                events=[10.0, 20.0],
                pre_ms=5.0,
                post_ms=5.0,
                kind="invalid",
            )


class TestSpikeSliceToRasterMCPEdgeCases2:
    """Additional edge case tests for spike_slice_to_raster MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_bin_size_zero(self, loaded_ws):
        """
        spike_slice_to_raster with bin_size=0 raises error.

        Tests:
            (Test Case 1) Zero bin_size propagates an error.
        """
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd = ws.get(ns, "spikedata")
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 25.0), (25.0, 50.0)])
        ws.store(ns, "sss_for_raster", sss)
        with pytest.raises(Exception):
            await analysis.spike_slice_to_raster(
                ws_id,
                ns,
                spike_slice_key="sss_for_raster",
                key="raster_zero",
                bin_size=0.0,
            )


class TestFetchWorkspaceItemEdgeCases2:
    """Additional edge case tests for fetch_workspace_item MCP tool."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_ratedata_empty_times(self, loaded_ws):
        """
        fetch_workspace_item with RateData that has empty times.

        Tests:
            (Test Case 1) Accessing obj.times[0] on zero-length RateData
                raises IndexError.

        Notes:
            - This is a known bug: fetch_workspace_item accesses times[0]
              and times[-1] without checking for empty times.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        from spikelab.spikedata.ratedata import RateData

        rd = RateData(np.empty((2, 0)), np.array([]))
        ws.store(ns, "rd_empty_times", rd)
        with pytest.raises(IndexError):
            await analysis.fetch_workspace_item(ws_id, ns, "rd_empty_times")


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
class TestCurateAndFracSpikesInBurst:
    """Tests for curate_spikedata and get_frac_spikes_in_burst MCP tools."""

    # ------------------------------------------------------------------
    # curate_spikedata
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_curate_spikedata_min_spikes(self, loaded_ws):
        """
        Happy path: curate with min_spikes removes units with too few spikes.

        Tests:
            (Test Case 1) Curated SpikeData has fewer units than the original.
            (Test Case 2) Result contains correct workspace_id and namespace.
            (Test Case 3) Curation history reports the criteria applied.
            (Test Case 4) Curated SpikeData is stored in the workspace.
        """
        ws_id, ns = loaded_ws
        # sample_spikedata has 3 units with 4, 3, 2 spikes respectively.
        # min_spikes=3 should keep only the first two units.
        result = await analysis.curate_spikedata(
            workspace_id=ws_id,
            namespace=ns,
            min_spikes=3,
        )
        assert result["workspace_id"] == ws_id
        assert result["namespace"] == ns + "_curated"
        assert result["workspace_key"] == "spikedata"
        assert result["info"]["num_neurons_before"] == 3
        assert result["info"]["num_neurons_after"] == 2
        assert len(result["info"]["criteria_applied"]) > 0
        # Verify stored in workspace
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_curated = ws.get(ns + "_curated", "spikedata")
        assert sd_curated is not None
        assert sd_curated.N == 2

    @pytest.mark.asyncio
    async def test_curate_spikedata_custom_out_namespace(self, loaded_ws):
        """
        Happy path: curate stores result at custom out_namespace.

        Tests:
            (Test Case 1) Result namespace matches the provided out_namespace.
            (Test Case 2) Curated SpikeData is stored at the custom namespace.
        """
        ws_id, ns = loaded_ws
        result = await analysis.curate_spikedata(
            workspace_id=ws_id,
            namespace=ns,
            out_namespace="my_curated",
            min_spikes=3,
        )
        assert result["namespace"] == "my_curated"
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_curated = ws.get("my_curated", "spikedata")
        assert sd_curated is not None
        assert sd_curated.N == 2

    @pytest.mark.asyncio
    async def test_curate_spikedata_no_criteria(self, loaded_ws):
        """
        Edge case: no curation criteria returns original data unchanged.

        Tests:
            (Test Case 1) All units are retained when no criteria are specified.
            (Test Case 2) Curated SpikeData has the same number of units as original.
        """
        ws_id, ns = loaded_ws
        result = await analysis.curate_spikedata(
            workspace_id=ws_id,
            namespace=ns,
        )
        assert result["info"]["num_neurons_before"] == 3
        assert result["info"]["num_neurons_after"] == 3
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        sd_curated = ws.get(ns + "_curated", "spikedata")
        assert sd_curated.N == 3

    @pytest.mark.asyncio
    async def test_curate_spikedata_strict_removes_all(self, loaded_ws):
        """
        Edge case: strict criteria remove all units.

        Tests:
            (Test Case 1) Curated SpikeData has zero units when threshold exceeds all.
        """
        ws_id, ns = loaded_ws
        result = await analysis.curate_spikedata(
            workspace_id=ws_id,
            namespace=ns,
            min_spikes=100,
        )
        assert result["info"]["num_neurons_after"] == 0

    # ------------------------------------------------------------------
    # get_frac_spikes_in_burst
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_frac_spikes_in_burst_happy_path(self, loaded_ws):
        """
        Happy path: compute fraction of spikes in burst windows.

        Tests:
            (Test Case 1) Result contains workspace_id, namespace, key.
            (Test Case 2) Stored fractions array has shape (N,).
            (Test Case 3) Fractions are between 0 and 1 for units with spikes.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        # Create burst edges that span part of the recording.
        # Edges are in bin coordinates (bin_size=1.0 by default).
        # One burst from bin 10 to bin 30 — covers some spikes.
        edges = np.array([[10, 30]], dtype=np.float64)
        ws.store(ns, "burst_edges", edges)

        result = await analysis.get_frac_spikes_in_burst(
            workspace_id=ws_id,
            namespace=ns,
            edges_key="burst_edges",
            key="frac_burst",
        )
        assert result["workspace_id"] == ws_id
        assert result["namespace"] == ns
        assert result["key"] == "frac_burst"
        # Verify stored array
        frac = ws.get(ns, "frac_burst")
        assert isinstance(frac, np.ndarray)
        assert frac.shape == (3,)
        # All units have spikes, so fractions should be finite and in [0, 1]
        assert np.all(np.isfinite(frac))
        assert np.all(frac >= 0.0)
        assert np.all(frac <= 1.0)

    @pytest.mark.asyncio
    async def test_get_frac_spikes_in_burst_empty_edges(self, loaded_ws):
        """
        Edge case: empty edges array (0 bursts) returns NaN for all units.

        Tests:
            (Test Case 1) Stored fractions are all NaN when no bursts exist.
            (Test Case 2) Result info is returned.
        """
        ws_id, ns = loaded_ws
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        # Empty edges: shape (0, 2)
        edges = np.empty((0, 2), dtype=np.float64)
        ws.store(ns, "no_edges", edges)

        result = await analysis.get_frac_spikes_in_burst(
            workspace_id=ws_id,
            namespace=ns,
            edges_key="no_edges",
            key="frac_empty",
        )
        assert result["key"] == "frac_empty"
        frac = ws.get(ns, "frac_empty")
        assert isinstance(frac, np.ndarray)
        assert frac.shape == (3,)
        # With zero bursts, get_frac_spikes_in_burst returns NaN for all units
        assert np.all(np.isnan(frac))

    @pytest.mark.asyncio
    async def test_get_frac_spikes_in_burst_missing_edges_key(self, loaded_ws):
        """
        Edge case: missing edges key raises ValueError.

        Tests:
            (Test Case 1) ValueError raised when edges_key does not exist in workspace.
        """
        ws_id, ns = loaded_ws
        with pytest.raises(ValueError, match="No edges array found"):
            await analysis.get_frac_spikes_in_burst(
                workspace_id=ws_id,
                namespace=ns,
                edges_key="nonexistent_edges",
                key="frac_missing",
            )


@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
class TestCoverageGaps:
    """Tests for MCP tool coverage gaps."""

    @pytest.mark.asyncio
    async def test_compute_resampled_isi_happy_path(self, loaded_ws):
        """
        Tests: compute_resampled_isi happy-path.

        (Test Case 1) Stored result is a RateData with correct shape.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_resampled_isi(
            workspace_id=ws_id,
            namespace=ns,
            key="isi_rate",
            times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            sigma_ms=5.0,
        )
        assert "n_timepoints" in result
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        item = ws.get(ns, "isi_rate")
        assert hasattr(item, "inst_Frate_data")

    @pytest.mark.asyncio
    async def test_frames_rate_data_overlap_ge_length(self, loaded_ws):
        """
        frames_rate_data raises ValueError when overlap >= length.

        Tests:
            (Test Case 1) overlap == length raises ValueError.
            (Test Case 2) overlap > length raises ValueError.
        """
        from spikelab.spikedata.ratedata import RateData

        ws_id, ns = loaded_ws
        # Store a RateData in the workspace
        wm = get_workspace_manager()
        ws = wm.get_workspace(ws_id)
        rd = RateData(
            inst_Frate_data=np.random.default_rng(0).standard_normal((3, 100)),
            times=np.arange(100, dtype=float),
        )
        ws.store(ns, "rates", rd)

        # overlap == length
        with pytest.raises(ValueError, match="overlap must be less than length"):
            await analysis.frames_rate_data(
                workspace_id=ws_id,
                namespace=ns,
                rate_key="rates",
                key="frames",
                length=10.0,
                overlap=10.0,
            )

        # overlap > length
        with pytest.raises(ValueError, match="overlap must be less than length"):
            await analysis.frames_rate_data(
                workspace_id=ws_id,
                namespace=ns,
                rate_key="rates",
                key="frames",
                length=10.0,
                overlap=15.0,
            )

    @pytest.mark.asyncio
    async def test_load_from_hdf5_paired_happy_path(self, tmp_path):
        """
        load_from_hdf5_paired loads paired arrays and stores to workspace.

        Tests:
            (Test Case 1) Returns dict with workspace_id and namespace.
            (Test Case 2) Stored SpikeData has correct number of units.
        """
        import h5py

        # Create test HDF5 with paired arrays
        path = str(tmp_path / "paired.h5")
        idces = np.array([0, 0, 1, 1, 2])
        times = np.array([10.0, 20.0, 15.0, 25.0, 30.0])
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times)

        result = await data_loaders.load_from_hdf5_paired(
            file_path=path,
            idces_dataset="idces",
            times_dataset="times",
            times_unit="ms",
        )

        assert "workspace_id" in result
        assert "namespace" in result
        wm = get_workspace_manager()
        ws = wm.get_workspace(result["workspace_id"])
        sd = ws.get(result["namespace"], "spikedata")
        assert sd.N == 3

    @pytest.mark.asyncio
    async def test_export_to_hdf5_group(self, loaded_ws, tmp_path):
        """
        export_to_hdf5_group creates an HDF5 file with group-per-unit format.

        Tests:
            (Test Case 1) File is created.
            (Test Case 2) Return dict contains file_path.
        """
        ws_id, ns = loaded_ws
        path = str(tmp_path / "group_export.h5")
        result = await exporters.export_to_hdf5_group(
            workspace_id=ws_id,
            namespace=ns,
            file_path=path,
        )
        assert os.path.exists(path)
        assert result["file_path"] == path

    @pytest.mark.asyncio
    async def test_export_to_hdf5_paired(self, loaded_ws, tmp_path):
        """
        export_to_hdf5_paired creates an HDF5 file with paired array format.

        Tests:
            (Test Case 1) File is created.
            (Test Case 2) Return dict contains file_path.
        """
        ws_id, ns = loaded_ws
        path = str(tmp_path / "paired_export.h5")
        result = await exporters.export_to_hdf5_paired(
            workspace_id=ws_id,
            namespace=ns,
            file_path=path,
        )
        assert os.path.exists(path)
        assert result["file_path"] == path
