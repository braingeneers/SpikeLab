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
import pathlib
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Check for MCP dependencies - import in stages
MCP_IMPORT_ERROR = None
MCP_AVAILABLE = False
MCP_SERVER_AVAILABLE = False

# Basic imports (no mcp dependency)
try:
    from SpikeLab.data_loaders.s3_utils import (
        download_from_s3,
        ensure_local_file,
        is_s3_url,
        parse_s3_url,
        upload_to_s3,
    )
    from SpikeLab.spikedata import SpikeData

    MCP_AVAILABLE = True
except ImportError as e:
    MCP_IMPORT_ERROR = str(e)

# Server imports (requires mcp package)
if MCP_AVAILABLE:
    try:
        from SpikeLab.mcp_server.server import server
        from SpikeLab.mcp_server.tools import (
            analysis,
            data_loaders,
            exporters,
        )
        from SpikeLab.workspace.workspace import get_workspace_manager

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
    @patch("SpikeLab.data_loaders.s3_utils.boto3")
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
    @patch("SpikeLab.data_loaders.s3_utils.boto3")
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
    @patch("SpikeLab.data_loaders.s3_utils.boto3")
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
    @patch("SpikeLab.data_loaders.s3_utils.boto3")
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
    @patch("SpikeLab.mcp_server.tools.data_loaders.ensure_local_file")
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
            result = await data_loaders.load_from_hdf5(
                "s3://bucket/data.h5",
                style="ragged",
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
        with pytest.raises(ValueError, match="fetch_workspace_item supports"):
            await analysis.fetch_workspace_item(ws_id, "ns", "obj")


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
        assert result["info_corr"]["type"] == "ndarray"
        assert result["info_corr"]["shape"] == [3, 3]
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
            result = await exporters.export_to_hdf5(
                ws_id, ns, tmp_path, style="ragged", spike_times_unit="s"
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
        with patch(
            "SpikeLab.data_loaders.s3_utils.upload_to_s3"
        ) as mock_upload:
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
        from SpikeLab.mcp_server.server import _list_tools

        tools = await _list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

        tool_names = [tool.name for tool in tools]
        assert "load_from_nwb" in tool_names
        assert "compute_rates" in tool_names
        assert "export_to_hdf5" in tool_names

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
        from SpikeLab.mcp_server.server import _list_tools

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
        from SpikeLab.mcp_server.server import _list_tools

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
        from SpikeLab.mcp_server.server import _list_tools

        tools = await _list_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            assert tool.inputSchema["type"] == "object"

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("SpikeLab.mcp_server.server.analysis.compute_rates")
    async def test_call_tool(self, mock_compute):
        """Test calling a tool through the server."""
        from SpikeLab.mcp_server.server import _call_tool

        mock_compute.return_value = {
            "rates": [0.1, 0.2, 0.3],
            "unit": "kHz",
            "num_neurons": 3,
        }

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
        mock_compute.assert_called_once()

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test error handling for unknown tool."""
        from SpikeLab.mcp_server.server import _call_tool

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
    from SpikeLab.spikedata.spikeslicestack import SpikeSliceStack

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
    from SpikeLab.spikedata.rateslicestack import RateSliceStack

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
        from SpikeLab.spikedata.pairwise import PairwiseCompMatrix

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
        from SpikeLab.spikedata.pairwise import PairwiseCompMatrixStack

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
    async def test_compute_sparse_raster(self, loaded_ws):
        """
        Test compute_sparse_raster stores a dense raster.

        Tests:
            (Test Case 1) Stored item is ndarray.
        """
        ws_id, ns = loaded_ws
        result = await analysis.compute_sparse_raster(ws_id, ns, "sparse", bin_size=5.0)
        assert result["key"] == "sparse"
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
        assert result["info"]["type"] == "ndarray"
        assert result["info"]["shape"] == [3, 3]

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
        assert result["info"]["type"] == "ndarray"
        assert result["info"]["shape"] == [3, 3]

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
    @patch("SpikeLab.mcp_server.tools.analysis.SpikeData.get_bursts")
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
    @patch("SpikeLab.mcp_server.tools.analysis.SpikeData.burst_sensitivity")
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
    @patch("SpikeLab.mcp_server.tools.analysis.SpikeData.get_frac_active")
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
    @patch("SpikeLab.mcp_server.tools.analysis.SpikeData.get_waveform_traces")
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
        result = await analysis.get_waveform_traces(
            ws_id, ns, key="wf_unit0", unit=0
        )
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
    @patch("SpikeLab.mcp_server.tools.analysis.SpikeData.fit_gplvm")
    async def test_fit_gplvm(self, mock_fit, loaded_ws):
        """
        Test fit_gplvm stores decode_res, reorder_indices, and binned_spike_counts.

        Tests:
            (Test Case 1) Three keys are stored in workspace.
            (Test Case 2) Return dict includes log_marginal_l and bin_size_ms.
        """
        n_time, n_units = 10, 3
        mock_fit.return_value = {
            "decode_res": {"posterior_latent_marg": np.random.default_rng(0).random((n_time, 5))},
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
            await analysis.umap_reduction(
                ws_id, ns, key="nonexistent", out_key="out"
            )

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
            await analysis.umap_reduction(
                ws_id, ns, key="arr1d", out_key="out"
            )


class TestLoaderMCPToolsCoverage:
    """Tests for untested loader MCP tools."""

    @pytestmark_server
    @pytest.mark.asyncio
    @patch("SpikeLab.mcp_server.tools.data_loaders.load_spikedata_from_hdf5_raw_thresholded")
    @patch("SpikeLab.mcp_server.tools.data_loaders.ensure_local_file")
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
    @patch("SpikeLab.mcp_server.tools.data_loaders.load_spikedata_from_ibl")
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
    @patch("SpikeLab.mcp_server.tools.data_loaders._query_ibl_probes")
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

        from SpikeLab.workspace.workspace import AnalysisWorkspace

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
