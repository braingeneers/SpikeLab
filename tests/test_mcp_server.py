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
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Check for MCP dependencies - import in stages
MCP_IMPORT_ERROR = None
MCP_AVAILABLE = False
MCP_SERVER_AVAILABLE = False

# Basic imports (no mcp dependency)
try:
    from data_loaders.s3_utils import (
        download_from_s3,
        ensure_local_file,
        is_s3_url,
        parse_s3_url,
        upload_to_s3,
    )
    from spikedata import SpikeData

    MCP_AVAILABLE = True
except ImportError as e:
    MCP_IMPORT_ERROR = str(e)

# Server imports (requires mcp package)
if MCP_AVAILABLE:
    try:
        from mcp_server.server import server
        from mcp_server.tools import analysis, data_loaders, exporters
        from workspace.workspace import get_workspace_manager

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
    @patch("data_loaders.s3_utils.boto3")
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
    @patch("data_loaders.s3_utils.boto3")
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
    @patch("data_loaders.s3_utils.boto3")
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
    @patch("data_loaders.s3_utils.boto3")
    def test_upload_to_s3_credential_error(self, mock_boto3):
        """
        Test that upload_to_s3 raises RuntimeError when AWS credentials are missing.

        Tests:
        (Method 1) Creates temp file
        (Method 2) Mocks upload_file to raise NoCredentialsError (lazy cred check on request)
        (Test Case 1) RuntimeError is raised with credentials message
        """
        try:
            from botocore.exceptions import NoCredentialsError
        except ImportError:
            NoCredentialsError = (
                Exception  # s3_utils falls back to Exception when botocore missing
            )

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
    @patch("mcp_server.tools.data_loaders.ensure_local")
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


# ============================================================================
# Server Integration Tests
# ============================================================================


class TestServerIntegration:
    """Test server integration and tool registration."""

    @pytestmark_server
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that tools are registered."""
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
    async def test_workspace_tools_registered(self):
        """
        Test that all workspace management tools are registered.

        Tests:
            (Test Case 1) create_workspace is registered.
            (Test Case 2) list_workspaces is registered.
            (Test Case 3) fetch_workspace_item is registered.
            (Test Case 4) _from_workspace analysis tools are registered.
        """
        from mcp_server.server import _list_tools

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
        from mcp_server.server import _list_tools

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
        from mcp_server.server import _call_tool

        result = await _call_tool("unknown_tool", {})
        data = json.loads(result[0].text)
        assert "error" in data
