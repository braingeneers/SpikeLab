"""
Coverage gap tests for data_loaders module.
"""

import pytest
import numpy as np
import os
import pathlib
from unittest import mock
from data_loaders import s3_utils, data_loaders, data_exporters

def test_is_s3_url():
    assert s3_utils.is_s3_url("s3://bucket/key") is True
    assert s3_utils.is_s3_url("https://mybucket.s3.amazonaws.com/mykey") is True
    assert s3_utils.is_s3_url("http://s3.amazonaws.com/bucket/key") is True
    assert s3_utils.is_s3_url("/local/path") is False

def test_parse_s3_url_success():
    # s3://
    assert s3_utils.parse_s3_url("s3://bucket/key/path") == ("bucket", "key/path")
    
    # Path-style: https://s3.amazonaws.com/bucket/key
    assert s3_utils.parse_s3_url("https://s3.amazonaws.com/mybucket/mykey") == ("mybucket", "mykey")
    
    # Virtual-hosted-style: https://bucket.s3.amazonaws.com/key
    assert s3_utils.parse_s3_url("https://mybucket.s3.amazonaws.com/mykey") == ("mybucket", "mykey")

def test_parse_s3_url_errors():
    # Line 67: s3://bucket (no key)
    assert s3_utils.parse_s3_url("s3://bucket") == ("bucket", "")
    
    # Line 79: Path-style without bucket
    with pytest.raises(ValueError, match="Invalid S3 URL format"):
        s3_utils.parse_s3_url("https://s3.amazonaws.com/")
    
    # Line 90: Invalid HTTPS but not S3
    with pytest.raises(ValueError, match="Invalid S3 URL format"):
        s3_utils.parse_s3_url("https://example.com/foo")
        
    # Line 92: Not a URL at all
    with pytest.raises(ValueError, match="Not an S3 URL"):
        s3_utils.parse_s3_url("just_a_string")

def test_download_from_s3_no_boto3():
    # Line 105: boto3 is None
    with mock.patch("data_loaders.s3_utils.boto3", None):
        with pytest.raises(ImportError, match="boto3 is required"):
            s3_utils.download_from_s3("s3://bucket/key")

def test_download_from_s3_invalid_url():
    # Line 110: not is_s3_url
    with pytest.raises(ValueError, match="Not an S3 URL"):
        s3_utils.download_from_s3("/local/file")

def test_download_from_s3_credentials_params():
    # Lines 116, 118, 120, 122
    mock_client = mock.MagicMock()
    with mock.patch("boto3.client", return_value=mock_client) as mock_boto:
        s3_utils.download_from_s3(
            "s3://bucket/key",
            aws_access_key_id="acc",
            aws_secret_access_key="sec",
            aws_session_token="tok",
            region_name="reg"
        )
        mock_boto.assert_called_with(
            "s3",
            aws_access_key_id="acc",
            aws_secret_access_key="sec",
            aws_session_token="tok",
            region_name="reg"
        )

def test_download_from_s3_tempfile_logic(tmp_path):
    # Lines 126-132: local_path is None
    mock_client = mock.MagicMock()
    with mock.patch("boto3.client", return_value=mock_client):
        # We need to mock download_file to not actually do anything
        path = s3_utils.download_from_s3("s3://bucket/key.txt")
        assert ".txt" in path
        assert os.path.exists(path)
        os.remove(path)

def test_download_from_s3_error_handling():
    # Lines 140-150: ClientError and NoCredentialsError
    mock_client = mock.MagicMock()
    
    # NoSuchBucket
    error_response = {"Error": {"Code": "NoSuchBucket"}}
    mock_client.download_file.side_effect = s3_utils.ClientError(error_response, "DownloadFile")
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(ValueError, match="S3 bucket not found"):
            s3_utils.download_from_s3("s3://bucket/key")
            
    # NoSuchKey
    error_response = {"Error": {"Code": "NoSuchKey"}}
    mock_client.download_file.side_effect = s3_utils.ClientError(error_response, "DownloadFile")
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(ValueError, match="S3 key not found"):
            s3_utils.download_from_s3("s3://bucket/key")
            
    # 403
    error_response = {"Error": {"Code": "403"}}
    mock_client.download_file.side_effect = s3_utils.ClientError(error_response, "DownloadFile")
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(PermissionError, match="Access denied"):
            s3_utils.download_from_s3("s3://bucket/key")

    # Generic RuntimeError
    error_response = {"Error": {"Code": "Unknown"}}
    mock_client.download_file.side_effect = s3_utils.ClientError(error_response, "DownloadFile")
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Error downloading from S3"):
            s3_utils.download_from_s3("s3://bucket/key")

    # NoCredentialsError
    from botocore.exceptions import NoCredentialsError
    mock_client.download_file.side_effect = NoCredentialsError()
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="AWS credentials not found"):
            s3_utils.download_from_s3("s3://bucket/key")

def test_ensure_local_file_s3():
    # Lines 165-172
    # Also coverage for download_from_s3 with local_path provided (Line 126->132 jump)
    with mock.patch("boto3.client") as mock_boto:
        with mock.patch("data_loaders.s3_utils.download_from_s3", wraps=s3_utils.download_from_s3) as mock_dl:
            path, is_temp = s3_utils.ensure_local_file("s3://bucket/key", aws_access_key_id="foo")
            assert is_temp is True
            mock_dl.assert_called()

def test_download_from_s3_with_local_path(tmp_path):
    # Coverage for Line 126->132 (local_path is NOT None)
    my_path = str(tmp_path / "manual_save.txt")
    with mock.patch("boto3.client"):
        path = s3_utils.download_from_s3("s3://bucket/key", local_path=my_path)
        assert path == my_path

def test_ensure_local_file_errors():
    # Line 175: FileNotFoundError
    with pytest.raises(FileNotFoundError):
        s3_utils.ensure_local_file("/non/existent/file")
