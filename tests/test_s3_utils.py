"""
Edge case tests for SpikeLab.data_loaders.s3_utils.

These tests verify boundary behaviors for S3 URL parsing, upload with empty
files, and ensure_local_file with mocked S3 backend.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.data_loaders.s3_utils import (
    parse_s3_url,
    is_s3_url,
    upload_to_s3,
    ensure_local_file,
)


class TestParseS3UrlEdgeCases:
    """Edge case tests for parse_s3_url."""

    def test_ec_s3_01_empty_key_trailing_slash(self):
        """
        EC-S3-01: Verify parse_s3_url with an empty key (s3://bucket/).
        The trailing slash produces an empty string as the key.

        Tests:
            (Test Case 1) bucket is "mybucket".
            (Test Case 2) key is "" (empty string).
        """
        bucket, key = parse_s3_url("s3://mybucket/")
        assert bucket == "mybucket"
        assert key == ""

    def test_ec_s3_01_no_trailing_slash(self):
        """
        EC-S3-01 variant: Verify parse_s3_url with no key and no trailing
        slash (s3://bucket).

        Tests:
            (Test Case 1) bucket is "mybucket".
            (Test Case 2) key is "" (empty string).
        """
        bucket, key = parse_s3_url("s3://mybucket")
        assert bucket == "mybucket"
        assert key == ""

    def test_ec_s3_02_special_characters_in_key(self):
        """
        EC-S3-02: Verify parse_s3_url handles special characters in the key
        (spaces encoded as %20, plus signs, unicode, etc.).

        Tests:
            (Test Case 1) Key with spaces and special chars is preserved as-is.
            (Test Case 2) Key with nested path and dots is preserved.
        """
        bucket, key = parse_s3_url("s3://mybucket/path/with spaces/file+name.h5")
        assert bucket == "mybucket"
        assert key == "path/with spaces/file+name.h5"

        bucket2, key2 = parse_s3_url("s3://mybucket/a/b/c/file.v2.0.tar.gz")
        assert bucket2 == "mybucket"
        assert key2 == "a/b/c/file.v2.0.tar.gz"

    def test_ec_s3_02_percent_encoded_key(self):
        """
        EC-S3-02 variant: Verify percent-encoded characters pass through.

        Tests:
            (Test Case 1) %20 in key is preserved literally.
        """
        bucket, key = parse_s3_url("s3://mybucket/path%20with%20encoding/file.h5")
        assert bucket == "mybucket"
        assert key == "path%20with%20encoding/file.h5"


class TestUploadToS3EdgeCases:
    """Edge case tests for upload_to_s3."""

    def test_ec_s3_03_empty_file(self, tmp_path):
        """
        EC-S3-03: Verify that uploading an empty (0-byte) file to S3 succeeds
        without error. The upload function should not reject empty files.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) upload_file is called on the S3 client.
        """
        empty_file = str(tmp_path / "empty.txt")
        with open(empty_file, "wb") as f:
            pass  # 0 bytes
        assert os.path.getsize(empty_file) == 0

        mock_client = MagicMock()
        with patch("SpikeLab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            result = upload_to_s3(empty_file, "s3://mybucket/empty.txt")

        assert result == "s3://mybucket/empty.txt"
        mock_client.upload_file.assert_called_once_with(
            empty_file, "mybucket", "empty.txt"
        )


class TestEnsureLocalFileEdgeCases:
    """Edge case tests for ensure_local_file."""

    def test_ec_s3_04_s3_url_download_path(self):
        """
        EC-S3-04: Verify that ensure_local_file with an S3 URL calls
        download_from_s3 and returns (local_path, True).

        Tests:
            (Test Case 1) download_from_s3 is called with the S3 URL.
            (Test Case 2) Returns is_temporary=True.
            (Test Case 3) local_path matches the mock return value.
        """
        with patch(
            "SpikeLab.data_loaders.s3_utils.download_from_s3"
        ) as mock_download:
            mock_download.return_value = "/tmp/downloaded_file.h5"
            local_path, is_temp = ensure_local_file(
                "s3://mybucket/data/file.h5",
                aws_access_key_id="AKID",
                aws_secret_access_key="SECRET",
            )

        assert local_path == "/tmp/downloaded_file.h5"
        assert is_temp is True
        mock_download.assert_called_once_with(
            "s3://mybucket/data/file.h5",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            aws_session_token=None,
            region_name=None,
        )

    def test_ensure_local_file_local_path(self, tmp_path):
        """
        Verify that ensure_local_file with a local path returns the path
        directly with is_temporary=False.

        Tests:
            (Test Case 1) Returns the same path.
            (Test Case 2) is_temporary is False.
        """
        path = str(tmp_path / "local.h5")
        with open(path, "wb") as f:
            f.write(b"data")

        local_path, is_temp = ensure_local_file(path)
        assert local_path == path
        assert is_temp is False

    def test_ensure_local_file_missing_local_raises(self):
        """
        Verify that ensure_local_file raises FileNotFoundError for a
        non-existent local path.

        Tests:
            (Test Case 1) FileNotFoundError is raised.
        """
        with pytest.raises(FileNotFoundError):
            ensure_local_file("/nonexistent/path/to/file.h5")
