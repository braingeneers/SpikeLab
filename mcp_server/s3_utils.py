"""
Utilities for handling S3 file downloads.

Supports downloading files from S3 URLs (both s3:// and https:// formats)
to temporary local files for processing.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception


def is_s3_url(url: str) -> bool:
    """
    Check if a URL is an S3 URL.

    Args:
        url: URL string to check

    Returns:
        True if the URL is an S3 URL, False otherwise
    """
    if url.startswith("s3://"):
        return True
    # Check for https://s3... URLs
    if url.startswith("https://"):
        parsed = urlparse(url)
        # Match patterns like s3.amazonaws.com, s3.region.amazonaws.com, etc.
        if "s3" in parsed.netloc and "amazonaws.com" in parsed.netloc:
            return True
    return False


def parse_s3_url(url: str) -> Tuple[str, str]:
    """
    Parse an S3 URL to extract bucket and key.

    Supports both s3://bucket/key and https://s3.../bucket/key formats.

    Args:
        url: S3 URL to parse

    Returns:
        Tuple of (bucket, key)

    Raises:
        ValueError: If the URL is not a valid S3 URL
    """
    if url.startswith("s3://"):
        # Remove s3:// prefix
        path = url[5:]
        parts = path.split("/", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    # Handle https:// URLs
    if url.startswith("https://"):
        parsed = urlparse(url)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) < 1:
            raise ValueError(f"Invalid S3 URL format: {url}")
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ""
        return bucket, key

    raise ValueError(f"Not an S3 URL: {url}")


def download_from_s3(
    url: str,
    local_path: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> str:
    """
    Download a file from S3 to a local temporary file.

    Args:
        url: S3 URL (s3://bucket/key or https://s3.../bucket/key)
        local_path: Optional local path to save the file. If None, creates a temp file.
        aws_access_key_id: Optional AWS access key ID
        aws_secret_access_key: Optional AWS secret access key
        aws_session_token: Optional AWS session token (for temporary credentials)
        region_name: Optional AWS region name

    Returns:
        Path to the downloaded local file

    Raises:
        ImportError: If boto3 is not installed
        ValueError: If the URL is not a valid S3 URL
        ClientError: If there's an error accessing S3
        NoCredentialsError: If AWS credentials are not available
    """
    if boto3 is None:
        raise ImportError(
            "boto3 is required for S3 downloads. Install it with: pip install boto3"
        )

    if not is_s3_url(url):
        raise ValueError(f"Not an S3 URL: {url}")

    bucket, key = parse_s3_url(url)

    # Create S3 client with credentials if provided
    s3_kwargs = {}
    if aws_access_key_id:
        s3_kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        s3_kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        s3_kwargs["aws_session_token"] = aws_session_token
    if region_name:
        s3_kwargs["region_name"] = region_name

    s3_client = boto3.client("s3", **s3_kwargs)

    # Determine local file path
    if local_path is None:
        # Create temporary file
        suffix = Path(key).suffix if key else ".tmp"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        local_path = temp_file.name
        temp_file.close()

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else ".", exist_ok=True)

    try:
        # Download the file
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchBucket":
            raise ValueError(f"S3 bucket not found: {bucket}") from e
        elif error_code == "NoSuchKey":
            raise ValueError(f"S3 key not found: {key} in bucket {bucket}") from e
        elif error_code == "403":
            raise PermissionError(f"Access denied to s3://{bucket}/{key}") from e
        else:
            raise RuntimeError(f"Error downloading from S3: {e}") from e
    except NoCredentialsError as e:
        raise RuntimeError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables or configure AWS credentials."
        ) from e


def ensure_local_file(
    file_path_or_url: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Ensure a file path or S3 URL is available as a local file.

    If the input is an S3 URL, downloads it to a temporary file.
    If it's already a local path, returns it as-is.

    Args:
        file_path_or_url: Local file path or S3 URL
        aws_access_key_id: Optional AWS access key ID
        aws_secret_access_key: Optional AWS secret access key
        aws_session_token: Optional AWS session token
        region_name: Optional AWS region name

    Returns:
        Tuple of (local_file_path, is_temporary) where is_temporary indicates
        if the file should be cleaned up after use
    """
    if is_s3_url(file_path_or_url):
        local_path = download_from_s3(
            file_path_or_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )
        return local_path, True

    # It's a local file path
    if not os.path.exists(file_path_or_url):
        raise FileNotFoundError(f"File not found: {file_path_or_url}")
    return file_path_or_url, False

