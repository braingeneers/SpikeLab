"""
Utilities for handling S3-backed inputs.

These helpers support:
- Detecting S3 URLs (`s3://...` and common `https://...amazonaws.com/...` forms)
- Parsing bucket/key pairs from S3 URLs
- Downloading S3 objects to local temporary files for downstream processing
- Treating local paths and S3 URLs uniformly (`ensure_local_file`)

This module intentionally has **no** dependency on the MCP server implementation
so it can be reused by the core analysis package and other integrations.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:  # pragma: no cover
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

__all__ = [
    "is_s3_url",
    "parse_s3_url",
    "download_from_s3",
    "upload_to_s3",
    "ensure_local_file",
]


def is_s3_url(url: str) -> bool:
    """Return True if `url` looks like an S3 URL (s3:// or https://...amazonaws.com)."""
    if url.startswith("s3://"):
        return True
    if url.startswith("https://") or url.startswith("http://"):
        parsed = urlparse(url)
        # Matches path-style and virtual-hosted-style S3 endpoints:
        # - s3.amazonaws.com
        # - s3.<region>.amazonaws.com
        # - <bucket>.s3.amazonaws.com
        # - <bucket>.s3.<region>.amazonaws.com
        return "s3" in parsed.netloc and "amazonaws.com" in parsed.netloc
    return False


def parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse an S3 URL into (bucket, key).

    Supported forms:
    - s3://bucket/key
    - https://s3.amazonaws.com/bucket/key
    - https://s3.<region>.amazonaws.com/bucket/key
    - https://<bucket>.s3.amazonaws.com/key
    - https://<bucket>.s3.<region>.amazonaws.com/key
    """
    if url.startswith("s3://"):
        path = url[5:]
        parts = path.split("/", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    if url.startswith("https://") or url.startswith("http://"):
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path.lstrip("/")

        # Path-style: https://s3.../bucket/key
        if host.startswith("s3") and "amazonaws.com" in host:
            parts = path.split("/", 1)
            if not parts or parts[0] == "":
                raise ValueError(f"Invalid S3 URL format: {url}")
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return bucket, key

        # Virtual-hosted-style: https://bucket.s3.../key
        if ".s3" in host and "amazonaws.com" in host:
            bucket = host.split(".s3", 1)[0]
            key = path
            return bucket, key

        raise ValueError(f"Invalid S3 URL format: {url}")

    raise ValueError(f"Not an S3 URL: {url}")


def download_from_s3(
    url: str,
    local_path: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> str:
    """Download a single S3 object to a local file and return the local path."""
    if boto3 is None:
        raise ImportError(
            "boto3 is required for S3 downloads. Install it with: pip install boto3"
        )

    if not is_s3_url(url):
        raise ValueError(f"Not an S3 URL: {url}")

    bucket, key = parse_s3_url(url)

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

    if local_path is None:
        suffix = Path(key).suffix if key else ".tmp"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        local_path = temp_file.name
        temp_file.close()

    os.makedirs(
        os.path.dirname(local_path) if os.path.dirname(local_path) else ".",
        exist_ok=True,
    )

    try:
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except ClientError as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "NoSuchBucket":
            raise ValueError(f"S3 bucket not found: {bucket}") from e
        if error_code == "NoSuchKey":
            raise ValueError(f"S3 key not found: {key} in bucket {bucket}") from e
        if error_code in ("AccessDenied", "Forbidden"):
            raise PermissionError(f"Access denied to s3://{bucket}/{key}") from e
        raise RuntimeError(f"Error downloading from S3: {e}") from e
    except NoCredentialsError as e:
        raise RuntimeError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables or configure AWS credentials."
        ) from e


def upload_to_s3(
    local_path: str,
    s3_url: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> str:
    """Upload a local file to S3 and return the S3 URL."""
    if not is_s3_url(s3_url):
        raise ValueError(f"Not an S3 URL: {s3_url}")

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if boto3 is None:
        raise ImportError(
            "boto3 is required for S3 uploads. Install it with: pip install boto3"
        )

    bucket, key = parse_s3_url(s3_url)

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

    try:
        s3_client.upload_file(local_path, bucket, key)
        return s3_url
    except ClientError as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "NoSuchBucket":
            raise ValueError(f"S3 bucket not found: {bucket}") from e
        if error_code in ("AccessDenied", "Forbidden"):
            raise PermissionError(f"Access denied to s3://{bucket}/{key}") from e
        raise RuntimeError(f"Error uploading to S3: {e}") from e
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
    """Return (local_path, is_temporary) for a local path or S3 URL."""
    if is_s3_url(file_path_or_url):
        local_path = download_from_s3(
            file_path_or_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )
        return local_path, True

    if not os.path.exists(file_path_or_url):
        raise FileNotFoundError(f"File not found: {file_path_or_url}")
    return file_path_or_url, False
