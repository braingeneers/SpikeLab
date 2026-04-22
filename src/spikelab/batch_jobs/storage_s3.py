"""S3-compatible storage helpers for batch job artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

from ..data_loaders.s3_utils import parse_s3_url
from .models import StoragePathTemplates


class S3StorageClient:
    """Small wrapper around boto3 for upload/download URI handling.

    Path layout is controlled by *path_templates* (a
    :class:`StoragePathTemplates` instance loaded from the active profile).
    """

    def __init__(
        self,
        *,
        prefix: Optional[str] = None,
        path_templates: Optional[StoragePathTemplates] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ) -> None:
        self.prefix = (
            (prefix if prefix.endswith("/") else f"{prefix}/") if prefix else None
        )
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self._templates = path_templates or StoragePathTemplates()
        if boto3 is None:
            raise ImportError("boto3 is required for S3 storage: pip install boto3")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    def build_uri(self, *, run_id: str, filename: str, category: str = "inputs") -> str:
        """Build an S3 URI for a file using the active path templates."""
        if not self.prefix:
            raise ValueError(
                "S3 prefix is not configured. Set it in the profile or command."
            )
        template = getattr(self._templates, category, self._templates.inputs)
        return template.format(prefix=self.prefix, run_id=run_id, filename=filename)

    def upload_file(self, *, local_path: str, s3_uri: str) -> str:
        """Upload a local file to S3 and return the URI."""
        bucket, key = parse_s3_url(s3_uri)
        self._client.upload_file(local_path, bucket, key)
        return s3_uri

    def upload_bundle(self, *, local_zip: str, run_id: str) -> str:
        """Upload a zip bundle to S3 under the inputs path template."""
        filename = Path(local_zip).name
        uri = self.build_uri(run_id=run_id, filename=filename, category="inputs")
        return self.upload_file(local_path=local_zip, s3_uri=uri)

    def output_prefix_for_run(self, run_id: str) -> str:
        """Return the S3 prefix for a run's output files."""
        if not self.prefix:
            return ""
        return self._templates.outputs.format(
            prefix=self.prefix, run_id=run_id, filename=""
        )

    def logs_prefix_for_run(self, run_id: str) -> str:
        """Return the S3 prefix for a run's log files."""
        if not self.prefix:
            return ""
        return self._templates.logs.format(
            prefix=self.prefix, run_id=run_id, filename=""
        )

    def download_file(self, *, s3_uri: str, local_path: str) -> str:
        """Download a single file from S3.

        Parameters:
            s3_uri (str): Full ``s3://bucket/key`` URI.
            local_path (str): Destination path on disk.

        Returns:
            local_path (str): The same *local_path* for convenience.
        """
        bucket, key = parse_s3_url(s3_uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(bucket, key, local_path)
        return local_path

    def download_output(self, *, run_id: str, filename: str, local_dir: str) -> str:
        """Download a file from the output prefix of a run.

        Parameters:
            run_id (str): Run identifier.
            filename (str): Name of the file within the output prefix.
            local_dir (str): Local directory to save the file into.

        Returns:
            local_path (str): Absolute path of the downloaded file.
        """
        prefix = self.output_prefix_for_run(run_id)
        s3_uri = prefix + filename
        local_path = str(Path(local_dir) / filename)
        return self.download_file(s3_uri=s3_uri, local_path=local_path)

    def list_output_files(self, run_id: str) -> list:
        """List object keys under the output prefix of a run.

        Parameters:
            run_id (str): Run identifier.

        Returns:
            keys (list[str]): S3 object keys found under the output prefix.
        """
        prefix = self.output_prefix_for_run(run_id)
        if not prefix:
            return []
        bucket, key_prefix = parse_s3_url(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
