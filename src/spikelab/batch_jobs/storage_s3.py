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
        if not self.prefix:
            raise ValueError(
                "S3 prefix is not configured. Set it in the profile or command."
            )
        template = getattr(self._templates, category, self._templates.inputs)
        return template.format(prefix=self.prefix, run_id=run_id, filename=filename)

    def upload_file(self, *, local_path: str, s3_uri: str) -> str:
        bucket, key = parse_s3_url(s3_uri)
        self._client.upload_file(local_path, bucket, key)
        return s3_uri

    def upload_bundle(self, *, local_zip: str, run_id: str) -> str:
        filename = Path(local_zip).name
        uri = self.build_uri(run_id=run_id, filename=filename, category="inputs")
        return self.upload_file(local_path=local_zip, s3_uri=uri)

    def output_prefix_for_run(self, run_id: str) -> str:
        if not self.prefix:
            return ""
        return self._templates.outputs.format(
            prefix=self.prefix, run_id=run_id, filename=""
        )

    def logs_prefix_for_run(self, run_id: str) -> str:
        if not self.prefix:
            return ""
        return self._templates.logs.format(
            prefix=self.prefix, run_id=run_id, filename=""
        )
