"""
Result store for large numpy array outputs from analysis tools.

Three backends are available, selected via the RESULT_BACKEND environment variable:

    RESULT_BACKEND=memory  (default) — in-process dict with TTL; lost on restart.
    RESULT_BACKEND=disk             — .npy files in a temp dir; survives restarts.
    RESULT_BACKEND=s3               — objects in an S3 bucket; works across remote hosts.

All backends implement the same interface (store / get / get_info / delete /
cleanup_expired) so no tool wrapper code changes when the backend is swapped.

Environment variables
---------------------
RESULT_BACKEND            : "memory" | "disk" | "s3"  (default: "memory")
RESULT_TTL_SECONDS        : integer TTL in seconds     (default: 3600)
RESULT_STORE_DIR          : disk backend only — directory for .npy files
RESULT_STORE_S3_BUCKET    : s3 backend only  — S3 bucket name (required)
RESULT_STORE_S3_PREFIX    : s3 backend only  — key prefix     (default: "iat-results/")
AWS_ACCESS_KEY_ID         : s3 backend only  — optional AWS credential
AWS_SECRET_ACCESS_KEY     : s3 backend only  — optional AWS credential
AWS_SESSION_TOKEN         : s3 backend only  — optional AWS credential
AWS_DEFAULT_REGION        : s3 backend only  — optional AWS region
"""

import io
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class ResultStore:
    """
    In-memory store for large numpy array analysis results.

    Results are keyed by a UUID and expire after a configurable TTL.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        """
        Initialize the in-memory result store.

        Parameters:
            default_ttl_seconds (int): Default time-to-live in seconds (default: 1 hour).
        """
        self._results: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds

    def store(
        self,
        array: np.ndarray,
        ttl_seconds: Optional[int] = None,
        session_ids: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a numpy array and return a result ID.

        Parameters:
            array (np.ndarray): The array to store.
            ttl_seconds (int, optional): Time-to-live in seconds. Uses default if None.
            session_ids (list[str], optional): Spike data session IDs to associate with
                this result. Supports multi-source results derived from more than one session.
            extra_meta (dict, optional): Additional JSON-serializable metadata to attach
                (e.g. times and step_size for RateSliceStack reconstruction).

        Returns:
            result_id (str): UUID string identifying this result.
        """
        result_id = str(uuid.uuid4())
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        entry: Dict[str, Any] = {
            "array": array,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "session_ids": session_ids,
        }
        if extra_meta is not None:
            entry["extra_meta"] = extra_meta
        self._results[result_id] = entry
        return result_id

    def get(
        self, result_id: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a stored array and its metadata.

        Parameters:
            result_id (str): The result ID returned by store().

        Returns:
            (array, metadata) tuple if found and not expired, None otherwise.
        """
        if result_id not in self._results:
            return None

        entry = self._results[result_id]
        if time.time() > entry["expires_at"]:
            del self._results[result_id]
            return None

        meta = {
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "created_at": entry["created_at"],
            "expires_at": entry["expires_at"],
            "session_ids": entry.get("session_ids"),
            "extra_meta": entry.get("extra_meta"),
        }
        return entry["array"], meta

    def get_info(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a stored result without returning the array.

        Parameters:
            result_id (str): The result ID.

        Returns:
            info (dict): shape, dtype, created_at, expires_at, session_ids, or None if not found.
        """
        if result_id not in self._results:
            return None

        entry = self._results[result_id]
        if time.time() > entry["expires_at"]:
            del self._results[result_id]
            return None

        return {
            "result_id": result_id,
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "created_at": entry["created_at"],
            "expires_at": entry["expires_at"],
            "session_ids": entry.get("session_ids"),
        }

    def list_by_session(self, session_id: str) -> list:
        """
        List all active results associated with a spike data session.

        Parameters:
            session_id (str): The spike data session ID.

        Returns:
            results (list[dict]): List of get_info() dicts for matching results.
        """
        self.cleanup_expired()
        return [
            self.get_info(rid)
            for rid, entry in list(self._results.items())
            if session_id in (entry.get("session_ids") or [])
        ]

    def delete(self, result_id: str) -> bool:
        """
        Delete a stored result.

        Parameters:
            result_id (str): The result ID to delete.

        Returns:
            deleted (bool): True if deleted, False if not found.
        """
        if result_id in self._results:
            del self._results[result_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired results.

        Returns:
            count (int): Number of results removed.
        """
        now = time.time()
        expired = [
            rid
            for rid, entry in self._results.items()
            if now > entry["expires_at"]
        ]
        for rid in expired:
            del self._results[rid]
        return len(expired)


# ---------------------------------------------------------------------------
# Disk backend
# ---------------------------------------------------------------------------


class DiskResultStore:
    """
    Disk-backed result store.

    Each result is saved as a pair of files in store_dir:
      <result_id>.npy  — the numpy array
      <result_id>.json — shape, dtype, created_at, expires_at metadata

    Survives server restarts. Files for expired results are removed lazily on
    get() and eagerly by cleanup_expired().
    """

    def __init__(
        self,
        store_dir: Optional[str] = None,
        default_ttl_seconds: int = 3600,
    ):
        """
        Initialize the disk result store.

        Parameters:
            store_dir (str, optional): Directory for result files. A temp
                directory is created automatically if not provided.
            default_ttl_seconds (int): Default TTL in seconds (default: 1 hour).
        """
        self.store_dir = store_dir or tempfile.mkdtemp(prefix="iat_results_")
        os.makedirs(self.store_dir, exist_ok=True)
        self.default_ttl = default_ttl_seconds

    def _npy_path(self, result_id: str) -> str:
        return os.path.join(self.store_dir, f"{result_id}.npy")

    def _meta_path(self, result_id: str) -> str:
        return os.path.join(self.store_dir, f"{result_id}.json")

    def store(
        self,
        array: np.ndarray,
        ttl_seconds: Optional[int] = None,
        session_ids: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a numpy array to disk and return a result ID.

        Parameters:
            array (np.ndarray): The array to store.
            ttl_seconds (int, optional): Time-to-live in seconds. Uses default if None.
            session_ids (list[str], optional): Spike data session IDs to associate with
                this result. Supports multi-source results derived from more than one session.
            extra_meta (dict, optional): Additional JSON-serializable metadata to attach.

        Returns:
            result_id (str): UUID string identifying this result.
        """
        result_id = str(uuid.uuid4())
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        np.save(self._npy_path(result_id), array)
        meta: Dict[str, Any] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "session_ids": session_ids,
        }
        if extra_meta is not None:
            meta["extra_meta"] = extra_meta
        with open(self._meta_path(result_id), "w") as f:
            json.dump(meta, f)
        return result_id

    def get(
        self, result_id: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Load a stored array from disk and return it with its metadata.

        Parameters:
            result_id (str): The result ID returned by store().

        Returns:
            (array, metadata) tuple if found and not expired, None otherwise.
        """
        meta_path = self._meta_path(result_id)
        npy_path = self._npy_path(result_id)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        if time.time() > meta["expires_at"]:
            self.delete(result_id)
            return None
        array = np.load(npy_path)
        return array, dict(meta)

    def get_info(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Read metadata for a stored result without loading the array.

        Parameters:
            result_id (str): The result ID.

        Returns:
            info (dict): shape, dtype, created_at, expires_at, session_ids, or None if not found.
        """
        meta_path = self._meta_path(result_id)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        if time.time() > meta["expires_at"]:
            self.delete(result_id)
            return None
        return {"result_id": result_id, **meta}

    def list_by_session(self, session_id: str) -> list:
        """
        List all active results on disk associated with a spike data session.

        Parameters:
            session_id (str): The spike data session ID.

        Returns:
            results (list[dict]): List of get_info() dicts for matching results.
        """
        results = []
        now = time.time()
        for filename in os.listdir(self.store_dir):
            if not filename.endswith(".json"):
                continue
            result_id = filename[:-5]
            meta_path = self._meta_path(result_id)
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if now > meta["expires_at"]:
                    continue
                if session_id in (meta.get("session_ids") or []):
                    results.append({"result_id": result_id, **meta})
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        return results

    def delete(self, result_id: str) -> bool:
        """
        Delete the .npy and .json files for a result.

        Parameters:
            result_id (str): The result ID to delete.

        Returns:
            deleted (bool): True if deleted, False if not found.
        """
        meta_path = self._meta_path(result_id)
        if not os.path.exists(meta_path):
            return False
        for path in (self._npy_path(result_id), meta_path):
            if os.path.exists(path):
                os.remove(path)
        return True

    def cleanup_expired(self) -> int:
        """
        Scan the store directory and delete all expired results.

        Returns:
            count (int): Number of results removed.
        """
        count = 0
        now = time.time()
        for filename in os.listdir(self.store_dir):
            if not filename.endswith(".json"):
                continue
            result_id = filename[:-5]
            meta_path = self._meta_path(result_id)
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if now > meta["expires_at"]:
                    self.delete(result_id)
                    count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        return count


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------


class S3ResultStore:
    """
    S3-backed result store.

    Each result is stored as a single .npy object in an S3 bucket. Shape, dtype,
    and TTL metadata are stored as S3 object metadata fields so that get_info()
    does not require downloading the array body.

    Requires boto3 (optional dependency — install with ``pip install boto3``).

    Notes:
        - cleanup_expired() lists all objects under the prefix and issues a
          HEAD request per object to read metadata. This can be expensive for
          large buckets; consider using S3 Lifecycle rules for bulk expiry.
        - For remote-hosted deployments, configure AWS credentials via env vars
          (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
          AWS_DEFAULT_REGION) rather than constructor arguments so that no
          credentials appear in MCP tool arguments.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "iat-results/",
        default_ttl_seconds: int = 3600,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """
        Initialize the S3 result store.

        Parameters:
            bucket (str): S3 bucket name.
            prefix (str): Key prefix for all result objects (default: "iat-results/").
            default_ttl_seconds (int): Default TTL in seconds (default: 1 hour).
            aws_access_key_id (str, optional): AWS access key.
            aws_secret_access_key (str, optional): AWS secret key.
            aws_session_token (str, optional): AWS session token.
            region_name (str, optional): AWS region.
        """
        try:
            import boto3
            import botocore.exceptions as _bce

            self._botocore_exceptions = _bce
        except ImportError:
            raise ImportError(
                "boto3 is required for S3ResultStore. "
                "Install with 'pip install boto3'."
            )

        self.bucket = bucket
        self.prefix = prefix
        self.default_ttl = default_ttl_seconds
        import boto3

        self._s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )

    def _key(self, result_id: str) -> str:
        return f"{self.prefix}{result_id}.npy"

    def _parse_meta(self, raw: Dict[str, str]) -> Dict[str, Any]:
        raw_session_ids = raw.get("session_ids")
        session_ids = json.loads(raw_session_ids) if raw_session_ids else None
        raw_extra_meta = raw.get("extra_meta")
        extra_meta = json.loads(raw_extra_meta) if raw_extra_meta else None
        return {
            "shape": json.loads(raw.get("shape", "[]")),
            "dtype": raw.get("dtype", ""),
            "created_at": float(raw.get("created_at", 0)),
            "expires_at": float(raw.get("expires_at", 0)),
            "session_ids": session_ids,
            "extra_meta": extra_meta,
        }

    def store(
        self,
        array: np.ndarray,
        ttl_seconds: Optional[int] = None,
        session_ids: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload a numpy array to S3 and return a result ID.

        Parameters:
            array (np.ndarray): The array to store.
            ttl_seconds (int, optional): Time-to-live in seconds. Uses default if None.
            session_ids (list[str], optional): Spike data session IDs to associate with
                this result. Supports multi-source results derived from more than one session.
            extra_meta (dict, optional): Additional JSON-serializable metadata to attach.
                Stored as a single JSON-encoded S3 metadata field.

        Returns:
            result_id (str): UUID string identifying this result.
        """
        result_id = str(uuid.uuid4())
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        buf = io.BytesIO()
        np.save(buf, array)
        s3_meta: Dict[str, str] = {
            "shape": json.dumps(list(array.shape)),
            "dtype": str(array.dtype),
            "created_at": str(time.time()),
            "expires_at": str(time.time() + ttl),
        }
        if session_ids is not None:
            s3_meta["session_ids"] = json.dumps(session_ids)
        if extra_meta is not None:
            s3_meta["extra_meta"] = json.dumps(extra_meta)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self._key(result_id),
            Body=buf.getvalue(),
            Metadata=s3_meta,
        )
        return result_id

    def get(
        self, result_id: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Download a stored array from S3 and return it with its metadata.

        Parameters:
            result_id (str): The result ID returned by store().

        Returns:
            (array, metadata) tuple if found and not expired, None otherwise.
        """
        try:
            response = self._s3.get_object(
                Bucket=self.bucket, Key=self._key(result_id)
            )
        except self._botocore_exceptions.ClientError:
            return None
        meta = self._parse_meta(response.get("Metadata", {}))
        if time.time() > meta["expires_at"]:
            self.delete(result_id)
            return None
        array = np.load(io.BytesIO(response["Body"].read()))
        return array, meta

    def get_info(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for a stored result without downloading the array.

        Parameters:
            result_id (str): The result ID.

        Returns:
            info (dict): shape, dtype, created_at, expires_at, session_ids, or None if not found.
        """
        try:
            response = self._s3.head_object(
                Bucket=self.bucket, Key=self._key(result_id)
            )
        except self._botocore_exceptions.ClientError:
            return None
        meta = self._parse_meta(response.get("Metadata", {}))
        if time.time() > meta["expires_at"]:
            self.delete(result_id)
            return None
        return {"result_id": result_id, **meta}

    def list_by_session(self, session_id: str) -> list:
        """
        List all active results in S3 associated with a spike data session.

        Parameters:
            session_id (str): The spike data session ID.

        Returns:
            results (list[dict]): List of get_info() dicts for matching results.

        Notes:
            - Issues one HEAD request per object to read metadata.
              For large buckets, prefer a dedicated index (e.g. DynamoDB) for efficient filtering.
        """
        results = []
        now = time.time()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                result_id = key[len(self.prefix) : -4]  # strip prefix + ".npy"
                try:
                    response = self._s3.head_object(Bucket=self.bucket, Key=key)
                    meta = self._parse_meta(response.get("Metadata", {}))
                    if now > meta["expires_at"]:
                        continue
                    if session_id in (meta.get("session_ids") or []):
                        results.append({"result_id": result_id, **meta})
                except self._botocore_exceptions.ClientError:
                    pass
        return results

    def delete(self, result_id: str) -> bool:
        """
        Delete a result object from S3.

        Parameters:
            result_id (str): The result ID to delete.

        Returns:
            deleted (bool): True if the delete call succeeded.
        """
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=self._key(result_id))
            return True
        except self._botocore_exceptions.ClientError:
            return False

    def cleanup_expired(self) -> int:
        """
        List all result objects in the bucket prefix and delete expired ones.

        Returns:
            count (int): Number of objects deleted.

        Notes:
            - Issues one HEAD request per object to read expiry metadata.
              For large buckets, prefer S3 Lifecycle rules for bulk expiry.
        """
        count = 0
        now = time.time()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                result_id = key[len(self.prefix) : -4]  # strip prefix + ".npy"
                try:
                    response = self._s3.head_object(Bucket=self.bucket, Key=key)
                    meta = self._parse_meta(response.get("Metadata", {}))
                    if now > meta["expires_at"]:
                        self._s3.delete_object(Bucket=self.bucket, Key=key)
                        count += 1
                except self._botocore_exceptions.ClientError:
                    pass
        return count


# ---------------------------------------------------------------------------
# Factory and singleton
# ---------------------------------------------------------------------------

AnyResultStore = Union[ResultStore, DiskResultStore, S3ResultStore]


def _make_result_store() -> AnyResultStore:
    """
    Instantiate the appropriate backend from environment variables.

    Returns:
        store: A ResultStore, DiskResultStore, or S3ResultStore instance.
    """
    backend = os.environ.get("RESULT_BACKEND", "memory").lower()
    ttl = int(os.environ.get("RESULT_TTL_SECONDS", "3600"))

    if backend == "memory":
        return ResultStore(default_ttl_seconds=ttl)

    if backend == "disk":
        store_dir = os.environ.get("RESULT_STORE_DIR")
        return DiskResultStore(store_dir=store_dir, default_ttl_seconds=ttl)

    if backend == "s3":
        bucket = os.environ.get("RESULT_STORE_S3_BUCKET")
        if not bucket:
            raise ValueError(
                "RESULT_STORE_S3_BUCKET env var is required when RESULT_BACKEND=s3"
            )
        return S3ResultStore(
            bucket=bucket,
            prefix=os.environ.get("RESULT_STORE_S3_PREFIX", "iat-results/"),
            default_ttl_seconds=ttl,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
        )

    raise ValueError(
        f"Unknown RESULT_BACKEND: '{backend}'. "
        "Expected 'memory', 'disk', or 's3'."
    )


# Lazily initialized singleton — created on first call to get_result_store()
_result_store: Optional[AnyResultStore] = None


def get_result_store() -> AnyResultStore:
    """Get the global result store instance, initializing it from env vars on first call."""
    global _result_store
    if _result_store is None:
        _result_store = _make_result_store()
    return _result_store
