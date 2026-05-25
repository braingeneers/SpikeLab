"""Kubernetes backend for batch job submission and monitoring."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator, List, Optional

import yaml

try:
    from kubernetes import client, config, watch
except ImportError:  # pragma: no cover
    client = None
    config = None
    watch = None


class KubernetesBatchJobBackend:
    """Backend wrapper around Kubernetes client with kubectl fallback."""

    def __init__(
        self,
        namespace: str = "default",
        kubeconfig: Optional[str] = None,
        use_kubectl_fallback: bool = True,
    ) -> None:
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.use_kubectl_fallback = use_kubectl_fallback
        self._batch_api = None
        self._core_api = None
        if client is not None and config is not None:
            try:
                config.load_kube_config(config_file=kubeconfig)
                self._batch_api = client.BatchV1Api()
                self._core_api = client.CoreV1Api()
            except config.ConfigException:
                pass  # No valid kubeconfig — fall back to kubectl

    def _run_kubectl(self, args: List[str]) -> str:
        command = ["kubectl"]
        if self.kubeconfig:
            command.extend(["--kubeconfig", self.kubeconfig])
        command.extend(args)
        out = subprocess.run(command, check=True, text=True, capture_output=True)
        return out.stdout.strip()

    def apply_manifest(self, manifest_path_or_str: str) -> str:
        """Apply a job manifest by YAML file path or raw YAML string.

        Returns the job's ``metadata.name`` (consistent across both
        paths). Previously the kubectl-fallback path returned the
        raw stdout of ``kubectl apply`` (e.g. ``"job.batch/myjob
        created\\n"``) while the Python-client path returned the
        clean name — callers had no portable way to extract the
        identifier without sniffing the backend.

        Raises ``ValueError`` if the manifest's ``metadata.namespace``
        is set and disagrees with the backend's ``self.namespace`` —
        this would otherwise silently deploy into the backend's
        namespace, contrary to the rendered manifest. Manifests with
        no ``metadata.namespace`` are accepted and assigned the
        backend's namespace as before.
        """

        def _check_manifest_namespace(payload: Any) -> None:
            try:
                manifest_ns = (payload or {}).get("metadata", {}).get("namespace")
            except AttributeError:
                manifest_ns = None
            if manifest_ns and manifest_ns != self.namespace:
                raise ValueError(
                    f"Manifest metadata.namespace={manifest_ns!r} disagrees "
                    f"with backend namespace={self.namespace!r}. Refusing "
                    "to deploy a manifest into a different namespace than "
                    "the rendered one."
                )

        def _extract_job_name(payload: Any) -> Optional[str]:
            """Best-effort extraction of metadata.name from a parsed manifest.

            Returns ``None`` when the structure isn't a dict-of-dicts;
            callers should fall back to the kubectl stdout (only the
            Python-client path requires the name to be present).
            """
            if not isinstance(payload, dict):
                return None
            meta = payload.get("metadata")
            if not isinstance(meta, dict):
                return None
            name = meta.get("name")
            return name if isinstance(name, str) and name else None

        path = Path(manifest_path_or_str)

        # Resolve the kubectl-fallback path first when the Python client
        # is unavailable. The fallback uses ``kubectl apply`` directly,
        # which itself rejects manifests missing ``metadata.name``, so
        # we only do a best-effort namespace check here without raising
        # on missing-name. The namespace check still runs on whatever
        # YAML we can parse; an unparseable payload skips the check.
        if self._batch_api is None:
            if not self.use_kubectl_fallback:
                raise RuntimeError(
                    "Kubernetes client unavailable and kubectl fallback disabled"
                )
            try:
                if path.exists():
                    payload_for_check = yaml.safe_load(path.read_text(encoding="utf-8"))
                else:
                    payload_for_check = yaml.safe_load(manifest_path_or_str)
            except yaml.YAMLError:
                payload_for_check = None
            if payload_for_check is not None:
                _check_manifest_namespace(payload_for_check)
            fallback_job_name = _extract_job_name(payload_for_check)

            if path.exists():
                stdout = self._run_kubectl(
                    ["apply", "-f", str(path), "-n", self.namespace]
                )
                return fallback_job_name or stdout
            temp_path = None
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", encoding="utf-8", delete=False
            ) as f:
                f.write(manifest_path_or_str)
                temp_path = f.name
            try:
                stdout = self._run_kubectl(
                    ["apply", "-f", temp_path, "-n", self.namespace]
                )
                return fallback_job_name or stdout
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

        # Python-client path: parse strictly. ``create_namespaced_job``
        # requires a structured payload with metadata.name, so we raise
        # early with a clear message rather than letting the API client
        # surface a less actionable error.
        if path.exists():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(manifest_path_or_str)
        _check_manifest_namespace(payload)
        job_name = _extract_job_name(payload)
        if job_name is None:
            raise ValueError(
                "Manifest does not contain metadata.name; cannot apply via "
                "the Kubernetes Python client."
            )
        self._batch_api.create_namespaced_job(namespace=self.namespace, body=payload)
        return job_name

    def delete_job(self, name: str) -> None:
        """Delete a job and its pods. Idempotent: missing jobs are a no-op.

        Matches the ``kubectl --ignore-not-found=true`` semantic on
        the fallback path so the two delete paths behave the same
        way for the missing-job case. Previously the Python
        kubernetes-client path propagated ``ApiException(404)``
        verbatim while the kubectl path exited cleanly.
        """
        if self._batch_api is None:
            self._run_kubectl(
                ["delete", "job", name, "-n", self.namespace, "--ignore-not-found=true"]
            )
            return
        try:
            self._batch_api.delete_namespaced_job(
                name=name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy="Background"),
            )
        except client.exceptions.ApiException as exc:
            if exc.status == 404:
                # Missing job — idempotent no-op, matches kubectl
                # ``--ignore-not-found`` behaviour. Any other API
                # error (403 Forbidden, 500 Server Error, etc.)
                # still propagates.
                return
            raise

    def job_status(self, name: str) -> str:
        """Return one of Pending/Running/Complete/Failed/Unknown."""
        if self._batch_api is None:
            out = self._run_kubectl(
                ["get", "job", name, "-n", self.namespace, "-o", "yaml"]
            )
            # ``yaml.safe_load("")`` returns ``None``; guard so transient empty
            # stdout (kubectl warming up, race during job restart, etc.) does
            # not raise AttributeError and silently break monitoring loops.
            payload = yaml.safe_load(out) or {}
            status = payload.get("status", {})
        else:
            status_obj = self._batch_api.read_namespaced_job_status(
                name, self.namespace
            )
            status = (
                status_obj.status.to_dict() if status_obj and status_obj.status else {}
            )

        if status.get("failed"):
            return "Failed"
        if status.get("succeeded"):
            return "Complete"
        if status.get("active"):
            return "Running"
        return "Pending"

    def pods_for_job(self, job_name: str) -> List[str]:
        """Return pod names associated with a job."""
        selector = f"job-name={job_name}"
        if self._core_api is None:
            out = self._run_kubectl(
                ["get", "pods", "-n", self.namespace, "-l", selector, "-o", "yaml"]
            )
            # Guard empty kubectl stdout — see ``job_status`` for rationale.
            payload = yaml.safe_load(out) or {}
            return [item["metadata"]["name"] for item in payload.get("items", [])]

        pods = self._core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=selector,
        )
        return [item.metadata.name for item in pods.items]

    def stream_logs(self, pod_name: str, follow: bool = True) -> Iterator[str]:
        """Yield log lines from a pod."""
        if self._core_api is None:
            args = ["logs", pod_name, "-n", self.namespace]
            if follow:
                args.append("-f")
            process = subprocess.Popen(
                ["kubectl", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                yield line.rstrip("\n")
            return

        if follow and watch is not None:
            watcher = watch.Watch()
            for line in watcher.stream(
                self._core_api.read_namespaced_pod_log,
                name=pod_name,
                namespace=self.namespace,
                follow=True,
            ):
                yield str(line)
            return

        # Stream the (non-follow) log via ``_preload_content=False``
        # so multi-GB pod logs don't materialise as a single string in
        # memory. The kubernetes client returns an
        # ``urllib3.response.HTTPResponse`` in this mode which exposes
        # ``stream(chunk_size)`` for chunked reading. We buffer
        # partial lines across chunks and yield only when we see a
        # newline (or at EOF), matching the line-at-a-time contract
        # used by the kubectl-fallback path above.
        response = self._core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=self.namespace,
            _preload_content=False,
        )
        try:
            buf = ""
            for chunk in response.stream(amt=64 * 1024, decode_content=True):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="replace")
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield line
            if buf:
                yield buf
        finally:
            response.release_conn()
