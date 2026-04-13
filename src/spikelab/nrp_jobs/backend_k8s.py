"""Kubernetes backend for batch job submission and monitoring."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

import yaml

try:
    from kubernetes import client, config, watch
except Exception:  # pragma: no cover
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
            config.load_kube_config(config_file=kubeconfig)
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()

    def _run_kubectl(self, args: List[str]) -> str:
        command = ["kubectl"]
        if self.kubeconfig:
            command.extend(["--kubeconfig", self.kubeconfig])
        command.extend(args)
        out = subprocess.run(command, check=True, text=True, capture_output=True)
        return out.stdout.strip()

    def apply_manifest(self, manifest_path_or_str: str) -> str:
        """Apply a job manifest by YAML file path or raw YAML string."""
        if self._batch_api is None:
            if not self.use_kubectl_fallback:
                raise RuntimeError(
                    "Kubernetes client unavailable and kubectl fallback disabled"
                )
            path = Path(manifest_path_or_str)
            if path.exists():
                return self._run_kubectl(
                    ["apply", "-f", str(path), "-n", self.namespace]
                )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", encoding="utf-8", delete=False
            ) as f:
                f.write(manifest_path_or_str)
                temp_path = f.name
            return self._run_kubectl(["apply", "-f", temp_path, "-n", self.namespace])

        path = Path(manifest_path_or_str)
        if path.exists():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(manifest_path_or_str)
        self._batch_api.create_namespaced_job(namespace=self.namespace, body=payload)
        return payload["metadata"]["name"]

    def delete_job(self, name: str) -> None:
        """Delete a job and its pods."""
        if self._batch_api is None:
            self._run_kubectl(
                ["delete", "job", name, "-n", self.namespace, "--ignore-not-found=true"]
            )
            return
        self._batch_api.delete_namespaced_job(
            name=name,
            namespace=self.namespace,
            propagation_policy="Background",
        )

    def job_status(self, name: str) -> str:
        """Return one of Pending/Running/Complete/Failed/Unknown."""
        if self._batch_api is None:
            out = self._run_kubectl(
                ["get", "job", name, "-n", self.namespace, "-o", "yaml"]
            )
            payload = yaml.safe_load(out)
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
            payload = yaml.safe_load(out)
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

        text = self._core_api.read_namespaced_pod_log(
            name=pod_name, namespace=self.namespace
        )
        for line in text.splitlines():
            yield line
