"""High-level run orchestration for packaging, uploading, and job submission."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from ..data_loaders.data_exporters import export_spikedata_to_nwb, export_to_pickle
from .artifact_packager import package_analysis_bundle
from .backend_k8s import KubernetesBatchJobBackend
from .credentials import ResolvedCredentials, resolve_credentials
from .models import ClusterProfile, JobSpec
from .policy_nrp import evaluate_nrp_policy, summarize_preflight
from .storage_s3 import S3StorageClient
from .templating import build_template_context, render_job_manifest


@dataclass
class SubmitResult:
    job_name: str
    manifest_yaml: str
    uploaded_input_uri: str
    output_prefix: str
    logs_prefix: str


class RunSession:
    """Coordinates artifact packaging + job submission lifecycle."""

    def __init__(
        self,
        *,
        profile: ClusterProfile,
        backend: KubernetesBatchJobBackend,
        storage_client: S3StorageClient,
        credentials: Optional[ResolvedCredentials] = None,
    ) -> None:
        self.profile = profile
        self.backend = backend
        self.storage = storage_client
        self.credentials = credentials or resolve_credentials()

    @staticmethod
    def _build_job_name(prefix: str) -> str:
        token = uuid4().hex[:8]
        return f"{prefix}-{token}"[:63]

    def _materialize_outputs(
        self,
        *,
        spikelab_object: Any,
        output_format: str,
        run_id: str,
        work_dir: str,
    ) -> list[str]:
        outputs: list[str] = []
        if output_format in {"pickle", "both"}:
            pickle_path = str(Path(work_dir) / f"{run_id}.pkl")
            outputs.append(export_to_pickle(spikelab_object, pickle_path))
        if output_format in {"nwb", "both"}:
            nwb_path = str(Path(work_dir) / f"{run_id}.nwb")
            export_spikedata_to_nwb(spikelab_object, nwb_path)
            outputs.append(nwb_path)
        return outputs

    def render_manifest(self, *, job_name: str, job_spec: JobSpec, run_id: str) -> str:
        context = build_template_context(
            job_name=job_name,
            job_spec=job_spec,
            profile=self.profile,
            extra_labels={"run_id": run_id},
        )
        return render_job_manifest(context)

    def submit_trial(
        self,
        *,
        spikelab_object: Any,
        job_spec: JobSpec,
        output_format: str = "pickle",
        allow_policy_risk: bool = False,
        bundle_input_paths: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SubmitResult:
        """Package outputs, upload bundle, and submit Kubernetes job."""
        findings = evaluate_nrp_policy(job_spec)
        status, summary = summarize_preflight(findings)
        if status == "BLOCK" and not allow_policy_risk:
            raise RuntimeError(
                f"NRP preflight blocked submission. Re-run with override if intentional.\n{summary}"
            )

        run_id = uuid4().hex
        job_name = self._build_job_name(job_spec.name_prefix)
        with tempfile.TemporaryDirectory(prefix=f"{run_id}-session-") as temp_dir:
            generated = self._materialize_outputs(
                spikelab_object=spikelab_object,
                output_format=output_format,
                run_id=run_id,
                work_dir=temp_dir,
            )
            input_files = [*generated, *(bundle_input_paths or [])]
            bundle_zip = package_analysis_bundle(
                input_paths=input_files,
                run_id=run_id,
                output_dir=temp_dir,
                output_format=output_format,  # type: ignore[arg-type]
                metadata=metadata,
            )
            uploaded_input_uri = self.storage.upload_bundle(
                local_zip=bundle_zip, run_id=run_id
            )
            manifest_text = self.render_manifest(
                job_name=job_name, job_spec=job_spec, run_id=run_id
            )
            manifest_path = Path(temp_dir) / f"{job_name}.yaml"
            manifest_path.write_text(manifest_text, encoding="utf-8")
            self.backend.apply_manifest(str(manifest_path))

            return SubmitResult(
                job_name=job_name,
                manifest_yaml=manifest_text,
                uploaded_input_uri=uploaded_input_uri,
                output_prefix=self.storage.output_prefix_for_run(run_id),
                logs_prefix=self.storage.logs_prefix_for_run(run_id),
            )

    def submit_prepared_job(
        self,
        *,
        job_spec: JobSpec,
        run_id: Optional[str] = None,
        allow_policy_risk: bool = False,
    ) -> SubmitResult:
        """Submit a job without generating bundle artifacts."""
        findings = evaluate_nrp_policy(job_spec)
        status, summary = summarize_preflight(findings)
        if status == "BLOCK" and not allow_policy_risk:
            raise RuntimeError(f"NRP preflight blocked submission.\n{summary}")

        current_run_id = run_id or uuid4().hex
        job_name = self._build_job_name(job_spec.name_prefix)
        manifest_text = self.render_manifest(
            job_name=job_name,
            job_spec=job_spec,
            run_id=current_run_id,
        )
        self.backend.apply_manifest(manifest_text)
        return SubmitResult(
            job_name=job_name,
            manifest_yaml=manifest_text,
            uploaded_input_uri="",
            output_prefix=self.storage.output_prefix_for_run(current_run_id),
            logs_prefix=self.storage.logs_prefix_for_run(current_run_id),
        )

    def wait_for_completion(
        self,
        *,
        job_name: str,
        max_wait_seconds: int = 3600,
        poll_interval_seconds: int = 10,
    ) -> str:
        """Poll until completion/failure or timeout and return final state."""
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            state = self.backend.job_status(job_name)
            if state in {"Complete", "Failed"}:
                return state
            time.sleep(poll_interval_seconds)
        return "Timeout"
