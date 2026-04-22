"""High-level run orchestration for packaging, uploading, and job submission."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
from uuid import uuid4

from .artifact_packager import package_analysis_bundle
from .backend_k8s import KubernetesBatchJobBackend
from .credentials import ResolvedCredentials, resolve_credentials
from .models import ClusterProfile, JobSpec, SubmitResult
from .policy import evaluate_policy, summarize_preflight
from .storage_s3 import S3StorageClient
from .templating import build_template_context, render_job_manifest

# Workspace path convention: save(base) produces base.h5 + base.json
_WORKSPACE_BASE_NAME = "workspace"


class RunSession:
    """Coordinates artifact packaging, job submission, and result retrieval."""

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_job_name(prefix: str) -> str:
        token = uuid4().hex[:8]
        max_prefix = 63 - 1 - len(token)  # 54
        truncated = prefix[:max_prefix].rstrip("-")
        return f"{truncated}-{token}"

    def _preflight(self, job_spec: JobSpec, allow_policy_risk: bool) -> None:
        """Run policy checks and raise on BLOCK unless overridden."""
        findings = evaluate_policy(job_spec, self.profile)
        status, summary = summarize_preflight(findings)
        if status == "BLOCK" and not allow_policy_risk:
            raise RuntimeError(
                "Policy preflight blocked submission. "
                f"Re-run with allow_policy_risk=True if intentional.\n{summary}"
            )

    def render_manifest(self, *, job_name: str, job_spec: JobSpec, run_id: str) -> str:
        """Render a Kubernetes Job manifest from a spec and profile."""
        context = build_template_context(
            job_name=job_name,
            job_spec=job_spec,
            profile=self.profile,
            extra_labels={"run_id": run_id},
        )
        return render_job_manifest(context)

    def _submit(
        self,
        *,
        job_spec: JobSpec,
        run_id: str,
        uploaded_input_uri: str,
        job_type: str,
    ) -> SubmitResult:
        """Render manifest, apply to cluster, return result."""
        job_name = self._build_job_name(job_spec.name_prefix)
        manifest_text = self.render_manifest(
            job_name=job_name, job_spec=job_spec, run_id=run_id
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"{job_name}-",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(manifest_text)
            manifest_path = f.name
        try:
            self.backend.apply_manifest(manifest_path)
        finally:
            os.unlink(manifest_path)

        return SubmitResult(
            job_name=job_name,
            manifest_yaml=manifest_text,
            run_id=run_id,
            uploaded_input_uri=uploaded_input_uri,
            output_prefix=self.storage.output_prefix_for_run(run_id),
            logs_prefix=self.storage.logs_prefix_for_run(run_id),
            job_type=job_type,
        )

    @staticmethod
    def _inject_env(job_spec: JobSpec, env: Dict[str, str]) -> JobSpec:
        """Return a copy of *job_spec* with additional env vars on the container."""
        merged = dict(job_spec.container.env)
        merged.update(env)
        updated_container = job_spec.container.model_copy(update={"env": merged})
        return job_spec.model_copy(update={"container": updated_container})

    # ------------------------------------------------------------------
    # Submission: workspace job
    # ------------------------------------------------------------------

    def submit_workspace_job(
        self,
        *,
        workspace: Any,
        script: str,
        job_spec: JobSpec,
        allow_policy_risk: bool = False,
        bundle_input_paths: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SubmitResult:
        """Save a workspace, bundle it with a script, and submit a job.

        Parameters:
            workspace: An ``AnalysisWorkspace`` instance or a ``str``
                path to an existing workspace base path (without
                extension).
            script (str): Path to the analysis script to run inside the
                container.
            job_spec (JobSpec): Kubernetes job specification.
            allow_policy_risk (bool): Override policy BLOCK findings.
            bundle_input_paths (iterable[str] | None): Extra files to
                include in the bundle.
            metadata (dict | None): Arbitrary metadata written into the
                bundle manifest.

        Returns:
            result (SubmitResult): Submission details including the
                output prefix where the updated workspace will appear.
        """
        self._preflight(job_spec, allow_policy_risk)

        run_id = uuid4().hex
        with tempfile.TemporaryDirectory(prefix=f"{run_id}-session-") as temp_dir:
            # Resolve workspace to .h5 + .json on disk
            workspace_base = self._save_workspace(workspace, temp_dir)

            script_path = Path(script)
            if not script_path.exists():
                raise FileNotFoundError(f"Analysis script not found: {script_path}")

            input_files = [
                f"{workspace_base}.h5",
                f"{workspace_base}.json",
                str(script_path),
                *(bundle_input_paths or []),
            ]

            bundle_zip = package_analysis_bundle(
                input_paths=input_files,
                run_id=run_id,
                output_dir=temp_dir,
                output_format="workspace",
                metadata=metadata,
            )

            uploaded_input_uri = self.storage.upload_bundle(
                local_zip=bundle_zip, run_id=run_id
            )

            enriched_spec = self._inject_env(
                job_spec,
                {
                    "INPUT_URI": uploaded_input_uri,
                    "OUTPUT_PREFIX": self.storage.output_prefix_for_run(run_id),
                    "SCRIPT_NAME": script_path.name,
                },
            )
            # Set container command to the workspace entrypoint
            enriched_spec = enriched_spec.model_copy(
                update={
                    "container": enriched_spec.container.model_copy(
                        update={
                            "command": [
                                "python",
                                "-m",
                                "spikelab.batch_jobs.entrypoints.workspace",
                            ],
                        }
                    )
                }
            )

            return self._submit(
                job_spec=enriched_spec,
                run_id=run_id,
                uploaded_input_uri=uploaded_input_uri,
                job_type="workspace",
            )

    # ------------------------------------------------------------------
    # Submission: sorting job
    # ------------------------------------------------------------------

    def submit_sorting_job(
        self,
        *,
        recording_paths: list,
        config: Any = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        job_spec: JobSpec,
        allow_policy_risk: bool = False,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SubmitResult:
        """Bundle recording files with a sorting config and submit a job.

        Parameters:
            recording_paths (list[str]): Paths to recording files.
            config: A ``SortingPipelineConfig`` instance, a preset name
                string (e.g. ``"kilosort4"``), or None for defaults.
            config_overrides (dict | None): Flat keyword overrides
                applied to the config via ``config.override()``.
            job_spec (JobSpec): Kubernetes job specification.
            allow_policy_risk (bool): Override policy BLOCK findings.
            metadata (dict | None): Arbitrary metadata written into the
                bundle manifest.

        Returns:
            result (SubmitResult): Submission details including the
                output prefix where sorted results will appear.
        """
        self._preflight(job_spec, allow_policy_risk)

        run_id = uuid4().hex
        with tempfile.TemporaryDirectory(prefix=f"{run_id}-session-") as temp_dir:
            config_dict = self._resolve_sorting_config(config, config_overrides)
            config_path = Path(temp_dir) / "sorting_config.json"
            config_path.write_text(
                json.dumps(config_dict, indent=2, default=str), encoding="utf-8"
            )

            # Validate recording paths
            for rpath in recording_paths:
                if not Path(rpath).exists():
                    raise FileNotFoundError(f"Recording file not found: {rpath}")

            input_files = [
                str(config_path),
                *[str(p) for p in recording_paths],
            ]

            bundle_zip = package_analysis_bundle(
                input_paths=input_files,
                run_id=run_id,
                output_dir=temp_dir,
                output_format="sorting",
                metadata=metadata,
            )

            uploaded_input_uri = self.storage.upload_bundle(
                local_zip=bundle_zip, run_id=run_id
            )

            enriched_spec = self._inject_env(
                job_spec,
                {
                    "INPUT_URI": uploaded_input_uri,
                    "OUTPUT_PREFIX": self.storage.output_prefix_for_run(run_id),
                },
            )
            enriched_spec = enriched_spec.model_copy(
                update={
                    "container": enriched_spec.container.model_copy(
                        update={
                            "command": [
                                "python",
                                "-m",
                                "spikelab.batch_jobs.entrypoints.sorting",
                            ],
                        }
                    )
                }
            )

            return self._submit(
                job_spec=enriched_spec,
                run_id=run_id,
                uploaded_input_uri=uploaded_input_uri,
                job_type="sorting",
            )

    # ------------------------------------------------------------------
    # Submission: prepared job (no bundling)
    # ------------------------------------------------------------------

    def submit_prepared_job(
        self,
        *,
        job_spec: JobSpec,
        run_id: Optional[str] = None,
        allow_policy_risk: bool = False,
    ) -> SubmitResult:
        """Submit a job without generating bundle artifacts."""
        self._preflight(job_spec, allow_policy_risk)
        current_run_id = run_id or uuid4().hex
        return self._submit(
            job_spec=job_spec,
            run_id=current_run_id,
            uploaded_input_uri="",
            job_type="prepared",
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_result(
        self,
        submit_result: SubmitResult,
        local_dir: str,
    ) -> Any:
        """Download job outputs and return an AnalysisWorkspace.

        Parameters:
            submit_result (SubmitResult): The result from a prior
                ``submit_workspace_job`` or ``submit_sorting_job`` call.
            local_dir (str): Local directory to download outputs into.

        Returns:
            workspace (AnalysisWorkspace): The workspace produced by the
                job. For workspace jobs this is the updated workspace;
                for sorting jobs it contains per-recording namespaces
                with SpikeData at key ``"spikedata"``.

        Notes:
            - Call ``wait_for_completion`` before calling this method to
              ensure the job has finished.
        """
        from ..workspace.workspace import AnalysisWorkspace

        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)

        if submit_result.job_type == "workspace":
            return self._retrieve_workspace(submit_result, local)
        elif submit_result.job_type == "sorting":
            return self._retrieve_sorting(submit_result, local)
        else:
            raise ValueError(
                f"Cannot retrieve results for job_type={submit_result.job_type!r}. "
                "Only 'workspace' and 'sorting' jobs produce retrievable outputs."
            )

    def _retrieve_workspace(
        self, result: SubmitResult, local_dir: Path
    ) -> Any:
        """Download workspace .h5 + .json and load."""
        from ..workspace.workspace import AnalysisWorkspace

        h5_name = f"{_WORKSPACE_BASE_NAME}.h5"
        json_name = f"{_WORKSPACE_BASE_NAME}.json"

        self.storage.download_output(
            run_id=result.run_id, filename=h5_name, local_dir=str(local_dir)
        )
        self.storage.download_output(
            run_id=result.run_id, filename=json_name, local_dir=str(local_dir)
        )

        base_path = str(local_dir / _WORKSPACE_BASE_NAME)
        return AnalysisWorkspace.load(base_path)

    def _retrieve_sorting(
        self, result: SubmitResult, local_dir: Path
    ) -> Any:
        """Download all sorting outputs, build workspace from pickles."""
        from ..data_loaders.data_loaders import load_spikedata_from_pickle
        from ..workspace.workspace import AnalysisWorkspace

        # Download everything under the output prefix
        keys = self.storage.list_output_files(result.run_id)
        if not keys:
            raise FileNotFoundError(
                f"No output files found for run_id={result.run_id}"
            )

        from ..data_loaders.s3_utils import parse_s3_url

        prefix = result.output_prefix
        bucket, prefix_key = parse_s3_url(prefix)

        downloaded = []
        for key in keys:
            # Derive relative path from prefix
            relative = key[len(prefix_key) :] if key.startswith(prefix_key) else key
            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_uri = f"s3://{bucket}/{key}"
            self.storage.download_file(s3_uri=s3_uri, local_path=str(local_path))
            downloaded.append((relative, str(local_path)))

        # Build workspace from downloaded SpikeData pickles
        ws = AnalysisWorkspace(name=f"sorting-{result.run_id[:8]}")

        for relative, local_path in downloaded:
            if local_path.endswith(".pkl"):
                try:
                    sd = load_spikedata_from_pickle(local_path)
                except Exception:
                    continue
                namespace = Path(relative).stem
                ws.store(namespace, "spikedata", sd)
            elif local_path.endswith(".json") and "config" not in relative:
                # Store sorting metadata
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    namespace = Path(relative).stem
                    ws.store(namespace, "sorting_metadata", meta)
                except Exception:
                    continue

        return ws

    # ------------------------------------------------------------------
    # Wait
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_workspace(workspace: Any, work_dir: str) -> str:
        """Ensure workspace is saved to disk, return the base path.

        Parameters:
            workspace: ``AnalysisWorkspace`` or ``str`` base path.
            work_dir (str): Directory to save into if workspace is an
                object.

        Returns:
            base_path (str): Path without extension.
        """
        if isinstance(workspace, str):
            # Assume it's a base path; verify .h5 exists
            if not Path(f"{workspace}.h5").exists():
                raise FileNotFoundError(
                    f"Workspace file not found: {workspace}.h5"
                )
            return workspace

        # It's an AnalysisWorkspace object
        base_path = str(Path(work_dir) / _WORKSPACE_BASE_NAME)
        workspace.save(base_path)
        return base_path

    @staticmethod
    def _resolve_sorting_config(
        config: Any, overrides: Optional[Dict[str, Any]]
    ) -> dict:
        """Resolve a sorting config to a serializable dict.

        Parameters:
            config: ``SortingPipelineConfig``, preset name string, or
                None.
            overrides (dict | None): Flat keyword overrides.

        Returns:
            config_dict (dict): JSON-serializable nested dict.
        """
        from ..spike_sorting.config import SortingPipelineConfig

        if config is None:
            resolved = SortingPipelineConfig()
        elif isinstance(config, str):
            # Treat as preset name
            import spikelab.spike_sorting.config as cfg_module

            preset = getattr(cfg_module, config.upper(), None)
            if preset is None:
                raise ValueError(
                    f"Unknown sorting preset: {config!r}. Available: "
                    "KILOSORT2, KILOSORT4, KILOSORT2_DOCKER, KILOSORT4_DOCKER, "
                    "RT_SORT_MEA, RT_SORT_NEUROPIXELS"
                )
            resolved = preset
        else:
            resolved = config

        if overrides:
            resolved = resolved.override(**overrides)

        return dataclasses.asdict(resolved)
