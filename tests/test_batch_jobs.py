"""Tests for the batch job-launcher package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("pydantic") is None or importlib.util.find_spec("yaml") is None:
    pytest.skip("batch-jobs dependencies not installed", allow_module_level=True)

import yaml

from spikelab.batch_jobs.credentials import redact_sensitive_map
from spikelab.batch_jobs.models import (
    ClusterProfile,
    JobSpec,
    NamespaceHookSpec,
    VolumeMountSpec,
)
from spikelab.batch_jobs.policy import evaluate_policy, summarize_preflight
from spikelab.batch_jobs.templating import build_template_context, render_job_manifest
from spikelab.batch_jobs.validation import validate_job_spec
import spikelab.batch_jobs.cli as cli


def _example_payload():
    return {
        "name_prefix": "analysis-job",
        "namespace": "default",
        "labels": {"analysis": "spikelab"},
        "container": {
            "image": "ghcr.io/example/image:latest",
            "command": ["python"],
            "args": ["-m", "run"],
            "env": {"OUTPUT_PREFIX": "s3://test-bucket/test-prefix/"},
        },
        "resources": {
            "requests_cpu": "2",
            "requests_memory": "8Gi",
            "limits_cpu": "2",
            "limits_memory": "8Gi",
            "requests_gpu": 1,
            "limits_gpu": 1,
            "node_selector": {},
        },
        "volumes": [],
    }


def _profile_with_hooks():
    """Profile with namespace hooks for testing the generic hook engine."""
    return ClusterProfile(
        name="test-cluster",
        namespace_hooks={
            "test-ns": NamespaceHookSpec(
                image_pull_policy="Always",
                default_command=["sh", "-c"],
                required_volumes=[
                    VolumeMountSpec(
                        name="test-secret",
                        mount_path="/etc/test-creds",
                        secret_name="test-secret",
                    ),
                ],
            ),
        },
    )


def test_validate_job_spec():
    job_spec = validate_job_spec(_example_payload())
    assert isinstance(job_spec, JobSpec)
    assert job_spec.container.image.startswith("ghcr.io/")


def test_render_job_manifest_contains_job_name():
    job_spec = validate_job_spec(_example_payload())
    profile = ClusterProfile(name="test")
    context = build_template_context(
        job_name="analysis-job-1234",
        job_spec=job_spec,
        profile=profile,
        extra_labels={"run_id": "abc"},
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    assert "name: analysis-job-1234" in manifest
    assert parsed["kind"] == "Job"
    assert "run_id" in manifest


def test_namespace_hooks_inject_required_mounts():
    """Profile-driven namespace hooks inject volumes for matching namespace."""
    payload = _example_payload()
    payload["namespace"] = "test-ns"
    job_spec = validate_job_spec(payload)
    profile = _profile_with_hooks()
    context = build_template_context(
        job_name="analysis-job-hooks",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    mounts = parsed["spec"]["template"]["spec"]["containers"][0].get("volumeMounts", [])
    mount_paths = {item["mountPath"] for item in mounts}
    assert "/etc/test-creds" in mount_paths
    # image_pull_policy should be overridden by hook
    container = parsed["spec"]["template"]["spec"]["containers"][0]
    assert container["imagePullPolicy"] == "Always"


def test_namespace_hooks_no_match_leaves_manifest_unchanged():
    """Non-matching namespace does not inject hook volumes."""
    payload = _example_payload()
    payload["namespace"] = "other-ns"
    job_spec = validate_job_spec(payload)
    profile = _profile_with_hooks()
    context = build_template_context(
        job_name="analysis-job-no-hook",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    mounts = parsed["spec"]["template"]["spec"]["containers"][0].get("volumeMounts", [])
    mount_paths = {item.get("mountPath") for item in mounts}
    assert "/etc/test-creds" not in mount_paths


def test_namespace_hooks_preserve_user_affinity():
    """Namespace hooks do not override user-specified affinity."""
    payload = _example_payload()
    payload["namespace"] = "test-ns"
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(
        name="test-with-affinity",
        affinity={
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "nvidia.com/gpu.product",
                                    "operator": "In",
                                    "values": ["NVIDIA-A40"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        namespace_hooks={
            "test-ns": NamespaceHookSpec(
                image_pull_policy="Always",
                required_volumes=[
                    VolumeMountSpec(
                        name="cred-vol",
                        mount_path="/etc/creds",
                        secret_name="cred-vol",
                    ),
                ],
            ),
        },
    )
    context = build_template_context(
        job_name="analysis-job-affinity",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    affinity = parsed["spec"]["template"]["spec"]["affinity"]
    values = affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ][0]["matchExpressions"][0]["values"]
    assert values == ["NVIDIA-A40"]


def test_namespace_hooks_inject_env_defaults():
    """Hook env_defaults are merged into the container env."""
    payload = _example_payload()
    payload["namespace"] = "test-ns"
    payload["container"]["env"]["USER_VAR"] = "user-value"
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(
        name="test-env",
        namespace_hooks={
            "test-ns": NamespaceHookSpec(
                env_defaults={
                    "AWS_SHARED_CREDENTIALS_FILE": "/etc/spikelab/aws/credentials",
                    "KUBECONFIG": "/etc/spikelab/kube/config",
                },
            ),
        },
    )
    context = build_template_context(
        job_name="env-hook-test",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    env_list = parsed["spec"]["template"]["spec"]["containers"][0].get("env", [])
    env_map = {item["name"]: item["value"] for item in env_list}
    # Hook defaults present
    assert env_map["AWS_SHARED_CREDENTIALS_FILE"] == "/etc/spikelab/aws/credentials"
    assert env_map["KUBECONFIG"] == "/etc/spikelab/kube/config"
    # User-specified env preserved
    assert env_map["USER_VAR"] == "user-value"


def test_namespace_hooks_env_defaults_user_overrides_hook():
    """User-specified env keys take precedence over hook env_defaults."""
    payload = _example_payload()
    payload["namespace"] = "test-ns"
    payload["container"]["env"]["KUBECONFIG"] = "/my/custom/kubeconfig"
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(
        name="test-env-override",
        namespace_hooks={
            "test-ns": NamespaceHookSpec(
                env_defaults={
                    "KUBECONFIG": "/etc/spikelab/kube/config",
                },
            ),
        },
    )
    context = build_template_context(
        job_name="env-override-test",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    env_list = parsed["spec"]["template"]["spec"]["containers"][0].get("env", [])
    env_map = {item["name"]: item["value"] for item in env_list}
    # User value wins over hook default
    assert env_map["KUBECONFIG"] == "/my/custom/kubeconfig"


from spikelab.batch_jobs.policy import _contains_disallowed_sleep


class TestSleepDetection:
    """Tests for the _contains_disallowed_sleep heuristic."""

    def test_sleep_infinity(self):
        assert _contains_disallowed_sleep(["sleep"], ["infinity"])

    def test_sleep_inf(self):
        assert _contains_disallowed_sleep(["sleep"], ["inf"])

    def test_sleep_infinity_in_sh_c(self):
        assert _contains_disallowed_sleep(["sh", "-c"], ["sleep infinity"])

    def test_bare_sleep(self):
        assert _contains_disallowed_sleep(["sleep"], [])

    def test_sleep_large_number(self):
        assert _contains_disallowed_sleep(["sleep"], ["999999999"])

    def test_sleep_24h(self):
        assert _contains_disallowed_sleep(["sleep"], ["86400"])

    def test_sleep_short_allowed(self):
        assert not _contains_disallowed_sleep(["sleep"], ["60"])

    def test_sleep_23h_allowed(self):
        assert not _contains_disallowed_sleep(["sleep"], ["82800"])

    def test_normal_command_allowed(self):
        assert not _contains_disallowed_sleep(["python"], ["-m", "my_script"])

    def test_sleep_as_substring_allowed(self):
        """'sleep' appearing as part of another word is not flagged."""
        assert not _contains_disallowed_sleep(["python"], ["-m", "sleeper_module"])

    def test_empty_command(self):
        assert not _contains_disallowed_sleep([], [])

    def test_sleep_with_non_numeric_arg(self):
        """sleep with a non-numeric arg (not 'infinity'/'inf') is not flagged."""
        assert not _contains_disallowed_sleep(["sleep"], ["10s"])


def test_policy_blocks_sleep_infinity():
    payload = _example_payload()
    payload["container"]["args"] = ["sleep", "infinity"]
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(name="test")
    findings = evaluate_policy(job_spec, profile)
    level, _ = summarize_preflight(findings)
    assert level == "BLOCK"


def test_policy_uses_profile_thresholds():
    """Policy thresholds come from the profile, not hardcoded values."""
    payload = _example_payload()
    payload["resources"]["requests_gpu"] = 3
    payload["resources"]["limits_gpu"] = 3
    job_spec = validate_job_spec(payload)
    # Default threshold is 2 — should warn
    profile_default = ClusterProfile(name="test")
    findings = evaluate_policy(job_spec, profile_default)
    gpu_finding = [f for f in findings if f.code == "interactive_gpu_limit"][0]
    assert gpu_finding.level == "WARN"
    # Raise threshold to 4 — should pass
    from spikelab.batch_jobs.models import PolicyConfig

    profile_relaxed = ClusterProfile(
        name="test-relaxed",
        policy=PolicyConfig(max_interactive_gpus=4),
    )
    findings_relaxed = evaluate_policy(job_spec, profile_relaxed)
    gpu_finding_relaxed = [
        f for f in findings_relaxed if f.code == "interactive_gpu_limit"
    ][0]
    assert gpu_finding_relaxed.level == "PASS"


def test_redaction_hides_sensitive_fields():
    redacted = redact_sensitive_map(
        {
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "super-secret",
            "AWS_SESSION_TOKEN": "tok-123",
            "DB_PASSWORD": "hunter2",
            "NORMAL_FIELD": "ok",
        }
    )
    assert redacted["AWS_SECRET_ACCESS_KEY"] == "***REDACTED***"
    assert redacted["AWS_SESSION_TOKEN"] == "***REDACTED***"
    assert redacted["DB_PASSWORD"] == "***REDACTED***"
    # Access key ID is the public half — should NOT be redacted
    assert redacted["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
    assert redacted["NORMAL_FIELD"] == "ok"


def test_cli_deploy_prints_job_name(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "job.yaml"
    config_path.write_text(
        """
name_prefix: analysis-job
namespace: default
container:
  image: ghcr.io/example/image:latest
  command: ["python"]
  args: ["-m", "run"]
  env: {}
resources:
  requests_cpu: "1"
  requests_memory: "2Gi"
  limits_cpu: "1"
  limits_memory: "2Gi"
  requests_gpu: 0
  limits_gpu: 0
  node_selector: {}
volumes: []
""".strip(),
        encoding="utf-8",
    )

    class DummySession:
        def submit_prepared_job(self, **kwargs):
            return SimpleNamespace(
                job_name="analysis-job-xyz",
                output_prefix="s3://test-bucket/test-prefix/outputs/run/",
                logs_prefix="s3://test-bucket/test-prefix/logs/run/",
            )

    monkeypatch.setattr(
        cli, "_load_profile", lambda *args, **kwargs: ClusterProfile(name="test")
    )
    monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: DummySession())
    args = SimpleNamespace(
        profile="defaults",
        profile_file=None,
        kubeconfig=None,
        job_config=str(config_path),
        allow_policy_risk=False,
        render_only=False,
        output_manifest=None,
        wait=False,
        max_wait_seconds=0,
        follow_logs=False,
        image_profile=None,
        image=None,
    )
    exit_code = cli._cmd_deploy(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "JOB_NAME=analysis-job-xyz" in out


def test_apply_image_selection_uses_profile_default():
    payload = {
        "container": {
            "command": ["python"],
            "args": ["-m", "run"],
            "env": {},
        }
    }
    profile = ClusterProfile(
        name="test",
        default_images={
            "cpu": "ghcr.io/example/cpu:latest",
            "gpu": "ghcr.io/example/gpu:latest",
        },
    )
    updated = cli._apply_image_selection(
        payload,
        profile=profile,
        image_profile="gpu",
        image_override=None,
    )
    assert updated["container"]["image"] == "ghcr.io/example/gpu:latest"


def test_render_path_applies_image_profile(monkeypatch, tmp_path):
    config_path = tmp_path / "render-job.yaml"
    config_path.write_text(
        """
name_prefix: analysis-job
namespace: default
container:
  command: ["python"]
  args: ["-m", "run"]
  env: {}
resources:
  requests_cpu: "1"
  requests_memory: "2Gi"
  limits_cpu: "1"
  limits_memory: "2Gi"
  requests_gpu: 0
  limits_gpu: 0
  node_selector: {}
volumes: []
""".strip(),
        encoding="utf-8",
    )

    class DummySession:
        def render_manifest(self, *, job_name, job_spec, run_id):
            assert job_spec.container.image == "ghcr.io/example/gpu:latest"
            return f"metadata:\n  name: {job_name}\n"

    monkeypatch.setattr(
        cli,
        "_load_profile",
        lambda *args, **kwargs: ClusterProfile(
            name="test",
            default_images={
                "cpu": "ghcr.io/example/cpu:latest",
                "gpu": "ghcr.io/example/gpu:latest",
            },
        ),
    )
    monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: DummySession())
    args = SimpleNamespace(
        profile="defaults",
        profile_file=None,
        kubeconfig=None,
        job_config=str(config_path),
        allow_policy_risk=False,
        render_only=True,
        output_manifest=None,
        wait=False,
        max_wait_seconds=0,
        follow_logs=False,
        image_profile="gpu",
        image=None,
    )
    exit_code = cli._cmd_render(args)
    assert exit_code == 0


# ---------------------------------------------------------------------------
# artifact_packager tests
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.artifact_packager import package_analysis_bundle, _sha256


class TestArtifactPackager:
    def test_creates_zip_with_manifest(self, tmp_path):
        """Bundle creates a zip containing copied files and manifest.json."""
        input_file = tmp_path / "data.pkl"
        input_file.write_bytes(b"fake pickle data")

        zip_path = package_analysis_bundle(
            input_paths=[str(input_file)],
            run_id="run-001",
            output_dir=str(tmp_path / "out"),
            output_format="pickle",
        )

        assert Path(zip_path).exists()
        assert zip_path.endswith(".zip")

        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "run-001/data.pkl" in names
            assert "run-001/manifest.json" in names

    def test_manifest_contains_sha256_and_metadata(self, tmp_path):
        """manifest.json includes per-file checksums and user metadata."""
        input_file = tmp_path / "result.nwb"
        input_file.write_bytes(b"fake nwb content")

        zip_path = package_analysis_bundle(
            input_paths=[str(input_file)],
            run_id="run-002",
            output_dir=str(tmp_path / "out"),
            output_format="nwb",
            metadata={"workspace_id": "ws-42"},
        )

        import json
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            manifest = json.loads(zf.read("run-002/manifest.json"))

        assert manifest["run_id"] == "run-002"
        assert manifest["output_format"] == "nwb"
        assert manifest["metadata"]["workspace_id"] == "ws-42"
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["name"] == "result.nwb"
        assert len(manifest["files"][0]["sha256"]) == 64  # hex SHA256

    def test_multiple_input_files(self, tmp_path):
        """Multiple input files are all included in the bundle."""
        f1 = tmp_path / "a.pkl"
        f2 = tmp_path / "b.nwb"
        f1.write_bytes(b"data1")
        f2.write_bytes(b"data2")

        zip_path = package_analysis_bundle(
            input_paths=[str(f1), str(f2)],
            run_id="run-multi",
            output_dir=str(tmp_path / "out"),
            output_format="both",
        )

        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "run-multi/a.pkl" in names
            assert "run-multi/b.nwb" in names
            assert "run-multi/manifest.json" in names

    def test_missing_input_file_raises(self, tmp_path):
        """FileNotFoundError raised when an input path does not exist."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            package_analysis_bundle(
                input_paths=["/nonexistent/file.pkl"],
                run_id="run-bad",
                output_dir=str(tmp_path / "out"),
                output_format="pickle",
            )

    def test_invalid_output_format_raises(self, tmp_path):
        """ValueError raised for unsupported output_format."""
        f = tmp_path / "data.pkl"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="output_format"):
            package_analysis_bundle(
                input_paths=[str(f)],
                run_id="run-fmt",
                output_dir=str(tmp_path / "out"),
                output_format="csv",  # type: ignore[arg-type]
            )

    def test_empty_input_paths(self, tmp_path):
        """Empty input_paths produces a zip with only manifest.json."""
        zip_path = package_analysis_bundle(
            input_paths=[],
            run_id="run-empty",
            output_dir=str(tmp_path / "out"),
            output_format="pickle",
        )

        import json
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "run-empty/manifest.json" in names
            manifest = json.loads(zf.read("run-empty/manifest.json"))
            assert manifest["files"] == []

    def test_sha256_correctness(self, tmp_path):
        """_sha256 produces correct hex digest for known content."""
        import hashlib

        f = tmp_path / "test.bin"
        content = b"hello world"
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert _sha256(f) == expected

    def test_output_dir_created_if_missing(self, tmp_path):
        """Output directory is created automatically if it does not exist."""
        f = tmp_path / "data.pkl"
        f.write_bytes(b"data")

        out_dir = tmp_path / "deeply" / "nested" / "output"
        zip_path = package_analysis_bundle(
            input_paths=[str(f)],
            run_id="run-nest",
            output_dir=str(out_dir),
            output_format="pickle",
        )
        assert Path(zip_path).exists()


# ---------------------------------------------------------------------------
# storage_s3 tests (mocked boto3)
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.storage_s3 import S3StorageClient
from spikelab.batch_jobs.models import StoragePathTemplates
from unittest.mock import MagicMock, patch


class TestS3StorageClient:
    def _make_client(self, prefix="s3://bucket/prefix/", templates=None):
        """Build an S3StorageClient with mocked boto3."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = S3StorageClient(
                prefix=prefix,
                path_templates=templates,
            )
        return client

    def test_build_uri_default_templates(self):
        """build_uri uses default path templates."""
        client = self._make_client(prefix="s3://bucket/prefix/")
        uri = client.build_uri(run_id="run-1", filename="data.pkl")
        assert uri == "s3://bucket/prefix/inputs/run-1/data.pkl"

    def test_build_uri_custom_templates(self):
        """build_uri respects custom StoragePathTemplates."""
        templates = StoragePathTemplates(
            inputs="{prefix}data/{run_id}/{filename}",
            outputs="{prefix}results/{run_id}/",
            logs="{prefix}log/{run_id}/",
        )
        client = self._make_client(
            prefix="s3://my-bucket/my-project/", templates=templates
        )
        uri = client.build_uri(run_id="r42", filename="bundle.zip")
        assert uri == "s3://my-bucket/my-project/data/r42/bundle.zip"

    def test_build_uri_outputs_category(self):
        """build_uri with category='outputs' uses outputs template (no filename)."""
        client = self._make_client(prefix="s3://bucket/pfx/")
        uri = client.build_uri(run_id="run-2", filename="out.pkl", category="outputs")
        # outputs template is "{prefix}outputs/{run_id}/" — filename not in template
        assert uri == "s3://bucket/pfx/outputs/run-2/"

    def test_build_uri_no_prefix_raises(self):
        """build_uri raises ValueError when prefix is not configured."""
        client = self._make_client(prefix=None)
        with pytest.raises(ValueError, match="S3 prefix is not configured"):
            client.build_uri(run_id="run-1", filename="data.pkl")

    def test_output_prefix_for_run(self):
        """output_prefix_for_run formats the outputs template."""
        client = self._make_client(prefix="s3://bucket/pfx/")
        assert client.output_prefix_for_run("run-3") == "s3://bucket/pfx/outputs/run-3/"

    def test_logs_prefix_for_run(self):
        """logs_prefix_for_run formats the logs template."""
        client = self._make_client(prefix="s3://bucket/pfx/")
        assert client.logs_prefix_for_run("run-4") == "s3://bucket/pfx/logs/run-4/"

    def test_output_prefix_no_prefix_returns_empty(self):
        """output_prefix_for_run returns empty string when prefix is None."""
        client = self._make_client(prefix=None)
        assert client.output_prefix_for_run("run-5") == ""

    def test_logs_prefix_no_prefix_returns_empty(self):
        """logs_prefix_for_run returns empty string when prefix is None."""
        client = self._make_client(prefix=None)
        assert client.logs_prefix_for_run("run-6") == ""

    def test_prefix_trailing_slash_normalization(self):
        """Prefix without trailing slash gets one appended."""
        client = self._make_client(prefix="s3://bucket/no-slash")
        assert client.prefix == "s3://bucket/no-slash/"

    def test_upload_file_calls_boto3(self):
        """upload_file delegates to the boto3 client."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            result = client.upload_file(
                local_path="/tmp/data.pkl",
                s3_uri="s3://bucket/pfx/inputs/run-1/data.pkl",
            )
            mock_s3.upload_file.assert_called_once_with(
                "/tmp/data.pkl", "bucket", "pfx/inputs/run-1/data.pkl"
            )
            assert result == "s3://bucket/pfx/inputs/run-1/data.pkl"

    def test_upload_bundle_builds_uri_and_uploads(self):
        """upload_bundle composes build_uri + upload_file."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            result = client.upload_bundle(local_zip="/tmp/run-7.zip", run_id="run-7")
            assert "run-7.zip" in result
            assert mock_s3.upload_file.called

    def test_custom_templates_for_output_and_logs(self):
        """Custom templates change output_prefix and logs_prefix paths."""
        templates = StoragePathTemplates(
            inputs="{prefix}in/{run_id}/{filename}",
            outputs="{prefix}out/{run_id}/",
            logs="{prefix}lg/{run_id}/",
        )
        client = self._make_client(prefix="s3://b/p/", templates=templates)
        assert client.output_prefix_for_run("r1") == "s3://b/p/out/r1/"
        assert client.logs_prefix_for_run("r1") == "s3://b/p/lg/r1/"


# ---------------------------------------------------------------------------
# backend_k8s tests (no real cluster)
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.backend_k8s import KubernetesBatchJobBackend


class TestKubernetesBatchJobBackend:
    def test_fallback_disabled_raises(self):
        """RuntimeError when kubernetes client unavailable and fallback disabled."""
        backend = KubernetesBatchJobBackend(
            namespace="test", use_kubectl_fallback=False
        )
        backend._batch_api = None
        with pytest.raises(RuntimeError, match="kubectl fallback disabled"):
            backend.apply_manifest("apiVersion: batch/v1\nkind: Job\n")

    def test_apply_manifest_from_file(self, tmp_path, monkeypatch):
        """apply_manifest with a file path calls kubectl apply -f."""
        manifest_path = tmp_path / "job.yaml"
        manifest_path.write_text("apiVersion: batch/v1\nkind: Job\n")

        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(stdout="job/test-job created", returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        backend._batch_api = None
        result = backend.apply_manifest(str(manifest_path))

        assert len(calls) == 1
        assert "apply" in calls[0]
        assert "-f" in calls[0]
        assert str(manifest_path) in calls[0]
        assert "-n" in calls[0]
        assert "test-ns" in calls[0]

    def test_apply_manifest_from_string_creates_temp_file(self, monkeypatch):
        """apply_manifest with a raw YAML string creates and cleans up a temp file."""
        created_temps = []

        def fake_run(command, **kwargs):
            # Capture the temp file path from the command
            f_idx = command.index("-f")
            created_temps.append(command[f_idx + 1])
            return SimpleNamespace(stdout="job/test-job created", returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        backend = KubernetesBatchJobBackend(namespace="default")
        backend._batch_api = None
        backend.apply_manifest("apiVersion: batch/v1\nkind: Job\n")

        # Temp file should have been cleaned up
        assert len(created_temps) == 1
        assert not Path(created_temps[0]).exists()

    def test_job_status_parsing(self, monkeypatch):
        """job_status parses kubectl YAML output into status strings."""
        test_cases = [
            ({"status": {"failed": 1}}, "Failed"),
            ({"status": {"succeeded": 1}}, "Complete"),
            ({"status": {"active": 1}}, "Running"),
            ({"status": {}}, "Pending"),
        ]

        for yaml_status, expected in test_cases:
            monkeypatch.setattr(
                "subprocess.run",
                lambda cmd, **kw: SimpleNamespace(
                    stdout=yaml.safe_dump(yaml_status), returncode=0
                ),
            )
            backend = KubernetesBatchJobBackend(namespace="ns")
            backend._batch_api = None
            assert backend.job_status("test-job") == expected

    def test_pods_for_job_kubectl(self, monkeypatch):
        """pods_for_job parses kubectl output for pod names."""
        pod_yaml = {
            "items": [
                {"metadata": {"name": "test-job-abc"}},
                {"metadata": {"name": "test-job-def"}},
            ]
        }
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: SimpleNamespace(
                stdout=yaml.safe_dump(pod_yaml), returncode=0
            ),
        )
        backend = KubernetesBatchJobBackend(namespace="ns")
        backend._batch_api = None
        pods = backend.pods_for_job("test-job")
        assert pods == ["test-job-abc", "test-job-def"]

    def test_pods_for_job_empty(self, monkeypatch):
        """pods_for_job returns empty list when no pods found."""
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: SimpleNamespace(
                stdout=yaml.safe_dump({"items": []}), returncode=0
            ),
        )
        backend = KubernetesBatchJobBackend(namespace="ns")
        backend._batch_api = None
        assert backend.pods_for_job("no-such-job") == []

    def test_kubeconfig_passed_to_kubectl(self, monkeypatch):
        """kubeconfig path is forwarded to kubectl commands."""
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(stdout=yaml.safe_dump({"status": {}}), returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        backend = KubernetesBatchJobBackend(
            namespace="ns", kubeconfig="/path/to/config"
        )
        backend._batch_api = None
        backend.job_status("test-job")

        assert "--kubeconfig" in calls[0]
        assert "/path/to/config" in calls[0]


# ---------------------------------------------------------------------------
# session tests (mocked dependencies)
# ---------------------------------------------------------------------------


class TestRunSession:
    def _make_session(self):
        """Build a RunSession with fully mocked backend/storage."""
        from spikelab.batch_jobs.session import RunSession

        profile = ClusterProfile(name="test")
        backend = MagicMock(spec=KubernetesBatchJobBackend)
        backend.apply_manifest.return_value = "test-job-abc"
        backend.job_status.return_value = "Complete"

        storage = MagicMock(spec=S3StorageClient)
        storage.upload_bundle.return_value = "s3://test/inputs/run/bundle.zip"
        storage.output_prefix_for_run.return_value = "s3://test/outputs/run/"
        storage.logs_prefix_for_run.return_value = "s3://test/logs/run/"

        creds = MagicMock()

        session = RunSession(
            profile=profile,
            backend=backend,
            storage_client=storage,
            credentials=creds,
        )
        return session, backend, storage

    def test_build_job_name_format(self):
        """Job name is prefix-<8hex>, within 63 chars."""
        from spikelab.batch_jobs.session import RunSession

        name = RunSession._build_job_name("analysis-job")
        assert name.startswith("analysis-job-")
        assert len(name) <= 63
        # 8 hex chars after the last hyphen
        token = name.split("-")[-1]
        assert len(token) == 8
        int(token, 16)  # must be valid hex

    def test_build_job_name_long_prefix_truncated(self):
        """Long prefix is truncated to keep the name under 63 chars."""
        from spikelab.batch_jobs.session import RunSession

        long_prefix = "a" * 60
        name = RunSession._build_job_name(long_prefix)
        assert len(name) <= 63

    def test_render_manifest_produces_yaml(self):
        """render_manifest returns valid YAML with the job name."""
        session, _, _ = self._make_session()
        job_spec = validate_job_spec(_example_payload())
        manifest = session.render_manifest(
            job_name="test-render", job_spec=job_spec, run_id="run-1"
        )
        parsed = yaml.safe_load(manifest)
        assert parsed["metadata"]["name"] == "test-render"
        assert parsed["kind"] == "Job"

    def test_submit_prepared_job_calls_backend(self):
        """submit_prepared_job applies the manifest and returns SubmitResult."""
        session, backend, storage = self._make_session()
        job_spec = validate_job_spec(_example_payload())

        result = session.submit_prepared_job(job_spec=job_spec, run_id="run-prep")

        backend.apply_manifest.assert_called_once()
        assert result.job_name.startswith("analysis-job-")
        assert result.output_prefix == "s3://test/outputs/run/"
        assert result.logs_prefix == "s3://test/logs/run/"

    def test_submit_prepared_job_blocked_by_policy(self):
        """submit_prepared_job raises when policy blocks and override is False."""
        session, _, _ = self._make_session()
        payload = _example_payload()
        payload["container"]["args"] = ["sleep", "infinity"]
        job_spec = validate_job_spec(payload)

        with pytest.raises(RuntimeError, match="Policy preflight blocked"):
            session.submit_prepared_job(job_spec=job_spec)

    def test_submit_prepared_job_policy_override(self):
        """submit_prepared_job succeeds with allow_policy_risk=True despite BLOCK."""
        session, backend, _ = self._make_session()
        payload = _example_payload()
        payload["container"]["args"] = ["sleep", "infinity"]
        job_spec = validate_job_spec(payload)

        result = session.submit_prepared_job(job_spec=job_spec, allow_policy_risk=True)
        assert result.job_name  # should succeed
        backend.apply_manifest.assert_called_once()

    def test_wait_for_completion_returns_complete(self):
        """wait_for_completion returns 'Complete' when job succeeds."""
        session, backend, _ = self._make_session()
        backend.job_status.return_value = "Complete"

        state = session.wait_for_completion(
            job_name="test-job", max_wait_seconds=5, poll_interval_seconds=0
        )
        assert state == "Complete"

    def test_wait_for_completion_returns_failed(self):
        """wait_for_completion returns 'Failed' when job fails."""
        session, backend, _ = self._make_session()
        backend.job_status.return_value = "Failed"

        state = session.wait_for_completion(
            job_name="test-job", max_wait_seconds=5, poll_interval_seconds=0
        )
        assert state == "Failed"

    def test_wait_for_completion_timeout(self):
        """wait_for_completion returns 'Timeout' when deadline exceeded."""
        session, backend, _ = self._make_session()
        backend.job_status.return_value = "Running"

        state = session.wait_for_completion(
            job_name="test-job", max_wait_seconds=0, poll_interval_seconds=0
        )
        assert state == "Timeout"


# ---------------------------------------------------------------------------
# profiles tests
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.profiles import load_profile_from_name


class TestProfiles:
    def test_load_defaults_profile(self):
        """'defaults' profile loads without error and has generic values."""
        profile = load_profile_from_name("defaults")
        assert profile.name == "defaults"
        assert profile.default_images == {}
        assert profile.namespace == "default"

    def test_load_nrp_profile(self):
        """'nrp' profile loads and has the expected namespace."""
        profile = load_profile_from_name("nrp")
        assert profile.name == "nrp"
        assert profile.namespace_hooks  # should have at least one hook

    def test_load_unknown_name_falls_back_to_defaults(self):
        """Unknown profile name falls back to defaults.yaml."""
        profile = load_profile_from_name("unknown-cluster")
        assert profile.name == "defaults"

    def test_nautilus_alias(self):
        """'nautilus' loads the same profile as 'nrp'."""
        profile = load_profile_from_name("nautilus")
        assert profile.name == "nrp"

    def test_load_profile_from_explicit_path(self, tmp_path):
        """load_cluster_profile reads a custom YAML file."""
        from spikelab.batch_jobs.profiles import load_cluster_profile

        profile_yaml = tmp_path / "custom.yaml"
        profile_yaml.write_text("name: custom\nnamespace: my-ns\n", encoding="utf-8")
        profile = load_cluster_profile(str(profile_yaml))
        assert profile.name == "custom"
        assert profile.namespace == "my-ns"

    def test_load_profile_file_not_found(self):
        """load_cluster_profile raises when file does not exist."""
        from spikelab.batch_jobs.profiles import load_cluster_profile

        with pytest.raises(FileNotFoundError):
            load_cluster_profile("/nonexistent/profile.yaml")

    def test_load_profile_non_dict_raises(self, tmp_path):
        """Profile file containing a list raises ValueError."""
        from spikelab.batch_jobs.profiles import load_cluster_profile

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid profile file"):
            load_cluster_profile(str(bad_yaml))


# ---------------------------------------------------------------------------
# Model validation edge cases
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.models import (
    ResourceSpec,
    ContainerSpec,
    StoragePathTemplates,
    PolicyConfig,
)
from pydantic import ValidationError as PydanticValidationError


class TestModelValidationEdgeCases:
    def test_gpu_requests_must_equal_limits(self):
        """ResourceSpec rejects mismatched GPU requests and limits."""
        with pytest.raises(
            PydanticValidationError, match="GPU requests and limits must match"
        ):
            ResourceSpec(requests_gpu=1, limits_gpu=2)

    def test_gpu_zero_zero_allowed(self):
        """ResourceSpec allows requests_gpu=0 and limits_gpu=0."""
        spec = ResourceSpec(requests_gpu=0, limits_gpu=0)
        assert spec.requests_gpu == 0

    def test_volume_mount_requires_source(self):
        """VolumeMountSpec rejects when neither secret_name nor pvc_name provided."""
        with pytest.raises(PydanticValidationError, match="secret_name or pvc_name"):
            VolumeMountSpec(name="vol", mount_path="/mnt")

    def test_volume_mount_both_sources_allowed(self):
        """VolumeMountSpec accepts both secret_name and pvc_name (no conflict error)."""
        vol = VolumeMountSpec(
            name="vol", mount_path="/mnt", secret_name="sec", pvc_name="pvc"
        )
        assert vol.secret_name == "sec"
        assert vol.pvc_name == "pvc"

    def test_name_prefix_special_chars_sanitized(self):
        """JobSpec sanitizes special characters in name_prefix to hyphens."""
        payload = _example_payload()
        payload["name_prefix"] = "my job!@#test"
        job_spec = validate_job_spec(payload)
        assert job_spec.name_prefix == "my-job---test"

    def test_name_prefix_all_special_chars_fallback(self):
        """JobSpec falls back to 'analysis-job' when prefix is all special chars."""
        payload = _example_payload()
        payload["name_prefix"] = "---"
        job_spec = validate_job_spec(payload)
        assert job_spec.name_prefix == "analysis-job"

    def test_name_prefix_truncated_to_40(self):
        """JobSpec truncates name_prefix to 40 characters."""
        payload = _example_payload()
        payload["name_prefix"] = "a" * 60
        job_spec = validate_job_spec(payload)
        assert len(job_spec.name_prefix) <= 40

    def test_container_spec_empty_image_rejected(self):
        """ContainerSpec rejects empty image string."""
        with pytest.raises(PydanticValidationError):
            ContainerSpec(image="")

    def test_active_deadline_seconds_zero_rejected(self):
        """JobSpec rejects active_deadline_seconds=0."""
        payload = _example_payload()
        payload["active_deadline_seconds"] = 0
        with pytest.raises(PydanticValidationError):
            validate_job_spec(payload)


# ---------------------------------------------------------------------------
# Validation module edge cases
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.validation import (
    validate_run_config,
    summarize_validation_error,
)


class TestValidationModule:
    def test_validate_run_config_happy_path(self):
        """validate_run_config parses a valid RunConfig payload."""
        config = validate_run_config({"input_path": "/data/recording.h5"})
        assert config.input_path == "/data/recording.h5"
        assert config.profile_name == "defaults"

    def test_validate_run_config_missing_required_field(self):
        """validate_run_config raises for missing input_path."""
        with pytest.raises(PydanticValidationError):
            validate_run_config({})

    def test_validate_run_config_invalid_format(self):
        """validate_run_config rejects invalid output_format."""
        with pytest.raises(PydanticValidationError):
            validate_run_config({"input_path": "/data/x.h5", "output_format": "csv"})

    def test_summarize_validation_error_format(self):
        """summarize_validation_error produces a readable string."""
        try:
            validate_run_config({})
        except PydanticValidationError as exc:
            summary = summarize_validation_error(exc)
            assert "input_path" in summary
            assert isinstance(summary, str)

    def test_summarize_validation_error_multiple_errors(self):
        """summarize_validation_error joins multiple errors with semicolons."""
        try:
            validate_job_spec({"container": {}})  # missing image + other issues
        except PydanticValidationError as exc:
            summary = summarize_validation_error(exc)
            assert ";" in summary  # multiple errors joined


# ---------------------------------------------------------------------------
# Credential edge cases
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.credentials import resolve_credentials


class TestCredentialEdgeCases:
    def test_resolve_credentials_explicit_wins(self, monkeypatch):
        """Explicit parameters take precedence over environment variables."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
        creds = resolve_credentials(
            aws_access_key_id="explicit-key",
            aws_secret_access_key="explicit-secret",
        )
        assert creds.aws_access_key_id == "explicit-key"
        assert creds.aws_secret_access_key == "explicit-secret"

    def test_resolve_credentials_falls_back_to_env(self, monkeypatch):
        """Missing explicit params fall back to environment variables."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        monkeypatch.setenv("KUBECONFIG", "/env/kube/config")
        creds = resolve_credentials()
        assert creds.aws_access_key_id == "env-key"
        assert creds.kubeconfig == "/env/kube/config"

    def test_resolve_credentials_all_none(self, monkeypatch):
        """All fields are None when no params or env vars are set."""
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
        monkeypatch.delenv("KUBECONFIG", raising=False)
        creds = resolve_credentials()
        assert creds.aws_access_key_id is None
        assert creds.aws_secret_access_key is None
        assert creds.kubeconfig is None

    def test_redact_none_values(self):
        """redact_sensitive_map converts None values to empty strings."""
        redacted = redact_sensitive_map({"FIELD": None, "OTHER": "ok"})
        assert redacted["FIELD"] == ""
        assert redacted["OTHER"] == "ok"


# ---------------------------------------------------------------------------
# Namespace hook edge cases
# ---------------------------------------------------------------------------


class TestNamespaceHookEdgeCases:
    def test_user_command_not_overridden_by_hook_default(self):
        """Hook default_command does not override user-specified command."""
        payload = _example_payload()
        payload["namespace"] = "test-ns"
        payload["container"]["command"] = ["python", "-m", "my_script"]
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            namespace_hooks={
                "test-ns": NamespaceHookSpec(
                    default_command=["sh", "-c"],
                ),
            },
        )
        context = build_template_context(
            job_name="cmd-test",
            job_spec=job_spec,
            profile=profile,
        )
        manifest = render_job_manifest(context)
        parsed = yaml.safe_load(manifest)
        container = parsed["spec"]["template"]["spec"]["containers"][0]
        assert container["command"] == ["python", "-m", "my_script"]

    def test_hook_default_command_applied_when_user_has_none(self):
        """Hook default_command is used when user provides no command."""
        payload = _example_payload()
        payload["namespace"] = "test-ns"
        payload["container"]["command"] = []
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            namespace_hooks={
                "test-ns": NamespaceHookSpec(
                    default_command=["sh", "-c"],
                ),
            },
        )
        context = build_template_context(
            job_name="cmd-default-test",
            job_spec=job_spec,
            profile=profile,
        )
        manifest = render_job_manifest(context)
        parsed = yaml.safe_load(manifest)
        container = parsed["spec"]["template"]["spec"]["containers"][0]
        assert container["command"] == ["sh", "-c"]

    def test_default_volumes_always_applied(self):
        """Profile default_volumes are injected regardless of namespace."""
        payload = _example_payload()
        payload["namespace"] = "any-namespace"
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test-with-defaults",
            default_volumes=[
                VolumeMountSpec(
                    name="shared-vol",
                    mount_path="/etc/shared",
                    secret_name="shared-secret",
                ),
            ],
        )
        context = build_template_context(
            job_name="default-vol-test",
            job_spec=job_spec,
            profile=profile,
        )
        manifest = render_job_manifest(context)
        parsed = yaml.safe_load(manifest)
        mounts = parsed["spec"]["template"]["spec"]["containers"][0].get(
            "volumeMounts", []
        )
        mount_paths = {item["mountPath"] for item in mounts}
        assert "/etc/shared" in mount_paths


# ---------------------------------------------------------------------------
# Policy edge cases
# ---------------------------------------------------------------------------


class TestPolicyEdgeCases:
    def test_summarize_preflight_empty_findings(self):
        """Empty findings list returns PASS with empty text."""
        level, text = summarize_preflight([])
        assert level == "PASS"
        assert text == ""

    def test_policy_long_runtime_warning(self):
        """active_deadline_seconds exceeding max triggers WARN."""
        payload = _example_payload()
        payload["active_deadline_seconds"] = 2_000_000
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings = evaluate_policy(job_spec, profile)
        codes = {f.code: f.level for f in findings}
        assert codes["long_runtime"] == "WARN"

    def test_policy_no_runtime_warning_when_not_set(self):
        """No long_runtime finding when active_deadline_seconds is None."""
        payload = _example_payload()
        # active_deadline_seconds defaults to None
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings = evaluate_policy(job_spec, profile)
        codes = {f.code for f in findings}
        assert "long_runtime" not in codes

    def test_policy_request_limit_mismatch_warning(self):
        """Mismatched CPU/memory requests and limits triggers WARN."""
        payload = _example_payload()
        payload["resources"]["requests_cpu"] = "1"
        payload["resources"]["limits_cpu"] = "4"
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings = evaluate_policy(job_spec, profile)
        codes = {f.code: f.level for f in findings}
        assert codes["request_limit_mismatch"] == "WARN"

    def test_policy_warn_mismatch_disabled_by_profile(self):
        """request_limit_mismatch check can be disabled via profile."""
        payload = _example_payload()
        payload["resources"]["requests_cpu"] = "1"
        payload["resources"]["limits_cpu"] = "4"
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(warn_request_limit_mismatch=False),
        )
        findings = evaluate_policy(job_spec, profile)
        codes = {f.code: f.level for f in findings}
        assert codes["request_limit_mismatch"] == "PASS"


# ---------------------------------------------------------------------------
# Backend edge cases
# ---------------------------------------------------------------------------


class TestBackendEdgeCases:
    def test_kubectl_failure_raises(self, monkeypatch):
        """CalledProcessError from kubectl propagates."""
        import subprocess

        def fake_run(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, stderr="error msg")

        monkeypatch.setattr("subprocess.run", fake_run)
        backend = KubernetesBatchJobBackend(namespace="ns")
        backend._batch_api = None
        with pytest.raises(subprocess.CalledProcessError):
            backend.job_status("test-job")

    def test_delete_job_kubectl_fallback(self, monkeypatch):
        """delete_job falls back to kubectl when K8s client unavailable."""
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(stdout="", returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        backend = KubernetesBatchJobBackend(namespace="ns")
        backend._batch_api = None
        backend.delete_job("test-job")
        assert any("delete" in cmd for cmd in calls)
        assert any("test-job" in cmd for cmd in calls)
