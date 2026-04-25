"""Tests for the batch job-launcher package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

if (
    importlib.util.find_spec("pydantic") is None
    or importlib.util.find_spec("yaml") is None
):
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

    def test_sleep_scientific_notation(self):
        """sleep 1e6 (scientific notation) should be caught as large number."""
        assert _contains_disallowed_sleep(["sleep"], ["1e6"])

    def test_sleep_negative_number(self):
        """sleep -1 should not be flagged (negative is below threshold)."""
        assert not _contains_disallowed_sleep(["sleep"], ["-1"])

    def test_nosleep_substring_false_positive(self):
        """Commands containing 'sleep' as substring should not be flagged."""
        # "nosleep" contains "sleep" but is not a sleep command
        assert not _contains_disallowed_sleep(["nosleep"], [])
        # "sleepless" as a command name
        assert not _contains_disallowed_sleep(["sleepless"], ["module"])


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
            output_format="workspace",
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
        input_file = tmp_path / "result.h5"
        input_file.write_bytes(b"fake workspace content")

        zip_path = package_analysis_bundle(
            input_paths=[str(input_file)],
            run_id="run-002",
            output_dir=str(tmp_path / "out"),
            output_format="sorting",
            metadata={"workspace_id": "ws-42"},
        )

        import json
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            manifest = json.loads(zf.read("run-002/manifest.json"))

        assert manifest["run_id"] == "run-002"
        assert manifest["output_format"] == "sorting"
        assert manifest["metadata"]["workspace_id"] == "ws-42"
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["name"] == "result.h5"
        assert len(manifest["files"][0]["sha256"]) == 64  # hex SHA256

    def test_multiple_input_files(self, tmp_path):
        """Multiple input files are all included in the bundle."""
        f1 = tmp_path / "a.pkl"
        f2 = tmp_path / "b.h5"
        f1.write_bytes(b"data1")
        f2.write_bytes(b"data2")

        zip_path = package_analysis_bundle(
            input_paths=[str(f1), str(f2)],
            run_id="run-multi",
            output_dir=str(tmp_path / "out"),
            output_format="custom",
        )

        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "run-multi/a.pkl" in names
            assert "run-multi/b.h5" in names
            assert "run-multi/manifest.json" in names

    def test_missing_input_file_raises(self, tmp_path):
        """FileNotFoundError raised when an input path does not exist."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            package_analysis_bundle(
                input_paths=["/nonexistent/file.pkl"],
                run_id="run-bad",
                output_dir=str(tmp_path / "out"),
                output_format="workspace",
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
            output_format="workspace",
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
            output_format="workspace",
        )
        assert Path(zip_path).exists()

    def test_duplicate_filenames_last_wins(self, tmp_path):
        """Duplicate filenames in input_paths: last file overwrites earlier ones."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()

        file_a = dir_a / "data.pkl"
        file_a.write_bytes(b"content_a")
        file_b = dir_b / "data.pkl"
        file_b.write_bytes(b"content_b")

        import zipfile

        zip_path = package_analysis_bundle(
            input_paths=[str(file_a), str(file_b)],
            run_id="run-dup",
            output_dir=str(tmp_path / "out"),
            output_format="workspace",
        )

        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("run-dup/data.pkl")
            # Last copy wins (shutil.copy2 overwrites)
            assert content == b"content_b"

    def test_large_file_hashing(self, tmp_path):
        """Files larger than the read chunk size are hashed correctly."""
        large_file = tmp_path / "large.bin"
        # Write >8192 bytes to trigger multi-chunk hashing
        content = b"x" * 20000
        large_file.write_bytes(content)

        import hashlib

        expected = hashlib.sha256(content).hexdigest()
        assert _sha256(Path(large_file)) == expected


# ---------------------------------------------------------------------------
# storage_s3 tests (mocked boto3)
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.storage_s3 import S3StorageClient
from spikelab.batch_jobs.models import StoragePathTemplates
from unittest.mock import MagicMock, patch


class TestS3StorageClient:
    def _make_client(self, prefix="s3://bucket/prefix/", templates=None):
        """Build an S3StorageClient with a mocked boto3 client."""
        client = S3StorageClient(
            prefix=prefix,
            path_templates=templates,
        )
        client._client = MagicMock()  # bypass lazy init
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

    def test_boto3_not_installed(self):
        """ImportError raised when boto3 is not available — deferred to first use."""
        with patch("spikelab.batch_jobs.storage_s3.boto3", None):
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            with pytest.raises(ImportError, match="boto3 is required"):
                client.upload_file(local_path="x", s3_uri="s3://bucket/pfx/x")

    def test_build_uri_invalid_category_falls_back_to_inputs(self):
        """Invalid category string falls back to inputs template."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = S3StorageClient(prefix="s3://bucket/pfx/")
        # "invalid_category" doesn't exist as a template attribute
        uri = client.build_uri(
            run_id="run-1", filename="data.pkl", category="invalid_category"
        )
        # Falls back to inputs template
        assert uri == "s3://bucket/pfx/inputs/run-1/data.pkl"

    def test_build_uri_special_chars_in_run_id(self):
        """Special characters in run_id are passed through to the URI."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = S3StorageClient(prefix="s3://bucket/pfx/")
        uri = client.build_uri(run_id="run/with spaces", filename="data.pkl")
        assert "run/with spaces" in uri

    def test_build_uri_special_chars_in_filename(self):
        """Special characters in filename are passed through to the URI."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = S3StorageClient(prefix="s3://bucket/pfx/")
        uri = client.build_uri(run_id="run-1", filename="my file (1).pkl")
        assert "my file (1).pkl" in uri


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

    def test_empty_yaml_file(self, tmp_path):
        """Empty YAML file produces a Pydantic ValidationError (missing 'name')."""
        from spikelab.batch_jobs.profiles import load_cluster_profile

        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(PydanticValidationError):
            load_cluster_profile(str(empty_file))

    def test_yaml_null_only(self, tmp_path):
        """YAML file containing only 'null' raises ValidationError."""
        from spikelab.batch_jobs.profiles import load_cluster_profile

        null_file = tmp_path / "null.yaml"
        null_file.write_text("null\n", encoding="utf-8")
        with pytest.raises(PydanticValidationError):
            load_cluster_profile(str(null_file))


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


class TestModelValidation:
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

    def test_validate_run_config_invalid_wait(self):
        """validate_run_config rejects max_wait_seconds below minimum."""
        with pytest.raises(PydanticValidationError):
            validate_run_config({"input_path": "/data/x.h5", "max_wait_seconds": 0})

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


class TestCredential:
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


class TestNamespaceHook:
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


class TestPolicy:
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


class TestBackend:
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


# ---------------------------------------------------------------------------
# Policy boundary and precedence edge cases
# ---------------------------------------------------------------------------


class TestPolicyBoundary:
    def test_gpu_exactly_at_threshold(self):
        """requests_gpu == max_interactive_gpus should PASS (not WARN)."""
        payload = _example_payload()
        payload["resources"]["requests_gpu"] = 2
        payload["resources"]["limits_gpu"] = 2
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(max_interactive_gpus=2),
        )
        findings = evaluate_policy(job_spec, profile)
        gpu_finding = [f for f in findings if f.code == "interactive_gpu_limit"][0]
        assert gpu_finding.level == "PASS"

    def test_block_sleep_infinity_disabled(self):
        """block_sleep_infinity=False allows sleep commands through."""
        payload = _example_payload()
        payload["container"]["args"] = ["sleep", "infinity"]
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(block_sleep_infinity=False),
        )
        findings = evaluate_policy(job_spec, profile)
        sleep_finding = [f for f in findings if f.code == "sleep_in_batch_job"][0]
        assert sleep_finding.level == "PASS"

    def test_active_deadline_at_boundary(self):
        """active_deadline_seconds == max_runtime_seconds should not WARN."""
        payload = _example_payload()
        payload["active_deadline_seconds"] = 1_209_600  # exactly 14 days
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(max_runtime_seconds=1_209_600),
        )
        findings = evaluate_policy(job_spec, profile)
        codes = {f.code for f in findings}
        assert "long_runtime" not in codes

    def test_mixed_block_and_warn_findings(self):
        """BLOCK takes precedence over WARN in summarize_preflight."""
        payload = _example_payload()
        # Trigger BLOCK: sleep infinity
        payload["container"]["args"] = ["sleep", "infinity"]
        # Trigger WARN: GPU above threshold
        payload["resources"]["requests_gpu"] = 5
        payload["resources"]["limits_gpu"] = 5
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings = evaluate_policy(job_spec, profile)
        level, text = summarize_preflight(findings)
        assert level == "BLOCK"
        # Both findings should appear in the summary text
        assert "sleep_in_batch_job" in text
        assert "interactive_gpu_limit" in text


# ---------------------------------------------------------------------------
# Sleep detection edge cases
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# _build_job_name edge cases
# ---------------------------------------------------------------------------


class TestBuildJobName:
    def test_empty_prefix(self):
        """Empty prefix produces a name that is just '-<token>'."""
        from spikelab.batch_jobs.session import RunSession

        name = RunSession._build_job_name("")
        # Empty string rstripped of hyphens is still empty, so name is "-<8hex>"
        assert len(name) <= 63
        token = name.split("-")[-1]
        assert len(token) == 8
        int(token, 16)

    def test_all_hyphens_prefix(self):
        """Prefix like '---' is rstripped to empty, producing '-<token>'."""
        from spikelab.batch_jobs.session import RunSession

        name = RunSession._build_job_name("---")
        assert len(name) <= 63
        token = name.split("-")[-1]
        assert len(token) == 8
        int(token, 16)

    def test_prefix_exactly_at_max_length(self):
        """54-char prefix fits exactly (54 + 1 + 8 = 63)."""
        from spikelab.batch_jobs.session import RunSession

        prefix = "a" * 54
        name = RunSession._build_job_name(prefix)
        assert len(name) == 63
        assert name.startswith("a" * 54 + "-")


# ---------------------------------------------------------------------------
# RunConfig validation edge cases
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.models import RunConfig


class TestRunConfigValidation:
    def test_max_wait_seconds_zero_rejected(self):
        """max_wait_seconds=0 should fail validation (ge=1)."""
        with pytest.raises(PydanticValidationError):
            RunConfig(input_path="/data/test.h5", max_wait_seconds=0)

    def test_max_wait_seconds_one_accepted(self):
        """max_wait_seconds=1 is the minimum allowed value."""
        config = RunConfig(input_path="/data/test.h5", max_wait_seconds=1)
        assert config.max_wait_seconds == 1

    def test_max_runtime_seconds_zero_rejected(self):
        """PolicyConfig max_runtime_seconds=0 should fail validation (ge=1)."""
        with pytest.raises(PydanticValidationError):
            PolicyConfig(max_runtime_seconds=0)


# ---------------------------------------------------------------------------
# SubmitResult model tests
# ---------------------------------------------------------------------------

from spikelab.batch_jobs.models import SubmitResult


class TestSubmitResult:
    def test_construction_workspace_type(self):
        """
        SubmitResult can be constructed with job_type='workspace'.

        Tests:
            (Test Case 1) All fields are stored correctly.
            (Test Case 2) job_type is 'workspace'.
        """
        result = SubmitResult(
            job_name="test-job-abc",
            manifest_yaml="apiVersion: batch/v1\n",
            run_id="abc123",
            uploaded_input_uri="s3://bucket/inputs/abc123/bundle.zip",
            output_prefix="s3://bucket/outputs/abc123/",
            logs_prefix="s3://bucket/logs/abc123/",
            job_type="workspace",
        )
        assert result.job_name == "test-job-abc"
        assert result.run_id == "abc123"
        assert result.job_type == "workspace"

    def test_construction_sorting_type(self):
        """
        SubmitResult accepts job_type='sorting'.

        Tests:
            (Test Case 1) job_type is 'sorting'.
        """
        result = SubmitResult(
            job_name="sort-job-def",
            manifest_yaml="kind: Job\n",
            run_id="def456",
            uploaded_input_uri="s3://bucket/inputs/def456/bundle.zip",
            output_prefix="s3://bucket/outputs/def456/",
            logs_prefix="s3://bucket/logs/def456/",
            job_type="sorting",
        )
        assert result.job_type == "sorting"

    def test_construction_prepared_type(self):
        """
        SubmitResult accepts job_type='prepared'.

        Tests:
            (Test Case 1) job_type is 'prepared'.
        """
        result = SubmitResult(
            job_name="prep-job",
            manifest_yaml="",
            run_id="ghi789",
            uploaded_input_uri="",
            output_prefix="",
            logs_prefix="",
            job_type="prepared",
        )
        assert result.job_type == "prepared"

    def test_invalid_job_type_rejected(self):
        """
        SubmitResult rejects invalid job_type values.

        Tests:
            (Test Case 1) job_type='pickle' is not accepted.
        """
        with pytest.raises(PydanticValidationError):
            SubmitResult(
                job_name="bad",
                manifest_yaml="",
                run_id="x",
                uploaded_input_uri="",
                output_prefix="",
                logs_prefix="",
                job_type="pickle",
            )


# ---------------------------------------------------------------------------
# S3StorageClient download/list tests
# ---------------------------------------------------------------------------


class TestS3StorageClientDownload:
    def _make_client(self, prefix="s3://bucket/prefix/"):
        """Build an S3StorageClient with a mocked boto3 client."""
        client = S3StorageClient(prefix=prefix)
        mock_s3 = MagicMock()
        client._client = mock_s3  # bypass lazy init
        return client, mock_s3

    def test_download_file_calls_boto3(self, tmp_path):
        """
        download_file delegates to boto3 client.download_file.

        Tests:
            (Test Case 1) Correct bucket and key parsed from URI.
            (Test Case 2) Returns the local_path.
        """
        client, mock_s3 = self._make_client()
        local = str(tmp_path / "out.h5")
        result = client.download_file(
            s3_uri="s3://bucket/prefix/outputs/run-1/workspace.h5",
            local_path=local,
        )
        mock_s3.download_file.assert_called_once_with(
            "bucket", "prefix/outputs/run-1/workspace.h5", local
        )
        assert result == local

    def test_download_file_creates_parent_dirs(self, tmp_path):
        """
        download_file creates intermediate directories if needed.

        Tests:
            (Test Case 1) Nested parent directories are created.
        """
        client, mock_s3 = self._make_client()
        local = str(tmp_path / "deep" / "nested" / "file.h5")
        client.download_file(
            s3_uri="s3://bucket/key/file.h5",
            local_path=local,
        )
        assert (tmp_path / "deep" / "nested").is_dir()

    def test_download_output_uses_output_prefix(self, tmp_path):
        """
        download_output composes the output prefix with the filename.

        Tests:
            (Test Case 1) Downloads from the correct S3 URI.
            (Test Case 2) Saves to the correct local path.
        """
        client, mock_s3 = self._make_client(prefix="s3://bucket/pfx/")
        local = client.download_output(
            run_id="run-1", filename="workspace.h5", local_dir=str(tmp_path)
        )
        expected_uri = "s3://bucket/pfx/outputs/run-1/workspace.h5"
        # download_file is called internally; check the underlying boto3 call
        mock_s3.download_file.assert_called_once()
        call_args = mock_s3.download_file.call_args
        assert call_args[0][1] == "pfx/outputs/run-1/workspace.h5"
        assert local == str(tmp_path / "workspace.h5")

    def test_list_output_files_paginates(self):
        """
        list_output_files uses a paginator to list all keys.

        Tests:
            (Test Case 1) Keys from multiple pages are combined.
            (Test Case 2) Returns full S3 keys.
        """
        client, mock_s3 = self._make_client(prefix="s3://bucket/pfx/")
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "pfx/outputs/r1/a.h5"}]},
            {"Contents": [{"Key": "pfx/outputs/r1/b.json"}]},
        ]

        keys = client.list_output_files("r1")
        assert keys == ["pfx/outputs/r1/a.h5", "pfx/outputs/r1/b.json"]
        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")

    def test_list_output_files_empty_prefix(self):
        """
        list_output_files returns empty list when prefix is None.

        Tests:
            (Test Case 1) No S3 calls made, empty list returned.
        """
        client, mock_s3 = self._make_client(prefix=None)
        keys = client.list_output_files("r1")
        assert keys == []

    def test_list_output_files_no_contents(self):
        """
        list_output_files handles pages with no Contents key.

        Tests:
            (Test Case 1) Returns empty list when no objects found.
        """
        client, mock_s3 = self._make_client(prefix="s3://bucket/pfx/")
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        keys = client.list_output_files("r1")
        assert keys == []


# ---------------------------------------------------------------------------
# RunSession: submit_workspace_job tests
# ---------------------------------------------------------------------------


class TestRunSessionWorkspaceJob:
    def _make_session(self):
        """Build a RunSession with fully mocked backend/storage."""
        from spikelab.batch_jobs.session import RunSession

        profile = ClusterProfile(name="test")
        backend = MagicMock(spec=KubernetesBatchJobBackend)
        backend.apply_manifest.return_value = "test-job-abc"

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

    def test_submit_workspace_job_with_object(self, tmp_path):
        """
        submit_workspace_job accepts an AnalysisWorkspace object.

        Tests:
            (Test Case 1) Returns SubmitResult with job_type='workspace'.
            (Test Case 2) Bundle is uploaded to S3.
            (Test Case 3) Container command is the workspace entrypoint.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, backend, storage = self._make_session()
        ws = AnalysisWorkspace(name="test-ws")

        script = tmp_path / "analyze.py"
        script.write_text("print('hello')", encoding="utf-8")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_workspace_job(
            workspace=ws,
            script=str(script),
            job_spec=job_spec,
        )

        assert result.job_type == "workspace"
        assert result.run_id  # non-empty
        storage.upload_bundle.assert_called_once()
        backend.apply_manifest.assert_called_once()

    def test_submit_workspace_job_with_path(self, tmp_path):
        """
        submit_workspace_job accepts a string path to a saved workspace.

        Tests:
            (Test Case 1) Workspace is loaded from the path.
            (Test Case 2) Returns SubmitResult.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, _, storage = self._make_session()

        # Save a workspace to disk
        ws = AnalysisWorkspace(name="saved-ws")
        base = str(tmp_path / "my_workspace")
        ws.save(base)

        script = tmp_path / "analyze.py"
        script.write_text("print('hello')", encoding="utf-8")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_workspace_job(
            workspace=base,
            script=str(script),
            job_spec=job_spec,
        )
        assert result.job_type == "workspace"

    def test_submit_workspace_job_missing_workspace_raises(self, tmp_path):
        """
        submit_workspace_job raises FileNotFoundError for missing workspace.

        Tests:
            (Test Case 1) Non-existent workspace path raises.
        """
        session, _, _ = self._make_session()
        script = tmp_path / "analyze.py"
        script.write_text("print('hello')", encoding="utf-8")

        job_spec = validate_job_spec(_example_payload())
        with pytest.raises(FileNotFoundError, match="Workspace file not found"):
            session.submit_workspace_job(
                workspace="/nonexistent/workspace",
                script=str(script),
                job_spec=job_spec,
            )

    def test_submit_workspace_job_missing_script_raises(self, tmp_path):
        """
        submit_workspace_job raises FileNotFoundError for missing script.

        Tests:
            (Test Case 1) Non-existent script path raises.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, _, _ = self._make_session()
        ws = AnalysisWorkspace(name="test-ws")
        job_spec = validate_job_spec(_example_payload())

        with pytest.raises(FileNotFoundError, match="Analysis script not found"):
            session.submit_workspace_job(
                workspace=ws,
                script="/nonexistent/script.py",
                job_spec=job_spec,
            )

    def test_submit_workspace_job_policy_block(self, tmp_path):
        """
        submit_workspace_job raises on policy BLOCK.

        Tests:
            (Test Case 1) Sleep infinity command triggers policy block.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, _, _ = self._make_session()
        ws = AnalysisWorkspace(name="test-ws")
        script = tmp_path / "analyze.py"
        script.write_text("print('hello')", encoding="utf-8")

        payload = _example_payload()
        payload["container"]["args"] = ["sleep", "infinity"]
        job_spec = validate_job_spec(payload)

        with pytest.raises(RuntimeError, match="Policy preflight blocked"):
            session.submit_workspace_job(
                workspace=ws, script=str(script), job_spec=job_spec
            )

    def test_submit_workspace_job_env_vars_set(self, tmp_path):
        """
        submit_workspace_job injects INPUT_URI, OUTPUT_PREFIX, SCRIPT_NAME.

        Tests:
            (Test Case 1) The manifest YAML contains the expected env vars.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, backend, storage = self._make_session()
        ws = AnalysisWorkspace(name="test-ws")
        script = tmp_path / "my_analysis.py"
        script.write_text("print('hello')", encoding="utf-8")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_workspace_job(
            workspace=ws, script=str(script), job_spec=job_spec
        )

        manifest = result.manifest_yaml
        assert "INPUT_URI" in manifest
        assert "OUTPUT_PREFIX" in manifest
        assert "SCRIPT_NAME" in manifest
        assert "my_analysis.py" in manifest


# ---------------------------------------------------------------------------
# RunSession: submit_sorting_job tests
# ---------------------------------------------------------------------------


class TestRunSessionSortingJob:
    def _make_session(self):
        """Build a RunSession with fully mocked backend/storage."""
        from spikelab.batch_jobs.session import RunSession

        profile = ClusterProfile(name="test")
        backend = MagicMock(spec=KubernetesBatchJobBackend)
        backend.apply_manifest.return_value = "sort-job-abc"

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

    def test_submit_sorting_job_default_config(self, tmp_path):
        """
        submit_sorting_job works with config=None (default config).

        Tests:
            (Test Case 1) Returns SubmitResult with job_type='sorting'.
            (Test Case 2) Bundle is uploaded to S3.
        """
        session, backend, storage = self._make_session()

        rec = tmp_path / "recording.h5"
        rec.write_bytes(b"fake recording")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_sorting_job(
            recording_paths=[str(rec)],
            config=None,
            job_spec=job_spec,
        )

        assert result.job_type == "sorting"
        storage.upload_bundle.assert_called_once()
        backend.apply_manifest.assert_called_once()

    def test_submit_sorting_job_preset_string(self, tmp_path):
        """
        submit_sorting_job accepts a preset name string.

        Tests:
            (Test Case 1) 'kilosort4' is resolved to a config.
            (Test Case 2) Returns valid SubmitResult.
        """
        session, _, _ = self._make_session()
        rec = tmp_path / "recording.h5"
        rec.write_bytes(b"fake recording")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_sorting_job(
            recording_paths=[str(rec)],
            config="kilosort4",
            job_spec=job_spec,
        )
        assert result.job_type == "sorting"

    def test_submit_sorting_job_config_object(self, tmp_path):
        """
        submit_sorting_job accepts a SortingPipelineConfig object.

        Tests:
            (Test Case 1) Config is serialized and bundled.
        """
        from spikelab.spike_sorting.config import SortingPipelineConfig

        session, _, _ = self._make_session()
        rec = tmp_path / "recording.h5"
        rec.write_bytes(b"fake recording")

        config = SortingPipelineConfig()
        job_spec = validate_job_spec(_example_payload())
        result = session.submit_sorting_job(
            recording_paths=[str(rec)],
            config=config,
            job_spec=job_spec,
        )
        assert result.job_type == "sorting"

    def test_submit_sorting_job_with_overrides(self, tmp_path):
        """
        submit_sorting_job applies config_overrides.

        Tests:
            (Test Case 1) Overrides are applied without error.
        """
        session, _, _ = self._make_session()
        rec = tmp_path / "recording.h5"
        rec.write_bytes(b"fake recording")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_sorting_job(
            recording_paths=[str(rec)],
            config=None,
            config_overrides={"freq_min": 200},
            job_spec=job_spec,
        )
        assert result.job_type == "sorting"

    def test_submit_sorting_job_invalid_preset_raises(self, tmp_path):
        """
        submit_sorting_job raises ValueError for unknown preset name.

        Tests:
            (Test Case 1) 'nonexistent' preset raises ValueError.
        """
        session, _, _ = self._make_session()
        rec = tmp_path / "recording.h5"
        rec.write_bytes(b"fake recording")

        job_spec = validate_job_spec(_example_payload())
        with pytest.raises(ValueError, match="Unknown sorting preset"):
            session.submit_sorting_job(
                recording_paths=[str(rec)],
                config="nonexistent",
                job_spec=job_spec,
            )

    def test_submit_sorting_job_missing_recording_raises(self):
        """
        submit_sorting_job raises FileNotFoundError for missing recording.

        Tests:
            (Test Case 1) Non-existent recording path raises.
        """
        session, _, _ = self._make_session()
        job_spec = validate_job_spec(_example_payload())
        with pytest.raises(FileNotFoundError, match="Recording file not found"):
            session.submit_sorting_job(
                recording_paths=["/nonexistent/recording.h5"],
                config=None,
                job_spec=job_spec,
            )

    def test_submit_sorting_job_container_command(self, tmp_path):
        """
        submit_sorting_job sets the sorting entrypoint as container command.

        Tests:
            (Test Case 1) Manifest contains the sorting entrypoint module.
        """
        session, _, _ = self._make_session()
        rec = tmp_path / "recording.h5"
        rec.write_bytes(b"fake recording")

        job_spec = validate_job_spec(_example_payload())
        result = session.submit_sorting_job(
            recording_paths=[str(rec)],
            config=None,
            job_spec=job_spec,
        )
        assert "spikelab.batch_jobs.entrypoints.sorting" in result.manifest_yaml


# ---------------------------------------------------------------------------
# RunSession: retrieve_result tests
# ---------------------------------------------------------------------------


class TestRunSessionRetrieve:
    def _make_session(self):
        """Build a RunSession with fully mocked backend/storage."""
        from spikelab.batch_jobs.session import RunSession

        profile = ClusterProfile(name="test")
        backend = MagicMock(spec=KubernetesBatchJobBackend)

        storage = MagicMock(spec=S3StorageClient)
        storage.output_prefix_for_run.return_value = "s3://test/outputs/run/"
        storage.logs_prefix_for_run.return_value = "s3://test/logs/run/"

        session = RunSession(
            profile=profile,
            backend=backend,
            storage_client=storage,
            credentials=MagicMock(),
        )
        return session, storage

    def test_retrieve_workspace_result(self, tmp_path):
        """
        retrieve_result downloads and loads workspace for workspace jobs.

        Tests:
            (Test Case 1) Calls download_output for .h5 and .json files.
            (Test Case 2) Returns an AnalysisWorkspace.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, storage = self._make_session()

        # Pre-create the workspace files that download_output would produce
        ws = AnalysisWorkspace(name="result-ws")
        base = str(tmp_path / "workspace")
        ws.save(base)

        # Mock download_output to be a no-op (files already exist)
        storage.download_output.side_effect = lambda **kwargs: str(
            tmp_path / kwargs["filename"]
        )

        submit_result = SubmitResult(
            job_name="test-job",
            manifest_yaml="",
            run_id="run-1",
            uploaded_input_uri="s3://test/inputs/run-1/bundle.zip",
            output_prefix="s3://test/outputs/run-1/",
            logs_prefix="s3://test/logs/run-1/",
            job_type="workspace",
        )

        result_ws = session.retrieve_result(submit_result, str(tmp_path))
        assert isinstance(result_ws, AnalysisWorkspace)
        assert storage.download_output.call_count == 2

    def test_retrieve_sorting_result(self, tmp_path):
        """
        retrieve_result builds workspace from sorting pickle outputs.

        Tests:
            (Test Case 1) Downloads all files from output prefix.
            (Test Case 2) SpikeData pickles are loaded into workspace namespaces.
        """
        import pickle

        import numpy as np
        from spikelab.spikedata.spikedata import SpikeData
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, storage = self._make_session()

        # Create a SpikeData pickle that will be "downloaded"
        sd = SpikeData([np.array([1.0, 2.0]), np.array([3.0])], length=10.0)
        pkl_path = tmp_path / "rec1_curated.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(sd, f)

        # Mock list_output_files to return one pickle
        storage.list_output_files.return_value = ["pfx/outputs/run-1/rec1_curated.pkl"]
        storage.output_prefix_for_run.return_value = "s3://bucket/pfx/outputs/run-1/"

        # Mock download_file to copy the pickle
        def fake_download(*, s3_uri, local_path):
            import shutil

            shutil.copy2(str(pkl_path), local_path)
            return local_path

        storage.download_file.side_effect = fake_download

        submit_result = SubmitResult(
            job_name="sort-job",
            manifest_yaml="",
            run_id="run-1",
            uploaded_input_uri="s3://bucket/pfx/inputs/run-1/bundle.zip",
            output_prefix="s3://bucket/pfx/outputs/run-1/",
            logs_prefix="s3://bucket/pfx/logs/run-1/",
            job_type="sorting",
        )

        result_ws = session.retrieve_result(submit_result, str(tmp_path / "out"))
        assert isinstance(result_ws, AnalysisWorkspace)
        # SpikeData should be stored under namespace derived from filename
        sd_loaded = result_ws.get("rec1_curated", "spikedata")
        assert sd_loaded is not None
        assert sd_loaded.N == 2

    def test_retrieve_prepared_raises(self, tmp_path):
        """
        retrieve_result raises ValueError for 'prepared' job type.

        Tests:
            (Test Case 1) Prepared jobs have no retrievable outputs.
        """
        session, _ = self._make_session()
        submit_result = SubmitResult(
            job_name="prep-job",
            manifest_yaml="",
            run_id="run-1",
            uploaded_input_uri="",
            output_prefix="",
            logs_prefix="",
            job_type="prepared",
        )
        with pytest.raises(ValueError, match="Cannot retrieve results"):
            session.retrieve_result(submit_result, str(tmp_path))

    def test_retrieve_sorting_no_files_raises(self, tmp_path):
        """
        retrieve_result raises FileNotFoundError when no output files exist.

        Tests:
            (Test Case 1) Empty output prefix raises.
        """
        session, storage = self._make_session()
        storage.list_output_files.return_value = []

        submit_result = SubmitResult(
            job_name="sort-job",
            manifest_yaml="",
            run_id="run-1",
            uploaded_input_uri="s3://bucket/inputs/run-1/bundle.zip",
            output_prefix="s3://bucket/outputs/run-1/",
            logs_prefix="s3://bucket/logs/run-1/",
            job_type="sorting",
        )
        with pytest.raises(FileNotFoundError, match="No output files found"):
            session.retrieve_result(submit_result, str(tmp_path))


# ---------------------------------------------------------------------------
# RunSession: _inject_env and _resolve_sorting_config tests
# ---------------------------------------------------------------------------


class TestRunSessionHelpers:
    def test_inject_env_adds_vars(self):
        """
        _inject_env adds environment variables to job spec container.

        Tests:
            (Test Case 1) New env vars are present.
            (Test Case 2) Existing env vars are preserved.
        """
        from spikelab.batch_jobs.session import RunSession

        job_spec = validate_job_spec(_example_payload())
        enriched = RunSession._inject_env(
            job_spec, {"NEW_VAR": "new_value", "OTHER": "other_value"}
        )
        assert enriched.container.env["NEW_VAR"] == "new_value"
        assert enriched.container.env["OTHER"] == "other_value"
        # Original env preserved
        assert (
            enriched.container.env["OUTPUT_PREFIX"] == "s3://test-bucket/test-prefix/"
        )

    def test_inject_env_overrides_existing(self):
        """
        _inject_env overrides existing env vars with new values.

        Tests:
            (Test Case 1) Existing key is overwritten by new value.
        """
        from spikelab.batch_jobs.session import RunSession

        job_spec = validate_job_spec(_example_payload())
        enriched = RunSession._inject_env(
            job_spec, {"OUTPUT_PREFIX": "s3://new-bucket/new-prefix/"}
        )
        assert enriched.container.env["OUTPUT_PREFIX"] == "s3://new-bucket/new-prefix/"

    def test_resolve_sorting_config_none(self):
        """
        _resolve_sorting_config with None returns default config dict.

        Tests:
            (Test Case 1) Returns a dict with expected sub-config keys.
        """
        from spikelab.batch_jobs.session import RunSession

        config_dict = RunSession._resolve_sorting_config(None, None)
        assert isinstance(config_dict, dict)
        assert "recording" in config_dict
        assert "sorter" in config_dict
        assert "curation" in config_dict

    def test_resolve_sorting_config_preset_string(self):
        """
        _resolve_sorting_config resolves a preset name string.

        Tests:
            (Test Case 1) 'kilosort4' resolves without error.
            (Test Case 2) Sorter name is 'kilosort4' in the output.
        """
        from spikelab.batch_jobs.session import RunSession

        config_dict = RunSession._resolve_sorting_config("kilosort4", None)
        assert config_dict["sorter"]["sorter_name"] == "kilosort4"

    def test_resolve_sorting_config_with_overrides(self):
        """
        _resolve_sorting_config applies overrides to the config.

        Tests:
            (Test Case 1) freq_min override is reflected in output.
        """
        from spikelab.batch_jobs.session import RunSession

        config_dict = RunSession._resolve_sorting_config(None, {"freq_min": 200})
        assert config_dict["recording"]["freq_min"] == 200

    def test_resolve_sorting_config_invalid_preset_raises(self):
        """
        _resolve_sorting_config raises ValueError for unknown preset.

        Tests:
            (Test Case 1) 'nonexistent' raises ValueError.
        """
        from spikelab.batch_jobs.session import RunSession

        with pytest.raises(ValueError, match="Unknown sorting preset"):
            RunSession._resolve_sorting_config("nonexistent", None)


# ---------------------------------------------------------------------------
# Entrypoint tests (mocked I/O)
# ---------------------------------------------------------------------------


class TestWorkspaceEntrypoint:
    def test_require_env_raises_on_missing(self):
        """
        _require_env raises RuntimeError for missing env var.

        Tests:
            (Test Case 1) Missing env var raises with descriptive message.
        """
        from spikelab.batch_jobs.entrypoints.workspace import _require_env

        with pytest.raises(RuntimeError, match="INPUT_URI"):
            _require_env("INPUT_URI")

    def test_require_env_returns_value(self, monkeypatch):
        """
        _require_env returns the env var value when set.

        Tests:
            (Test Case 1) Set env var is returned.
        """
        from spikelab.batch_jobs.entrypoints.workspace import _require_env

        monkeypatch.setenv("INPUT_URI", "s3://bucket/input.zip")
        assert _require_env("INPUT_URI") == "s3://bucket/input.zip"

    def test_main_runs_script_with_workspace(self, tmp_path, monkeypatch):
        """
        Workspace entrypoint loads workspace, runs script, uploads result.

        Tests:
            (Test Case 1) Script receives workspace object.
            (Test Case 2) Updated workspace is uploaded as .h5 + .json.
        """
        import json
        import zipfile

        from spikelab.workspace.workspace import AnalysisWorkspace

        # Create a workspace and save it
        ws = AnalysisWorkspace(name="entry-test")
        ws_base = str(tmp_path / "workspace")
        ws.save(ws_base)

        # Create a script that modifies the workspace
        script = tmp_path / "my_script.py"
        script.write_text(
            "import numpy as np\nworkspace.store('ns', 'marker', np.array([1, 2, 3]))\n",
            encoding="utf-8",
        )

        # Create bundle zip
        bundle_dir = tmp_path / "bundle" / "run-1"
        bundle_dir.mkdir(parents=True)
        import shutil

        shutil.copy2(f"{ws_base}.h5", bundle_dir / "workspace.h5")
        shutil.copy2(f"{ws_base}.json", bundle_dir / "workspace.json")
        shutil.copy2(str(script), bundle_dir / "my_script.py")
        # Write manifest
        manifest = {"run_id": "run-1", "output_format": "workspace", "files": []}
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        zip_path = str(tmp_path / "bundle.zip")
        shutil.make_archive(
            str(tmp_path / "bundle"),
            "zip",
            root_dir=str(tmp_path / "bundle"),
            base_dir="run-1",
        )

        # Mock S3StorageClient
        upload_calls = []

        def fake_download(*, s3_uri, local_path):
            shutil.copy2(zip_path, local_path)
            return local_path

        def fake_upload(*, local_path, s3_uri):
            upload_calls.append((local_path, s3_uri))
            return s3_uri

        mock_storage = MagicMock()
        mock_storage.download_file.side_effect = fake_download
        mock_storage.upload_file.side_effect = fake_upload

        monkeypatch.setenv("INPUT_URI", "s3://bucket/input/bundle.zip")
        monkeypatch.setenv("OUTPUT_PREFIX", "s3://bucket/outputs/run-1/")
        monkeypatch.setenv("SCRIPT_NAME", "my_script.py")
        monkeypatch.setattr(
            "spikelab.batch_jobs.storage_s3.S3StorageClient",
            lambda **kwargs: mock_storage,
        )

        from spikelab.batch_jobs.entrypoints.workspace import main

        main()

        # Verify uploads happened (workspace.h5 + workspace.json)
        assert len(upload_calls) == 2
        uploaded_uris = {uri for _, uri in upload_calls}
        assert "s3://bucket/outputs/run-1/workspace.h5" in uploaded_uris
        assert "s3://bucket/outputs/run-1/workspace.json" in uploaded_uris


class TestSortingEntrypoint:
    def test_require_env_raises_on_missing(self):
        """
        _require_env raises RuntimeError for missing env var.

        Tests:
            (Test Case 1) Missing env var raises with descriptive message.
        """
        from spikelab.batch_jobs.entrypoints.sorting import _require_env

        with pytest.raises(RuntimeError, match="INPUT_URI"):
            _require_env("INPUT_URI")

    def test_reconstruct_config(self):
        """
        _reconstruct_config rebuilds SortingPipelineConfig from dict.

        Tests:
            (Test Case 1) Reconstructed config matches original.
            (Test Case 2) Sub-configs have correct field values.
        """
        import dataclasses

        from spikelab.batch_jobs.entrypoints.sorting import _reconstruct_config
        from spikelab.spike_sorting.config import SortingPipelineConfig

        original = SortingPipelineConfig()
        original_dict = dataclasses.asdict(original)
        reconstructed = _reconstruct_config(original_dict)

        assert isinstance(reconstructed, SortingPipelineConfig)
        assert reconstructed.recording.freq_min == original.recording.freq_min
        assert reconstructed.sorter.sorter_name == original.sorter.sorter_name
        assert reconstructed.curation.fr_min == original.curation.fr_min

    def test_reconstruct_config_with_overrides(self):
        """
        _reconstruct_config preserves non-default values.

        Tests:
            (Test Case 1) Custom freq_min is preserved after roundtrip.
        """
        import dataclasses

        from spikelab.batch_jobs.entrypoints.sorting import _reconstruct_config
        from spikelab.spike_sorting.config import SortingPipelineConfig

        config = SortingPipelineConfig()
        config = config.override(freq_min=200)
        config_dict = dataclasses.asdict(config)
        reconstructed = _reconstruct_config(config_dict)
        assert reconstructed.recording.freq_min == 200


# ---------------------------------------------------------------------------
# Edge case tests — batch_jobs (HIGH and MEDIUM severity findings)
# ---------------------------------------------------------------------------


class TestVolumeMountSpec:
    """Edge cases for VolumeMountSpec._validate_source."""

    def test_empty_string_secret_name_rejected(self):
        """Empty string secret_name (falsy) should fail validation."""
        with pytest.raises(PydanticValidationError, match="secret_name or pvc_name"):
            VolumeMountSpec(name="vol", mount_path="/mnt", secret_name="")

    def test_empty_string_pvc_name_rejected(self):
        """Empty string pvc_name (falsy) should fail validation."""
        with pytest.raises(PydanticValidationError, match="secret_name or pvc_name"):
            VolumeMountSpec(name="vol", mount_path="/mnt", pvc_name="")

    def test_empty_string_both_sources_rejected(self):
        """Both empty string sources should fail validation."""
        with pytest.raises(PydanticValidationError, match="secret_name or pvc_name"):
            VolumeMountSpec(name="vol", mount_path="/mnt", secret_name="", pvc_name="")


class TestJobSpecNamePrefix:
    """Edge cases for JobSpec._validate_name_prefix."""

    def test_unicode_characters_pass_through(self):
        """Unicode letters pass isalnum() but are invalid for K8s RFC 1123 labels.

        This documents a known gap: the sanitizer uses Python's isalnum()
        which accepts Unicode letters, but K8s labels require ASCII only.
        """
        payload = _example_payload()
        payload["name_prefix"] = "análysis-jöb"
        job_spec = validate_job_spec(payload)
        # Unicode chars pass through isalnum() check — this is the gap
        assert "á" in job_spec.name_prefix or "ö" in job_spec.name_prefix
        # Verify the result is NOT valid ASCII (documenting the issue)
        with pytest.raises(UnicodeEncodeError):
            job_spec.name_prefix.encode("ascii")

    def test_trailing_hyphens_after_truncation(self):
        """Truncation at 40 chars can leave trailing hyphens.

        This documents a known gap: strip("-") runs before [:40], so
        truncation can expose interior hyphens at the boundary.
        """
        payload = _example_payload()
        # Create a prefix where position 40 falls right after hyphens
        payload["name_prefix"] = "a" * 37 + "---xyz"
        job_spec = validate_job_spec(payload)
        assert len(job_spec.name_prefix) <= 40
        # Documenting the gap: trailing hyphens remain after truncation
        assert job_spec.name_prefix.endswith("-")


class TestSleepDetectionMore:
    """Additional edge cases for _contains_disallowed_sleep."""

    def test_sleep_in_quoted_string_not_flagged(self):
        """sleep inside a quoted argument is not flagged (token-based matching)."""
        result = _contains_disallowed_sleep(
            ["sh", "-c"], ['echo "do not sleep infinity"']
        )
        # Token split produces ["echo", '"do', "not", "sleep", 'infinity"']
        # "sleep" matches but 'infinity"' (with trailing quote) doesn't
        # match "infinity" exactly — so this is correctly not flagged.
        assert result is False


class TestPolicySummarizePreflight:
    """Edge cases for summarize_preflight aggregation."""

    def test_multiple_findings_same_level_all_in_text(self):
        """Multiple WARN findings should all appear in the summary text."""
        from spikelab.batch_jobs.policy import PolicyFinding

        findings = [
            PolicyFinding("check_a", "WARN", "Warning A"),
            PolicyFinding("check_b", "WARN", "Warning B"),
            PolicyFinding("check_c", "PASS", "Passed C"),
        ]
        level, text = summarize_preflight(findings)
        assert level == "WARN"
        assert "check_a" in text
        assert "check_b" in text
        assert "check_c" in text
        # All three lines present
        assert text.count("\n") == 2

    def test_warn_only_findings(self):
        """All WARN findings produce aggregate WARN level."""
        from spikelab.batch_jobs.policy import PolicyFinding

        findings = [
            PolicyFinding("c1", "WARN", "w1"),
            PolicyFinding("c2", "WARN", "w2"),
        ]
        level, _ = summarize_preflight(findings)
        assert level == "WARN"

    def test_all_pass_findings(self):
        """All PASS findings produce aggregate PASS level."""
        from spikelab.batch_jobs.policy import PolicyFinding

        findings = [
            PolicyFinding("c1", "PASS", "p1"),
            PolicyFinding("c2", "PASS", "p2"),
        ]
        level, _ = summarize_preflight(findings)
        assert level == "PASS"


class TestWaitForCompletion:
    """Edge cases for RunSession.wait_for_completion."""

    def test_zero_wait_immediately_complete_returns_timeout(self):
        """max_wait_seconds=0 returns Timeout even if job is already complete."""
        from spikelab.batch_jobs.session import RunSession

        profile = ClusterProfile(name="test")
        backend = MagicMock(spec=KubernetesBatchJobBackend)
        backend.job_status.return_value = "Complete"
        storage = MagicMock(spec=S3StorageClient)
        session = RunSession(
            profile=profile,
            backend=backend,
            storage_client=storage,
            credentials=MagicMock(),
        )
        # max_wait_seconds=0 means the deadline is already in the past
        state = session.wait_for_completion(
            job_name="test-job", max_wait_seconds=0, poll_interval_seconds=0
        )
        assert state == "Timeout"


class TestKubernetesBatchJobBackendK8sClientPath:
    """Tests for K8s client code paths (HIGH severity — previously untested)."""

    def test_apply_manifest_k8s_client_from_string(self):
        """apply_manifest via K8s client parses YAML string and calls create_namespaced_job."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        manifest = "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: my-job\n"
        result = backend.apply_manifest(manifest)

        mock_batch_api.create_namespaced_job.assert_called_once()
        call_kwargs = mock_batch_api.create_namespaced_job.call_args
        assert call_kwargs[1]["namespace"] == "test-ns"
        assert result == "my-job"

    def test_apply_manifest_k8s_client_from_file(self, tmp_path):
        """apply_manifest via K8s client reads YAML file and calls create_namespaced_job."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        manifest_file = tmp_path / "job.yaml"
        manifest_file.write_text(
            "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: file-job\n",
            encoding="utf-8",
        )
        result = backend.apply_manifest(str(manifest_file))

        mock_batch_api.create_namespaced_job.assert_called_once()
        assert result == "file-job"

    def test_apply_manifest_k8s_client_invalid_yaml(self):
        """apply_manifest via K8s client with invalid YAML raises."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        # YAML that parses as a string, not a dict — will fail on metadata access
        with pytest.raises((TypeError, KeyError, AttributeError)):
            backend.apply_manifest("just a plain string without yaml structure")

    def test_apply_manifest_k8s_client_missing_metadata_name(self):
        """apply_manifest via K8s client raises when metadata.name is missing."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        manifest = "apiVersion: batch/v1\nkind: Job\nmetadata:\n  labels: {}\n"
        with pytest.raises(KeyError):
            backend.apply_manifest(manifest)

    def test_delete_job_k8s_client(self):
        """delete_job via K8s client calls delete_namespaced_job."""
        mock_client_module = MagicMock()
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        with patch("spikelab.batch_jobs.backend_k8s.client", mock_client_module):
            backend.delete_job("my-job")
        mock_batch_api.delete_namespaced_job.assert_called_once()
        call_kwargs = mock_batch_api.delete_namespaced_job.call_args[1]
        assert call_kwargs["name"] == "my-job"
        assert call_kwargs["namespace"] == "test-ns"

    def test_job_status_k8s_client_complete(self):
        """job_status via K8s client returns 'Complete' for succeeded job."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        status_mock = MagicMock()
        status_mock.status.to_dict.return_value = {"succeeded": 1}
        mock_batch_api.read_namespaced_job_status.return_value = status_mock

        assert backend.job_status("my-job") == "Complete"

    def test_job_status_k8s_client_failed(self):
        """job_status via K8s client returns 'Failed' for failed job."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        status_mock = MagicMock()
        status_mock.status.to_dict.return_value = {"failed": 1}
        mock_batch_api.read_namespaced_job_status.return_value = status_mock

        assert backend.job_status("my-job") == "Failed"

    def test_job_status_k8s_client_running(self):
        """job_status via K8s client returns 'Running' for active job."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        status_mock = MagicMock()
        status_mock.status.to_dict.return_value = {"active": 1}
        mock_batch_api.read_namespaced_job_status.return_value = status_mock

        assert backend.job_status("my-job") == "Running"

    def test_job_status_k8s_client_pending(self):
        """job_status via K8s client returns 'Pending' for empty status."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        status_mock = MagicMock()
        status_mock.status.to_dict.return_value = {}
        mock_batch_api.read_namespaced_job_status.return_value = status_mock

        assert backend.job_status("my-job") == "Pending"

    def test_job_status_k8s_client_status_obj_none(self):
        """job_status via K8s client returns 'Pending' when status_obj is None."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        mock_batch_api.read_namespaced_job_status.return_value = None

        assert backend.job_status("my-job") == "Pending"

    def test_job_status_k8s_client_status_attr_none(self):
        """job_status via K8s client returns 'Pending' when status_obj.status is None."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        status_mock = MagicMock()
        status_mock.status = None
        mock_batch_api.read_namespaced_job_status.return_value = status_mock

        assert backend.job_status("my-job") == "Pending"

    def test_pods_for_job_k8s_client(self):
        """pods_for_job via K8s client returns pod names."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_core_api = MagicMock()
        backend._core_api = mock_core_api

        pod1 = MagicMock()
        pod1.metadata.name = "pod-abc"
        pod2 = MagicMock()
        pod2.metadata.name = "pod-def"
        mock_core_api.list_namespaced_pod.return_value = MagicMock(items=[pod1, pod2])

        pods = backend.pods_for_job("my-job")
        assert pods == ["pod-abc", "pod-def"]
        mock_core_api.list_namespaced_pod.assert_called_once_with(
            namespace="test-ns", label_selector="job-name=my-job"
        )


class TestStreamLogs:
    """Tests for KubernetesBatchJobBackend.stream_logs (HIGH — zero coverage)."""

    def test_stream_logs_kubectl_follow(self, monkeypatch):
        """stream_logs with follow=True via kubectl uses -f flag."""
        import subprocess

        backend = KubernetesBatchJobBackend(namespace="test-ns")
        backend._core_api = None

        mock_process = MagicMock()
        mock_process.stdout = iter(["line 1\n", "line 2\n"])

        def fake_popen(cmd, **kwargs):
            assert "-f" in cmd
            assert "test-pod" in cmd
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        lines = list(backend.stream_logs("test-pod", follow=True))
        assert lines == ["line 1", "line 2"]

    def test_stream_logs_kubectl_no_follow(self, monkeypatch):
        """stream_logs with follow=False via kubectl does not use -f flag."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        backend._core_api = None

        mock_process = MagicMock()
        mock_process.stdout = iter(["log line\n"])

        def fake_popen(cmd, **kwargs):
            assert "-f" not in cmd
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        lines = list(backend.stream_logs("test-pod", follow=False))
        assert lines == ["log line"]

    def test_stream_logs_k8s_client_follow_with_watch(self):
        """stream_logs with follow=True via K8s client uses watch.Watch."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_core_api = MagicMock()
        backend._core_api = mock_core_api

        mock_watcher = MagicMock()
        mock_watcher.stream.return_value = iter(["log line 1", "log line 2"])

        with patch("spikelab.batch_jobs.backend_k8s.watch") as mock_watch_module:
            mock_watch_module.Watch.return_value = mock_watcher
            lines = list(backend.stream_logs("test-pod", follow=True))

        assert lines == ["log line 1", "log line 2"]
        mock_watcher.stream.assert_called_once()

    def test_stream_logs_k8s_client_follow_without_watch(self):
        """stream_logs with follow=True but watch is None falls back to non-follow."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_core_api = MagicMock()
        backend._core_api = mock_core_api
        mock_core_api.read_namespaced_pod_log.return_value = "line a\nline b"

        with patch("spikelab.batch_jobs.backend_k8s.watch", None):
            lines = list(backend.stream_logs("test-pod", follow=True))

        assert lines == ["line a", "line b"]

    def test_stream_logs_k8s_client_no_follow(self):
        """stream_logs with follow=False via K8s client reads log text."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_core_api = MagicMock()
        backend._core_api = mock_core_api
        mock_core_api.read_namespaced_pod_log.return_value = "hello\nworld"

        lines = list(backend.stream_logs("test-pod", follow=False))
        assert lines == ["hello", "world"]


class TestK8sBackendConfigException:
    """Edge case for KubernetesBatchJobBackend.__init__ config failure."""

    def test_config_exception_falls_back_to_kubectl(self):
        """ConfigException during init leaves _batch_api as None (kubectl fallback)."""
        mock_client = MagicMock()
        mock_config = MagicMock()
        mock_config.ConfigException = type("ConfigException", (Exception,), {})
        mock_config.load_kube_config.side_effect = mock_config.ConfigException("fail")

        with patch.dict(
            "sys.modules",
            {
                "kubernetes": MagicMock(),
                "kubernetes.client": mock_client,
                "kubernetes.config": mock_config,
                "kubernetes.watch": MagicMock(),
            },
        ):
            with patch("spikelab.batch_jobs.backend_k8s.client", mock_client):
                with patch("spikelab.batch_jobs.backend_k8s.config", mock_config):
                    backend = KubernetesBatchJobBackend(namespace="test")
                    assert backend._batch_api is None
                    assert backend._core_api is None


class TestCli:
    """Edge cases for CLI module."""

    def test_load_payload_json_file(self, tmp_path):
        """_load_payload reads a JSON config file."""
        config_path = tmp_path / "job.json"
        import json

        payload = _example_payload()
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = cli._load_payload(str(config_path))
        assert loaded["name_prefix"] == "analysis-job"
        assert isinstance(loaded, dict)

    def test_load_payload_non_dict_raises(self, tmp_path):
        """_load_payload raises ValueError for non-dict content."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must contain an object"):
            cli._load_payload(str(config_path))

    def test_load_payload_non_dict_json_raises(self, tmp_path):
        """_load_payload raises ValueError for JSON array content."""
        import json

        config_path = tmp_path / "bad.json"
        config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with pytest.raises(ValueError, match="must contain an object"):
            cli._load_payload(str(config_path))

    def test_apply_image_selection_override_takes_precedence(self):
        """image_override takes precedence over image_profile."""
        payload = {
            "container": {
                "image": "existing:v1",
                "command": ["python"],
                "args": [],
                "env": {},
            }
        }
        profile = ClusterProfile(
            name="test",
            default_images={"gpu": "ghcr.io/example/gpu:latest"},
        )
        updated = cli._apply_image_selection(
            payload,
            profile=profile,
            image_profile="gpu",
            image_override="my-custom:v2",
        )
        assert updated["container"]["image"] == "my-custom:v2"

    def test_apply_image_selection_profile_not_found(self):
        """Image profile not in default_images leaves container image unchanged."""
        payload = {
            "container": {
                "image": "original:v1",
                "command": ["python"],
                "args": [],
                "env": {},
            }
        }
        profile = ClusterProfile(name="test", default_images={})
        updated = cli._apply_image_selection(
            payload,
            profile=profile,
            image_profile="nonexistent",
            image_override=None,
        )
        # Image not changed because profile has no such key
        assert updated["container"]["image"] == "original:v1"

    def test_apply_image_selection_container_not_dict_raises(self):
        """container field that is not a dict raises ValueError."""
        payload = {"container": "not-a-dict"}
        profile = ClusterProfile(name="test")
        with pytest.raises(ValueError, match="container.*must be an object"):
            cli._apply_image_selection(
                payload,
                profile=profile,
                image_profile=None,
                image_override=None,
            )

    def test_cmd_deploy_render_only_with_output_manifest(
        self, monkeypatch, tmp_path, capsys
    ):
        """render_only=True with output_manifest writes to file."""
        config_path = tmp_path / "job.yaml"
        config_path.write_text(
            "name_prefix: analysis-job\nnamespace: default\n"
            "container:\n  image: ghcr.io/example/image:latest\n"
            "  command: ['python']\n  args: ['-m', 'run']\n  env: {}\n"
            "resources:\n  requests_cpu: '1'\n  requests_memory: 2Gi\n"
            "  limits_cpu: '1'\n  limits_memory: 2Gi\n"
            "  requests_gpu: 0\n  limits_gpu: 0\n  node_selector: {}\n"
            "volumes: []\n",
            encoding="utf-8",
        )

        class DummySession:
            def render_manifest(self, *, job_name, job_spec, run_id):
                return f"metadata:\n  name: {job_name}\n"

        monkeypatch.setattr(
            cli,
            "_load_profile",
            lambda *args, **kwargs: ClusterProfile(name="test"),
        )
        monkeypatch.setattr(
            cli, "_build_session", lambda *args, **kwargs: DummySession()
        )

        output_file = tmp_path / "output.yaml"
        args = SimpleNamespace(
            profile="defaults",
            profile_file=None,
            kubeconfig=None,
            job_config=str(config_path),
            allow_policy_risk=False,
            render_only=True,
            output_manifest=str(output_file),
            wait=False,
            max_wait_seconds=0,
            follow_logs=False,
            image_profile=None,
            image=None,
        )
        exit_code = cli._cmd_deploy(args)
        assert exit_code == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "metadata:" in content
        out = capsys.readouterr().out
        assert f"MANIFEST_PATH={output_file}" in out

    def test_cmd_deploy_validation_error_without_errors_attr(
        self, monkeypatch, tmp_path
    ):
        """Validation error without .errors() attribute uses str(exc)."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(
            "name_prefix: analysis-job\nnamespace: default\n"
            "container:\n  image: ''\n  command: []\n  args: []\n  env: {}\n"
            "resources:\n  requests_cpu: '1'\n  requests_memory: 2Gi\n"
            "  limits_cpu: '1'\n  limits_memory: 2Gi\n"
            "  requests_gpu: 0\n  limits_gpu: 0\n  node_selector: {}\n"
            "volumes: []\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            cli,
            "_load_profile",
            lambda *args, **kwargs: ClusterProfile(name="test"),
        )
        monkeypatch.setattr(
            cli,
            "_build_session",
            lambda *args, **kwargs: MagicMock(),
        )

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
        with pytest.raises(SystemExit, match="Invalid job config"):
            cli._cmd_deploy(args)

    def test_cmd_logs_no_pods_raises(self, monkeypatch):
        """_cmd_logs raises SystemExit when no pods found."""
        mock_session = MagicMock()
        mock_session.backend.pods_for_job.return_value = []

        monkeypatch.setattr(
            cli,
            "_load_profile",
            lambda *args, **kwargs: ClusterProfile(name="test"),
        )
        monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: mock_session)

        args = SimpleNamespace(
            profile="defaults",
            profile_file=None,
            kubeconfig=None,
            job_name="test-job",
            follow=False,
        )
        with pytest.raises(SystemExit, match="No pods found"):
            cli._cmd_logs(args)

    def test_cmd_status(self, monkeypatch, capsys):
        """_cmd_status prints job status."""
        mock_session = MagicMock()
        mock_session.backend.job_status.return_value = "Running"

        monkeypatch.setattr(
            cli,
            "_load_profile",
            lambda *args, **kwargs: ClusterProfile(name="test"),
        )
        monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: mock_session)

        args = SimpleNamespace(
            profile="defaults",
            profile_file=None,
            kubeconfig=None,
            job_name="test-job",
        )
        exit_code = cli._cmd_status(args)
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "JOB_STATUS=Running" in out

    def test_cmd_delete(self, monkeypatch, capsys):
        """_cmd_delete prints deleted job name."""
        mock_session = MagicMock()

        monkeypatch.setattr(
            cli,
            "_load_profile",
            lambda *args, **kwargs: ClusterProfile(name="test"),
        )
        monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: mock_session)

        args = SimpleNamespace(
            profile="defaults",
            profile_file=None,
            kubeconfig=None,
            job_name="delete-me",
        )
        exit_code = cli._cmd_delete(args)
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "DELETED_JOB=delete-me" in out
        mock_session.backend.delete_job.assert_called_once_with("delete-me")


class TestTemplating:
    """Edge cases for the templating module."""

    def test_sanitize_yaml_value_strips_unsafe_chars(self):
        """_sanitize_yaml_value removes newlines, tabs, quotes, backslashes."""
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        result = _sanitize_yaml_value('hello\nworld\t"test\\value')
        assert "\n" not in result
        assert "\t" not in result
        assert '"' not in result
        assert "\\" not in result
        assert result == "helloworldtestvalue"

    def test_sanitize_yaml_value_injection_attempt(self):
        """YAML injection via label values is stripped."""
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        malicious = 'value"\ninjected_key: injected_value'
        result = _sanitize_yaml_value(malicious)
        assert "\n" not in result
        assert '"' not in result

    def test_build_pod_volumes_mount_with_no_name_skipped(self):
        """Mounts with no name are silently dropped."""
        from spikelab.batch_jobs.templating import _build_pod_volumes

        mounts = [
            {"name": "vol1", "mount_path": "/a", "secret_name": "sec1"},
            {"mount_path": "/b", "secret_name": "sec2"},  # no name
            {"name": "", "mount_path": "/c", "secret_name": "sec3"},  # empty name
        ]
        volumes = _build_pod_volumes(mounts)
        names = [v["name"] for v in volumes]
        assert "vol1" in names
        # Empty string name may or may not be included; no-name mount is skipped
        assert len(volumes) <= 2

    def test_build_template_context_empty_namespace_fallback(self):
        """Empty string namespace falls back to profile namespace."""
        payload = _example_payload()
        payload["namespace"] = ""
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test", namespace="fallback-ns")
        context = build_template_context(
            job_name="test-ctx",
            job_spec=job_spec,
            profile=profile,
        )
        assert context["namespace"] == "fallback-ns"

    def test_build_template_context_empty_labels(self):
        """Empty labels from all sources still produces a dict."""
        payload = _example_payload()
        payload["labels"] = {}
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test", labels={})
        context = build_template_context(
            job_name="test-empty-labels",
            job_spec=job_spec,
            profile=profile,
            extra_labels=None,
        )
        assert isinstance(context["labels"], dict)


class TestValidation:
    """Edge cases for the validation module."""

    def test_summarize_validation_error_empty_loc(self):
        """summarize_validation_error handles error with empty loc tuple."""
        from spikelab.batch_jobs.validation import summarize_validation_error

        # Create a mock ValidationError with an empty loc
        mock_exc = MagicMock()
        mock_exc.errors.return_value = [
            {"loc": (), "msg": "custom error message"},
        ]
        result = summarize_validation_error(mock_exc)
        assert result == "custom error message"


class TestInitLazyImport:
    """Edge cases for batch_jobs.__init__.__getattr__."""

    def test_nonexistent_attribute_raises_attribute_error(self):
        """Accessing a non-existent attribute raises AttributeError."""
        import spikelab.batch_jobs as batch_jobs_pkg

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = batch_jobs_pkg.NoSuchThing

    def test_known_symbol_accessible(self):
        """Public symbols like JobSpec are accessible via lazy import."""
        import spikelab.batch_jobs as batch_jobs_pkg

        js = batch_jobs_pkg.JobSpec
        assert js is JobSpec


class TestCredentialExtended:
    """Additional edge cases for credentials module."""

    def test_resolve_credentials_empty_string_falls_back_to_env(self, monkeypatch):
        """Empty string explicit arg falls back to environment variable."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        creds = resolve_credentials(aws_access_key_id="")
        # Empty string is falsy, so `or` falls back to env var
        assert creds.aws_access_key_id == "env-key"

    def test_redact_sensitive_map_case_insensitive(self):
        """redact_sensitive_map matches keys case-insensitively (via upper)."""
        redacted = redact_sensitive_map(
            {
                "db_password": "hunter2",
                "Api_Secret": "key123",
                "auth_token": "tok-abc",
            }
        )
        assert redacted["db_password"] == "***REDACTED***"
        assert redacted["Api_Secret"] == "***REDACTED***"
        assert redacted["auth_token"] == "***REDACTED***"

    def test_redact_sensitive_map_no_false_positive_on_access_key_id(self):
        """AWS_ACCESS_KEY_ID should not be redacted (no SECRET/TOKEN/PASSWORD)."""
        redacted = redact_sensitive_map({"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE"})
        assert redacted["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
