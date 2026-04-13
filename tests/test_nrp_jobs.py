"""Tests for the NRP job-launcher package."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import yaml

if importlib.util.find_spec("pydantic") is None:
    pytest.skip("pydantic not installed", allow_module_level=True)

from spikelab.nrp_jobs.credentials import redact_sensitive_map
from spikelab.nrp_jobs.models import ClusterProfile, JobSpec
from spikelab.nrp_jobs.policy_nrp import evaluate_nrp_policy, summarize_preflight
from spikelab.nrp_jobs.templating import build_template_context, render_job_manifest
from spikelab.nrp_jobs.validation import validate_job_spec
import spikelab.nrp_jobs.cli as cli


def _example_payload():
    return {
        "name_prefix": "analysis-job",
        "namespace": "default",
        "labels": {"analysis": "spikelab"},
        "container": {
            "image": "ghcr.io/example/image:latest",
            "command": ["python"],
            "args": ["-m", "run"],
            "env": {"OUTPUT_PREFIX": "s3://braingeneers/ephys-analysis/"},
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


def test_validate_job_spec():
    job_spec = validate_job_spec(_example_payload())
    assert isinstance(job_spec, JobSpec)
    assert job_spec.container.image.startswith("ghcr.io/")


def test_render_job_manifest_contains_job_name():
    job_spec = validate_job_spec(_example_payload())
    profile = ClusterProfile(name="nrp")
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


def test_braingeneers_namespace_injects_required_mounts_without_forced_affinity():
    payload = _example_payload()
    payload["namespace"] = "braingeneers"
    payload["resources"]["requests_gpu"] = 1
    payload["resources"]["limits_gpu"] = 1
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(name="nrp")
    context = build_template_context(
        job_name="analysis-job-braingeneers",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    mounts = parsed["spec"]["template"]["spec"]["containers"][0].get("volumeMounts", [])
    mount_paths = {item["mountPath"] for item in mounts}
    assert "/root/.aws/credentials" in mount_paths
    assert "/root/.aws/.s3cfg" in mount_paths
    assert "/root/.kube" in mount_paths
    assert "affinity" not in parsed["spec"]["template"]["spec"]


def test_braingeneers_keeps_user_profile_affinity():
    payload = _example_payload()
    payload["namespace"] = "braingeneers"
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(
        name="nrp",
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
    )
    context = build_template_context(
        job_name="analysis-job-braingeneers",
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


def test_policy_blocks_sleep_infinity():
    payload = _example_payload()
    payload["container"]["args"] = ["sleep", "infinity"]
    job_spec = validate_job_spec(payload)
    findings = evaluate_nrp_policy(job_spec)
    level, _ = summarize_preflight(findings)
    assert level == "BLOCK"


def test_redaction_hides_sensitive_fields():
    redacted = redact_sensitive_map(
        {
            "AWS_ACCESS_KEY_ID": "abc",
            "AWS_SECRET_ACCESS_KEY": "super-secret",
            "NORMAL_FIELD": "ok",
        }
    )
    assert redacted["AWS_SECRET_ACCESS_KEY"] == "***REDACTED***"
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
                output_prefix="s3://braingeneers/ephys-analysis/outputs/run/",
                logs_prefix="s3://braingeneers/ephys-analysis/logs/run/",
            )

    monkeypatch.setattr(
        cli, "_load_profile", lambda *args, **kwargs: ClusterProfile(name="nrp")
    )
    monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: DummySession())
    args = SimpleNamespace(
        profile="nrp",
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
        name="nrp",
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
            name="nrp",
            default_images={
                "cpu": "ghcr.io/example/cpu:latest",
                "gpu": "ghcr.io/example/gpu:latest",
            },
        ),
    )
    monkeypatch.setattr(cli, "_build_session", lambda *args, **kwargs: DummySession())
    args = SimpleNamespace(
        profile="nrp",
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
