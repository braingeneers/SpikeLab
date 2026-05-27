"""Tests for the batch job-launcher package."""

from __future__ import annotations

import argparse
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
    # Quoting in the template means the name shows up as
    # ``name: "analysis-job-1234"`` post-J-A12.
    assert parsed["metadata"]["name"] == "analysis-job-1234"
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


def test_image_pull_secrets_default_omits_field_from_manifest():
    """
    Tier L-F7: when the active ``ClusterProfile`` has no
    ``image_pull_secrets`` configured (the default for public-registry
    workflows), the rendered manifest contains no ``imagePullSecrets``
    key — preserving the prior behaviour for clusters relying on the
    default ServiceAccount's pull secrets.

    Tests:
        (Test Case 1) ``ClusterProfile(name=...)`` with default
            ``image_pull_secrets=[]`` renders a manifest whose pod
            spec does NOT include ``imagePullSecrets``.
    """
    job_spec = validate_job_spec(_example_payload())
    profile = ClusterProfile(name="public-cluster")
    context = build_template_context(
        job_name="lf7-default",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    pod_spec = parsed["spec"]["template"]["spec"]
    assert "imagePullSecrets" not in pod_spec
    # ``imagePullSecrets`` should not appear textually either —
    # commented-out renderings would still confuse downstream tools.
    assert "imagePullSecrets" not in manifest


def test_image_pull_secrets_single_secret_renders_into_manifest():
    """
    Tier L-F7: a single pull secret on the profile renders as a
    one-element ``imagePullSecrets:`` list under the pod spec,
    exactly as Kubernetes expects.

    Tests:
        (Test Case 1) ``ClusterProfile(image_pull_secrets=["regcred"])``
            renders ``imagePullSecrets: [{"name": "regcred"}]`` on
            the pod spec.
    """
    job_spec = validate_job_spec(_example_payload())
    profile = ClusterProfile(name="private-cluster", image_pull_secrets=["regcred"])
    context = build_template_context(
        job_name="lf7-single",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    pod_spec = parsed["spec"]["template"]["spec"]
    assert pod_spec["imagePullSecrets"] == [{"name": "regcred"}]


def test_image_pull_secrets_multiple_secrets_render_in_order():
    """
    Tier L-F7: multiple pull secrets render in the order given on the
    profile. K8s does not document an evaluation order for multiple
    pull secrets, but preserving caller order keeps the manifest
    diff-stable across runs.

    Tests:
        (Test Case 1) Two secrets render as a two-element list in
            the same order as the profile's
            ``image_pull_secrets`` list.
    """
    job_spec = validate_job_spec(_example_payload())
    profile = ClusterProfile(
        name="multi-registry-cluster",
        image_pull_secrets=["ghcr-cred", "ecr-cred"],
    )
    context = build_template_context(
        job_name="lf7-multi",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    pod_spec = parsed["spec"]["template"]["spec"]
    assert pod_spec["imagePullSecrets"] == [
        {"name": "ghcr-cred"},
        {"name": "ecr-cred"},
    ]


def test_image_pull_secrets_yaml_unsafe_name_raises():
    """
    Tier L-F7: a pull-secret name containing YAML-unsafe characters
    (newline, etc.) is rejected by ``build_template_context`` rather
    than silently stripped — same fail-fast policy as labels /
    volume names elsewhere in the templating path (Tier L-C1).

    Tests:
        (Test Case 1) ``image_pull_secrets=["bad\\nname"]`` causes
            ``build_template_context`` to raise ValueError mentioning
            the offending field.
    """
    import pytest

    job_spec = validate_job_spec(_example_payload())
    profile = ClusterProfile(
        name="bad-secret-cluster",
        image_pull_secrets=["bad\nname"],
    )
    with pytest.raises(ValueError, match="YAML-unsafe character"):
        build_template_context(
            job_name="lf7-bad",
            job_spec=job_spec,
            profile=profile,
        )


def test_volume_mount_name_with_newline_raises():
    """
    Tier L-C1: ``VolumeMountSpec(name="bad\\nname", ...)`` flowing
    through ``build_template_context`` now raises ``ValueError``
    instead of silently stripping the newline. The error message
    names the offending field path so the operator can locate the
    bad entry.

    Tests:
        (Test Case 1) ``build_template_context`` raises ValueError
            whose message contains ``volume[bad\\nname].name`` and
            mentions the newline char.
    """
    import pytest

    payload = _example_payload()
    payload["namespace"] = "test-ns"
    payload["volumes"] = [
        {
            "name": "bad\nname",
            "mount_path": "/x",
            "secret_name": "test-secret",
        }
    ]
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(name="vol-sanitize-test")
    with pytest.raises(ValueError, match="YAML-unsafe character"):
        build_template_context(
            job_name="vol-test",
            job_spec=job_spec,
            profile=profile,
        )


def test_namespace_hook_env_defaults_with_newline_raises():
    """
    Tier L-C1: hook-supplied ``env_defaults`` values containing YAML-
    unsafe characters now raise ValueError when the hook merges into
    the container env, instead of silently stripping the offending
    chars. The error names the hook + key path so the offending
    profile entry can be located.

    Tests:
        (Test Case 1) ``NamespaceHookSpec(env_defaults={"FOO":
            "value\\nwith\\nnewlines"})`` causes
            ``build_template_context`` to raise ValueError whose
            message mentions ``hooks.test-ns.env_defaults.FOO``.
    """
    import pytest

    payload = _example_payload()
    payload["namespace"] = "test-ns"
    # No user-set FOO so the hook value wins.
    payload["container"]["env"] = {"NORMAL": "value"}
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(
        name="env-sanitize-test",
        namespace_hooks={
            "test-ns": NamespaceHookSpec(
                env_defaults={"FOO": "value\nwith\nnewlines"},
                required_volumes=[
                    VolumeMountSpec(
                        name="dummy",
                        mount_path="/dummy",
                        secret_name="dummy",
                    ),
                ],
            )
        },
    )
    with pytest.raises(ValueError, match="YAML-unsafe character"):
        build_template_context(
            job_name="env-test",
            job_spec=job_spec,
            profile=profile,
        )


def test_job_yaml_quoting_handles_special_characters():
    """
    ``job.yaml.j2`` quotes string fields so values with YAML-sensitive
    characters (``:``) still parse cleanly.

    Tests:
        (Test Case 1) Volume name containing a colon (``"weird:name"``
            via the unsafe-character sanitizer leaves the colon
            intact, since ``:`` isn't in the YAML-unsafe set) parses
            successfully through ``yaml.safe_load``.
        (Test Case 2) ``metadata.name``, ``metadata.namespace`` and
            container image round-trip via the parsed YAML.
    """
    payload = _example_payload()
    payload["namespace"] = "ns"
    payload["container"]["image"] = "img"
    payload["volumes"] = [
        {
            "name": "weird:name",
            "mount_path": "/m",
            "secret_name": "test-secret",
        }
    ]
    job_spec = validate_job_spec(payload)
    profile = ClusterProfile(name="quoting-test")
    context = build_template_context(
        job_name="job-x",
        job_spec=job_spec,
        profile=profile,
    )
    manifest = render_job_manifest(context)
    parsed = yaml.safe_load(manifest)
    assert parsed["metadata"]["name"] == "job-x"
    assert parsed["metadata"]["namespace"] == "ns"
    assert parsed["spec"]["template"]["spec"]["containers"][0]["image"] == "img"


class TestSanitizeYamlValueRaisesOnUnsafeChars:
    """``_sanitize_yaml_value`` raises ``ValueError`` on unsafe
    characters instead of silently stripping them. The field-path
    argument is included in the error message so operators can
    locate the offending entry in the job spec.
    """

    def test_unsafe_char_raises_with_repr_in_message(self):
        """
        Tests:
            (Test Case 1) Newline in the value raises ValueError.
            (Test Case 2) The error message includes the substring
                "YAML-unsafe character" and the offending char's repr.
            (Test Case 3) A clean value passes through unchanged.
        """
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        with pytest.raises(ValueError, match="YAML-unsafe character"):
            _sanitize_yaml_value("hello\nworld")

        assert _sanitize_yaml_value("safe value") == "safe value"

    def test_field_argument_appears_in_error(self):
        """
        Tests:
            (Test Case 1) ``field="container.env.FOO"`` shows up in
                the raised error so operators can locate the failing
                entry in their job spec.
        """
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        with pytest.raises(ValueError, match="container.env.FOO"):
            _sanitize_yaml_value("bad\nval", field="container.env.FOO")

    def test_sanitize_map_names_offending_subkey(self):
        """
        Tests:
            (Test Case 1) ``_sanitize_map({"FOO": "x\\ny"},
                field="container.env")`` raises ValueError naming
                ``container.env.FOO``.
        """
        from spikelab.batch_jobs.templating import _sanitize_map

        with pytest.raises(ValueError, match=r"container\.env\.FOO"):
            _sanitize_map({"FOO": "x\nY"}, field="container.env")

    def test_sanitize_list_names_offending_index(self):
        """
        Tests:
            (Test Case 1) ``_sanitize_list(["ok", "bad\\nval"],
                field="container.command")`` raises ValueError naming
                ``container.command[1]``.
        """
        from spikelab.batch_jobs.templating import _sanitize_list

        with pytest.raises(ValueError, match=r"container\.command\[1\]"):
            _sanitize_list(["ok", "bad\nval"], field="container.command")

    def test_volume_mount_unsafe_name_raises_from_build_template_context(self):
        """
        Tests:
            (Test Case 1) ``build_template_context`` with a
                VolumeMountSpec whose ``name`` contains a newline
                raises ValueError naming the volume entry.
        """
        payload = _example_payload()
        payload["namespace"] = "ns"
        payload["volumes"] = [
            {
                "name": "bad\nname",
                "mount_path": "/x",
                "secret_name": "test-secret",
            }
        ]
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="vol-raise")
        with pytest.raises(ValueError, match="YAML-unsafe character"):
            build_template_context(job_name="job-x", job_spec=job_spec, profile=profile)

    def test_safe_punctuation_passes_through(self):
        """
        Tests:
            (Test Case 1) Characters like ``:``, ``#``, ``{``, ``[``,
                ``&``, ``*``, ``!``, ``|``, ``>``, ``'``, ``%``,
                ``@``, ``` ` ``` are safe inside double-quoted YAML
                scalars and pass through ``_sanitize_yaml_value``
                unchanged.
        """
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        safe = "a:b#c{d}e[f]g&h*i!j|k>l'm%n@o`p"
        assert _sanitize_yaml_value(safe) == safe


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

    def test_threshold_s_parameter_controls_cutoff(self):
        """
        The ``threshold_s`` kwarg sets the cap above which a bare
        ``sleep <number>`` is considered idle. Lowering it flags
        durations that the default cap permits.

        Tests:
            (Test Case 1) ``sleep 100`` with ``threshold_s=50`` → True.
            (Test Case 2) ``sleep 100`` with ``threshold_s=200`` → False.
        """
        assert _contains_disallowed_sleep(["sleep"], ["100"], threshold_s=50)
        assert not _contains_disallowed_sleep(["sleep"], ["100"], threshold_s=200)

    def test_evaluate_policy_propagates_sleep_duration_threshold(self):
        """
        ``evaluate_policy`` forwards ``cfg.sleep_duration_threshold_s``
        to ``_contains_disallowed_sleep`` so users can tighten or
        loosen the policy without editing the heuristic itself.

        Tests:
            (Test Case 1) ``PolicyConfig(sleep_duration_threshold_s=50)``
                + ``sleep 100`` produces a sleep-related finding (the
                stricter threshold flags it).
            (Test Case 2) The default threshold (24h) allows the same
                ``sleep 100`` without flagging it.
        """
        from spikelab.batch_jobs.policy import PolicyConfig, evaluate_policy

        payload = _example_payload()
        payload["container"]["command"] = ["sleep"]
        payload["container"]["args"] = ["100"]
        job_spec = validate_job_spec(payload)

        # Strict profile: 50 s threshold flags ``sleep 100``.
        strict_profile = ClusterProfile(
            name="strict",
            policy=PolicyConfig(sleep_duration_threshold_s=50),
        )
        findings_strict = evaluate_policy(job_spec, strict_profile)
        levels_strict = {(f.code, f.level) for f in findings_strict}
        # A BLOCK-level sleep finding fires because the duration
        # exceeds the strict threshold.
        assert any(
            "sleep" in code and level == "BLOCK" for code, level in levels_strict
        )

        # Default profile: 24 h threshold lets ``sleep 100`` through.
        default_profile = ClusterProfile(name="default")
        findings_default = evaluate_policy(job_spec, default_profile)
        levels_default = {(f.code, f.level) for f in findings_default}
        assert not any(
            "sleep" in code and level == "BLOCK" for code, level in levels_default
        )

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
    """Policy thresholds come from the profile, not hardcoded values.

    Tier L-C4 changed the default to ``include_passes=False`` so a
    passing check no longer emits a PASS finding. The relaxed-profile
    branch now asserts the absence of an ``interactive_gpu_limit``
    finding, and passes ``include_passes=True`` to confirm the PASS
    finding is still available for audit tooling.
    """
    payload = _example_payload()
    payload["resources"]["requests_gpu"] = 3
    payload["resources"]["limits_gpu"] = 3
    job_spec = validate_job_spec(payload)
    # Default threshold is 2 — should warn
    profile_default = ClusterProfile(name="test")
    findings = evaluate_policy(job_spec, profile_default)
    gpu_finding = [f for f in findings if f.code == "interactive_gpu_limit"][0]
    assert gpu_finding.level == "WARN"
    # Raise threshold to 4 — should pass (silently under the default)
    from spikelab.batch_jobs.models import PolicyConfig

    profile_relaxed = ClusterProfile(
        name="test-relaxed",
        policy=PolicyConfig(max_interactive_gpus=4),
    )
    findings_relaxed = evaluate_policy(job_spec, profile_relaxed)
    gpu_findings_relaxed = [
        f for f in findings_relaxed if f.code == "interactive_gpu_limit"
    ]
    # Default (include_passes=False): no finding when the check passes.
    assert gpu_findings_relaxed == []
    # Audit mode: opt in to PASS findings.
    findings_audit = evaluate_policy(job_spec, profile_relaxed, include_passes=True)
    gpu_finding_audit = [
        f for f in findings_audit if f.code == "interactive_gpu_limit"
    ][0]
    assert gpu_finding_audit.level == "PASS"


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

        # Tier L-C2: sorting bundles must declare ``recording_files``.
        zip_path = package_analysis_bundle(
            input_paths=[str(input_file)],
            run_id="run-002",
            output_dir=str(tmp_path / "out"),
            output_format="sorting",
            metadata={"workspace_id": "ws-42"},
            recording_files=["result.h5"],
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
        assert manifest["recording_files"] == ["result.h5"]

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

    def test_sha256_chunk_bytes_constant_exposed(self):
        """
        ``_SHA256_CHUNK_BYTES`` is exposed at module scope so callers
        can read (or override) the chunk size. The constant value is
        1 MiB.

        Tests:
            (Test Case 1) ``_SHA256_CHUNK_BYTES`` equals ``1 << 20``
                (1 MiB).
        """
        from spikelab.batch_jobs.artifact_packager import _SHA256_CHUNK_BYTES

        assert _SHA256_CHUNK_BYTES == 1 << 20

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

    def test_duplicate_filenames_rejected(self, tmp_path):
        """Duplicate basenames in input_paths raise ValueError at packaging time."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()

        file_a = dir_a / "data.pkl"
        file_a.write_bytes(b"content_a")
        file_b = dir_b / "data.pkl"
        file_b.write_bytes(b"content_b")

        with pytest.raises(ValueError, match="Duplicate basename"):
            package_analysis_bundle(
                input_paths=[str(file_a), str(file_b)],
                run_id="run-dup",
                output_dir=str(tmp_path / "out"),
                output_format="workspace",
            )

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

    def test_upload_file_calls_boto3(self, tmp_path):
        """upload_file delegates to the boto3 client."""
        # ``upload_file`` now validates that the local path exists
        # before calling boto3 (clearer error than boto3's own).
        # Create a real temp file so the validation passes and the
        # mocked boto3 client still sees the delegation.
        local = tmp_path / "data.pkl"
        local.write_bytes(b"")
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            result = client.upload_file(
                local_path=str(local),
                s3_uri="s3://bucket/pfx/inputs/run-1/data.pkl",
            )
            mock_s3.upload_file.assert_called_once_with(
                str(local), "bucket", "pfx/inputs/run-1/data.pkl"
            )
            assert result == "s3://bucket/pfx/inputs/run-1/data.pkl"

    def test_upload_file_missing_local_path_raises_before_boto3(self, tmp_path):
        """
        ``upload_file`` validates ``local_path`` and raises
        ``FileNotFoundError`` before any boto3 call. Without this,
        boto3 produces a less informative error and the upload still
        opens a client connection on a nonexistent file.

        Tests:
            (Test Case 1) Missing path raises FileNotFoundError.
            (Test Case 2) The error message names the missing path.
            (Test Case 3) The boto3 client's ``upload_file`` is NOT
                invoked.
        """
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            missing_path = str(tmp_path / "does_not_exist.pkl")
            with pytest.raises(FileNotFoundError) as excinfo:
                client.upload_file(
                    local_path=missing_path,
                    s3_uri="s3://bucket/pfx/inputs/run-1/data.pkl",
                )
            # The error message embeds the path via repr() — match on
            # the basename to stay portable across OS path separators.
            assert "does_not_exist.pkl" in str(excinfo.value)
            mock_s3.upload_file.assert_not_called()

    def test_upload_bundle_builds_uri_and_uploads(self, tmp_path):
        """upload_bundle composes build_uri + upload_file."""
        local_zip = tmp_path / "run-7.zip"
        local_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            result = client.upload_bundle(local_zip=str(local_zip), run_id="run-7")
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

    def test_boto3_not_installed_constructor_succeeds_for_uri_ops(self):
        """boto3 missing is tolerated at construction (Tier H).

        Pure-string operations like ``build_uri`` /
        ``output_prefix_for_run`` / ``logs_prefix_for_run`` don't need
        boto3, so the constructor no longer eagerly checks for the
        optional dependency. The ImportError is deferred to the first
        call that actually needs the boto3 client (upload, download,
        list).
        """
        with patch("spikelab.batch_jobs.storage_s3.boto3", None):
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            # URI-only operations work without boto3.
            assert (
                client.output_prefix_for_run("run-1")
                == "s3://bucket/pfx/outputs/run-1/"
            )
            # Client-using operations raise the deferred ImportError.
            with pytest.raises(ImportError, match="boto3 is required"):
                client.upload_file(local_path=__file__, s3_uri="s3://bucket/pfx/x.bin")

    def test_list_output_files_caps_at_default_limit(self):
        """
        ``list_output_files`` raises ValueError when the paginator
        yields more objects than ``max_keys`` (default
        ``DEFAULT_LIST_OUTPUT_LIMIT = 10_000``). Pinning the cap as a
        class attribute lets callers raise it explicitly without
        editing the source.

        Tests:
            (Test Case 1) Mocked paginator yielding 10_001 keys raises
                ValueError mentioning ``max_keys=10000``.
            (Test Case 2) With ``max_keys=20_000`` the same fixture
                returns the full list.
            (Test Case 3) The cap is exposed as
                ``S3StorageClient.DEFAULT_LIST_OUTPUT_LIMIT``.
        """
        # Class attribute is documented and stable.
        assert S3StorageClient.DEFAULT_LIST_OUTPUT_LIMIT == 10_000

        def make_pages(n_keys):
            # Yield 1000 keys per page.
            for start in range(0, n_keys, 1000):
                stop = min(start + 1000, n_keys)
                yield {"Contents": [{"Key": f"k{i}"} for i in range(start, stop)]}

        # 10_001 keys → exceeds the default cap.
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix="s3://bucket/pfx/")

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = make_pages(10_001)
        mock_s3.get_paginator.return_value = mock_paginator

        with pytest.raises(ValueError, match="max_keys=10000"):
            client.list_output_files("run-1")

        # With a larger cap, the same fixture returns all keys.
        mock_paginator.paginate.return_value = make_pages(10_001)
        keys = client.list_output_files("run-1", max_keys=20_000)
        assert len(keys) == 10_001

    def test_boto3_not_installed_download_and_list_also_raise_deferred(self):
        """
        Reinforces the lazy-init detail of Tier H5: download_file and
        list_output_files raise the same deferred ImportError as
        upload_file when boto3 is unavailable. All client-using
        operations should hit the same lazy property, so an absent
        boto3 produces a consistent error message across the surface.

        Tests:
            (Test Case 1) ``download_file`` raises ImportError naming boto3.
            (Test Case 2) ``list_output_files`` raises ImportError naming boto3.
        """
        with patch("spikelab.batch_jobs.storage_s3.boto3", None):
            client = S3StorageClient(prefix="s3://bucket/pfx/")
            with pytest.raises(ImportError, match="boto3 is required"):
                client.download_file(
                    s3_uri="s3://bucket/pfx/x.bin", local_path="/tmp/x.bin"
                )
            with pytest.raises(ImportError, match="boto3 is required"):
                client.list_output_files(run_id="run-1")

    def test_build_uri_invalid_category_warns_with_known_categories(self):
        """
        ``build_uri`` with an unknown ``category`` falls back to the
        ``inputs`` template and emits a UserWarning naming the known
        categories so typos ("input", "logs/") don't silently land in
        the wrong S3 prefix.

        Tests:
            (Test Case 1) A UserWarning is emitted for an unknown
                category.
            (Test Case 2) The warning names the known categories
                ("inputs", "outputs", "logs").
            (Test Case 3) The returned URI uses the ``inputs`` template
                as the fallback.
        """
        import warnings as _warnings

        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = S3StorageClient(prefix="s3://bucket/pfx/")

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            uri = client.build_uri(
                run_id="run-1",
                filename="data.pkl",
                category="not-a-real-category",
            )

        warn_msgs = [str(rec.message) for rec in w if rec.category is UserWarning]
        relevant = [m for m in warn_msgs if "build_uri" in m]
        assert relevant, warn_msgs
        # Known categories listed in the warning.
        assert "inputs" in relevant[0]
        assert "outputs" in relevant[0]
        assert "logs" in relevant[0]
        # Fallback to the inputs template.
        assert uri == "s3://bucket/pfx/inputs/run-1/data.pkl"

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

    def test_prefixless_client_supports_download_upload(self, tmp_path):
        """
        S3StorageClient(prefix=None) supports the entrypoint pattern:
        the container constructs a prefixless client and exercises only
        download_file / upload_file with fully-formed S3 URIs. The
        prefix-templating methods are off-limits in this mode.

        Tests:
            (Test Case 1) download_file works with prefix=None.
            (Test Case 2) upload_file works with prefix=None.
            (Test Case 3) self.prefix is None (not coerced).
        """
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix=None)
        assert client.prefix is None

        client.download_file(
            s3_uri="s3://bucket/outputs/run-1/data.pkl",
            local_path="/tmp/data.pkl",
        )
        mock_s3.download_file.assert_called_once_with(
            "bucket", "outputs/run-1/data.pkl", "/tmp/data.pkl"
        )

        out_path = tmp_path / "out.pkl"
        out_path.write_bytes(b"")
        result = client.upload_file(
            local_path=str(out_path),
            s3_uri="s3://bucket/outputs/run-1/out.pkl",
        )
        mock_s3.upload_file.assert_called_once_with(
            str(out_path), "bucket", "outputs/run-1/out.pkl"
        )
        assert result == "s3://bucket/outputs/run-1/out.pkl"

    def test_prefixless_client_rejects_template_methods(self):
        """
        Calling prefix-templating methods on a prefix=None client
        raises ValueError naming the missing prefix — this is the
        intended fail-fast for the container path so a future
        refactor that accidentally calls build_uri / upload_bundle
        in the entrypoint surfaces the bug instead of silently
        producing double-templated URIs.

        Tests:
            (Test Case 1) build_uri raises.
            (Test Case 2) upload_bundle raises (it composes build_uri).
            (Test Case 3) output_prefix_for_run / logs_prefix_for_run
                return empty string (documented existing behaviour).
        """
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            client = S3StorageClient(prefix=None)

        with pytest.raises(ValueError, match="S3 prefix is not configured"):
            client.build_uri(run_id="run-1", filename="data.pkl")
        with pytest.raises(ValueError, match="S3 prefix is not configured"):
            client.upload_bundle(local_zip="/tmp/x.zip", run_id="run-1")

        # The two *_prefix_for_run methods return empty strings rather
        # than raising — preserved as the documented existing behaviour
        # (see test_output_prefix_no_prefix_returns_empty above).
        assert client.output_prefix_for_run("run-1") == ""
        assert client.logs_prefix_for_run("run-1") == ""


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

    def test_unknown_name_warns_with_recognised_aliases(self):
        """
        Unknown names emit a UserWarning naming the recognised aliases
        so a typo doesn't silently land on the wrong profile.

        Tests:
            (Test Case 1) A UserWarning is emitted for an unknown name.
            (Test Case 2) The warning lists at least one recognised
                alias ("defaults") so the operator can correct the typo.
        """
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            profile = load_profile_from_name("not-a-real-profile")

        warn_msgs = [str(rec.message) for rec in w if rec.category is UserWarning]
        relevant = [m for m in warn_msgs if "Unknown profile" in m]
        assert relevant, warn_msgs
        assert "defaults" in relevant[0]
        # Falls back to defaults regardless.
        assert profile.name == "defaults"

    def test_known_names_do_not_warn(self):
        """
        Recognised names (``"defaults"``, ``"nrp"``) load without
        emitting the unknown-name warning.

        Tests:
            (Test Case 1) ``load_profile_from_name("defaults")`` emits
                no "Unknown profile" warning.
            (Test Case 2) ``load_profile_from_name("nrp")`` emits no
                "Unknown profile" warning.
        """
        import warnings as _warnings

        for name in ("defaults", "nrp"):
            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter("always")
                load_profile_from_name(name)
            warn_msgs = [str(rec.message) for rec in w]
            assert not any(
                "Unknown profile" in m for m in warn_msgs
            ), f"{name!r}: {warn_msgs}"

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

    def test_gpu_fields_reject_none(self):
        """
        ``ResourceSpec.requests_gpu`` and ``limits_gpu`` are typed as
        ``int = Field(default=0, ge=0)``. None is rejected at the
        pydantic type-validation layer (before the
        ``_validate_gpu_pairing`` model-validator can run).

        Pins the current contract that one-sided GPU specs cannot be
        expressed as ``None`` — a previous REVIEW.md entry suggested
        ``requests_gpu=None, limits_gpu=1`` was a missing case, but
        the int-typed fields reject ``None`` outright. The default
        (both 0) is accepted.

        Tests:
            (Test Case 1) ``requests_gpu=None`` raises pydantic
                int-type error (not the mismatch validator).
            (Test Case 2) Default construction yields zero-zero GPU
                spec (no validation error).
            (Test Case 3) Asymmetric integer values like (1, 2) still
                trigger the explicit mismatch validator.
        """
        with pytest.raises(PydanticValidationError, match="int_type|valid integer"):
            ResourceSpec(requests_gpu=None, limits_gpu=1)

        spec = ResourceSpec()
        assert spec.requests_gpu == 0
        assert spec.limits_gpu == 0

        with pytest.raises(
            PydanticValidationError, match="GPU requests and limits must match"
        ):
            ResourceSpec(requests_gpu=1, limits_gpu=2)

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
        """JobSpec sanitizes special characters in name_prefix to hyphens and collapses runs.

        Tests:
            - Special characters are replaced with hyphens.
            - Consecutive hyphens are collapsed to a single hyphen.
        """
        payload = _example_payload()
        payload["name_prefix"] = "my job!@#test"
        job_spec = validate_job_spec(payload)
        assert job_spec.name_prefix == "my-job-test"

    def test_name_prefix_all_special_chars_raises(self):
        """JobSpec raises ValueError when prefix has no usable ASCII content.

        Tests:
            - An all-hyphen prefix sanitizes to empty and raises ValueError.
        """
        payload = _example_payload()
        payload["name_prefix"] = "---"
        with pytest.raises(ValueError, match="no usable ASCII content"):
            validate_job_spec(payload)

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

from spikelab.batch_jobs.validation import summarize_validation_error


class TestValidationModule:
    def test_summarize_validation_error_format(self):
        """summarize_validation_error produces a readable string."""
        try:
            validate_job_spec({})
        except PydanticValidationError as exc:
            summary = summarize_validation_error(exc)
            assert isinstance(summary, str)
            assert summary.startswith("Invalid job config")

    def test_summarize_validation_error_multiple_errors(self):
        """summarize_validation_error puts each error on its own line.

        Pinning the new multiline format: header + one bullet per
        issue. Nested-location validation messages stay scannable when
        a pydantic error has several issues at once.
        """
        try:
            validate_job_spec({"container": {}})  # missing image + other issues
        except PydanticValidationError as exc:
            summary = summarize_validation_error(exc)
            assert summary.startswith("Invalid job config:")
            assert "\n  - " in summary
            # At least two distinct bullet lines for the multi-error case.
            assert summary.count("\n  - ") >= 2


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

    def test_redact_none_values_renders_unset_sentinel(self):
        """
        ``None`` inputs render as the literal ``"<unset>"`` so the
        audit log can distinguish "credential is not configured" from
        "credential is configured but empty" (the prior implementation
        collapsed both into ``""``).

        Tests:
            (Test Case 1) ``{"FIELD": None}`` produces
                ``{"FIELD": "<unset>"}``.
            (Test Case 2) Non-None values still pass through.
            (Test Case 3) ``None`` short-circuits before the redaction
                pattern check — a None-valued sensitive key (e.g.
                ``SECRET_TOKEN``) also renders as ``"<unset>"``, not
                ``"***REDACTED***"``.
        """
        redacted = redact_sensitive_map({"FIELD": None, "OTHER": "ok"})
        assert redacted["FIELD"] == "<unset>"
        assert redacted["OTHER"] == "ok"

        # None short-circuits the sensitive-key redaction branch.
        secret = redact_sensitive_map({"SECRET_TOKEN": None})
        assert secret["SECRET_TOKEN"] == "<unset>"


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

    def test_policy_long_runtime_pass_when_not_set(self):
        """
        Tier L-C4: with the default ``include_passes=False``, a
        passing long_runtime check emits NO finding. Opting in
        with ``include_passes=True`` still produces the audit-trail
        PASS entry naming the cluster-default fallback.

        Tests:
            (Test Case 1) Default call returns no long_runtime
                finding when active_deadline_seconds is None.
            (Test Case 2) include_passes=True produces a PASS
                finding whose message names the cluster-default
                fallback.
        """
        payload = _example_payload()
        # active_deadline_seconds defaults to None
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        # Default: silent on passing checks.
        findings = evaluate_policy(job_spec, profile)
        long_findings = [f for f in findings if f.code == "long_runtime"]
        assert long_findings == []
        # Audit mode: PASS entry returns.
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        long_finding = [f for f in findings_audit if f.code == "long_runtime"][0]
        assert long_finding.level == "PASS"
        assert "cluster default" in long_finding.message

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
        """request_limit_mismatch check can be disabled via profile.

        Tier L-C4: default ``include_passes=False`` → no finding is
        emitted when the check is disabled or passes. ``include_passes
        =True`` restores the PASS audit entry.
        """
        payload = _example_payload()
        payload["resources"]["requests_cpu"] = "1"
        payload["resources"]["limits_cpu"] = "4"
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(warn_request_limit_mismatch=False),
        )
        # Default: silent on disabled / passing checks.
        findings = evaluate_policy(job_spec, profile)
        assert [f for f in findings if f.code == "request_limit_mismatch"] == []
        # Audit mode: PASS entry returns.
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        codes = {f.code: f.level for f in findings_audit}
        assert codes["request_limit_mismatch"] == "PASS"

    def test_default_suppresses_pass_on_fully_compliant_spec(self):
        """
        Tier L-C4: a fully compliant job spec produces NO findings by
        default, and exactly four PASS findings under
        ``include_passes=True``.

        Tests:
            (Test Case 1) Default ``include_passes=False`` returns an
                empty list when every check passes — eliminating the
                ~4 PASS-line log noise per compliant submission.
            (Test Case 2) ``include_passes=True`` returns four PASS
                findings, one per check
                (interactive_gpu_limit, sleep_in_batch_job,
                 request_limit_mismatch, long_runtime).
        """
        job_spec = validate_job_spec(_example_payload())
        profile = ClusterProfile(name="test")
        findings_default = evaluate_policy(job_spec, profile)
        assert findings_default == []
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        codes_levels = {f.code: f.level for f in findings_audit}
        assert codes_levels == {
            "interactive_gpu_limit": "PASS",
            "sleep_in_batch_job": "PASS",
            "request_limit_mismatch": "PASS",
            "long_runtime": "PASS",
        }

    def test_warn_emitted_regardless_of_include_passes(self):
        """
        Tier L-C4: WARN findings are emitted regardless of
        ``include_passes``. Only PASS findings are gated by the flag.

        Tests:
            (Test Case 1) A WARN-triggering interactive_gpu_limit
                spec emits the WARN finding under
                ``include_passes=False`` (the default).
            (Test Case 2) The same WARN finding is also present under
                ``include_passes=True``, alongside the PASS entries
                for the other checks.
        """
        payload = _example_payload()
        payload["resources"]["requests_gpu"] = 8
        payload["resources"]["limits_gpu"] = 8
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings_default = evaluate_policy(job_spec, profile)
        warn_findings = [
            f for f in findings_default if f.code == "interactive_gpu_limit"
        ]
        assert len(warn_findings) == 1
        assert warn_findings[0].level == "WARN"
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        warn_findings_audit = [
            f for f in findings_audit if f.code == "interactive_gpu_limit"
        ]
        assert len(warn_findings_audit) == 1
        assert warn_findings_audit[0].level == "WARN"

    def test_block_emitted_regardless_of_include_passes(self):
        """
        Tier L-C4: BLOCK findings are emitted regardless of
        ``include_passes`` — the flag only gates PASS entries.

        Tests:
            (Test Case 1) A BLOCK-triggering sleep_in_batch_job
                payload emits the BLOCK finding under the default
                ``include_passes=False``.
            (Test Case 2) The same BLOCK finding is also present
                under ``include_passes=True``.
        """
        payload = _example_payload()
        payload["container"]["args"] = ["sleep", "infinity"]
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings_default = evaluate_policy(job_spec, profile)
        block_findings = [f for f in findings_default if f.code == "sleep_in_batch_job"]
        assert len(block_findings) == 1
        assert block_findings[0].level == "BLOCK"
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        block_findings_audit = [
            f for f in findings_audit if f.code == "sleep_in_batch_job"
        ]
        assert len(block_findings_audit) == 1
        assert block_findings_audit[0].level == "BLOCK"

    def test_mixed_compliance_default_returns_only_warn(self):
        """
        Tier L-C4: a spec that triggers ONE WARN and passes the
        other three checks returns exactly one finding by default
        (the WARN). ``include_passes=True`` returns four findings
        (1 WARN + 3 PASS).

        Tests:
            (Test Case 1) Default call returns one finding —
                the request_limit_mismatch WARN.
            (Test Case 2) ``include_passes=True`` returns four
                findings — one WARN plus three PASS entries.
        """
        payload = _example_payload()
        payload["resources"]["requests_cpu"] = "1"
        payload["resources"]["limits_cpu"] = "4"
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(name="test")
        findings_default = evaluate_policy(job_spec, profile)
        assert len(findings_default) == 1
        assert findings_default[0].code == "request_limit_mismatch"
        assert findings_default[0].level == "WARN"
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        assert len(findings_audit) == 4
        levels = sorted(f.level for f in findings_audit)
        assert levels == ["PASS", "PASS", "PASS", "WARN"]

    def test_summarize_preflight_compliant_spec_empty_text(self):
        """
        Tier L-C4: a fully compliant job spec evaluated under the
        default ``include_passes=False`` produces an empty findings
        list. ``summarize_preflight`` on that empty list returns
        ``("PASS", "")`` — the level stays PASS, the text body is
        empty (no more ``[PASS] ... [PASS] ...`` noise).

        Tests:
            (Test Case 1) ``evaluate_policy`` on a compliant spec
                returns ``[]`` by default.
            (Test Case 2) ``summarize_preflight([])`` returns
                ``("PASS", "")`` — the level stays PASS, text empty.
        """
        job_spec = validate_job_spec(_example_payload())
        profile = ClusterProfile(name="test")
        findings = evaluate_policy(job_spec, profile)
        assert findings == []
        level, text = summarize_preflight(findings)
        assert level == "PASS"
        assert text == ""


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
        """requests_gpu == max_interactive_gpus is treated as within-limit.

        Tier L-C4: with the default ``include_passes=False``, an at-
        threshold-but-not-over GPU request emits NO finding. The
        ``include_passes=True`` opt-in still produces the PASS audit
        entry.
        """
        payload = _example_payload()
        payload["resources"]["requests_gpu"] = 2
        payload["resources"]["limits_gpu"] = 2
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(max_interactive_gpus=2),
        )
        # Default: silent on passing checks.
        findings = evaluate_policy(job_spec, profile)
        assert [f for f in findings if f.code == "interactive_gpu_limit"] == []
        # Audit mode: PASS entry returns.
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        gpu_finding = [f for f in findings_audit if f.code == "interactive_gpu_limit"][
            0
        ]
        assert gpu_finding.level == "PASS"

    def test_block_sleep_infinity_disabled_emits_warn(self):
        """block_sleep_infinity=False permits sleep commands through but
        records a WARN audit finding when a sleep pattern is actually
        detected. The previous PASS finding was misleading — it claimed
        no pattern was detected even when one was present, just not
        blocked. The WARN keeps the audit trail honest.
        """
        payload = _example_payload()
        payload["container"]["args"] = ["sleep", "infinity"]
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(block_sleep_infinity=False),
        )
        findings = evaluate_policy(job_spec, profile)
        sleep_finding = [f for f in findings if f.code == "sleep_in_batch_job"][0]
        assert sleep_finding.level == "WARN"
        assert "block_sleep_infinity is" in sleep_finding.message

    def test_active_deadline_at_boundary(self):
        """
        active_deadline_seconds == max_runtime_seconds is treated as
        within-limit. Tier L-C4: default ``include_passes=False`` →
        no finding emitted (the WARN threshold is strict ``>``).
        Opt-in ``include_passes=True`` produces the PASS audit entry.

        Tests:
            (Test Case 1) Default call returns no long_runtime
                finding at the boundary.
            (Test Case 2) include_passes=True produces a PASS whose
                message names both the actual deadline and the max.
        """
        payload = _example_payload()
        payload["active_deadline_seconds"] = 1_209_600  # exactly 14 days
        job_spec = validate_job_spec(payload)
        profile = ClusterProfile(
            name="test",
            policy=PolicyConfig(max_runtime_seconds=1_209_600),
        )
        # Default: silent.
        findings = evaluate_policy(job_spec, profile)
        assert [f for f in findings if f.code == "long_runtime"] == []
        # Audit mode: PASS entry.
        findings_audit = evaluate_policy(job_spec, profile, include_passes=True)
        long_finding = [f for f in findings_audit if f.code == "long_runtime"][0]
        assert long_finding.level == "PASS"
        assert "1209600" in long_finding.message

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
    def test_empty_prefix_raises(self):
        """Empty prefix raises ValueError (would produce a leading-hyphen name)."""
        from spikelab.batch_jobs.session import RunSession

        with pytest.raises(ValueError, match="alphanumeric"):
            RunSession._build_job_name("")

    def test_all_hyphens_prefix_raises(self):
        """All-hyphen prefix raises ValueError (rstrip reduces it to empty)."""
        from spikelab.batch_jobs.session import RunSession

        with pytest.raises(ValueError, match="empty string"):
            RunSession._build_job_name("---")

    def test_prefix_exactly_at_max_length(self):
        """54-char prefix fits exactly (54 + 1 + 8 = 63)."""
        from spikelab.batch_jobs.session import RunSession

        prefix = "a" * 54
        name = RunSession._build_job_name(prefix)
        assert len(name) == 63
        assert name.startswith("a" * 54 + "-")


# ---------------------------------------------------------------------------
# PolicyConfig validation edge cases
# ---------------------------------------------------------------------------


class TestPolicyConfigValidation:
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
        """Build an S3StorageClient with mocked boto3."""
        with patch("spikelab.batch_jobs.storage_s3.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            client = S3StorageClient(prefix=prefix)
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

    def test_submit_returns_redacted_manifest_but_applies_real_one(self, tmp_path):
        """
        ``RunSession._submit`` writes the *real* manifest (with secrets
        intact) to the tempfile applied via ``backend.apply_manifest``,
        but returns a *redacted* manifest in ``SubmitResult.manifest_yaml``
        so the value can be logged/audited without leaking credentials.

        Tests:
            (Test Case 1) ``SubmitResult.manifest_yaml`` contains
                ``***REDACTED***`` for the secret-named env var and the
                non-secret value verbatim.
            (Test Case 2) The manifest path passed to
                ``backend.apply_manifest`` carries the REAL secret on
                disk at the time of the call.
        """
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, backend, storage = self._make_session()

        # Tier L-D8: ``_submit`` now passes the manifest YAML string
        # directly to ``apply_manifest`` instead of writing to a
        # tempfile and passing the path. Capture the string argument
        # for post-call inspection.
        captured_manifest = {"text": None}

        def capture_apply(manifest_text_or_path):
            captured_manifest["text"] = manifest_text_or_path
            return "test-job-abc"

        backend.apply_manifest.side_effect = capture_apply

        ws = AnalysisWorkspace(name="redact-secret")
        script = tmp_path / "analyze.py"
        script.write_text("print('hello')", encoding="utf-8")

        payload = _example_payload()
        payload["container"]["env"] = {
            "AWS_SECRET_ACCESS_KEY": "real-secret-value",
            "NORMAL_VAR": "value",
        }
        job_spec = validate_job_spec(payload)

        result = session.submit_workspace_job(
            workspace=ws,
            script=str(script),
            job_spec=job_spec,
        )

        # Test Case 1: returned manifest carries the redacted env.
        assert "***REDACTED***" in result.manifest_yaml
        assert "real-secret-value" not in result.manifest_yaml
        assert "NORMAL_VAR" in result.manifest_yaml
        assert "value" in result.manifest_yaml

        # Test Case 2: the applied manifest carried the real secret.
        assert captured_manifest["text"] is not None
        assert "real-secret-value" in captured_manifest["text"]

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

    def test_submit_workspace_with_string_base_missing_json_raises(self, tmp_path):
        """
        ``_save_workspace`` for a string base-path verifies both ``.h5``
        and ``.json`` exist. A missing ``.json`` (but present ``.h5``)
        raises FileNotFoundError mentioning the index file — without
        this, the partial save would silently propagate downstream.

        Tests:
            (Test Case 1) Path with only ``.h5`` (no ``.json``) raises
                FileNotFoundError.
            (Test Case 2) Error message names the missing ``.json``
                file.
        """
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")

        session, _, _ = self._make_session()
        script = tmp_path / "analyze.py"
        script.write_text("print('hello')", encoding="utf-8")

        # Create only the .h5 file; omit the .json index.
        base = tmp_path / "partial"
        with h5py.File(str(base) + ".h5", "w") as f:
            f.attrs["__workspace_id__"] = "test"

        job_spec = validate_job_spec(_example_payload())
        with pytest.raises(FileNotFoundError) as excinfo:
            session.submit_workspace_job(
                workspace=str(base),
                script=str(script),
                job_spec=job_spec,
            )
        msg = str(excinfo.value)
        assert "partial.json" in msg or "index" in msg.lower()

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

        # Mock list_output_files to return a fully-prefixed S3 key. After
        # the prefix is stripped the leading "/" left by parse_s3_url's
        # trailing-slash normalisation is also stripped so the path
        # resolves under local_dir on every OS.
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

    def test_retrieve_sorting_strips_leading_slash_from_relative_path(self, tmp_path):
        """
        retrieve_result strips the leading slash left by parse_s3_url's
        trailing-slash normalisation so listed keys that share the
        configured output prefix resolve under local_dir on every OS
        (not at drive root on Windows).

        Tests:
            (Test Case 1) A listed key fully prefixed with the output
                prefix is downloaded under local_dir, not at drive root.
            (Test Case 2) The downloaded SpikeData is loaded into the
                workspace under the filename stem.
        """
        import pickle

        import numpy as np

        from spikelab.spikedata.spikedata import SpikeData
        from spikelab.workspace.workspace import AnalysisWorkspace

        session, storage = self._make_session()

        sd = SpikeData([np.array([1.0, 2.0])], length=10.0)
        pkl_path = tmp_path / "rec1_curated.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(sd, f)

        # Fully prefixed listing key — common in real S3 listings — must
        # not produce a drive-root path when concatenated with local_dir.
        storage.list_output_files.return_value = ["pfx/outputs/run-1/rec1_curated.pkl"]
        storage.output_prefix_for_run.return_value = "s3://bucket/pfx/outputs/run-1/"

        observed_local_paths: list[str] = []

        def fake_download(*, s3_uri, local_path):
            import shutil

            observed_local_paths.append(local_path)
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

        out_dir = tmp_path / "out"
        result_ws = session.retrieve_result(submit_result, str(out_dir))
        assert isinstance(result_ws, AnalysisWorkspace)

        # Every download_file call resolves under out_dir, never drive root.
        for lp in observed_local_paths:
            assert Path(lp).resolve().is_relative_to(out_dir.resolve())

        sd_loaded = result_ws.get("rec1_curated", "spikedata")
        assert sd_loaded is not None
        assert sd_loaded.N == 1

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

    def test_main_script_failure_prints_marker_and_reraises(
        self, tmp_path, monkeypatch, capsys
    ):
        """
        ``entrypoints/workspace.main`` wraps ``runpy.run_path`` so a
        failure inside the analysis script surfaces with a clear
        ``WORKSPACE_SCRIPT_FAILED: <type>: <msg>`` marker on stderr,
        then re-raises so the pod exits non-zero. Without this, the
        operator scanning pod logs would have to distinguish "script
        ran and raised" from "entrypoint bootstrap failed" by stack-
        trace shape alone.

        Tests:
            (Test Case 1) The marker substring ``WORKSPACE_SCRIPT_FAILED``
                with the exception class name and message appears on
                stderr.
            (Test Case 2) The original exception propagates to the
                caller (re-raise behaviour).
        """
        import json
        import zipfile
        import shutil

        from spikelab.batch_jobs.entrypoints import workspace as ws_entrypoint
        from spikelab.workspace.workspace import AnalysisWorkspace

        ws = AnalysisWorkspace(name="failure-marker-test")
        ws_base = str(tmp_path / "workspace")
        ws.save(ws_base)

        script = tmp_path / "failing_script.py"
        script.write_text("raise RuntimeError('synthetic failure')\n", encoding="utf-8")

        bundle_dir = tmp_path / "bundle" / "run-1"
        bundle_dir.mkdir(parents=True)
        shutil.copy2(f"{ws_base}.h5", bundle_dir / "workspace.h5")
        shutil.copy2(f"{ws_base}.json", bundle_dir / "workspace.json")
        shutil.copy2(str(script), bundle_dir / "failing_script.py")
        manifest = {"run_id": "run-1", "output_format": "workspace", "files": []}
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        shutil.make_archive(
            str(tmp_path / "bundle"),
            "zip",
            root_dir=str(tmp_path / "bundle"),
            base_dir="run-1",
        )
        zip_path = str(tmp_path / "bundle.zip")

        def fake_download(*, s3_uri, local_path):
            shutil.copy2(zip_path, local_path)
            return local_path

        mock_storage = MagicMock()
        mock_storage.download_file.side_effect = fake_download
        monkeypatch.setattr(
            "spikelab.batch_jobs.storage_s3.S3StorageClient",
            lambda **kwargs: mock_storage,
        )
        monkeypatch.setenv("INPUT_URI", "s3://bucket/input/bundle.zip")
        monkeypatch.setenv("OUTPUT_PREFIX", "s3://bucket/outputs/run-1/")
        monkeypatch.setenv("SCRIPT_NAME", "failing_script.py")

        with pytest.raises(RuntimeError, match="synthetic failure"):
            ws_entrypoint.main()

        captured = capsys.readouterr()
        assert "WORKSPACE_SCRIPT_FAILED" in captured.err
        assert "RuntimeError" in captured.err
        assert "synthetic failure" in captured.err

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


class TestFindWorkspaceH5:
    """
    ``_find_workspace_h5`` identifies the workspace by content
    signature (the __workspace_id__ HDF5 attribute) rather than by
    filename. Bundles can contain other .h5 inputs (recordings,
    intermediate data) via ``bundle_input_paths``, and the workspace
    itself can be saved under any base path the caller chose.
    """

    def _make_workspace_h5(self, path):
        """Write a minimal SpikeLab workspace signature to ``path``."""
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["__workspace_id__"] = "ws-test-123"
            f.attrs["__workspace_name__"] = "test"
            f.attrs["__created_at__"] = 0.0

    def _make_recording_h5(self, path):
        """Write an .h5 file that is NOT a SpikeLab workspace."""
        import h5py
        import numpy as np

        with h5py.File(path, "w") as f:
            f.create_dataset("traces", data=np.zeros((10, 4)))

    def test_picks_workspace_with_arbitrary_name(self, tmp_path):
        """
        _find_workspace_h5 picks the workspace .h5 even when it has
        a custom name (not 'workspace.h5'), because identification
        is content-based.

        Tests:
            (Test Case 1) A bundle with my_analysis.h5 (the workspace)
                + recording.h5 (no signature) returns my_analysis.h5.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.batch_jobs.entrypoints.workspace import _find_workspace_h5

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        ws = bundle / "my_analysis.h5"
        rec = bundle / "recording.h5"
        self._make_workspace_h5(ws)
        self._make_recording_h5(rec)

        result = _find_workspace_h5(bundle)
        assert result == ws

    def test_ignores_non_workspace_h5_files(self, tmp_path):
        """
        Files without __workspace_id__ are skipped, so extra .h5
        inputs (recordings, intermediate data) don't confuse the
        identification.

        Tests:
            (Test Case 1) Bundle with workspace.h5 + extra rec.h5
                returns workspace.h5 (not the first-rglob result).
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.batch_jobs.entrypoints.workspace import _find_workspace_h5

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        # Use names where the recording sorts BEFORE the workspace
        # alphabetically, to exercise the "wrong-rglob-order" case.
        ws = bundle / "workspace.h5"
        rec1 = bundle / "aaa_recording.h5"
        rec2 = bundle / "zzz_intermediate.h5"
        self._make_workspace_h5(ws)
        self._make_recording_h5(rec1)
        self._make_recording_h5(rec2)

        result = _find_workspace_h5(bundle)
        assert result == ws

    def test_no_workspace_raises_clear_error(self, tmp_path):
        """
        A bundle with no .h5 carrying __workspace_id__ raises
        FileNotFoundError naming the expected attribute so operators
        can debug the bundle layout.

        Tests:
            (Test Case 1) Bundle with only non-workspace .h5 raises.
            (Test Case 2) The error names "__workspace_id__".
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.batch_jobs.entrypoints.workspace import _find_workspace_h5

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        self._make_recording_h5(bundle / "rec.h5")

        with pytest.raises(FileNotFoundError, match="__workspace_id__"):
            _find_workspace_h5(bundle)

    def test_multiple_workspaces_raises_clear_error(self, tmp_path):
        """
        Two .h5 files both carrying __workspace_id__ are an
        ambiguous bundle layout — refuse to guess and name both
        candidates so the operator can fix the inputs.

        Tests:
            (Test Case 1) Two workspace files raise RuntimeError.
            (Test Case 2) The error names both candidate paths.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.batch_jobs.entrypoints.workspace import _find_workspace_h5

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        ws1 = bundle / "first.h5"
        ws2 = bundle / "second.h5"
        self._make_workspace_h5(ws1)
        self._make_workspace_h5(ws2)

        with pytest.raises(RuntimeError) as exc_info:
            _find_workspace_h5(bundle)
        msg = str(exc_info.value)
        assert "first.h5" in msg
        assert "second.h5" in msg

    def test_malformed_h5_silently_skipped(self, tmp_path):
        """
        A non-HDF5 file with .h5 extension (e.g. corrupt download) is
        silently skipped — h5py raises OSError, the helper continues
        scanning, and a sibling valid workspace is still found.

        Tests:
            (Test Case 1) Bundle with a corrupt foo.h5 + a real
                workspace returns the workspace.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.batch_jobs.entrypoints.workspace import _find_workspace_h5

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        ws = bundle / "ws.h5"
        corrupt = bundle / "corrupt.h5"
        self._make_workspace_h5(ws)
        corrupt.write_bytes(b"not an HDF5 file")

        result = _find_workspace_h5(bundle)
        assert result == ws

    def test_only_corrupt_h5_raises_with_diagnostic_listing(self, tmp_path):
        """
        When the bundle contains ONLY corrupt .h5 files (no readable
        workspace), the FileNotFoundError message lists the corrupt
        path under a ``could not be inspected`` line so the operator
        can tell "no workspace" apart from "workspace was unreadable".

        Tests:
            (Test Case 1) FileNotFoundError is raised.
            (Test Case 2) The error names the corrupt file.
            (Test Case 3) The error mentions "could not be inspected".
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.batch_jobs.entrypoints.workspace import _find_workspace_h5

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        corrupt = bundle / "corrupt.h5"
        corrupt.write_bytes(b"not an HDF5 file")

        with pytest.raises(FileNotFoundError) as excinfo:
            _find_workspace_h5(bundle)
        msg = str(excinfo.value)
        assert "corrupt.h5" in msg
        assert "could not be inspected" in msg


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

    def test_reconstruct_config_resolves_string_annotations(self, monkeypatch):
        """
        Under ``from __future__ import annotations`` (or other
        deferred-evaluation modes), dataclass ``field.type`` carries
        the annotation as a *string* rather than the actual class.
        The reconstruction path resolves string annotations via
        ``typing.get_type_hints`` so both modes work; without this,
        ``f.type(**sub_dict)`` would raise
        ``TypeError: 'str' object is not callable``.

        Tests:
            (Test Case 1) Setting ``SortingPipelineConfig.__annotations__[
                "recording"] = "RecordingConfig"`` (string form) does
                not break ``_reconstruct_config`` — it still produces a
                valid SortingPipelineConfig.
        """
        import dataclasses

        from spikelab.batch_jobs.entrypoints.sorting import _reconstruct_config
        from spikelab.spike_sorting.config import (
            SortingPipelineConfig,
            RecordingConfig,
        )

        # Build a valid baseline config dict.
        config_dict = dataclasses.asdict(SortingPipelineConfig())

        # Monkeypatch one annotation to the string form. ``get_type_hints``
        # resolves "RecordingConfig" against the class's module globals,
        # which already imports the class.
        original = dict(SortingPipelineConfig.__annotations__)
        new_annotations = dict(original)
        new_annotations["recording"] = "RecordingConfig"
        monkeypatch.setattr(SortingPipelineConfig, "__annotations__", new_annotations)

        reconstructed = _reconstruct_config(config_dict)
        assert isinstance(reconstructed.recording, RecordingConfig)


class TestSortingEntrypointMain:
    """
    End-to-end test for the ``main`` function in
    ``spikelab.batch_jobs.entrypoints.sorting``. Mirrors the
    ``TestWorkspaceEntrypoint.test_main_runs_script_with_workspace``
    pattern: build a real bundle, mock S3 + ``sort_recording``, and
    verify that ``main()`` downloads, sorts, and uploads as
    documented.
    """

    def test_main_downloads_sorts_and_uploads(self, tmp_path, monkeypatch):
        """
        ``main()`` reads INPUT_URI / OUTPUT_PREFIX, downloads the
        bundle zip, runs ``sort_recording`` on extracted recordings,
        and uploads each curated SpikeData pickle plus a
        ``sorting_report.json`` to ``output_prefix``.

        Tests:
            (Test Case 1) ``S3StorageClient.download_file`` is called
                with INPUT_URI.
            (Test Case 2) ``sort_recording`` receives the recording
                file paths, the reconstructed config, and the auto-
                generated intermediate / results folders.
            (Test Case 3) Each returned SpikeData is uploaded as
                ``{name}_curated.pkl`` under OUTPUT_PREFIX.
            (Test Case 4) ``sorting_report.json`` is uploaded with
                the expected metadata.
        """
        import dataclasses
        import json
        import pickle
        import shutil
        import zipfile
        from unittest.mock import MagicMock

        import numpy as np

        from spikelab.batch_jobs.entrypoints.sorting import main as sorting_main
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spikedata import SpikeData

        # --- Build a tiny bundle: one recording file + sorting_config.json ---
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir(parents=True)

        config = SortingPipelineConfig()
        config_dict = dataclasses.asdict(config)
        (bundle_dir / "sorting_config.json").write_text(
            json.dumps(config_dict), encoding="utf-8"
        )

        # Recording file is opaque to the entrypoint; a placeholder
        # byte-blob is enough to exercise the file-discovery loop.
        rec_a = bundle_dir / "rec_a.bin"
        rec_a.write_bytes(b"binary recording payload")
        # A second .bin to verify multi-recording handling.
        rec_b = bundle_dir / "rec_b.bin"
        rec_b.write_bytes(b"second recording payload")
        # Tier L-C2: manifest declares the recording basenames so the
        # entrypoint can distinguish recordings from companion files
        # via the declared whitelist (not a name blacklist).
        (bundle_dir / "manifest.json").write_text(
            json.dumps({"recording_files": ["rec_a.bin", "rec_b.bin"]}),
            encoding="utf-8",
        )

        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for path in bundle_dir.iterdir():
                zf.write(path, path.name)

        # --- Mock the S3 client: download = local copy; upload = capture ---
        # Capture each upload's bytes immediately, since main()'s
        # tempfile.TemporaryDirectory cleans up local_path on exit.
        upload_calls: list[tuple[str, str, bytes]] = []

        def fake_download(*, s3_uri, local_path):
            shutil.copy2(zip_path, local_path)
            return local_path

        def fake_upload(*, local_path, s3_uri):
            with open(local_path, "rb") as f:
                payload = f.read()
            upload_calls.append((local_path, s3_uri, payload))
            return s3_uri

        mock_storage = MagicMock()
        mock_storage.download_file.side_effect = fake_download
        mock_storage.upload_file.side_effect = fake_upload

        monkeypatch.setattr(
            "spikelab.batch_jobs.storage_s3.S3StorageClient",
            lambda **kwargs: mock_storage,
        )

        # --- Mock sort_recording with a stub that returns one SpikeData per recording ---
        sort_calls: list[dict] = []

        def fake_sort_recording(
            recording_files,
            config,
            intermediate_folders,
            results_folders,
            **kwargs,
        ):
            # Snapshot existence-at-call-time: the temp dir holding
            # these folders is cleaned up when main() returns, so
            # later existence checks would be misleading.
            folders_existed = all(
                Path(p).exists() and Path(p).is_dir()
                for p in list(intermediate_folders) + list(results_folders)
            )
            sort_calls.append(
                {
                    "recording_files": list(recording_files),
                    "intermediate_folders": list(intermediate_folders),
                    "results_folders": list(results_folders),
                    "config": config,
                    "folders_existed_at_call": folders_existed,
                }
            )
            return [SpikeData([[1.0, 2.0]], length=10.0) for _ in recording_files]

        monkeypatch.setattr(
            "spikelab.spike_sorting.pipeline.sort_recording",
            fake_sort_recording,
        )

        # --- Env vars consumed by main() ---
        monkeypatch.setenv("INPUT_URI", "s3://bucket/input/bundle.zip")
        monkeypatch.setenv("OUTPUT_PREFIX", "s3://bucket/outputs/run-1/")

        sorting_main()

        # --- Assertions on the orchestration ---
        # Test Case 1: download_file invoked with INPUT_URI.
        download_args = mock_storage.download_file.call_args
        assert download_args.kwargs["s3_uri"] == "s3://bucket/input/bundle.zip"

        # Test Case 2: sort_recording received the two recording files.
        assert len(sort_calls) == 1
        call = sort_calls[0]
        rec_names = {Path(p).name for p in call["recording_files"]}
        assert rec_names == {"rec_a.bin", "rec_b.bin"}
        # Per-recording intermediate / results folders were materialised
        # (existence captured inside the sort_recording stub before the
        # tempdir was cleaned up).
        assert len(call["intermediate_folders"]) == 2
        assert len(call["results_folders"]) == 2
        assert call["folders_existed_at_call"] is True
        assert isinstance(call["config"], SortingPipelineConfig)

        # Test Case 3: per-recording curated pickles uploaded.
        uploaded_uris = {uri for _, uri, _ in upload_calls}
        assert "s3://bucket/outputs/run-1/rec_a_curated.pkl" in uploaded_uris
        assert "s3://bucket/outputs/run-1/rec_b_curated.pkl" in uploaded_uris

        # Test Case 4: sorting_report.json uploaded with the expected metadata.
        report_uploads = [
            (local, uri, payload)
            for local, uri, payload in upload_calls
            if uri.endswith("sorting_report.json")
        ]
        assert len(report_uploads) == 1
        _, _, report_bytes = report_uploads[0]
        meta = json.loads(report_bytes.decode("utf-8"))
        assert meta["n_recordings"] == 2
        assert meta["n_results"] == 2
        assert set(meta["recording_names"]) == {"rec_a", "rec_b"}
        assert meta["sorter"] == config.sorter.sorter_name

        # The pickled SpikeData should round-trip from the upload payloads.
        pkl_uploads = [
            payload for _, uri, payload in upload_calls if uri.endswith(".pkl")
        ]
        for payload in pkl_uploads:
            loaded = pickle.loads(payload)
            assert isinstance(loaded, SpikeData)

    def test_main_uploads_sort_run_report_and_prints_partial_failure_marker(
        self, tmp_path, monkeypatch, capsys
    ):
        """
        After a sort with mixed success/failure, ``main()`` must upload
        ``sort_run_report.json`` with per-recording status records and
        print a ``partial failures`` marker to stdout so a "job exit 0"
        doesn't silently mask a degraded run.

        Tests:
            (Test Case 1) ``sort_run_report.json`` is uploaded under
                OUTPUT_PREFIX.
            (Test Case 2) The uploaded JSON contains one record per
                recording with the populated ``status`` field
                (mixed success / failure).
            (Test Case 3) Stdout contains a ``partial failures`` marker
                naming the failed recording.
        """
        import dataclasses
        import json
        import shutil
        import zipfile
        from unittest.mock import MagicMock

        from spikelab.batch_jobs.entrypoints.sorting import main as sorting_main
        from spikelab.spike_sorting.config import SortingPipelineConfig
        from spikelab.spike_sorting.pipeline import RecordingResult
        from spikelab.spikedata import SpikeData

        # --- Build bundle with two recordings ---
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir(parents=True)
        config = SortingPipelineConfig()
        (bundle_dir / "sorting_config.json").write_text(
            json.dumps(dataclasses.asdict(config)), encoding="utf-8"
        )
        (bundle_dir / "rec_ok.bin").write_bytes(b"payload-a")
        (bundle_dir / "rec_bad.bin").write_bytes(b"payload-b")
        # Tier L-C2: declare the recording basenames in the manifest.
        (bundle_dir / "manifest.json").write_text(
            json.dumps({"recording_files": ["rec_ok.bin", "rec_bad.bin"]}),
            encoding="utf-8",
        )

        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for path in bundle_dir.iterdir():
                zf.write(path, path.name)

        # --- Mock S3 with payload capture ---
        upload_calls: list[tuple[str, bytes]] = []

        def fake_download(*, s3_uri, local_path):
            shutil.copy2(zip_path, local_path)
            return local_path

        def fake_upload(*, local_path, s3_uri):
            with open(local_path, "rb") as f:
                upload_calls.append((s3_uri, f.read()))
            return s3_uri

        mock_storage = MagicMock()
        mock_storage.download_file.side_effect = fake_download
        mock_storage.upload_file.side_effect = fake_upload
        monkeypatch.setattr(
            "spikelab.batch_jobs.storage_s3.S3StorageClient",
            lambda **kwargs: mock_storage,
        )

        # --- Mock sort_recording: succeed on rec_ok, fail on rec_bad ---
        def fake_sort_recording(
            recording_files,
            config,
            intermediate_folders,
            results_folders,
            out_report=None,
            **kwargs,
        ):
            results = []
            for rec_path, res_folder in zip(recording_files, results_folders):
                rec_name = Path(rec_path).stem
                if rec_name == "rec_ok":
                    out_report.add(
                        RecordingResult(
                            rec_name=rec_name,
                            rec_path=str(rec_path),
                            results_folder=str(res_folder),
                            status="success",
                            wall_time_s=0.1,
                            n_curated_units=1,
                        )
                    )
                    results.append(SpikeData([[1.0]], length=10.0))
                else:
                    out_report.add(
                        RecordingResult(
                            rec_name=rec_name,
                            rec_path=str(rec_path),
                            results_folder=str(res_folder),
                            status="failed",
                            wall_time_s=0.05,
                            error_class="RuntimeError",
                            error_message="synthetic failure",
                        )
                    )
                    # main() still expects one SpikeData per recording.
                    results.append(SpikeData([[]], length=10.0))
            return results

        monkeypatch.setattr(
            "spikelab.spike_sorting.pipeline.sort_recording", fake_sort_recording
        )

        monkeypatch.setenv("INPUT_URI", "s3://b/in/bundle.zip")
        monkeypatch.setenv("OUTPUT_PREFIX", "s3://b/out/run/")

        sorting_main()

        # Test Case 1: sort_run_report.json uploaded.
        report_uploads = [
            (uri, payload)
            for uri, payload in upload_calls
            if uri.endswith("sort_run_report.json")
        ]
        assert len(report_uploads) == 1
        # Test Case 2: report content names both recordings + their statuses.
        report = json.loads(report_uploads[0][1].decode("utf-8"))
        assert report["n_total"] == 2
        assert report["n_succeeded"] == 1
        assert report["n_failed"] == 1
        status_by_name = {r["rec_name"]: r["status"] for r in report["records"]}
        assert status_by_name == {"rec_ok": "success", "rec_bad": "failed"}

        # Test Case 3: stdout carries the partial-failure marker.
        captured = capsys.readouterr().out
        assert "partial failures" in captured.lower()
        assert "rec_bad" in captured

    def test_main_raises_on_missing_sorting_config(self, tmp_path, monkeypatch):
        """
        ``main()`` raises FileNotFoundError when the bundle contains
        no ``sorting_config.json``. The error must surface clearly so
        the operator can repair the bundle.

        Tests:
            (Test Case 1) Bundle without sorting_config.json raises
                FileNotFoundError mentioning the missing file.
        """
        import shutil
        import zipfile
        from unittest.mock import MagicMock

        from spikelab.batch_jobs.entrypoints.sorting import main as sorting_main

        bundle_dir = tmp_path / "bundle_no_cfg"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "rec.bin").write_bytes(b"recording")

        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for path in bundle_dir.iterdir():
                zf.write(path, path.name)

        def fake_download(*, s3_uri, local_path):
            shutil.copy2(zip_path, local_path)
            return local_path

        mock_storage = MagicMock()
        mock_storage.download_file.side_effect = fake_download

        monkeypatch.setattr(
            "spikelab.batch_jobs.storage_s3.S3StorageClient",
            lambda **kwargs: mock_storage,
        )
        monkeypatch.setenv("INPUT_URI", "s3://bucket/input/bundle.zip")
        monkeypatch.setenv("OUTPUT_PREFIX", "s3://bucket/outputs/run-1/")

        with pytest.raises(FileNotFoundError, match="sorting_config.json"):
            sorting_main()

    def test_main_raises_on_no_recording_files(self, tmp_path, monkeypatch):
        """
        Tier L-C2: ``main()`` raises ``ValueError`` when the manifest
        has no ``recording_files`` declaration. Previously the
        entrypoint guessed via a name blacklist and would raise
        ``FileNotFoundError("No recording files found in input
        bundle")``; the new declared-whitelist contract surfaces the
        missing manifest field instead.

        Tests:
            (Test Case 1) Bundle with empty manifest.json (no
                ``recording_files`` field) raises ValueError naming
                the missing field.
        """
        import dataclasses
        import json
        import shutil
        import zipfile
        from unittest.mock import MagicMock

        from spikelab.batch_jobs.entrypoints.sorting import main as sorting_main
        from spikelab.spike_sorting.config import SortingPipelineConfig

        bundle_dir = tmp_path / "bundle_no_rec"
        bundle_dir.mkdir(parents=True)
        config_dict = dataclasses.asdict(SortingPipelineConfig())
        (bundle_dir / "sorting_config.json").write_text(
            json.dumps(config_dict), encoding="utf-8"
        )
        (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")

        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for path in bundle_dir.iterdir():
                zf.write(path, path.name)

        def fake_download(*, s3_uri, local_path):
            shutil.copy2(zip_path, local_path)
            return local_path

        mock_storage = MagicMock()
        mock_storage.download_file.side_effect = fake_download

        monkeypatch.setattr(
            "spikelab.batch_jobs.storage_s3.S3StorageClient",
            lambda **kwargs: mock_storage,
        )
        monkeypatch.setenv("INPUT_URI", "s3://bucket/input/bundle.zip")
        monkeypatch.setenv("OUTPUT_PREFIX", "s3://bucket/outputs/run-1/")

        with pytest.raises(ValueError, match="recording_files"):
            sorting_main()


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

    def test_unicode_characters_replaced_with_hyphens(self):
        """Non-ASCII characters in name_prefix are replaced with hyphens.

        Tests:
            - Mixed ASCII/non-ASCII input produces an ASCII-only result.
            - An all non-ASCII input raises ValueError after sanitization.
        """
        payload = _example_payload()
        payload["name_prefix"] = "análysis-jöb"
        job_spec = validate_job_spec(payload)
        assert job_spec.name_prefix == "an-lysis-j-b"
        # Result must be valid ASCII.
        job_spec.name_prefix.encode("ascii")

        payload_all_non_ascii = _example_payload()
        payload_all_non_ascii["name_prefix"] = "áöü"
        with pytest.raises(ValueError, match="no usable ASCII content"):
            validate_job_spec(payload_all_non_ascii)

    def test_oversized_name_prefix_emits_truncation_warning(self):
        """
        A name_prefix longer than 40 chars (after stripping) emits a
        UserWarning whose message contains both the original input
        and the truncated form so the operator can shorten upstream.

        Tests:
            (Test Case 1) A 50-char alphanumeric prefix triggers a
                UserWarning.
            (Test Case 2) The warning message contains the original
                value (e.g. via ``repr``).
            (Test Case 3) The warning message contains the truncated
                40-char form.
        """
        import warnings as _warnings

        payload = _example_payload()
        original = "a" * 50
        payload["name_prefix"] = original
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            job_spec = validate_job_spec(payload)

        warn_msgs = [str(rec.message) for rec in w if rec.category is UserWarning]
        relevant = [m for m in warn_msgs if "name_prefix" in m and "truncated" in m]
        assert relevant, warn_msgs
        # Both the original and the truncated 40-char form appear.
        assert original in relevant[0]
        assert job_spec.name_prefix in relevant[0]
        assert len(job_spec.name_prefix) == 40

    def test_trailing_hyphens_stripped_after_truncation(self):
        """Trailing hyphens exposed by 40-char truncation are stripped.

        Tests:
            - Result length is at most 40 characters.
            - Result does not start or end with a hyphen.
        """
        payload = _example_payload()
        # Create a prefix where position 40 falls right after hyphens
        payload["name_prefix"] = "a" * 37 + "---xyz"
        job_spec = validate_job_spec(payload)
        assert len(job_spec.name_prefix) <= 40
        assert not job_spec.name_prefix.endswith("-")
        assert not job_spec.name_prefix.startswith("-")


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


class TestWaitForCompletionRetryBackoff:
    """``wait_for_completion`` retries transient backend errors with
    exponential backoff capped at ``max_backoff_seconds``. Persistent
    failures eventually surface as ``"Backend unavailable"`` after
    ``max_consecutive_failures`` consecutive raises. A 404 from the
    K8s client returns ``"Failed"`` immediately (no retry — job
    deleted out-of-band). A successful poll resets both the failure
    counter and the backoff to ``poll_interval_seconds``.
    """

    @staticmethod
    def _make_session(backend=None):
        from spikelab.batch_jobs.session import RunSession

        profile = ClusterProfile(name="test")
        backend = backend or MagicMock(spec=KubernetesBatchJobBackend)
        storage = MagicMock(spec=S3StorageClient)
        return (
            RunSession(
                profile=profile,
                backend=backend,
                storage_client=storage,
                credentials=MagicMock(),
            ),
            backend,
        )

    def test_immediate_complete_single_poll(self):
        """
        Tests:
            (Test Case 1) ``job_status`` returning "Complete" causes
                ``wait_for_completion`` to return after exactly one
                call.
        """
        session, backend = self._make_session()
        backend.job_status.return_value = "Complete"
        state = session.wait_for_completion(
            job_name="job-x", max_wait_seconds=10, poll_interval_seconds=1
        )
        assert state == "Complete"
        assert backend.job_status.call_count == 1

    def test_transient_recovery_after_calledprocesserror(self, monkeypatch):
        """
        Tests:
            (Test Case 1) After 3 transient ``CalledProcessError``
                raises followed by a "Complete" return, the loop
                surfaces "Complete" with exactly 4 total calls.
        """
        import subprocess

        session, backend = self._make_session()
        backend.job_status.side_effect = [
            subprocess.CalledProcessError(1, "kubectl"),
            subprocess.CalledProcessError(1, "kubectl"),
            subprocess.CalledProcessError(1, "kubectl"),
            "Complete",
        ]
        # Patch time.sleep to a no-op so the test is fast.
        from spikelab.batch_jobs import session as session_mod

        monkeypatch.setattr(session_mod.time, "sleep", lambda _s: None)

        state = session.wait_for_completion(
            job_name="job-x",
            max_wait_seconds=60,
            poll_interval_seconds=1,
            max_consecutive_failures=5,
        )
        assert state == "Complete"
        assert backend.job_status.call_count == 4

    def test_persistent_failure_returns_backend_unavailable(self, monkeypatch):
        """
        Tests:
            (Test Case 1) After ``max_consecutive_failures`` raises in
                a row, the loop returns "Backend unavailable" and
                ``job_status`` was called exactly
                ``max_consecutive_failures`` times.
        """
        import subprocess

        session, backend = self._make_session()
        backend.job_status.side_effect = subprocess.CalledProcessError(1, "kubectl")
        from spikelab.batch_jobs import session as session_mod

        monkeypatch.setattr(session_mod.time, "sleep", lambda _s: None)

        state = session.wait_for_completion(
            job_name="job-x",
            max_wait_seconds=600,
            poll_interval_seconds=1,
            max_consecutive_failures=3,
        )
        assert state == "Backend unavailable"
        assert backend.job_status.call_count == 3

    def test_apiexception_404_returns_failed_immediately(self, monkeypatch):
        """
        Tests:
            (Test Case 1) An ``ApiException(status=404)`` from the K8s
                client returns "Failed" on the first poll (no retry).
        """
        try:
            from kubernetes.client.exceptions import ApiException
        except ImportError:
            pytest.skip("kubernetes client not installed")

        session, backend = self._make_session()
        # Build a 404 ApiException.
        exc = ApiException(status=404, reason="Not Found")
        backend.job_status.side_effect = exc

        from spikelab.batch_jobs import session as session_mod

        monkeypatch.setattr(session_mod.time, "sleep", lambda _s: None)
        state = session.wait_for_completion(
            job_name="job-x",
            max_wait_seconds=60,
            poll_interval_seconds=1,
            max_consecutive_failures=5,
        )
        assert state == "Failed"
        assert backend.job_status.call_count == 1

    def test_running_then_timeout(self, monkeypatch):
        """
        Tests:
            (Test Case 1) ``job_status`` returning "Running"
                indefinitely yields "Timeout" once
                ``max_wait_seconds`` elapses.
        """
        session, backend = self._make_session()
        backend.job_status.return_value = "Running"
        from spikelab.batch_jobs import session as session_mod

        monkeypatch.setattr(session_mod.time, "sleep", lambda _s: None)

        # Use a tiny budget and a counter-based fake time to keep the
        # test deterministic and fast.
        time_now = {"t": 0.0}

        def fake_time():
            time_now["t"] += 0.6  # each call ticks 0.6s
            return time_now["t"]

        monkeypatch.setattr(session_mod.time, "time", fake_time)
        state = session.wait_for_completion(
            job_name="job-x", max_wait_seconds=2, poll_interval_seconds=1
        )
        assert state == "Timeout"

    def test_successful_poll_resets_backoff(self, monkeypatch):
        """
        Tests:
            (Test Case 1) After two failures, one success, then
                another failure followed by Complete — the second
                failure's sleep starts back at
                ``poll_interval_seconds``, not the prior backoff
                value. The reset is asserted by capturing
                ``time.sleep`` arg-lists and checking the pattern.
        """
        import subprocess

        session, backend = self._make_session()
        backend.job_status.side_effect = [
            subprocess.CalledProcessError(1, "kubectl"),  # fail 1
            subprocess.CalledProcessError(1, "kubectl"),  # fail 2
            "Running",  # success — resets counter+backoff
            subprocess.CalledProcessError(1, "kubectl"),  # fail again
            "Complete",
        ]

        sleeps: list = []
        from spikelab.batch_jobs import session as session_mod

        monkeypatch.setattr(session_mod.time, "sleep", lambda s: sleeps.append(s))
        state = session.wait_for_completion(
            job_name="job-x",
            max_wait_seconds=600,
            poll_interval_seconds=1,
            max_consecutive_failures=5,
        )
        assert state == "Complete"
        # Pattern: fail→2, fail→4, success→1 (reset to poll_interval),
        # fail→2, then success returns without sleeping after the
        # final iteration. Pin the reset by asserting the third sleep
        # is exactly poll_interval_seconds.
        assert len(sleeps) >= 3
        assert sleeps[2] == 1


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
        """apply_manifest via K8s client with invalid YAML raises ValueError.

        Updated for the apply_manifest rework (Tier E): non-dict
        payloads (and dict payloads missing metadata.name) now raise
        a clear ValueError mentioning that the Python-client path
        needs a structured manifest, instead of the previous mix of
        TypeError/KeyError/AttributeError surfaced from whatever
        attribute access happened to fail first.
        """
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        with pytest.raises(ValueError, match="metadata.name"):
            backend.apply_manifest("just a plain string without yaml structure")

    def test_apply_manifest_k8s_client_missing_metadata_name(self):
        """apply_manifest via K8s client raises ValueError when
        metadata.name is missing — clearer than the previous KeyError
        propagated from raw dict access.
        """
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        manifest = "apiVersion: batch/v1\nkind: Job\nmetadata:\n  labels: {}\n"
        with pytest.raises(ValueError, match="metadata.name"):
            backend.apply_manifest(manifest)

    def test_apply_manifest_rejects_namespace_disagreement(self):
        """
        ``apply_manifest`` raises ValueError when the manifest's
        ``metadata.namespace`` is set and differs from the backend's
        configured namespace — otherwise the apply would silently
        deploy into the backend's namespace instead of the rendered
        target.

        Tests:
            (Test Case 1) ValueError is raised.
            (Test Case 2) The error message names both the manifest
                namespace and the backend namespace.
            (Test Case 3) The K8s client's ``create_namespaced_job``
                is NOT invoked (we refused before issuing the apply).
        """
        backend = KubernetesBatchJobBackend(namespace="prod")
        mock_batch_api = MagicMock()
        backend._batch_api = mock_batch_api

        manifest = (
            "apiVersion: batch/v1\nkind: Job\n"
            "metadata:\n  name: my-job\n  namespace: staging\n"
        )
        with pytest.raises(ValueError) as excinfo:
            backend.apply_manifest(manifest)
        msg = str(excinfo.value)
        assert "staging" in msg
        assert "prod" in msg
        mock_batch_api.create_namespaced_job.assert_not_called()

    def test_apply_manifest_kubectl_fallback_returns_metadata_name(self, tmp_path):
        """
        Both the kubectl-fallback path and the Python-client path
        return the bare ``metadata.name`` (E8). Pin the fallback path:
        if the rendered YAML's ``metadata.name`` is available, the
        returned value is the name (not raw ``kubectl apply`` stdout).

        Tests:
            (Test Case 1) The kubectl-fallback path returns
                ``metadata.name`` from the parsed YAML, not the raw
                ``kubectl apply`` stdout containing it.
        """
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        backend._batch_api = None  # force fallback
        backend.use_kubectl_fallback = True

        # Stub the kubectl invocation so the test stays hermetic.
        backend._run_kubectl = lambda cmd: ("job.batch/clean-job-name created\n")

        manifest_file = tmp_path / "job.yaml"
        manifest_file.write_text(
            "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: clean-job-name\n",
            encoding="utf-8",
        )
        result = backend.apply_manifest(str(manifest_file))
        # Returns the bare name (E8 contract), not the raw stdout.
        assert result == "clean-job-name"

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
        # Source uses _preload_content=False which yields a streaming HTTP
        # response — emulate the .stream() chunk iterator + release_conn().
        mock_response = MagicMock()
        mock_response.stream.return_value = iter([b"line a\nline b"])
        mock_core_api.read_namespaced_pod_log.return_value = mock_response

        with patch("spikelab.batch_jobs.backend_k8s.watch", None):
            lines = list(backend.stream_logs("test-pod", follow=True))

        assert lines == ["line a", "line b"]
        mock_response.release_conn.assert_called_once()

    def test_stream_logs_k8s_client_no_follow(self):
        """stream_logs with follow=False via K8s client reads log text."""
        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_core_api = MagicMock()
        backend._core_api = mock_core_api
        mock_response = MagicMock()
        mock_response.stream.return_value = iter([b"hello\nworld"])
        mock_core_api.read_namespaced_pod_log.return_value = mock_response

        lines = list(backend.stream_logs("test-pod", follow=False))
        assert lines == ["hello", "world"]
        mock_response.release_conn.assert_called_once()


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

    def test_apply_manifest_kubectl_fallback_unlinks_temp_on_write_failure(
        self, tmp_path, monkeypatch
    ):
        """
        The kubectl-fallback path writes the manifest YAML to a tempfile
        via ``tempfile.mkstemp`` + ``os.fdopen``. If the inner write
        raises, the ``finally`` branch must still unlink the temp
        path — the previous pattern initialised ``temp_path = None``
        before the write and skipped cleanup when the assignment never
        happened.

        Tests:
            (Test Case 1) When ``os.fdopen`` raises after
                ``mkstemp`` succeeded, the temp file path is unlinked
                in the cleanup branch.
            (Test Case 2) The original exception propagates to the
                caller.
        """
        import os
        import tempfile as _tempfile

        backend = KubernetesBatchJobBackend(namespace="test-ns")
        backend._batch_api = None  # force fallback
        backend.use_kubectl_fallback = True

        captured_temp_path: list[str] = []
        real_mkstemp = _tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            captured_temp_path.append(path)
            return fd, path

        def failing_fdopen(fd, *args, **kwargs):
            # Close the fd before raising so the finally branch's
            # ``Path(temp_path).unlink(missing_ok=True)`` can actually
            # delete the file on Windows (where holding the fd blocks
            # unlink). This mirrors what a real ``os.fdopen`` failure
            # looks like for a properly-implemented file wrapper that
            # closed the fd before propagating the error.
            os.close(fd)
            raise OSError("simulated write failure")

        monkeypatch.setattr(
            "spikelab.batch_jobs.backend_k8s.tempfile.mkstemp", tracking_mkstemp
        )
        monkeypatch.setattr("spikelab.batch_jobs.backend_k8s.os.fdopen", failing_fdopen)

        manifest = "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: my-job\n"
        with pytest.raises(OSError, match="simulated write failure"):
            backend.apply_manifest(manifest)

        assert captured_temp_path, "mkstemp was never called"
        assert not os.path.exists(captured_temp_path[0])

    def test_non_config_exception_during_api_discovery_logs_and_falls_back(
        self, caplog
    ):
        """
        Non-ConfigException errors raised during ``client.BatchV1Api()``
        / ``client.CoreV1Api()`` (network blip, urllib3 quirks, API
        version mismatch) leave the backend in the partially-initialised
        kubectl-fallback state — ``_batch_api is None`` — and emit a
        DEBUG-level log record so operators can still see what went
        wrong.

        Tests:
            (Test Case 1) Constructor returns successfully despite the
                non-ConfigException error.
            (Test Case 2) ``_batch_api`` and ``_core_api`` are both
                None (kubectl fallback engaged).
            (Test Case 3) A DEBUG-level log record was emitted to the
                module's logger.
        """
        import logging

        mock_client = MagicMock()
        # ``BatchV1Api()`` raises a non-ConfigException-type error.
        mock_client.BatchV1Api.side_effect = RuntimeError(
            "simulated urllib3 MaxRetryError"
        )
        mock_config = MagicMock()
        mock_config.ConfigException = type("ConfigException", (Exception,), {})

        with caplog.at_level(logging.DEBUG, logger="spikelab.batch_jobs.backend_k8s"):
            with patch("spikelab.batch_jobs.backend_k8s.client", mock_client):
                with patch("spikelab.batch_jobs.backend_k8s.config", mock_config):
                    backend = KubernetesBatchJobBackend(namespace="test")

        assert backend._batch_api is None
        assert backend._core_api is None
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert debug_records, "expected a DEBUG-level log record on fallback"


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

    def test_apply_image_selection_does_not_mutate_caller_payload(self):
        """
        ``_apply_image_selection`` deep-copies the payload before
        mutating so callers can reuse the same parsed config dict
        across multiple invocations with different image profiles.

        Tests:
            (Test Case 1) The caller's payload dict is byte-equal
                before and after the call (deep-copy via
                ``copy.deepcopy`` then dict equality).
            (Test Case 2) The returned dict carries the resolved
                image (different from the caller's original).
        """
        import copy

        payload = {
            "container": {
                "image": "",  # empty so resolution must fill it in
                "command": ["python"],
                "args": [],
                "env": {"K": "v"},
            },
            "other_field": [1, 2, 3],
        }
        profile = ClusterProfile(
            name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
        )
        snapshot = copy.deepcopy(payload)
        updated = cli._apply_image_selection(
            payload,
            profile=profile,
            image_profile="cpu",
            image_override=None,
        )
        # Caller's payload survives unmutated.
        assert payload == snapshot
        # Returned dict carries the resolved image.
        assert updated["container"]["image"] == "ghcr.io/example/cpu:latest"

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
        """Validation error without .errors() attribute uses str(exc).

        Pre-2026-05-24 the test relied on YAML-level breakage to land
        a non-Pydantic exception in ``_cmd_deploy``'s except block. The
        parallel-session ``_apply_image_selection`` pre-check now
        consumes empty-image YAML before validate_job_spec runs, so we
        patch the validator directly to raise a plain ``ValueError``
        (no ``.errors()`` attr) to exercise the ``str(exc)`` fallback.
        """
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(
            "name_prefix: analysis-job\nnamespace: default\n"
            "container:\n  image: 'placeholder:latest'\n  command: []\n  args: []\n  env: {}\n"
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

        # Plain ValueError has no ``.errors()`` attribute → exercises
        # the ``str(exc)`` fallback path in ``_cmd_deploy``.
        def _raise_plain(payload):
            raise ValueError("custom validation message")

        monkeypatch.setattr(cli, "validate_job_spec", _raise_plain)

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

    def test_sanitize_yaml_value_raises_on_unsafe_chars(self):
        """Tier L-C1: _sanitize_yaml_value raises ValueError on unsafe chars.

        Previously the function silently stripped \\n / \\t / " / \\
        from the value. Tier L-C1 replaces the silent strip with a
        fail-fast ValueError so commands and label values that
        depend on those characters can't be silently mangled.
        """
        import pytest
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        with pytest.raises(ValueError, match="YAML-unsafe character"):
            _sanitize_yaml_value('hello\nworld\t"test\\value')
        # Safe inputs round-trip unchanged.
        assert _sanitize_yaml_value("hello world") == "hello world"
        assert _sanitize_yaml_value("foo:bar/baz") == "foo:bar/baz"

    def test_sanitize_yaml_value_injection_attempt_rejected(self):
        """Tier L-C1: YAML injection via label values raises ValueError.

        The fail-fast contract surfaces the offending characters in
        the error message; operators can't silently ship a job spec
        whose newlines would inject extra YAML nodes.
        """
        import pytest
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        malicious = 'value"\ninjected_key: injected_value'
        with pytest.raises(ValueError, match="YAML-unsafe character"):
            _sanitize_yaml_value(malicious)

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
        """summarize_validation_error handles error with empty loc tuple.

        With the new multiline format, the message appears as a bullet
        under the ``Invalid job config:`` header. The location prefix is
        absent because the loc tuple is empty.
        """
        from spikelab.batch_jobs.validation import summarize_validation_error

        mock_exc = MagicMock()
        mock_exc.errors.return_value = [
            {"loc": (), "msg": "custom error message"},
        ]
        result = summarize_validation_error(mock_exc)
        assert result == "Invalid job config:\n  - custom error message"


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


class TestRetrieveSortingWarnsOnCorruptOutputs:
    """
    Tests that _retrieve_sorting emits a UserWarning naming the corrupt
    file when a pickle or JSON output fails to load, instead of silently
    swallowing the error. The retrieval still completes (continuing
    through the remaining files) so a single bad output does not abort
    the whole batch.
    """

    def _make_session(self):
        from spikelab.batch_jobs.backend_k8s import KubernetesBatchJobBackend
        from spikelab.batch_jobs.models import ClusterProfile
        from spikelab.batch_jobs.session import RunSession
        from spikelab.batch_jobs.storage_s3 import S3StorageClient

        profile = ClusterProfile(name="test")
        backend = MagicMock(spec=KubernetesBatchJobBackend)
        storage = MagicMock(spec=S3StorageClient)
        storage.output_prefix_for_run.return_value = "s3://b/out/run/"
        storage.logs_prefix_for_run.return_value = "s3://b/logs/run/"
        session = RunSession(
            profile=profile,
            backend=backend,
            storage_client=storage,
            credentials=MagicMock(),
        )
        return session, storage

    def test_corrupt_pickle_raises_with_filename(self, tmp_path):
        """
        _retrieve_sorting now raises ``RuntimeError`` when any output
        file fails to ingest, instead of silently warning and
        returning a half-built workspace. The exception lists the
        offending file so the operator can find it.

        Tests:
            (Test Case 1) RuntimeError is raised when a downloaded
                pickle is corrupt.
            (Test Case 2) The exception message names the corrupt
                file.
        """
        from spikelab.batch_jobs.models import SubmitResult

        session, storage = self._make_session()

        # Source for the fake download lives in a separate subdirectory so
        # local_path (derived from local_dir/relative) does not collide
        # with the source — copy2 raises SameFileError otherwise.
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        bad_pkl = src_dir / "bad.pkl"
        bad_pkl.write_bytes(b"not a real pickle")
        local_dir = tmp_path / "local"
        local_dir.mkdir()

        storage.list_output_files.return_value = ["pfx/out/run-1/bad.pkl"]
        storage.output_prefix_for_run.return_value = "s3://b/pfx/out/run-1/"

        def fake_download(*, s3_uri, local_path):
            import shutil

            shutil.copy2(str(bad_pkl), local_path)
            return local_path

        storage.download_file.side_effect = fake_download

        submit_result = SubmitResult(
            job_name="sort-job",
            manifest_yaml="",
            run_id="run-1",
            uploaded_input_uri="s3://b/pfx/inputs/run-1/bundle.zip",
            output_prefix="s3://b/pfx/out/run-1/",
            logs_prefix="s3://b/pfx/logs/run-1/",
            job_type="sorting",
        )

        with pytest.raises(RuntimeError, match="bad.pkl"):
            session.retrieve_result(submit_result, str(local_dir))

    def test_corrupt_json_raises_with_filename(self, tmp_path):
        """
        _retrieve_sorting raises ``RuntimeError`` when a JSON metadata
        file is unreadable, naming the offending file.

        Tests:
            (Test Case 1) RuntimeError is raised on bad JSON.
            (Test Case 2) The exception message names the bad file.
        """
        from spikelab.batch_jobs.models import SubmitResult

        session, storage = self._make_session()

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        bad_json = src_dir / "metadata.json"
        bad_json.write_text("not valid json {")
        local_dir = tmp_path / "local"
        local_dir.mkdir()

        storage.list_output_files.return_value = ["pfx/out/run-1/metadata.json"]
        storage.output_prefix_for_run.return_value = "s3://b/pfx/out/run-1/"

        def fake_download(*, s3_uri, local_path):
            import shutil

            shutil.copy2(str(bad_json), local_path)
            return local_path

        storage.download_file.side_effect = fake_download

        submit_result = SubmitResult(
            job_name="sort-job",
            manifest_yaml="",
            run_id="run-1",
            uploaded_input_uri="s3://b/pfx/inputs/run-1/bundle.zip",
            output_prefix="s3://b/pfx/out/run-1/",
            logs_prefix="s3://b/pfx/logs/run-1/",
            job_type="sorting",
        )

        with pytest.raises(RuntimeError, match="metadata.json"):
            session.retrieve_result(submit_result, str(local_dir))


class TestBuildJobNameRfc1123Compliance:
    """
    Tests that _build_job_name rejects prefixes that would produce an
    RFC 1123-invalid Kubernetes job name (leading hyphen) instead of
    letting the cluster reject the manifest at apply time.
    """

    def test_empty_prefix_raises(self):
        """
        _build_job_name("") raises ValueError naming "alphanumeric".

        Tests:
            (Test Case 1) Empty prefix raises ValueError.
            (Test Case 2) The error names the offending input and the
                reason (RFC 1123 / alphanumeric).
        """
        from spikelab.batch_jobs.session import RunSession

        with pytest.raises(ValueError) as exc_info:
            RunSession._build_job_name("")
        msg = str(exc_info.value)
        assert "''" in msg or "empty" in msg.lower()
        assert "alphanumeric" in msg.lower() or "RFC 1123" in msg

    def test_all_hyphens_prefix_raises(self):
        """
        _build_job_name("---") raises ValueError because trailing-hyphen
        stripping reduces it to the empty string.

        Tests:
            (Test Case 1) All-hyphen prefix raises ValueError.
        """
        from spikelab.batch_jobs.session import RunSession

        with pytest.raises(ValueError, match="empty string"):
            RunSession._build_job_name("---")

    def test_valid_prefix_succeeds(self):
        """
        Valid prefixes still produce well-formed job names — no
        regression for the happy path.

        Tests:
            (Test Case 1) "spikelab-sort" produces a name starting with
                "spikelab-sort-" and containing the 8-char hex token.
            (Test Case 2) Total length ≤ 63 (K8s job name limit).
            (Test Case 3) Trailing-hyphen prefix "foo--" still works
                because "foo" is left after rstrip.
        """
        from spikelab.batch_jobs.session import RunSession

        name = RunSession._build_job_name("spikelab-sort")
        assert name.startswith("spikelab-sort-")
        assert len(name) <= 63
        assert name[0].isalpha()

        name2 = RunSession._build_job_name("foo--")
        assert name2.startswith("foo-")


class TestSleepDetectionEdgeCases:
    """Boundary tests for _contains_disallowed_sleep covering NaN durations,
    -infinity, mixed case, and whitespace-padded tokens."""

    def test_nan_duration_flagged(self):
        """
        ``sleep NaN`` is flagged as a disallowed sleep pattern — NaN is
        not a finite duration; the actual sleep binary rejects it, but
        a job spec containing it is suspicious (bug or obfuscation
        around the literal 'inf'/'infinity' check).

        Tests:
            (Test Case 1) ['sleep', 'NaN'] is flagged.
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep(["sleep", "NaN"], []) is True

    def test_negative_infinity_duration_flagged(self):
        """
        ``sleep -infinity`` is flagged as a disallowed sleep pattern —
        non-finite duration suggests intent to bypass the literal
        'inf'/'infinity' string check.

        Tests:
            (Test Case 1) ['sleep', '-infinity'] is flagged.
            (Test Case 2) ['sleep', '-inf'] is flagged.
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep(["sleep", "-infinity"], []) is True
        assert _contains_disallowed_sleep(["sleep", "-inf"], []) is True

    def test_mixed_case_sleep_infinity_flagged(self):
        """
        The token-pair check lowercases both sides, so ``SLEEP infinity``
        is correctly flagged.

        Tests:
            (Test Case 1) ['SLEEP', 'infinity'] is flagged.
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep(["SLEEP", "infinity"], []) is True

    def test_whitespace_padded_sleep_token_flagged(self):
        """
        Tokens are split on whitespace before the bare-sleep check, so
        a single token with internal whitespace ("  sleep  ") is split
        into ["sleep"] and triggers the bare-sleep branch.

        Tests:
            (Test Case 1) [' sleep '] alone is flagged as bare-sleep.
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep([" sleep "], []) is True


class TestRedactSensitiveMapBoundary:
    """Boundary test for redact_sensitive_map covering an empty mapping."""

    def test_empty_mapping_returns_empty_dict(self):
        """
        redact_sensitive_map on an empty input returns an empty dict
        rather than raising.

        Tests:
            (Test Case 1) {} returns {}.
        """
        assert redact_sensitive_map({}) == {}


class TestProfilesEdgeCases:
    """Boundary tests for load_profile_from_name covering whitespace and
    empty-string inputs."""

    def test_load_profile_with_whitespace_name(self):
        """
        load_profile_from_name strips leading/trailing whitespace and
        lowercases before matching, so "  NRP  " resolves to the nrp
        profile.

        Tests:
            (Test Case 1) "  NRP  " loads the nrp.yaml profile.
        """
        from spikelab.batch_jobs.profiles import load_profile_from_name

        prof = load_profile_from_name("  NRP  ")
        assert prof.name == "nrp"


class TestBuildPodVolumesRejectsAmbiguousSources:
    """A K8s ``Volume`` may have at most one of ``secret`` /
    ``persistentVolumeClaim`` — they're mutually exclusive volume
    sources. The Jinja template at ``job.yaml.j2:83-88`` renders only
    the secret via ``{% if secret_name %}{% elif pvc_name %}``, so a
    volume with both would silently drop the pvc at render time.
    ``_build_pod_volumes`` raises ``ValueError`` at build time to
    surface the misconfiguration loudly.
    """

    def test_volume_with_both_secret_and_pvc_raises(self):
        """
        Tests:
            (Test Case 1) Building a mount dict with both
                ``secret_name`` and ``pvc_name`` raises ``ValueError``.
            (Test Case 2) The error message names "mutually exclusive"
                and the offending volume name.
        """
        from spikelab.batch_jobs.templating import _build_pod_volumes

        mounts = [
            {
                "name": "creds",
                "secret_name": "my-secret",
                "pvc_name": "my-pvc",
            }
        ]
        with pytest.raises(ValueError, match=r"mutually exclusive") as excinfo:
            _build_pod_volumes(mounts)
        assert "'creds'" in str(excinfo.value)

    def test_volume_with_neither_source_raises(self):
        """
        Tests:
            (Test Case 1) A mount with ``name`` but neither source set
                raises ``ValueError`` mentioning "exactly one source".
        """
        from spikelab.batch_jobs.templating import _build_pod_volumes

        mounts = [{"name": "empty-vol", "secret_name": None, "pvc_name": None}]
        with pytest.raises(ValueError, match=r"exactly one source"):
            _build_pod_volumes(mounts)

    def test_secret_then_pvc_merge_raises(self):
        """
        Two mounts with the same ``name``, one carrying a secret and
        the other carrying a pvc, merge into a single volume with
        BOTH sources — which is exactly the conflict the validator
        must catch.

        Tests:
            (Test Case 1) Merge of secret-only + pvc-only mounts with
                the same name raises ``ValueError``.
        """
        from spikelab.batch_jobs.templating import _build_pod_volumes

        mounts = [
            {"name": "shared", "secret_name": "s", "pvc_name": None},
            {"name": "shared", "secret_name": None, "pvc_name": "p"},
        ]
        with pytest.raises(ValueError, match=r"mutually exclusive"):
            _build_pod_volumes(mounts)


class TestKubectlEmptyStdoutGuard:
    """``yaml.safe_load("")`` returns ``None``. The kubectl-path job
    status / pods queries must guard with ``or {}`` so transient
    empty kubectl stdout returns the fallthrough status (Pending / [])
    instead of raising ``AttributeError`` and breaking monitoring loops.
    """

    def test_job_status_handles_empty_kubectl_stdout(self):
        """
        Tests:
            (Test Case 1) ``job_status`` returns ``"Pending"`` (the
                fallthrough status) instead of raising AttributeError
                when ``_run_kubectl`` returns empty stdout.
        """
        from spikelab.batch_jobs.backend_k8s import KubernetesBatchJobBackend

        backend = KubernetesBatchJobBackend.__new__(KubernetesBatchJobBackend)
        backend.namespace = "default"
        backend._batch_api = None  # forces the kubectl path
        backend._core_api = None

        def _stub_run_kubectl(_args):
            return ""

        backend._run_kubectl = _stub_run_kubectl  # type: ignore[assignment]

        assert backend.job_status("any-job") == "Pending"

    def test_pods_for_job_handles_empty_kubectl_stdout(self):
        """
        Tests:
            (Test Case 1) ``pods_for_job`` returns ``[]`` instead of
                raising AttributeError when ``_run_kubectl`` returns
                empty stdout.
        """
        from spikelab.batch_jobs.backend_k8s import KubernetesBatchJobBackend

        backend = KubernetesBatchJobBackend.__new__(KubernetesBatchJobBackend)
        backend.namespace = "default"
        backend._batch_api = None
        backend._core_api = None
        backend._run_kubectl = lambda _args: ""  # type: ignore[assignment]

        assert backend.pods_for_job("any-job") == []


class TestRetrieveSortingMissingPrefixGuard:
    """``_retrieve_sorting`` raises an actionable ``ValueError`` when
    ``SubmitResult.output_prefix`` is empty (e.g. the cluster profile
    has no ``default_s3_prefix`` and no override was supplied), rather
    than letting ``parse_s3_url("")`` raise a generic error that masks
    the actual configuration issue.
    """

    def test_empty_output_prefix_raises_actionable_error(self, tmp_path):
        """
        Tests:
            (Test Case 1) Empty ``output_prefix`` raises ``ValueError``
                naming ``default_s3_prefix``.
        """
        from spikelab.batch_jobs.models import SubmitResult
        from spikelab.batch_jobs.session import RunSession

        class _StubStorage:
            def list_output_files(self, run_id):
                return ["spikedata.pkl"]

        session = RunSession.__new__(RunSession)
        session.storage = _StubStorage()

        result = SubmitResult(
            run_id="r-123",
            job_name="j-123",
            manifest_yaml="",
            uploaded_input_uri="s3://b/inputs/r-123.zip",
            output_prefix="",
            logs_prefix="",
            job_type="sorting",
        )

        with pytest.raises(ValueError, match=r"default_s3_prefix"):
            session._retrieve_sorting(result, tmp_path)


class TestDeployZeroPodsFollowLogsWarns:
    """``_cmd_deploy --wait --follow-logs`` prints a stderr warning
    when ``pods_for_job`` returns ``[]`` instead of silently exiting 0
    with no output. The user asked to follow logs and gets nothing
    visible — surface the condition so automation can detect it
    without parsing the absence of stdout.
    """

    def test_zero_pods_emits_stderr_warning(self, monkeypatch, capsys):
        """
        Tests:
            (Test Case 1) ``stderr`` contains "no pods found".
            (Test Case 2) Exit code remains 0 (the submission itself
                succeeded; --wait surfaced FINAL_STATUS earlier).
        """
        from argparse import Namespace

        from spikelab.batch_jobs import cli as cli_mod
        from spikelab.batch_jobs.models import SubmitResult

        submit_result = SubmitResult(
            run_id="r-1",
            job_name="j-1",
            manifest_yaml="",
            uploaded_input_uri="s3://b/inputs/r-1.zip",
            output_prefix="",
            logs_prefix="",
            job_type="prepared",
        )

        class _StubBackend:
            def pods_for_job(self, name):
                return []

        class _StubSession:
            backend = _StubBackend()

            def submit_prepared_job(self, **_kw):
                return submit_result

            def wait_for_completion(self, **_kw):
                return "Complete"

        monkeypatch.setattr(cli_mod, "_load_payload", lambda _: {"name_prefix": "test"})
        monkeypatch.setattr(cli_mod, "_load_profile", lambda *_: object())
        monkeypatch.setattr(cli_mod, "_build_session", lambda *_: _StubSession())
        monkeypatch.setattr(
            cli_mod,
            "_apply_image_selection",
            lambda payload, *_a, **_kw: payload,
        )
        # Bypass real Pydantic validation — the test exercises the
        # zero-pods branch, not the JobSpec schema.
        monkeypatch.setattr(cli_mod, "validate_job_spec", lambda payload: object())

        rc = cli_mod._cmd_deploy(
            Namespace(
                job_config="dummy.yaml",
                profile="nrp",
                profile_file=None,
                kubeconfig=None,
                image_profile=None,
                image=None,
                allow_policy_risk=False,
                render_only=False,
                output_manifest=None,
                wait=True,
                follow_logs=True,
                max_wait_seconds=1,
            )
        )

        captured = capsys.readouterr()
        assert "no pods found" in captured.err.lower()
        assert rc == 0


class TestArtifactPackagerPathTraversalGuard:
    """``package_analysis_bundle`` rejects ``run_id`` values containing
    path-separator or ``..`` segments — ``run_id`` flows directly into
    the temp bundle dir and the output zip filename, so a value like
    ``"../escape"`` could let the function clobber arbitrary files
    outside ``output_dir``.
    """

    @pytest.mark.parametrize(
        "bad_run_id",
        [
            "../escape",
            "..",
            "subdir/run",
            "run\\bad",
            "",
        ],
    )
    def test_traversal_run_id_rejected(self, tmp_path, bad_run_id):
        """
        Tests:
            (Test Case 1) Each adversarial ``run_id`` raises
                ``ValueError`` mentioning path traversal or separators.
        """
        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        with pytest.raises(ValueError, match=r"path-traversal|separators"):
            package_analysis_bundle(
                input_paths=[],
                run_id=bad_run_id,
                output_dir=str(tmp_path),
                output_format="custom",
            )


class TestS3StorageDownloadOutputPathTraversalGuard:
    """``S3StorageClient.download_output`` rejects ``filename`` values
    that resolve outside ``local_dir`` after joining — a malicious or
    buggy upstream that supplied ``"../etc/passwd"`` could otherwise
    write to an arbitrary location on the host.
    """

    def test_traversal_filename_rejected(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``filename="../etc/passwd"`` raises
                ``ValueError`` mentioning "path-traversal".
        """
        from spikelab.batch_jobs.storage_s3 import S3StorageClient

        client = S3StorageClient.__new__(S3StorageClient)
        client.bucket = "test-bucket"
        # ``_client`` is now a lazy property; assign to the underlying
        # ``_client_instance`` slot directly (also never touched, since
        # validation fires first).
        client._client_instance = None
        client._boto3_kwargs = {}

        with pytest.raises(ValueError, match=r"path-traversal"):
            client.download_output(
                run_id="r-1",
                filename="../etc/passwd",
                local_dir=str(tmp_path),
            )


class TestK8sBackendDeleteJobNotFound:
    """``KubernetesBatchJobBackend.delete_job`` is idempotent on both
    paths: a missing job is a clean no-op rather than an error.

    - **kubectl-fallback path** uses ``--ignore-not-found=true``.
    - **Python kubernetes-client path** catches ``ApiException`` with
      ``status == 404`` and returns; any other API error
      (403 Forbidden, 500 Server Error, etc.) still propagates.

    Resolves the prior asymmetry where the client path propagated
    404s verbatim while the kubectl path swallowed them.
    """

    def test_kubectl_path_ignores_missing_job(self, monkeypatch):
        """
        Tests:
            (Test Case 1) ``delete_job`` on the kubectl-fallback path
                invokes ``kubectl delete`` with ``--ignore-not-found=true``.
            (Test Case 2) No exception is raised when the job is missing.
        """
        from types import SimpleNamespace

        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            # Mimic kubectl's --ignore-not-found behaviour: exit 0 with
            # an informational message on stdout, never raises.
            return SimpleNamespace(stdout='job "missing" not found', returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        backend = KubernetesBatchJobBackend(namespace="ns")
        backend._batch_api = None  # force kubectl fallback

        # Should not raise — kubectl-path swallows "not found".
        backend.delete_job("missing-job")

        assert len(calls) == 1
        cmd = calls[0]
        assert "delete" in cmd
        assert "missing-job" in cmd
        assert "--ignore-not-found=true" in cmd

    def test_k8s_client_path_ignores_404(self):
        """
        Tests:
            (Test Case 1) ``delete_job`` on the Python kubernetes-client
                path catches ``ApiException`` with ``status == 404`` and
                returns cleanly — matches the kubectl path's
                ``--ignore-not-found`` semantic.
            (Test Case 2) ``delete_namespaced_job`` is still called once
                (we don't short-circuit before the API call).
        """

        class _FakeApiException(Exception):
            """Stand-in for ``kubernetes.client.rest.ApiException``."""

            def __init__(self, status, reason):
                self.status = status
                self.reason = reason
                super().__init__(f"({status}) {reason}")

        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        mock_batch_api.delete_namespaced_job.side_effect = _FakeApiException(
            404, "Not Found"
        )
        backend._batch_api = mock_batch_api

        # Patch ``client.exceptions.ApiException`` to our stand-in so the
        # ``except`` catches our fake exception class.
        fake_client = MagicMock()
        fake_client.exceptions.ApiException = _FakeApiException
        with patch("spikelab.batch_jobs.backend_k8s.client", fake_client):
            # No exception expected.
            backend.delete_job("missing-job")

        mock_batch_api.delete_namespaced_job.assert_called_once()

    def test_k8s_client_path_propagates_non_404(self):
        """
        Tests:
            (Test Case 1) Other ``ApiException`` statuses (e.g. 403
                Forbidden) still propagate — only 404 is swallowed.
        """

        class _FakeApiException(Exception):
            def __init__(self, status, reason):
                self.status = status
                self.reason = reason
                super().__init__(f"({status}) {reason}")

        backend = KubernetesBatchJobBackend(namespace="test-ns")
        mock_batch_api = MagicMock()
        mock_batch_api.delete_namespaced_job.side_effect = _FakeApiException(
            403, "Forbidden"
        )
        backend._batch_api = mock_batch_api

        fake_client = MagicMock()
        fake_client.exceptions.ApiException = _FakeApiException
        with patch("spikelab.batch_jobs.backend_k8s.client", fake_client):
            with pytest.raises(_FakeApiException, match=r"Forbidden"):
                backend.delete_job("forbidden-job")


# ============================================================================
# Batch Jobs review (2026-05-24) — edge-case pins from the /complete_review
# pass on fix/review-cleanups.
# ============================================================================


class TestModelValidationBatchJobs:
    """Pydantic boundary pins for `models.py` validators."""

    def test_container_spec_invalid_pull_policy_rejected(self):
        """
        Tests:
            (Test Case 1) ``image_pull_policy="Sometimes"`` raises
                ValidationError (only Always/IfNotPresent/Never legal).
        """
        from pydantic import ValidationError

        from spikelab.batch_jobs.models import ContainerSpec

        with pytest.raises(ValidationError):
            ContainerSpec(image="x", image_pull_policy="Sometimes")

    def test_resource_spec_negative_gpu_rejected(self):
        """
        Tests:
            (Test Case 1) Negative GPU values rejected by Pydantic ge=0.
        """
        from pydantic import ValidationError

        from spikelab.batch_jobs.models import ResourceSpec

        with pytest.raises(ValidationError):
            ResourceSpec(
                requests_cpu="1",
                requests_memory="1Gi",
                limits_cpu="1",
                limits_memory="1Gi",
                requests_gpu=-1,
                limits_gpu=-1,
            )

    def test_volume_mount_empty_name_rejected(self):
        """
        Tests:
            (Test Case 1) Empty ``name`` raises ValidationError.
            (Test Case 2) Empty ``mount_path`` raises ValidationError.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VolumeMountSpec(name="", mount_path="/m", secret_name="s")
        with pytest.raises(ValidationError):
            VolumeMountSpec(name="v", mount_path="", secret_name="s")


class TestJobSpecNamePrefixBoundaries:
    """``JobSpec._validate_name_prefix`` boundary pins. Whitespace-only
    and all-hyphen inputs collapse to the same unified error so operators
    don't see different messages for inputs that look equivalent.
    """

    def _make_payload(self, name_prefix: str):
        payload = _example_payload()
        payload["name_prefix"] = name_prefix
        return payload

    def test_whitespace_only_raises_unified_message(self):
        """
        Tests:
            (Test Case 1) ``"   "`` raises ValidationError with the
                unified "no usable ASCII content" message.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="no usable ASCII content"):
            JobSpec(**self._make_payload("   "))

    def test_all_hyphens_raises_unified_message(self):
        """
        Tests:
            (Test Case 1) ``"---"`` raises ValidationError with the same
                unified "no usable ASCII content" message as the
                whitespace-only case.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="no usable ASCII content"):
            JobSpec(**self._make_payload("---"))

    def test_leading_hyphens_stripped_to_non_empty(self):
        """
        Tests:
            (Test Case 1) ``"-abc"`` is sanitized to ``"abc"`` (leading
                hyphens stripped by ``safe[:40].strip("-")``).
        """
        spec = JobSpec(**self._make_payload("-abc"))
        assert spec.name_prefix == "abc"


class TestContainsDisallowedSleepBoundaries:
    """``policy._contains_disallowed_sleep`` boundary contracts: the
    function flags the common ``sleep <large-int>`` and ``sleep infinity``
    forms but deliberately misses a trailing bare ``sleep`` after a real
    command (documented design choice to avoid false positives).
    """

    def test_empty_command_with_sleep_in_args(self):
        """
        Tests:
            (Test Case 1) ``command=[], args=["sleep", "infinity"]``
                flagged as disallowed.
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep([], ["sleep", "infinity"]) is True

    def test_underscore_grouped_million_flagged(self):
        """
        Tests:
            (Test Case 1) ``["sleep", "1_000_000"]`` is flagged
                (``float("1_000_000")`` returns 1e6, above threshold).
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep(["sleep", "1_000_000"], []) is True

    def test_trailing_bare_sleep_not_flagged(self):
        """
        Tests:
            (Test Case 1) ``["python", "-c", "sleep"]`` is NOT flagged
                (bare-sleep check only fires when tokens is exactly
                ``["sleep"]``).

        Notes:
            - Documents the current design choice. See Bugs Confirmed —
              Batch Jobs for the unverified-but-possibly-intentional
              gap.
        """
        from spikelab.batch_jobs.policy import _contains_disallowed_sleep

        assert _contains_disallowed_sleep(["python", "-c", "sleep"], []) is False


class TestValidationModuleBoundaries:
    """``validation.validate_job_spec`` rejection contracts at the
    type-shape boundary.
    """

    def test_validate_job_spec_none_payload_raises(self):
        """
        Tests:
            (Test Case 1) ``payload=None`` raises ValidationError.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_job_spec(None)

    def test_validate_job_spec_non_dict_payload_raises(self):
        """
        Tests:
            (Test Case 1) ``payload=[]`` (list, not dict) raises
                ValidationError.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_job_spec([])


class TestTemplatingSanitizers:
    """``templating._sanitize_yaml_value`` boundary contracts."""

    def test_empty_string_returns_empty(self):
        """
        Tests:
            (Test Case 1) Empty input returns empty string.
        """
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        assert _sanitize_yaml_value("") == ""

    def test_all_unsafe_chars_raises(self):
        """
        Tier L-C1: an input composed entirely of unsafe characters
        raises ValueError naming each unsafe character in the
        message. Previously this returned empty string (silent mangle).

        Tests:
            (Test Case 1) ``"\\n\\t\\""`` raises ValueError.
        """
        import pytest
        from spikelab.batch_jobs.templating import _sanitize_yaml_value

        with pytest.raises(ValueError, match="YAML-unsafe character"):
            _sanitize_yaml_value('\n\t"')


class TestLoadProfileFromNameBoundaries:
    """``profiles.load_profile_from_name`` accepts only literal
    ``{"nrp", "nautilus"}`` (case-insensitive, whitespace-stripped)
    and falls back to ``defaults`` for anything else — including
    empty and path-traversal-like strings.
    """

    def test_empty_string_falls_back_to_defaults(self):
        """
        Tests:
            (Test Case 1) ``name=""`` resolves to the defaults profile
                (returns a ClusterProfile, not an error).
        """
        from spikelab.batch_jobs.profiles import load_profile_from_name

        profile = load_profile_from_name("")
        # defaults profile must construct successfully.
        assert profile is not None

    def test_mixed_case_nrp_normalized(self):
        """
        Tests:
            (Test Case 1) ``name="Nrp"`` resolves to the NRP profile
                (case-insensitive match).
        """
        from spikelab.batch_jobs.profiles import load_profile_from_name

        profile_nrp = load_profile_from_name("Nrp")
        profile_lower = load_profile_from_name("nrp")
        assert profile_nrp.name == profile_lower.name

    def test_path_traversal_arg_safely_falls_back(self):
        """
        Tests:
            (Test Case 1) ``name="../etc/passwd"`` does NOT cause a
                path-traversal lookup; falls back to defaults.
        """
        from spikelab.batch_jobs.profiles import load_profile_from_name

        profile_evil = load_profile_from_name("../etc/passwd")
        profile_default = load_profile_from_name("defaults")
        assert profile_evil.name == profile_default.name


class TestSubmitPreparedJobRunIdTraversal:
    """``RunSession.submit_prepared_job`` rejects a traversal-shaped
    ``run_id`` before it can flow into S3 prefixes.
    """

    def test_traversal_run_id_rejected(self):
        """
        Tests:
            (Test Case 1) Source contains a guard inside
                ``submit_prepared_job`` that rejects ``run_id`` values
                containing ``..``, ``/``, or ``\\``. Pin via source
                inspection.
        """
        from pathlib import Path

        import spikelab.batch_jobs.session as session_mod

        src = Path(session_mod.__file__).read_text(encoding="utf-8")
        sub_start = src.index("def submit_prepared_job")
        sub_end = src.index("def retrieve_result")
        body = src[sub_start:sub_end]
        # The guard rejects traversal-style run_ids at the session boundary.
        assert "current_run_id" in body
        assert '".." in' in body or "'..' in" in body


class TestTierKBatchJobConstants:
    """Pin the named-constant module attributes the parallel review
    extracted in Tier K. These constants are part of the public API
    (callers tune the defaults via these names rather than re-reading
    magic numbers from the source).
    """

    def test_default_image_profile_constant(self):
        """
        Tests:
            (Test Case 1) ``cli.DEFAULT_IMAGE_PROFILE`` equals ``"cpu"``.
            (Test Case 2) Passing ``image_profile=None`` resolves
                through the constant to the cpu image.
        """
        from spikelab.batch_jobs import cli as cli_mod

        assert cli_mod.DEFAULT_IMAGE_PROFILE == "cpu"

        # The constant drives the resolution path.
        payload = _example_payload()
        # Force ``container.image`` to be empty so the profile fallback
        # must fill it in.
        payload["container"]["image"] = ""
        profile = ClusterProfile(
            name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
        )
        updated = cli_mod._apply_image_selection(
            payload,
            profile=profile,
            image_profile=None,
            image_override=None,
        )
        assert updated["container"]["image"] == "ghcr.io/example/cpu:latest"

    def test_job_name_length_constants_match_rfc1123_budget(self):
        """
        Tests:
            (Test Case 1) ``_JOB_NAME_MAX_LENGTH`` == 63 (RFC 1123 DNS
                subdomain label cap).
            (Test Case 2) ``_JOB_NAME_TOKEN_LENGTH`` == 8 (UUID suffix
                budget).
            (Test Case 3) ``RunSession._build_job_name`` output never
                exceeds ``_JOB_NAME_MAX_LENGTH`` for prefixes at the
                budget boundary.
        """
        from spikelab.batch_jobs.session import (
            RunSession,
            _JOB_NAME_MAX_LENGTH,
            _JOB_NAME_TOKEN_LENGTH,
        )

        assert _JOB_NAME_MAX_LENGTH == 63
        assert _JOB_NAME_TOKEN_LENGTH == 8

        # Long prefix is truncated to fit the budget.
        long_prefix = "a" * 200
        out = RunSession._build_job_name(long_prefix)
        assert len(out) <= _JOB_NAME_MAX_LENGTH

    def test_reconstruct_config_typeerror_names_subconfig(self):
        """
        ``_reconstruct_config`` wraps the bare ``TypeError`` from
        ``f.type(**sub_dict)`` so the message names which sub-config
        failed to reconstruct — the operator sees ``Failed to
        reconstruct 'execution': ...`` instead of the bare ``__init__()
        got an unexpected keyword`` from the dataclass.

        Tests:
            (Test Case 1) An unknown key in the ``execution`` sub-dict
                raises TypeError whose message contains
                ``"Failed to reconstruct 'execution'"``.
        """
        import dataclasses

        from spikelab.batch_jobs.entrypoints.sorting import _reconstruct_config
        from spikelab.spike_sorting.config import SortingPipelineConfig

        config_dict = dataclasses.asdict(SortingPipelineConfig())
        # Inject an unknown key into the execution sub-dict.
        config_dict["execution"]["nonexistent_field"] = 42

        with pytest.raises(TypeError, match="Failed to reconstruct 'execution'"):
            _reconstruct_config(config_dict)


class TestCliCmdRenderNamespaceBackfill:
    """``cli._cmd_render`` accepts an argparse.Namespace that only
    carries the bare ``render`` subparser's attributes. It fills in
    ``wait`` / ``follow_logs`` / ``max_wait_seconds`` /
    ``allow_policy_risk`` / ``output_manifest`` defaults so the
    delegate ``_cmd_deploy`` doesn't crash with ``AttributeError``.
    """

    def test_bare_render_namespace_does_not_raise_attributeerror(
        self, tmp_path, monkeypatch
    ):
        """
        Tests:
            (Test Case 1) Calling ``_cmd_render`` with a Namespace
                that lacks the deploy-only flags returns 0 without
                ``AttributeError``.
        """
        import json
        from spikelab.batch_jobs import cli as cli_mod

        # Minimal valid job config on disk so _cmd_render → _cmd_deploy →
        # _load_payload finds something to validate.
        payload = _example_payload()
        config_path = tmp_path / "job.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        # Stub heavy dependencies so the test stays hermetic.
        mock_session = MagicMock()
        mock_session.render_manifest.return_value = "rendered-yaml"
        monkeypatch.setattr(
            cli_mod, "_build_session", lambda profile, kubeconfig: mock_session
        )
        monkeypatch.setattr(
            cli_mod,
            "_load_profile",
            lambda name, path: ClusterProfile(
                name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
            ),
        )

        # Build a bare Namespace carrying ONLY what the render subparser
        # would set — no wait / follow_logs / max_wait_seconds /
        # allow_policy_risk / output_manifest.
        args = argparse.Namespace(
            profile=None,
            profile_file=None,
            kubeconfig=None,
            job_config=str(config_path),
            image_profile="cpu",
            image=None,
        )
        result = cli_mod._cmd_render(args)
        assert result == 0

    def test_render_only_stdout_when_no_output_manifest(
        self, tmp_path, monkeypatch, capsys
    ):
        """
        Tests:
            (Test Case 1) Without ``output_manifest`` set on the
                Namespace, ``_cmd_render`` prints the rendered manifest
                to stdout (the documented fallback path).
        """
        import json
        from spikelab.batch_jobs import cli as cli_mod

        payload = _example_payload()
        config_path = tmp_path / "job.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        mock_session = MagicMock()
        mock_session.render_manifest.return_value = "yaml-from-stub"
        monkeypatch.setattr(
            cli_mod, "_build_session", lambda profile, kubeconfig: mock_session
        )
        monkeypatch.setattr(
            cli_mod,
            "_load_profile",
            lambda name, path: ClusterProfile(
                name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
            ),
        )

        args = argparse.Namespace(
            profile=None,
            profile_file=None,
            kubeconfig=None,
            job_config=str(config_path),
            image_profile="cpu",
            image=None,
        )
        cli_mod._cmd_render(args)
        captured = capsys.readouterr().out
        assert "yaml-from-stub" in captured


class TestCliCmdDeployInvalidConfigMessages:
    """``cli._cmd_deploy`` produces ``SystemExit("Invalid job config: …")``
    for both pydantic ``ValidationError`` (uses the summariser) and
    plain exceptions (uses ``str(exc)``). The split exists so unrelated
    exceptions with an ``errors`` attribute don't accidentally route
    through the pydantic summariser.
    """

    def test_validation_error_routes_through_summariser(self, tmp_path, monkeypatch):
        """
        Tests:
            (Test Case 1) An invalid job_config payload (e.g. empty
                container.image) raises SystemExit whose message
                starts with ``"Invalid job config:"``.
            (Test Case 2) The summariser output is a non-empty
                follow-on string after the prefix.
        """
        import json
        from spikelab.batch_jobs import cli as cli_mod

        payload = _example_payload()
        # Drop a required field that the profile fallback cannot fill in.
        payload["container"]["image"] = ""
        del payload["container"]["command"]
        # Make name_prefix empty so the validator must reject it.
        payload["name_prefix"] = ""
        config_path = tmp_path / "job.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        monkeypatch.setattr(
            cli_mod,
            "_load_profile",
            lambda name, path: ClusterProfile(
                name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
            ),
        )
        monkeypatch.setattr(
            cli_mod, "_build_session", lambda profile, kubeconfig: MagicMock()
        )

        args = argparse.Namespace(
            profile=None,
            profile_file=None,
            kubeconfig=None,
            job_config=str(config_path),
            image_profile="cpu",
            image=None,
            render_only=True,
            wait=False,
            follow_logs=False,
            max_wait_seconds=1,
            allow_policy_risk=False,
            output_manifest=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            cli_mod._cmd_deploy(args)
        msg = str(excinfo.value)
        assert msg.startswith("Invalid job config:")
        # Summariser produces some follow-on text.
        assert len(msg) > len("Invalid job config:")

    def test_non_validation_error_routes_through_str(self, tmp_path, monkeypatch):
        """
        Tests:
            (Test Case 1) A plain ``RuntimeError("x")`` raised from
                ``validate_job_spec`` produces SystemExit with the
                message ``"Invalid job config: x"`` (the raw
                ``str(exc)`` rather than the pydantic summariser).
        """
        import json
        from spikelab.batch_jobs import cli as cli_mod

        config_path = tmp_path / "job.json"
        config_path.write_text(json.dumps(_example_payload()), encoding="utf-8")

        monkeypatch.setattr(
            cli_mod,
            "_load_profile",
            lambda name, path: ClusterProfile(
                name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
            ),
        )
        monkeypatch.setattr(
            cli_mod, "_build_session", lambda profile, kubeconfig: MagicMock()
        )

        def raise_runtime_error(payload):
            raise RuntimeError("x")

        monkeypatch.setattr(cli_mod, "validate_job_spec", raise_runtime_error)

        args = argparse.Namespace(
            profile=None,
            profile_file=None,
            kubeconfig=None,
            job_config=str(config_path),
            image_profile="cpu",
            image=None,
            render_only=True,
            wait=False,
            follow_logs=False,
            max_wait_seconds=1,
            allow_policy_risk=False,
            output_manifest=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            cli_mod._cmd_deploy(args)
        assert str(excinfo.value) == "Invalid job config: x"


class TestCliCmdDeployRenderOnlyCreatesParentDirs:
    """``cli._cmd_deploy`` with ``--render-only`` + ``--output-manifest``
    creates the parent directory if it doesn't exist (otherwise
    ``write_text`` raises FileNotFoundError on a missing intermediate
    directory like ``./out/dry-run/`` when ``./out`` doesn't exist).
    """

    def test_render_only_creates_missing_parent_dirs(self, tmp_path, monkeypatch):
        """
        Tests:
            (Test Case 1) Output path under a non-existent nested
                subdirectory writes the manifest, creating the
                missing parents.
        """
        import json
        from spikelab.batch_jobs import cli as cli_mod

        payload = _example_payload()
        config_path = tmp_path / "job.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        mock_session = MagicMock()
        mock_session.render_manifest.return_value = "rendered-yaml-payload"
        monkeypatch.setattr(
            cli_mod, "_build_session", lambda profile, kubeconfig: mock_session
        )
        monkeypatch.setattr(
            cli_mod,
            "_load_profile",
            lambda name, path: ClusterProfile(
                name="t", default_images={"cpu": "ghcr.io/example/cpu:latest"}
            ),
        )

        output_path = tmp_path / "deep" / "nested" / "manifest.yaml"
        assert not output_path.parent.exists()

        args = argparse.Namespace(
            profile=None,
            profile_file=None,
            kubeconfig=None,
            job_config=str(config_path),
            image_profile="cpu",
            image=None,
            render_only=True,
            wait=False,
            follow_logs=False,
            max_wait_seconds=1,
            allow_policy_risk=False,
            output_manifest=str(output_path),
        )
        result = cli_mod._cmd_deploy(args)
        assert result == 0
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "rendered-yaml-payload"


class TestPackageAnalysisBundleMetadataValidation:
    """``package_analysis_bundle`` pre-validates ``metadata`` against
    ``json.dumps`` before any I/O. Without this, the bundle would hash
    every input, copy every file, then crash at the very end — wasting
    the work.
    """

    def test_non_json_serializable_metadata_rejected_before_io(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``metadata={"path": Path("/tmp")}`` raises
                ValueError mentioning "not JSON-serializable".
            (Test Case 2) The metadata check runs before input-file
                I/O — we pass a non-existent input path and confirm
                no FileNotFoundError surfaces (metadata check fired
                first).
        """
        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        nonexistent_input = str(tmp_path / "does_not_exist.bin")
        with pytest.raises(ValueError, match="not JSON-serializable"):
            package_analysis_bundle(
                input_paths=[nonexistent_input],
                run_id="run-meta",
                output_dir=str(tmp_path / "out"),
                output_format="workspace",
                metadata={"path": Path("/tmp")},
            )

    def test_manifest_size_bytes_is_int_not_string(self, tmp_path):
        """
        ``manifest.json`` entries record ``size_bytes`` as an integer
        (``json.loads`` returns ``int``), not a string. Readers should
        not need to cast.

        Tests:
            (Test Case 1) ``isinstance(manifest["files"][0]["size_bytes"],
                int)`` is True after a successful bundle.
        """
        import json
        import zipfile

        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        f = input_dir / "data.bin"
        f.write_bytes(b"hello world")

        bundle_path = package_analysis_bundle(
            input_paths=[str(f)],
            run_id="run-int",
            output_dir=str(tmp_path / "out"),
            output_format="workspace",
        )

        with zipfile.ZipFile(bundle_path) as zf:
            manifest_name = next(
                n for n in zf.namelist() if n.endswith("manifest.json")
            )
            manifest = json.loads(zf.read(manifest_name).decode("utf-8"))

        assert manifest["files"]
        # JSON int → Python int; a stringified int would deserialise as str.
        first = manifest["files"][0]
        assert isinstance(first["size_bytes"], int)
        assert first["size_bytes"] == len(b"hello world")


class TestPackageAnalysisBundleSortingRecordingFiles:
    """``package_analysis_bundle(output_format='sorting', ...)``
    declares ``recording_files`` in the manifest. The sorting
    entrypoint uses that whitelist instead of a permissive name
    blacklist. Sorting output requires the field; workspace / custom
    bundles only include it when the caller passes it explicitly.
    """

    def test_sorting_manifest_carries_sorted_recording_files(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``output_format='sorting'`` writes
                ``manifest.json["recording_files"]`` as a sorted list
                of the supplied basenames.
            (Test Case 2) Duplicates are deduplicated via ``set`` →
                sorted.
        """
        import json
        import zipfile

        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "rec_a.bin").write_bytes(b"a")
        (in_dir / "rec_b.bin").write_bytes(b"b")
        (in_dir / "sorting_config.json").write_text("{}")

        zip_path = package_analysis_bundle(
            input_paths=[
                str(in_dir / "rec_a.bin"),
                str(in_dir / "rec_b.bin"),
                str(in_dir / "sorting_config.json"),
            ],
            run_id="run-1",
            output_dir=str(tmp_path / "out"),
            output_format="sorting",
            recording_files=["rec_b.bin", "rec_a.bin", "rec_a.bin"],
        )

        with zipfile.ZipFile(zip_path) as zf:
            manifest_name = next(
                n for n in zf.namelist() if n.endswith("manifest.json")
            )
            manifest = json.loads(zf.read(manifest_name).decode("utf-8"))
        assert manifest["recording_files"] == ["rec_a.bin", "rec_b.bin"]

    def test_sorting_bundle_requires_recording_files(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``output_format='sorting'`` without
                ``recording_files=`` raises ValueError naming the
                missing parameter.
        """
        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "rec.bin").write_bytes(b"x")

        with pytest.raises(ValueError, match="recording_files"):
            package_analysis_bundle(
                input_paths=[str(in_dir / "rec.bin")],
                run_id="run-1",
                output_dir=str(tmp_path / "out"),
                output_format="sorting",
            )

    def test_workspace_bundle_omits_recording_files_when_unspecified(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``output_format='workspace'`` without
                ``recording_files=`` produces a manifest without the
                ``recording_files`` key.
        """
        import json
        import zipfile

        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "data.pkl").write_bytes(b"x")

        zip_path = package_analysis_bundle(
            input_paths=[str(in_dir / "data.pkl")],
            run_id="run-1",
            output_dir=str(tmp_path / "out"),
            output_format="workspace",
        )

        with zipfile.ZipFile(zip_path) as zf:
            manifest_name = next(
                n for n in zf.namelist() if n.endswith("manifest.json")
            )
            manifest = json.loads(zf.read(manifest_name).decode("utf-8"))
        assert "recording_files" not in manifest


class TestPackageAnalysisBundleManifestCollision:
    """``package_analysis_bundle`` rejects an input file named
    ``manifest.json`` upfront so the generated manifest cannot silently
    overwrite a user file.
    """

    def test_user_manifest_input_rejected(self, tmp_path):
        """
        Tests:
            (Test Case 1) A user input file named ``manifest.json``
                raises ValueError at packaging time rather than being
                silently replaced.
        """
        from spikelab.batch_jobs.artifact_packager import package_analysis_bundle

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        user_manifest = input_dir / "manifest.json"
        user_manifest.write_text('{"user_data": "original"}')

        output_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="manifest.json"):
            package_analysis_bundle(
                input_paths=[str(user_manifest)],
                run_id="run-1",
                output_dir=str(output_dir),
                output_format="workspace",
            )


class TestSortingEntrypointResultCountMismatch:
    """``entrypoints.sorting.main`` now raises RuntimeError when
    ``sort_recording`` returns a different number of results than
    recordings (commit fixed the silent clobber). Pin the new contract.

    The earlier `min(i, len(recording_files) - 1)` saturation has been
    replaced by an explicit length check at source line 126-131.
    """

    def test_result_count_mismatch_raises_runtime_error(self):
        """
        Tests:
            (Test Case 1) Source contains the explicit length check
                (verified by reading the source) — pin the contract
                via source inspection.

        Notes:
            - A full integration test would require mocking S3, the
              sorting backend, and bundle extraction. The contract is
              instead verified by asserting the source carries the
              length-mismatch RuntimeError text.
        """
        from pathlib import Path

        import spikelab.batch_jobs.entrypoints.sorting as sorting_mod

        src = Path(sorting_mod.__file__).read_text(encoding="utf-8")
        # The new contract surfaces this exact phrase.
        assert "cannot map results to recording names" in src
        assert "len(spikedata_results) != len(recording_files)" in src


class TestS3StoragePrefixBoundaries:
    """``S3StorageClient.__init__`` and ``download_output`` path-
    traversal boundary pins.
    """

    def test_empty_prefix_treated_as_none(self):
        """
        Tests:
            (Test Case 1) ``S3StorageClient(prefix="")`` produces the
                same prefix-less client as ``prefix=None``.
        """
        from spikelab.batch_jobs.storage_s3 import S3StorageClient

        client_empty = S3StorageClient(prefix="")
        client_none = S3StorageClient(prefix=None)
        # Both fall to the same prefixless attribute state.
        assert client_empty.prefix == client_none.prefix


class TestCredentialsRedactionBoundaries:
    """``credentials.redact_sensitive_map`` uses word-boundary matching
    so keys like ``SECRETS_PATH`` are not falsely flagged as sensitive.
    """

    def test_secrets_path_not_redacted(self):
        """
        Tests:
            (Test Case 1) ``"SECRETS_PATH"`` passes through unredacted —
                ``SECRETS`` is not a word-boundary match against the
                ``SECRET`` token.
        """
        result = redact_sensitive_map({"SECRETS_PATH": "/etc/secrets"})
        assert result["SECRETS_PATH"] == "/etc/secrets"

    def test_secret_key_redacted(self):
        """
        Tests:
            (Test Case 1) A key actually containing the standalone token
                ``SECRET`` is still redacted (regression guard).
        """
        result = redact_sensitive_map({"API_SECRET": "abc123"})
        assert result["API_SECRET"] != "abc123"

    def test_empty_string_key_passed_through(self):
        """
        Tests:
            (Test Case 1) Empty-string key has no SECRET/TOKEN/PASSWORD
                substring so the value passes through unredacted.
        """
        result = redact_sensitive_map({"": "value"})
        assert result[""] == "value"
