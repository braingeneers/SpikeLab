"""Typed models used by the batch job launcher."""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ContainerSpec(BaseModel):
    """Container runtime details for a single-job pod."""

    image: str = Field(min_length=1)
    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"
    command: List[str] = Field(default_factory=list)
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


class ResourceSpec(BaseModel):
    """Resource requests/limits for a job container."""

    requests_cpu: str = "1"
    requests_memory: str = "2Gi"
    limits_cpu: str = "1"
    limits_memory: str = "2Gi"
    requests_gpu: int = Field(default=0, ge=0)
    limits_gpu: int = Field(default=0, ge=0)
    node_selector: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_gpu_pairing(self) -> "ResourceSpec":
        if self.requests_gpu != self.limits_gpu:
            raise ValueError("GPU requests and limits must match")
        return self


class VolumeMountSpec(BaseModel):
    """Pod volume + mount target information."""

    name: str = Field(min_length=1)
    mount_path: str = Field(min_length=1)
    sub_path: Optional[str] = None
    secret_name: Optional[str] = None
    pvc_name: Optional[str] = None
    read_only: bool = True

    @model_validator(mode="after")
    def _validate_source(self) -> "VolumeMountSpec":
        if not self.secret_name and not self.pvc_name:
            raise ValueError("Volume must reference either secret_name or pvc_name")
        return self


class NamespaceHookSpec(BaseModel):
    """Per-namespace overrides applied when a job targets a specific namespace."""

    image_pull_policy: Optional[Literal["Always", "IfNotPresent", "Never"]] = None
    default_command: List[str] = Field(default_factory=list)
    required_volumes: List[VolumeMountSpec] = Field(default_factory=list)
    env_defaults: Dict[str, str] = Field(default_factory=dict)


class StoragePathTemplates(BaseModel):
    """Python format-string templates for S3 artifact paths.

    Available placeholders: ``{prefix}``, ``{run_id}``, ``{filename}``.
    """

    inputs: str = "{prefix}inputs/{run_id}/{filename}"
    outputs: str = "{prefix}outputs/{run_id}/"
    logs: str = "{prefix}logs/{run_id}/"


class PolicyConfig(BaseModel):
    """Configurable thresholds for the cluster policy engine."""

    max_interactive_gpus: int = Field(default=2, ge=0)
    max_runtime_seconds: int = Field(default=1_209_600, ge=1)  # 14 days
    block_sleep_infinity: bool = True
    warn_request_limit_mismatch: bool = True
    sleep_duration_threshold_s: int = Field(
        default=86_400, ge=1
    )  # 24 hours — bare ``sleep <n>`` durations >= this are flagged
    # as idle-placeholders. Separate from ``max_runtime_seconds``
    # because the check targets compute-masquerading sleeps, not
    # the job's overall wall-clock budget.


class JobSpec(BaseModel):
    """High-level description of a Kubernetes batch job.

    Single-container assumption: ``container`` is a single
    :class:`ContainerSpec`, not a list. The rendered ``job.yaml.j2``
    template targets one container per pod (named ``analysis``).
    Multi-container patterns (sidecars for log shipping, init
    containers for fetch) are not supported by the current template.
    """

    name_prefix: str = "analysis-job"
    namespace: str = "default"
    labels: Dict[str, str] = Field(default_factory=dict)
    container: ContainerSpec
    resources: ResourceSpec
    volumes: List[VolumeMountSpec] = Field(default_factory=list)
    #: Kubernetes ``ttlSecondsAfterFinished``: how long the K8s job
    #: object and its pod logs persist after the pod terminates,
    #: before the cluster reaps them. The default 3600 (1 hour)
    #: optimises for cluster cleanliness, not forensics — useful
    #: dials when you need pod logs available post-completion for
    #: debugging:
    #:
    #: - ``21600`` (6h): end-of-day check
    #: - ``86400`` (24h): next-business-day investigation
    #: - ``172800`` (48h): covers a weekend gap
    #: - ``604800`` (1 week): covers a vacation
    #:
    #: Results stored in S3 are retained regardless — this TTL only
    #: affects the K8s-side job object and its pod logs.
    ttl_seconds_after_finished: int = Field(default=3600, ge=0)
    backoff_limit: int = Field(default=0, ge=0)
    active_deadline_seconds: Optional[int] = Field(default=None, ge=1)

    @field_validator("name_prefix")
    @classmethod
    def _validate_name_prefix(cls, value: str) -> str:
        # Unified validation: both whitespace-only and all-hyphen inputs
        # produce a single "no usable ASCII content" message so operators
        # don't see different errors for inputs that look equivalent.
        # Previously ``"   "`` raised "cannot be empty" while ``"---"``
        # raised "empty after ASCII sanitization" — same root cause,
        # different wording, operator confusion.
        stripped = value.strip().lower()
        # Replace any character outside the RFC 1123 ASCII subset with '-'.
        safe = re.sub(r"[^a-z0-9-]", "-", stripped)
        # Collapse runs of hyphens.
        safe = re.sub(r"-+", "-", safe)
        # Truncate, then strip leading/trailing hyphens so the result never
        # ends in '-' after truncation (RFC 1123 violation).
        pre_truncate = safe
        safe = safe[:40].strip("-")
        # Warn when the input was meaningfully truncated. The operator
        # may have expected the full string to survive into the Job
        # name; surfacing the truncation lets them shorten the prefix
        # upstream rather than discovering a mangled name in kubectl.
        if len(pre_truncate) > 40 and safe and safe != pre_truncate:
            warnings.warn(
                f"name_prefix={value!r} truncated to {safe!r} to fit the "
                "40-character RFC 1123 budget. Job names that need to "
                "round-trip the full prefix should pass a shorter "
                "name_prefix upstream.",
                UserWarning,
                stacklevel=2,
            )
        if not safe:
            raise ValueError(
                f"name_prefix={value!r} has no usable ASCII content (after "
                "stripping whitespace and reducing to RFC 1123 characters). "
                "Pass a non-empty alphanumeric prefix."
            )
        return safe


class ClusterProfile(BaseModel):
    """Cluster defaults that can be merged with a JobSpec.

    All organisation-specific configuration (images, secrets, S3 buckets,
    namespace hooks) belongs in profile YAML files, not in Python source.
    """

    name: str
    namespace: str = "default"
    labels: Dict[str, str] = Field(default_factory=dict)
    default_s3_prefix: Optional[str] = None
    affinity: Dict[str, object] = Field(default_factory=dict)
    tolerations: List[Dict[str, object]] = Field(default_factory=list)
    default_secrets_mapping: Dict[str, str] = Field(default_factory=dict)
    default_images: Dict[str, str] = Field(default_factory=dict)
    default_volumes: List[VolumeMountSpec] = Field(default_factory=list)
    namespace_hooks: Dict[str, NamespaceHookSpec] = Field(default_factory=dict)
    storage: StoragePathTemplates = Field(default_factory=StoragePathTemplates)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    endpoint_url: Optional[str] = None
    region_name: Optional[str] = None
    #: Tier L-F7: Kubernetes ``imagePullSecrets`` to attach to the pod
    #: spec. Each entry is the name of a ``kubernetes.io/dockerconfigjson``
    #: secret that already exists in the target namespace. Empty (the
    #: default) means no pull secrets are rendered — appropriate for
    #: public registries and clusters whose default ServiceAccount
    #: already carries the necessary secrets. Cluster-level rather
    #: than per-job because the same registry is typically reused
    #: across every job submitted to the cluster.
    image_pull_secrets: List[str] = Field(default_factory=list)


class SubmitResult(BaseModel):
    """Result returned by job submission methods."""

    job_name: str
    manifest_yaml: str
    run_id: str
    uploaded_input_uri: str
    output_prefix: str
    logs_prefix: str
    job_type: Literal["workspace", "sorting", "prepared"]
