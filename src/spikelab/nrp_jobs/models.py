"""Typed models used by the NRP job launcher."""

from __future__ import annotations

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


class JobSpec(BaseModel):
    """High-level description of a Kubernetes batch job."""

    name_prefix: str = "analysis-job"
    namespace: str = "default"
    labels: Dict[str, str] = Field(default_factory=dict)
    container: ContainerSpec
    resources: ResourceSpec
    volumes: List[VolumeMountSpec] = Field(default_factory=list)
    ttl_seconds_after_finished: int = Field(default=3600, ge=0)
    backoff_limit: int = Field(default=0, ge=0)
    active_deadline_seconds: Optional[int] = Field(default=None, ge=1)

    @field_validator("name_prefix")
    @classmethod
    def _validate_name_prefix(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("name_prefix cannot be empty")
        safe = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in value)
        return safe.strip("-")[:40] or "analysis-job"


class ClusterProfile(BaseModel):
    """Cluster defaults that can be merged with a JobSpec."""

    name: str
    namespace: str = "default"
    labels: Dict[str, str] = Field(default_factory=dict)
    default_s3_prefix: Optional[str] = None
    affinity: Dict[str, object] = Field(default_factory=dict)
    tolerations: List[Dict[str, object]] = Field(default_factory=list)
    default_secrets_mapping: Dict[str, str] = Field(default_factory=dict)
    default_images: Dict[str, str] = Field(
        default_factory=lambda: {
            "cpu": "ghcr.io/braingeneers/spikelab-analysis-base:cpu",
            "gpu": "ghcr.io/braingeneers/spikelab-analysis-base:gpu",
        }
    )
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    endpoint_url: Optional[str] = None
    region_name: Optional[str] = None


class RunConfig(BaseModel):
    """User-facing run config consumed by CLI/session."""

    profile_name: str = "nrp"
    output_format: Literal["pickle", "nwb", "both"] = "pickle"
    input_path: str
    output_prefix: Optional[str] = None
    workspace_id: Optional[str] = None
    namespace: Optional[str] = None
    allow_policy_risk: bool = False
    max_wait_seconds: int = Field(default=3600, ge=1)
    wait_for_completion: bool = False
    follow_logs: bool = False
