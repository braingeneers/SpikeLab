"""Template rendering for Kubernetes Job manifests."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
import yaml

from .models import ClusterProfile, JobSpec


def _template_env() -> Environment:
    templates_dir = files("spikelab.nrp_jobs").joinpath("templates")
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _volume_entry_key(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(entry.get("name", "")),
        str(entry.get("mount_path", "")),
        str(entry.get("sub_path") or ""),
    )


def _apply_braingeneers_namespace_defaults(
    namespace: str,
    container: Dict[str, Any],
    mounts: List[Dict[str, Any]],
    affinity: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    if namespace != "braingeneers":
        return container, mounts, affinity

    updated_container = dict(container)
    updated_container["image_pull_policy"] = "Always"
    if not updated_container.get("command"):
        updated_container["command"] = ["sh", "-c"]

    required_mounts: List[Dict[str, Any]] = [
        {
            "name": "prp-s3-credentials",
            "mount_path": "/root/.aws/credentials",
            "sub_path": "credentials",
            "secret_name": "prp-s3-credentials",
            "pvc_name": None,
            "read_only": True,
        },
        {
            "name": "prp-s3-credentials",
            "mount_path": "/root/.aws/.s3cfg",
            "sub_path": ".s3cfg",
            "secret_name": "prp-s3-credentials",
            "pvc_name": None,
            "read_only": True,
        },
        {
            "name": "kube-config",
            "mount_path": "/root/.kube",
            "sub_path": None,
            "secret_name": "kube-config",
            "pvc_name": None,
            "read_only": True,
        },
    ]

    seen = {_volume_entry_key(item) for item in mounts}
    merged_mounts = list(mounts)
    for entry in required_mounts:
        if _volume_entry_key(entry) not in seen:
            merged_mounts.append(entry)

    # Keep affinity fully user/profile-driven; do not inject GPU product preferences.
    return updated_container, merged_mounts, affinity


def _build_pod_volumes(mounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    volumes_by_name: Dict[str, Dict[str, Any]] = {}
    for mount in mounts:
        name = mount.get("name")
        if not name:
            continue
        secret_name = mount.get("secret_name")
        pvc_name = mount.get("pvc_name")
        if name not in volumes_by_name:
            volumes_by_name[name] = {
                "name": name,
                "secret_name": secret_name,
                "pvc_name": pvc_name,
            }
            continue
        if not volumes_by_name[name].get("secret_name") and secret_name:
            volumes_by_name[name]["secret_name"] = secret_name
        if not volumes_by_name[name].get("pvc_name") and pvc_name:
            volumes_by_name[name]["pvc_name"] = pvc_name
    return list(volumes_by_name.values())


def build_template_context(
    *,
    job_name: str,
    job_spec: JobSpec,
    profile: ClusterProfile,
    extra_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    labels = dict(profile.labels)
    labels.update(job_spec.labels)
    if extra_labels:
        labels.update(extra_labels)
    namespace = job_spec.namespace or profile.namespace
    mounts = [volume.model_dump() for volume in job_spec.volumes]
    container = job_spec.container.model_dump()
    affinity = profile.affinity
    container, mounts, affinity = _apply_braingeneers_namespace_defaults(
        namespace=namespace,
        container=container,
        mounts=mounts,
        affinity=affinity,
    )
    pod_volumes = _build_pod_volumes(mounts)
    return {
        "job_name": job_name,
        "namespace": namespace,
        "labels": labels,
        "container": container,
        "resources": job_spec.resources.model_dump(),
        "volume_mounts": mounts,
        "pod_volumes": pod_volumes,
        "affinity": affinity,
        "affinity_yaml": (
            yaml.safe_dump(affinity, sort_keys=False).rstrip() if affinity else ""
        ),
        "tolerations": profile.tolerations,
        "tolerations_yaml": (
            yaml.safe_dump(profile.tolerations, sort_keys=False).rstrip()
            if profile.tolerations
            else ""
        ),
        "ttl_seconds_after_finished": job_spec.ttl_seconds_after_finished,
        "backoff_limit": job_spec.backoff_limit,
        "active_deadline_seconds": job_spec.active_deadline_seconds,
    }


def render_job_manifest(context: Dict[str, Any]) -> str:
    """Render a Kubernetes Job manifest as YAML."""
    env = _template_env()
    template = env.get_template("job.yaml.j2")
    return template.render(**context).strip() + "\n"
