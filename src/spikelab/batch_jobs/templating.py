"""Template rendering for Kubernetes Job manifests."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
import yaml

from .models import ClusterProfile, JobSpec

_YAML_UNSAFE_CHARS = set('\n\r\t"\\')


def _sanitize_yaml_value(value: str) -> str:
    """Strip characters that could break out of a quoted YAML value."""
    return "".join(ch for ch in value if ch not in _YAML_UNSAFE_CHARS)


def _sanitize_map(mapping: Dict[str, str]) -> Dict[str, str]:
    """Sanitize all values in a string->string mapping for YAML embedding."""
    return {k: _sanitize_yaml_value(str(v)) for k, v in mapping.items()}


def _sanitize_list(items: List[str]) -> List[str]:
    """Sanitize a list of strings for YAML embedding."""
    return [_sanitize_yaml_value(str(item)) for item in items]


def _template_env() -> Environment:
    templates_dir = files("spikelab.batch_jobs").joinpath("templates")
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


def _apply_namespace_hooks(
    namespace: str,
    container: Dict[str, Any],
    mounts: List[Dict[str, Any]],
    affinity: Dict[str, Any],
    profile: ClusterProfile,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply profile-driven default volumes and namespace-specific hooks.

    1. Merge ``profile.default_volumes`` (always applied).
    2. If *namespace* matches a key in ``profile.namespace_hooks``, apply
       that hook's ``image_pull_policy``, ``default_command``, and
       ``required_volumes``.
    """
    seen = {_volume_entry_key(item) for item in mounts}
    merged_mounts = list(mounts)

    # Always-on default volumes from profile
    for vol in profile.default_volumes:
        entry = vol.model_dump()
        key = _volume_entry_key(entry)
        if key not in seen:
            merged_mounts.append(entry)
            seen.add(key)

    # Namespace-specific hook
    hook = profile.namespace_hooks.get(namespace)
    if hook is None:
        return container, merged_mounts, affinity

    updated_container = dict(container)
    if hook.image_pull_policy:
        updated_container["image_pull_policy"] = hook.image_pull_policy
    if hook.default_command and not updated_container.get("command"):
        updated_container["command"] = hook.default_command

    for vol in hook.required_volumes:
        entry = vol.model_dump()
        key = _volume_entry_key(entry)
        if key not in seen:
            merged_mounts.append(entry)
            seen.add(key)

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
    labels = _sanitize_map(labels)
    namespace = job_spec.namespace or profile.namespace
    mounts = [volume.model_dump() for volume in job_spec.volumes]
    container = job_spec.container.model_dump()
    container["env"] = _sanitize_map(container.get("env", {}))
    container["command"] = _sanitize_list(container.get("command", []))
    container["args"] = _sanitize_list(container.get("args", []))
    affinity = profile.affinity
    container, mounts, affinity = _apply_namespace_hooks(
        namespace=namespace,
        container=container,
        mounts=mounts,
        affinity=affinity,
        profile=profile,
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
