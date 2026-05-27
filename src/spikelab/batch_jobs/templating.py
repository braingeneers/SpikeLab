"""Template rendering for Kubernetes Job manifests."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
import yaml

from .models import ClusterProfile, JobSpec

# Characters that can break out of a double-quoted YAML scalar. Each
# entry is documented for posterity:
#   ``\n`` / ``\r``  — start a new YAML node; terminate the scalar early.
#   ``\t``           — YAML rejects tabs in indentation, and many parsers
#                       reject them inside flow scalars too.
#   ``"``            — closes the surrounding double-quoted scalar.
#   ``\\``           — would start a YAML escape sequence.
# Characters that are safe inside double-quoted scalars and therefore
# intentionally NOT in this set: ``:``, ``#``, ``{`` / ``}``, ``[`` /
# ``]``, ``&``, ``*``, ``!``, ``|``, ``>``, ``'``, ``%``, ``@``,
# `` ` ``. None of these can terminate a double-quoted scalar.
_YAML_UNSAFE_CHARS = set('\n\r\t"\\')


def _sanitize_yaml_value(value: str, *, field: Optional[str] = None) -> str:
    """Validate that *value* is safe to embed inside a quoted YAML scalar.

    Previously this function silently stripped any character in
    ``_YAML_UNSAFE_CHARS`` from *value* and returned the truncated
    string. That mangled commands and label values without notice —
    e.g. a ``command=['python', '-c "print(\\"x\\")"']`` lost both
    double-quotes and ran ``python -c print(x)`` inside the container,
    silently. Tier L-C1 replaces the silent strip with a fail-fast
    ``ValueError`` naming the offending character(s) and (if known)
    the field that contained them. Callers that legitimately need
    multi-line or quote-containing content should pre-process: base64-
    encode the payload, or write the literal value to a sidecar file
    referenced by path. Returns *value* unchanged on success.

    Parameters:
        value (str): The string to validate.
        field (str | None): Optional dotted-path field name used in
            the error message (e.g. ``"container.env.FOO"``). Helps
            operators locate the offending entry in the job spec.

    Raises:
        ValueError: If *value* contains any character in
            ``_YAML_UNSAFE_CHARS``.
    """
    bad = sorted({ch for ch in value if ch in _YAML_UNSAFE_CHARS})
    if bad:
        bad_repr = ", ".join(repr(c) for c in bad)
        field_str = f" in field {field!r}" if field else ""
        raise ValueError(
            f"YAML-unsafe character(s){field_str}: {bad_repr} in "
            f"value={value!r}. These characters can break out of a "
            "double-quoted YAML scalar and would corrupt the rendered "
            "manifest. Pre-process the value (e.g. base64-encode "
            "multi-line content, or write to a sidecar file referenced "
            "by path) before passing it to the job spec."
        )
    return value


def _sanitize_map(
    mapping: Dict[str, str], *, field: Optional[str] = None
) -> Dict[str, str]:
    """Validate all values in a string->string mapping for YAML embedding.

    Per-entry failures include the dotted-path field name (e.g.
    ``"container.env.FOO"``) when *field* is provided, so the
    ValueError pinpoints the offending entry.
    """
    out: Dict[str, str] = {}
    for k, v in mapping.items():
        sub_field = f"{field}.{k}" if field else k
        out[k] = _sanitize_yaml_value(str(v), field=sub_field)
    return out


def _sanitize_list(items: List[str], *, field: Optional[str] = None) -> List[str]:
    """Validate a list of strings for YAML embedding.

    Per-entry failures include the indexed field name (e.g.
    ``"container.command[2]"``) when *field* is provided.
    """
    out: List[str] = []
    for i, item in enumerate(items):
        sub_field = f"{field}[{i}]" if field else f"[{i}]"
        out.append(_sanitize_yaml_value(str(item), field=sub_field))
    return out


def _template_env() -> Environment:
    templates_dir = files("spikelab.batch_jobs").joinpath("templates")
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


_VOLUME_STRING_FIELDS = (
    "name",
    "mount_path",
    "sub_path",
    "secret_name",
    "pvc_name",
)


def _sanitize_volume_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate string fields on a single volume-mount dict for YAML embedding.

    Mirrors the ``container.env`` / ``container.command`` validation
    pass in ``build_template_context`` but applied to the volume-mount
    payload. A volume name containing ``\\n`` or ``"`` could break
    the rendered YAML structure; this raises ``ValueError`` per-field
    if any unsafe character is present (Tier L-C1).
    """
    cleaned = dict(entry)
    vol_name = cleaned.get("name", "<unnamed>")
    for field in _VOLUME_STRING_FIELDS:
        value = cleaned.get(field)
        if isinstance(value, str):
            cleaned[field] = _sanitize_yaml_value(
                value, field=f"volume[{vol_name}].{field}"
            )
    return cleaned


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
    profile: ClusterProfile,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply profile-driven default volumes and namespace-specific hooks.

    1. Merge ``profile.default_volumes`` (always applied).
    2. If *namespace* matches a key in ``profile.namespace_hooks``, apply
       that hook's ``image_pull_policy``, ``default_command``, and
       ``required_volumes``.

    Returns the updated ``(container, mounts)`` pair. ``affinity`` used
    to be passed through this function unchanged — that parameter was
    dead weight and has been removed; the caller now owns the
    affinity dict directly.
    """
    seen = {_volume_entry_key(item) for item in mounts}
    merged_mounts = list(mounts)

    # Always-on default volumes from profile
    for vol in profile.default_volumes:
        entry = _sanitize_volume_entry(vol.model_dump())
        key = _volume_entry_key(entry)
        if key not in seen:
            merged_mounts.append(entry)
            seen.add(key)

    # Namespace-specific hook
    hook = profile.namespace_hooks.get(namespace)
    if hook is None:
        return container, merged_mounts

    updated_container = dict(container)
    if hook.image_pull_policy:
        updated_container["image_pull_policy"] = hook.image_pull_policy
    if hook.default_command and not updated_container.get("command"):
        updated_container["command"] = hook.default_command

    # Merge hook env_defaults (hook values do not override user-specified keys)
    if hook.env_defaults:
        existing_env = updated_container.get("env", {})
        # Validate hook env_defaults BEFORE the merge — the user's
        # ``container.env`` was already validated at the call site in
        # ``build_template_context``, but hook values bypass that pass.
        # Tier L-C1: ``_sanitize_map`` now raises rather than silently
        # stripping, so an unsafe character in a hook's env_defaults
        # surfaces immediately with the field name
        # ``hooks.<namespace>.env_defaults.<KEY>``.
        merged_env = _sanitize_map(
            dict(hook.env_defaults),
            field=f"hooks.{namespace}.env_defaults",
        )
        merged_env.update(existing_env)  # user keys take precedence
        updated_container["env"] = merged_env

    for vol in hook.required_volumes:
        entry = _sanitize_volume_entry(vol.model_dump())
        key = _volume_entry_key(entry)
        if key not in seen:
            merged_mounts.append(entry)
            seen.add(key)

    return updated_container, merged_mounts


def _build_pod_volumes(mounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate volume mounts by name into pod volume specs.

    First-writer-wins: when the same volume name appears in multiple
    mounts, the first mount's secret_name/pvc_name is kept. Later
    mounts only fill in missing fields (e.g., a mount with secret_name
    followed by one with pvc_name merges both).
    """
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

    # Final validation: every volume must have exactly one source. A K8s
    # `Volume` cannot have both ``secret`` and ``persistentVolumeClaim``
    # populated (they are mutually exclusive sources), and the template
    # would silently drop the pvc via ``{% if secret_name %}{% elif
    # pvc_name %}`` if both were set. Reject the misconfiguration loudly
    # rather than producing an invalid manifest.
    for v in volumes_by_name.values():
        has_secret = bool(v.get("secret_name"))
        has_pvc = bool(v.get("pvc_name"))
        if has_secret and has_pvc:
            raise ValueError(
                f"Volume {v['name']!r} has both secret_name "
                f"({v['secret_name']!r}) and pvc_name ({v['pvc_name']!r}) "
                "after merging. K8s Volume sources are mutually exclusive — "
                "split into two separate volume mounts with distinct names."
            )
        if not has_secret and not has_pvc:
            raise ValueError(
                f"Volume {v['name']!r} has neither secret_name nor pvc_name. "
                "Every K8s Volume must specify exactly one source."
            )
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
    labels = _sanitize_map(labels, field="labels")
    namespace = job_spec.namespace or profile.namespace
    mounts = [
        _sanitize_volume_entry(volume.model_dump()) for volume in job_spec.volumes
    ]
    container = job_spec.container.model_dump()
    container["env"] = _sanitize_map(container.get("env", {}), field="container.env")
    container["command"] = _sanitize_list(
        container.get("command", []), field="container.command"
    )
    container["args"] = _sanitize_list(
        container.get("args", []), field="container.args"
    )
    affinity = profile.affinity
    container, mounts = _apply_namespace_hooks(
        namespace=namespace,
        container=container,
        mounts=mounts,
        profile=profile,
    )
    pod_volumes = _build_pod_volumes(mounts)
    return {
        "job_name": _sanitize_yaml_value(job_name, field="job_name"),
        "namespace": _sanitize_yaml_value(namespace, field="namespace"),
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
        "image_pull_secrets": [
            _sanitize_yaml_value(s, field="image_pull_secrets")
            for s in profile.image_pull_secrets
        ],
    }


def render_job_manifest(context: Dict[str, Any]) -> str:
    """Render a Kubernetes Job manifest as YAML."""
    env = _template_env()
    template = env.get_template("job.yaml.j2")
    return template.render(**context).strip() + "\n"
