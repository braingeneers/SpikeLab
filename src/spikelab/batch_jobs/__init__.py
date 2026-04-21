"""Kubernetes batch-job launching helpers for SpikeLab.

Requires the ``batch-jobs`` optional dependency group::

    pip install spikelab[batch-jobs]
"""


def __getattr__(name):
    """Lazy-import public symbols so the package is importable without extras."""
    _public = {
        "ClusterProfile",
        "ContainerSpec",
        "JobSpec",
        "ResourceSpec",
        "RunConfig",
        "VolumeMountSpec",
        "RunSession",
        "load_cluster_profile",
        "load_profile_from_name",
    }
    if name in _public:
        try:
            if name in {
                "ClusterProfile",
                "ContainerSpec",
                "JobSpec",
                "ResourceSpec",
                "RunConfig",
                "VolumeMountSpec",
            }:
                from .models import (
                    ClusterProfile,
                    ContainerSpec,
                    JobSpec,
                    ResourceSpec,
                    RunConfig,
                    VolumeMountSpec,
                )

                return locals()[name]
            if name in {"load_cluster_profile", "load_profile_from_name"}:
                from .profiles import load_cluster_profile, load_profile_from_name

                return locals()[name]
            if name == "RunSession":
                from .session import RunSession

                return RunSession
        except ImportError as exc:
            raise ImportError(
                f"Cannot import '{name}' — install the nrp-jobs extra: "
                "pip install spikelab[batch-jobs]"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClusterProfile",
    "ContainerSpec",
    "JobSpec",
    "ResourceSpec",
    "RunConfig",
    "VolumeMountSpec",
    "RunSession",
    "load_cluster_profile",
    "load_profile_from_name",
]
