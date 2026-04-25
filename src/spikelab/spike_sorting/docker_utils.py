"""Docker image selection utilities for spike sorting.

Provides auto-detection of the host GPU's CUDA driver version and
selects the most compatible pre-built Docker image tag. This ensures
that Docker-based sorting works across different GPU architectures
without manual image selection.
"""

import subprocess
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

# ---------------------------------------------------------------------------
# CUDA driver → maximum supported toolkit version mapping
# See: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# ---------------------------------------------------------------------------
# Each entry: (minimum_driver_version, cuda_toolkit_tag)
# Ordered newest first; the first match wins.
_DRIVER_TO_CUDA: list[tuple[int, str]] = [
    (560, "cu130"),  # Driver 560+ → CUDA 13.0
    (550, "cu126"),  # Driver 550+ → CUDA 12.6
    (545, "cu124"),  # Driver 545+ → CUDA 12.4
    (535, "cu121"),  # Driver 535+ → CUDA 12.1
    (525, "cu118"),  # Driver 525+ → CUDA 11.8
]

# ---------------------------------------------------------------------------
# Pre-built image registry
# Maps (sorter, cuda_tag) → full Docker image name.
# When a cuda_tag has no entry, falls back to the newest available.
# ---------------------------------------------------------------------------
_IMAGE_REGISTRY: dict[str, dict[str, str]] = {
    "kilosort2": {
        # KS2 uses compiled MATLAB Runtime — MW_CUDA_FORWARD_COMPATIBILITY
        # handles GPU compatibility, so one image works for all CUDA versions.
        "default": "spikeinterface/kilosort2-compiled-base:py310-si0.104",
    },
    "kilosort4": {
        "cu130": "spikeinterface/kilosort4-base:py311-si0.104",
        "cu126": "spikeinterface/kilosort4-base:py311-si0.104",
        # CUDA 11.8 would need a separate image with PyTorch+cu118
        # "cu118": "spikeinterface/kilosort4-base:py311-si0.104-cu118",
    },
}


def get_host_cuda_driver_version() -> int | None:
    """Query the host's NVIDIA driver major version.

    Returns:
        version (int or None): Major driver version (e.g. 590),
            or None if nvidia-smi is unavailable.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        # Driver version format: "590.44.01" — take major
        return int(output.split(".")[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return None


def get_host_cuda_tag() -> str | None:
    """Determine the highest CUDA toolkit tag supported by the host driver.

    Returns:
        tag (str or None): CUDA tag (e.g. "cu130"), or None if
            the driver version cannot be determined or is too old.
    """
    driver_ver = get_host_cuda_driver_version()
    if driver_ver is None:
        return None
    for min_driver, tag in _DRIVER_TO_CUDA:
        if driver_ver >= min_driver:
            return tag
    return None


def get_docker_image(sorter: str, cuda_tag: str | None = None) -> str:
    """Select the best Docker image for a sorter and CUDA version.

    Parameters:
        sorter (str): Sorter name (e.g. "kilosort2", "kilosort4").
        cuda_tag (str or None): CUDA toolkit tag (e.g. "cu130").
            If None, auto-detected from the host GPU.

    Returns:
        image (str): Full Docker image name with tag.

    Raises:
        ValueError: If the sorter has no registered images.
        RuntimeError: If no compatible image is found for the
            detected CUDA version.
    """
    if sorter not in _IMAGE_REGISTRY:
        available = ", ".join(sorted(_IMAGE_REGISTRY.keys()))
        raise ValueError(
            f"No Docker images registered for sorter '{sorter}'. "
            f"Available: {available}"
        )

    images = _IMAGE_REGISTRY[sorter]

    # KS2: single image works for all GPUs
    if "default" in images:
        return images["default"]

    # Auto-detect CUDA if not provided
    if cuda_tag is None:
        cuda_tag = get_host_cuda_tag()
        if cuda_tag is None:
            raise RuntimeError(
                "Could not detect CUDA driver version. Ensure nvidia-smi "
                "is available, or pass a specific docker_image to sort_recording()."
            )

    # Exact match
    if cuda_tag in images:
        return images[cuda_tag]

    raise RuntimeError(
        f"No compatible Docker image for '{sorter}' with CUDA {cuda_tag}. "
        f"Available CUDA tags: {list(images.keys())}. "
        f"To build a custom image: edit SpikeLab/docker/{sorter}/Dockerfile "
        f"and change the PyTorch --index-url from "
        f"https://download.pytorch.org/whl/cu126 to "
        f"https://download.pytorch.org/whl/{cuda_tag}, then run: "
        f"docker build -t spikeinterface/{sorter}-base:py311-si0.104-{cuda_tag} "
        f"-f SpikeLab/docker/{sorter}/Dockerfile SpikeLab/docker/{sorter}/ "
        f"— then pass the image via "
        f'use_docker="spikeinterface/{sorter}-base:py311-si0.104-{cuda_tag}". '
        f"Alternatively, run {sorter} locally without Docker."
    )


@contextmanager
def patched_container_client(
    extra_env: Optional[Dict[str, str]] = None,
    mem_limit_frac: Optional[float] = 0.8,
) -> Iterator[None]:
    """Patch SpikeInterface's ``ContainerClient`` for Docker sorter runs.

    Injects extra environment variables and an optional memory cap into
    every Docker container started by SpikeInterface for the duration
    of the context. On exit, the original ``ContainerClient.__init__``
    is restored unconditionally.

    Parameters:
        extra_env (dict[str, str] or None): Environment variables to
            inject into the container (e.g.
            ``{"MW_CUDA_FORWARD_COMPATIBILITY": "1"}`` for Kilosort2).
        mem_limit_frac (float or None): Fraction of host system RAM to
            cap the container's memory at via Docker's ``mem_limit``.
            Defaults to ``0.8`` (80%). Pass ``None`` to disable the
            memory cap. If system RAM cannot be detected, no cap is
            applied.

    Notes:
        - No-op when SpikeInterface is not installed.
        - Only the ``"docker"`` mode is patched; ``"singularity"`` runs
          are unaffected.
    """
    try:
        from spikeinterface.sorters.container_tools import ContainerClient
    except ImportError:
        yield
        return

    from .sorting_utils import get_system_ram_bytes

    _orig_init = ContainerClient.__init__

    def _patched_init(self, mode, container_image, volumes, py_user_base, extra_kwargs):
        if mode == "docker":
            if extra_env:
                extra_kwargs.setdefault("environment", {})
                extra_kwargs["environment"].update(extra_env)
            if mem_limit_frac is not None:
                ram_bytes = get_system_ram_bytes()
                if ram_bytes is not None:
                    extra_kwargs["mem_limit"] = int(ram_bytes * mem_limit_frac)
        _orig_init(self, mode, container_image, volumes, py_user_base, extra_kwargs)

    ContainerClient.__init__ = _patched_init
    try:
        yield
    finally:
        ContainerClient.__init__ = _orig_init
