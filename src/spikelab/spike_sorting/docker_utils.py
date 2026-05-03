"""Docker image selection utilities for spike sorting.

Provides auto-detection of the host GPU's CUDA driver version and
selects the most compatible pre-built Docker image tag. This ensures
that Docker-based sorting works across different GPU architectures
without manual image selection.
"""

import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

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


def get_local_image_digest(image_tag: str) -> Optional[str]:
    """Return the locally-cached image's digest (``sha256:...``), or None.

    Used by the Markdown sorting report and the optional
    ``ExecutionConfig.docker_image_expected_digest`` mismatch check
    so two sorts months apart can be compared at the bit level rather
    than only by the mutable image tag.

    Tries the python ``docker`` client first (``images.get(...).id``)
    and falls back to ``docker inspect --format={{.Id}}``. Returns
    ``None`` on any failure — caller should treat the absence as
    "digest unknown" rather than a hard error.

    Parameters:
        image_tag (str): Docker image tag, e.g.
            ``"spikeinterface/kilosort4-base:py311-si0.104"``.

    Returns:
        digest (str or None): Image ID string (typically
            ``"sha256:..."``), or ``None`` when the image is not in
            the local cache, the daemon is unreachable, or the
            ``docker`` CLI is missing.
    """
    if not image_tag:
        return None

    try:
        import docker as _docker  # type: ignore[import-not-found]

        try:
            image = _docker.from_env().images.get(image_tag)
            digest = getattr(image, "id", None)
            if isinstance(digest, str) and digest:
                return digest
        except Exception:
            pass
    except ImportError:
        pass

    try:
        out = subprocess.run(
            ["docker", "inspect", "--format={{.Id}}", image_tag],
            check=True,
            timeout=5,
            capture_output=True,
            text=True,
        )
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None
    digest = out.stdout.strip()
    return digest or None


@contextmanager
def patched_container_client(
    extra_env: Optional[Dict[str, str]] = None,
    mem_limit_frac: Optional[float] = 0.8,
) -> Iterator[None]:
    """Patch SpikeInterface's ``ContainerClient`` for Docker sorter runs.

    Injects extra environment variables, an optional memory cap, and
    a host-memory-watchdog kill callback into every Docker container
    started by SpikeInterface for the duration of the context. On
    exit, the original ``ContainerClient.__init__`` is restored
    unconditionally.

    Three layers of host protection are applied to each container:

    1. ``extra_env`` — environment variables (e.g. KS2's
       ``MW_CUDA_FORWARD_COMPATIBILITY``).
    2. ``mem_limit`` — Docker-enforced container memory cap at
       ``mem_limit_frac`` of host RAM.
    3. **Host-memory watchdog kill hook** (new) — when an active
       :class:`spikelab.spike_sorting.guards.HostMemoryWatchdog`
       trips, the container is ``stop()``-and-``kill()``-ed
       alongside any registered subprocesses. Closes the gap on
       Windows-Docker where the WSL2 VM can drag the host into
       thrash even with ``mem_limit`` set.

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
          are unaffected by both the memory cap and the kill hook.
        - The kill hook uses a weak reference to the container so SI's
          normal teardown can garbage-collect it; the registration is
          auto-cleaned via ``weakref.finalize`` when that happens.
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

        # Register a kill callback with the active host-memory
        # watchdog so the container is terminated promptly on host
        # RAM pressure. Uses weakrefs to avoid keeping the container
        # alive past SI's normal teardown.
        if mode == "docker":
            _try_register_container_kill(self)

    ContainerClient.__init__ = _patched_init
    try:
        yield
    finally:
        ContainerClient.__init__ = _orig_init


def _try_register_container_kill(client: Any) -> None:
    """Best-effort: register kill hooks for *client*'s container.

    Two hooks are installed when applicable:

    1. **Host-memory watchdog kill callback** — when an active
       :class:`HostMemoryWatchdog` trips, the container is
       ``stop()``-and-``kill()``-ed alongside any registered
       subprocesses.
    2. **Container-aware log inactivity watchdog** — when both an
       active log path and an active inactivity tolerance are
       published (typically by the backend's ``sort()`` method via
       :func:`set_active_inactivity_timeout_s`), a
       :class:`LogInactivityWatchdog` is started that watches the
       Tee log; on trip it stops the container directly. Closes
       the gap on Docker-backed sorts that hang without consuming
       memory.

    Both hooks are auto-unregistered via ``weakref.finalize`` when
    the container is garbage-collected (i.e. when SI's normal
    teardown releases it).

    Failures during registration are swallowed so a guards-side
    bug never breaks the sort.

    Parameters:
        client: The patched ``ContainerClient`` instance whose
            ``docker_container`` was just created.
    """
    try:
        import weakref

        from .guards import (
            LogInactivityWatchdog,
            get_active_inactivity_timeout_s,
            get_active_log_path,
            get_active_watchdog,
        )

        container = getattr(client, "docker_container", None)
        if container is None:
            return

        # Use a weakref so closures do not keep the container alive
        # once SI releases it. Callbacks no-op gracefully when the
        # referent is already gone.
        container_ref = weakref.ref(container)

        def _kill_container() -> None:
            c = container_ref()
            if c is None:
                return
            try:
                c.stop(timeout=2)
            except Exception as exc:
                print(f"[container kill] container.stop() failed: {exc!r}")
            try:
                # ``stop`` should have done it; ``kill`` is the
                # belt-and-braces path for hung containers that
                # ignored SIGTERM.
                if hasattr(c, "kill"):
                    c.kill()
            except Exception:
                # Container almost certainly already exited; this is
                # the expected path. Swallow.
                pass

        # ---- Hook 1: host-memory watchdog kill callback ----
        watchdog = get_active_watchdog()
        if watchdog is not None:
            watchdog.register_kill_callback(_kill_container)
            weakref.finalize(
                container, watchdog.unregister_kill_callback, _kill_container
            )

        # ---- Hook 2: container-aware inactivity watchdog ----
        log_path = get_active_log_path()
        inactivity_s = get_active_inactivity_timeout_s()
        if log_path is not None and inactivity_s and inactivity_s > 0:
            # Scale the poll interval with the inactivity tolerance so
            # short timeouts (test scenarios; very short recordings)
            # don't pay the full default poll cadence. For typical
            # production timeouts (≥ 600 s), poll_interval stays at
            # the default 5 s.
            poll_interval_s = min(5.0, max(0.1, float(inactivity_s) / 4.0))
            inactivity_wd = LogInactivityWatchdog(
                log_path=log_path,
                popen=None,
                inactivity_s=inactivity_s,
                sorter="docker_container",
                poll_interval_s=poll_interval_s,
                kill_callback=_kill_container,
            )
            try:
                inactivity_wd.__enter__()
            except Exception as exc:
                print(
                    f"[container kill] failed to start container "
                    f"inactivity watchdog: {exc!r}"
                )
            else:
                # Auto-stop the watchdog when the container is GC'd
                # so the polling thread joins and we don't leak it.
                weakref.finalize(
                    container,
                    _safe_exit_inactivity_watchdog,
                    inactivity_wd,
                )
    except Exception as exc:
        print(
            f"[container kill] failed to register kill hooks: {exc!r}; "
            "container will rely on Docker's mem_limit and SI's teardown only."
        )


def _safe_exit_inactivity_watchdog(watchdog: Any) -> None:
    """Best-effort ``__exit__`` for a finalizer-driven cleanup.

    Used by ``weakref.finalize`` to stop a container-aware
    ``LogInactivityWatchdog`` when SI releases the container.
    Failures are swallowed because the finalizer runs from a
    GC-driven context where re-raising would lose the original
    error path.
    """
    try:
        watchdog.__exit__(None, None, None)
    except Exception as exc:
        print(f"[container kill] inactivity watchdog __exit__ failed: {exc!r}")
