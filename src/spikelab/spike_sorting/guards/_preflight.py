"""Pre-loop resource checks for the spike-sorting pipeline.

:func:`run_preflight` inspects the host before any sorter is spawned
and returns a list of :class:`PreflightFinding` records. Each finding
carries a level (``"warn"`` or ``"fail"``) and a human-readable
remediation hint.

The intent is to surface predictable failure causes — disk full, RAM
already exhausted, GPU saturated by another process, ``HDF5_PLUGIN_PATH``
pointing at a missing directory — at the start of the run rather than
after a long sort has already crashed the workstation.

Default behaviour is permissive: every finding is reported but only
``"fail"``-level findings raise. ``ExecutionConfig.preflight_strict``
flips ``"warn"`` findings into ``"fail"`` for stricter deployments.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._exceptions import (
    EnvironmentSortFailure,
    HDF5PluginMissingError,
    ResourceSortFailure,
)


@dataclass
class PreflightFinding:
    """A single resource-check finding from :func:`run_preflight`.

    Parameters:
        level (str): Either ``"warn"`` or ``"fail"``.
        code (str): Short stable identifier (e.g.
            ``"low_disk_inter"``, ``"low_vram"``).
        message (str): One-line description of what was observed.
        remediation (str or None): Suggested action for the operator.
        category (str): One of ``"resource"`` or ``"environment"`` —
            controls which exception subclass is raised when the
            finding is escalated.
    """

    level: str
    code: str
    message: str
    remediation: Optional[str] = None
    category: str = "resource"


_GB = 1024**3


def _disk_free_gb(path: Path) -> Optional[float]:
    """Return free disk space in GB at *path*'s nearest existing parent."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / _GB
    except OSError:
        return None


def _available_ram_gb() -> Optional[float]:
    """Return available host RAM in GB via psutil, or ``None``."""
    try:
        import psutil

        return psutil.virtual_memory().available / _GB
    except ImportError:
        return None


def _free_vram_gb() -> Optional[float]:
    """Return free GPU memory (sum across devices) in GB.

    Tries ``pynvml`` first, falls back to parsing ``nvidia-smi``. Returns
    ``None`` when no GPU/driver is detectable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            free_total = 0
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_total += info.free
            return free_total / _GB
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    free_total_mib = 0.0
    for line in out.strip().splitlines():
        try:
            free_total_mib += float(line.strip())
        except ValueError:
            continue
    return free_total_mib / 1024.0


def _sorter_uses_gpu(config: Any) -> bool:
    """Return True when the configured sorter is GPU-backed.

    KS2 only uses the GPU through Docker (the MATLAB host path is CPU-
    only here); KS4 and RT-Sort always use the GPU.
    """
    name = getattr(config.sorter, "sorter_name", "").lower()
    if name in ("kilosort4", "rt_sort"):
        return True
    if name == "kilosort2" and getattr(config.sorter, "use_docker", False):
        return True
    return False


def _parse_wslconfig_memory_gb(text: str) -> Optional[float]:
    """Parse the ``[wsl2] memory=`` value from a ``.wslconfig`` body.

    Accepts ``memory=8GB``, ``memory=8192MB``, ``memory=8gb`` (any
    case), with or without surrounding whitespace. Returns ``None``
    when the key is absent, malformed, or expressed in an unknown
    unit.

    Parameters:
        text (str): Full text of ``~/.wslconfig``.

    Returns:
        memory_gb (float or None): Configured WSL2 memory ceiling in
            GB. Returns ``None`` when not configured.
    """
    in_wsl2 = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("["):
            in_wsl2 = line.lower().startswith("[wsl2")
            continue
        if not in_wsl2:
            continue
        m = re.match(r"^memory\s*=\s*([\d.]+)\s*([a-zA-Z]+)?\s*$", line)
        if m is None:
            continue
        value = float(m.group(1))
        unit = (m.group(2) or "GB").upper()
        if unit in ("GB", "G"):
            return value
        if unit in ("MB", "M"):
            return value / 1024.0
        if unit in ("KB", "K"):
            return value / (1024.0 * 1024.0)
        return None
    return None


def estimate_rt_sort_intermediate_gb(
    *,
    n_channels: int,
    n_samples: int,
) -> float:
    """Project the on-disk size of RT-Sort's intermediate files in GB.

    RT-Sort writes three large per-recording artefacts under the
    intermediate folder during ``detect_sequences``:

    * scaled traces (float32, 4 bytes/sample)
    * model traces (float16, 2 bytes/sample)
    * model outputs (float16, 2 bytes/sample)

    Total per-sample byte cost is therefore ``4 + 2 + 2 = 8`` bytes
    per channel-sample.

    Parameters:
        n_channels (int): Channel count of the recording.
        n_samples (int): Total samples (per channel) over the
            recording duration.

    Returns:
        gb (float): Projected on-disk footprint in GB.
    """
    bytes_per_channel_sample = 4 + 2 + 2
    total_bytes = float(n_channels) * float(n_samples) * float(bytes_per_channel_sample)
    return total_bytes / _GB


def _rt_sort_disk_finding(
    config: Any,
    recording_files: Sequence[Any],
    intermediate_folders: Sequence[Any],
) -> Optional[PreflightFinding]:
    """Warn when RT-Sort's intermediate-folder footprint will not fit.

    Only fires for the RT-Sort sorter. Skips when channel count or
    sample count cannot be determined from the inputs (the recording
    is loaded lazily by ``load_recording`` later, so at preflight we
    only know the path — to keep the check cheap we accept that the
    estimate is ``None`` for inputs given as paths/strings instead of
    pre-loaded recordings).
    """
    if getattr(config.sorter, "sorter_name", "").lower() != "rt_sort":
        return None
    if not recording_files:
        return None

    # Try to extract channel and sample counts from any pre-loaded
    # recordings. Path-only inputs would require loading the file
    # here, which is too expensive for a preflight; we silently skip
    # those.
    estimates: List[Tuple[Any, float]] = []
    for rec in recording_files:
        try:
            n_ch = int(rec.get_num_channels())
            n_smp = int(rec.get_num_samples())
        except Exception:
            continue
        estimates.append(
            (rec, estimate_rt_sort_intermediate_gb(n_channels=n_ch, n_samples=n_smp))
        )

    if not estimates:
        return None

    # Compare the largest estimate to the smallest free-disk among
    # intermediate folders. If we cannot read free disk (e.g. unusual
    # OS), bail out cleanly.
    largest_gb = max(g for _, g in estimates)
    free_gbs = []
    for folder in intermediate_folders:
        free = _disk_free_gb(Path(folder))
        if free is not None:
            free_gbs.append(free)
    if not free_gbs:
        return None
    smallest_free_gb = min(free_gbs)

    if largest_gb <= smallest_free_gb:
        return None

    return PreflightFinding(
        level="warn",
        code="rt_sort_disk_projection",
        category="resource",
        message=(
            f"RT-Sort: intermediate-folder projection of "
            f"{largest_gb:.1f} GB exceeds free disk "
            f"({smallest_free_gb:.1f} GB) on at least one "
            "intermediate path. RT-Sort writes scaled traces, model "
            "traces, and model outputs to disk during sequence "
            "detection."
        ),
        remediation=(
            "Free disk space, point intermediate_folders at a larger "
            "volume, or shorten the recording window via "
            "RTSortConfig.recording_window_ms."
        ),
    )


def _wslconfig_finding(config: Any) -> Optional[PreflightFinding]:
    """Warn when running Docker on Windows without a sane ``.wslconfig``.

    Docker Desktop on Windows runs in a WSL2 VM whose memory ceiling
    is governed by ``%USERPROFILE%\\.wslconfig`` ([wsl2] memory=...).
    When the file is missing or the limit is unset, the VM can grow
    beyond a safe fraction of physical RAM and drag the host into
    thrash even with a Docker ``mem_limit`` configured. Only relevant
    when:

    * Host platform is Windows.
    * The configured sorter uses Docker
      (KS2/KS4 with ``use_docker=True``).

    Returns ``None`` when neither condition holds, when ``.wslconfig``
    is configured with a sensible memory ceiling, or when host RAM
    cannot be detected.
    """
    if sys.platform != "win32":
        return None
    if not getattr(config.sorter, "use_docker", False):
        return None

    wslconfig = Path(os.path.expanduser("~")) / ".wslconfig"
    if not wslconfig.is_file():
        return PreflightFinding(
            level="warn",
            code="wslconfig_missing",
            category="environment",
            message=(
                "Docker-on-Windows: ~/.wslconfig is missing. The WSL2 "
                "VM hosting Docker has no host-side memory ceiling, so "
                "a runaway sort can take Windows down even with the "
                "container's mem_limit set."
            ),
            remediation=(
                "Create %USERPROFILE%\\.wslconfig with [wsl2] "
                "memory=<N>GB where <N> is roughly 75% of host RAM, "
                "then run `wsl --shutdown` and restart Docker Desktop."
            ),
        )

    try:
        text = wslconfig.read_text(errors="replace")
    except OSError:
        return None

    memory_gb = _parse_wslconfig_memory_gb(text)
    if memory_gb is None:
        return PreflightFinding(
            level="warn",
            code="wslconfig_no_memory",
            category="environment",
            message=(
                "Docker-on-Windows: ~/.wslconfig exists but has no "
                "[wsl2] memory= setting. Without it WSL2 can grow to "
                "consume up to half of host RAM by default."
            ),
            remediation=(
                "Add a [wsl2] section with memory=<N>GB (~75% of host "
                "RAM) to ~/.wslconfig, then run `wsl --shutdown`."
            ),
        )

    # Guard against an over-generous setting relative to host RAM.
    try:
        from ..sorting_utils import get_system_ram_bytes

        host_bytes = get_system_ram_bytes()
    except Exception:
        host_bytes = None
    if host_bytes is not None:
        host_gb = host_bytes / _GB
        if memory_gb > 0.85 * host_gb:
            return PreflightFinding(
                level="warn",
                code="wslconfig_memory_too_high",
                category="environment",
                message=(
                    f"Docker-on-Windows: ~/.wslconfig sets WSL2 "
                    f"memory={memory_gb:.1f} GB on a {host_gb:.1f} GB "
                    "host (>85% of physical RAM). A runaway sort can "
                    "still drag Windows into swap."
                ),
                remediation=(
                    "Lower [wsl2] memory= in ~/.wslconfig to ~75% of "
                    "host RAM, then run `wsl --shutdown` and restart "
                    "Docker Desktop."
                ),
            )

    return None


_KNOWN_RECORDING_EXTENSIONS = (".h5", ".nwb", ".dat", ".raw")


def _validate_recording_inputs(
    recording_files: Sequence[Any],
) -> List[PreflightFinding]:
    """Quick existence + extension checks for path-style recording inputs.

    Catches typos and missing files in microseconds rather than the
    seconds-to-minutes that a full ``load_recording`` failure costs.
    Skips entries that are pre-loaded SpikeInterface ``BaseRecording``
    objects — those have already been validated by the loader.

    Returns a list of findings:

    * ``recording_missing`` (level=fail, environment) — path doesn't
      exist on disk.
    * ``recording_extension_unknown`` (level=warn, environment) — file
      extension is not in the known list.

    Parameters:
        recording_files (sequence): Per-recording inputs. Can mix
            paths (``str`` / ``Path``) and pre-loaded ``BaseRecording``
            objects; only path-style entries are checked.

    Returns:
        findings (list[PreflightFinding]): One finding per problem
            recording. Empty when all recordings exist and have a
            known extension.
    """
    findings: List[PreflightFinding] = []
    for rec in recording_files:
        if not isinstance(rec, (str, Path)):
            # Pre-loaded recording object — skip.
            continue
        p = Path(rec)
        if not p.exists():
            findings.append(
                PreflightFinding(
                    level="fail",
                    code="recording_missing",
                    category="environment",
                    message=(f"Recording {p!s} does not exist on disk."),
                    remediation=(
                        "Verify the path is correct, the file has not "
                        "been moved, and any networked storage is "
                        "mounted."
                    ),
                )
            )
            continue
        if p.is_dir():
            # Directory inputs are concatenated by the loader; no
            # extension check applies.
            continue
        # Get the full multi-suffix tail (e.g. ``.raw.h5``) and check
        # whether *any* extension in it is in the known list.
        suffixes = [s.lower() for s in p.suffixes]
        if not any(s in _KNOWN_RECORDING_EXTENSIONS for s in suffixes):
            ext_str = "".join(p.suffixes) if p.suffixes else "(no extension)"
            findings.append(
                PreflightFinding(
                    level="warn",
                    code="recording_extension_unknown",
                    category="environment",
                    message=(
                        f"Recording {p.name} has unfamiliar extension "
                        f"{ext_str}. Known extensions: "
                        f"{', '.join(_KNOWN_RECORDING_EXTENSIONS)}."
                    ),
                    remediation=(
                        "If the file is genuinely a supported format "
                        "with an unusual extension this warning can be "
                        "ignored; otherwise verify the path."
                    ),
                )
            )
    return findings


_TESTED_SI_VERSION_RANGE = ("0.100.0", "0.110.0")
_TESTED_KILOSORT4_VERSION_RANGE = ("4.0.0", "5.0.0")


def _check_kilosort2_host(config: Any) -> List[PreflightFinding]:
    """Probe local Kilosort2 dependencies (host path, no Docker).

    KS2's host path needs MATLAB on PATH plus a checkout of the
    Kilosort2 sources containing ``master_kilosort.m``. The sources
    location is taken from ``SorterConfig.sorter_path`` when set,
    otherwise from the ``KILOSORT_PATH`` environment variable.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        findings (list[PreflightFinding]): Up to two fail-level
            findings (missing matlab, missing/invalid KILOSORT_PATH).
    """
    findings: List[PreflightFinding] = []

    if shutil.which("matlab") is None:
        findings.append(
            PreflightFinding(
                level="fail",
                code="sorter_dependency_missing",
                category="environment",
                message=(
                    "Kilosort2 (host) requires MATLAB but `matlab` was "
                    "not found on PATH."
                ),
                remediation=(
                    "Install MATLAB and ensure `matlab` resolves on "
                    "PATH, or switch to Kilosort2 Docker via "
                    "SorterConfig(use_docker=True)."
                ),
            )
        )

    ks_path = getattr(config.sorter, "sorter_path", None) or os.environ.get(
        "KILOSORT_PATH"
    )
    if not ks_path:
        findings.append(
            PreflightFinding(
                level="fail",
                code="sorter_dependency_missing",
                category="environment",
                message=(
                    "Kilosort2 (host) requires the Kilosort2 source "
                    "directory but neither SorterConfig.sorter_path "
                    "nor the KILOSORT_PATH environment variable is set."
                ),
                remediation=(
                    "Clone https://github.com/MouseLand/Kilosort and "
                    "set KILOSORT_PATH (or SorterConfig.sorter_path) "
                    "to the directory containing master_kilosort.m."
                ),
            )
        )
    else:
        ks_dir = Path(ks_path)
        master_m = ks_dir / "master_kilosort.m"
        if not ks_dir.is_dir():
            findings.append(
                PreflightFinding(
                    level="fail",
                    code="sorter_dependency_missing",
                    category="environment",
                    message=(
                        f"Kilosort2 sources directory {ks_dir!s} does " "not exist."
                    ),
                    remediation=(
                        "Set KILOSORT_PATH (or SorterConfig.sorter_path) "
                        "to a valid Kilosort2 source directory."
                    ),
                )
            )
        elif not master_m.is_file():
            findings.append(
                PreflightFinding(
                    level="fail",
                    code="sorter_dependency_missing",
                    category="environment",
                    message=(
                        f"Kilosort2 sources directory {ks_dir!s} does "
                        "not contain master_kilosort.m."
                    ),
                    remediation=(
                        "Verify KILOSORT_PATH points to the root of a "
                        "Kilosort2 checkout (the directory holding "
                        "master_kilosort.m)."
                    ),
                )
            )

    return findings


def _check_kilosort4_host(config: Any) -> List[PreflightFinding]:
    """Probe local Kilosort4 dependencies (host path, no Docker).

    Verifies the ``kilosort`` package imports and falls inside the
    SpikeLab-tested major-version window. Out-of-range versions warn
    rather than fail because newer KS4 releases sometimes work without
    incident — the warning just makes the operator aware.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration
            (unused, kept for signature symmetry with the other
            ``_check_*_host`` helpers).

    Returns:
        findings (list[PreflightFinding]): Empty when KS4 imports and
            the version is in-range; otherwise one fail- or warn-level
            finding.
    """
    try:
        import kilosort as _ks4
    except ImportError as exc:
        return [
            PreflightFinding(
                level="fail",
                code="sorter_dependency_missing",
                category="environment",
                message=(
                    f"Kilosort4 (host) requires the `kilosort` Python "
                    f"package but it is not importable: {exc}."
                ),
                remediation=(
                    "Install Kilosort4 in the active environment "
                    "(`pip install kilosort`), or switch to Kilosort4 "
                    "Docker via SorterConfig(use_docker=True)."
                ),
            )
        ]

    version = getattr(_ks4, "__version__", None)
    if version is None:
        return []
    parsed = _parse_version_tuple(version)
    low = _parse_version_tuple(_TESTED_KILOSORT4_VERSION_RANGE[0])
    high = _parse_version_tuple(_TESTED_KILOSORT4_VERSION_RANGE[1])
    if parsed is None or low is None or high is None:
        return []
    if low <= parsed < high:
        return []
    return [
        PreflightFinding(
            level="warn",
            code="kilosort4_version_outside_tested_range",
            category="environment",
            message=(
                f"Kilosort4 {version} is outside the SpikeLab tested "
                f"range [{_TESTED_KILOSORT4_VERSION_RANGE[0]}, "
                f"{_TESTED_KILOSORT4_VERSION_RANGE[1]})."
            ),
            remediation=(
                f"Pin Kilosort4 to a version inside "
                f"[{_TESTED_KILOSORT4_VERSION_RANGE[0]}, "
                f"{_TESTED_KILOSORT4_VERSION_RANGE[1]}), or run a smoke "
                "test before relying on a new release."
            ),
        )
    ]


def _check_docker_sorter(config: Any) -> List[PreflightFinding]:
    """Probe Docker-backed sorter dependencies (daemon + image cache).

    Validates that the Docker daemon is reachable and that the image
    selected by :func:`docker_utils.get_docker_image` for the chosen
    sorter is present in the local image cache. Pull-ability is not
    probed — that requires a registry round-trip and would defeat the
    "milliseconds-cheap" preflight goal; SpikeInterface will pull on
    first use if the image is missing, but we surface a warn-level
    finding so the operator knows ahead of time.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        findings (list[PreflightFinding]): Daemon failures are
            ``level="fail"``; missing local image is ``level="warn"``.
    """
    findings: List[PreflightFinding] = []

    daemon_ok = False
    try:
        import docker as _docker  # type: ignore[import-not-found]

        try:
            _docker.from_env().ping()
            daemon_ok = True
        except Exception as exc:
            findings.append(
                PreflightFinding(
                    level="fail",
                    code="sorter_dependency_missing",
                    category="environment",
                    message=(f"Docker daemon ping failed via docker-py: " f"{exc!r}."),
                    remediation=(
                        "Start Docker Desktop / the docker service, or "
                        "switch to a host-path sorter (set "
                        "SorterConfig.use_docker=False)."
                    ),
                )
            )
    except ImportError:
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                timeout=5,
                capture_output=True,
            )
            daemon_ok = True
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as exc:
            findings.append(
                PreflightFinding(
                    level="fail",
                    code="sorter_dependency_missing",
                    category="environment",
                    message=(
                        f"Docker daemon is not reachable: `docker info` "
                        f"failed ({exc!r})."
                    ),
                    remediation=(
                        "Start Docker Desktop / the docker service, or "
                        "switch to a host-path sorter (set "
                        "SorterConfig.use_docker=False)."
                    ),
                )
            )

    if not daemon_ok:
        return findings

    # Resolve the image tag the sort would actually use, then check
    # the local cache. Tag resolution can raise (e.g. unsupported
    # CUDA version) — surface that as a finding rather than letting
    # it crash the preflight.
    try:
        from ..docker_utils import get_docker_image

        image_tag = get_docker_image(getattr(config.sorter, "sorter_name", ""))
    except Exception as exc:
        findings.append(
            PreflightFinding(
                level="fail",
                code="sorter_dependency_missing",
                category="environment",
                message=(
                    f"Could not resolve the Docker image for "
                    f"{getattr(config.sorter, 'sorter_name', '?')}: "
                    f"{exc!r}."
                ),
                remediation=(
                    "Check sorter name and CUDA driver compatibility. "
                    "See docker_utils._IMAGE_REGISTRY for available tags."
                ),
            )
        )
        return findings

    cached = False
    try:
        import docker as _docker  # type: ignore[import-not-found]

        try:
            _docker.from_env().images.get(image_tag)
            cached = True
        except Exception:
            cached = False
    except ImportError:
        try:
            subprocess.run(
                ["docker", "image", "inspect", image_tag],
                check=True,
                timeout=5,
                capture_output=True,
            )
            cached = True
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            cached = False

    if not cached:
        findings.append(
            PreflightFinding(
                level="warn",
                code="sorter_dependency_missing",
                category="environment",
                message=(
                    f"Docker image {image_tag!s} is not in the local "
                    "cache. SpikeInterface will attempt to pull it on "
                    "first use, which can take minutes and fails "
                    "without network connectivity to the registry."
                ),
                remediation=(
                    f"Pre-pull the image with `docker pull {image_tag}` "
                    "before launching the sort to fail fast on network "
                    "or auth issues."
                ),
            )
        )

    return findings


def _check_rt_sort(config: Any) -> List[PreflightFinding]:
    """Probe RT-Sort runtime dependencies.

    RT-Sort needs PyTorch (DL detection model), diptest (amplitude
    unimodality test), scikit-learn (Gaussian mixture clustering),
    h5py (intermediate I/O), and tqdm (progress bars). When the
    configured device is CUDA, also verifies that ``torch.cuda`` is
    actually available so a missing driver does not surface as a
    cryptic kernel-launch error mid-sort.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        findings (list[PreflightFinding]): One fail-level finding per
            missing import; one additional fail when ``device="cuda"``
            but ``torch.cuda.is_available()`` is False.
    """
    findings: List[PreflightFinding] = []

    required = [
        ("torch", "PyTorch — required for the DL detection model"),
        ("diptest", "amplitude unimodality test in cluster splitting"),
        ("sklearn", "Gaussian mixture clustering"),
        ("h5py", "intermediate scaled-traces I/O"),
        ("tqdm", "progress bars"),
    ]
    for module_name, role in required:
        try:
            __import__(module_name)
        except ImportError as exc:
            findings.append(
                PreflightFinding(
                    level="fail",
                    code="sorter_dependency_missing",
                    category="environment",
                    message=(
                        f"RT-Sort requires `{module_name}` ({role}) but "
                        f"it is not importable: {exc}."
                    ),
                    remediation=(
                        f"Install the missing dependency, e.g. "
                        f"`pip install {module_name}`."
                    ),
                )
            )

    device = str(getattr(config.rt_sort, "device", "") or "")
    if device.startswith("cuda"):
        try:
            import torch as _torch

            if not _torch.cuda.is_available():
                findings.append(
                    PreflightFinding(
                        level="fail",
                        code="sorter_dependency_missing",
                        category="environment",
                        message=(
                            f"RT-Sort is configured with device={device!r} "
                            "but torch.cuda.is_available() is False."
                        ),
                        remediation=(
                            "Verify the NVIDIA driver, install a CUDA-"
                            "enabled PyTorch build, or switch RTSortConfig"
                            "(device='cpu')."
                        ),
                    )
                )
        except ImportError:
            # Already reported above as a missing dependency.
            pass

    return findings


def _check_sorter_dependencies(config: Any) -> List[PreflightFinding]:
    """Dispatch to the per-sorter dependency probe for the active config.

    Catches the most common environment-shaped failures (wrong conda
    env, missing CUDA wheels, broken Docker daemon, unset MATLAB path)
    in milliseconds rather than letting them surface as cryptic
    tracebacks deep inside the sort.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        findings (list[PreflightFinding]): All findings produced by
            the per-sorter probe; empty when dependencies look healthy
            or when the sorter is unrecognized.
    """
    sorter_name = getattr(config.sorter, "sorter_name", "").lower()
    use_docker = bool(getattr(config.sorter, "use_docker", False))

    if sorter_name in ("kilosort2", "kilosort4") and use_docker:
        return _check_docker_sorter(config)
    if sorter_name == "kilosort2":
        return _check_kilosort2_host(config)
    if sorter_name == "kilosort4":
        return _check_kilosort4_host(config)
    if sorter_name == "rt_sort":
        return _check_rt_sort(config)
    return []


def _resolve_target_device_index(config: Any) -> int:
    """Resolve the GPU device index the configured sorter would use.

    Mirrors :func:`._gpu_watchdog.resolve_active_device` so the
    preflight check sees the same target device the watchdog will
    monitor at run time.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        index (int): Device index (defaults to 0 when the sorter does
            not expose an explicit device).
    """
    from ._gpu_watchdog import _resolve_device_index

    sorter_name = getattr(config.sorter, "sorter_name", "").lower()
    if sorter_name == "rt_sort":
        return _resolve_device_index(getattr(config.rt_sort, "device", None))
    if sorter_name == "kilosort4":
        params = getattr(config.sorter, "sorter_params", None) or {}
        return _resolve_device_index(params.get("torch_device"))
    return 0


def _detect_gpu_device_count() -> Optional[int]:
    """Return the number of CUDA devices visible to the host.

    Tries pynvml first (cheapest, no torch import cost), then
    ``torch.cuda.device_count()``, then ``nvidia-smi``. Returns
    ``None`` when none of the three is available — callers should
    treat that as "cannot validate" and stay silent rather than
    emit noise (the existing ``vram_unknown`` finding already
    flags the broader detection gap).
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    # ``nvidia-smi --query-gpu=count`` repeats the count once per GPU
    # row, so any line is a valid sample.
    for line in out.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def _check_gpu_device_present(config: Any) -> Optional[PreflightFinding]:
    """Verify the configured GPU device index actually exists.

    A user setting ``torch_device="cuda:1"`` (or ``RTSortConfig.device=
    "cuda:2"``) on a host with one GPU otherwise discovers the mistake
    a minute or more into the sort, when CUDA reports an opaque
    invalid-device error from the kernel launch.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration. Must
            describe a GPU-backed sorter (caller is expected to gate
            on :func:`_sorter_uses_gpu`).

    Returns:
        finding (PreflightFinding or None): Fail-level finding when
            the resolved index is out of range. ``None`` when the
            index is valid or when the device count cannot be
            detected (silent skip — already covered by
            ``vram_unknown``).
    """
    target = _resolve_target_device_index(config)
    count = _detect_gpu_device_count()
    if count is None or count <= 0:
        return None
    if target < count:
        return None
    valid_indices = ", ".join(str(i) for i in range(count))
    return PreflightFinding(
        level="fail",
        code="gpu_device_not_present",
        category="environment",
        message=(
            f"Configured GPU device index {target} is out of range; "
            f"host exposes {count} CUDA device(s)."
        ),
        remediation=(
            f"Pick an available device index ({valid_indices}) via "
            "RTSortConfig.device='cuda:N' or "
            "SorterConfig.sorter_params={'torch_device': 'cuda:N'}."
        ),
    )


# Sample-rate windows beyond which sorter output becomes unreliable.
# KS2/KS4: bandpass + drift correction adapt across a wide range, but
# below 10 kHz the assumed spike timescales no longer hold and above
# 50 kHz the templates and PCA become numerically degenerate. RT-Sort:
# the bundled detection model is rate-locked — feeding rates outside
# the trained sampling-clock tolerance puts the model out of
# distribution and silently degrades quality.
_SAMPLE_RATE_RANGES_HZ: Dict[str, Tuple[float, float]] = {
    "kilosort2": (10_000.0, 50_000.0),
    "kilosort4": (10_000.0, 50_000.0),
}
_RT_SORT_NOMINAL_HZ: Dict[str, float] = {
    "mea": 20_000.0,
    "neuropixels": 30_000.0,
}
_RT_SORT_TOLERANCE_FRAC: float = 0.005  # 0.5 % recording-clock jitter


def _expected_sample_rate_window(config: Any) -> Optional[Tuple[float, float, str]]:
    """Return ``(low_hz, high_hz, label)`` for the configured sorter.

    Returns ``None`` when the sorter has no defined window (e.g. an
    unrecognized name or an RT-Sort probe variant we don't have
    nominal rates for).
    """
    sorter_name = getattr(config.sorter, "sorter_name", "").lower()
    if sorter_name in _SAMPLE_RATE_RANGES_HZ:
        low, high = _SAMPLE_RATE_RANGES_HZ[sorter_name]
        return low, high, sorter_name
    if sorter_name == "rt_sort":
        probe = str(getattr(config.rt_sort, "probe", "") or "").lower()
        nominal = _RT_SORT_NOMINAL_HZ.get(probe)
        if nominal is None:
            return None
        tol = nominal * _RT_SORT_TOLERANCE_FRAC
        return nominal - tol, nominal + tol, f"rt_sort/{probe}"
    return None


def _check_recording_sample_rate(
    config: Any,
    recording_files: Sequence[Any],
) -> List[PreflightFinding]:
    """Warn when a pre-loaded recording's rate sits outside the sorter window.

    Only inspects pre-loaded recordings (entries with
    ``get_sampling_frequency``). Path-only inputs are skipped — we do
    not load the recording for preflight; the existing
    :func:`_validate_recording_inputs` only confirms the file exists.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.
        recording_files (sequence): Per-recording inputs.

    Returns:
        findings (list[PreflightFinding]): One ``warn``-level finding
            per pre-loaded recording whose rate falls outside the
            sorter-specific window. Strict mode flips warnings into
            hard failures.
    """
    window = _expected_sample_rate_window(config)
    if window is None:
        return []
    low_hz, high_hz, label = window

    findings: List[PreflightFinding] = []
    for rec in recording_files:
        get_fs = getattr(rec, "get_sampling_frequency", None)
        if not callable(get_fs):
            continue
        try:
            fs_hz = float(get_fs())
        except Exception:
            continue
        if low_hz <= fs_hz <= high_hz:
            continue
        findings.append(
            PreflightFinding(
                level="warn",
                code="sample_rate_out_of_window",
                category="resource",
                message=(
                    f"Recording sampling rate {fs_hz / 1000.0:.2f} kHz "
                    f"is outside the {label} window "
                    f"[{low_hz / 1000.0:.2f}, {high_hz / 1000.0:.2f}] "
                    "kHz. Sorter output may degrade."
                ),
                remediation=(
                    "Resample the recording to within the supported "
                    "window, or pick a sorter whose window matches the "
                    "recording's native rate."
                ),
            )
        )
    return findings


def _parse_version_tuple(version: str) -> Optional[Tuple[int, ...]]:
    """Parse a dotted version string to a comparable tuple of ints."""
    try:
        parts = version.strip().split(".")
        return tuple(int("".join(c for c in p if c.isdigit())) for p in parts[:3])
    except Exception:
        return None


def _check_spikeinterface_version() -> Optional[PreflightFinding]:
    """Warn when SpikeInterface's version is outside the tested range.

    SpikeLab is verified against a specific SI version window. Older
    SI may lack APIs we depend on (e.g. ``run_sorter`` keyword
    arguments, ``ContainerClient`` fields). Newer SI may have
    introduced incompatibilities we have not yet caught.

    Returns ``None`` when SI is absent (no preflight to add — the
    relevant sort backend will fail later with a clearer message)
    or when the version is inside the tested range.
    """
    try:
        import spikeinterface as _si
    except ImportError:
        return None
    version = getattr(_si, "__version__", None)
    if version is None:
        return None
    parsed = _parse_version_tuple(version)
    low = _parse_version_tuple(_TESTED_SI_VERSION_RANGE[0])
    high = _parse_version_tuple(_TESTED_SI_VERSION_RANGE[1])
    if parsed is None or low is None or high is None:
        return None
    if low <= parsed < high:
        return None
    return PreflightFinding(
        level="warn",
        code="spikeinterface_version_outside_tested_range",
        category="environment",
        message=(
            f"SpikeInterface {version} is outside the SpikeLab tested "
            f"range [{_TESTED_SI_VERSION_RANGE[0]}, "
            f"{_TESTED_SI_VERSION_RANGE[1]}). Some sort paths may "
            "behave unexpectedly."
        ),
        remediation=(
            f"Pin SpikeInterface to a version inside "
            f"[{_TESTED_SI_VERSION_RANGE[0]}, "
            f"{_TESTED_SI_VERSION_RANGE[1]}), or run a smoke test to "
            "verify your sort path before relying on a new release."
        ),
    )


def _check_resource_rlimits(config: Any) -> List[PreflightFinding]:
    """POSIX-only: warn when ``RLIMIT_NOFILE`` / ``RLIMIT_NPROC`` are tight.

    RT-Sort opens many file descriptors during chunked I/O; KS4
    spawns multiple worker processes when ``num_processes > 1``.
    Constrained limits (some CI containers, shared hosts) cause
    opaque ``OSError [Errno 24] Too many open files`` or
    ``BlockingIOError`` failures deep inside the sort.

    Linux thresholds:

    * ``RLIMIT_NOFILE`` < 4096 → warn (RT-Sort's chunked I/O can
      hold thousands of FDs at once on dense MEAs).
    * ``RLIMIT_NPROC`` < 256 → warn (KS4 + RT-Sort spawn workers
      proportional to ``num_processes``).

    Returns an empty list on Windows where ``resource`` is
    unavailable.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration —
            inspected so RT-Sort's ``num_processes`` informs the
            NPROC threshold.

    Returns:
        findings (list[PreflightFinding]): Up to two warn-level
            findings. Empty when limits are healthy or the OS
            does not expose them.
    """
    try:
        import resource as _resource
    except ImportError:
        return []

    findings: List[PreflightFinding] = []

    try:
        soft_nofile, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        soft_nofile = None
    if soft_nofile is not None and 0 < soft_nofile < 4096:
        findings.append(
            PreflightFinding(
                level="warn",
                code="low_rlimit_nofile",
                category="environment",
                message=(
                    f"RLIMIT_NOFILE soft limit is {soft_nofile} "
                    "(< 4096). RT-Sort's chunked I/O and KS4 worker "
                    "pools may exhaust file descriptors during the sort."
                ),
                remediation=(
                    "Raise the limit before launching the sort, e.g. "
                    "`ulimit -n 65536` on bash, or via the systemd "
                    "service unit's LimitNOFILE setting."
                ),
            )
        )

    try:
        soft_nproc, _hard = _resource.getrlimit(_resource.RLIMIT_NPROC)
    except (AttributeError, ValueError, OSError):
        soft_nproc = None
    # NPROC threshold scales with the configured worker count for
    # RT-Sort. Default to 256 if no explicit setting is given.
    nproc_needed = 256
    rt = getattr(config, "rt_sort", None)
    if rt is not None:
        cfg_n = getattr(rt, "num_processes", None)
        if isinstance(cfg_n, int) and cfg_n > 0:
            nproc_needed = max(256, 4 * cfg_n)
    if soft_nproc is not None and 0 < soft_nproc < nproc_needed:
        findings.append(
            PreflightFinding(
                level="warn",
                code="low_rlimit_nproc",
                category="environment",
                message=(
                    f"RLIMIT_NPROC soft limit is {soft_nproc} "
                    f"(< {nproc_needed}). Worker spawning may fail "
                    "with BlockingIOError partway through the sort."
                ),
                remediation=(
                    "Raise the limit before launching the sort, e.g. "
                    "`ulimit -u 4096`, or reduce "
                    "``RTSortConfig.num_processes`` if you cannot."
                ),
            )
        )

    return findings


def _check_filesystem_writable(
    folders: Sequence[Any],
    *,
    label: str,
    code_prefix: str,
) -> List[PreflightFinding]:
    """Verify that *folders* live on writable filesystems.

    A read-only mount (e.g. an NFS export that flipped to RO after a
    storage event, or a mistakenly-mounted snapshot) passes the
    free-disk check but fails on the first write. Catching this in
    preflight surfaces the misconfiguration in milliseconds rather
    than seconds-to-minutes into a sort.

    For folders that do not yet exist, the nearest existing parent
    is checked instead — the sort will create the folder later.

    Parameters:
        folders (sequence of path-like): Folders to validate.
        label (str): Human-readable folder kind (``"intermediate"``
            or ``"results"``) used in the message.
        code_prefix (str): Stable prefix for the finding code
            (``"intermediate"`` → ``"intermediate_readonly"``).

    Returns:
        findings (list[PreflightFinding]): One ``fail``-level
            finding per folder whose nearest existing parent is not
            writable.
    """
    findings: List[PreflightFinding] = []
    for folder in folders:
        p = Path(folder)
        while not p.exists() and p.parent != p:
            p = p.parent
        if not p.exists():
            continue
        if os.access(p, os.W_OK):
            continue
        findings.append(
            PreflightFinding(
                level="fail",
                code=f"{code_prefix}_readonly",
                category="environment",
                message=(
                    f"{label.capitalize()} folder {folder!s} is on a "
                    f"non-writable filesystem (nearest existing "
                    f"parent {p!s} fails W_OK)."
                ),
                remediation=(
                    "Pick a writable path or remount the volume "
                    "read-write. Common causes: NFS export flipped "
                    "to RO after a storage event, mounted snapshot, "
                    "or insufficient permissions on a shared drive."
                ),
            )
        )
    return findings


def _hdf5_plugin_finding(config: Any) -> Optional[PreflightFinding]:
    """Validate ``HDF5_PLUGIN_PATH`` when configured.

    Surfaces the same root cause the post-mortem classifier
    (:class:`HDF5PluginMissingError`) detects — but before any data is
    loaded, so an early operator can fix the path without waiting for
    the sort to fail.
    """
    configured = getattr(config.recording, "hdf5_plugin_path", None)
    if configured is None:
        configured = os.environ.get("HDF5_PLUGIN_PATH")
    if not configured:
        return None
    path = Path(configured)
    if path.is_dir():
        return None
    return PreflightFinding(
        level="fail",
        code="hdf5_plugin_missing",
        category="environment",
        message=(
            f"HDF5_PLUGIN_PATH points to {path!s} but the directory " "does not exist."
        ),
        remediation=(
            "Set HDF5_PLUGIN_PATH (via RecordingConfig.hdf5_plugin_path "
            "or the environment) to a directory containing the HDF5 "
            "compression plugin needed for the recording."
        ),
    )


def run_preflight(
    config: Any,
    recording_files: Sequence[Any],
    intermediate_folders: Sequence[Any],
    results_folders: Sequence[Any],
) -> List[PreflightFinding]:
    """Run pre-loop resource checks; return all findings.

    Findings are not raised by this function — the caller decides
    whether to escalate based on ``ExecutionConfig.preflight_strict``.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration. Reads
            thresholds from ``config.execution``.
        recording_files (sequence): Recording inputs (used for length
            sanity in future checks; currently unused but kept in the
            signature for forward compatibility).
        intermediate_folders (sequence of path-like): Per-recording
            intermediate folders. Disk free space is checked at each
            folder's nearest existing ancestor.
        results_folders (sequence of path-like): Per-recording results
            folders. Disk free space is checked similarly.

    Returns:
        findings (list[PreflightFinding]): All findings produced by
            the checks. May be empty when the host has plenty of
            headroom.
    """
    exe = config.execution
    findings: List[PreflightFinding] = []

    min_free_inter = float(exe.preflight_min_free_inter_gb)
    min_free_results = float(exe.preflight_min_free_results_gb)
    min_avail_ram = float(exe.preflight_min_available_ram_gb)
    min_free_vram = float(exe.preflight_min_free_vram_gb)

    # ---------- Disk -----------------------------------------------------
    for folder in intermediate_folders:
        free_gb = _disk_free_gb(Path(folder))
        if free_gb is None:
            continue
        if free_gb < min_free_inter:
            findings.append(
                PreflightFinding(
                    level="warn",
                    code="low_disk_inter",
                    message=(
                        f"Intermediate folder {folder!s} parent has only "
                        f"{free_gb:.1f} GB free (< {min_free_inter:.1f} GB)."
                    ),
                    remediation=(
                        "Free disk space or point intermediate_folders at "
                        "a larger volume. RT-Sort and Kilosort write large "
                        "temporary files."
                    ),
                )
            )

    for folder in results_folders:
        free_gb = _disk_free_gb(Path(folder))
        if free_gb is None:
            continue
        if free_gb < min_free_results:
            findings.append(
                PreflightFinding(
                    level="warn",
                    code="low_disk_results",
                    message=(
                        f"Results folder {folder!s} parent has only "
                        f"{free_gb:.1f} GB free (< {min_free_results:.1f} GB)."
                    ),
                    remediation=(
                        "Free disk space or point results_folders at a "
                        "larger volume."
                    ),
                )
            )

    # ---------- RAM ------------------------------------------------------
    avail_ram = _available_ram_gb()
    if avail_ram is None:
        findings.append(
            PreflightFinding(
                level="warn",
                code="ram_unknown",
                message=(
                    "psutil not installed — cannot check available host "
                    "RAM. The host-memory watchdog will also be disabled."
                ),
                remediation="Install psutil to enable RAM-based safety checks.",
            )
        )
    elif avail_ram < min_avail_ram:
        findings.append(
            PreflightFinding(
                level="warn",
                code="low_ram",
                message=(
                    f"Only {avail_ram:.1f} GB host RAM available "
                    f"(< {min_avail_ram:.1f} GB). Sort may trigger the "
                    "watchdog or thrash on Windows."
                ),
                remediation=(
                    "Close other applications or shorten the recording "
                    "before sorting."
                ),
            )
        )

    # ---------- GPU VRAM -------------------------------------------------
    if _sorter_uses_gpu(config):
        free_vram = _free_vram_gb()
        if free_vram is None:
            findings.append(
                PreflightFinding(
                    level="warn",
                    code="vram_unknown",
                    message=(
                        "Sorter requires a GPU but VRAM availability could "
                        "not be detected (no pynvml, no nvidia-smi)."
                    ),
                    remediation=(
                        "Install pynvml or ensure nvidia-smi is on PATH so "
                        "VRAM headroom can be checked before the sort."
                    ),
                )
            )
        elif free_vram < min_free_vram:
            findings.append(
                PreflightFinding(
                    level="warn",
                    code="low_vram",
                    message=(
                        f"Only {free_vram:.1f} GB GPU memory free "
                        f"(< {min_free_vram:.1f} GB). Risk of "
                        "GPUOutOfMemoryError during sort."
                    ),
                    remediation=(
                        "Close other GPU consumers, reduce batch size, or "
                        "switch to a larger-memory GPU."
                    ),
                )
            )

    # ---------- Recording inputs -----------------------------------------
    findings.extend(_validate_recording_inputs(recording_files))

    # ---------- Recording sample-rate window -----------------------------
    findings.extend(_check_recording_sample_rate(config, recording_files))

    # ---------- Sorter dependency probes ---------------------------------
    findings.extend(_check_sorter_dependencies(config))

    # ---------- GPU device existence -------------------------------------
    if _sorter_uses_gpu(config):
        gpu_dev = _check_gpu_device_present(config)
        if gpu_dev is not None:
            findings.append(gpu_dev)

    # ---------- POSIX resource limits ------------------------------------
    findings.extend(_check_resource_rlimits(config))

    # ---------- SpikeInterface version range -----------------------------
    si_finding = _check_spikeinterface_version()
    if si_finding is not None:
        findings.append(si_finding)

    # ---------- Filesystem writability -----------------------------------
    findings.extend(
        _check_filesystem_writable(
            intermediate_folders, label="intermediate", code_prefix="intermediate"
        )
    )
    findings.extend(
        _check_filesystem_writable(
            results_folders, label="results", code_prefix="results"
        )
    )

    # ---------- HDF5 plugin path -----------------------------------------
    hdf5 = _hdf5_plugin_finding(config)
    if hdf5 is not None:
        findings.append(hdf5)

    # ---------- .wslconfig (Docker-on-Windows) ---------------------------
    wsl = _wslconfig_finding(config)
    if wsl is not None:
        findings.append(wsl)

    # ---------- RT-Sort intermediate-disk projection ---------------------
    rt_disk = _rt_sort_disk_finding(config, recording_files, intermediate_folders)
    if rt_disk is not None:
        findings.append(rt_disk)

    return findings


def report_findings(
    findings: Sequence[PreflightFinding], *, strict: bool = False
) -> None:
    """Print findings and raise if any escalate to a hard failure.

    Parameters:
        findings (sequence[PreflightFinding]): Output of
            :func:`run_preflight`.
        strict (bool): When True, every ``"warn"`` finding is treated
            as ``"fail"``. Defaults to False.

    Raises:
        EnvironmentSortFailure: If a finding has ``level == "fail"``
            (or ``"warn"`` under *strict*) and category
            ``"environment"``.
        ResourceSortFailure: If a finding has ``level == "fail"`` (or
            ``"warn"`` under *strict*) and category ``"resource"``.
    """
    if not findings:
        print("[preflight] all checks passed")
        return

    print("[preflight] findings:")
    fatal: List[PreflightFinding] = []
    for f in findings:
        effective_level = "fail" if (strict and f.level == "warn") else f.level
        marker = "FAIL" if effective_level == "fail" else "WARN"
        print(f"  [{marker}] {f.code}: {f.message}")
        if f.remediation:
            print(f"         -> {f.remediation}")
        if effective_level == "fail":
            fatal.append(f)

    if not fatal:
        return

    # Prefer a categorical match for the first fatal finding so callers
    # can branch on EnvironmentSortFailure vs ResourceSortFailure.
    first = fatal[0]
    summary = (
        f"Preflight failed: {len(fatal)} fatal finding(s). "
        f"First: {first.code} — {first.message}"
    )
    if first.code == "hdf5_plugin_missing":
        configured = os.environ.get("HDF5_PLUGIN_PATH")
        raise HDF5PluginMissingError(summary, configured_path=configured)
    if first.category == "environment":
        raise EnvironmentSortFailure(summary)
    raise ResourceSortFailure(summary)
