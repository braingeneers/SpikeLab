"""Post-failure classifiers for spike-sorting exceptions.

Each ``_classify_*`` helper inspects either an exception chain or a
sorter log and returns a specific
:class:`SpikeSortingClassifiedError` subclass when it recognises the
signature, or ``None`` to let the caller keep the original exception.

Dispatchers :func:`classify_ks2_failure` and
:func:`classify_ks4_failure` run the applicable helpers in priority
order (environment and resource signatures before biology, so a
genuine config problem on an active well is not misclassified as
"insufficient activity").

All regex signatures are tolerant of surrounding formatting so they
work across SpikeInterface versions and sorter log formats. They do
not depend on deployment-specific paths.
"""

import os
import re
from pathlib import Path
from typing import Optional

from ._exceptions import (
    DockerEnvironmentError,
    GPUOutOfMemoryError,
    HDF5PluginMissingError,
    InsufficientActivityError,
    NoGoodChannelsError,
    SpikeSortingClassifiedError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_exception_chain(exc: Optional[BaseException]) -> str:
    """Concatenate all messages in an exception's cause/context chain.

    Uses identity checks to break cycles. Handy for matching signatures
    produced by wrappers (SpikeInterface re-raises sklearn errors) where
    the interesting message is on an inner link.
    """
    messages: list[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(messages)


def _read_log_if_exists(path: Optional[Path]) -> Optional[str]:
    """Read a log file, returning ``None`` on any failure."""
    if path is None or not path.is_file():
        return None
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _find_ks2_log(output_folder: Path) -> Optional[Path]:
    """Locate ``kilosort2.log`` for either Docker or MATLAB execution paths."""
    for candidate in (
        output_folder / "kilosort2.log",
        output_folder / "sorter_output" / "kilosort2.log",
    ):
        if candidate.is_file():
            return candidate
    return None


def _find_ks4_log(output_folder: Path) -> Optional[Path]:
    """Locate ``kilosort4.log`` when present."""
    for candidate in (
        output_folder / "kilosort4.log",
        output_folder / "sorter_output" / "kilosort4.log",
    ):
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Environment classifiers (match first — env issues can occur on any data)
# ---------------------------------------------------------------------------


_HDF5_PLUGIN_MARKERS = (
    "HDF5_PLUGIN_PATH",
    "HDF5 filter",
    "Can't open directory",
    "libcompression",
    "Unable to synchronously read data",
)


def _classify_hdf5_plugin_missing(
    chain_text: str, log_text: Optional[str]
) -> Optional[HDF5PluginMissingError]:
    """HDF5 plugin load failure — unable to open a compressed dataset."""
    haystack = chain_text if log_text is None else f"{chain_text}\n{log_text}"
    if not any(marker in haystack for marker in _HDF5_PLUGIN_MARKERS):
        return None
    # Only treat it as a plugin issue when the error is about filter
    # decoding, not a generic "file not found" on the recording itself.
    if (
        "HDF5_PLUGIN_PATH" not in haystack
        and "filter" not in haystack.lower()
        and "compression" not in haystack.lower()
    ):
        return None
    configured = os.environ.get("HDF5_PLUGIN_PATH")
    message = (
        "HDF5 filter plugin is missing or HDF5_PLUGIN_PATH is misconfigured. "
        "Set HDF5_PLUGIN_PATH to a directory containing the compression "
        "plugin required by the recording before importing h5py-based "
        "loaders."
    )
    return HDF5PluginMissingError(message, configured_path=configured)


_DOCKER_DAEMON_MARKERS = (
    "Cannot connect to the Docker daemon",
    "Is the docker daemon running",
    "connect: no such file or directory",
)
_DOCKER_CLIENT_MISSING_MARKERS = (
    "No module named 'docker'",
    "ModuleNotFoundError: No module named \"docker\"",
)
_DOCKER_PERMISSION_MARKERS = (
    "permission denied while trying to connect to the Docker daemon",
    "docker: Got permission denied",
)
_DOCKER_PULL_MARKERS = (
    "manifest unknown",
    "pull access denied",
    "failed to resolve reference",
    "failed to pull and unpack image",
    "dial tcp: lookup",
    "docker.errors.ImageNotFound",
    "error pulling image",
)


def _classify_docker_environment(
    chain_text: str, log_text: Optional[str]
) -> Optional[DockerEnvironmentError]:
    haystack = chain_text if log_text is None else f"{chain_text}\n{log_text}"
    if any(marker in haystack for marker in _DOCKER_CLIENT_MISSING_MARKERS):
        return DockerEnvironmentError(
            "Python docker client library is not installed in the sorting "
            "environment. Install 'docker' (docker-py) before using the "
            "Docker-backed sort path.",
            reason="client_missing",
        )
    if any(marker in haystack for marker in _DOCKER_PERMISSION_MARKERS):
        return DockerEnvironmentError(
            "Permission denied connecting to the Docker daemon. The user "
            "running the sort is not authorised to access the Docker "
            "socket.",
            reason="permission_denied",
        )
    if any(marker in haystack for marker in _DOCKER_DAEMON_MARKERS):
        return DockerEnvironmentError(
            "Cannot reach the Docker daemon. Confirm Docker is running "
            "and the socket is accessible before retrying.",
            reason="daemon_down",
        )
    if any(marker in haystack for marker in _DOCKER_PULL_MARKERS):
        return DockerEnvironmentError(
            "Docker image pull failed. Image may be missing, registry "
            "auth may be stale, or the host cannot reach the registry.",
            reason="image_pull_failed",
        )
    return None


# ---------------------------------------------------------------------------
# Resource classifiers
# ---------------------------------------------------------------------------


_GPU_OOM_MARKERS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "OutOfMemoryError",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "Out of memory on device",
    "cudaErrorMemoryAllocation",
)


def _classify_gpu_oom(
    sorter: str,
    chain_text: str,
    log_text: Optional[str],
    log_path: Optional[Path],
) -> Optional[GPUOutOfMemoryError]:
    haystack = chain_text if log_text is None else f"{chain_text}\n{log_text}"
    if not any(marker in haystack for marker in _GPU_OOM_MARKERS):
        return None
    message = (
        f"{sorter} ran out of GPU memory. Reduce batch size / NT / nPCs, "
        "shorten the recording window, or switch to a GPU with more "
        "memory before retrying."
    )
    return GPUOutOfMemoryError(message, sorter=sorter, log_path=log_path)


# ---------------------------------------------------------------------------
# Biology classifiers — KS2 log-based
# ---------------------------------------------------------------------------


_KS2_CUDA_INVALID_CONFIG = "invalid configuration argument"
_KS2_THRESH_CROSS_RE = re.compile(r"found\s+(\d+)\s+threshold crossings")
_KS2_TEMPLATE_OPT_RE = re.compile(
    r"\b(\d+)\s*/\s*\d+\s*batches,\s*(\d+)\s*units,\s*nspks:\s*([0-9.]+)"
)
_KS2_NCHAN_RE = re.compile(r"Recording has\s+(\d+)\s+channels")
_KS2_BAD_CHANNELS_RE = re.compile(r"found\s+(\d+)\s+bad channels")
_KS2_ZERO_GOOD_CHANNELS_RE = re.compile(r"found\s+0\s+good channels\b")

_KS2_MIN_THRESHOLD_CROSSINGS = 20_000
_KS2_MAX_UNITS_AT_FAILURE = 5
_KS2_MAX_NSPKS_AT_FAILURE = 5.0


def _classify_no_good_channels_ks2(
    log_text: Optional[str], log_path: Optional[Path]
) -> Optional[NoGoodChannelsError]:
    """KS2 flagged every channel as bad."""
    if not log_text:
        return None
    nchan_m = _KS2_NCHAN_RE.search(log_text)
    bad_m = _KS2_BAD_CHANNELS_RE.search(log_text)
    zero_good = _KS2_ZERO_GOOD_CHANNELS_RE.search(log_text) is not None

    total = int(nchan_m.group(1)) if nchan_m else None
    bad = int(bad_m.group(1)) if bad_m else None

    all_bad = total is not None and bad is not None and bad >= total
    if not (zero_good or all_bad):
        return None

    message = (
        "Kilosort2 flagged every channel as bad; no good channels "
        "remained for sorting. Check electrode contact, recording "
        "gain, and the minfr_goodchannels parameter."
    )
    return NoGoodChannelsError(
        message,
        sorter="kilosort2",
        total_channels=total,
        bad_channels=bad,
        log_path=log_path,
    )


def _classify_insufficient_activity_ks2(
    log_text: Optional[str], log_path: Optional[Path], exc: BaseException
) -> Optional[InsufficientActivityError]:
    """KS2 crashed on near-silent data with degenerate kernel launches."""
    if not log_text or _KS2_CUDA_INVALID_CONFIG not in log_text:
        return None

    thresh_match = _KS2_THRESH_CROSS_RE.search(log_text)
    threshold_crossings = int(thresh_match.group(1)) if thresh_match else None

    # Take the last template-optimization line before the crash.
    opt_matches = list(_KS2_TEMPLATE_OPT_RE.finditer(log_text))
    if opt_matches:
        last = opt_matches[-1]
        units_at_failure: Optional[int] = int(last.group(2))
        nspks_at_failure: Optional[float] = float(last.group(3))
    else:
        units_at_failure = None
        nspks_at_failure = None

    low_crossings = (
        threshold_crossings is not None
        and threshold_crossings < _KS2_MIN_THRESHOLD_CROSSINGS
    )
    few_units = (
        units_at_failure is not None
        and units_at_failure <= _KS2_MAX_UNITS_AT_FAILURE
    )
    low_nspks = (
        nspks_at_failure is not None
        and nspks_at_failure <= _KS2_MAX_NSPKS_AT_FAILURE
    )
    if not (low_crossings or few_units or low_nspks):
        return None

    parts = []
    if threshold_crossings is not None:
        parts.append(f"{threshold_crossings} threshold crossings")
    if units_at_failure is not None:
        parts.append(f"{units_at_failure} templates at crash")
    if nspks_at_failure is not None:
        parts.append(f"nspks={nspks_at_failure:g}")
    evidence = "; ".join(parts) if parts else "no activity metrics parsed"

    message = (
        "Kilosort2 crashed on near-silent recording — insufficient "
        "spiking activity for sorting. Evidence from log: "
        f"{evidence}. CUDA kernel error ('invalid configuration "
        "argument') is a known symptom of degenerate kernel launches "
        f"on low-unit templates. Original exception: {exc!r}."
        + (f" See {log_path} for full trace." if log_path else "")
    )
    return InsufficientActivityError(
        message,
        sorter="kilosort2",
        threshold_crossings=threshold_crossings,
        units_at_failure=units_at_failure,
        nspks_at_failure=nspks_at_failure,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# Biology classifiers — KS4 chain-based
# ---------------------------------------------------------------------------


_KS4_SVD_EMPTY_RE = re.compile(
    r"Found array with\s+(\d+)\s+sample\(s\).*?required by TruncatedSVD",
    re.DOTALL,
)
_KS4_KMEANS_RE = re.compile(r"n_samples=(\d+)\s+should be\s+>=\s+n_clusters=(\d+)")


def _classify_insufficient_activity_ks4(
    chain_text: str, log_path: Optional[Path], exc: BaseException
) -> Optional[InsufficientActivityError]:
    svd_match = _KS4_SVD_EMPTY_RE.search(chain_text)
    kmeans_match = _KS4_KMEANS_RE.search(chain_text)
    if svd_match is None and kmeans_match is None:
        return None

    if svd_match is not None:
        n_samples = int(svd_match.group(1))
        reason = (
            f"Kilosort4 spike detection returned {n_samples} events — "
            "TruncatedSVD requires at least 1. Well is effectively silent."
        )
    else:
        assert kmeans_match is not None
        n_samples = int(kmeans_match.group(1))
        n_clusters = int(kmeans_match.group(2))
        reason = (
            f"Kilosort4 spike detection returned only {n_samples} events, "
            f"below the KMeans n_clusters={n_clusters} minimum. Well has "
            "too little activity to cluster."
        )

    message = (
        f"{reason} Original exception: {exc!r}."
        + (f" See {log_path} for full trace." if log_path else "")
    )
    return InsufficientActivityError(
        message,
        sorter="kilosort4",
        units_at_failure=n_samples,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------


def classify_ks2_failure(
    output_folder: Path, exc: BaseException
) -> Optional[SpikeSortingClassifiedError]:
    """Return a classified exception for a Kilosort2 failure, or ``None``.

    Priority: environment → resource → biology. Environment and resource
    errors can appear on any recording, so they take precedence over
    biology signatures that would otherwise be consistent with them.
    """
    output_folder = Path(output_folder)
    log_path = _find_ks2_log(output_folder)
    log_text = _read_log_if_exists(log_path)
    chain_text = _walk_exception_chain(exc)

    hdf5 = _classify_hdf5_plugin_missing(chain_text, log_text)
    if hdf5 is not None:
        return hdf5
    docker_err = _classify_docker_environment(chain_text, log_text)
    if docker_err is not None:
        return docker_err
    oom = _classify_gpu_oom("kilosort2", chain_text, log_text, log_path)
    if oom is not None:
        return oom
    no_channels = _classify_no_good_channels_ks2(log_text, log_path)
    if no_channels is not None:
        return no_channels
    return _classify_insufficient_activity_ks2(log_text, log_path, exc)


def classify_ks4_failure(
    output_folder: Path, exc: BaseException
) -> Optional[SpikeSortingClassifiedError]:
    """Return a classified exception for a Kilosort4 failure, or ``None``.

    Priority mirrors KS2. KS4 does not expose a distinct "all channels bad"
    diagnostic the same way KS2 does, so only the generic biology
    classifier (insufficient activity) is applied.
    """
    output_folder = Path(output_folder)
    log_path = _find_ks4_log(output_folder)
    log_text = _read_log_if_exists(log_path)
    chain_text = _walk_exception_chain(exc)

    hdf5 = _classify_hdf5_plugin_missing(chain_text, log_text)
    if hdf5 is not None:
        return hdf5
    docker_err = _classify_docker_environment(chain_text, log_text)
    if docker_err is not None:
        return docker_err
    oom = _classify_gpu_oom("kilosort4", chain_text, log_text, log_path)
    if oom is not None:
        return oom
    return _classify_insufficient_activity_ks4(chain_text, log_path, exc)
