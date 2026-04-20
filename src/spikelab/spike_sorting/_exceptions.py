"""Classified spike-sorting exceptions shared across runners and curation.

Failures from Kilosort2, Kilosort4, and the downstream curation/waveform
code are grouped into three categories so callers can implement retry /
skip / hard-stop policies without parsing generic ``Exception`` messages:

* :class:`BiologicalSortFailure` — the recording itself cannot be sorted
  (too silent, all channels bad, no waveforms to compute metrics on).
  Recommended policy: mark the target as not-sortable, move on, do not
  retry.

* :class:`EnvironmentSortFailure` — the host environment or container
  runtime is misconfigured. Recommended policy: hard stop and surface
  to the operator; retrying without intervention will loop.

* :class:`ResourceSortFailure` — the job exhausted a machine resource
  (GPU memory today; disk/CPU in future). Recommended policy: retry
  with reduced parameters rather than skip or hard-stop.

Classifiers in :mod:`._classifier` inspect sorter logs and exception
chains to re-raise generic failures as one of the specific types below.
The classes are also usable directly from non-classifier paths (e.g.
curation code that already knows the exact condition).
"""

from pathlib import Path
from typing import Optional


class SpikeSortingClassifiedError(RuntimeError):
    """Base class for all classified sort-pipeline failures.

    Catch this when you want to treat any identified failure uniformly.
    Prefer catching the more specific categorical bases
    (:class:`BiologicalSortFailure`, :class:`EnvironmentSortFailure`,
    :class:`ResourceSortFailure`) when the policy differs by category.
    """


class BiologicalSortFailure(SpikeSortingClassifiedError):
    """Failure caused by the recording itself (too little signal)."""


class EnvironmentSortFailure(SpikeSortingClassifiedError):
    """Failure caused by host or container environment misconfiguration."""


class ResourceSortFailure(SpikeSortingClassifiedError):
    """Failure caused by exhausting a machine resource."""


# ---------------------------------------------------------------------------
# Biological failures
# ---------------------------------------------------------------------------


class InsufficientActivityError(BiologicalSortFailure):
    """Sorting crashed because the recording has too little spiking activity.

    Both Kilosort2 and Kilosort4 fall over on near-silent recordings, but in
    different ways:

    * **Kilosort2:** mex kernels launch with degenerate grid/block
      configurations when template counts and per-batch spike counts
      approach zero. Pre-Blackwell GPUs tolerated these launches; newer
      architectures (compute capability ≥ 12) reject them with
      ``CUDA error: invalid configuration argument``.

    * **Kilosort4:** sklearn's ``TruncatedSVD`` rejects an empty feature
      matrix, or ``KMeans`` fails the ``n_samples >= n_clusters`` check,
      when the initial spike-detection pass finds essentially no events.

    Attributes:
        threshold_crossings: KS2 only; count of detected threshold
            crossings parsed from ``kilosort2.log``. ``None`` for KS4.
        units_at_failure: KS2 template count at the crash, or KS4
            ``n_samples`` when KMeans complained. ``None`` when the log
            did not expose the value.
        nspks_at_failure: KS2 only; spikes-per-batch at the failing
            template-optimization step.
        log_path: Sorter log file carrying the full trace when located.
        sorter: Short identifier of the sorter that raised
            (``"kilosort2"``, ``"kilosort4"``).
    """

    def __init__(
        self,
        message: str,
        *,
        sorter: str,
        threshold_crossings: Optional[int] = None,
        units_at_failure: Optional[int] = None,
        nspks_at_failure: Optional[float] = None,
        log_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.sorter = sorter
        self.threshold_crossings = threshold_crossings
        self.units_at_failure = units_at_failure
        self.nspks_at_failure = nspks_at_failure
        self.log_path = log_path


class NoGoodChannelsError(BiologicalSortFailure):
    """All channels were flagged as bad by the sorter's good-channel check.

    Distinct from :class:`InsufficientActivityError`: the signal may be
    noisy/present but no channel passes the sorter's ``minfr_goodchannels``
    (or equivalent) firing-rate threshold.

    Attributes:
        total_channels: Total channel count in the recording, when parsed.
        bad_channels: Channels flagged as bad.
        log_path: Sorter log file carrying the full trace when located.
        sorter: Short identifier of the sorter that raised.
    """

    def __init__(
        self,
        message: str,
        *,
        sorter: str,
        total_channels: Optional[int] = None,
        bad_channels: Optional[int] = None,
        log_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.sorter = sorter
        self.total_channels = total_channels
        self.bad_channels = bad_channels
        self.log_path = log_path


class SaturatedSignalError(BiologicalSortFailure):
    """Recording appears flat or rail-saturated across all channels.

    Typical causes: disconnected electrodes, loss of fluid contact, broken
    amplifier front-end, or a saved recording that never received real
    data. Distinct from :class:`InsufficientActivityError` because it
    reflects a hardware/acquisition fault rather than biology.

    The sort-time log signatures are ambiguous with near-silent biology,
    so this class is currently intended to be raised by dedicated
    pre-sort validators (e.g. per-channel variance / rail-clip checks)
    rather than by the post-failure classifiers. Callers that already
    know the condition may raise it directly.

    Attributes:
        channels_saturated: Number of channels identified as saturated,
            when the caller provides this.
        total_channels: Total channel count in the recording.
    """

    def __init__(
        self,
        message: str,
        *,
        channels_saturated: Optional[int] = None,
        total_channels: Optional[int] = None,
    ):
        super().__init__(message)
        self.channels_saturated = channels_saturated
        self.total_channels = total_channels


class EmptyWaveformMetricsError(BiologicalSortFailure):
    """Waveform metrics (SNR, std-norm) cannot be computed.

    Raised when curation requests a waveform-based metric but no
    precomputed values exist and ``raw_data`` on the ``SpikeData`` is
    empty, so there is nothing to extract waveforms from.

    This is biology-adjacent: it typically means the upstream sorter
    produced units that have no usable waveform evidence attached, or
    that the pipeline skipped the waveform-extraction stage. Callers
    should treat it as "cannot curate this target" rather than retry.

    Attributes:
        metric_name: The metric that could not be computed.
    """

    def __init__(self, message: str, *, metric_name: Optional[str] = None):
        super().__init__(message)
        self.metric_name = metric_name


# ---------------------------------------------------------------------------
# Environment failures
# ---------------------------------------------------------------------------


class HDF5PluginMissingError(EnvironmentSortFailure):
    """HDF5 filter plugin is missing or the plugin path is misconfigured.

    Typical signatures in the underlying exception chain: h5py / HDF5
    errors about being unable to open a compressed dataset, or the
    inherited ``HDF5_PLUGIN_PATH`` environment variable pointing to a
    non-existent directory.

    Recommended remediation (operator, not the library): set
    ``HDF5_PLUGIN_PATH`` to a directory containing the compression
    plugin required by the recording's HDF5 build before any h5py import.
    The exact directory and plugin name are deployment-specific.

    Attributes:
        configured_path: The value of ``HDF5_PLUGIN_PATH`` at failure
            time, if known.
    """

    def __init__(self, message: str, *, configured_path: Optional[str] = None):
        super().__init__(message)
        self.configured_path = configured_path


class DockerEnvironmentError(EnvironmentSortFailure):
    """Docker daemon, client library, or image is unusable for sorting.

    The ``reason`` string narrows the failure mode so callers can render
    better diagnostics or choose different remediations without catching
    sub-exceptions.

    Recognized ``reason`` values:

    * ``"daemon_down"`` — Cannot connect to the Docker daemon.
    * ``"client_missing"`` — The Python ``docker`` client library is not
      installed in the sorting env.
    * ``"image_pull_failed"`` — Image pull returned an error (network,
      auth, or manifest-not-found).
    * ``"permission_denied"`` — Socket permission denied; user not in
      the ``docker`` group or equivalent.
    * ``"other"`` — Docker is broken in a way that did not match any
      known signature; inspect ``__cause__`` for details.

    Attributes:
        reason: One of the strings above.
    """

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Resource failures
# ---------------------------------------------------------------------------


class GPUOutOfMemoryError(ResourceSortFailure):
    """The sorter exhausted GPU memory.

    Raised when either a PyTorch ``CUDA out of memory`` error (KS4) or a
    MATLAB/mex ``CUDA_ERROR_OUT_OF_MEMORY`` diagnostic (KS2) appears in
    the exception chain or sorter log.

    Recommended remediation: reduce batch size / ``NT`` / ``nPCs``, split
    the recording into shorter segments, or run on a larger-memory GPU.
    Retrying the same command unchanged will loop.

    Attributes:
        sorter: Short identifier of the sorter that raised.
        log_path: Sorter log file carrying the full trace when located.
    """

    def __init__(
        self,
        message: str,
        *,
        sorter: str,
        log_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.sorter = sorter
        self.log_path = log_path
