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
from typing import Any, Optional


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

    Kilosort2, Kilosort4, and RT-Sort all fail on near-silent recordings,
    but in different ways:

    * **Kilosort2:** mex kernels launch with degenerate grid/block
      configurations when template counts and per-batch spike counts
      approach zero. Pre-Blackwell GPUs tolerated these launches; newer
      architectures (compute capability ≥ 12) reject them with
      ``CUDA error: invalid configuration argument``.

    * **Kilosort4:** sklearn's ``TruncatedSVD`` rejects an empty feature
      matrix, or ``KMeans`` fails the ``n_samples >= n_clusters`` check,
      when the initial spike-detection pass finds essentially no events.

    * **RT-Sort:** ``detect_sequences`` produces zero propagation
      sequences when the recording lacks sufficient spiking activity for
      clustering. Returns ``None``, which causes an ``AttributeError``
      when ``sort_offline`` is subsequently called.

    Attributes:
        threshold_crossings: KS2 only; count of detected threshold
            crossings parsed from ``kilosort2.log``. ``None`` for
            KS4 / RT-Sort.
        units_at_failure: KS2 template count at the crash, or KS4
            ``n_samples`` when KMeans complained. ``None`` when the log
            did not expose the value.
        nspks_at_failure: KS2 only; spikes-per-batch at the failing
            template-optimization step.
        log_path: Sorter log file carrying the full trace when located.
        sorter: Short identifier of the sorter that raised
            (``"kilosort2"``, ``"kilosort4"``, ``"rt_sort"``).
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


class EmptyWaveformMetricsError(BiologicalSortFailure, ValueError):
    """Waveform metrics (SNR, std-norm) cannot be computed.

    Raised when curation requests a waveform-based metric but no
    precomputed values exist and ``raw_data`` on the ``SpikeData`` is
    empty, so there is nothing to extract waveforms from.

    This is biology-adjacent: it typically means the upstream sorter
    produced units that have no usable waveform evidence attached, or
    that the pipeline skipped the waveform-extraction stage. Callers
    should treat it as "cannot curate this target" rather than retry.

    Inherits from both :class:`BiologicalSortFailure` (for
    category-aware handling) and :class:`ValueError` (for backward
    compatibility with callers that historically caught ``ValueError``
    from this site).

    Attributes:
        metric_name: The metric that could not be computed.
    """

    def __init__(self, message: str, *, metric_name: Optional[str] = None):
        super().__init__(message)
        self.metric_name = metric_name


# ---------------------------------------------------------------------------
# Environment failures
# ---------------------------------------------------------------------------


class ConcurrentSortError(EnvironmentSortFailure):
    """Another sort is already in progress on the same intermediate folder.

    Raised by
    :func:`spikelab.spike_sorting.guards.acquire_sort_lock` when a
    pre-existing lock file points at an alive PID on the same host.
    Two concurrent sorts against the same intermediate folder would
    corrupt each other's binary artefacts (KS2 ``.dat`` file,
    RT-Sort scaled traces, curation cache), so the second sort
    fails fast rather than racing.

    Recommended remediation: wait for the running sort to finish,
    or point the second sort at a different ``intermediate_folders``
    path. If you believe the holder is dead but the lock persists,
    delete ``<inter_path>/.spikelab_sort.lock`` by hand.

    Attributes:
        lock_path: Path to the lock file that triggered the abort.
        holder_pid: PID listed in the lock file (when readable).
        holder_hostname: Hostname listed in the lock file (when
            readable).
        started_at: ISO timestamp recorded when the holder acquired
            the lock.
    """

    def __init__(
        self,
        message: str,
        *,
        lock_path: Optional[Path] = None,
        holder_pid: Optional[int] = None,
        holder_hostname: Optional[str] = None,
        started_at: Optional[str] = None,
    ):
        super().__init__(message)
        self.lock_path = lock_path
        self.holder_pid = holder_pid
        self.holder_hostname = holder_hostname
        self.started_at = started_at


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


class ModelLoadingError(EnvironmentSortFailure):
    """Detection model could not be loaded or is unusable.

    Raised when RT-Sort's ``ModelSpikeSorter.load()`` fails — typically
    because PyTorch is missing, weights are corrupt, the model folder
    does not exist, or the architecture parameters do not match the
    saved state dict.

    Attributes:
        model_path: Path that was attempted, when known.
        sorter: Short identifier of the sorter that raised.
    """

    def __init__(
        self,
        message: str,
        *,
        sorter: str = "rt_sort",
        model_path: Optional[str] = None,
    ):
        super().__init__(message)
        self.sorter = sorter
        self.model_path = model_path


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


class SorterTimeoutError(ResourceSortFailure):
    """The sorter subprocess produced no output for too long.

    Raised by
    :class:`spikelab.spike_sorting.guards.LogInactivityWatchdog` when
    the sorter's log file has not been updated within the configured
    inactivity tolerance. Distinct from a hard wall-clock timeout:
    this fires only when the sort has stopped making progress (no log
    writes), so legitimate long sorts on dense MEAs / multi-hour
    recordings are not falsely killed.

    Recommended remediation: skip the recording and continue. Retrying
    without intervention will likely hang again at the same stage.
    Investigate the sorter log up to the inactivity point for the
    proximate cause (CUDA hang, MATLAB JVM deadlock, mex kernel
    failure mode, disk-full stall).

    Attributes:
        sorter: Short identifier of the sorter that hung.
        inactivity_s: Configured inactivity tolerance at the time of
            the trip, in seconds.
        log_path: Path to the sorter log file the watchdog was
            polling, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        sorter: str,
        inactivity_s: Optional[float] = None,
        log_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.sorter = sorter
        self.inactivity_s = inactivity_s
        self.log_path = log_path


class DiskExhaustionError(ResourceSortFailure):
    """Free disk space crossed the watchdog abort threshold mid-sort.

    Raised by :class:`spikelab.spike_sorting.guards.DiskUsageWatchdog`
    when ``shutil.disk_usage(folder).free`` drops below the configured
    abort threshold while a sort is in progress. RT-Sort especially
    can fill a volume mid-run by writing scaled traces, model traces,
    and model outputs as large ``.npy`` files.

    The exception carries a :class:`DiskExhaustionReport` describing
    free space, projected need, top disk consumers in the watched
    folder, and suggested operator actions.

    Recommended remediation: free disk space (or shorten the
    recording window via ``RTSortConfig.recording_window_ms`` /
    ``first_n_mins``) and rerun. The report's ``top_consumers`` field
    flags the largest existing files in the watched folder so the
    operator can clean up safely.

    Attributes:
        folder: The folder whose free space crossed the threshold.
        free_gb_at_trip: Free space (GB) at the moment of the trip.
        abort_threshold_gb: Configured abort threshold (GB).
        report: Optional :class:`DiskExhaustionReport` with the full
            diagnostic payload. ``None`` only when the report could
            not be assembled (e.g. ``os.walk`` failed).
    """

    def __init__(
        self,
        message: str,
        *,
        folder: Optional[Path] = None,
        free_gb_at_trip: Optional[float] = None,
        abort_threshold_gb: Optional[float] = None,
        report: Optional[Any] = None,
    ):
        super().__init__(message)
        self.folder = folder
        self.free_gb_at_trip = free_gb_at_trip
        self.abort_threshold_gb = abort_threshold_gb
        self.report = report


class GpuMemoryWatchdogError(ResourceSortFailure):
    """GPU VRAM crossed the watchdog abort threshold mid-sort.

    Raised by :class:`spikelab.spike_sorting.guards.GpuMemoryWatchdog`
    when free VRAM on the device-in-use drops below the configured
    abort threshold (or used VRAM crosses the abort percentage). Sharp
    GPU OOMs typically come from PyTorch allocator fragmentation
    rather than a clean ``cudaMalloc`` failure, so a percentage-based
    early warning lets the pipeline trigger the existing OOM-retry
    path with a reduced batch *before* the next allocation hits the
    wall.

    Recommended remediation: rerun with reduced sorter batch params
    (the existing OOM-retry path handles this automatically through
    ``GPUOutOfMemoryError`` classification, which this exception
    subclasses-by-symmetry — both surface as ``oom_gpu`` status).

    Attributes:
        device_index: Index of the GPU device that crossed the
            threshold.
        used_pct_at_trip: GPU memory used percentage at the moment
            of the trip.
        abort_pct: Configured abort percentage threshold.
    """

    def __init__(
        self,
        message: str,
        *,
        device_index: Optional[int] = None,
        used_pct_at_trip: Optional[float] = None,
        abort_pct: Optional[float] = None,
    ):
        super().__init__(message)
        self.device_index = device_index
        self.used_pct_at_trip = used_pct_at_trip
        self.abort_pct = abort_pct


class IOStallError(ResourceSortFailure):
    """Disk I/O stalled mid-sort.

    Raised by
    :class:`spikelab.spike_sorting.guards.IOStallWatchdog` when
    ``psutil.disk_io_counters()`` for the watched volume shows no
    byte-counter movement for the configured tolerance — typical
    of a hung NFS / SMB / S3-fuse mount that's still accepting
    file handles but not actually reading or writing.

    The inactivity watchdog catches some I/O stalls (no log
    output → trip), but a sorter that keeps logging while waiting
    for I/O can defeat that signal. The I/O stall watchdog adds a
    second layer specifically targeting kernel-level read/write
    progress.

    Attributes:
        device: Volume identifier (e.g. ``"sda1"``, ``"C:"``).
        stall_s: Configured stall tolerance at the time of the trip.
    """

    def __init__(
        self,
        message: str,
        *,
        device: Optional[str] = None,
        stall_s: Optional[float] = None,
    ):
        super().__init__(message)
        self.device = device
        self.stall_s = stall_s


class HostMemoryWatchdogError(ResourceSortFailure):
    """Host RAM pressure exceeded the watchdog abort threshold.

    Raised by :class:`spikelab.spike_sorting.guards.HostMemoryWatchdog`
    when ``psutil.virtual_memory().percent`` crosses the configured
    abort percentage. Distinct from a Python ``MemoryError`` (which
    fires on a failed allocation): this signals impending host-level
    thrash before any individual allocation has hit a wall, so the
    pipeline can skip the current recording and let the workstation
    recover.

    Recommended remediation: skip the current recording, free
    references and call ``gc.collect()``/``torch.cuda.empty_cache()``,
    then continue with the next recording. Investigate the recording
    that tripped the trigger — long durations, very high unit counts,
    or oversized intermediate buffers are common causes.

    Attributes:
        percent_at_trip: ``psutil`` system memory percentage at the
            moment the watchdog tripped.
        abort_pct: Configured abort threshold.
    """

    def __init__(
        self,
        message: str,
        *,
        percent_at_trip: Optional[float] = None,
        abort_pct: Optional[float] = None,
    ):
        super().__init__(message)
        self.percent_at_trip = percent_at_trip
        self.abort_pct = abort_pct
