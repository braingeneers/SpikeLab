"""Spike-sorting exceptions shared across runners.

Keeping these in a sorter-neutral module (rather than on ``ks2_runner`` or
``ks4_runner``) lets callers import a single symbol regardless of which
sorter produced the failure and lets new runners reuse the class without
a circular import.
"""

from pathlib import Path
from typing import Optional


class InsufficientActivityError(RuntimeError):
    """Sorting failed because the recording has too little spiking activity.

    Both Kilosort2 and Kilosort4 fall over on near-silent recordings, but in
    different ways:

    * **Kilosort2:** mex kernels launch with degenerate grid/block
      configurations when the template count and per-batch spike counts
      approach zero. Pre-Blackwell GPUs tolerated these launches; newer
      architectures (RTX 50-series, compute capability 12.x) reject them
      with ``CUDA error: invalid configuration argument`` during template
      optimization.
    * **Kilosort4:** sklearn's ``TruncatedSVD`` rejects an empty feature
      matrix, or ``KMeans`` fails the ``n_samples >= n_clusters`` check,
      when the initial spike-detection pass finds ~no events.

    All of these are biology (empty/near-silent well), not a tooling fault —
    the correct handling is to flag the well as insufficiently active and
    move on, not to retry with different parameters or swap sorters.

    Attributes:
        threshold_crossings: KS2-only; count of detected threshold crossings
            parsed from ``kilosort2.log``. ``None`` for KS4.
        units_at_failure: KS2 template count at crash, or KS4 ``n_samples``
            when KMeans complained about too few samples. ``None`` when
            the log didn't expose the value.
        nspks_at_failure: KS2-only; spikes-per-batch at the failing
            template-optimization step. ``None`` for KS4.
        log_path: Path to the sorter log file carrying the full trace,
            when one was located.
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
