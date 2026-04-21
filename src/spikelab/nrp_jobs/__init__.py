"""Deprecated: use ``spikelab.batch_jobs`` instead."""

import warnings

warnings.warn(
    "spikelab.nrp_jobs is deprecated; use spikelab.batch_jobs instead. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from spikelab.batch_jobs import *  # noqa: F401, F403
