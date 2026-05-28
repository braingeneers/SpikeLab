"""Centralised one-shot warning for missing :mod:`psutil`.

Several guards in :mod:`spikelab.spike_sorting` degrade to a no-op
when ``psutil`` is unavailable. The host-memory watchdog emits its
own warning at entry, but the preflight available-RAM check and the
I/O-stall watchdog were previously silent — leaving operators with
no signal that those guards were inactive.

This module provides a single latched warning so that each distinct
*purpose* warns at most once per process. Polling sites (e.g.
``_io_stall._read_io_bytes`` called every ``poll_interval_s``) can
therefore call :func:`warn_psutil_missing_once` on every miss
without flooding the log.
"""

from __future__ import annotations

import logging
from typing import Set

_warned: Set[str] = set()


def warn_psutil_missing_once(logger: logging.Logger, purpose: str) -> None:
    """Emit a single ``WARNING`` about missing :mod:`psutil`.

    Each unique ``purpose`` warns at most once per process, regardless
    of how many call sites share that purpose. The ``logger`` argument
    is used so the warning is attributed to the calling module in log
    output, but it is not part of the dedup key — two call sites in
    different modules that share a ``purpose`` still warn only once
    combined.

    Parameters:
        logger (logging.Logger): Logger to emit the warning through.
        purpose (str): Short human-readable description of the feature
            that is disabled (e.g. ``"I/O stall watchdog"``). Used as
            the dedup key.
    """
    if purpose in _warned:
        return
    _warned.add(purpose)
    logger.warning(
        "psutil not installed — %s disabled. "
        "Install psutil (pip install psutil) to enable.",
        purpose,
    )


def _reset_for_tests() -> None:
    """Clear the dedup latch. Intended for use only by the test suite."""
    _warned.clear()
