"""Pipeline canary — short-window smoke test before each full sort.

A multi-hour KS2 / KS4 sort can fail at the MEX / preprocessing stage
hours into the run because of an architecture incompatibility, missing
CUDA kernel, broken Docker image, or any other "first-time" failure
that only manifests once the sorter actually starts processing data.
The fix that already happened to the configured environment is the
goal of the preflight in :mod:`.guards._preflight`; the canary covers
the residual: failures that need the data + the code paths to actually
run end-to-end before they show themselves.

Operation
---------
When ``ExecutionConfig.canary_first_n_s > 0``,
:func:`run_canary` clones the live config, restricts the recording
window to ``[0, canary_first_n_s]`` seconds, turns off post-sort
exporters and figure generation, relaxes curation so a low-yield
window does not falsely fail, and runs the same backend on the same
recording into ``<inter_path>/_canary/``. The MEX compile, model
loads, Docker image start, and the sorter's first preprocessing pass
all execute under realistic conditions — that is the point.

Failure handling
----------------
* **Classified failure** (``InsufficientActivityError``,
  ``BiologicalSortFailure``, ``EnvironmentSortFailure``,
  ``ResourceSortFailure``): returned to the caller so the full sort
  is aborted before launch and the canary's exception is propagated
  as the recording's classified result.
* **Unexpected failure** (anything else, including the canary itself
  hitting OOM or running out of disk): logged but **not** propagated —
  the canary is a smoke test, not a hard gate. The full sort proceeds
  and the live watchdogs handle resource-shaped issues at runtime.

Edge cases
----------
* Recording shorter than ``canary_first_n_s`` → skip is performed by
  the call site in :mod:`spikelab.spike_sorting.pipeline` (where the
  recording duration is known), not inside :func:`run_canary`. If the
  caller invokes :func:`run_canary` on a too-short recording, the
  backend's slicing clamps to the actual duration and the canary
  effectively runs against the whole recording.
* Canary leaves a small amount of intermediate state under
  ``<inter_path>/_canary/``; cleanup is best-effort on success.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import _globals
from ._exceptions import CLASSIFIED_FAILURES as _CLASSIFIED_FAILURES

_logger = logging.getLogger(__name__)


def _snapshot_pipeline_globals() -> dict:
    """Capture the current state of all uppercase-named globals in
    :mod:`._globals`. The canary backend mutates these via
    ``_sync_globals_from_config`` to apply its overrides; without a
    matching restore the next sort would inherit canary values
    (curation flags off, narrowed time window, etc.) since the full
    backend's ``_sync_globals`` is only called at ``__init__`` and
    not before the next ``load_recording`` / ``sort``.
    """
    snapshot: dict = {}
    for name in vars(_globals):
        if name.startswith("__"):
            continue
        if not (name[:1].isupper() or name == "_REC_CHUNK_NAMES"):
            continue
        value = getattr(_globals, name)
        if isinstance(value, list):
            snapshot[name] = list(value)
        elif isinstance(value, dict):
            snapshot[name] = dict(value)
        else:
            snapshot[name] = value
    return snapshot


def _restore_pipeline_globals(snapshot: dict) -> None:
    """Restore globals from a snapshot taken by
    :func:`_snapshot_pipeline_globals`. Mirrors the original list /
    dict identities are not preserved (callers should treat the
    globals as opaque values), but the values are correct."""
    for name, value in snapshot.items():
        setattr(_globals, name, value)


def _build_canary_config(config: Any, canary_window_s: float) -> Any:
    """Return a config clone tuned for a short smoke-test sort.

    The clone preserves everything that affects whether the sort can
    *start* (sorter selection, sorter params, MEX compile, Docker
    image, model paths, recording loader settings) and turns off
    everything that only matters for the *output* of a real sort
    (post-sort exporters, figures, the per-recording report, the
    second preflight pass that already ran).

    Curation is disabled because a tiny window legitimately yields
    too few units, and a curation-driven empty result would surface
    as a false-positive canary failure.

    Parameters:
        config (SortingPipelineConfig): The original pipeline config.
        canary_window_s (float): Seconds of the recording the canary
            should sort.

    Returns:
        canary_config (SortingPipelineConfig): Deep copy with canary
            overrides applied.
    """
    # Scale the inactivity baseline with the canary window: a 30s
    # smoke test should not wait the same 5 min as a multi-hour sort
    # before flagging a hang. The floor (120s) absorbs Docker / MEX
    # cold-start; the ceiling (300s) caps the timeout at the original
    # full-sort default for unusually large windows.
    canary_inactivity_base_s = min(300.0, max(120.0, 4.0 * canary_window_s))
    overrides = {
        # Restrict to the leading window; clear any rec_chunks that
        # would otherwise force the loader into a multi-segment path.
        # Note: ``start_time_s`` / ``end_time_s`` are allowed to
        # coexist with the per-file rec_chunks that directory
        # concatenation auto-populates — the loader treats time
        # slicing as an explicit override of those auto-populated
        # boundaries (see ``load_recording`` and the
        # ``REC_CHUNKS_FROM_CONCAT`` flag).
        "start_time_s": 0.0,
        "end_time_s": float(canary_window_s),
        "rec_chunks": [],
        "rec_chunks_s": [],
        # Skip curation — too few units in a 30 s window is normal.
        "curate_first": False,
        "curate_second": False,
        # ``tee_log_policy="keep"`` preserves the canary's tee log
        # for debugging. The tee log lives outside the per-pid
        # ``_canary_<pid>/`` folder, so ``_wipe_canary_folder`` does
        # not clean it up — operators should sweep stale canary tee
        # logs from the parent results folder periodically.
        # Skip post-sort exporters — canary outputs are discarded.
        "compile_single_recording": False,
        "compile_to_mat": False,
        "compile_to_npz": False,
        "compile_waveforms": False,
        "save_raw_pkl": False,
        "save_dl_data": False,
        # Skip figures and the human-readable report.
        "create_figures": False,
        "create_unit_figures": False,
        "generate_sorting_report": False,
        # Avoid double-running preflight (the outer caller already
        # ran it) and keep the canary log file in place so a failure
        # is debuggable.
        "preflight": False,
        "tee_log_policy": "keep",
        "sorter_inactivity_base_s": canary_inactivity_base_s,
    }
    return config.override(**overrides)


def _wipe_canary_folder(folder: Path, *, strict: bool = False) -> None:
    """Best-effort cleanup of the canary's intermediate folder.

    Parameters:
        folder (Path): Folder to wipe.
        strict (bool): When True, raises if the wipe fails. Used at
            entry-time so a permission-denied wipe is surfaced
            rather than silently running the canary against a
            partially-cleaned folder. Defaults to False (best-effort
            cleanup) for the post-canary cleanup call site.
    """
    try:
        if folder.exists():
            if strict:
                # ignore_errors=False — any failure raises so the
                # caller can decide how to surface it.
                shutil.rmtree(folder)
            else:
                shutil.rmtree(folder, ignore_errors=True)
    except Exception as exc:
        if strict:
            raise
        _logger.warning("cleanup of %s failed: %r", folder, exc)


def run_canary(
    config: Any,
    recording: Any,
    rec_path: Any,
    inter_path: Any,
    *,
    sorter_name: Optional[str] = None,
    rec_name: str = "canary",
    rng: Any = None,
) -> Optional[BaseException]:
    """Run a short-window smoke test of the configured backend.

    Builds a canary clone of *config* (see :func:`_build_canary_config`),
    spins up a fresh backend instance against that clone, and invokes
    :func:`spikelab.spike_sorting.pipeline.process_recording` against
    a ``<inter_path>/_canary/`` subdirectory.

    Parameters:
        config (SortingPipelineConfig): Live pipeline configuration.
            Read but never mutated.
        recording: Pre-loaded ``BaseRecording`` for the canary, or
            ``None`` when only a path is available.
        rec_path: Path to the recording on disk. Used by the backend
            loader when *recording* is ``None``.
        inter_path: The recording's intermediate folder. The canary
            writes under a ``_canary`` sub-folder so the real sort's
            artefacts are untouched.
        sorter_name (str or None): Override the sorter resolved from
            ``config.sorter.sorter_name``. Mostly used by tests.
        rec_name (str): Short identifier for the canary in log
            output.
        rng (np.random.Generator or None): Optional RNG passed
            through to ``process_recording`` for reproducibility.

    Returns:
        result (BaseException or None): A classified exception when
            the canary discovered a failure the full sort would also
            have hit; ``None`` when the canary succeeded *or* when
            the canary itself hit an unexpected non-classified
            failure (which the live watchdogs are responsible for
            during the real run).
    """
    canary_window_s = float(getattr(config.execution, "canary_first_n_s", 0.0))
    # NaN comparisons are always False, so a NaN window would skip the
    # ``<= 0`` guard and proceed to build a meaningless canary with
    # ``end_time_s=NaN``. Treat NaN the same as "disabled".
    if math.isnan(canary_window_s) or canary_window_s <= 0:
        return None

    # Snapshot the live ``_globals`` state. The canary backend's
    # ``__init__`` (below) re-syncs globals from the canary config —
    # narrower curation, no figures, etc. — and the next sort's
    # already-initialised backend does not re-sync before
    # ``load_recording``. Without this snapshot/restore the canary's
    # values would silently bleed into the full sort.
    globals_snapshot = _snapshot_pipeline_globals()
    try:
        canary_config = _build_canary_config(config, canary_window_s)
        # Per-pid subfolder so two direct callers of run_canary
        # against the same inter_path cannot race on the wipe +
        # mkdir. The standard pipeline flow already serialises via
        # acquire_sort_lock on inter_path, but run_canary is also
        # exposed as a public function and direct callers have no
        # such protection.
        canary_root = Path(inter_path) / f"_canary_{os.getpid()}"
        # Strict wipe at entry — running the canary against a
        # partially-cleaned folder could mask sorter behaviour. The
        # cleanup-phase wipe at exit stays best-effort.
        _wipe_canary_folder(canary_root, strict=True)
        canary_root.mkdir(parents=True, exist_ok=True)
        canary_inter = canary_root / "inter"
        canary_results = canary_root / "results"

        sorter = sorter_name or getattr(config.sorter, "sorter_name", "")
        _logger.info(
            "running %.1fs smoke test for %s via %s",
            canary_window_s,
            rec_name,
            sorter,
        )

        started_t = time.monotonic()
        try:
            from .backends import get_backend_class
            from .pipeline import process_recording

            backend_cls = get_backend_class(sorter)
            canary_backend = backend_cls(canary_config)

            result = process_recording(
                canary_backend,
                canary_config,
                rec_name,
                rec_path,
                canary_inter,
                canary_results,
                rec_loaded=recording,
                rec_chunks=None,
                rec_chunk_names=None,
                rng=rng,
            )
        except _CLASSIFIED_FAILURES as exc:
            _logger.warning("classified failure: %s: %s", type(exc).__name__, exc)
            _wipe_canary_folder(canary_root)
            return exc
        except (KeyboardInterrupt, SystemExit):
            # User abort or watchdog interrupt — never swallow.
            # Clean up the canary folder and let the interrupt
            # propagate so the outer pipeline tears down promptly.
            _wipe_canary_folder(canary_root)
            raise
        except Exception as exc:
            # Unexpected failure — the canary is a smoke test, not
            # a hard gate. Log and let the full sort proceed; live
            # watchdogs handle resource-shaped issues at runtime.
            _logger.warning(
                "non-classified failure (%s: %s); proceeding with the " "full sort.",
                type(exc).__name__,
                exc,
            )
            _wipe_canary_folder(canary_root)
            return None

        if isinstance(result, _CLASSIFIED_FAILURES):
            _logger.warning("classified failure: %s: %s", type(result).__name__, result)
            _wipe_canary_folder(canary_root)
            return result
        if isinstance(result, (KeyboardInterrupt, SystemExit)):
            # process_recording returned the interrupt as a value
            # rather than raising; surface it the same way as the
            # raised path.
            _wipe_canary_folder(canary_root)
            raise result
        if isinstance(result, BaseException):
            _logger.warning(
                "non-classified failure (%s: %s); proceeding with the " "full sort.",
                type(result).__name__,
                result,
            )
            _wipe_canary_folder(canary_root)
            return None

        elapsed_s = time.monotonic() - started_t
        n_units = _extract_unit_count(result)
        if n_units is None:
            _logger.info(
                "passed in %.1fs; proceeding with the full sort.",
                elapsed_s,
            )
        else:
            _logger.info(
                "passed: produced %d unit(s) in %.1fs; proceeding with "
                "the full sort.",
                n_units,
                elapsed_s,
            )
        _wipe_canary_folder(canary_root)
        return None
    finally:
        _restore_pipeline_globals(globals_snapshot)


def _extract_unit_count(result: Any) -> Optional[int]:
    """Best-effort unit count from a ``process_recording`` success result.

    ``process_recording`` returns either a single ``SpikeData`` or a
    ``(sd, sd_curated)`` tuple depending on
    ``ExecutionConfig.save_raw_pkl``. When neither shape matches, or
    when the object lacks ``N`` (number of units), returns ``None``
    so the caller falls back to a unit-count-less log line.
    """
    candidate = result
    if isinstance(result, tuple) and result:
        # Prefer the curated SpikeData if present (last entry).
        candidate = result[-1]
    n = getattr(candidate, "N", None)
    # Accept numpy integer types (np.int64, etc.) as well as Python int.
    # SpikeData.N is sometimes assigned from numpy operations such as
    # np.unique(...).size, which returns a numpy scalar.
    if isinstance(n, (int, np.integer)):
        return int(n)
    return None
