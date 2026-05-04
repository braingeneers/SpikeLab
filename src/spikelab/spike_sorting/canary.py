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

import math
import shutil
from pathlib import Path
from typing import Any, Optional

from ._exceptions import (
    BiologicalSortFailure,
    EnvironmentSortFailure,
    InsufficientActivityError,
    ResourceSortFailure,
)

# Exceptions whose appearance in the canary indicates the full sort
# would have hit the same wall — propagated to the caller.
_CLASSIFIED_FAILURES: tuple = (
    InsufficientActivityError,
    BiologicalSortFailure,
    EnvironmentSortFailure,
    ResourceSortFailure,
)


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
    overrides = {
        # Restrict to the leading window; clear any rec_chunks that
        # would otherwise force the loader into a multi-segment path.
        "start_time_s": 0.0,
        "end_time_s": float(canary_window_s),
        "rec_chunks": [],
        "rec_chunks_s": [],
        # Skip curation — too few units in a 30 s window is normal.
        "curate_first": False,
        "curate_second": False,
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
        # Half the inactivity baseline so a hung canary doesn't sit
        # for 10 minutes before tripping; still well above the time
        # any non-pathological smoke test needs.
        "sorter_inactivity_base_s": 300.0,
    }
    return config.override(**overrides)


def _wipe_canary_folder(folder: Path) -> None:
    """Best-effort cleanup of the canary's intermediate folder."""
    try:
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    except Exception as exc:
        print(f"[canary] cleanup of {folder!s} failed: {exc!r}")


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

    canary_config = _build_canary_config(config, canary_window_s)
    canary_root = Path(inter_path) / "_canary"
    _wipe_canary_folder(canary_root)
    canary_root.mkdir(parents=True, exist_ok=True)
    canary_inter = canary_root / "inter"
    canary_results = canary_root / "results"

    sorter = sorter_name or getattr(config.sorter, "sorter_name", "")
    print(
        f"[canary] running {canary_window_s:.1f}s smoke test for "
        f"{rec_name} via {sorter}"
    )

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
        print(f"[canary] classified failure: {type(exc).__name__}: {exc}")
        _wipe_canary_folder(canary_root)
        return exc
    except (KeyboardInterrupt, SystemExit):
        # User abort or watchdog interrupt — never swallow. Clean up
        # the canary folder and let the interrupt propagate so the
        # outer pipeline tears down promptly.
        _wipe_canary_folder(canary_root)
        raise
    except Exception as exc:
        # Unexpected failure — the canary is a smoke test, not a
        # hard gate. Log and let the full sort proceed; live
        # watchdogs handle resource-shaped issues at runtime.
        print(
            f"[canary] non-classified failure ({type(exc).__name__}: {exc}); "
            "proceeding with the full sort."
        )
        _wipe_canary_folder(canary_root)
        return None

    if isinstance(result, _CLASSIFIED_FAILURES):
        print(f"[canary] classified failure: {type(result).__name__}: {result}")
        _wipe_canary_folder(canary_root)
        return result
    if isinstance(result, (KeyboardInterrupt, SystemExit)):
        # process_recording returned the interrupt as a value rather
        # than raising; surface it the same way as the raised path.
        _wipe_canary_folder(canary_root)
        raise result
    if isinstance(result, BaseException):
        print(
            f"[canary] non-classified failure "
            f"({type(result).__name__}: {result}); "
            "proceeding with the full sort."
        )
        _wipe_canary_folder(canary_root)
        return None

    print("[canary] passed; proceeding with the full sort")
    _wipe_canary_folder(canary_root)
    return None
