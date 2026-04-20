"""Kilosort4 sorting runner.

Runs Kilosort4 via SpikeInterface's ``run_sorter("kilosort4", ...)``.
Mirrors the structure of ``ks2_runner.py`` for symmetry — backends
should delegate sorting to a dedicated runner module.
"""

import re
from pathlib import Path
from typing import Any, Optional

from . import _globals
from ._exceptions import InsufficientActivityError
from .docker_utils import get_docker_image
from .sorting_extractor import KilosortSortingExtractor
from .sorting_utils import Stopwatch, print_stage


# Kilosort4 surfaces low-activity failures as sklearn ValueError messages
# that bubble out of run_sorter. Two flavours seen in the wild:
#
#   1. TruncatedSVD: spike detection returned no events.
#      "Found array with 0 sample(s) (shape=(0, N)) while a minimum of 1
#       is required by TruncatedSVD."
#
#   2. KMeans: spike detection returned fewer events than the configured
#      n_clusters.
#      "n_samples=3 should be >= n_clusters=6."
#
# Both are biology (well is too young / too quiet to sort), not bugs.
_KS4_SVD_EMPTY_RE = re.compile(
    r"Found array with\s+(\d+)\s+sample\(s\).*?required by TruncatedSVD",
    re.DOTALL,
)
_KS4_KMEANS_RE = re.compile(r"n_samples=(\d+)\s+should be\s+>=\s+n_clusters=(\d+)")


def _classify_ks4_failure(
    original_exception: BaseException, log_path: Optional[Path] = None
) -> Optional[InsufficientActivityError]:
    """Inspect a Kilosort4 exception for the low-activity signature.

    Returns an :class:`InsufficientActivityError` carrying parsed metrics
    when the message matches one of the known "not enough spikes" patterns,
    otherwise ``None`` so the caller keeps the original exception.
    """
    # Walk the chained exceptions so we catch SpikeInterface wrappers too.
    messages: list[str] = []
    current: Optional[BaseException] = original_exception
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__

    text = "\n".join(messages)

    svd_match = _KS4_SVD_EMPTY_RE.search(text)
    kmeans_match = _KS4_KMEANS_RE.search(text)

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
        f"{reason} Original exception: {original_exception!r}."
        + (f" See {log_path} for full trace." if log_path is not None else "")
    )
    return InsufficientActivityError(
        message,
        sorter="kilosort4",
        units_at_failure=n_samples,
        log_path=log_path,
    )


def spike_sort(
    rec_cache: Any,
    rec_path: Any,
    recording_dat_path: Any,
    output_folder: Any,
) -> Any:
    """Run Kilosort4 spike sorting on a single recording.

    Uses ``spikeinterface.sorters.run_sorter("kilosort4", ...)`` which
    handles binary conversion, parameter passing, and result loading.
    When ``_globals.USE_DOCKER`` is truthy, runs in a Docker container
    using an auto-detected image (or a user-supplied image string).

    Parameters:
        rec_cache: Scaled and filtered SpikeInterface recording.
        rec_path: Path to the original recording file (unused, kept
            for interface parity with ``ks2_runner.spike_sort``).
        recording_dat_path: Path to the binary .dat file (unused, kept
            for interface parity).
        output_folder (Path): Directory for Kilosort4 output files.

    Returns:
        sorting: A ``KilosortSortingExtractor`` pointing at the output
            folder, or the caught exception if sorting failed.
    """
    import spikeinterface.sorters as ss

    print_stage("SPIKE SORTING WITH KILOSORT4")
    stopwatch = Stopwatch()

    sorter_params = dict(_globals.KILOSORT_PARAMS)

    output_folder_path = output_folder
    if hasattr(output_folder, "__fspath__") or isinstance(output_folder, str):
        output_folder_path = Path(output_folder)

    # Reuse existing results if present and we're not forced to recompute
    if (
        not _globals.RECOMPUTE_SORTING
        and output_folder_path.exists()
        and (output_folder_path / "spike_times.npy").exists()
    ):
        print("Loading existing Kilosort4 results")
        sorting = KilosortSortingExtractor(folder_path=output_folder_path)
        stopwatch.log_time("Done loading existing results.")
        return sorting

    try:
        docker_kwargs = {}
        if _globals.USE_DOCKER:
            docker_kwargs["docker_image"] = (
                _globals.USE_DOCKER
                if isinstance(_globals.USE_DOCKER, str)
                else get_docker_image("kilosort4")
            )
            # Use "pypi" instead of "no-install" to work around an SI
            # 0.104 bug where extra_requirements triggers an undefined
            # 'cmd' variable when installation_mode="no-install".
            # SI will detect the pre-installed version and skip the install.
            docker_kwargs["installation_mode"] = "pypi"

        ss.run_sorter(
            "kilosort4",
            rec_cache,
            folder=str(output_folder),
            remove_existing_folder=True,
            verbose=True,
            **sorter_params,
            **docker_kwargs,
        )
    except Exception as e:
        ks4_log_path = output_folder_path / "sorter_output" / "kilosort4.log"
        classified = _classify_ks4_failure(
            e, log_path=ks4_log_path if ks4_log_path.is_file() else None
        )
        if classified is not None:
            # Propagate biology signals so callers can distinguish insufficient
            # activity from a real tooling failure without message parsing.
            raise classified from e
        print(f"Kilosort4 sorting failed: {e}")
        stopwatch.log_time("Sorting failed.")
        return e

    # Load results using the shared KilosortSortingExtractor
    # (KS4 output format is compatible: spike_times.npy, spike_clusters.npy)
    sorter_output = output_folder_path
    if (output_folder_path / "sorter_output").exists():
        sorter_output = output_folder_path / "sorter_output"

    sorting = KilosortSortingExtractor(folder_path=sorter_output)
    stopwatch.log_time("Done sorting with Kilosort4.")
    return sorting
