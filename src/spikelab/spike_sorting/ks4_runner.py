"""Kilosort4 sorting runner.

Runs Kilosort4 via SpikeInterface's ``run_sorter("kilosort4", ...)``.
Mirrors the structure of ``ks2_runner.py`` for symmetry — backends
should delegate sorting to a dedicated runner module.
"""

from pathlib import Path
from typing import Any

from . import _globals
from .docker_utils import get_docker_image
from .sorting_extractor import KilosortSortingExtractor
from .sorting_utils import Stopwatch, print_stage


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
