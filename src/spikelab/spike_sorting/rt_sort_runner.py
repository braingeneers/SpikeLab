"""RT-Sort sorting runner.

Runs the RT-Sort offline pipeline (``detect_sequences`` followed by
``RTSort.sort_offline``) and returns a SpikeInterface ``NumpySorting``
object so the rest of the SpikeLab pipeline (waveform extraction,
SpikeData conversion, curation, compilation) can consume it through the
same path used by Kilosort2/4.

Mirrors the structure of ``ks2_runner.py`` and ``ks4_runner.py`` for
symmetry — backends delegate sorting to a dedicated runner module.

The underlying RT-Sort algorithm is vendored in ``rt_sort/`` and is
attributed to van der Molen, Lim et al. 2024 (PLOS ONE, DOI
10.1371/journal.pone.0312438).
"""

from pathlib import Path
from typing import Any

from . import _globals
from .sorting_utils import Stopwatch, print_stage


def _load_detection_model(model_path, probe):
    """Load a pretrained RT-Sort detection model.

    Parameters:
        model_path (str or Path or None): Explicit model folder.  When
            None, the bundled model for ``probe`` is loaded.
        probe (str): ``"mea"`` or ``"neuropixels"``; selects the
            bundled model when ``model_path`` is None.

    Returns:
        model (ModelSpikeSorter): The loaded detection model.
    """
    from .rt_sort.model import ModelSpikeSorter
    from .rt_sort import DEFAULT_MEA_MODEL_PATH, DEFAULT_NEUROPIXELS_MODEL_PATH

    if model_path is not None:
        return ModelSpikeSorter.load(Path(model_path))
    if probe == "mea":
        return ModelSpikeSorter.load(DEFAULT_MEA_MODEL_PATH)
    if probe == "neuropixels":
        return ModelSpikeSorter.load(DEFAULT_NEUROPIXELS_MODEL_PATH)
    raise ValueError(f"Unknown probe {probe!r}; expected 'mea' or 'neuropixels'.")


def spike_sort(
    rec_cache: Any,
    rec_path: Any,
    recording_dat_path: Any,
    output_folder: Any,
) -> Any:
    """Run RT-Sort offline spike sorting on a single recording.

    Executes the two-stage RT-Sort pipeline:
      1. ``detect_sequences`` — trains sequences from the recording by
         running the DL detection model, clustering codetections, and
         merging preliminary sequences.
      2. ``RTSort.sort_offline`` — assigns spikes in the recording to
         the detected sequences.

    Reads RT-Sort parameters from ``_globals`` (populated by
    ``RTSortBackend._sync_globals``).  The serialized ``RTSort`` object
    is optionally written to ``output_folder/rt_sort.pickle`` for reuse
    by the Phase 2 stim-aware sorting pipeline.

    Parameters:
        rec_cache: Scaled and filtered SpikeInterface recording.
        rec_path: Path to the original recording file (used by
            RT-Sort's internal trace caching).
        recording_dat_path: Unused (kept for interface parity with the
            Kilosort runners).
        output_folder (Path): Directory where RT-Sort intermediate
            files and the serialized ``RTSort`` object are stored.

    Returns:
        sorting: A SpikeInterface ``NumpySorting`` with one unit per
            detected sequence, or the caught exception if sorting
            failed.
    """
    from .rt_sort import detect_sequences

    print_stage("SPIKE SORTING WITH RT-SORT")
    stopwatch = Stopwatch()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    rt_sort_pickle = output_folder / "rt_sort.pickle"
    cached_sorting_npz = output_folder / "sorting.npz"

    # Reuse cached results when recompute is not forced
    if (
        not _globals.RECOMPUTE_SORTING
        and rt_sort_pickle.exists()
        and cached_sorting_npz.exists()
    ):
        print("Loading existing RT-Sort results")
        try:
            sorting = _load_cached_sorting(cached_sorting_npz, rec_cache)
            stopwatch.log_time("Done loading existing results.")
            return sorting
        except Exception as exc:
            print(f"Failed to load cached sorting ({exc}); recomputing.")

    try:
        detection_model = _load_detection_model(
            _globals.RT_SORT_MODEL_PATH,
            probe=(_globals.RT_SORT_PARAMS or {}).get("probe", "mea"),
        )

        # Assemble the detect_sequences kwargs from globals + param
        # override dict.  The override dict wins.
        ds_kwargs = dict(
            recording_window_ms=_globals.RT_SORT_RECORDING_WINDOW_MS,
            device=_globals.RT_SORT_DEVICE,
            num_processes=_globals.RT_SORT_NUM_PROCESSES,
            delete_inter=_globals.RT_SORT_DELETE_INTER,
            verbose=_globals.RT_SORT_VERBOSE,
        )
        param_overrides = dict(_globals.RT_SORT_PARAMS or {})
        param_overrides.pop("probe", None)  # consumed above
        ds_kwargs.update(param_overrides)

        rt_sort = detect_sequences(
            recording=rec_cache,
            inter_path=output_folder,
            detection_model=detection_model,
            **ds_kwargs,
        )
    except Exception as exc:
        print(f"RT-Sort sequence detection failed: {exc}")
        stopwatch.log_time("Sequence detection failed.")
        return exc

    try:
        sorting = rt_sort.sort_offline(
            recording=rec_cache,
            inter_path=output_folder,
            recording_window_ms=_globals.RT_SORT_RECORDING_WINDOW_MS,
            return_spikeinterface_sorter=True,
            verbose=_globals.RT_SORT_VERBOSE,
        )
    except Exception as exc:
        print(f"RT-Sort offline sorting failed: {exc}")
        stopwatch.log_time("Offline sorting failed.")
        return exc

    # Persist the trained sequences for Phase 2 reuse
    if _globals.RT_SORT_SAVE_PICKLE:
        import pickle

        with open(rt_sort_pickle, "wb") as f:
            pickle.dump(rt_sort, f)

    # Cache the sorting for fast reload on subsequent runs
    _save_sorting_cache(sorting, cached_sorting_npz)

    stopwatch.log_time("Done sorting with RT-Sort.")
    return sorting


def _save_sorting_cache(sorting, path):
    """Persist a NumpySorting to a .npz file for fast reloading.

    NumpySorting is not directly picklable in a stable, portable form
    across SpikeInterface versions, so we save the per-unit spike
    times (in samples) plus the sampling frequency and rebuild a fresh
    NumpySorting on reload.
    """
    import numpy as np

    unit_ids = sorting.get_unit_ids()
    fs = sorting.get_sampling_frequency()
    data = {"unit_ids": np.asarray(unit_ids), "fs": np.asarray(fs)}
    for uid in unit_ids:
        data[f"u{uid}"] = sorting.get_unit_spike_train(uid)
    np.savez(path, **data)


def _load_cached_sorting(path, recording):
    """Rebuild a NumpySorting from a cached .npz file."""
    import numpy as np
    from spikeinterface.extractors import NumpySorting

    with np.load(path, allow_pickle=True) as data:
        unit_ids = list(data["unit_ids"])
        fs = float(data["fs"])
        spikes_by_unit = {uid: data[f"u{uid}"] for uid in unit_ids}

    return NumpySorting.from_unit_dict([spikes_by_unit], sampling_frequency=fs)
