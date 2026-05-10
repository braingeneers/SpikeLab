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

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import _globals
from ._classifier import classify_rt_sort_failure
from ._exceptions import SpikeSortingClassifiedError
from .config import SortingPipelineConfig
from .sorting_utils import Stopwatch, Tee, print_stage


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
    *,
    config: Optional[SortingPipelineConfig] = None,
) -> Any:
    """Run RT-Sort offline spike sorting on a single recording.

    Executes the two-stage RT-Sort pipeline:
      1. ``detect_sequences`` — trains sequences from the recording by
         running the DL detection model, clustering codetections, and
         merging preliminary sequences.
      2. ``RTSort.sort_offline`` — assigns spikes in the recording to
         the detected sequences.

    Reads RT-Sort parameters from *config* (when provided) or from
    the legacy module-level globals in ``_globals.py`` (populated by
    ``RTSortBackend._sync_globals``). The serialized ``RTSort`` object
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
        config (SortingPipelineConfig or None): Pipeline configuration.
            When ``None``, falls back to the legacy module-level
            globals (``RECOMPUTE_SORTING``, ``RT_SORT_MODEL_PATH``,
            ``RT_SORT_DEVICE``, ``RT_SORT_NUM_PROCESSES``,
            ``RT_SORT_RECORDING_WINDOW_MS``,
            ``RT_SORT_DETECTION_WINDOW_S``, ``RT_SORT_VERBOSE``,
            ``RT_SORT_DELETE_INTER``, ``RT_SORT_PARAMS``,
            ``RT_SORT_SAVE_PICKLE``) and emits a ``DeprecationWarning``.

    Returns:
        sorting: A SpikeInterface ``NumpySorting`` with one unit per
            detected sequence, or the caught exception if sorting
            failed.
    """
    from .rt_sort import detect_sequences

    if config is None:
        warnings.warn(
            "rt_sort_runner.spike_sort called without an explicit "
            "`config`; falling back to the legacy module-level globals "
            "in spikelab.spike_sorting._globals. Pass "
            "config=<SortingPipelineConfig> to silence this warning. "
            "The legacy path will be removed once the _globals.py "
            "refactor lands (see iat/TO_IMPLEMENT.md).",
            DeprecationWarning,
            stacklevel=2,
        )
        recompute_sorting = _globals.RECOMPUTE_SORTING
        rt_model_path = _globals.RT_SORT_MODEL_PATH
        rt_device = _globals.RT_SORT_DEVICE
        rt_num_processes = _globals.RT_SORT_NUM_PROCESSES
        sort_window_ms = _globals.RT_SORT_RECORDING_WINDOW_MS
        det_window_s = _globals.RT_SORT_DETECTION_WINDOW_S
        rt_verbose = _globals.RT_SORT_VERBOSE
        rt_delete_inter = _globals.RT_SORT_DELETE_INTER
        # Mirror RTSortBackend._sync_globals: the live RT_SORT_PARAMS
        # already contains the merged probe key, so we use it directly.
        rt_params = dict(_globals.RT_SORT_PARAMS or {})
        save_rt_sort_pickle = _globals.RT_SORT_SAVE_PICKLE
    else:
        rts = config.rt_sort
        recompute_sorting = config.execution.recompute_sorting
        rt_model_path = rts.model_path
        rt_device = rts.device
        rt_num_processes = rts.num_processes
        sort_window_ms = rts.recording_window_ms
        det_window_s = rts.detection_window_s
        rt_verbose = rts.verbose
        rt_delete_inter = rts.delete_inter
        # Mirror RTSortBackend._sync_globals: merge probe into params
        # so the runner sees both as a single dict.
        rt_params = {"probe": rts.probe}
        if rts.params:
            rt_params.update(rts.params)
        save_rt_sort_pickle = rts.save_rt_sort_pickle

    print_stage("SPIKE SORTING WITH RT-SORT")
    stopwatch = Stopwatch()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    log_path = output_folder / "rt_sort.log"
    # rt_sort.pickle is saved to the recording dir (parent of inter_path),
    # NOT inside the inter dir. This keeps it persistent across spikelab's
    # inter cleanup (RTSortConfig.delete_inter=True default would otherwise
    # remove the pickle along with the cache npy files). Stable path used
    # for the cache-hit reuse check below and for sort_stim_recording reuse.
    rt_sort_pickle = output_folder.parent.parent / "rt_sort.pickle"
    cached_sorting_npz = output_folder / "sorting.npz"

    with Tee(log_path, file_mode="w"):
        # Reuse cached results when recompute is not forced
        if (
            not recompute_sorting
            and rt_sort_pickle.exists()
            and cached_sorting_npz.exists()
        ):
            print("Loading existing RT-Sort results")
            try:
                sorting = _load_cached_sorting(cached_sorting_npz, rec_cache)
                root_elecs_path = output_folder / "root_elecs.npy"
                root_elecs = (
                    list(np.load(str(root_elecs_path)))
                    if root_elecs_path.exists()
                    else None
                )
                stopwatch.log_time("Done loading existing results.")
                return sorting, root_elecs
            except Exception as exc:
                print(f"Failed to load cached sorting ({exc}); recomputing.")

        try:
            detection_model = _load_detection_model(
                rt_model_path,
                probe=rt_params.get("probe", "mea"),
            )

            # Resolve the detection window.  If the user set
            # ``detection_window_s``, it narrows the window used *only*
            # during sequence detection — ``sort_offline`` below still uses
            # the full ``recording_window_ms`` so every spike in
            # the recording is assigned to one of the detected sequences.
            if det_window_s is not None:
                start_ms = sort_window_ms[0] if sort_window_ms is not None else 0.0
                detect_window_ms = (
                    start_ms,
                    start_ms + float(det_window_s) * 1000.0,
                )
                if rt_verbose:
                    print(
                        f"[rt_sort] Detection window narrowed to "
                        f"{detect_window_ms[0]/1000:.1f}-"
                        f"{detect_window_ms[1]/1000:.1f} s "
                        f"(sort_offline still covers full recording)."
                    )
            else:
                detect_window_ms = sort_window_ms

            # Assemble the detect_sequences kwargs from resolved
            # params + override dict. Override dict wins.
            ds_kwargs = dict(
                recording_window_ms=detect_window_ms,
                device=rt_device,
                num_processes=rt_num_processes,
                delete_inter=rt_delete_inter,
                verbose=rt_verbose,
            )
            param_overrides = dict(rt_params)
            param_overrides.pop("probe", None)  # consumed above
            ds_kwargs.update(param_overrides)

            rt_sort = detect_sequences(
                recording=rec_cache,
                inter_path=output_folder,
                detection_model=detection_model,
                **ds_kwargs,
            )
        except SpikeSortingClassifiedError:
            raise
        except Exception as exc:
            print(f"RT-Sort sequence detection failed: {exc}")
            stopwatch.log_time("Sequence detection failed.")
            classified = classify_rt_sort_failure(output_folder, exc)
            if classified is not None:
                raise classified from exc
            return exc

        try:
            sorting = rt_sort.sort_offline(
                recording=rec_cache,
                inter_path=output_folder,
                recording_window_ms=sort_window_ms,
                return_spikeinterface_sorter=True,
                verbose=rt_verbose,
            )
        except SpikeSortingClassifiedError:
            raise
        except Exception as exc:
            print(f"RT-Sort offline sorting failed: {exc}")
            stopwatch.log_time("Offline sorting failed.")
            classified = classify_rt_sort_failure(output_folder, exc)
            if classified is not None:
                raise classified from exc
            return exc

        # Persist the trained sequences for Phase 2 reuse.
        # RTSort.save() strips the unpicklable compiled model before
        # serialization and restores it in-memory afterward.
        # The pickle is saved to the recording dir (parent of inter_path) so it
        # survives RTSortConfig.delete_inter=True. We also copy compiled.ts
        # alongside so load_rt_sort() can auto-detect the model on reload
        # (it looks for `pickle_path.parent / "compiled.ts"`).
        if save_rt_sort_pickle:
            rt_sort.save(rt_sort_pickle)
            compiled_src = output_folder / "compiled.ts"
            compiled_dst = rt_sort_pickle.parent / "compiled.ts"
            if compiled_src.exists() and not compiled_dst.exists():
                import shutil as _shutil

                _shutil.copy2(compiled_src, compiled_dst)

        # Cache the sorting for fast reload on subsequent runs
        _save_sorting_cache(sorting, cached_sorting_npz)

        # Save root electrodes for the KilosortSortingExtractor conversion
        root_elecs = list(rt_sort._seq_root_elecs)
        np.save(str(output_folder / "root_elecs.npy"), np.array(root_elecs))

        stopwatch.log_time("Done sorting with RT-Sort.")
        return sorting, root_elecs


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


def load_rt_sort(pickle_path, model=None, model_path=None):
    """Load a saved RTSort object and reattach its detection model.

    ``RTSort.save()`` strips the compiled model before pickling because
    TensorRT/TorchScript modules are not picklable.  This function
    reattaches the model so the returned object is ready for
    ``sort_offline()`` or Phase 2 stim-aware sorting.

    The model can be supplied in three ways (checked in order):

    1. *model* — an already-loaded ``ModelSpikeSorter`` instance.
    2. *model_path* — path to a folder containing ``init_dict.json``
       and ``state_dict.pt`` (loads and compiles fresh).
    3. If neither is given, the function looks for ``compiled.ts`` in
       the same directory as *pickle_path* (the default location where
       ``detect_sequences`` caches the compiled model).

    Parameters:
        pickle_path (str or Path): Path to the ``rt_sort.pickle`` file
            written by ``RTSort.save()``.
        model (ModelSpikeSorter or None): Pre-loaded detection model.
        model_path (str or Path or None): Path to a model folder.

    Returns:
        rt_sort (RTSort): The loaded RTSort object with model attached.
    """
    from .rt_sort._algorithm import RTSort as _RTSort

    pickle_path = Path(pickle_path)

    if model is not None:
        return _RTSort.load_from_file(pickle_path, model=model)

    if model_path is not None:
        return _RTSort.load_from_file(pickle_path, model=model_path)

    # Try the compiled model cached alongside the pickle
    compiled_ts = pickle_path.parent / "compiled.ts"
    if compiled_ts.exists():
        from .rt_sort.model import ModelSpikeSorter

        rt_sort = _RTSort.load_from_file(pickle_path, model=None)
        rt_sort.model = ModelSpikeSorter.load_compiled(pickle_path.parent)
        return rt_sort

    # Fall back to loading without a model — caller must set it later
    return _RTSort.load_from_file(pickle_path, model=None)
