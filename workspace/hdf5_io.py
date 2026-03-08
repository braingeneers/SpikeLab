"""
HDF5 serialization and deserialization for AnalysisWorkspace.

Each workspace is stored in a single .h5 file with the following structure:

    workspace.h5
    ├── {namespace}/                 (group)
    │   └── {key}/                   (group)
    │       ├── __type__             (attr): IAT class name or "ndarray"
    │       ├── __created_at__       (attr): float POSIX timestamp
    │       ├── __note__             (attr, optional): free-text annotation
    │       └── ...                  type-specific datasets and attrs

Supported types
---------------
ndarray, SpikeData, RateData, RateSliceStack, SpikeSliceStack,
PairwiseCompMatrix, PairwiseCompMatrixStack.
"""

import json
import time
from typing import Any, Optional, Tuple

import numpy as np

try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def _require_h5py() -> None:
    if not _H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 workspace I/O. Install with: pip install h5py"
        )


# ===========================================================================
# Public API
# ===========================================================================


def dump_workspace(ws, path: str) -> None:
    """
    Write a full AnalysisWorkspace to an HDF5 file at {path}.h5.

    Parameters:
        ws (AnalysisWorkspace): The workspace to serialise.
        path (str): Base path without file extension.
    """
    _require_h5py()
    h5_path = f"{path}.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["__workspace_id__"] = ws.workspace_id
        f.attrs["__workspace_name__"] = ws.name or ""
        f.attrs["__created_at__"] = ws.created_at
        for ns, keys in ws._items.items():
            ns_grp = f.require_group(ns)
            for key, obj in keys.items():
                key_grp = ns_grp.require_group(key)
                index_entry = ws._index.get(ns, {}).get(key, {})
                created_at = index_entry.get("created_at", time.time())
                note = index_entry.get("note")
                _dump_item(key_grp, obj, created_at, note)


def load_workspace_full(path: str):
    """
    Load a full AnalysisWorkspace from {path}.h5, reconstructing all objects.

    Parameters:
        path (str): Base path without file extension.

    Returns:
        ws (AnalysisWorkspace): Reconstructed workspace with all items restored
            to their original IAT data class types.
    """
    _require_h5py()
    from workspace.workspace import AnalysisWorkspace

    h5_path = f"{path}.h5"
    with h5py.File(h5_path, "r") as f:
        ws = AnalysisWorkspace.__new__(AnalysisWorkspace)
        ws.workspace_id = str(f.attrs["__workspace_id__"])
        name = str(f.attrs["__workspace_name__"])
        ws.name = name if name else None
        ws.created_at = float(f.attrs["__created_at__"])
        ws._items = {}
        ws._index = {}
        for ns in f.keys():
            ns_grp = f[ns]
            ws._items[ns] = {}
            ws._index[ns] = {}
            for key in ns_grp.keys():
                key_grp = ns_grp[key]
                obj, index_entry = _load_item(key_grp)
                ws._items[ns][key] = obj
                ws._index[ns][key] = index_entry
    return ws


def load_workspace_item(path: str, namespace: str, key: str) -> Any:
    """
    Load a single item from a saved workspace HDF5 file, reconstructing its
    original IAT data class.

    Parameters:
        path (str): Base path without file extension.
        namespace (str): Namespace the item was stored under.
        key (str): Key the item was stored under.

    Returns:
        obj: Reconstructed IAT data object or numpy array.
    """
    _require_h5py()
    h5_path = f"{path}.h5"
    with h5py.File(h5_path, "r") as f:
        if namespace not in f:
            raise KeyError(f"Namespace '{namespace}' not found in workspace file.")
        if key not in f[namespace]:
            raise KeyError(f"Key '{key}' not found in namespace '{namespace}'.")
        obj, _ = _load_item(f[namespace][key])
    return obj


def dump_item_to_file(
    h5_path: str,
    namespace: str,
    key: str,
    obj: Any,
    created_at: float,
    note: Optional[str] = None,
) -> None:
    """
    Write a single item to an HDF5 file, creating or overwriting the item group.

    Parameters:
        h5_path (str): Full path to the HDF5 file (including .h5 extension).
        namespace (str): Namespace group to write into.
        key (str): Key group to write into.
        obj: Object to serialise.
        created_at (float): POSIX timestamp for the item.
        note (str | None): Optional annotation.
    """
    _require_h5py()
    with h5py.File(h5_path, "a") as f:
        ns_grp = f.require_group(namespace)
        if key in ns_grp:
            del ns_grp[key]
        key_grp = ns_grp.create_group(key)
        _dump_item(key_grp, obj, created_at, note)


def load_item_from_file(h5_path: str, namespace: str, key: str) -> Any:
    """
    Load a single item from an HDF5 file by its full path.

    Parameters:
        h5_path (str): Full path to the HDF5 file (including .h5 extension).
        namespace (str): Namespace the item was stored under.
        key (str): Key the item was stored under.

    Returns:
        obj: Reconstructed IAT data object or numpy array.
    """
    _require_h5py()
    with h5py.File(h5_path, "r") as f:
        if namespace not in f:
            raise KeyError(f"Namespace '{namespace}' not found in workspace file.")
        if key not in f[namespace]:
            raise KeyError(f"Key '{key}' not found in namespace '{namespace}'.")
        obj, _ = _load_item(f[namespace][key])
    return obj


def delete_item_from_file(
    h5_path: str, namespace: str, key: Optional[str] = None
) -> None:
    """
    Delete a single item or entire namespace from an HDF5 file.

    Parameters:
        h5_path (str): Full path to the HDF5 file (including .h5 extension).
        namespace (str): Namespace to delete from.
        key (str | None): Key to delete. If None, deletes the entire namespace.
    """
    _require_h5py()
    with h5py.File(h5_path, "a") as f:
        if namespace not in f:
            return
        if key is None:
            del f[namespace]
        elif key in f[namespace]:
            del f[namespace][key]


# ===========================================================================
# Item-level dump / load
# ===========================================================================


def _dump_item(grp, obj: Any, created_at: float, note: Optional[str]) -> None:
    """Write one object to an HDF5 group, tagging with __type__ and metadata attrs."""
    try:
        from spikedata.spikedata import SpikeData
    except ImportError:
        SpikeData = None
    try:
        from spikedata.ratedata import RateData
    except ImportError:
        RateData = None
    try:
        from spikedata.rateslicestack import RateSliceStack
    except ImportError:
        RateSliceStack = None
    try:
        from spikedata.spikeslicestack import SpikeSliceStack
    except ImportError:
        SpikeSliceStack = None
    try:
        from spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
    except ImportError:
        PairwiseCompMatrix = None
        PairwiseCompMatrixStack = None

    grp.attrs["__created_at__"] = created_at
    if note is not None:
        grp.attrs["__note__"] = note

    # PairwiseCompMatrixStack must be checked before PairwiseCompMatrix
    if isinstance(obj, np.ndarray):
        grp.attrs["__type__"] = "ndarray"
        _dump_ndarray(grp, obj)
    elif SpikeData is not None and isinstance(obj, SpikeData):
        grp.attrs["__type__"] = "SpikeData"
        _dump_spikedata(grp, obj)
    elif RateData is not None and isinstance(obj, RateData):
        grp.attrs["__type__"] = "RateData"
        _dump_ratedata(grp, obj)
    elif RateSliceStack is not None and isinstance(obj, RateSliceStack):
        grp.attrs["__type__"] = "RateSliceStack"
        _dump_rateslicestack(grp, obj)
    elif SpikeSliceStack is not None and isinstance(obj, SpikeSliceStack):
        grp.attrs["__type__"] = "SpikeSliceStack"
        _dump_spikeslicestack(grp, obj)
    elif PairwiseCompMatrixStack is not None and isinstance(
        obj, PairwiseCompMatrixStack
    ):
        grp.attrs["__type__"] = "PairwiseCompMatrixStack"
        _dump_pairwise_stack(grp, obj)
    elif PairwiseCompMatrix is not None and isinstance(obj, PairwiseCompMatrix):
        grp.attrs["__type__"] = "PairwiseCompMatrix"
        _dump_pairwise(grp, obj)
    else:
        raise TypeError(
            f"Cannot serialise object of type '{type(obj).__name__}' to HDF5. "
            "Supported types: ndarray, SpikeData, RateData, RateSliceStack, "
            "SpikeSliceStack, PairwiseCompMatrix, PairwiseCompMatrixStack."
        )


def _load_item(grp) -> Tuple[Any, dict]:
    """
    Read and reconstruct one object from an HDF5 group.

    Returns:
        obj: Reconstructed IAT data object or numpy array.
        index_entry (dict): Summary metadata for the workspace index.
    """
    from workspace.workspace import _make_summary

    type_tag = str(grp.attrs.get("__type__", ""))
    created_at = float(grp.attrs.get("__created_at__", 0.0))
    note_raw = grp.attrs.get("__note__", None)
    note = str(note_raw) if note_raw is not None else None

    _dispatch = {
        "ndarray": _load_ndarray,
        "SpikeData": _load_spikedata,
        "RateData": _load_ratedata,
        "RateSliceStack": _load_rateslicestack,
        "SpikeSliceStack": _load_spikeslicestack,
        "PairwiseCompMatrixStack": _load_pairwise_stack,
        "PairwiseCompMatrix": _load_pairwise,
    }

    if type_tag not in _dispatch:
        raise ValueError(
            f"Unknown __type__ '{type_tag}' in HDF5 group '{grp.name}'. "
            f"Supported: {list(_dispatch.keys())}"
        )

    obj = _dispatch[type_tag](grp)

    entry = _make_summary(obj)
    entry["created_at"] = created_at
    if note:
        entry["note"] = note

    return obj, entry


# ===========================================================================
# ndarray
# ===========================================================================


def _dump_ndarray(grp, arr: np.ndarray) -> None:
    grp.create_dataset("data", data=arr)


def _load_ndarray(grp) -> np.ndarray:
    return np.array(grp["data"])


# ===========================================================================
# Shared helpers
# ===========================================================================


def _dump_neuron_attributes(grp, neuron_attributes: list) -> None:
    """
    Serialise a list of N per-unit attribute dicts to an HDF5 sub-group.

    Each unique attribute key becomes one dataset of length N.  Numeric values
    are stored as float64 (NaN for missing entries).  String values are stored
    as variable-length strings (empty string for missing entries).

    Parameters:
        grp: Open h5py Group to write into.
        neuron_attributes (list[dict]): List of N dicts, one per unit.
    """
    if not neuron_attributes:
        return

    N = len(neuron_attributes)
    na_grp = grp.create_group("neuron_attributes")

    all_keys: set = set()
    for d in neuron_attributes:
        all_keys.update(d.keys())

    for attr_key in all_keys:
        values = [d.get(attr_key) for d in neuron_attributes]
        non_none = [v for v in values if v is not None]

        if not non_none:
            na_grp.create_dataset(attr_key, data=np.full(N, np.nan))
            continue

        use_string = any(isinstance(v, str) for v in non_none)
        if use_string:
            str_values = [str(v) if v is not None else "" for v in values]
            dt = h5py.string_dtype()
            na_grp.create_dataset(
                attr_key,
                data=np.array(str_values, dtype=object),
                dtype=dt,
            )
        else:
            float_values = [float(v) if v is not None else np.nan for v in values]
            na_grp.create_dataset(
                attr_key, data=np.array(float_values, dtype=np.float64)
            )


def _load_neuron_attributes(grp) -> Optional[list]:
    """
    Reconstruct list of N per-unit dicts from a neuron_attributes HDF5 sub-group.

    Returns:
        neuron_attributes (list[dict] | None): Reconstructed list, or None if
            no neuron_attributes group exists or all dicts are empty.
    """
    if "neuron_attributes" not in grp:
        return None

    na_grp = grp["neuron_attributes"]
    if len(na_grp.keys()) == 0:
        return None

    first_key = next(iter(na_grp.keys()))
    N = len(na_grp[first_key])
    result = [{} for _ in range(N)]

    for attr_key in na_grp.keys():
        raw = na_grp[attr_key][:]
        if raw.dtype.kind in ("S", "O"):
            values = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw
            ]
        else:
            values = raw.tolist()

        for i, v in enumerate(values):
            # Skip NaN sentinels used for missing float values
            if isinstance(v, float) and np.isnan(v):
                continue
            result[i][attr_key] = v

    if all(len(d) == 0 for d in result):
        return None
    return result


def _dump_labels(grp, labels: Optional[list]) -> None:
    """
    Store unit labels (list of int or str) as an HDF5 dataset named 'labels'.

    Parameters:
        grp: Open h5py Group to write into.
        labels (list | None): Per-unit label values, or None to skip.
    """
    if labels is None:
        return
    non_none = [l for l in labels if l is not None]
    if not non_none:
        return
    use_string = any(isinstance(l, str) for l in non_none)
    if use_string:
        dt = h5py.string_dtype()
        grp.create_dataset(
            "labels",
            data=np.array([str(l) for l in labels], dtype=object),
            dtype=dt,
        )
    else:
        grp.create_dataset("labels", data=np.array(labels))


def _load_labels(grp) -> Optional[list]:
    """
    Reconstruct labels list from an HDF5 group, or None if not present.

    Returns:
        labels (list | None): Reconstructed labels, or None.
    """
    if "labels" not in grp:
        return None
    raw = grp["labels"][:]
    if raw.dtype.kind in ("S", "O"):
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw]
    return raw.tolist()


def _dump_times_tuples(grp, times: Optional[list], key: str = "times") -> None:
    """
    Store a list of (start, end) tuples as a (S, 2) float64 dataset.

    Parameters:
        grp: Open h5py Group to write into.
        times (list[tuple] | None): List of (start, end) pairs.
        key (str): Dataset name within the group.
    """
    if times is None:
        return
    arr = np.array(times, dtype=np.float64)
    grp.create_dataset(key, data=arr)


def _load_times_tuples(grp, key: str = "times") -> Optional[list]:
    """
    Reconstruct a list of (start, end) tuples from a (S, 2) HDF5 dataset.

    Returns:
        times (list[tuple] | None): Reconstructed list, or None if not present.
    """
    if key not in grp:
        return None
    arr = grp[key][:]
    return [(float(row[0]), float(row[1])) for row in arr]


def _dump_metadata_json(grp, metadata: dict) -> None:
    """
    Store a metadata dict as a JSON string attribute '__metadata__'.

    Parameters:
        grp: Open h5py Group to write into.
        metadata (dict): Must be JSON-serialisable.

    Raises:
        ValueError: If metadata contains non-JSON-serialisable values.
    """
    try:
        grp.attrs["__metadata__"] = json.dumps(metadata)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"metadata contains non-JSON-serialisable values and cannot be saved "
            f"to HDF5. Offending value: {e}"
        )


def _load_metadata_json(grp) -> dict:
    """
    Reconstruct a metadata dict from the '__metadata__' JSON string attribute.

    Returns:
        metadata (dict): Reconstructed metadata, or empty dict if not present.
    """
    raw = grp.attrs.get("__metadata__", "{}")
    return json.loads(raw)


# ===========================================================================
# SpikeData
# ===========================================================================


def _dump_spikedata(grp, sd) -> None:
    flat = (
        np.concatenate(sd.train)
        if any(len(t) > 0 for t in sd.train)
        else np.array([], dtype=np.float64)
    )
    index = np.cumsum([len(t) for t in sd.train], dtype=np.int64)
    grp.create_dataset("spike_times", data=flat.astype(np.float64))
    grp.create_dataset("spike_times_index", data=index)
    grp.attrs["length_ms"] = float(sd.length)
    grp.attrs["N"] = int(sd.N)
    _dump_metadata_json(grp, sd.metadata)
    if sd.raw_data.size > 0:
        grp.create_dataset("raw_data", data=sd.raw_data)
        grp.create_dataset("raw_time", data=sd.raw_time)
    if sd.neuron_attributes is not None:
        _dump_neuron_attributes(grp, sd.neuron_attributes)


def _load_spikedata(grp):
    from spikedata.spikedata import SpikeData

    flat = np.array(grp["spike_times"], dtype=np.float64)
    index = np.array(grp["spike_times_index"], dtype=np.int64)
    N = int(grp.attrs["N"])
    length_ms = float(grp.attrs["length_ms"])
    metadata = _load_metadata_json(grp)

    train = []
    prev = 0
    for end in index:
        train.append(flat[prev:end])
        prev = int(end)

    raw_data = np.array(grp["raw_data"]) if "raw_data" in grp else None
    raw_time = np.array(grp["raw_time"]) if "raw_time" in grp else None
    neuron_attributes = _load_neuron_attributes(grp)

    return SpikeData(
        train,
        length=length_ms,
        N=N,
        metadata=metadata,
        neuron_attributes=neuron_attributes,
        raw_data=raw_data,
        raw_time=raw_time,
    )


# ===========================================================================
# RateData
# ===========================================================================


def _dump_ratedata(grp, rd) -> None:
    grp.create_dataset("inst_Frate_data", data=rd.inst_Frate_data.astype(np.float64))
    grp.create_dataset("times", data=rd.times.astype(np.float64))
    if rd.neuron_attributes is not None:
        _dump_neuron_attributes(grp, rd.neuron_attributes)


def _load_ratedata(grp):
    from spikedata.ratedata import RateData

    inst_Frate_data = np.array(grp["inst_Frate_data"])
    times = np.array(grp["times"])
    neuron_attributes = _load_neuron_attributes(grp)
    return RateData(inst_Frate_data, times, neuron_attributes=neuron_attributes)


# ===========================================================================
# RateSliceStack
# ===========================================================================


def _dump_rateslicestack(grp, rss) -> None:
    grp.create_dataset("event_stack", data=rss.event_stack.astype(np.float64))
    _dump_times_tuples(grp, rss.times)
    grp.attrs["step_size"] = float(rss.step_size)


def _load_rateslicestack(grp):
    from spikedata.rateslicestack import RateSliceStack

    event_stack = np.array(grp["event_stack"])
    times = _load_times_tuples(grp)
    step_size = float(grp.attrs["step_size"])
    return RateSliceStack(
        data_obj=None,
        event_matrix=event_stack,
        times_start_to_end=times,
        step_size=step_size,
    )


# ===========================================================================
# SpikeSliceStack
# ===========================================================================


def _dump_spikeslicestack(grp, sss) -> None:
    _dump_times_tuples(grp, sss.times)
    slices_grp = grp.create_group("spike_stack")
    for i, sd in enumerate(sss.spike_stack):
        sd_grp = slices_grp.create_group(str(i))
        _dump_spikedata(sd_grp, sd)


def _load_spikeslicestack(grp):
    from spikedata.spikeslicestack import SpikeSliceStack

    times = _load_times_tuples(grp)
    slices_grp = grp["spike_stack"]
    n_slices = len(slices_grp)
    spike_stack = [_load_spikedata(slices_grp[str(i)]) for i in range(n_slices)]

    # Bypass the constructor (which requires a full SpikeData + subtime slicing)
    # and set fields directly, as all slice data is already reconstructed.
    sss = SpikeSliceStack.__new__(SpikeSliceStack)
    sss.spike_stack = spike_stack
    sss.times = times
    return sss


# ===========================================================================
# PairwiseCompMatrix
# ===========================================================================


def _dump_pairwise(grp, pcm) -> None:
    grp.create_dataset("matrix", data=pcm.matrix.astype(np.float64))
    _dump_labels(grp, pcm.labels)
    _dump_metadata_json(grp, pcm.metadata)


def _load_pairwise(grp):
    from spikedata.pairwise import PairwiseCompMatrix

    matrix = np.array(grp["matrix"])
    labels = _load_labels(grp)
    metadata = _load_metadata_json(grp)
    return PairwiseCompMatrix(matrix=matrix, labels=labels, metadata=metadata)


# ===========================================================================
# PairwiseCompMatrixStack
# ===========================================================================


def _dump_pairwise_stack(grp, pcms) -> None:
    grp.create_dataset("stack", data=pcms.stack.astype(np.float64))
    _dump_labels(grp, pcms.labels)
    _dump_times_tuples(grp, pcms.times)
    _dump_metadata_json(grp, pcms.metadata)


def _load_pairwise_stack(grp):
    from spikedata.pairwise import PairwiseCompMatrixStack

    stack = np.array(grp["stack"])
    labels = _load_labels(grp)
    times = _load_times_tuples(grp)
    metadata = _load_metadata_json(grp)
    return PairwiseCompMatrixStack(
        stack=stack, labels=labels, times=times, metadata=metadata
    )
