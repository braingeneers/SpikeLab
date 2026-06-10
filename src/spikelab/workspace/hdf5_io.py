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
Top-level values stored in a namespace:
ndarray, SpikeData, RateData, RateSliceStack, SpikeSliceStack,
PairwiseCompMatrix, PairwiseCompMatrixStack, dict.

Inside a dict (recursive), the supported leaf types additionally
include: int, float, bool, str, None, list (lossy — round-trips
as ndarray), tuple, set, frozenset, plus any of the top-level
types above. See ``_dump_dict`` for the full per-type schema
and round-trip semantics (e.g. tuple/set/frozenset preserve
their Python type via ``__type__`` tags; ndarray of unicode
strings is supported via h5py's variable-length string dtype).
"""

import json
import time
import warnings
from typing import Any, Optional, Tuple

import numpy as np

import h5py

_NDARRAY_SENTINEL = "__ndarray__"

# Per-unit neuron-attribute keys that are intentionally NOT persisted to the
# workspace. ``spike_train_samples`` (written by the spike sorter) is transient
# export scratch: it duplicates ``SpikeData.train`` expressed in sample units,
# is consumed only in-memory by the .mat/.npz DL compiler during the sort run,
# and is read by nothing after a workspace round-trip. It is also ragged (one
# entry per spike, so its length differs per unit), which the uniform-shape
# neuron-attribute serializer cannot store anyway. It is fully and exactly
# recoverable from ``train`` and the recording sampling rate:
# ``samples = round(train_ms * fs_Hz / 1000)``.
_NON_PERSISTED_NEURON_ATTRS = frozenset({"spike_train_samples"})


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays and scalar types to Python primitives.

    Tier L-F3: ndarrays are wrapped in a sentinel dict (``{"__ndarray__":
    True, "dtype": ..., "shape": ..., "values": ...}``) so the load path
    can rebuild the original ndarray with its dtype and shape preserved.
    The previous behaviour — ``ndarray.tolist()`` — silently degraded
    every ndarray-valued metadata entry to a Python list, breaking
    downstream code that did e.g. ``metadata["trial_start_times"] /
    1000`` because list/scalar is a TypeError.
    """

    def default(self, obj):  # noqa: D102
        """Convert numpy types to JSON-serializable Python primitives."""
        if isinstance(obj, np.ndarray):
            return {
                _NDARRAY_SENTINEL: True,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "values": obj.tolist(),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.str_):
            return str(obj)
        return super().default(obj)


def _restore_ndarrays(obj):
    """Recursively reconstitute ndarrays from ``_NumpyEncoder`` sentinels.

    Walks dicts and lists. A dict carrying ``__ndarray__: True`` is
    rebuilt via ``np.asarray(values, dtype=dtype).reshape(shape)``;
    every other value is passed through unchanged. Legacy files
    (encoded before this change) contain plain lists with no sentinel
    and therefore round-trip as lists — matching the pre-L-F3
    behaviour for files on disk that pre-date the upgrade.
    """
    if isinstance(obj, dict):
        if obj.get(_NDARRAY_SENTINEL) is True:
            values = obj["values"]
            dtype = obj["dtype"]
            shape = obj["shape"]
            arr = np.asarray(values, dtype=dtype)
            # ``np.asarray`` already gives the right shape for non-empty
            # nested lists; ``reshape`` is needed for the empty / 0-D
            # cases where ``values`` is ``[]`` or a bare scalar.
            return arr.reshape(shape)
        return {k: _restore_ndarrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_ndarrays(v) for v in obj]
    return obj


# ===========================================================================
# Public API
# ===========================================================================


def dump_workspace(ws, path: str) -> None:
    """Write a full AnalysisWorkspace to an HDF5 file at {path}.h5.

    Parameters:
        ws (AnalysisWorkspace): The workspace to serialise.
        path (str): Base path without file extension.

    Notes:
        - The HDF5 file is written to a sibling ``.tmp`` path and
          ``os.replace``-d into position on success. A partial write
          (disk full, permission, interrupted process) therefore can
          no longer leave a half-populated ``{path}.h5`` next to a
          stale ``{path}.json`` index.
    """
    import os

    h5_path = f"{path}.h5"
    tmp_path = f"{h5_path}.tmp"
    try:
        with h5py.File(tmp_path, "w") as f:
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
        os.replace(tmp_path, h5_path)
    except BaseException:
        # Best-effort cleanup; the underlying error is re-raised
        # regardless. ``BaseException`` is used so KeyboardInterrupt
        # mid-write also tidies up the half-written tmp.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_workspace_full(path: str):
    """Load a full AnalysisWorkspace from {path}.h5, reconstructing all objects.

    Parameters:
        path (str): Base path without file extension.

    Returns:
        ws (AnalysisWorkspace): Reconstructed workspace with all items restored
            to their original IAT data class types.
    """

    from .workspace import AnalysisWorkspace

    h5_path = f"{path}.h5"
    with h5py.File(h5_path, "r") as f:
        if "__workspace_id__" not in f.attrs:
            raise ValueError(
                f"The HDF5 file '{h5_path}' does not appear to be a SpikeLab "
                "workspace (missing __workspace_id__ attribute)."
            )
        ws = AnalysisWorkspace.__new__(AnalysisWorkspace)
        # ``str(bytes_obj)`` returns the literal ``"b'foo'"`` repr —
        # decode explicitly so a workspace written under h5py's
        # variable-length-bytes mode does not silently corrupt the
        # ``workspace_id`` / ``name`` attrs on reload.
        wid_attr = f.attrs["__workspace_id__"]
        ws.workspace_id = (
            wid_attr.decode("utf-8") if isinstance(wid_attr, bytes) else str(wid_attr)
        )
        name_attr = f.attrs["__workspace_name__"]
        name = (
            name_attr.decode("utf-8")
            if isinstance(name_attr, bytes)
            else str(name_attr)
        )
        ws.name = name if name else None
        ws.created_at = float(f.attrs["__created_at__"])
        ws._items = {}
        ws._index = {}
        for ns in f.keys():
            # Skip any top-level group whose name starts with ``__``.
            # This mirrors the ``__workspace_id__`` / ``__workspace_name__``
            # / ``__created_at__`` attribute convention and reserves
            # the namespace for future metadata sub-groups (e.g. a
            # ``/__history__/`` audit log). Without this filter, the
            # loader would treat such a group as a user namespace and
            # ``_load_item`` would fail on its first child (missing
            # ``__type__`` attr).
            if ns.startswith("__"):
                continue
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
    """Load a single item from a saved workspace HDF5 file.

    Reconstructs the original IAT data class from the stored HDF5 data.

    Parameters:
        path (str): Base path without file extension.
        namespace (str): Namespace the item was stored under.
        key (str): Key the item was stored under.

    Returns:
        obj: Reconstructed IAT data object or numpy array.
    """

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
    """Write a single item to an HDF5 file, creating or overwriting the item group.

    .. warning::
        HDF5 does **not** support concurrent writes to the same file.
        Two processes (or threads) calling ``dump_item_to_file`` on
        the same ``h5_path`` simultaneously will corrupt the file.
        The caller is responsible for serialising writes — typically
        by using per-process workspaces and ``AnalysisWorkspace.merge_from``
        for aggregation (see the ``merge_from`` docstring for the
        merge protocol). ``LazyAnalysisWorkspace`` is safe by virtue
        of each instance owning a unique ``tempfile.mkstemp`` path.

    Parameters:
        h5_path (str): Full path to the HDF5 file (including .h5 extension).
        namespace (str): Namespace group to write into.
        key (str): Key group to write into.
        obj: Object to serialise.
        created_at (float): POSIX timestamp for the item.
        note (str | None): Optional annotation.
    """

    with h5py.File(h5_path, "a") as f:
        ns_grp = f.require_group(namespace)
        if key in ns_grp:
            del ns_grp[key]
        key_grp = ns_grp.create_group(key)
        _dump_item(key_grp, obj, created_at, note)


def load_item_from_file(h5_path: str, namespace: str, key: str) -> Any:
    """Load a single item from an HDF5 file by its full path.

    Parameters:
        h5_path (str): Full path to the HDF5 file (including .h5 extension).
        namespace (str): Namespace the item was stored under.
        key (str): Key the item was stored under.

    Returns:
        obj: Reconstructed IAT data object or numpy array.
    """

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
    """Delete a single item or entire namespace from an HDF5 file.

    Parameters:
        h5_path (str): Full path to the HDF5 file (including .h5 extension).
        namespace (str): Namespace to delete from.
        key (str | None): Key to delete. If None, deletes the entire namespace.
    """

    with h5py.File(h5_path, "a") as f:
        if namespace not in f:
            return
        if key is None:
            del f[namespace]
        elif key in f[namespace]:
            del f[namespace][key]


def set_note_in_file(h5_path: str, namespace: str, key: str, note: str) -> None:
    """Write a note attribute onto an existing item in an HDF5 workspace file.

    Parameters:
        h5_path (str): Path to the HDF5 workspace file.
        namespace (str): Namespace containing the target item.
        key (str): Item key within the namespace.
        note (str): Note text to attach.

    Raises:
        KeyError: If the item ``(namespace, key)`` does not exist.
    """
    with h5py.File(h5_path, "a") as f:
        if namespace not in f or key not in f[namespace]:
            raise KeyError(f"workspace item not found: ({namespace!r}, {key!r})")
        f[namespace][key].attrs["__note__"] = note


# ===========================================================================
# Item-level dump / load
# ===========================================================================


def _dump_item(grp, obj: Any, created_at: float, note: Optional[str]) -> None:
    """Write one object to an HDF5 group, tagging with __type__ and metadata attrs."""
    try:
        from ..spikedata.spikedata import SpikeData
    except ImportError:
        SpikeData = None
    try:
        from ..spikedata.ratedata import RateData
    except ImportError:
        RateData = None
    try:
        from ..spikedata.rateslicestack import RateSliceStack
    except ImportError:
        RateSliceStack = None
    try:
        from ..spikedata.spikeslicestack import SpikeSliceStack
    except ImportError:
        SpikeSliceStack = None
    try:
        from ..spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
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
    elif isinstance(obj, dict):
        grp.attrs["__type__"] = "dict"
        _dump_dict(grp, obj, created_at)
    else:
        raise TypeError(
            f"Cannot serialise object of type '{type(obj).__name__}' to HDF5. "
            "Supported types: ndarray, SpikeData, RateData, RateSliceStack, "
            "SpikeSliceStack, PairwiseCompMatrix, PairwiseCompMatrixStack, "
            "dict. Inside a dict, additional types are supported: int, "
            "float, bool, str, None, list (lossy → ndarray), tuple, set, "
            "frozenset. See ``_dump_dict`` for the full schema."
        )


def _load_item(grp) -> Tuple[Any, dict]:
    """Read and reconstruct one object from an HDF5 group.

    Parameters:
        grp: Open h5py Group to read from.

    Returns:
        obj: Reconstructed IAT data object or numpy array.
        index_entry (dict): Summary metadata for the workspace index.
    """
    from .workspace import _make_summary

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
        "dict": _load_dict,
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
    """Write an ndarray to the group's ``data`` dataset.

    Fixed-width unicode/byte-string arrays (dtype kinds ``U`` / ``S``)
    are stored via h5py's variable-length string dtype because h5py
    cannot persist ``dtype('<Un')`` directly. The on-disk dataset is
    tagged with ``__string_array__ = True`` so the load side knows to
    decode bytes back into Python strings.
    """
    if arr.dtype.kind in ("U", "S"):
        import h5py  # type: ignore

        str_dtype = h5py.string_dtype(encoding="utf-8")
        ds = grp.create_dataset("data", data=arr.astype(object), dtype=str_dtype)
        ds.attrs["__string_array__"] = True
    else:
        grp.create_dataset("data", data=arr)


def _load_ndarray(grp) -> np.ndarray:
    """Reconstruct an ndarray from the group's ``data`` dataset.

    String arrays come back from h5py as ``object`` arrays of bytes
    (older h5py) or Python strings (newer h5py). Coerce to a numpy
    unicode array so callers see consistent semantics regardless of
    h5py version.
    """
    ds = grp["data"]
    arr = np.array(ds)
    if ds.attrs.get("__string_array__", False):
        # Coerce to Python str array; bytes decode to utf-8.
        decoded = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in arr.ravel().tolist()
        ]
        arr = np.array(decoded).reshape(arr.shape)
    return arr


# ===========================================================================
# dict
# ===========================================================================


def _dump_dict(grp, d: dict, created_at: float) -> None:
    """Recursively serialise a plain dict to an HDF5 group.

    Each dict key becomes a child group whose value is serialised
    according to its type.

    Supported value types (and how they round-trip):

      - ``int``, ``float``, ``bool`` (incl. numpy scalar variants):
        stored as ``__type__ = "scalar"`` attrs. Round-trip preserves
        scalar kind (int / float / bool) via ``__scalar_kind__``.
      - ``str``: stored as ``__type__ = "scalar_str"`` attrs.
      - ``None``: stored as ``__type__ = "none"`` (no payload).
        Round-trips back to ``None``.
      - ``list``: converted to ``ndarray`` and stored as
        ``__type__ = "ndarray"``. **Lossy**: round-trips as ndarray,
        not list. Heterogeneous / ragged lists raise ``TypeError``.
      - ``tuple``: converted to ``ndarray`` and stored as
        ``__type__ = "tuple"`` with the same heterogeneity check as
        lists. Round-trips as ``tuple`` (type preserved).
      - ``set`` / ``frozenset``: sorted into a canonical order, then
        stored as ``ndarray`` with ``__type__ = "set"`` /
        ``"frozenset"``. Round-trips as ``set`` / ``frozenset`` (type
        preserved, order not). Elements must be orderable and
        homogeneous. **Lossy for mixed-numeric content**: a set such
        as ``{1, 2.5}`` widens to a float ndarray (``[1.0, 2.5]``)
        and reloads as a set of floats (``{1.0, 2.5}``) — the
        original ``int`` element is lost. Pure-int and pure-float
        sets round-trip with dtype preserved; only the
        int/float-mixed case widens.
      - ``dict``: recursively serialised via this function.
      - ``ndarray``, ``SpikeData``, ``RateData``, slice stacks,
        pairwise matrices, and pairwise stacks: routed through
        ``_dump_item``'s dedicated serialisers.

    Anything else triggers a ``TypeError`` from ``_dump_item`` listing
    the supported types.

    Raises:
        ValueError: If any dict key is not a non-empty string, or
            contains a forward slash (h5py interprets ``/`` as a
            group-path separator and would silently corrupt the
            round-trip).
        TypeError: If any value is a ragged / mixed-type list or
            tuple, a mixed-type set, or a type not in the supported
            list above.
    """
    for k, v in d.items():
        # Reject keys that h5py would either reject cryptically
        # (empty / non-string) or silently misinterpret (slash). Up-front
        # validation gives a clear error and avoids silent corruption.
        if not isinstance(k, str):
            raise ValueError(
                f"Dict key {k!r} is not a string. HDF5 group names must "
                f"be strings (got {type(k).__name__})."
            )
        if not k:
            raise ValueError("Dict key is empty. HDF5 group names must be non-empty.")
        if "/" in k:
            raise ValueError(
                f"Dict key {k!r} contains a forward slash. h5py treats "
                f"'/' as a group-path separator; use a different "
                f"separator (e.g. '_' or '.') in dict keys."
            )
        if isinstance(v, list):
            v = np.asarray(v)
            if v.dtype == object:
                raise TypeError(
                    f"Cannot serialize ragged or mixed-type list for key {k!r}. "
                    "All elements must have the same shape and type."
                )
        if v is None:
            child = grp.create_group(k)
            child.attrs["__type__"] = "none"
        elif isinstance(v, tuple):
            arr = np.asarray(v)
            if arr.dtype == object:
                raise TypeError(
                    f"Cannot serialize ragged or mixed-type tuple for key {k!r}. "
                    "All elements must have the same shape and type."
                )
            child = grp.create_group(k)
            child.attrs["__type__"] = "tuple"
            _dump_ndarray(child, arr)
        elif isinstance(v, (set, frozenset)):
            try:
                ordered = sorted(v)
            except TypeError as exc:
                raise TypeError(
                    f"Cannot serialize set/frozenset for key {k!r} with "
                    f"unorderable elements ({exc}). All elements must be "
                    "mutually orderable so the on-disk representation is "
                    "deterministic."
                ) from exc
            arr = np.asarray(ordered)
            if arr.dtype == object:
                raise TypeError(
                    f"Cannot serialize mixed-type set/frozenset for key "
                    f"{k!r}. All elements must have the same shape and type."
                )
            child = grp.create_group(k)
            child.attrs["__type__"] = "frozenset" if isinstance(v, frozenset) else "set"
            _dump_ndarray(child, arr)
        elif isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            child = grp.create_group(k)
            child.attrs["__type__"] = "scalar"
            if isinstance(v, (bool, np.bool_)):
                child.attrs["__scalar_value__"] = bool(v)
            elif isinstance(v, (int, np.integer)):
                child.attrs["__scalar_value__"] = int(v)
            else:
                child.attrs["__scalar_value__"] = float(v)
            child.attrs["__scalar_kind__"] = type(v).__name__
        elif isinstance(v, str):
            child = grp.create_group(k)
            child.attrs["__type__"] = "scalar_str"
            child.attrs["__scalar_value__"] = v
        else:
            child = grp.create_group(k)
            _dump_item(child, v, created_at, note=None)


def _load_dict(grp) -> dict:
    """Reconstruct a dict from an HDF5 group written by ``_dump_dict``.

    Recognises the type tags written by :func:`_dump_dict`:
    ``scalar``, ``scalar_str``, ``none``, ``tuple``, ``set``,
    ``frozenset``, and everything else (``ndarray``, ``dict``,
    ``SpikeData``, etc.) routes through :func:`_load_item`.
    """
    result = {}
    for k in grp.keys():
        child = grp[k]
        type_tag = str(child.attrs.get("__type__", ""))
        if type_tag == "scalar":
            kind = str(child.attrs.get("__scalar_kind__", "float"))
            val = child.attrs["__scalar_value__"]
            if "int" in kind or kind.startswith("uint"):
                val = int(val)
            elif kind in ("bool", "bool_"):
                val = bool(val)
            else:
                val = float(val)
            result[k] = val
        elif type_tag == "scalar_str":
            result[k] = str(child.attrs["__scalar_value__"])
        elif type_tag == "none":
            result[k] = None
        elif type_tag == "tuple":
            result[k] = tuple(_load_ndarray(child).tolist())
        elif type_tag == "set":
            result[k] = set(_load_ndarray(child).tolist())
        elif type_tag == "frozenset":
            result[k] = frozenset(_load_ndarray(child).tolist())
        else:
            obj, _ = _load_item(child)
            result[k] = obj
    return result


# ===========================================================================
# Shared helpers
# ===========================================================================


def _dump_neuron_attributes(grp, neuron_attributes: list) -> None:
    """Serialise a list of N per-unit attribute dicts to an HDF5 sub-group.

    Each unique attribute key becomes one dataset of length N. Numeric
    values are stored as float64 (NaN for missing entries). String
    values are stored as variable-length strings (empty string for
    missing entries).

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

    # Drop transient/non-persisted per-unit attributes (see
    # _NON_PERSISTED_NEURON_ATTRS). Removing them here means a ragged
    # attribute like spike_train_samples never reaches the uniform-shape
    # validation below, so it no longer blocks the save.
    all_keys -= _NON_PERSISTED_NEURON_ATTRS

    _SUPPORTED_SCALAR_TYPES = (
        str,
        bool,
        int,
        float,
        np.integer,
        np.floating,
        np.bool_,
    )
    _SUPPORTED_ARRAY_TYPES = (np.ndarray, list, tuple)

    for attr_key in all_keys:
        # h5py interprets '/' in dataset names as a path separator, so a
        # key like 'meta/info' would create a nested group rather than a
        # literal attribute. Reject up front instead of silently
        # corrupting the round-trip.
        if "/" in attr_key:
            raise ValueError(
                f"Neuron attribute key {attr_key!r} contains a forward "
                f"slash. h5py treats '/' as a group-path separator; use "
                f"a different separator (e.g. '_' or '.') in attribute "
                f"keys."
            )

        values = [d.get(attr_key) for d in neuron_attributes]
        non_none = [v for v in values if v is not None]

        # Reject unsupported value types upfront with a clear message
        # naming the attribute and offending type. This guards against
        # deep TypeError/IndexError later in dataset construction.
        for v in non_none:
            if not isinstance(v, _SUPPORTED_SCALAR_TYPES + _SUPPORTED_ARRAY_TYPES):
                raise ValueError(
                    f"Neuron attribute {attr_key!r} contains an unsupported "
                    f"value type: {type(v).__name__}. Supported types are "
                    f"numeric scalars, strings, and array-likes "
                    f"(ndarray/list/tuple)."
                )

        if not non_none:
            na_grp.create_dataset(attr_key, data=np.full(N, np.nan))
            continue

        use_array = any(isinstance(v, _SUPPORTED_ARRAY_TYPES) for v in non_none)
        use_string = any(isinstance(v, str) for v in non_none)

        if use_array:
            # Reject mixed scalar + array under the same key. Previously the
            # shape-validation loop only iterated array-typed values, so a
            # scalar passed through silently and was broadcast across the
            # array dimensions at write time — producing a phantom row
            # with the scalar repeated at every position. Surface the
            # mistake at dump time with a clear message naming the unit
            # whose value is the wrong type.
            for i, v in enumerate(values):
                if v is None or isinstance(v, _SUPPORTED_ARRAY_TYPES):
                    continue
                raise ValueError(
                    f"Neuron attribute {attr_key!r} mixes scalar and array "
                    f"values across units: unit {i} has scalar "
                    f"{type(v).__name__} {v!r} but other units have arrays. "
                    "All units must use the same value shape for a given "
                    "attribute. Either wrap the scalar in a length-1 array "
                    "to match, or split the attribute into separate keys."
                )
            # Infer dtype from the first non-None array value rather
            # than always coercing to float64. Two reasons:
            #   (1) Integer-valued attributes (per-unit channel-index
            #       arrays, template indices) silently widened to
            #       float64 on every save/load round-trip.
            #   (2) NaN can't serve as a missing-entry sentinel for
            #       integer dtypes, and for float dtypes it collides
            #       with legitimate NaN inside the stored array (e.g.
            #       a NaN waveform sample would be indistinguishable
            #       from "this unit had no value for this attribute"
            #       on reload).
            # We store missing units in a ``__missing_unit_indices__``
            # HDF5 attribute on the dataset, so the dtype-preserving
            # buffer's fill value never has to encode missingness.
            sample = np.asarray(
                next(v for v in non_none if isinstance(v, (np.ndarray, list, tuple)))
            )
            arr_shape = sample.shape
            arr_dtype = sample.dtype
            for v in non_none:
                if isinstance(v, (np.ndarray, list, tuple)):
                    v_shape = np.asarray(v).shape
                    if v_shape != arr_shape:
                        raise ValueError(
                            f"Neuron attribute {attr_key!r} has inconsistent "
                            f"shapes: expected {arr_shape}, got {v_shape}. "
                            f"All units must have the same array shape for "
                            f"a given attribute."
                        )
            if np.issubdtype(arr_dtype, np.floating):
                fill: Any = np.nan
            elif np.issubdtype(arr_dtype, np.bool_):
                fill = False
            else:
                # integer, complex, etc. — fill with 0 cast to the
                # target dtype. The ``__missing_unit_indices__`` attr
                # is authoritative, so the actual fill value is only
                # cosmetic.
                fill = 0
            stacked = np.full((N, *arr_shape), fill, dtype=arr_dtype)
            missing_indices = []
            for i, v in enumerate(values):
                if v is None:
                    missing_indices.append(i)
                else:
                    stacked[i] = np.asarray(v, dtype=arr_dtype)
            ds = na_grp.create_dataset(attr_key, data=stacked)
            if missing_indices:
                ds.attrs["__missing_unit_indices__"] = np.asarray(
                    missing_indices, dtype=np.int64
                )
        elif use_string:
            str_values = [str(v) if v is not None else "" for v in values]
            dt = h5py.string_dtype()
            na_grp.create_dataset(
                attr_key,
                data=np.array(str_values, dtype=object),
                dtype=dt,
            )
        else:
            # NaN doubles as the missing-entry sentinel here, so a
            # legitimate NaN supplied by the caller would be silently
            # dropped on reload. Warn so the caller can pick a different
            # convention (e.g. omit the attribute, or use a sentinel
            # like -1).
            if any(
                isinstance(v, (float, np.floating)) and np.isnan(v) for v in non_none
            ):
                warnings.warn(
                    f"Neuron attribute {attr_key!r} contains NaN values "
                    f"that will be indistinguishable from missing "
                    f"entries when reloaded (NaN is the missing-entry "
                    f"sentinel for float attributes). Drop the "
                    f"attribute or use a different convention if NaN "
                    f"is meaningful here.",
                    UserWarning,
                    stacklevel=2,
                )
            # Infer the scalar Python kind from the first non-None
            # value so the load path can restore the original type.
            # Previously every scalar attribute round-tripped as
            # Python ``float``, silently changing the type of
            # integer-valued attributes like ``electrode = 47`` to
            # ``47.0`` — downstream ``isinstance(v, int)`` checks
            # would then break.
            sample_scalar = non_none[0]
            if isinstance(sample_scalar, (bool, np.bool_)):
                scalar_kind = "bool"
            elif isinstance(sample_scalar, (int, np.integer)):
                scalar_kind = "int"
            else:
                scalar_kind = "float"
            float_values = [float(v) if v is not None else np.nan for v in values]
            ds = na_grp.create_dataset(
                attr_key, data=np.array(float_values, dtype=np.float64)
            )
            if scalar_kind != "float":
                ds.attrs["__scalar_kind__"] = scalar_kind


def _load_neuron_attributes(grp) -> Optional[list]:
    """Reconstruct list of N per-unit dicts from a neuron_attributes HDF5 sub-group.

    Parameters:
        grp: Open h5py Group to read from.

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
        ds = na_grp[attr_key]
        raw = ds[:]
        if raw.dtype.kind in ("S", "O"):
            for i, v in enumerate(raw):
                decoded = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                if decoded:  # empty string is sentinel for missing
                    result[i][attr_key] = decoded
        elif raw.ndim > 1:
            # Array-valued attribute: each row is one unit's array.
            # Modern format records missing entries in the
            # ``__missing_unit_indices__`` HDF5 attr (preserves dtype
            # and disambiguates from legitimate NaN inside the data).
            # Legacy format (pre-dtype-preservation) used all-NaN rows
            # as the missing sentinel.
            missing_attr = ds.attrs.get("__missing_unit_indices__")
            if missing_attr is not None:
                missing_set = {int(i) for i in np.asarray(missing_attr).ravel()}
                for i in range(len(raw)):
                    if i in missing_set:
                        continue
                    result[i][attr_key] = raw[i].copy()
            else:
                # Legacy fallback: all-NaN row marks a missing entry.
                # Only meaningful for float dtypes — integer/bool
                # arrays never used this path (those were silently
                # widened to float64 by the old writer, so they will
                # still load as float64 from legacy files).
                for i in range(len(raw)):
                    row = raw[i]
                    if np.issubdtype(row.dtype, np.floating) and np.all(np.isnan(row)):
                        continue
                    result[i][attr_key] = row.copy()
        else:
            # Scalar attribute. Modern format records the original
            # Python kind in ``__scalar_kind__`` ("int" or "bool")
            # so integer-valued attributes like ``electrode = 47``
            # don't silently round-trip as ``47.0``. Legacy files
            # without the attr load as float, matching the
            # pre-2026-05 contract.
            scalar_kind = ds.attrs.get("__scalar_kind__")
            if scalar_kind is not None and isinstance(scalar_kind, bytes):
                scalar_kind = scalar_kind.decode("utf-8")
            for i, v in enumerate(raw.tolist()):
                # Skip NaN sentinels used for missing float values
                if isinstance(v, float) and np.isnan(v):
                    continue
                if scalar_kind == "int":
                    v = int(v)
                elif scalar_kind == "bool":
                    v = bool(v)
                result[i][attr_key] = v

    if all(len(d) == 0 for d in result):
        return None
    return result


def _dump_labels(grp, labels: Optional[list]) -> None:
    """Store unit labels (list of int or str) as an HDF5 dataset named 'labels'.

    Parameters:
        grp: Open h5py Group to write into.
        labels (list | None): Per-unit label values, or None to skip.
    """
    if labels is None:
        return
    non_none = [lbl for lbl in labels if lbl is not None]
    if not non_none:
        return
    use_string = any(isinstance(lbl, str) for lbl in non_none)
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
    """Reconstruct labels list from an HDF5 group, or None if not present.

    Parameters:
        grp: Open h5py Group to read from.

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
    """Store a list of (start, end) tuples as a (S, 2) float64 dataset.

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
    """Reconstruct a list of (start, end) tuples from a (S, 2) HDF5 dataset.

    Parameters:
        grp: Open h5py Group to read from.
        key (str): Dataset name within the group.

    Returns:
        times (list[tuple] | None): Reconstructed list, or None if not present.
    """
    if key not in grp:
        return None
    arr = grp[key][:]
    return [(float(row[0]), float(row[1])) for row in arr]


def _dump_metadata_json(grp, metadata: dict) -> None:
    """Store a metadata dict as a JSON string attribute '__metadata__'.

    Parameters:
        grp: Open h5py Group to write into.
        metadata (dict): Must be JSON-serialisable.

    Raises:
        ValueError: If metadata contains non-JSON-serialisable values.
    """
    try:
        grp.attrs["__metadata__"] = json.dumps(metadata, cls=_NumpyEncoder)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"metadata contains non-JSON-serialisable values and cannot be saved "
            f"to HDF5. Offending value: {e}"
        )


def _load_metadata_json(grp) -> dict:
    """Reconstruct a metadata dict from the '__metadata__' JSON string attribute.

    Parameters:
        grp: Open h5py Group to read from.

    Returns:
        metadata (dict): Reconstructed metadata, or empty dict if not
            present. Ndarray-valued entries written by the Tier L-F3
            sentinel-encoded path are restored with their original
            dtype and shape; legacy entries written before that change
            still load as plain Python lists.
    """
    raw = grp.attrs.get("__metadata__", "{}")
    return _restore_ndarrays(json.loads(raw))


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
    grp.attrs["start_time"] = float(sd.start_time)
    grp.attrs["N"] = int(sd.N)
    _dump_metadata_json(grp, sd.metadata)
    # Preserve the independent-write semantics for raw_data and
    # raw_time: write each if it carries content. The original bug was
    # at LOAD time, where a scalar raw_time + missing raw_data tried
    # ``len(scalar)`` and crashed. The load path below now handles the
    # scalar case explicitly, so the dump path can stay permissive.
    if getattr(sd, "raw_data", None) is not None and sd.raw_data.size > 0:
        grp.create_dataset("raw_data", data=sd.raw_data)
    if getattr(sd, "raw_time", None) is not None and np.asarray(sd.raw_time).size > 0:
        grp.create_dataset("raw_time", data=sd.raw_time)
    if sd.neuron_attributes is not None:
        _dump_neuron_attributes(grp, sd.neuron_attributes)


def _load_spikedata(grp):
    from ..spikedata.spikedata import SpikeData

    flat = np.array(grp["spike_times"], dtype=np.float64)
    index = np.array(grp["spike_times_index"], dtype=np.int64)
    N = int(grp.attrs["N"])
    length_ms = float(grp.attrs["length_ms"])
    start_time = float(grp.attrs.get("start_time", 0.0))
    metadata = _load_metadata_json(grp)

    train = []
    prev = 0
    for end in index:
        train.append(flat[prev:end])
        prev = int(end)

    # _dump_spikedata now writes raw_data and raw_time atomically.
    raw_data = np.array(grp["raw_data"]) if "raw_data" in grp else None
    raw_time = np.array(grp["raw_time"]) if "raw_time" in grp else None
    # SpikeData requires both or neither. When only raw_time was
    # persisted (legitimate case: original raw_data had size 0),
    # fabricate the empty matching raw_data using the time vector's
    # trailing-axis length. Use np.atleast_1d so the construction
    # works for both arrays and scalars — the original ``len(raw_time)``
    # path crashed on scalar raw_time.
    if raw_data is None and raw_time is not None:
        time_len = np.atleast_1d(raw_time).shape[0]
        raw_data = np.zeros((0, time_len))
    neuron_attributes = _load_neuron_attributes(grp)

    return SpikeData(
        train,
        length=length_ms,
        start_time=start_time,
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
    from ..spikedata.ratedata import RateData

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
    if rss.neuron_attributes is not None:
        _dump_neuron_attributes(grp, rss.neuron_attributes)


def _load_rateslicestack(grp):
    from ..spikedata.rateslicestack import RateSliceStack

    event_stack = np.array(grp["event_stack"])
    times = _load_times_tuples(grp)
    step_size = float(grp.attrs["step_size"])
    neuron_attributes = _load_neuron_attributes(grp)
    return RateSliceStack(
        data_obj=None,
        event_matrix=event_stack,
        times_start_to_end=times,
        step_size=step_size,
        neuron_attributes=neuron_attributes,
    )


# ===========================================================================
# SpikeSliceStack
# ===========================================================================


def _dump_spikeslicestack(grp, sss) -> None:
    grp.attrs["N"] = sss.N
    _dump_times_tuples(grp, sss.times)
    slices_grp = grp.create_group("spike_stack")
    for i, sd in enumerate(sss.spike_stack):
        sd_grp = slices_grp.create_group(str(i))
        _dump_spikedata(sd_grp, sd)
    if sss.neuron_attributes is not None:
        _dump_neuron_attributes(grp, sss.neuron_attributes)


def _load_spikeslicestack(grp):
    from ..spikedata.spikeslicestack import SpikeSliceStack

    times = _load_times_tuples(grp)
    slices_grp = grp["spike_stack"]
    n_slices = len(slices_grp)
    spike_stack = [_load_spikedata(slices_grp[str(i)]) for i in range(n_slices)]
    neuron_attributes = _load_neuron_attributes(grp)

    # Bypass the constructor (which requires a full SpikeData + subtime slicing)
    # and set fields directly, as all slice data is already reconstructed.
    sss = SpikeSliceStack.__new__(SpikeSliceStack)
    sss.spike_stack = spike_stack
    sss.times = times
    sss.N = (
        int(grp.attrs["N"])
        if "N" in grp.attrs
        else (spike_stack[0].N if spike_stack else 0)
    )
    sss.neuron_attributes = neuron_attributes
    return sss


# ===========================================================================
# PairwiseCompMatrix
# ===========================================================================


def _dump_pairwise(grp, pcm) -> None:
    grp.create_dataset("matrix", data=pcm.matrix.astype(np.float64))
    _dump_labels(grp, pcm.labels)
    _dump_metadata_json(grp, pcm.metadata)


def _load_pairwise(grp):
    from ..spikedata.pairwise import PairwiseCompMatrix

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
    from ..spikedata.pairwise import PairwiseCompMatrixStack

    stack = np.array(grp["stack"])
    labels = _load_labels(grp)
    times = _load_times_tuples(grp)
    metadata = _load_metadata_json(grp)
    return PairwiseCompMatrixStack(
        stack=stack, labels=labels, times=times, metadata=metadata
    )
