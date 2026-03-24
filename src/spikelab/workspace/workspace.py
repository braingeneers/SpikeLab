"""
AnalysisWorkspace — named, namespaced container for analysis results.

Stores IAT data class objects and numpy arrays under two-level keys
(namespace, key). Supports save/load to disk (.h5 data + .json index).
Individual items can be loaded selectively from disk without loading
the full workspace.
"""

import json
import warnings
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np


def _make_summary(obj: Any) -> dict:
    """
    Build a JSON-serializable summary dict describing a stored object.

    Parameters:
        obj: Any supported IAT type or numpy array.

    Returns:
        summary (dict): Type and shape/attribute information.
    """
    # Lazy imports to avoid circular dependencies and keep optional deps optional.
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

    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}

    if SpikeData is not None and isinstance(obj, SpikeData):
        return {"type": "SpikeData", "N": obj.N, "length_ms": obj.length}

    if RateData is not None and isinstance(obj, RateData):
        return {"type": "RateData", "shape": list(obj.inst_Frate_data.shape)}

    if RateSliceStack is not None and isinstance(obj, RateSliceStack):
        return {"type": "RateSliceStack", "shape": list(obj.event_stack.shape)}

    if SpikeSliceStack is not None and isinstance(obj, SpikeSliceStack):
        length_ms = (
            float(obj.times[0][1] - obj.times[0][0]) if len(obj.times) > 0 else None
        )
        n_units = obj.spike_stack[0].N if len(obj.spike_stack) > 0 else 0
        return {
            "type": "SpikeSliceStack",
            "N_slices": len(obj.spike_stack),
            "N_units": n_units,
            "length_ms": length_ms,
        }

    # PairwiseCompMatrixStack must be checked before PairwiseCompMatrix since
    # it is not a subclass, but both are dataclasses from the same module.
    if PairwiseCompMatrixStack is not None and isinstance(obj, PairwiseCompMatrixStack):
        return {
            "type": "PairwiseCompMatrixStack",
            "shape": list(obj.stack.shape),
        }

    if PairwiseCompMatrix is not None and isinstance(obj, PairwiseCompMatrix):
        return {"type": "PairwiseCompMatrix", "shape": list(obj.matrix.shape)}

    return {"type": type(obj).__name__}


class AnalysisWorkspace:
    """
    Named, namespaced container for storing analysis results.

    Results are organised under two-level keys: a namespace (typically
    the name of a recording or comparison group) and a key (the specific
    result within that namespace). Supports saving and loading the full
    workspace to and from disk.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Create a new empty workspace.

        Parameters:
            name (str | None): Optional human-readable label for the workspace.
        """
        self.workspace_id: str = str(uuid.uuid4())
        self.name: Optional[str] = name
        self.created_at: float = time.time()
        self._items: Dict[str, Dict[str, Any]] = {}
        self._index: Dict[str, Dict[str, dict]] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        namespace: str,
        key: str,
        obj: Any,
        note: Optional[str] = None,
    ) -> None:
        """
        Store an object under (namespace, key).

        Parameters:
            namespace (str): Namespace grouping related results (e.g., a
                recording name).
            key (str): Human-readable key identifying this result within
                the namespace.
            obj: Object to store. Supported types: SpikeData, RateData,
                RateSliceStack, SpikeSliceStack, PairwiseCompMatrix,
                PairwiseCompMatrixStack, np.ndarray. Other types are
                accepted and stored, but their summary will only contain
                the class name.
            note (str | None): Optional free-text annotation attached to
                the index entry.

        Notes:
            - Storing under an existing (namespace, key) overwrites the
              previous value and refreshes the index entry.
        """
        if namespace not in self._items:
            self._items[namespace] = {}
            self._index[namespace] = {}

        self._items[namespace][key] = obj

        entry = _make_summary(obj)
        entry["created_at"] = time.time()
        if note is not None:
            entry["note"] = note
        self._index[namespace][key] = entry

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Retrieve a stored object.

        Parameters:
            namespace (str): Namespace the object was stored under.
            key (str): Key the object was stored under.

        Returns:
            obj: The stored object, or None if not found.
        """
        return self._items.get(namespace, {}).get(key)

    def get_info(self, namespace: str, key: str) -> Optional[dict]:
        """
        Return the index entry for an item without loading the object itself.

        Parameters:
            namespace (str): Namespace to look up.
            key (str): Key to look up.

        Returns:
            info (dict | None): Summary dict (type, shape/attributes, note,
                created_at), or None if not found.
        """
        return self._index.get(namespace, {}).get(key)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """
        Return the full index as a JSON-serializable dict.

        Returns:
            index (dict): Nested dict ``{namespace: {key: summary_dict}}``.
        """
        return {ns: dict(keys) for ns, keys in self._index.items()}

    def list_keys(self, namespace: Optional[str] = None) -> "dict | list":
        """
        List stored keys, optionally filtered to a single namespace.

        Parameters:
            namespace (str | None): If provided, returns the list of keys
                for that namespace. If None, returns a dict mapping each
                namespace to its list of keys.

        Returns:
            keys (dict | list): ``{namespace: [keys]}`` when namespace is
                None, otherwise ``[keys]``.
        """
        if namespace is not None:
            return list(self._items.get(namespace, {}).keys())
        return {ns: list(keys.keys()) for ns, keys in self._items.items()}

    def list_namespaces(self) -> list:
        """
        Return the names of all top-level namespaces in the workspace.

        Returns:
            namespaces (list[str]): Sorted list of namespace names.
        """
        return list(self._items.keys())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def rename(
        self, namespace: str, old_key: str, new_key: str, overwrite: bool = False
    ) -> bool:
        """
        Rename a key within a namespace.

        Parameters:
            namespace (str): Namespace containing the key.
            old_key (str): Existing key name.
            new_key (str): New key name.
            overwrite (bool): If False (default) and *new_key* already exists,
                a warning is printed and the rename is aborted. Set to True to
                silently overwrite the existing entry.

        Returns:
            success (bool): True if renamed, False if namespace or
                old_key not found or rename was blocked by existing key.
        """
        if namespace not in self._items or old_key not in self._items[namespace]:
            return False
        if not overwrite and new_key in self._items[namespace]:
            warnings.warn(
                f"Key '{new_key}' already exists in namespace '{namespace}'. "
                "Use overwrite=True to replace it.",
                UserWarning,
            )
            return False
        self._items[namespace][new_key] = self._items[namespace].pop(old_key)
        self._index[namespace][new_key] = self._index[namespace].pop(old_key)
        return True

    def add_note(self, namespace: str, key: str, note: str) -> bool:
        """
        Add or replace the note attached to a stored item.

        Parameters:
            namespace (str): Namespace of the item.
            key (str): Key of the item.
            note (str): Note text to attach.

        Returns:
            success (bool): True if updated, False if item not found.
        """
        if namespace not in self._index or key not in self._index[namespace]:
            return False
        self._index[namespace][key]["note"] = note
        return True

    def delete(self, namespace: str, key: Optional[str] = None) -> bool:
        """
        Delete a single item or an entire namespace.

        Parameters:
            namespace (str): Namespace to delete from.
            key (str | None): Key to delete. If None, the entire namespace
                and all its contents are deleted.

        Returns:
            success (bool): True if deleted, False if not found.
        """
        if namespace not in self._items:
            return False
        if key is None:
            del self._items[namespace]
            del self._index[namespace]
            return True
        if key not in self._items[namespace]:
            return False
        del self._items[namespace][key]
        del self._index[namespace][key]
        if not self._items[namespace]:
            del self._items[namespace]
            del self._index[namespace]
        return True

    def merge_from(self, other: "AnalysisWorkspace", overwrite: bool = False) -> dict:
        """
        Copy all items from another workspace into this one.

        Parameters:
            other (AnalysisWorkspace): Source workspace to merge from.
                May be a regular or lazy workspace.
            overwrite (bool): If True, existing (namespace, key) pairs in
                this workspace are replaced by the incoming values. If
                False (default), existing keys are kept and incoming
                duplicates are skipped.

        Returns:
            result (dict): Summary with keys ``merged`` (int),
                ``skipped`` (int), and ``skipped_keys``
                (list[tuple[str, str]]).

        Notes:
            - HDF5 does not support concurrent writes to the same file.
              When multiple processes (e.g. parallel Claude Code instances
              or MCP agents) need to store analysis results, each process
              should create its own workspace and save to a separate file.
              After all processes finish, a single orchestrator loads each
              file and calls ``merge_from`` to combine the results::

                  ws_main = AnalysisWorkspace(name="combined")
                  for path in agent_output_paths:
                      ws_main.merge_from(AnalysisWorkspace.load(path))
                  ws_main.save("path/to/combined_workspace")

            - Only object data and notes are copied. The source
              workspace's ``workspace_id``, ``name``, and ``created_at``
              are not transferred.
        """
        merged = 0
        skipped = 0
        skipped_keys: list = []

        for namespace in other.list_namespaces():
            for key in other.list_keys(namespace):
                if (
                    not overwrite
                    and namespace in self._index
                    and key in self._index.get(namespace, {})
                ):
                    skipped += 1
                    skipped_keys.append((namespace, key))
                    continue

                obj = other.get(namespace, key)
                other_info = other.get_info(namespace, key)
                note = other_info.get("note") if other_info else None
                self.store(namespace, key, obj, note=note)
                merged += 1

        return {"merged": merged, "skipped": skipped, "skipped_keys": skipped_keys}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the workspace to disk.

        Writes two files: ``{path}.h5`` (full object data, HDF5) and
        ``{path}.json`` (index/metadata, human-readable).  All stored
        objects are serialised to their constituent arrays so that
        individual items can be loaded selectively without reading the
        entire file.

        Parameters:
            path (str): Base path without file extension.
        """
        from .hdf5_io import dump_workspace

        dump_workspace(self, path)

        json_path = f"{path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "workspace_id": self.workspace_id,
                    "name": self.name,
                    "created_at": self.created_at,
                    "index": self._index,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "AnalysisWorkspace":
        """
        Load a workspace from disk, reconstructing all stored objects to
        their original IAT data class types.

        Parameters:
            path (str): Base path without file extension (the same value
                that was passed to ``save``).

        Returns:
            workspace (AnalysisWorkspace): Reconstructed workspace instance.
        """
        from .hdf5_io import load_workspace_full

        return load_workspace_full(path)

    @classmethod
    def load_item(cls, path: str, namespace: str, key: str) -> Any:
        """
        Load a single item from a saved workspace file without reading
        the entire workspace into memory.

        Parameters:
            path (str): Base path without file extension.
            namespace (str): Namespace the item was stored under.
            key (str): Key the item was stored under.

        Returns:
            obj: Reconstructed IAT data object or numpy array.
        """
        from .hdf5_io import load_workspace_item

        return load_workspace_item(path, namespace, key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def comparison_namespace(*namespaces: str) -> str:
        """
        Build a conventional namespace string for cross-recording comparisons.

        Parameters:
            *namespaces (str): Names of the recording namespaces involved
                in the comparison (in any order).

        Returns:
            name (str): A string of the form ``C_ns1_ns2_...``.

        Notes:
            - By convention, pass the same namespace strings used when
              storing the individual recording results.
        """
        return "C_" + "_".join(namespaces)

    def __repr__(self) -> str:
        ns_count = len(self._items)
        item_count = sum(len(v) for v in self._items.values())
        name_part = f" {self.name!r}" if self.name else ""
        return (
            f"AnalysisWorkspace{name_part}("
            f"id={self.workspace_id[:8]}…, "
            f"{ns_count} namespace(s), {item_count} item(s))"
        )


class LazyAnalysisWorkspace(AnalysisWorkspace):
    """
    Disk-backed variant of AnalysisWorkspace for low-RAM environments.

    Each stored object is immediately serialised to a temporary HDF5 file and
    removed from process memory. Only the lightweight index metadata is kept in
    RAM. Objects are deserialised from the temp file on each ``get()`` call.

    Use this when working with large recordings and limited available RAM.
    The temp file is deleted automatically when the workspace is garbage-collected.

    Notes:
        - Requires h5py. If h5py is not installed, construction will raise
          ImportError.
        - Every ``get()`` call performs a disk read; repeated access to the same
          item is slower than with the default in-memory workspace.
        - ``save()`` copies the temp file to the destination path, so it is as
          fast as a file copy rather than a full re-serialisation.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Create a new empty lazy workspace backed by a temp HDF5 file.

        Parameters:
            name (str | None): Optional human-readable label.
        """
        from .hdf5_io import _require_h5py

        _require_h5py()
        super().__init__(name=name)

        # Create the backing temp file.
        fd, self._h5_path = tempfile.mkstemp(suffix=".h5", prefix="iat_lazy_ws_")
        os.close(fd)

        try:
            import h5py

            with h5py.File(self._h5_path, "w") as f:
                f.attrs["__workspace_id__"] = self.workspace_id
                f.attrs["__workspace_name__"] = self.name or ""
                f.attrs["__created_at__"] = self.created_at
        except Exception:
            os.unlink(self._h5_path)
            raise

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        namespace: str,
        key: str,
        obj: Any,
        note: Optional[str] = None,
    ) -> None:
        """
        Serialise an object to the temp HDF5 file and record it in the index.

        Parameters:
            namespace (str): Namespace grouping related results.
            key (str): Key identifying this result within the namespace.
            obj: Object to store.
            note (str | None): Optional free-text annotation.
        """
        from .hdf5_io import dump_item_to_file

        if namespace not in self._index:
            self._index[namespace] = {}

        entry = _make_summary(obj)
        created_at = time.time()
        entry["created_at"] = created_at
        if note is not None:
            entry["note"] = note
        self._index[namespace][key] = entry

        dump_item_to_file(self._h5_path, namespace, key, obj, created_at, note)

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Deserialise and return a stored object from the temp HDF5 file.

        Parameters:
            namespace (str): Namespace the object was stored under.
            key (str): Key the object was stored under.

        Returns:
            obj: Reconstructed object, or None if not found.
        """
        from .hdf5_io import load_item_from_file

        if namespace not in self._index or key not in self._index[namespace]:
            return None
        try:
            return load_item_from_file(self._h5_path, namespace, key)
        except KeyError:
            return None

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_keys(self, namespace: Optional[str] = None) -> "dict | list":
        """
        List stored keys from the index (no disk read required).

        Parameters:
            namespace (str | None): If provided, returns the list of keys for
                that namespace. If None, returns a dict mapping each namespace
                to its list of keys.

        Returns:
            keys (dict | list): Key listing derived from the in-memory index.
        """
        if namespace is not None:
            return list(self._index.get(namespace, {}).keys())
        return {ns: list(keys.keys()) for ns, keys in self._index.items()}

    def list_namespaces(self) -> list:
        """
        Return the names of all top-level namespaces in the workspace.

        Returns:
            namespaces (list[str]): List of namespace names derived from the in-memory index.
        """
        return list(self._index.keys())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def rename(
        self, namespace: str, old_key: str, new_key: str, overwrite: bool = False
    ) -> bool:
        """
        Rename a key within a namespace.

        Parameters:
            namespace (str): Namespace containing the key.
            old_key (str): Existing key name.
            new_key (str): New key name.
            overwrite (bool): If False (default) and *new_key* already exists,
                a warning is printed and the rename is aborted. Set to True to
                silently overwrite the existing entry.

        Returns:
            success (bool): True if renamed, False if namespace or old_key
                not found or rename was blocked by existing key.
        """
        from .hdf5_io import delete_item_from_file, dump_item_to_file

        if namespace not in self._index or old_key not in self._index[namespace]:
            return False
        if not overwrite and new_key in self._index[namespace]:
            warnings.warn(
                f"Key '{new_key}' already exists in namespace '{namespace}'. "
                "Use overwrite=True to replace it.",
                UserWarning,
            )
            return False

        obj = self.get(namespace, old_key)
        if obj is None:
            return False

        old_entry = self._index[namespace][old_key]
        dump_item_to_file(
            self._h5_path,
            namespace,
            new_key,
            obj,
            old_entry["created_at"],
            old_entry.get("note"),
        )
        delete_item_from_file(self._h5_path, namespace, old_key)

        self._index[namespace][new_key] = self._index[namespace].pop(old_key)
        return True

    def delete(self, namespace: str, key: Optional[str] = None) -> bool:
        """
        Delete a single item or an entire namespace from the temp file and index.

        Parameters:
            namespace (str): Namespace to delete from.
            key (str | None): Key to delete. If None, deletes the entire namespace.

        Returns:
            success (bool): True if deleted, False if not found.
        """
        from .hdf5_io import delete_item_from_file

        if namespace not in self._index:
            return False
        if key is None:
            del self._index[namespace]
            delete_item_from_file(self._h5_path, namespace)
            return True
        if key not in self._index[namespace]:
            return False
        del self._index[namespace][key]
        if not self._index[namespace]:
            del self._index[namespace]
        delete_item_from_file(self._h5_path, namespace, key)
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the workspace to disk by copying the temp HDF5 file.

        Parameters:
            path (str): Base path without file extension.
        """
        shutil.copy2(self._h5_path, f"{path}.h5")
        json_path = f"{path}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "workspace_id": self.workspace_id,
                    "name": self.name,
                    "created_at": self.created_at,
                    "index": self._index,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        try:
            if hasattr(self, "_h5_path") and os.path.exists(self._h5_path):
                os.unlink(self._h5_path)
        except Exception:
            pass

    def __repr__(self) -> str:
        ns_count = len(self._index)
        item_count = sum(len(v) for v in self._index.values())
        name_part = f" {self.name!r}" if self.name else ""
        return (
            f"LazyAnalysisWorkspace{name_part}("
            f"id={self.workspace_id[:8]}…, "
            f"{ns_count} namespace(s), {item_count} item(s), "
            f"backed by temp HDF5)"
        )


class WorkspaceManager:
    """
    Registry for multiple AnalysisWorkspace instances within a single process.

    Provides create, retrieve, delete, list, save, and load operations.
    Use ``get_workspace_manager()`` to access the module-level singleton.
    """

    def __init__(self) -> None:
        """Initialize an empty WorkspaceManager."""
        import threading

        self._workspaces: Dict[str, AnalysisWorkspace] = {}
        self._lock = threading.Lock()

    def create_workspace(self, name: Optional[str] = None, lazy: bool = False) -> str:
        """
        Create and register a new empty workspace.

        Parameters:
            name (str | None): Optional human-readable label.
            lazy (bool): If True, creates a disk-backed LazyAnalysisWorkspace
                that serialises each item to a temp HDF5 file on store() and
                deserialises on get(). Only index metadata is kept in RAM.
                Use this when working with large recordings and limited RAM.
                Requires h5py. Defaults to False (fully in-memory).

        Returns:
            workspace_id (str): UUID of the new workspace.
        """
        if lazy:
            ws = LazyAnalysisWorkspace(name=name)
        else:
            ws = AnalysisWorkspace(name=name)
        with self._lock:
            self._workspaces[ws.workspace_id] = ws
        return ws.workspace_id

    def get_workspace(self, workspace_id: str) -> Optional[AnalysisWorkspace]:
        """
        Retrieve a workspace by ID.

        Parameters:
            workspace_id (str): UUID of the workspace.

        Returns:
            workspace (AnalysisWorkspace | None): The workspace, or None
                if not found.
        """
        with self._lock:
            return self._workspaces.get(workspace_id)

    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace and all its contents.

        Parameters:
            workspace_id (str): UUID of the workspace to delete.

        Returns:
            success (bool): True if deleted, False if not found.
        """
        with self._lock:
            if workspace_id in self._workspaces:
                del self._workspaces[workspace_id]
                return True
            return False

    def list_workspaces(self) -> List[dict]:
        """
        List all registered workspaces with summary information.

        Returns:
            workspaces (list[dict]): Each entry contains workspace_id, name,
                created_at, namespace_count, and item_count.
        """
        with self._lock:
            result = []
            for ws in self._workspaces.values():
                index = ws._index if hasattr(ws, "_index") else ws._items
                item_count = sum(len(v) for v in index.values())
                result.append(
                    {
                        "workspace_id": ws.workspace_id,
                        "name": ws.name,
                        "created_at": ws.created_at,
                        "namespace_count": len(index),
                        "item_count": item_count,
                    }
                )
            return result

    def save_workspace(self, workspace_id: str, path: str) -> None:
        """
        Save a workspace to disk.

        Parameters:
            workspace_id (str): UUID of the workspace to save.
            path (str): Base path without file extension (passed through
                to ``AnalysisWorkspace.save``).

        Notes:
            - Raises KeyError if workspace_id is not registered.
        """
        ws = self._workspaces[workspace_id]
        ws.save(path)

    def load_workspace(self, path: str) -> str:
        """
        Load a workspace from disk and register it in the manager,
        reconstructing all stored objects to their original IAT data class
        types.

        Parameters:
            path (str): Base path without file extension (the same value
                that was passed to ``save``).

        Returns:
            workspace_id (str): UUID of the loaded workspace.

        Notes:
            - If a workspace with the same ID is already registered, it
              will be overwritten by the loaded version.
        """
        ws = AnalysisWorkspace.load(path)
        with self._lock:
            self._workspaces[ws.workspace_id] = ws
        return ws.workspace_id

    def load_workspace_item(
        self, path: str, namespace: str, key: str, workspace_id: str
    ) -> None:
        """
        Load a single item from a saved workspace file and store it in an
        already-registered in-memory workspace, reconstructing the original
        IAT data class.

        Parameters:
            path (str): Base path without file extension.
            namespace (str): Namespace the item was stored under.
            key (str): Key the item was stored under.
            workspace_id (str): ID of the in-memory workspace to store the
                loaded item into.

        Notes:
            - Raises KeyError if workspace_id is not registered.
            - Raises KeyError if namespace or key is not found in the file.
        """
        with self._lock:
            ws = self._workspaces[workspace_id]
        obj = AnalysisWorkspace.load_item(path, namespace, key)
        ws.store(namespace, key, obj)


# Module-level singleton
_workspace_manager: Optional[WorkspaceManager] = None
_workspace_manager_lock: "threading.Lock | None" = None


def _get_singleton_lock():
    """Lazily create the module-level lock (avoids top-level threading import)."""
    global _workspace_manager_lock
    if _workspace_manager_lock is None:
        import threading

        _workspace_manager_lock = threading.Lock()
    return _workspace_manager_lock


def get_workspace_manager() -> WorkspaceManager:
    """
    Return the global WorkspaceManager singleton.

    Returns:
        manager (WorkspaceManager): The global instance, created on first call.
    """
    global _workspace_manager
    if _workspace_manager is None:
        with _get_singleton_lock():
            if _workspace_manager is None:
                _workspace_manager = WorkspaceManager()
    return _workspace_manager
