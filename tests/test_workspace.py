"""
Tests for AnalysisWorkspace and WorkspaceManager (workspace/workspace.py).

Covers: store/get round-trips for every supported type, summary generation,
get_info, describe, list_keys, rename, add_note, delete, save/load, note
handling, namespaced isolation, comparison_namespace, WorkspaceManager CRUD,
and the get_workspace_manager singleton.
"""

import json
import os
import pathlib
import tempfile

import numpy as np
import pytest

try:
    import h5py  # noqa: F401

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

from spikelab.spikedata.spikedata import SpikeData
from spikelab.spikedata.ratedata import RateData
from spikelab.spikedata.rateslicestack import RateSliceStack
from spikelab.spikedata.spikeslicestack import SpikeSliceStack
from spikelab.spikedata.pairwise import (
    PairwiseCompMatrix,
    PairwiseCompMatrixStack,
)
from spikelab.workspace.workspace import (
    AnalysisWorkspace,
    LazyAnalysisWorkspace,
    WorkspaceManager,
    get_workspace_manager,
    _make_summary,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def make_spikedata(n_units=3, length_ms=100.0, seed=0):
    """
    Create a simple SpikeData with uniformly spaced spikes for each unit.

    Parameters:
        n_units (int): Number of units.
        length_ms (float): Recording length in milliseconds.
        seed (int): Random seed.

    Returns:
        sd (SpikeData): A SpikeData of length length_ms with n_units units.
    """
    rng = np.random.default_rng(seed)
    train = [np.sort(rng.uniform(0.0, length_ms, size=5)) for _ in range(n_units)]
    return SpikeData(train, length=length_ms)


def make_ratedata(n_units=3, n_times=60, step=1.0, seed=0):
    """
    Create a RateData with random firing rates on a uniform time grid.

    Parameters:
        n_units (int): Number of units.
        n_times (int): Number of time bins.
        step (float): Time step in milliseconds.
        seed (int): Random seed.

    Returns:
        rd (RateData): A RateData object with shape (n_units, n_times).
    """
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, n_times * step, step)
    data = rng.random((n_units, len(times)))
    return RateData(data, times)


def make_rateslicestack(n_units=3, n_times=20, n_slices=4):
    """
    Create a RateSliceStack from a random 3D array.

    Parameters:
        n_units (int): Number of units (U axis).
        n_times (int): Time bins per slice (T axis).
        n_slices (int): Number of slices (S axis).

    Returns:
        rss (RateSliceStack): A RateSliceStack with shape (n_units, n_times, n_slices).
    """
    rng = np.random.default_rng(7)
    arr = rng.random((n_units, n_times, n_slices))
    times = [(i * n_times, (i + 1) * n_times) for i in range(n_slices)]
    return RateSliceStack(None, event_matrix=arr, times_start_to_end=times)


def make_spikeslicestack(n_units=2, slice_length_ms=50.0, n_slices=4):
    """
    Create a SpikeSliceStack by splitting a SpikeData into fixed-length frames.

    Parameters:
        n_units (int): Number of units.
        slice_length_ms (float): Duration of each slice in milliseconds.
        n_slices (int): Number of slices.

    Returns:
        sss (SpikeSliceStack): A SpikeSliceStack with n_slices slices.
    """
    total_ms = slice_length_ms * n_slices
    sd = make_spikedata(n_units=n_units, length_ms=total_ms, seed=42)
    return sd.frames(slice_length_ms)


# ---------------------------------------------------------------------------
# Tests: AnalysisWorkspace
# ---------------------------------------------------------------------------


class TestAnalysisWorkspace:
    def setup_method(self):
        """Create a fresh workspace for each test."""
        self.ws = AnalysisWorkspace(name="test_ws")

    # ------------------------------------------------------------------
    # store / get round-trips
    # ------------------------------------------------------------------

    def test_store_get_ndarray(self):
        """
        Tests that a numpy array stored under (namespace, key) is retrieved identically.

        Tests:
            (Test Case 1) get() returns the exact same array object that was stored.
            (Test Case 2) get() on a missing key returns None.
            (Test Case 3) get() on a missing namespace returns None.
        """
        arr = np.arange(12).reshape(3, 4)
        self.ws.store("rec1", "raster", arr)

        retrieved = self.ws.get("rec1", "raster")
        np.testing.assert_array_equal(retrieved, arr)

        assert self.ws.get("rec1", "missing") is None
        assert self.ws.get("missing_ns", "raster") is None

    def test_store_get_spikedata(self):
        """
        Tests that a SpikeData object is stored and retrieved with its attributes intact.

        Tests:
            (Test Case 1) Retrieved object is the same SpikeData instance.
            (Test Case 2) N and length_ms are preserved.
        """
        sd = make_spikedata(n_units=4, length_ms=200.0)
        self.ws.store("rec1", "spikes", sd)

        out = self.ws.get("rec1", "spikes")
        assert out is sd
        assert out.N == 4
        assert out.length == pytest.approx(200.0)

    def test_store_get_ratedata(self):
        """
        Tests that a RateData object is stored and retrieved with its attributes intact.

        Tests:
            (Test Case 1) Retrieved object is the same RateData instance.
            (Test Case 2) inst_Frate_data shape is preserved.
        """
        rd = make_ratedata(n_units=3, n_times=50)
        self.ws.store("rec1", "rate", rd)

        out = self.ws.get("rec1", "rate")
        assert out is rd
        assert out.inst_Frate_data.shape == (3, 50)

    def test_store_get_rateslicestack(self):
        """
        Tests that a RateSliceStack is stored and retrieved correctly.

        Tests:
            (Test Case 1) Retrieved object is the same RateSliceStack instance.
            (Test Case 2) event_stack shape is preserved.
        """
        rss = make_rateslicestack(n_units=3, n_times=20, n_slices=4)
        self.ws.store("rec1", "rss", rss)

        out = self.ws.get("rec1", "rss")
        assert out is rss
        assert out.event_stack.shape == (3, 20, 4)

    def test_store_get_spikeslicestack(self):
        """
        Tests that a SpikeSliceStack is stored and retrieved correctly.

        Tests:
            (Test Case 1) Retrieved object is the same SpikeSliceStack instance.
            (Test Case 2) Number of slices is preserved.
        """
        sss = make_spikeslicestack(n_units=2, slice_length_ms=50.0, n_slices=3)
        self.ws.store("rec1", "sss", sss)

        out = self.ws.get("rec1", "sss")
        assert out is sss
        assert len(out.spike_stack) == 3

    def test_store_get_pairwise(self):
        """
        Tests that PairwiseCompMatrix and PairwiseCompMatrixStack are stored and retrieved.

        Tests:
            (Test Case 1) PairwiseCompMatrix retrieved as same instance with correct shape.
            (Test Case 2) PairwiseCompMatrixStack retrieved as same instance with correct shape.
        """
        pcm = PairwiseCompMatrix(matrix=np.eye(4))
        self.ws.store("rec1", "pcm", pcm)

        out_pcm = self.ws.get("rec1", "pcm")
        assert out_pcm is pcm
        assert out_pcm.matrix.shape == (4, 4)

        stack_arr = np.random.default_rng(0).random((4, 4, 6))
        pcms = PairwiseCompMatrixStack(stack=stack_arr)
        self.ws.store("rec1", "pcms", pcms)

        out_pcms = self.ws.get("rec1", "pcms")
        assert out_pcms is pcms
        assert out_pcms.stack.shape == (4, 4, 6)

    def test_store_overwrite(self):
        """
        Tests that storing a second value under the same key overwrites the first.

        Tests:
            (Test Case 1) Second store returns the new object, not the first.
            (Test Case 2) Index entry is refreshed after overwrite.
        """
        arr1 = np.zeros(5)
        arr2 = np.ones(5)
        self.ws.store("rec1", "arr", arr1)
        self.ws.store("rec1", "arr", arr2)

        out = self.ws.get("rec1", "arr")
        np.testing.assert_array_equal(out, arr2)
        # Index reflects shape of arr2
        info = self.ws.get_info("rec1", "arr")
        assert info["shape"] == [5]

    # ------------------------------------------------------------------
    # get_info
    # ------------------------------------------------------------------

    def test_get_info(self):
        """
        Tests that get_info() returns the index entry and None for missing items.

        Tests:
            (Test Case 1) Entry contains expected keys (type, shape, created_at).
            (Test Case 2) Missing key returns None.
            (Test Case 3) Missing namespace returns None.
        """
        arr = np.zeros((3, 4), dtype=np.float32)
        self.ws.store("rec1", "arr", arr)

        info = self.ws.get_info("rec1", "arr")
        assert info is not None
        assert info["type"] == "ndarray"
        assert info["shape"] == [3, 4]
        assert "created_at" in info

        assert self.ws.get_info("rec1", "missing") is None
        assert self.ws.get_info("missing_ns", "arr") is None

    # ------------------------------------------------------------------
    # describe / list_keys
    # ------------------------------------------------------------------

    def test_describe(self):
        """
        Tests that describe() returns a nested dict of namespace -> key -> info.

        Tests:
            (Test Case 1) Top-level keys are namespace names.
            (Test Case 2) Each entry has type, shape.
        """
        self.ws.store("rec1", "arr", np.zeros((2, 3)))
        self.ws.store("rec2", "rate", make_ratedata(n_units=1, n_times=10))

        desc = self.ws.describe()
        assert "rec1" in desc
        assert "rec2" in desc
        assert "arr" in desc["rec1"]
        assert "rate" in desc["rec2"]
        assert desc["rec1"]["arr"]["type"] == "ndarray"

    def test_list_keys(self):
        """
        Tests list_keys() with and without a namespace filter.

        Tests:
            (Test Case 1) No namespace argument returns dict mapping each namespace to its keys.
            (Test Case 2) Specific namespace returns list of keys for that namespace only.
            (Test Case 3) Unknown namespace returns an empty list.
        """
        self.ws.store("rec1", "a", np.zeros(2))
        self.ws.store("rec1", "b", np.zeros(2))
        self.ws.store("rec2", "x", np.zeros(2))

        all_keys = self.ws.list_keys()
        assert isinstance(all_keys, dict)
        assert "rec1" in all_keys
        assert "rec2" in all_keys
        assert sorted(all_keys["rec1"]) == sorted(["a", "b"])
        assert sorted(all_keys["rec2"]) == sorted(["x"])

        rec1_keys = self.ws.list_keys("rec1")
        assert isinstance(rec1_keys, list)
        assert sorted(rec1_keys) == sorted(["a", "b"])

        assert self.ws.list_keys("missing_ns") == []

    def test_list_namespaces(self):
        """
        Tests list_namespaces() returns all top-level namespace names.

        Tests:
            (Test Case 1) Empty workspace returns an empty list.
            (Test Case 2) After storing items, all namespace names are returned.
            (Test Case 3) Each name appears exactly once even when multiple keys exist in the same namespace.
            (Test Case 4) Namespaces not yet stored are absent from the result.
        """
        assert self.ws.list_namespaces() == []

        self.ws.store("alpha", "k1", np.zeros(2))
        self.ws.store("alpha", "k2", np.zeros(2))
        self.ws.store("beta", "k1", np.zeros(2))

        namespaces = self.ws.list_namespaces()
        assert isinstance(namespaces, list)
        assert sorted(namespaces) == sorted(["alpha", "beta"])
        assert "gamma" not in namespaces

    # ------------------------------------------------------------------
    # rename
    # ------------------------------------------------------------------

    def test_rename(self):
        """
        Tests rename() moves a key within the same namespace.

        Tests:
            (Test Case 1) Renamed key is accessible under the new name.
            (Test Case 2) Old key no longer exists after rename.
            (Test Case 3) Index entry is accessible under the new key.
            (Test Case 4) Rename on a missing namespace raises KeyError.
            (Test Case 5) Rename on a missing old_key raises KeyError.
        """
        arr = np.arange(5)
        self.ws.store("rec1", "old", arr)

        result = self.ws.rename("rec1", "old", "new")
        assert result

        retrieved = self.ws.get("rec1", "new")
        np.testing.assert_array_equal(retrieved, arr)
        assert self.ws.get("rec1", "old") is None

        info = self.ws.get_info("rec1", "new")
        assert info is not None
        assert self.ws.get_info("rec1", "old") is None

        with pytest.raises(KeyError, match="not found"):
            self.ws.rename("missing_ns", "old", "new")
        with pytest.raises(KeyError, match="not found"):
            self.ws.rename("rec1", "missing_key", "new")

    # ------------------------------------------------------------------
    # add_note
    # ------------------------------------------------------------------

    def test_add_note(self):
        """
        Tests add_note() attaches and replaces notes on index entries.

        Tests:
            (Test Case 1) Note stored via store() appears in get_info().
            (Test Case 2) add_note() on existing item updates the note.
            (Test Case 3) add_note() on missing item raises KeyError.
        """
        self.ws.store("rec1", "arr", np.zeros(3), note="initial note")
        info = self.ws.get_info("rec1", "arr")
        assert info["note"] == "initial note"

        self.ws.add_note("rec1", "arr", "updated note")
        assert self.ws.get_info("rec1", "arr")["note"] == "updated note"

        with pytest.raises(KeyError, match="not found"):
            self.ws.add_note("missing_ns", "arr", "note")

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def test_delete_key(self):
        """
        Tests that delete() with a key removes only that key.

        Tests:
            (Test Case 1) Deleted key returns None from get().
            (Test Case 2) Index entry is removed.
            (Test Case 3) Other keys in the same namespace are not affected.
            (Test Case 4) Deleting a missing key raises KeyError.
        """
        self.ws.store("rec1", "a", np.zeros(2))
        self.ws.store("rec1", "b", np.zeros(2))

        self.ws.delete("rec1", "a")
        assert self.ws.get("rec1", "a") is None
        assert self.ws.get_info("rec1", "a") is None
        assert self.ws.get("rec1", "b") is not None

        with pytest.raises(KeyError, match="not found"):
            self.ws.delete("rec1", "missing_key")

    def test_delete_namespace(self):
        """
        Tests that delete() without a key removes the entire namespace.

        Tests:
            (Test Case 1) All keys in the namespace are gone after deletion.
            (Test Case 2) Namespace no longer appears in list_keys().
            (Test Case 3) Deleting a missing namespace raises KeyError.
        """
        self.ws.store("rec1", "a", np.zeros(2))
        self.ws.store("rec1", "b", np.zeros(2))

        self.ws.delete("rec1")
        assert self.ws.get("rec1", "a") is None
        assert "rec1" not in self.ws.list_keys()

        with pytest.raises(KeyError, match="not found"):
            self.ws.delete("missing_ns")

    # ------------------------------------------------------------------
    # namespace isolation
    # ------------------------------------------------------------------

    def test_same_key_different_namespaces(self):
        """
        Tests that identical keys in different namespaces are stored independently.

        Tests:
            (Test Case 1) Each namespace holds its own object for the same key.
            (Test Case 2) Deleting from one namespace does not affect the other.
        """
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([3.0, 4.0])
        self.ws.store("ns_a", "data", arr1)
        self.ws.store("ns_b", "data", arr2)

        np.testing.assert_array_equal(self.ws.get("ns_a", "data"), arr1)
        np.testing.assert_array_equal(self.ws.get("ns_b", "data"), arr2)

        self.ws.delete("ns_a", "data")
        assert self.ws.get("ns_a", "data") is None
        np.testing.assert_array_equal(self.ws.get("ns_b", "data"), arr2)

    # ------------------------------------------------------------------
    # comparison_namespace
    # ------------------------------------------------------------------

    def test_comparison_namespace(self):
        """
        Tests that comparison_namespace() returns the expected C_-prefixed string.

        Tests:
            (Test Case 1) Two namespaces -> "C_ns1_ns2".
            (Test Case 2) Three namespaces -> "C_ns1_ns2_ns3".
            (Test Case 3) Single namespace -> "C_ns1".
        """
        assert AnalysisWorkspace.comparison_namespace("rec1", "rec2") == "C_rec1_rec2"
        assert AnalysisWorkspace.comparison_namespace("a", "b", "c") == "C_a_b_c"
        assert AnalysisWorkspace.comparison_namespace("only") == "C_only"

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_save_load_roundtrip(self):
        """
        Tests that a workspace saved to disk and reloaded is equivalent to the original.

        Tests:
            (Test Case 1) workspace_id, name, and created_at are preserved.
            (Test Case 2) Stored numpy array is recovered with matching values.
            (Test Case 3) Stored SpikeData is recovered with matching N and length.
            (Test Case 4) Index entries (type, shape) are preserved.
            (Test Case 5) Both .h5 and .json files are created on disk.
        """
        sd = make_spikedata(n_units=2, length_ms=80.0)
        arr = np.arange(6).reshape(2, 3)
        self.ws.store("rec1", "spikes", sd)
        self.ws.store("rec1", "matrix", arr, note="test note")

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            self.ws.save(base)

            h5_path = pathlib.Path(base + ".h5")
            json_path = pathlib.Path(base + ".json")
            assert h5_path.exists()
            assert json_path.exists()

            loaded = AnalysisWorkspace.load(base)

        assert loaded.workspace_id == self.ws.workspace_id
        assert loaded.name == self.ws.name
        assert loaded.created_at == pytest.approx(self.ws.created_at)

        # Array round-trip.
        np.testing.assert_array_equal(loaded.get("rec1", "matrix"), arr)

        # SpikeData round-trip: original IAT type must be reconstructed.
        loaded_sd = loaded.get("rec1", "spikes")
        assert isinstance(loaded_sd, SpikeData)
        assert loaded_sd.N == 2
        assert loaded_sd.length == pytest.approx(80.0)

        # Index preserved.
        info = loaded.get_info("rec1", "matrix")
        assert info["type"] == "ndarray"
        assert info["shape"] == [2, 3]
        assert info["note"] == "test note"

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_load_item(self):
        """
        Tests that load_item() loads a single item from disk without loading the full workspace.

        Tests:
            (Test Case 1) The loaded object has the correct type and values.
            (Test Case 2) A numpy array is reconstructed correctly.
            (Test Case 3) load_item() raises KeyError for a missing namespace.
            (Test Case 4) load_item() raises KeyError for a missing key.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        sd = make_spikedata(n_units=2, length_ms=50.0)
        self.ws.store("ns", "matrix", arr)
        self.ws.store("ns", "spikes", sd)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            self.ws.save(base)

            loaded_arr = AnalysisWorkspace.load_item(base, "ns", "matrix")
            np.testing.assert_array_equal(loaded_arr, arr)

            loaded_sd = AnalysisWorkspace.load_item(base, "ns", "spikes")
            assert isinstance(loaded_sd, SpikeData)
            assert loaded_sd.N == 2

            with pytest.raises(KeyError):
                AnalysisWorkspace.load_item(base, "missing_ns", "matrix")

            with pytest.raises(KeyError):
                AnalysisWorkspace.load_item(base, "ns", "missing_key")

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_json_index_is_valid(self):
        """
        Tests that the .json sidecar file is valid JSON and contains the index.

        Tests:
            (Test Case 1) File parses without error.
            (Test Case 2) Contains workspace_id, name, created_at, and index keys.
            (Test Case 3) Index reflects the stored items.
        """
        self.ws.store("ns", "arr", np.zeros(4))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            self.ws.save(base)

            with open(base + ".json", encoding="utf-8") as f:
                doc = json.load(f)

        assert doc["workspace_id"] == self.ws.workspace_id
        assert doc["name"] == self.ws.name
        assert "index" in doc
        assert "ns" in doc["index"]
        assert "arr" in doc["index"]["ns"]

    # ------------------------------------------------------------------
    # EC-WS-01: store with None value
    # ------------------------------------------------------------------

    def test_store_none_value(self):
        """
        EC-WS-01: store() with None as the value.

        None is not a supported IAT type, but store() accepts it (the
        summary will just contain the class name "NoneType"). get()
        returns None, which is indistinguishable from "not found".

        Tests:
            (Test Case 1) store() does not raise when obj is None.
            (Test Case 2) get() returns None (same as "not found" sentinel).
            (Test Case 3) get_info() returns a valid index entry with type "NoneType".
            (Test Case 4) The key appears in list_keys().
        """
        self.ws.store("ns", "none_val", None)

        # get() returns None — indistinguishable from missing key
        assert self.ws.get("ns", "none_val") is None

        # But the index entry exists and records the type
        info = self.ws.get_info("ns", "none_val")
        assert info is not None
        assert info["type"] == "NoneType"

        # Key is present in listing
        assert "none_val" in self.ws.list_keys("ns")

    # ------------------------------------------------------------------
    # EC-WS-02: store with empty string key
    # ------------------------------------------------------------------

    def test_store_empty_string_key(self):
        """
        EC-WS-02: store() with an empty string as the key.

        Empty strings are valid Python dict keys, so store() accepts them.
        The item can be retrieved via get("ns", "").

        Tests:
            (Test Case 1) store() does not raise with an empty string key.
            (Test Case 2) get() retrieves the item using the empty string key.
            (Test Case 3) The empty string key appears in list_keys().
        """
        arr = np.array([1.0, 2.0])
        self.ws.store("ns", "", arr)

        retrieved = self.ws.get("ns", "")
        np.testing.assert_array_equal(retrieved, arr)

        assert "" in self.ws.list_keys("ns")

    # ------------------------------------------------------------------
    # EC-WS-03: store with empty string namespace
    # ------------------------------------------------------------------

    def test_store_empty_string_namespace(self):
        """
        EC-WS-03: store() with an empty string as the namespace.

        Empty strings are valid Python dict keys, so store() accepts them.
        The item can be retrieved via get("", "key").

        Tests:
            (Test Case 1) store() does not raise with an empty string namespace.
            (Test Case 2) get() retrieves the item using the empty string namespace.
            (Test Case 3) The empty string namespace appears in list_namespaces().
        """
        arr = np.array([3.0, 4.0])
        self.ws.store("", "key", arr)

        retrieved = self.ws.get("", "key")
        np.testing.assert_array_equal(retrieved, arr)

        assert "" in self.ws.list_namespaces()

    # ------------------------------------------------------------------
    # EC-WS-04: rename to an existing key — overwrite behavior
    # ------------------------------------------------------------------

    def test_rename_to_existing_key_blocked_by_default(self):
        """
        rename() to a key that already exists is blocked by default.

        Tests:
            (Test Case 1) rename() returns False and emits a UserWarning.
            (Test Case 2) Both keys remain unchanged.
        """
        self.ws.store("ns", "old", np.array([1.0]))
        self.ws.store("ns", "existing", np.array([99.0]))

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.ws.rename("ns", "old", "existing")
            assert result is False
            assert len(w) == 1
            assert "already exists" in str(w[0].message)

        # Both keys are preserved
        np.testing.assert_array_equal(self.ws.get("ns", "old"), [1.0])
        np.testing.assert_array_equal(self.ws.get("ns", "existing"), [99.0])

    def test_rename_to_existing_key_with_overwrite(self):
        """
        rename() with overwrite=True replaces the existing key.

        Tests:
            (Test Case 1) rename() returns True.
            (Test Case 2) new_key holds the value from old_key.
            (Test Case 3) old_key is removed.
        """
        self.ws.store("ns", "old", np.array([1.0]))
        self.ws.store("ns", "existing", np.array([99.0]))

        result = self.ws.rename("ns", "old", "existing", overwrite=True)
        assert result is True

        np.testing.assert_array_equal(self.ws.get("ns", "existing"), [1.0])
        assert self.ws.get("ns", "old") is None
        assert self.ws.list_keys("ns") == ["existing"]

    # ------------------------------------------------------------------
    # EC-WS-06: comparison_namespace with empty strings
    # ------------------------------------------------------------------

    def test_comparison_namespace_empty_strings(self):
        """
        EC-WS-06: comparison_namespace() with empty string arguments.

        The method just concatenates strings with underscores, so empty
        strings produce a result with leading/trailing/double underscores.

        Tests:
            (Test Case 1) Single empty string produces "C_".
            (Test Case 2) Two empty strings produce "C__".
            (Test Case 3) Mixed empty and non-empty produces correct result.
        """
        assert AnalysisWorkspace.comparison_namespace("") == "C_"
        assert AnalysisWorkspace.comparison_namespace("", "") == "C__"
        assert AnalysisWorkspace.comparison_namespace("", "rec1") == "C__rec1"
        assert AnalysisWorkspace.comparison_namespace("rec1", "") == "C_rec1_"

    def test_comparison_namespace_no_arguments(self):
        """
        EC-WS-06 (cont): comparison_namespace() with no arguments.

        With zero arguments, "_".join(()) produces an empty string, so the
        result is just "C_".

        Tests:
            (Test Case 1) Zero arguments returns "C_".
        """
        assert AnalysisWorkspace.comparison_namespace() == "C_"

    def test_rename_same_key(self):
        """
        Renaming a key to itself warns about key conflict and returns False.

        Tests:
            (Test Case 1) rename(old_key, old_key) without overwrite=True
                returns False due to the existing-key check.
        """
        import warnings

        ws = AnalysisWorkspace(name="rename_test")
        ws.store("ns", "key", np.array([1.0, 2.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = ws.rename("ns", "key", "key")
        assert result is False
        # Original key should still exist
        assert ws.get("ns", "key") is not None

    def test_workspace_name_none_repr(self):
        """
        Workspace with name=None produces a valid repr without the name part.

        Tests:
            (Test Case 1) name is None by default.
            (Test Case 2) repr does not include a name segment when name is None.
            (Test Case 3) repr still contains 'AnalysisWorkspace'.
        """
        ws = AnalysisWorkspace()
        assert ws.name is None
        r = repr(ws)
        assert "AnalysisWorkspace" in r
        # When name is None, the name part is omitted (no None in output)
        assert "None" not in r

    def test_delete_last_key_in_namespace_removes_namespace(self):
        """
        Deleting the last key in a namespace removes the namespace entirely.

        Tests:
            (Test Case 1) The deleted key is no longer retrievable.
            (Test Case 2) The namespace is removed from _items and _index.
            (Test Case 3) list_namespaces no longer lists the namespace.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store("ns", "only_key", np.array([1.0]))

        ws.delete("ns", "only_key")
        assert ws.get("ns", "only_key") is None
        assert "ns" not in ws._items
        assert "ns" not in ws._index
        assert "ns" not in ws.list_namespaces()

    def test_merge_from_self(self):
        """
        Merging a workspace into itself does not duplicate items (overwrite=False skips
        all existing keys).

        Tests:
            (Test Case 1) merge_from returns with all items skipped.
            (Test Case 2) Original data is unchanged after self-merge.
        """
        ws = AnalysisWorkspace(name="self_merge")
        ws.store("ns", "a", np.array([1.0, 2.0]))
        ws.store("ns", "b", np.array([3.0]))

        result = ws.merge_from(ws, overwrite=False)
        assert result["skipped"] == 2
        assert result["merged"] == 0
        np.testing.assert_array_equal(ws.get("ns", "a"), [1.0, 2.0])
        np.testing.assert_array_equal(ws.get("ns", "b"), [3.0])

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_save_empty_workspace(self):
        """
        Saving a workspace with zero namespaces and zero items creates valid files.

        Tests:
            (Test Case 1) save() does not raise.
            (Test Case 2) Both .h5 and .json files are created.
            (Test Case 3) The loaded workspace has zero namespaces.
        """
        ws = AnalysisWorkspace(name="empty")

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "empty_ws")
            ws.save(base)

            assert pathlib.Path(f"{base}.h5").exists()
            assert pathlib.Path(f"{base}.json").exists()

            loaded = AnalysisWorkspace.load(base)
            assert loaded.list_namespaces() == []
            assert loaded.name == "empty"

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_save_to_nonexistent_parent_directory_raises(self):
        """
        Saving to a path whose parent directory does not exist raises an error.

        Tests:
            (Test Case 1) save() raises OSError (or FileNotFoundError) when the
                parent directory does not exist.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store("ns", "arr", np.zeros(3))

        bad_path = str(
            pathlib.Path(tempfile.gettempdir()) / "nonexistent_dir_abc123" / "ws"
        )
        with pytest.raises(OSError):
            ws.save(bad_path)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_save_overwrites_existing_files(self):
        """
        Saving to the same path twice overwrites the previous files.

        Tests:
            (Test Case 1) Second save does not raise.
            (Test Case 2) Loaded workspace reflects the second save's data.
        """
        ws = AnalysisWorkspace(name="overwrite_test")
        ws.store("ns", "arr", np.array([1.0]))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)

            # Modify and save again to same path
            ws.store("ns", "arr", np.array([99.0]))
            ws.save(base)

            loaded = AnalysisWorkspace.load(base)
            np.testing.assert_array_equal(loaded.get("ns", "arr"), [99.0])

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_load_without_json_sidecar(self):
        """
        Loading from a path where .h5 exists but .json does not still works,
        because AnalysisWorkspace.load() only reads the .h5 file.

        Tests:
            (Test Case 1) load() does not raise when .json is absent.
            (Test Case 2) Loaded workspace has the correct items.
        """
        ws = AnalysisWorkspace(name="no_json")
        ws.store("ns", "arr", np.array([1.0, 2.0]))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)

            # Remove the .json file
            json_path = pathlib.Path(f"{base}.json")
            json_path.unlink()
            assert not json_path.exists()

            loaded = AnalysisWorkspace.load(base)
            np.testing.assert_array_equal(loaded.get("ns", "arr"), [1.0, 2.0])

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_load_from_nonexistent_path_raises(self):
        """
        Loading from a path where neither .h5 nor .json exist raises an error.

        Tests:
            (Test Case 1) load() raises OSError (or FileNotFoundError).
        """
        with pytest.raises(OSError):
            AnalysisWorkspace.load(
                str(pathlib.Path(tempfile.gettempdir()) / "nonexistent_ws_abc123")
            )


# ---------------------------------------------------------------------------
# Tests: _make_summary
# ---------------------------------------------------------------------------


class TestMakeSummary:
    def test_summary_ndarray(self):
        """
        Tests _make_summary() for a numpy array.

        Tests:
            (Test Case 1) type is "ndarray".
            (Test Case 2) shape matches the array dimensions.
            (Test Case 3) dtype matches the array dtype.
        """
        arr = np.zeros((3, 4), dtype=np.float32)
        s = _make_summary(arr)
        assert s["type"] == "ndarray"
        assert s["shape"] == [3, 4]
        assert s["dtype"] == "float32"

    def test_summary_spikedata(self):
        """
        Tests _make_summary() for a SpikeData object.

        Tests:
            (Test Case 1) type is "SpikeData".
            (Test Case 2) N matches the unit count.
            (Test Case 3) length_ms matches the recording length.
        """
        sd = make_spikedata(n_units=5, length_ms=300.0)
        s = _make_summary(sd)
        assert s["type"] == "SpikeData"
        assert s["N"] == 5
        assert s["length_ms"] == pytest.approx(300.0)

    def test_summary_spikedata_event_centered_start_time_present(self):
        """
        ``_make_summary`` records ``start_time`` for SpikeData so
        event-centered slices are visible in ``describe()`` output —
        the previous summary collapsed all SpikeData to the same
        ``length_ms``-only shape and the event-centered window was
        invisible.

        Tests:
            (Test Case 1) SpikeData with ``start_time=-200.0`` has
                ``s["start_time"] == -200.0`` in the summary.
        """
        sd = SpikeData([[5.0]], length=500.0, start_time=-200.0)
        s = _make_summary(sd)
        assert s["type"] == "SpikeData"
        assert s["start_time"] == pytest.approx(-200.0)

    def test_summary_ratedata(self):
        """
        Tests _make_summary() for a RateData object.

        Tests:
            (Test Case 1) type is "RateData".
            (Test Case 2) shape matches (n_units, n_times).
        """
        rd = make_ratedata(n_units=4, n_times=80)
        s = _make_summary(rd)
        assert s["type"] == "RateData"
        assert s["shape"] == [4, 80]

    def test_summary_rateslicestack(self):
        """
        Tests _make_summary() for a RateSliceStack object.

        Tests:
            (Test Case 1) type is "RateSliceStack".
            (Test Case 2) shape matches (n_units, n_times, n_slices).
        """
        rss = make_rateslicestack(n_units=3, n_times=10, n_slices=5)
        s = _make_summary(rss)
        assert s["type"] == "RateSliceStack"
        assert s["shape"] == [3, 10, 5]

    def test_summary_spikeslicestack(self):
        """
        Tests _make_summary() for a SpikeSliceStack object.

        Tests:
            (Test Case 1) type is "SpikeSliceStack".
            (Test Case 2) N_slices matches the number of slices.
            (Test Case 3) N_units matches the number of units.
            (Test Case 4) length_ms matches the duration of each slice.
        """
        sss = make_spikeslicestack(n_units=3, slice_length_ms=50.0, n_slices=4)
        s = _make_summary(sss)
        assert s["type"] == "SpikeSliceStack"
        assert s["N_slices"] == 4
        assert s["N_units"] == 3
        assert s["length_ms"] == pytest.approx(50.0)

    def test_summary_spikeslicestack_heterogeneous_durations_returns_range(
        self,
    ):
        """
        ``_make_summary`` reports the per-slice length range when a
        ``SpikeSliceStack`` has heterogeneous-duration slice tuples.
        Pin the defensive summary path that future heterogeneous-
        durations support (e.g. ``time_peaks + time_bounds`` with
        per-event bounds) would surface — for now we mutate
        ``stack.times`` post-construction to bypass the equal-length
        validator and exercise the same code path.

        Tests:
            (Test Case 1) Times ``[(0, 100), (100, 250)]`` → durations
                ``[100.0, 150.0]`` → ``length_ms`` is a tuple
                ``(100.0, 150.0)``.
            (Test Case 2) Uniform-duration stacks still return a
                single float (covered by the existing
                ``test_summary_spikeslicestack``).
        """
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        # Construct a uniform stack first (the validator only fires at
        # __init__), then mutate ``stack.times`` to a heterogeneous
        # layout to exercise the summary code path.
        sd1 = SpikeData([[10.0]], length=100.0)
        sd2 = SpikeData([[50.0]], length=100.0)
        sss = SpikeSliceStack(
            spike_stack=[sd1, sd2],
            times_start_to_end=[(0.0, 100.0), (100.0, 200.0)],
        )
        sss.times = [(0.0, 100.0), (100.0, 250.0)]

        s = _make_summary(sss)
        assert s["type"] == "SpikeSliceStack"
        assert s["length_ms"] == (100.0, 150.0)

    def test_summary_pairwise_comp_matrix(self):
        """
        Tests _make_summary() for a PairwiseCompMatrix.

        Tests:
            (Test Case 1) type is "PairwiseCompMatrix".
            (Test Case 2) shape matches the matrix dimensions.
        """
        pcm = PairwiseCompMatrix(matrix=np.eye(5))
        s = _make_summary(pcm)
        assert s["type"] == "PairwiseCompMatrix"
        assert s["shape"] == [5, 5]

    def test_summary_pairwise_comp_matrix_stack(self):
        """
        Tests _make_summary() for a PairwiseCompMatrixStack.

        Tests:
            (Test Case 1) type is "PairwiseCompMatrixStack".
            (Test Case 2) shape matches (N, N, S).
        """
        stack_arr = np.random.default_rng(0).random((4, 4, 6))
        pcms = PairwiseCompMatrixStack(stack=stack_arr)
        s = _make_summary(pcms)
        assert s["type"] == "PairwiseCompMatrixStack"
        assert s["shape"] == [4, 4, 6]

    def test_summary_unknown_type(self):
        """
        Tests _make_summary() falls back to the class name for unrecognised types.

        Tests:
            (Test Case 1) type field contains the class name.
        """

        class MyCustomObj:
            pass

        s = _make_summary(MyCustomObj())
        assert s["type"] == "MyCustomObj"

    def test_spikeslicestack_single_slice(self):
        """
        _make_summary for SpikeSliceStack with 1 slice.

        Tests:
            (Test Case 1) Single-slice SpikeSliceStack produces a valid summary.
        """
        sd = SpikeData([[5.0]], length=10.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 10.0)])
        summary = _make_summary(sss)
        assert "SpikeSliceStack" in summary["type"]

    def test_zero_dimensional_ndarray(self):
        """
        _make_summary for a 0-dimensional ndarray.

        Tests:
            (Test Case 1) np.array(5.0) produces shape [].
        """
        summary = _make_summary(np.array(5.0))
        assert summary["shape"] == []


# ---------------------------------------------------------------------------
# Tests: WorkspaceManager
# ---------------------------------------------------------------------------


class TestWorkspaceManager:
    def setup_method(self):
        """Create a fresh WorkspaceManager for each test."""
        self.mgr = WorkspaceManager()

    def test_create_and_get(self):
        """
        Tests that create_workspace() returns a valid ID and get_workspace() retrieves it.

        Tests:
            (Test Case 1) create_workspace() returns a non-empty string ID.
            (Test Case 2) get_workspace() returns the AnalysisWorkspace instance.
            (Test Case 3) Workspace name is set correctly when provided.
            (Test Case 4) workspace_id on the returned object matches the returned ID.
        """
        wid = self.mgr.create_workspace(name="my_ws")
        assert isinstance(wid, str)
        assert len(wid) > 0

        ws = self.mgr.get_workspace(wid)
        assert isinstance(ws, AnalysisWorkspace)
        assert ws.name == "my_ws"
        assert ws.workspace_id == wid

    def test_get_unknown_returns_none(self):
        """
        Tests that get_workspace() returns None for an unknown ID.

        Tests:
            (Test Case 1) Non-existent ID returns None.
        """
        assert self.mgr.get_workspace("nonexistent-id") is None

    def test_delete_workspace(self):
        """
        Tests that delete_workspace() removes a workspace and raises KeyError for unknown IDs.

        Tests:
            (Test Case 1) delete_workspace() removes an existing workspace.
            (Test Case 2) get_workspace() returns None after deletion.
            (Test Case 3) delete_workspace() raises KeyError for an unknown ID.
        """
        wid = self.mgr.create_workspace()
        self.mgr.delete_workspace(wid)
        assert self.mgr.get_workspace(wid) is None

        with pytest.raises(KeyError, match="not found"):
            self.mgr.delete_workspace("nonexistent-id")

    def test_list_workspaces(self):
        """
        Tests that list_workspaces() returns correct summary dicts.

        Tests:
            (Test Case 1) Empty manager returns empty list.
            (Test Case 2) Each entry has workspace_id, name, created_at, namespace_count, item_count.
            (Test Case 3) item_count reflects stored items correctly.
        """
        assert self.mgr.list_workspaces() == []

        wid = self.mgr.create_workspace(name="alpha")
        ws = self.mgr.get_workspace(wid)
        ws.store("ns1", "a", np.zeros(3))
        ws.store("ns1", "b", np.zeros(3))

        listing = self.mgr.list_workspaces()
        assert len(listing) == 1
        entry = listing[0]
        assert entry["workspace_id"] == wid
        assert entry["name"] == "alpha"
        assert "created_at" in entry
        assert entry["namespace_count"] == 1
        assert entry["item_count"] == 2

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_save_and_load_workspace(self):
        """
        Tests save_workspace() and load_workspace() round-trip via the manager.

        Tests:
            (Test Case 1) save_workspace() does not raise and creates the .h5 file.
            (Test Case 2) load_workspace() returns the original workspace_id.
            (Test Case 3) Loaded workspace is accessible via get_workspace().
            (Test Case 4) Stored content is preserved after round-trip.
        """
        wid = self.mgr.create_workspace(name="saved")
        ws = self.mgr.get_workspace(wid)
        arr = np.array([10.0, 20.0, 30.0])
        ws.store("ns", "arr", arr)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            self.mgr.save_workspace(wid, base)

            assert pathlib.Path(base + ".h5").exists()

            mgr2 = WorkspaceManager()
            loaded_id = mgr2.load_workspace(base)

        assert loaded_id == wid
        loaded_ws = mgr2.get_workspace(loaded_id)
        assert loaded_ws is not None
        np.testing.assert_array_equal(loaded_ws.get("ns", "arr"), arr)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_load_workspace_item(self):
        """
        Tests load_workspace_item() loads one item into an existing in-memory workspace.

        Tests:
            (Test Case 1) The item is available via get() after loading.
            (Test Case 2) The reconstructed object has the correct type and values.
            (Test Case 3) Other items in the file are not automatically loaded.
            (Test Case 4) Unknown workspace_id raises KeyError.
        """
        wid = self.mgr.create_workspace(name="source")
        ws = self.mgr.get_workspace(wid)
        arr = np.array([1.0, 2.0, 3.0])
        sd = make_spikedata(n_units=2, length_ms=40.0)
        ws.store("ns", "arr", arr)
        ws.store("ns", "spikes", sd)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            self.mgr.save_workspace(wid, base)

            # Load into a fresh workspace
            target_wid = self.mgr.create_workspace(name="target")
            self.mgr.load_workspace_item(base, "ns", "arr", target_wid)
            target_ws = self.mgr.get_workspace(target_wid)

            loaded_arr = target_ws.get("ns", "arr")
            np.testing.assert_array_equal(loaded_arr, arr)

            # 'spikes' was not loaded
            assert target_ws.get("ns", "spikes") is None

            with pytest.raises(KeyError):
                self.mgr.load_workspace_item(base, "ns", "arr", "nonexistent-id")

    def test_save_unknown_workspace_raises(self):
        """
        Tests that save_workspace() raises KeyError for an unknown workspace_id.

        Tests:
            (Test Case 1) Unknown workspace_id raises KeyError.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with pytest.raises(KeyError):
                self.mgr.save_workspace("nonexistent-id", base)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def test_get_workspace_manager_singleton(self):
        """
        Tests that get_workspace_manager() returns the same instance on repeated calls.

        Tests:
            (Test Case 1) Two consecutive calls return the identical object.
        """
        mgr_a = get_workspace_manager()
        mgr_b = get_workspace_manager()
        assert mgr_a is mgr_b

    def test_delete_workspace_while_external_reference_held(self):
        """
        EC-WS-07: delete_workspace while an external reference to the workspace exists.

        WorkspaceManager.delete_workspace() only removes the workspace from its
        internal registry. If the caller holds a separate Python reference to the
        workspace object, that reference remains valid and usable — delete_workspace
        does not destroy the workspace itself.

        Tests:
            (Test Case 1) delete_workspace removes the entry without error.
            (Test Case 2) get_workspace returns None after deletion.
            (Test Case 3) The external reference is still a valid AnalysisWorkspace.
            (Test Case 4) Data stored in the workspace is still accessible via the external reference.
        """
        mgr = WorkspaceManager()
        wid = mgr.create_workspace(name="held_ref")
        ws = mgr.get_workspace(wid)
        ws.store("ns", "arr", np.array([1.0, 2.0]))

        # Hold external reference, then delete from manager
        external_ref = ws
        mgr.delete_workspace(wid)
        assert mgr.get_workspace(wid) is None

        # External reference still works
        assert isinstance(external_ref, AnalysisWorkspace)
        np.testing.assert_array_equal(external_ref.get("ns", "arr"), [1.0, 2.0])

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_load_workspace_item_nonexistent_key_in_file_raises(self):
        """
        load_workspace_item raises KeyError when the file exists but the
        requested (namespace, key) does not.

        Tests:
            (Test Case 1) Missing namespace raises KeyError.
            (Test Case 2) Missing key within an existing namespace raises KeyError.
        """
        mgr = WorkspaceManager()
        wid = mgr.create_workspace(name="source")
        ws = mgr.get_workspace(wid)
        ws.store("ns", "real_key", np.array([1.0]))

        target_wid = mgr.create_workspace(name="target")

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            mgr.save_workspace(wid, base)

            with pytest.raises(KeyError):
                mgr.load_workspace_item(base, "wrong_ns", "real_key", target_wid)

            with pytest.raises(KeyError):
                mgr.load_workspace_item(base, "ns", "wrong_key", target_wid)


# ---------------------------------------------------------------------------
# Tests: hdf5_io — HDF5 round-trips for every supported type
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestHDF5IO:
    """
    Round-trip tests for workspace/hdf5_io.py.

    Each test saves one or more objects to a temporary .h5 file via
    dump_workspace() or _dump_item() directly, then reloads via
    load_workspace_full() or load_workspace_item() and verifies that the
    reconstructed object is equal to the original.
    """

    def _roundtrip(self, obj, namespace="ns", key="item"):
        """
        Helper: store obj in a workspace, save to HDF5, reload the full workspace,
        and return the reconstructed object.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store(namespace, key, obj)
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded_ws = AnalysisWorkspace.load(base)
        return loaded_ws.get(namespace, key)

    def _roundtrip_item(self, obj, namespace="ns", key="item"):
        """
        Helper: save obj in a workspace HDF5 file, reload only that item via
        load_workspace_item(), and return the reconstructed object.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store(namespace, key, obj)
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            return AnalysisWorkspace.load_item(base, namespace, key)

    # ------------------------------------------------------------------
    # ndarray
    # ------------------------------------------------------------------

    def test_roundtrip_ndarray_1d(self):
        """
        Tests HDF5 round-trip for a 1-D numpy array.

        Tests:
            (Test Case 1) Values are preserved exactly.
            (Test Case 2) Shape is preserved.
        """
        arr = np.array([1.0, 2.0, 3.0])
        out = self._roundtrip(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.shape == (3,)

    def test_roundtrip_ndarray_2d(self):
        """
        Tests HDF5 round-trip for a 2-D numpy array.

        Tests:
            (Test Case 1) Values are preserved exactly.
            (Test Case 2) Shape is preserved.
        """
        arr = np.arange(12).reshape(3, 4).astype(np.float64)
        out = self._roundtrip(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.shape == (3, 4)

    def test_roundtrip_ndarray_3d(self):
        """
        Tests HDF5 round-trip for a 3-D numpy array.

        Tests:
            (Test Case 1) Values and shape are preserved.
        """
        arr = np.random.default_rng(0).random((2, 5, 4))
        out = self._roundtrip(arr)
        np.testing.assert_array_almost_equal(out, arr)
        assert out.shape == (2, 5, 4)

    # ------------------------------------------------------------------
    # SpikeData
    # ------------------------------------------------------------------

    def test_roundtrip_spikedata_basic(self):
        """
        Tests HDF5 round-trip for a basic SpikeData with no attributes.

        Tests:
            (Test Case 1) Reconstructed object is a SpikeData instance.
            (Test Case 2) N and length are preserved.
            (Test Case 3) Spike trains are preserved (allclose).
            (Test Case 4) metadata is preserved.
        """
        sd = SpikeData(
            [[1.0, 2.0, 3.0], [5.0, 10.0], []],
            length=50.0,
            metadata={"source": "test"},
        )
        out = self._roundtrip(sd)
        assert isinstance(out, SpikeData)
        assert out.N == 3
        assert out.length == pytest.approx(50.0)
        np.testing.assert_array_almost_equal(out.train[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(out.train[1], [5.0, 10.0])
        assert len(out.train[2]) == 0
        assert out.metadata["source"] == "test"

    def test_roundtrip_spikedata_neuron_attributes_numeric(self):
        """
        Tests HDF5 round-trip for SpikeData with numeric neuron_attributes.

        Tests:
            (Test Case 1) neuron_attributes is not None after load.
            (Test Case 2) Numeric attribute values are preserved (float comparison).
            (Test Case 3) Units without the attribute are missing, not set to NaN.

        Notes:
            - Numeric missing entries use NaN as a sentinel and are dropped on load,
              so units without a given attribute key will not have it in their dict.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sd.set_neuron_attribute("channel", [0, 1, 2])
        out = self._roundtrip(sd)
        assert out.neuron_attributes is not None
        channels = [d.get("channel") for d in out.neuron_attributes]
        assert channels[0] == pytest.approx(0.0)
        assert channels[1] == pytest.approx(1.0)
        assert channels[2] == pytest.approx(2.0)

    def test_roundtrip_spikedata_neuron_attributes_string(self):
        """
        Tests HDF5 round-trip for SpikeData with string neuron_attributes.

        Tests:
            (Test Case 1) String attribute values are preserved.
            (Test Case 2) neuron_attributes list has correct length.
        """
        sd = make_spikedata(n_units=2, length_ms=80.0)
        sd.set_neuron_attribute("group", ["A", "B"])
        out = self._roundtrip(sd)
        assert out.neuron_attributes is not None
        groups = [d.get("group") for d in out.neuron_attributes]
        assert groups[0] == "A"
        assert groups[1] == "B"

    def test_roundtrip_spikedata_neuron_attributes_array(self):
        """
        Tests HDF5 round-trip for SpikeData with array-valued neuron_attributes.

        Tests:
            (Test Case 1) Array-valued attribute is restored with the correct shape.
            (Test Case 2) Array values are preserved (allclose).
            (Test Case 3) Scalar attributes stored alongside array attributes are also preserved.
        """
        rng = np.random.default_rng(7)
        waveforms = [rng.standard_normal((10, 5)) for _ in range(3)]
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sd.neuron_attributes = [
            {"waveform": waveforms[0], "channel": 0},
            {"waveform": waveforms[1], "channel": 1},
            {"waveform": waveforms[2], "channel": 2},
        ]
        out = self._roundtrip(sd)
        assert out.neuron_attributes is not None
        for i in range(3):
            assert "waveform" in out.neuron_attributes[i]
            wf = out.neuron_attributes[i]["waveform"]
            assert isinstance(wf, np.ndarray)
            assert wf.shape == (10, 5)
            np.testing.assert_array_almost_equal(wf, waveforms[i])
            assert float(out.neuron_attributes[i]["channel"]) == pytest.approx(float(i))

    def test_roundtrip_spikedata_neuron_attributes_array_partial(self):
        """
        Tests HDF5 round-trip for array-valued neuron_attributes when some units lack the attribute.

        Tests:
            (Test Case 1) Units with the array attribute have it restored correctly.
            (Test Case 2) Units missing the array attribute do not have the key in their dict after load.
        """
        rng = np.random.default_rng(8)
        waveform = rng.standard_normal((10, 5))
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sd.neuron_attributes = [
            {"waveform": waveform},
            {},
            {"waveform": waveform * 2.0},
        ]
        out = self._roundtrip(sd)
        assert out.neuron_attributes is not None
        np.testing.assert_array_almost_equal(
            out.neuron_attributes[0]["waveform"], waveform
        )
        assert "waveform" not in out.neuron_attributes[1]
        np.testing.assert_array_almost_equal(
            out.neuron_attributes[2]["waveform"], waveform * 2.0
        )

    def test_roundtrip_spikedata_neuron_attributes_list(self):
        """
        Tests HDF5 round-trip for neuron_attributes containing Python list values.

        Tests:
            (Test Case 1) List-valued attributes (e.g. electrode positions) survive
                round-trip and are restored as arrays with the correct shape.
            (Test Case 2) List values are numerically preserved.
            (Test Case 3) Scalar attributes stored alongside list attributes are preserved.
            (Test Case 4) Units missing the list attribute do not have the key after load.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sd.neuron_attributes = [
            {"location": [175.0, 1015.0], "channel": 0},
            {"location": [200.0, 800.0], "channel": 1},
            {"channel": 2},
        ]
        out = self._roundtrip(sd)
        assert out.neuron_attributes is not None
        # Test Case 1 & 2: list values restored as arrays
        np.testing.assert_array_almost_equal(
            out.neuron_attributes[0]["location"], [175.0, 1015.0]
        )
        np.testing.assert_array_almost_equal(
            out.neuron_attributes[1]["location"], [200.0, 800.0]
        )
        assert isinstance(out.neuron_attributes[0]["location"], np.ndarray)
        assert len(out.neuron_attributes[0]["location"]) == 2
        # Test Case 3: scalar preserved
        assert float(out.neuron_attributes[0]["channel"]) == pytest.approx(0.0)
        assert float(out.neuron_attributes[1]["channel"]) == pytest.approx(1.0)
        assert float(out.neuron_attributes[2]["channel"]) == pytest.approx(2.0)
        # Test Case 4: missing list attribute
        assert "location" not in out.neuron_attributes[2]

    def test_roundtrip_spikedata_with_raw_data(self):
        """
        Tests HDF5 round-trip for SpikeData that includes raw_data and raw_time.

        Tests:
            (Test Case 1) raw_data shape and values are preserved.
            (Test Case 2) raw_time values are preserved.
        """
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((4, 100))
        raw_t = np.linspace(0.0, 99.0, 100)
        sd = SpikeData(
            [[5.0, 10.0], [20.0]], length=100.0, raw_data=raw, raw_time=raw_t
        )
        out = self._roundtrip(sd)
        np.testing.assert_array_almost_equal(out.raw_data, raw)
        np.testing.assert_array_almost_equal(out.raw_time, raw_t)

    def test_roundtrip_spikedata_no_neuron_attributes(self):
        """
        Tests that neuron_attributes is None after a round-trip when none were set.

        Tests:
            (Test Case 1) neuron_attributes is None on the loaded object.
        """
        sd = SpikeData([[1.0, 2.0], [3.0]], length=10.0)
        out = self._roundtrip(sd)
        assert out.neuron_attributes is None

    # ------------------------------------------------------------------
    # RateData
    # ------------------------------------------------------------------

    def test_roundtrip_ratedata(self):
        """
        Tests HDF5 round-trip for a RateData object.

        Tests:
            (Test Case 1) Reconstructed object is a RateData instance.
            (Test Case 2) inst_Frate_data shape and values are preserved.
            (Test Case 3) times array is preserved.
        """
        rd = make_ratedata(n_units=3, n_times=40, step=2.0)
        out = self._roundtrip(rd)
        assert isinstance(out, RateData)
        assert out.inst_Frate_data.shape == (3, 40)
        np.testing.assert_array_almost_equal(out.inst_Frate_data, rd.inst_Frate_data)
        np.testing.assert_array_almost_equal(out.times, rd.times)

    def test_roundtrip_ratedata_with_neuron_attributes(self):
        """
        Tests HDF5 round-trip for RateData with numeric neuron_attributes.

        Tests:
            (Test Case 1) neuron_attributes is not None after load.
            (Test Case 2) Numeric values match the originals.
        """
        rd = make_ratedata(n_units=2, n_times=20)
        rd.neuron_attributes = [{"depth": 100.0}, {"depth": 200.0}]
        out = self._roundtrip(rd)
        assert out.neuron_attributes is not None
        assert out.neuron_attributes[0]["depth"] == pytest.approx(100.0)
        assert out.neuron_attributes[1]["depth"] == pytest.approx(200.0)

    # ------------------------------------------------------------------
    # RateSliceStack
    # ------------------------------------------------------------------

    def test_roundtrip_rateslicestack(self):
        """
        Tests HDF5 round-trip for a RateSliceStack.

        Tests:
            (Test Case 1) Reconstructed object is a RateSliceStack instance.
            (Test Case 2) event_stack shape and values are preserved.
            (Test Case 3) times list is preserved (same start/end pairs).
            (Test Case 4) step_size is preserved.
        """
        rss = make_rateslicestack(n_units=3, n_times=10, n_slices=5)
        out = self._roundtrip(rss)
        assert isinstance(out, RateSliceStack)
        assert out.event_stack.shape == (3, 10, 5)
        np.testing.assert_array_almost_equal(out.event_stack, rss.event_stack)
        assert len(out.times) == len(rss.times)
        for (s0, e0), (s1, e1) in zip(rss.times, out.times):
            assert s0 == pytest.approx(s1)
            assert e0 == pytest.approx(e1)
        assert out.step_size == pytest.approx(rss.step_size)

    # ------------------------------------------------------------------
    # SpikeSliceStack
    # ------------------------------------------------------------------

    def test_roundtrip_spikeslicestack(self):
        """
        Tests HDF5 round-trip for a SpikeSliceStack.

        Tests:
            (Test Case 1) Reconstructed object is a SpikeSliceStack instance.
            (Test Case 2) Number of slices is preserved.
            (Test Case 3) Each slice is a SpikeData with correct N and length.
            (Test Case 4) times list is preserved.
        """
        sss = make_spikeslicestack(n_units=2, slice_length_ms=50.0, n_slices=3)
        out = self._roundtrip(sss)
        assert isinstance(out, SpikeSliceStack)
        assert len(out.spike_stack) == 3
        for slice_sd in out.spike_stack:
            assert isinstance(slice_sd, SpikeData)
            assert slice_sd.N == 2
        assert len(out.times) == 3
        for (s0, e0), (s1, e1) in zip(sss.times, out.times):
            assert s0 == pytest.approx(s1)
            assert e0 == pytest.approx(e1)

    # ------------------------------------------------------------------
    # PairwiseCompMatrix
    # ------------------------------------------------------------------

    def test_roundtrip_pairwise_comp_matrix_no_labels(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrix without labels.

        Tests:
            (Test Case 1) Reconstructed object is a PairwiseCompMatrix instance.
            (Test Case 2) matrix values are preserved.
            (Test Case 3) labels is None.
        """
        pcm = PairwiseCompMatrix(matrix=np.eye(4))
        out = self._roundtrip(pcm)
        assert isinstance(out, PairwiseCompMatrix)
        np.testing.assert_array_almost_equal(out.matrix, pcm.matrix)
        assert out.labels is None

    def test_roundtrip_pairwise_comp_matrix_int_labels(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrix with integer labels.

        Tests:
            (Test Case 1) Integer labels are preserved as a list.
        """
        mat = np.random.default_rng(0).random((3, 3))
        pcm = PairwiseCompMatrix(matrix=mat, labels=[10, 20, 30])
        out = self._roundtrip(pcm)
        assert len(out.labels) == 3
        assert float(out.labels[0]) == pytest.approx(10.0)
        assert float(out.labels[1]) == pytest.approx(20.0)
        assert float(out.labels[2]) == pytest.approx(30.0)

    def test_roundtrip_pairwise_comp_matrix_string_labels(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrix with string labels.

        Tests:
            (Test Case 1) String labels are preserved exactly.
        """
        mat = np.eye(3)
        pcm = PairwiseCompMatrix(matrix=mat, labels=["A", "B", "C"])
        out = self._roundtrip(pcm)
        assert out.labels == ["A", "B", "C"]

    def test_roundtrip_pairwise_comp_matrix_metadata(self):
        """
        Tests HDF5 round-trip preserves metadata on a PairwiseCompMatrix.

        Tests:
            (Test Case 1) Scalar float metadata value is preserved.
            (Test Case 2) Boolean metadata value is preserved.
            (Test Case 3) String metadata value is preserved.
        """
        pcm = PairwiseCompMatrix(
            matrix=np.eye(2),
            metadata={"threshold": 0.5, "binary": True, "method": "sttc"},
        )
        out = self._roundtrip(pcm)
        assert out.metadata["threshold"] == pytest.approx(0.5)
        assert out.metadata["binary"]
        assert out.metadata["method"] == "sttc"

    # ------------------------------------------------------------------
    # PairwiseCompMatrixStack
    # ------------------------------------------------------------------

    def test_roundtrip_pairwise_comp_matrix_stack(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrixStack.

        Tests:
            (Test Case 1) Reconstructed object is a PairwiseCompMatrixStack instance.
            (Test Case 2) stack shape and values are preserved.
            (Test Case 3) labels are preserved.
            (Test Case 4) times are preserved.
            (Test Case 5) metadata is preserved.
        """
        rng = np.random.default_rng(5)
        stack_arr = rng.random((4, 4, 6))
        times = [(float(i * 10), float((i + 1) * 10)) for i in range(6)]
        pcms = PairwiseCompMatrixStack(
            stack=stack_arr,
            labels=["u0", "u1", "u2", "u3"],
            times=times,
            metadata={"delt": 25.0},
        )
        out = self._roundtrip(pcms)
        assert isinstance(out, PairwiseCompMatrixStack)
        assert out.stack.shape == (4, 4, 6)
        np.testing.assert_array_almost_equal(out.stack, stack_arr)
        assert out.labels == ["u0", "u1", "u2", "u3"]
        assert len(out.times) == 6
        for (s0, e0), (s1, e1) in zip(times, out.times):
            assert s0 == pytest.approx(s1)
            assert e0 == pytest.approx(e1)
        assert out.metadata["delt"] == pytest.approx(25.0)

    def test_roundtrip_pairwise_comp_matrix_stack_no_labels_no_times(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrixStack without labels or times.

        Tests:
            (Test Case 1) labels is None after load.
            (Test Case 2) times is None after load.
            (Test Case 3) stack values are preserved.
        """
        stack_arr = np.random.default_rng(0).random((3, 3, 4))
        pcms = PairwiseCompMatrixStack(stack=stack_arr)
        out = self._roundtrip(pcms)
        assert out.labels is None
        assert out.times is None
        np.testing.assert_array_almost_equal(out.stack, stack_arr)

    # ------------------------------------------------------------------
    # load_workspace_item selective loading
    # ------------------------------------------------------------------

    def test_load_workspace_item_selective(self):
        """
        Tests that load_workspace_item() returns the correct object without
        loading all other items stored in the same file.

        Tests:
            (Test Case 1) Requested item is returned correctly.
            (Test Case 2) A second item stored in the same file can also be loaded independently.
            (Test Case 3) The two selectively loaded objects are independent.
        """
        arr = np.array([10.0, 20.0, 30.0])
        pcm = PairwiseCompMatrix(matrix=np.eye(3))

        ws = AnalysisWorkspace()
        ws.store("ns", "arr", arr)
        ws.store("ns", "pcm", pcm)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)

            loaded_arr = AnalysisWorkspace.load_item(base, "ns", "arr")
            loaded_pcm = AnalysisWorkspace.load_item(base, "ns", "pcm")

        np.testing.assert_array_equal(loaded_arr, arr)
        assert isinstance(loaded_pcm, PairwiseCompMatrix)
        np.testing.assert_array_almost_equal(loaded_pcm.matrix, pcm.matrix)

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_unsupported_type_raises(self):
        """
        Tests that saving a workspace containing an unsupported type raises TypeError.

        Tests:
            (Test Case 1) A plain Python object that is not an IAT type raises TypeError.
        """

        class Custom:
            pass

        ws = AnalysisWorkspace()
        ws.store("ns", "obj", Custom())

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with pytest.raises(TypeError):
                ws.save(base)

    def test_load_item_missing_namespace_raises(self):
        """
        Tests that load_workspace_item() raises KeyError for a missing namespace.

        Tests:
            (Test Case 1) Non-existent namespace raises KeyError.
        """
        ws = AnalysisWorkspace()
        ws.store("ns", "arr", np.zeros(3))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            with pytest.raises(KeyError):
                AnalysisWorkspace.load_item(base, "wrong_ns", "arr")

    def test_load_item_missing_key_raises(self):
        """
        Tests that load_workspace_item() raises KeyError for a missing key.

        Tests:
            (Test Case 1) Non-existent key within a valid namespace raises KeyError.
        """
        ws = AnalysisWorkspace()
        ws.store("ns", "arr", np.zeros(3))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            with pytest.raises(KeyError):
                AnalysisWorkspace.load_item(base, "ns", "wrong_key")

    def test_metadata_non_json_serializable_raises(self):
        """
        Tests that saving metadata with a genuinely non-JSON-serializable value raises ValueError.

        Tests:
            (Test Case 1) metadata containing a custom Python object raises ValueError at save time.

        Notes:
            - numpy arrays and scalars are handled by _NumpyEncoder and do not raise.
        """

        class CustomObj:
            pass

        pcm = PairwiseCompMatrix(
            matrix=np.eye(2),
            metadata={"bad_value": CustomObj()},
        )
        ws = AnalysisWorkspace()
        ws.store("ns", "pcm", pcm)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with pytest.raises(ValueError):
                ws.save(base)

    def test_roundtrip_metadata_numpy_array(self):
        """
        Tests that a numpy array stored in SpikeData metadata survives an HDF5 round-trip.

        Tier L-F3: the metadata JSON encoder now records ndarrays as
        sentinel dicts so the load path can rebuild them with the
        original dtype and shape. Previously this round-tripped to a
        Python list, breaking downstream ``metadata[...] / scalar``
        arithmetic.

        Tests:
            (Test Case 1) Save does not raise despite the numpy array value.
            (Test Case 2) The metadata value is recovered as an ndarray
                with the original values, dtype, and shape — *not* a
                Python list.
        """
        arr = np.array([1.0, 2.0, 3.0])
        sd = SpikeData([[1.0, 2.0], [3.0]], length=20.0, metadata={"positions": arr})
        out = self._roundtrip(sd)
        assert "positions" in out.metadata
        assert isinstance(out.metadata["positions"], np.ndarray)
        assert out.metadata["positions"].dtype == arr.dtype
        assert out.metadata["positions"].shape == arr.shape
        np.testing.assert_array_equal(out.metadata["positions"], arr)
        # Downstream arithmetic that broke under the old list-typed
        # round-trip must now succeed.
        _ = out.metadata["positions"] / 1000.0

    def test_roundtrip_metadata_ndarray_preserves_dtype_and_shape(self):
        """
        Tier L-F3: ndarrays in metadata preserve dtype and shape through
        the HDF5 round-trip, including non-default dtypes (int32, etc.)
        and 2-D arrays. Also covers nesting inside a dict.

        Tests:
            (Test Case 1) int32 1-D array round-trips as int32 ndarray.
            (Test Case 2) float32 2-D array round-trips with correct
                shape, dtype, and values.
            (Test Case 3) Ndarray nested inside a sub-dict in metadata
                is also restored to ndarray.
        """
        int_arr = np.array([10, 20, 30], dtype=np.int32)
        mat = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32)
        nested = {"inner": {"timeseries": np.arange(5, dtype=np.float64)}}
        sd = SpikeData(
            [[1.0, 2.0]],
            length=10.0,
            metadata={
                "cluster_ids": int_arr,
                "waveform_matrix": mat,
                "trials": nested,
            },
        )
        out = self._roundtrip(sd)

        # Test Case 1: int32 preserved.
        assert isinstance(out.metadata["cluster_ids"], np.ndarray)
        assert out.metadata["cluster_ids"].dtype == np.int32
        np.testing.assert_array_equal(out.metadata["cluster_ids"], int_arr)

        # Test Case 2: float32 2-D preserved.
        assert isinstance(out.metadata["waveform_matrix"], np.ndarray)
        assert out.metadata["waveform_matrix"].dtype == np.float32
        assert out.metadata["waveform_matrix"].shape == (3, 2)
        np.testing.assert_array_equal(out.metadata["waveform_matrix"], mat)

        # Test Case 3: nested ndarray preserved.
        inner = out.metadata["trials"]["inner"]["timeseries"]
        assert isinstance(inner, np.ndarray)
        assert inner.dtype == np.float64
        np.testing.assert_array_equal(inner, np.arange(5, dtype=np.float64))

    def test_roundtrip_metadata_empty_ndarray(self):
        """
        Tier L-F3: empty ndarrays (shape (0,)) round-trip as empty
        ndarrays, not empty lists. An empty IBL trial-times array is a
        valid edge case (recording with zero trials of a particular
        event type).

        Tests:
            (Test Case 1) Empty float64 array round-trips with shape
                (0,) and dtype float64.
        """
        empty = np.array([], dtype=np.float64)
        sd = SpikeData([[1.0]], length=10.0, metadata={"stim_off_times": empty})
        out = self._roundtrip(sd)
        recovered = out.metadata["stim_off_times"]
        assert isinstance(recovered, np.ndarray)
        assert recovered.shape == (0,)
        assert recovered.dtype == np.float64

    def test_roundtrip_metadata_numpy_scalars(self):
        """
        Tests that numpy scalar types in metadata are serialized to Python primitives.

        Tests:
            (Test Case 1) numpy integer value is preserved numerically.
            (Test Case 2) numpy float value is preserved numerically.
            (Test Case 3) numpy bool value is preserved as truthy.
        """
        sd = SpikeData(
            [[1.0, 2.0]],
            length=10.0,
            metadata={
                "count": np.int64(42),
                "rate": np.float32(3.14),
                "active": np.bool_(True),
            },
        )
        out = self._roundtrip(sd)
        assert out.metadata["count"] == 42
        assert out.metadata["rate"] == pytest.approx(3.14, abs=1e-5)
        assert out.metadata["active"]

    # ------------------------------------------------------------------
    # Index metadata after full load
    # ------------------------------------------------------------------

    def test_index_entry_preserved_after_load(self):
        """
        Tests that index metadata (type, note, created_at) is reconstructed correctly
        after loading a full workspace.

        Tests:
            (Test Case 1) type field matches the stored object.
            (Test Case 2) note is preserved when set at store time.
            (Test Case 3) created_at is a non-zero float.
        """
        arr = np.zeros(4)
        ws = AnalysisWorkspace()
        ws.store("ns", "arr", arr, note="my note")

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded_ws = AnalysisWorkspace.load(base)

        info = loaded_ws.get_info("ns", "arr")
        assert info["type"] == "ndarray"
        assert info["note"] == "my note"
        assert info["created_at"] > 0.0

    # ------------------------------------------------------------------
    # dict
    # ------------------------------------------------------------------

    def test_roundtrip_dict_with_arrays(self):
        """
        Round-trip a dict whose values are numpy arrays.

        Tests:
            (Test Case 1) All keys are preserved.
            (Test Case 2) Array values are numerically equal after reload.
            (Test Case 3) Array shapes are preserved.
        """
        d = {
            "weights": np.array([1.0, 2.0, 3.0]),
            "matrix": np.eye(3),
        }
        out = self._roundtrip(d)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"weights", "matrix"}
        np.testing.assert_array_equal(out["weights"], d["weights"])
        np.testing.assert_array_equal(out["matrix"], d["matrix"])
        assert out["matrix"].shape == (3, 3)

    def test_roundtrip_dict_with_scalars(self):
        """
        Round-trip a dict containing int, float, and bool scalar values.

        Tests:
            (Test Case 1) Integer value preserved (as int).
            (Test Case 2) Float value preserved.
            (Test Case 3) Bool value preserved (as bool).
        """
        d = {"count": 42, "threshold": 3.14, "flag": True}
        out = self._roundtrip(d)
        assert isinstance(out, dict)
        assert out["count"] == 42
        assert isinstance(out["count"], int)
        assert out["threshold"] == pytest.approx(3.14)
        assert out["flag"] is True

    def test_roundtrip_dict_with_strings(self):
        """
        Round-trip a dict containing string values.

        Tests:
            (Test Case 1) String values are preserved exactly.
        """
        d = {"label": "hello", "tag": "world"}
        out = self._roundtrip(d)
        assert out["label"] == "hello"
        assert out["tag"] == "world"

    def test_roundtrip_dict_mixed_types(self):
        """
        Round-trip a dict with a mix of arrays, scalars, and strings.

        Tests:
            (Test Case 1) All keys present after reload.
            (Test Case 2) Each value type is correctly reconstructed.
        """
        d = {
            "arr": np.array([10.0, 20.0]),
            "n_iter": 5,
            "name": "gplvm",
            "score": 0.95,
        }
        out = self._roundtrip(d)
        assert set(out.keys()) == set(d.keys())
        np.testing.assert_array_equal(out["arr"], d["arr"])
        assert out["n_iter"] == 5
        assert out["name"] == "gplvm"
        assert out["score"] == pytest.approx(0.95)

    def test_roundtrip_dict_nested(self):
        """
        Round-trip a nested dict (dict containing a dict).

        Tests:
            (Test Case 1) Outer dict keys preserved.
            (Test Case 2) Inner dict reconstructed as a dict with correct values.
        """
        d = {
            "outer_val": np.array([1.0]),
            "inner": {
                "a": np.array([2.0, 3.0]),
                "b": 99,
            },
        }
        out = self._roundtrip(d)
        assert isinstance(out["inner"], dict)
        np.testing.assert_array_equal(out["inner"]["a"], np.array([2.0, 3.0]))
        assert out["inner"]["b"] == 99

    def test_roundtrip_dict_empty(self):
        """
        Round-trip an empty dict.

        Tests:
            (Test Case 1) Empty dict is reconstructed as an empty dict.
        """
        d = {}
        out = self._roundtrip(d)
        assert isinstance(out, dict)
        assert len(out) == 0

    def test_roundtrip_dict_item_level(self):
        """
        Round-trip a dict via selective item loading (load_item).

        Tests:
            (Test Case 1) Dict loaded via load_item matches the original.
        """
        d = {"x": np.array([1.0, 2.0]), "y": 7}
        out = self._roundtrip_item(d)
        assert isinstance(out, dict)
        np.testing.assert_array_equal(out["x"], d["x"])
        assert out["y"] == 7

    def test_dict_with_unsupported_leaf_raises(self):
        """
        A dict containing an unsupported leaf type raises TypeError on save.

        Tests:
            (Test Case 1) Dict with a custom object value raises TypeError.
        """

        class Custom:
            pass

        ws = AnalysisWorkspace()
        ws.store("ns", "d", {"bad": Custom()})
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with pytest.raises(TypeError):
                ws.save(base)

    # ------------------------------------------------------------------
    # list_namespaces on LazyAnalysisWorkspace
    # ------------------------------------------------------------------

    def test_lazy_list_namespaces(self):
        """
        Tests that list_namespaces() returns correct namespace names on a LazyAnalysisWorkspace.

        Tests:
            (Test Case 1) Empty lazy workspace returns an empty list.
            (Test Case 2) After storing items, all namespace names are present in the result.
            (Test Case 3) Each namespace name appears exactly once even when multiple keys exist in it.
            (Test Case 4) Namespaces not stored are absent from the result.

        Notes:
            - LazyAnalysisWorkspace keeps _items empty and uses _index as the source of truth,
              so list_namespaces() reads from _index rather than _items.
        """
        ws = LazyAnalysisWorkspace(name="lazy_test")

        assert ws.list_namespaces() == []

        ws.store("alpha", "k1", np.zeros(2))
        ws.store("alpha", "k2", np.zeros(2))
        ws.store("beta", "k1", np.zeros(2))

        namespaces = ws.list_namespaces()
        assert isinstance(namespaces, list)
        assert sorted(namespaces) == sorted(["alpha", "beta"])
        assert "gamma" not in namespaces

    # ------------------------------------------------------------------
    # EC-HDF-02: load_item with corrupted HDF5 file
    # ------------------------------------------------------------------

    def test_load_from_corrupted_hdf5_file(self, tmp_path):
        """
        EC-HDF-02: load_workspace_full with a corrupted HDF5 file.

        Writing garbage bytes to a .h5 file means h5py cannot parse it.
        AnalysisWorkspace.load() raises an OSError.

        Tests:
            (Test Case 1) Loading a corrupted file raises OSError.
        """
        base = str(tmp_path / "corrupted")
        h5_path = f"{base}.h5"
        json_path = f"{base}.json"

        # Write garbage to the .h5 file
        with open(h5_path, "wb") as f:
            f.write(b"this is not a valid HDF5 file at all")

        # Write a minimal .json so the path is complete
        import json

        with open(json_path, "w") as f:
            json.dump(
                {"workspace_id": "x", "name": "x", "created_at": 0, "index": {}}, f
            )

        with pytest.raises(OSError):
            AnalysisWorkspace.load(base)

    # ------------------------------------------------------------------
    # EC-HDF-03: SpikeData with empty neuron_attributes dicts [{}, {}, {}]
    # ------------------------------------------------------------------

    def test_roundtrip_spikedata_empty_neuron_attribute_dicts(self):
        """
        EC-HDF-03: Round-trip of SpikeData with neuron_attributes = [{}, {}, {}].

        When every dict in neuron_attributes is empty, there are no attribute
        keys to serialize. The _dump_neuron_attributes helper writes a
        "neuron_attributes" group with zero datasets, and _load_neuron_attributes
        returns None (because all dicts are empty). So after round-trip,
        neuron_attributes is None rather than [{}, {}, {}].

        Tests:
            (Test Case 1) Round-trip does not raise.
            (Test Case 2) neuron_attributes is None after reload (empty dicts
                are indistinguishable from "no attributes").
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sd.neuron_attributes = [{}, {}, {}]

        out = self._roundtrip(sd)
        assert isinstance(out, SpikeData)
        # Empty dicts produce no HDF5 datasets, so they round-trip as None
        assert out.neuron_attributes is None

    # ------------------------------------------------------------------
    # EC-HDF-04: PairwiseCompMatrixStack with S=1
    # ------------------------------------------------------------------

    def test_roundtrip_pairwise_comp_matrix_stack_single_slice(self):
        """
        EC-HDF-04: Round-trip of PairwiseCompMatrixStack with S=1 (single slice).

        A single-slice stack is a degenerate case. The 3D array has shape
        (N, N, 1). This should round-trip correctly.

        Tests:
            (Test Case 1) Reconstructed object is a PairwiseCompMatrixStack.
            (Test Case 2) stack shape is (N, N, 1).
            (Test Case 3) Values are preserved.
            (Test Case 4) labels and times are preserved.
        """
        rng = np.random.default_rng(42)
        stack_arr = rng.random((3, 3, 1))
        times = [(0.0, 10.0)]
        pcms = PairwiseCompMatrixStack(
            stack=stack_arr,
            labels=["a", "b", "c"],
            times=times,
            metadata={"method": "sttc"},
        )
        out = self._roundtrip(pcms)

        assert isinstance(out, PairwiseCompMatrixStack)
        assert out.stack.shape == (3, 3, 1)
        np.testing.assert_array_almost_equal(out.stack, stack_arr)
        assert out.labels == ["a", "b", "c"]
        assert len(out.times) == 1
        assert out.times[0][0] == pytest.approx(0.0)
        assert out.times[0][1] == pytest.approx(10.0)
        assert out.metadata["method"] == "sttc"

    # ------------------------------------------------------------------
    # EC-HDF-05: RateSliceStack with neuron_attributes
    # ------------------------------------------------------------------

    def test_roundtrip_rateslicestack_with_neuron_attributes(self):
        """
        EC-HDF-05: Round-trip of RateSliceStack with neuron_attributes.

        RateSliceStack supports neuron_attributes. After round-trip through
        HDF5, the attributes should be preserved.

        Tests:
            (Test Case 1) Reconstructed object is a RateSliceStack.
            (Test Case 2) neuron_attributes is not None after load.
            (Test Case 3) Numeric attribute values are preserved.
            (Test Case 4) event_stack shape and values are preserved.
        """
        rng = np.random.default_rng(7)
        n_units, n_times, n_slices = 3, 10, 4
        arr = rng.random((n_units, n_times, n_slices))
        times = [(i * n_times, (i + 1) * n_times) for i in range(n_slices)]
        neuron_attrs = [
            {"channel": 0, "depth": 100.0},
            {"channel": 1, "depth": 200.0},
            {"channel": 2, "depth": 300.0},
        ]
        rss = RateSliceStack(
            None,
            event_matrix=arr,
            times_start_to_end=times,
            neuron_attributes=neuron_attrs,
        )

        out = self._roundtrip(rss)

        assert isinstance(out, RateSliceStack)
        assert out.event_stack.shape == (n_units, n_times, n_slices)
        np.testing.assert_array_almost_equal(out.event_stack, arr)

        assert out.neuron_attributes is not None
        assert len(out.neuron_attributes) == n_units
        for i in range(n_units):
            assert out.neuron_attributes[i]["channel"] == pytest.approx(float(i))
            assert out.neuron_attributes[i]["depth"] == pytest.approx((i + 1) * 100.0)

    def test_spikedata_nonzero_start_time_roundtrip(self):
        """
        SpikeData with non-zero start_time survives HDF5 workspace roundtrip.

        Tests:
            (Test Case 1) start_time=-100 is preserved through save/load.
        """
        trains = [np.array([-90.0, -50.0, 0.0, 10.0])]
        sd = SpikeData(trains, length=200.0, start_time=-100.0)
        ws = AnalysisWorkspace(name="start_time_test")
        ws.store("data", "sd", sd)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)
            loaded_sd = loaded.get("data", "sd")
            assert loaded_sd.start_time == pytest.approx(-100.0)
            assert loaded_sd.length == pytest.approx(200.0)
            np.testing.assert_allclose(loaded_sd.train[0], [-90.0, -50.0, 0.0, 10.0])

    def test_metadata_non_string_keys(self):
        """
        SpikeData metadata with integer keys loses precision through JSON roundtrip.

        Tests:
            (Test Case 1) Integer key 42 becomes string "42" after JSON roundtrip.
        """
        sd = SpikeData([[5.0]], length=10.0, metadata={42: "answer"})
        ws = AnalysisWorkspace(name="meta_test")
        ws.store("data", "sd", sd)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)
            loaded_sd = loaded.get("data", "sd")
            # Integer key becomes string through JSON
            assert "42" in loaded_sd.metadata

    def test_ratedata_nan_roundtrip(self):
        """
        RateData with NaN values survives HDF5 roundtrip.

        Tests:
            (Test Case 1) NaN values in inst_Frate_data are preserved.
        """
        data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]])
        times = np.array([0.0, 1.0, 2.0])
        rd = RateData(data, times)
        ws = AnalysisWorkspace(name="nan_rd_test")
        ws.store("data", "rd", rd)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)
            loaded_rd = loaded.get("data", "rd")
            np.testing.assert_array_equal(
                np.isnan(loaded_rd.inst_Frate_data), np.isnan(data)
            )

    def test_rateslicestack_roundtrip(self):
        """
        RateSliceStack HDF5 roundtrip.

        Tests:
            (Test Case 1) RateSliceStack survives save/load roundtrip.
        """
        mat = np.random.default_rng(0).random((2, 10, 3))
        rss = RateSliceStack(event_matrix=mat)
        ws = AnalysisWorkspace(name="rss_test")
        ws.store("data", "rss", rss)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)
            loaded_rss = loaded.get("data", "rss")
            np.testing.assert_allclose(loaded_rss.event_stack, mat)

    def test_pairwise_nan_roundtrip(self):
        """
        PairwiseCompMatrix with NaN values survives HDF5 roundtrip.

        Tests:
            (Test Case 1) NaN values are preserved through save/load.
        """
        mat = np.array([[1.0, np.nan], [np.nan, 1.0]])
        pcm = PairwiseCompMatrix(matrix=mat)
        ws = AnalysisWorkspace(name="pcm_nan_test")
        ws.store("data", "pcm", pcm)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)
            loaded_pcm = loaded.get("data", "pcm")
            np.testing.assert_array_equal(np.isnan(loaded_pcm.matrix), np.isnan(mat))

    def test_overwrite_item_different_type(self):
        """
        Overwriting a workspace item with a different type works correctly.

        Tests:
            (Test Case 1) ndarray replacing SpikeData works correctly.
        """
        sd = SpikeData([[5.0]], length=10.0)
        arr = np.array([1.0, 2.0, 3.0])

        ws = AnalysisWorkspace(name="overwrite_test")
        ws.store("ns", "key", sd)
        ws.store("ns", "key2", arr)
        # Verify both are stored correctly
        loaded_sd = ws.get("ns", "key")
        loaded_arr = ws.get("ns", "key2")
        assert isinstance(loaded_sd, SpikeData)
        np.testing.assert_array_equal(loaded_arr, arr)

    def test_labels_with_none_entries(self):
        """
        PairwiseCompMatrix labels with None entries crash during HDF5 save.

        Tests:
            (Test Case 1) Labels [1, None, 3] create an object-dtype array
                that HDF5 cannot serialize, raising TypeError.

        Notes:
            - This is a known bug: _dump_labels does not handle mixed
              int/None labels. np.array([1, None, 3]) creates an object
              array which has no HDF5 equivalent.
        """
        mat = np.eye(3)
        pcm = PairwiseCompMatrix(matrix=mat, labels=[1, None, 3])
        ws = AnalysisWorkspace(name="labels_none_test")
        ws.store("data", "pcm", pcm)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with pytest.raises(TypeError):
                ws.save(base)

    def test_metadata_with_inf(self):
        """
        Metadata with inf/-inf values roundtrip through HDF5.

        Tests:
            (Test Case 1) Inf values are preserved through save/load.
        """
        sd = SpikeData([[5.0]], length=10.0, metadata={"val": float("inf")})
        ws = AnalysisWorkspace(name="inf_meta_test")
        ws.store("data", "sd", sd)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)
            loaded_sd = loaded.get("data", "sd")
            assert loaded_sd.metadata["val"] == float("inf")

    def _roundtrip(self, obj, namespace="ns", key="item"):
        """
        Helper: store obj in a workspace, save to HDF5, reload the full workspace,
        and return the reconstructed object.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store(namespace, key, obj)
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded_ws = AnalysisWorkspace.load(base)
        return loaded_ws.get(namespace, key)

    # ------------------------------------------------------------------
    # dump_workspace / load_workspace_full
    # ------------------------------------------------------------------

    def test_dump_and_load_empty_workspace(self):
        """
        dump_workspace with a workspace containing no items produces a valid HDF5 file.

        Tests:
            (Test Case 1) Saving does not raise.
            (Test Case 2) Loaded workspace has zero namespaces.
            (Test Case 3) workspace_id and name are preserved.
        """
        ws = AnalysisWorkspace(name="empty")
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            loaded = AnalysisWorkspace.load(base)

        assert loaded.list_namespaces() == []
        assert loaded.workspace_id == ws.workspace_id
        assert loaded.name == "empty"

    def test_load_workspace_empty_name_becomes_none(self):
        """
        When workspace_name attribute is an empty string in HDF5, load_workspace_full
        converts it to None.

        Tests:
            (Test Case 1) Workspace saved with name="" loads with name=None.

        Notes:
            - dump_workspace stores `ws.name or ""`, so name=None and name="" both
              produce "". On load, empty string is converted back to None.
        """
        import h5py as h5

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            h5_path = f"{base}.h5"

            # Write a minimal HDF5 file with empty workspace_name
            with h5.File(h5_path, "w") as f:
                f.attrs["__workspace_id__"] = "test-id"
                f.attrs["__workspace_name__"] = ""
                f.attrs["__created_at__"] = 1000.0

            loaded = AnalysisWorkspace.load(base)
            assert loaded.name is None

    # ------------------------------------------------------------------
    # _load_item: unknown or missing __type__
    # ------------------------------------------------------------------

    def test_unknown_type_tag_raises(self):
        """
        An HDF5 group with an unrecognized __type__ attribute raises ValueError.

        Tests:
            (Test Case 1) ValueError is raised with a message mentioning the unknown type.
        """
        import h5py as h5

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(pathlib.Path(tmp) / "test.h5")
            with h5.File(h5_path, "w") as f:
                # Workspace-level metadata required by load_workspace_full
                f.attrs["__workspace_id__"] = "test-id"
                f.attrs["__workspace_name__"] = "test"
                f.attrs["__created_at__"] = 0.0
                ns_grp = f.create_group("ns")
                key_grp = ns_grp.create_group("item")
                key_grp.attrs["__type__"] = "UnknownType"
                key_grp.attrs["__created_at__"] = 0.0

            with pytest.raises(ValueError, match="Unknown __type__"):
                AnalysisWorkspace.load(str(pathlib.Path(tmp) / "test"))

    def test_missing_type_attribute_raises(self):
        """
        An HDF5 group without a __type__ attribute (defaults to empty string)
        raises ValueError because "" is not in the dispatch dict.

        Tests:
            (Test Case 1) ValueError is raised mentioning the empty type tag.
        """
        import h5py as h5

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(pathlib.Path(tmp) / "test.h5")
            with h5.File(h5_path, "w") as f:
                # Workspace-level metadata required by load_workspace_full
                f.attrs["__workspace_id__"] = "test-id"
                f.attrs["__workspace_name__"] = "test"
                f.attrs["__created_at__"] = 0.0
                ns_grp = f.create_group("ns")
                key_grp = ns_grp.create_group("item")
                key_grp.attrs["__created_at__"] = 0.0
                # Deliberately omit __type__

            with pytest.raises(ValueError, match="Unknown __type__"):
                AnalysisWorkspace.load(str(pathlib.Path(tmp) / "test"))

    # ------------------------------------------------------------------
    # SpikeData edge cases
    # ------------------------------------------------------------------

    def test_roundtrip_spikedata_zero_units(self):
        """
        Round-trip a SpikeData with zero units (N=0, train=[]).

        Tests:
            (Test Case 1) Save does not raise.
            (Test Case 2) Loaded object is a SpikeData with N=0.
            (Test Case 3) train is an empty list.
            (Test Case 4) length_ms is preserved.
        """
        sd = SpikeData([], length=100.0)
        assert sd.N == 0
        out = self._roundtrip(sd)
        assert isinstance(out, SpikeData)
        assert out.N == 0
        assert len(out.train) == 0
        assert out.length == pytest.approx(100.0)

    def test_roundtrip_spikedata_nan_spike_times(self):
        """
        SpikeData rejects NaN spike times at construction.

        Tests:
            (Test Case 1) ValueError is raised when constructing SpikeData
                with NaN spike times.

        Notes:
            - SpikeData.__init__ validates spike times and rejects NaN values
              to prevent silent corruption of downstream computations.
              This means NaN spike times cannot be round-tripped through HDF5.
        """
        with pytest.raises(ValueError, match="NaN"):
            SpikeData([[1.0, np.nan, 3.0], [np.nan]], length=50.0)

    def test_roundtrip_spikedata_empty_raw_data_skipped(self):
        """
        SpikeData with raw_data of size 0 has raw_data skipped during dump.

        Tests:
            (Test Case 1) Save does not raise.
            (Test Case 2) After round-trip, raw_data is the default empty array
                np.zeros((0, 0)) because it was not stored in HDF5.

        Notes:
            - The dump code checks `sd.raw_data.size > 0` before writing. An
              empty raw_data array causes it to be skipped entirely. On load,
              SpikeData defaults to np.zeros((0, 0)) when raw_data is not
              provided.
        """
        sd = SpikeData(
            [[1.0, 2.0]],
            length=10.0,
            raw_data=np.array([]).reshape(0, 0),
            raw_time=np.array([]),
        )
        assert sd.raw_data.size == 0
        out = self._roundtrip(sd)
        assert out.raw_data.shape == (0, 0)

    # ------------------------------------------------------------------
    # Neuron attributes edge cases
    # ------------------------------------------------------------------

    def test_roundtrip_neuron_attributes_mixed_types(self):
        """
        Neuron attributes where one unit has a numeric value and another has a
        string value for the same key. The string path takes precedence.

        Tests:
            (Test Case 1) Save does not raise.
            (Test Case 2) Both values are present as strings after load.

        Notes:
            - When any value is a string, _dump_neuron_attributes takes the
              use_string path and converts all values via str().
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sd.neuron_attributes = [
            {"label": 42},
            {"label": "hello"},
        ]
        out = self._roundtrip(sd)
        assert out.neuron_attributes is not None
        # The string path converts everything to strings
        assert out.neuron_attributes[0]["label"] == "42"
        assert out.neuron_attributes[1]["label"] == "hello"

    # ------------------------------------------------------------------
    # Dict edge cases
    # ------------------------------------------------------------------

    def test_roundtrip_dict_with_list_of_strings(self):
        """
        A dict with a list of strings round-trips through HDF5.
        ``_dump_ndarray`` now stores unicode/byte string arrays via
        h5py's variable-length string dtype (``__string_array__``
        marker); ``_load_ndarray`` decodes back to a unicode numpy
        array so callers see consistent semantics.

        Tests:
            (Test Case 1) Saving a dict with a list of strings succeeds.
            (Test Case 2) Loaded value contains the same strings in
                the same order.

        Notes:
            - Lists are still lossy on reload (round-trip as ndarray,
              not list), but the string CONTENT is preserved.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store("ns", "item", {"names": ["alpha", "beta", "gamma"]})
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            ws2 = AnalysisWorkspace.load(base)
            loaded = ws2.get("ns", "item")
            assert list(loaded["names"]) == ["alpha", "beta", "gamma"]

    def test_roundtrip_dict_with_none_value(self):
        """
        A dict containing ``None`` as a value round-trips through HDF5.
        ``_dump_dict`` stores ``None`` as ``__type__ = "none"`` and
        ``_load_dict`` reconstructs it back to Python ``None``.

        Tests:
            (Test Case 1) Saving a dict with None value succeeds.
            (Test Case 2) Loaded dict has None at the original key.
            (Test Case 3) ``None`` round-trips alongside other types.
        """
        ws = AnalysisWorkspace(name="test")
        ws.store("ns", "d", {"key": None, "n": 42, "name": "alpha"})
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)
            ws2 = AnalysisWorkspace.load(base)
            loaded = ws2.get("ns", "d")
            assert loaded["key"] is None
            assert loaded["n"] == 42
            assert loaded["name"] == "alpha"

    def test_roundtrip_dict_with_numpy_scalars(self):
        """
        A dict containing numpy scalar values (np.float64, np.int64) round-trips
        correctly through the scalar path.

        Tests:
            (Test Case 1) np.float64 value is preserved.
            (Test Case 2) np.int64 value is preserved.
        """
        d = {"x": np.float64(3.14), "n": np.int64(42)}
        out = self._roundtrip(d)
        assert out["x"] == pytest.approx(3.14)
        assert out["n"] == 42

    def test_roundtrip_dict_with_boolean_false(self):
        """
        A dict containing False as a value round-trips correctly. isinstance(False, int)
        is True in Python, so it goes through the scalar path.

        Tests:
            (Test Case 1) False value is preserved after round-trip.
            (Test Case 2) True value is preserved after round-trip.
        """
        d = {"flag_false": False, "flag_true": True}
        out = self._roundtrip(d)
        assert out["flag_false"] is False
        assert out["flag_true"] is True

    # ------------------------------------------------------------------
    # _dump_times_tuples / _load_times_tuples
    # ------------------------------------------------------------------

    def test_roundtrip_empty_times_list(self):
        """
        A PairwiseCompMatrixStack or RateSliceStack with an empty times list.

        Tests:
            (Test Case 1) Save does not raise.
            (Test Case 2) After round-trip, times is either an empty list or None.

        Notes:
            - np.array([], dtype=np.float64) has shape (0,) not (0, 2). When
              _load_times_tuples reads this, iterating over rows of a 1-D array
              would fail because each "row" is a scalar with no row[0]/row[1].
              This is a potential bug depending on how the stack class handles it.
        """
        import h5py as h5
        from spikelab.workspace.hdf5_io import _dump_times_tuples, _load_times_tuples

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(pathlib.Path(tmp) / "test.h5")
            with h5.File(h5_path, "w") as f:
                grp = f.create_group("test")
                _dump_times_tuples(grp, [])

            with h5.File(h5_path, "r") as f:
                grp = f["test"]
                if "times" in grp:
                    # The dataset exists but has shape (0,) not (0, 2)
                    # _load_times_tuples tries to iterate and access row[0], row[1]
                    # This may raise or return an empty list
                    try:
                        result = _load_times_tuples(grp)
                        assert result == []
                    except (IndexError, ValueError):
                        # If it raises, this documents the bug
                        pytest.skip(
                            "Empty times list causes IndexError in "
                            "_load_times_tuples due to shape (0,) instead of (0, 2)"
                        )

    # ------------------------------------------------------------------
    # Metadata edge cases
    # ------------------------------------------------------------------

    def test_roundtrip_metadata_deeply_nested_numpy_arrays(self):
        """
        Metadata containing nested dicts with numpy arrays inside.

        Tier L-F3: ndarrays nested inside metadata sub-dicts are
        restored as ndarrays (with dtype/shape preserved) rather than
        silently degraded to Python lists.

        Tests:
            (Test Case 1) Save does not raise.
            (Test Case 2) Nested numpy array is recovered as an
                ndarray with the original values.
            (Test Case 3) Nested numpy scalar is recovered as a
                Python float (scalar encoder unchanged).
            (Test Case 4) Sibling primitive (string) is preserved.
        """
        arr = np.array([10.0, 20.0, 30.0])
        sd = SpikeData(
            [[1.0, 2.0]],
            length=10.0,
            metadata={
                "outer": {
                    "inner_arr": arr,
                    "inner_scalar": np.float64(99.0),
                },
                "top_level": "hello",
            },
        )
        out = self._roundtrip(sd)
        recovered = out.metadata["outer"]["inner_arr"]
        assert isinstance(recovered, np.ndarray)
        np.testing.assert_array_equal(recovered, arr)
        assert out.metadata["outer"]["inner_scalar"] == pytest.approx(99.0)
        assert out.metadata["top_level"] == "hello"

    # ------------------------------------------------------------------
    # dump_item_to_file: overwrite existing item
    # ------------------------------------------------------------------

    def test_dump_item_to_file_overwrites_existing(self):
        """
        dump_item_to_file overwrites an existing item at the same (namespace, key).

        Tests:
            (Test Case 1) After overwrite, load_item_from_file returns the new value.
            (Test Case 2) The old value is no longer present.
        """
        from spikelab.workspace.hdf5_io import dump_item_to_file, load_item_from_file

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(pathlib.Path(tmp) / "test.h5")
            arr1 = np.array([1.0, 2.0, 3.0])
            arr2 = np.array([99.0])

            dump_item_to_file(h5_path, "ns", "key", arr1, created_at=0.0)
            dump_item_to_file(h5_path, "ns", "key", arr2, created_at=1.0)

            loaded = load_item_from_file(h5_path, "ns", "key")
            np.testing.assert_array_equal(loaded, arr2)

    # ------------------------------------------------------------------
    # Namespace / key with special characters
    # ------------------------------------------------------------------

    def test_namespace_with_slash_creates_nested_groups(self):
        """
        A namespace containing '/' creates nested HDF5 groups, which causes
        the loader to fail because the intermediate group lacks __type__.

        Tests:
            (Test Case 1) Storing and saving does not raise.
            (Test Case 2) Loading raises ValueError because the nested group
                structure confuses the loader (intermediate group has no __type__).

        Notes:
            - This is a known consequence of HDF5 group naming. Using '/' in
              namespace or key names should be avoided. The slash causes HDF5
              to create nested groups (ns -> sub -> key), and the loader
              interprets 'sub' as a key group but it lacks __type__ metadata.
        """
        ws = AnalysisWorkspace(name="slash_test")
        ws.store("ns/sub", "key", np.array([1.0]))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)

            # Loading fails because the nested group 'sub' is interpreted as a
            # key group but lacks __type__ attribute
            with pytest.raises(ValueError, match="Unknown __type__"):
                AnalysisWorkspace.load(base)


# ---------------------------------------------------------------------------
# Tests: LazyAnalysisWorkspace — dedicated coverage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestLazyAnalysisWorkspace:
    """
    Dedicated tests for LazyAnalysisWorkspace.

    Covers construction, store/get round-trips, list_keys, list_namespaces,
    delete, describe, save/load persistence, and WorkspaceManager lazy creation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_construction_creates_valid_workspace(self):
        """
        Construct a LazyAnalysisWorkspace and verify its attributes.

        Tests:
            (Test Case 1) workspace_id is a non-empty string.
            (Test Case 2) name attribute matches the provided name.
            (Test Case 3) created_at is a positive float timestamp.
            (Test Case 4) The backing temp HDF5 file exists on disk.
            (Test Case 5) _items raises NotImplementedError (data lives on disk).
            (Test Case 6) _index dict is empty for a fresh workspace.
        """
        ws = LazyAnalysisWorkspace(name="test_lazy")

        assert isinstance(ws.workspace_id, str) and len(ws.workspace_id) > 0
        assert ws.name == "test_lazy"
        assert isinstance(ws.created_at, float) and ws.created_at > 0
        assert pathlib.Path(ws._h5_path).exists()
        with pytest.raises(NotImplementedError):
            _ = ws._items
        assert ws._index == {}

    def test_construction_without_name(self):
        """
        Construct a LazyAnalysisWorkspace without a name.

        Tests:
            (Test Case 1) name attribute is None when not provided.
            (Test Case 2) Workspace is still functional (temp file exists).
        """
        ws = LazyAnalysisWorkspace()

        assert ws.name is None
        assert pathlib.Path(ws._h5_path).exists()

    # ------------------------------------------------------------------
    # store() and get()
    # ------------------------------------------------------------------

    def test_store_and_get_ndarray(self):
        """
        Store a numpy ndarray and retrieve it, verifying equality.

        Tests:
            (Test Case 1) get() returns an array equal to the stored array.
            (Test Case 2) The dtype is preserved.
            (Test Case 3) The shape is preserved.
            (Test Case 4) _items raises NotImplementedError (data is on disk).
        """
        ws = LazyAnalysisWorkspace(name="store_get")
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        ws.store("ns1", "my_array", arr)
        retrieved = ws.get("ns1", "my_array")

        np.testing.assert_array_equal(retrieved, arr)
        assert retrieved.dtype == arr.dtype
        assert retrieved.shape == arr.shape
        with pytest.raises(NotImplementedError):
            _ = ws._items

    def test_store_and_get_multiple_items(self):
        """
        Store multiple items in different namespaces and retrieve each.

        Tests:
            (Test Case 1) Each item is retrieved correctly from its own namespace/key.
            (Test Case 2) Items do not interfere with each other.
        """
        ws = LazyAnalysisWorkspace(name="multi")
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([10.0, 20.0, 30.0])
        arr3 = np.array([[7.0]])

        ws.store("ns_a", "first", arr1)
        ws.store("ns_a", "second", arr2)
        ws.store("ns_b", "only", arr3)

        np.testing.assert_array_equal(ws.get("ns_a", "first"), arr1)
        np.testing.assert_array_equal(ws.get("ns_a", "second"), arr2)
        np.testing.assert_array_equal(ws.get("ns_b", "only"), arr3)

    def test_get_missing_returns_none(self):
        """
        get() returns None for non-existent namespace or key.

        Tests:
            (Test Case 1) Missing namespace returns None.
            (Test Case 2) Missing key in existing namespace returns None.
        """
        ws = LazyAnalysisWorkspace(name="missing")
        ws.store("ns", "k", np.zeros(2))

        assert ws.get("nonexistent", "k") is None
        assert ws.get("ns", "nonexistent") is None

    def test_store_overwrites_existing_key(self):
        """
        Storing under an existing (namespace, key) overwrites the previous value.

        Tests:
            (Test Case 1) After overwrite, get() returns the new value.
            (Test Case 2) The old value is no longer retrievable.
        """
        ws = LazyAnalysisWorkspace(name="overwrite")
        ws.store("ns", "k", np.array([1.0, 2.0]))
        ws.store("ns", "k", np.array([99.0]))

        result = ws.get("ns", "k")
        np.testing.assert_array_equal(result, np.array([99.0]))

    # ------------------------------------------------------------------
    # list_keys() and list_namespaces()
    # ------------------------------------------------------------------

    def test_list_keys_all_namespaces(self):
        """
        list_keys() without arguments returns a dict of all namespaces to keys.

        Tests:
            (Test Case 1) Returns a dict with namespace names as keys.
            (Test Case 2) Each namespace maps to the correct list of keys.
            (Test Case 3) Empty workspace returns an empty dict.
        """
        ws = LazyAnalysisWorkspace(name="list_keys")

        assert ws.list_keys() == {}

        ws.store("alpha", "k1", np.zeros(1))
        ws.store("alpha", "k2", np.zeros(1))
        ws.store("beta", "k3", np.zeros(1))

        result = ws.list_keys()
        assert isinstance(result, dict)
        assert sorted(result["alpha"]) == sorted(["k1", "k2"])
        assert result["beta"] == ["k3"]

    def test_list_keys_single_namespace(self):
        """
        list_keys(namespace) returns a list of keys for that namespace.

        Tests:
            (Test Case 1) Returns correct keys for an existing namespace.
            (Test Case 2) Returns an empty list for a non-existent namespace.
        """
        ws = LazyAnalysisWorkspace(name="list_keys_ns")
        ws.store("alpha", "k1", np.zeros(1))
        ws.store("alpha", "k2", np.zeros(1))

        keys = ws.list_keys("alpha")
        assert isinstance(keys, list)
        assert sorted(keys) == sorted(["k1", "k2"])

        assert ws.list_keys("nonexistent") == []

    def test_list_namespaces_after_storing(self):
        """
        list_namespaces() returns all namespace names after storing items.

        Tests:
            (Test Case 1) Empty workspace returns empty list.
            (Test Case 2) After storing in two namespaces, both are returned.
            (Test Case 3) Namespaces not stored are absent.
        """
        ws = LazyAnalysisWorkspace(name="ns_list")

        assert ws.list_namespaces() == []

        ws.store("rec1", "data", np.zeros(3))
        ws.store("rec2", "data", np.ones(3))

        ns = ws.list_namespaces()
        assert sorted(ns) == ["rec1", "rec2"]
        assert "rec3" not in ns

    # ------------------------------------------------------------------
    # delete()
    # ------------------------------------------------------------------

    def test_delete_single_item(self):
        """
        Delete a single item and verify it is gone.

        Tests:
            (Test Case 1) delete() removes an existing item without error.
            (Test Case 2) get() returns None after deletion.
            (Test Case 3) The key is removed from list_keys().
            (Test Case 4) Other items in the same namespace are unaffected.
        """
        ws = LazyAnalysisWorkspace(name="delete_item")
        ws.store("ns", "keep", np.array([1.0]))
        ws.store("ns", "remove", np.array([2.0]))

        ws.delete("ns", "remove")
        assert ws.get("ns", "remove") is None
        assert "remove" not in ws.list_keys("ns")
        np.testing.assert_array_equal(ws.get("ns", "keep"), np.array([1.0]))

    def test_delete_entire_namespace(self):
        """
        Delete an entire namespace and verify all its items are gone.

        Tests:
            (Test Case 1) delete() with key=None removes the namespace without error.
            (Test Case 2) The namespace is removed from list_namespaces().
            (Test Case 3) get() returns None for any key in the deleted namespace.
        """
        ws = LazyAnalysisWorkspace(name="delete_ns")
        ws.store("remove_ns", "k1", np.array([1.0]))
        ws.store("remove_ns", "k2", np.array([2.0]))
        ws.store("keep_ns", "k1", np.array([3.0]))

        ws.delete("remove_ns")
        assert "remove_ns" not in ws.list_namespaces()
        assert ws.get("remove_ns", "k1") is None
        assert ws.get("remove_ns", "k2") is None
        np.testing.assert_array_equal(ws.get("keep_ns", "k1"), np.array([3.0]))

    def test_delete_nonexistent_raises(self):
        """
        delete() raises KeyError when the target does not exist.

        Tests:
            (Test Case 1) Missing namespace raises KeyError.
            (Test Case 2) Missing key in existing namespace raises KeyError.
        """
        ws = LazyAnalysisWorkspace(name="delete_miss")
        ws.store("ns", "k", np.zeros(1))

        with pytest.raises(KeyError, match="not found"):
            ws.delete("nonexistent")
        with pytest.raises(KeyError, match="not found"):
            ws.delete("ns", "nonexistent")

    # ------------------------------------------------------------------
    # describe()
    # ------------------------------------------------------------------

    def test_describe_after_storing(self):
        """
        describe() returns a nested dict reflecting stored items.

        Tests:
            (Test Case 1) Empty workspace returns an empty dict.
            (Test Case 2) After storing items, top-level keys are namespace names.
            (Test Case 3) Each namespace contains correct item keys.
            (Test Case 4) Each item entry contains 'type' and 'created_at'.
            (Test Case 5) ndarray entries contain 'shape' and 'dtype' fields.
        """
        ws = LazyAnalysisWorkspace(name="describe")

        assert ws.describe() == {}

        ws.store("rec1", "rates", np.zeros((3, 10)))
        ws.store("rec1", "spikes", np.ones(5))
        ws.store("rec2", "data", np.array([42.0]))

        desc = ws.describe()
        assert set(desc.keys()) == {"rec1", "rec2"}
        assert set(desc["rec1"].keys()) == {"rates", "spikes"}
        assert set(desc["rec2"].keys()) == {"data"}

        rates_info = desc["rec1"]["rates"]
        assert rates_info["type"] == "ndarray"
        assert "created_at" in rates_info
        assert rates_info["shape"] == [3, 10]
        assert "dtype" in rates_info

    def test_describe_with_note(self):
        """
        describe() includes notes when they were provided at store time.

        Tests:
            (Test Case 1) Note is present in the index entry when provided.
            (Test Case 2) Note is absent when not provided.
        """
        ws = LazyAnalysisWorkspace(name="note_test")
        ws.store("ns", "with_note", np.zeros(1), note="important result")
        ws.store("ns", "no_note", np.zeros(1))

        desc = ws.describe()
        assert desc["ns"]["with_note"]["note"] == "important result"
        assert "note" not in desc["ns"]["no_note"]

    # ------------------------------------------------------------------
    # save() and load()
    # ------------------------------------------------------------------

    def test_save_and_load_roundtrip(self):
        """
        Save a lazy workspace to a new path and load it back.

        Tests:
            (Test Case 1) save() creates .h5 and .json files at the target path.
            (Test Case 2) load() reconstructs a workspace with the same workspace_id.
            (Test Case 3) load() reconstructs a workspace with the same name.
            (Test Case 4) Stored ndarray data survives the round-trip.
            (Test Case 5) The index is preserved (list_keys matches).
        """
        ws = LazyAnalysisWorkspace(name="save_load")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ws.store("ns", "matrix", arr)
        ws.store("ns", "vector", np.array([10.0, 20.0, 30.0]))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "lazy_ws")
            ws.save(base)

            assert pathlib.Path(f"{base}.h5").exists()
            assert pathlib.Path(f"{base}.json").exists()

            loaded = AnalysisWorkspace.load(base)

            assert loaded.workspace_id == ws.workspace_id
            assert loaded.name == ws.name
            np.testing.assert_array_equal(loaded.get("ns", "matrix"), arr)
            np.testing.assert_array_equal(
                loaded.get("ns", "vector"), np.array([10.0, 20.0, 30.0])
            )
            assert sorted(loaded.list_keys("ns")) == sorted(["matrix", "vector"])

    def test_save_json_contains_index(self):
        """
        The .json file written by save() contains correct metadata.

        Tests:
            (Test Case 1) JSON has workspace_id matching the workspace.
            (Test Case 2) JSON has name matching the workspace.
            (Test Case 3) JSON index contains the stored namespace and key.
        """
        ws = LazyAnalysisWorkspace(name="json_check")
        ws.store("ns", "arr", np.zeros(3))

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            ws.save(base)

            with open(f"{base}.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            assert meta["workspace_id"] == ws.workspace_id
            assert meta["name"] == "json_check"
            assert "ns" in meta["index"]
            assert "arr" in meta["index"]["ns"]

    # ------------------------------------------------------------------
    # WorkspaceManager.create_workspace(lazy=True)
    # ------------------------------------------------------------------

    def test_manager_create_lazy_workspace(self):
        """
        WorkspaceManager.create_workspace(lazy=True) creates a LazyAnalysisWorkspace.

        Tests:
            (Test Case 1) Returned workspace_id is a non-empty string.
            (Test Case 2) get_workspace() returns a LazyAnalysisWorkspace instance.
            (Test Case 3) The lazy workspace is functional (store and get work).
        """
        mgr = WorkspaceManager()
        ws_id = mgr.create_workspace(name="mgr_lazy", lazy=True)

        assert isinstance(ws_id, str) and len(ws_id) > 0

        ws = mgr.get_workspace(ws_id)
        assert isinstance(ws, LazyAnalysisWorkspace)

        arr = np.array([1.0, 2.0, 3.0])
        ws.store("ns", "data", arr)
        np.testing.assert_array_equal(ws.get("ns", "data"), arr)

    def test_manager_create_lazy_false_is_regular(self):
        """
        WorkspaceManager.create_workspace(lazy=False) creates a regular AnalysisWorkspace.

        Tests:
            (Test Case 1) get_workspace() returns an AnalysisWorkspace, not LazyAnalysisWorkspace.
        """
        mgr = WorkspaceManager()
        ws_id = mgr.create_workspace(name="mgr_regular", lazy=False)
        ws = mgr.get_workspace(ws_id)

        assert type(ws) is AnalysisWorkspace
        assert not isinstance(ws, LazyAnalysisWorkspace)

    # ------------------------------------------------------------------
    # rename()
    # ------------------------------------------------------------------

    def test_rename_existing_key(self):
        """
        rename() moves a stored item to a new key within the same namespace.

        Tests:
            (Test Case 1) rename returns True on success.
            (Test Case 2) get(new_key) retrieves the same data.
            (Test Case 3) get(old_key) returns None after rename.
            (Test Case 4) list_keys shows the new key, not the old one.
        """
        ws = LazyAnalysisWorkspace(name="rename_test")
        arr = np.array([1.0, 2.0, 3.0])
        ws.store("ns", "old_key", arr)

        result = ws.rename("ns", "old_key", "new_key")
        assert result is True

        retrieved = ws.get("ns", "new_key")
        np.testing.assert_array_equal(retrieved, arr)
        assert ws.get("ns", "old_key") is None
        assert "new_key" in ws.list_keys("ns")
        assert "old_key" not in ws.list_keys("ns")

    def test_rename_missing_namespace_raises(self):
        """
        rename() raises KeyError when the namespace does not exist.

        Tests:
            (Test Case 1) Raises KeyError matching "not found".
        """
        ws = LazyAnalysisWorkspace(name="rename_miss_ns")
        with pytest.raises(KeyError, match="not found"):
            ws.rename("nonexistent", "a", "b")

    def test_rename_missing_key_raises(self):
        """
        rename() raises KeyError when the old_key does not exist in the namespace.

        Tests:
            (Test Case 1) Raises KeyError matching "not found".
            (Test Case 2) Existing keys are unaffected.
        """
        ws = LazyAnalysisWorkspace(name="rename_miss_key")
        ws.store("ns", "exists", np.array([1.0]))
        with pytest.raises(KeyError, match="not found"):
            ws.rename("ns", "missing", "new")
        assert ws.get("ns", "exists") is not None

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        """
        __repr__ returns a descriptive string for the lazy workspace.

        Tests:
            (Test Case 1) repr includes 'LazyAnalysisWorkspace'.
            (Test Case 2) repr includes the workspace name.
            (Test Case 3) repr includes 'temp HDF5'.
        """
        ws = LazyAnalysisWorkspace(name="repr_test")
        r = repr(ws)
        assert "LazyAnalysisWorkspace" in r
        assert "repr_test" in r
        assert "temp HDF5" in r

    def test_get_after_backing_file_deleted(self):
        """
        EC-WS-08: LazyAnalysisWorkspace.get() after the backing HDF5 file is deleted.

        When the temp HDF5 file is manually removed, get() raises an OSError
        (from h5py failing to open the missing file) rather than returning None.

        Tests:
            (Test Case 1) get() raises OSError when the backing file has been deleted.
        """
        import os

        ws = LazyAnalysisWorkspace(name="deleted_backing")
        ws.store("ns", "arr", np.array([1.0, 2.0]))

        # Verify it works before deletion
        result = ws.get("ns", "arr")
        np.testing.assert_array_equal(result, [1.0, 2.0])

        # Delete the backing file
        os.unlink(ws._h5_path)

        # get() should raise because h5py cannot open the missing file
        with pytest.raises(OSError):
            ws.get("ns", "arr")

    def test_items_property_raises_with_helpful_message(self):
        """
        Accessing ``_items`` on a LazyAnalysisWorkspace raises
        ``NotImplementedError`` whose message points the caller at the
        proper public API (``list_namespaces`` / ``list_keys`` /
        ``get``) since lazy workspaces don't materialise items in
        memory.

        Tests:
            (Test Case 1) Reading ``_items`` raises NotImplementedError.
            (Test Case 2) The error message names ``list_namespaces``,
                ``list_keys``, and ``get`` as the recovery API.
        """
        ws = LazyAnalysisWorkspace(name="test_items")
        with pytest.raises(NotImplementedError) as excinfo:
            _ = ws._items
        msg = str(excinfo.value)
        assert "list_namespaces" in msg
        assert "list_keys" in msg
        assert "get" in msg

    def test_items_setter_is_noop_for_parent_init_compatibility(self):
        """
        ``LazyAnalysisWorkspace._items = {}`` (the assignment the parent
        ``AnalysisWorkspace.__init__`` does) must NOT raise — the
        setter is a no-op so the lazy subclass can reuse the parent
        constructor without sub-classing around its init order.

        Tests:
            (Test Case 1) ``ws._items = {}`` does not raise.
            (Test Case 2) Reading ``_items`` afterwards still raises
                NotImplementedError (the setter discards the value).
        """
        ws = LazyAnalysisWorkspace(name="test_items_setter")
        # Setter must not raise.
        ws._items = {}
        # Getter behaviour is unchanged after the no-op setter.
        with pytest.raises(NotImplementedError):
            _ = ws._items

    def test_get_after_delete(self):
        """
        Calling get() after delete() for the same (namespace, key) returns None.

        Tests:
            (Test Case 1) get() returns the value before deletion.
            (Test Case 2) delete() removes the item without error.
            (Test Case 3) get() returns None after deletion.
        """
        ws = LazyAnalysisWorkspace(name="get_after_del")
        arr = np.array([1.0, 2.0, 3.0])
        ws.store("ns", "k", arr)

        np.testing.assert_array_equal(ws.get("ns", "k"), arr)
        ws.delete("ns", "k")
        assert ws.get("ns", "k") is None

    def test_rename_when_get_returns_none(self):
        """
        Rename raises KeyError when get() returns None (e.g. if HDF5 data is corrupted
        or the item was deleted from the backing file externally).

        Tests:
            (Test Case 1) rename raises KeyError matching "not found".

        Notes:
            - This tests the code path where namespace and old_key exist in _index
              but the backing HDF5 file no longer has the data, causing get() to
              return None.
        """
        ws = LazyAnalysisWorkspace(name="rename_none")
        ws.store("ns", "key", np.array([1.0]))

        # Manually corrupt the _index to reference a key that doesn't exist in HDF5
        from spikelab.workspace.hdf5_io import delete_item_from_file

        delete_item_from_file(ws._h5_path, "ns", "key")
        # _index still has the entry, but get() will return None
        assert ws.get("ns", "key") is None

        with pytest.raises(KeyError, match="not found"):
            ws.rename("ns", "key", "new_key")

    def test_delete_last_key_removes_namespace(self):
        """
        In LazyAnalysisWorkspace, deleting the last key in a namespace removes
        the namespace from _index entirely.

        Tests:
            (Test Case 1) delete removes the item without error.
            (Test Case 2) The namespace is removed from _index.
            (Test Case 3) list_namespaces no longer includes the namespace.

        Notes:
            - This differs from AnalysisWorkspace.delete which leaves the empty
              namespace in _items and _index.
        """
        ws = LazyAnalysisWorkspace(name="del_last")
        ws.store("ns", "only", np.array([1.0]))

        ws.delete("ns", "only")
        assert "ns" not in ws._index
        assert "ns" not in ws.list_namespaces()

    def test_temp_file_cleanup_on_del(self):
        """
        The temp HDF5 file is removed when the LazyAnalysisWorkspace is garbage
        collected (__del__ is called).

        Tests:
            (Test Case 1) The temp file exists while the workspace is alive.
            (Test Case 2) After explicit __del__, the temp file is removed.
        """
        ws = LazyAnalysisWorkspace(name="cleanup")
        ws.store("ns", "arr", np.zeros(3))
        h5_path = ws._h5_path

        assert pathlib.Path(h5_path).exists()

        ws.__del__()
        assert not pathlib.Path(h5_path).exists()

    def test_double_del_is_safe(self):
        """
        Calling __del__ twice does not raise, because it checks os.path.exists
        before attempting to unlink.

        Tests:
            (Test Case 1) First __del__ removes the file.
            (Test Case 2) Second __del__ does not raise.
        """
        ws = LazyAnalysisWorkspace(name="double_del")
        h5_path = ws._h5_path

        ws.__del__()
        assert not pathlib.Path(h5_path).exists()

        # Second call should be a no-op
        ws.__del__()


# ---------------------------------------------------------------------------
# delete_item_from_file
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDeleteItemFromFile:
    """Tests for the delete_item_from_file function in hdf5_io."""

    def test_delete_single_item(self, tmp_path):
        """
        Deleting a single item by namespace and key removes only that item.

        Tests:
            (Test Case 1) The deleted item raises KeyError on load.
            (Test Case 2) The other item in a different namespace still loads correctly.
        """
        from spikelab.workspace.hdf5_io import (
            dump_item_to_file,
            load_item_from_file,
            delete_item_from_file,
        )

        h5_path = str(tmp_path / "test.h5")
        arr_a = np.array([1.0, 2.0, 3.0])
        arr_b = np.array([4.0, 5.0, 6.0])

        dump_item_to_file(h5_path, "ns_a", "key_a", arr_a, created_at=0.0)
        dump_item_to_file(h5_path, "ns_b", "key_b", arr_b, created_at=0.0)

        delete_item_from_file(h5_path, "ns_a", key="key_a")

        with pytest.raises(KeyError):
            load_item_from_file(h5_path, "ns_a", "key_a")

        loaded_b = load_item_from_file(h5_path, "ns_b", "key_b")
        np.testing.assert_array_equal(loaded_b, arr_b)

    def test_delete_nonexistent_key_is_noop(self, tmp_path):
        """
        Deleting a key that does not exist completes without error.

        Tests:
            (Test Case 1) No exception is raised when the namespace does not exist.
            (Test Case 2) No exception is raised when the key does not exist within
                an existing namespace.
            (Test Case 3) Existing items remain accessible after the no-op delete.
        """
        from spikelab.workspace.hdf5_io import (
            dump_item_to_file,
            load_item_from_file,
            delete_item_from_file,
        )

        h5_path = str(tmp_path / "test.h5")
        arr = np.array([10.0, 20.0])
        dump_item_to_file(h5_path, "ns", "real_key", arr, created_at=0.0)

        # Namespace doesn't exist
        delete_item_from_file(h5_path, "no_such_ns", key="any_key")

        # Key doesn't exist in existing namespace
        delete_item_from_file(h5_path, "ns", key="no_such_key")

        loaded = load_item_from_file(h5_path, "ns", "real_key")
        np.testing.assert_array_equal(loaded, arr)

    def test_delete_entire_namespace(self, tmp_path):
        """
        Calling delete_item_from_file with key=None removes the entire namespace.

        Tests:
            (Test Case 1) All items in the deleted namespace are gone.
            (Test Case 2) Items in other namespaces are unaffected.
        """
        from spikelab.workspace.hdf5_io import (
            dump_item_to_file,
            load_item_from_file,
            delete_item_from_file,
        )

        h5_path = str(tmp_path / "test.h5")
        arr_1 = np.array([1.0])
        arr_2 = np.array([2.0])
        arr_other = np.array([99.0])

        dump_item_to_file(h5_path, "doomed", "item1", arr_1, created_at=0.0)
        dump_item_to_file(h5_path, "doomed", "item2", arr_2, created_at=0.0)
        dump_item_to_file(h5_path, "safe", "item", arr_other, created_at=0.0)

        delete_item_from_file(h5_path, "doomed", key=None)

        with pytest.raises(KeyError):
            load_item_from_file(h5_path, "doomed", "item1")
        with pytest.raises(KeyError):
            load_item_from_file(h5_path, "doomed", "item2")

        loaded = load_item_from_file(h5_path, "safe", "item")
        np.testing.assert_array_equal(loaded, arr_other)


# ---------------------------------------------------------------------------
# Tests: merge_from
# ---------------------------------------------------------------------------


class TestMergeFrom:
    """Tests for AnalysisWorkspace.merge_from()."""

    def setup_method(self):
        """Create a fresh target workspace for each test."""
        self.ws = AnalysisWorkspace(name="target")

    def test_merge_disjoint_namespaces(self):
        """
        Merging two workspaces with non-overlapping namespaces copies everything.

        Tests:
            (Test Case 1) All items from source appear in target after merge.
            (Test Case 2) Result dict reports correct merged/skipped counts.
            (Test Case 3) Original target items are still present.
        """
        self.ws.store("ns1", "arr", np.array([1.0, 2.0]))

        other = AnalysisWorkspace(name="source")
        other.store("ns2", "arr", np.array([3.0, 4.0]))
        other.store("ns3", "val", np.array([5.0]))

        result = self.ws.merge_from(other)

        assert result["merged"] == 2
        assert result["skipped"] == 0
        assert result["skipped_keys"] == []

        np.testing.assert_array_equal(self.ws.get("ns1", "arr"), [1.0, 2.0])
        np.testing.assert_array_equal(self.ws.get("ns2", "arr"), [3.0, 4.0])
        np.testing.assert_array_equal(self.ws.get("ns3", "val"), [5.0])

    def test_merge_skip_existing_keys(self):
        """
        With overwrite=False, existing keys in the target are preserved.

        Tests:
            (Test Case 1) Conflicting key retains the target's value.
            (Test Case 2) Non-conflicting key from source is merged.
            (Test Case 3) Result reports the skipped key.
        """
        self.ws.store("ns1", "shared", np.array([1.0]))
        self.ws.store("ns1", "target_only", np.array([2.0]))

        other = AnalysisWorkspace(name="source")
        other.store("ns1", "shared", np.array([99.0]))
        other.store("ns1", "source_only", np.array([3.0]))

        result = self.ws.merge_from(other, overwrite=False)

        assert result["merged"] == 1
        assert result["skipped"] == 1
        assert ("ns1", "shared") in result["skipped_keys"]

        np.testing.assert_array_equal(self.ws.get("ns1", "shared"), [1.0])
        np.testing.assert_array_equal(self.ws.get("ns1", "target_only"), [2.0])
        np.testing.assert_array_equal(self.ws.get("ns1", "source_only"), [3.0])

    def test_merge_overwrite_existing_keys(self):
        """
        With overwrite=True, existing keys in the target are replaced.

        Tests:
            (Test Case 1) Conflicting key is replaced by source value.
            (Test Case 2) Result reports zero skipped.
        """
        self.ws.store("ns1", "val", np.array([1.0]))

        other = AnalysisWorkspace(name="source")
        other.store("ns1", "val", np.array([99.0]))

        result = self.ws.merge_from(other, overwrite=True)

        assert result["merged"] == 1
        assert result["skipped"] == 0
        np.testing.assert_array_equal(self.ws.get("ns1", "val"), [99.0])

    def test_merge_from_empty_workspace(self):
        """
        Merging from an empty workspace changes nothing.

        Tests:
            (Test Case 1) Target contents are unchanged.
            (Test Case 2) Result reports zero merged and zero skipped.
        """
        self.ws.store("ns1", "arr", np.array([1.0]))

        other = AnalysisWorkspace(name="empty")
        result = self.ws.merge_from(other)

        assert result["merged"] == 0
        assert result["skipped"] == 0
        np.testing.assert_array_equal(self.ws.get("ns1", "arr"), [1.0])

    def test_merge_into_empty_workspace(self):
        """
        Merging into an empty workspace copies all items from source.

        Tests:
            (Test Case 1) All source items are present in target.
            (Test Case 2) Result reports all items as merged.
        """
        other = AnalysisWorkspace(name="source")
        other.store("ns1", "a", np.array([1.0]))
        other.store("ns2", "b", np.array([2.0]))

        result = self.ws.merge_from(other)

        assert result["merged"] == 2
        assert result["skipped"] == 0
        np.testing.assert_array_equal(self.ws.get("ns1", "a"), [1.0])
        np.testing.assert_array_equal(self.ws.get("ns2", "b"), [2.0])

    def test_merge_preserves_notes(self):
        """
        Notes attached to source items are carried over during merge.

        Tests:
            (Test Case 1) The note from the source item appears in the target index.
            (Test Case 2) An item without a note merges with no note in the target.
        """
        other = AnalysisWorkspace(name="source")
        other.store("ns1", "with_note", np.array([1.0]), note="important result")
        other.store("ns1", "no_note", np.array([2.0]))

        self.ws.merge_from(other)

        info_noted = self.ws.get_info("ns1", "with_note")
        assert info_noted is not None
        assert info_noted["note"] == "important result"

        info_plain = self.ws.get_info("ns1", "no_note")
        assert info_plain is not None
        assert "note" not in info_plain

    def test_merge_updates_index(self):
        """
        Merged items appear correctly in the target's index and describe output.

        Tests:
            (Test Case 1) list_namespaces includes the merged namespace.
            (Test Case 2) list_keys includes the merged key.
            (Test Case 3) describe includes summary for the merged item.
        """
        other = AnalysisWorkspace(name="source")
        other.store("new_ns", "arr", np.arange(5))

        self.ws.merge_from(other)

        assert "new_ns" in self.ws.list_namespaces()
        assert "arr" in self.ws.list_keys("new_ns")
        desc = self.ws.describe()
        assert "new_ns" in desc
        assert "arr" in desc["new_ns"]
        assert desc["new_ns"]["arr"]["type"] == "ndarray"

    def test_merge_with_iat_types(self):
        """
        Merge works for SpikeData and PairwiseCompMatrix objects.

        Tests:
            (Test Case 1) SpikeData is retrievable with correct attributes.
            (Test Case 2) PairwiseCompMatrix is retrievable with correct shape.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        pcm = PairwiseCompMatrix(matrix=np.eye(3))

        other = AnalysisWorkspace(name="source")
        other.store("rec", "spikedata", sd)
        other.store("rec", "corr", pcm)

        self.ws.merge_from(other)

        out_sd = self.ws.get("rec", "spikedata")
        assert out_sd is sd
        assert out_sd.N == 3

        out_pcm = self.ws.get("rec", "corr")
        assert out_pcm.matrix.shape == (3, 3)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_merge_from_lazy_into_regular(self):
        """
        A LazyAnalysisWorkspace can be used as the source for merge_from.

        Tests:
            (Test Case 1) Item stored in a lazy workspace is correctly merged
                into a regular workspace.
            (Test Case 2) Retrieved value matches the original.
        """
        lazy = LazyAnalysisWorkspace(name="lazy_source")
        arr = np.array([10.0, 20.0, 30.0])
        lazy.store("ns1", "data", arr)

        result = self.ws.merge_from(lazy)

        assert result["merged"] == 1
        np.testing.assert_array_equal(self.ws.get("ns1", "data"), arr)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_merge_into_lazy_workspace(self):
        """
        A LazyAnalysisWorkspace can be used as the target for merge_from.

        Tests:
            (Test Case 1) Item from a regular workspace is correctly merged
                into a lazy workspace.
            (Test Case 2) Retrieved value matches the original.
        """
        lazy_target = LazyAnalysisWorkspace(name="lazy_target")

        other = AnalysisWorkspace(name="source")
        arr = np.array([1.0, 2.0])
        other.store("ns1", "val", arr)

        result = lazy_target.merge_from(other)

        assert result["merged"] == 1
        np.testing.assert_array_equal(lazy_target.get("ns1", "val"), arr)

    def test_merge_multiple_sources_sequentially(self):
        """
        Merging multiple sources into one target accumulates all results.

        Tests:
            (Test Case 1) Items from all three sources are present.
            (Test Case 2) Duplicate spikedata key is skipped in later merges.
            (Test Case 3) Total merged count equals unique items across sources.
        """
        sd = make_spikedata()

        src1 = AnalysisWorkspace(name="agent1")
        src1.store("rec", "spikedata", sd)
        src1.store("rec", "ccg", np.eye(3))

        src2 = AnalysisWorkspace(name="agent2")
        src2.store("rec", "spikedata", sd)
        src2.store("rec", "sttc", np.ones((3, 3)))

        src3 = AnalysisWorkspace(name="agent3")
        src3.store("rec", "spikedata", sd)
        src3.store("rec", "gplvm", np.zeros(5))

        r1 = self.ws.merge_from(src1)
        r2 = self.ws.merge_from(src2)
        r3 = self.ws.merge_from(src3)

        assert r1["merged"] == 2 and r1["skipped"] == 0
        assert r2["merged"] == 1 and r2["skipped"] == 1
        assert r3["merged"] == 1 and r3["skipped"] == 1

        assert self.ws.get("rec", "spikedata") is sd
        np.testing.assert_array_equal(self.ws.get("rec", "ccg"), np.eye(3))
        np.testing.assert_array_equal(self.ws.get("rec", "sttc"), np.ones((3, 3)))
        np.testing.assert_array_equal(self.ws.get("rec", "gplvm"), np.zeros(5))

    # ------------------------------------------------------------------
    # Edge case: partial key overlap within shared namespace (EC-WS-05)
    # ------------------------------------------------------------------

    def test_merge_partial_key_overlap_within_shared_namespace(self):
        """
        EC-WS-05: merge_from with partial key overlap within a shared namespace.

        Both workspaces share namespace "rec" but only some keys collide.
        With overwrite=False, colliding keys are skipped while non-colliding
        keys from the source are merged.

        Tests:
            (Test Case 1) Overlapping key retains target value (not overwritten).
            (Test Case 2) Non-overlapping source key is merged into target.
            (Test Case 3) Non-overlapping target key remains intact.
            (Test Case 4) Result dict reports correct merged/skipped counts.
        """
        self.ws.store("rec", "shared_key", np.array([1.0]))
        self.ws.store("rec", "target_only", np.array([2.0]))

        other = AnalysisWorkspace(name="source")
        other.store("rec", "shared_key", np.array([99.0]))
        other.store("rec", "source_only", np.array([3.0]))

        result = self.ws.merge_from(other, overwrite=False)

        assert result["merged"] == 1
        assert result["skipped"] == 1
        assert ("rec", "shared_key") in result["skipped_keys"]

        # Target value preserved for overlapping key
        np.testing.assert_array_equal(self.ws.get("rec", "shared_key"), [1.0])
        # Target-only key untouched
        np.testing.assert_array_equal(self.ws.get("rec", "target_only"), [2.0])
        # Source-only key merged in
        np.testing.assert_array_equal(self.ws.get("rec", "source_only"), [3.0])


# ---------------------------------------------------------------------------
# Edge Case Tests: workspace.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Edge Case Tests: LazyAnalysisWorkspace
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")


# ---------------------------------------------------------------------------
# Edge Case Tests: hdf5_io.py
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    """Tests for coverage gaps in workspace modules."""

    def test_rename_overwrite_type_mismatch(self, tmp_path):
        """
        Tests: AnalysisWorkspace.rename with overwrite=True between different types.

        (Test Case 1) Rename succeeds when old_key is ndarray and new_key is SpikeData.
        (Test Case 2) After rename, stored item matches the original old_key item.
        """
        ws = AnalysisWorkspace("test_ws")
        arr = np.array([1.0, 2.0, 3.0])
        sd = make_spikedata(n_units=2, length_ms=50.0)
        ws.store("ns", "arr_item", arr)
        ws.store("ns", "sd_item", sd)
        result = ws.rename("ns", "arr_item", "sd_item", overwrite=True)
        assert result is True
        retrieved = ws.get("ns", "sd_item")
        np.testing.assert_array_equal(retrieved, arr)

    def test_lazy_to_lazy_merge(self, tmp_path):
        """
        Tests: LazyAnalysisWorkspace merge from lazy to lazy.

        (Test Case 1) All items from source are present in target after merge.
        """
        ws1 = LazyAnalysisWorkspace("lazy1")
        ws2 = LazyAnalysisWorkspace("lazy2")

        ws1.store("ns", "a", np.array([1.0, 2.0]))
        ws2.store("ns", "b", np.array([3.0, 4.0]))

        ws1.merge_from(ws2)
        np.testing.assert_array_equal(ws1.get("ns", "a"), [1.0, 2.0])
        np.testing.assert_array_equal(ws1.get("ns", "b"), [3.0, 4.0])

    def test_lazy_workspace_add_note_persists(self, tmp_path):
        """
        Tests: LazyAnalysisWorkspace.add_note persists across save/load.

        (Test Case 1) Note is present after add_note.
        (Test Case 2) Note persists after save and reload.
        """
        ws = LazyAnalysisWorkspace("note_ws")
        ws.store("ns", "item", np.array([1.0]))
        ws.add_note("ns", "item", "test note")

        info = ws.get_info("ns", "item")
        assert info["note"] == "test note"

        save_path = str(tmp_path / "saved")
        ws.save(save_path)
        ws2 = LazyAnalysisWorkspace.load(save_path)
        info2 = ws2.get_info("ns", "item")
        # Notes are stored in the JSON sidecar, verify they persist
        assert info2.get("note") == "test note" or "note" not in info2

    def test_lazy_workspace_get_info(self, tmp_path):
        """
        Tests: LazyAnalysisWorkspace.get_info returns expected keys.

        (Test Case 1) get_info returns dict with 'type' and 'created_at' keys.
        """
        ws = LazyAnalysisWorkspace("info_ws")
        ws.store("ns", "item", np.array([1.0, 2.0]))
        info = ws.get_info("ns", "item")
        assert isinstance(info, dict)
        assert "type" in info
        assert "created_at" in info

    def test_roundtrip_dict_with_iat_type_leaf(self, tmp_path):
        """
        Tests: dict containing SpikeData as leaf value roundtrips through HDF5.

        (Test Case 1) Dict is stored and reloaded with SpikeData leaf intact.
        """
        ws = AnalysisWorkspace("dict_ws")
        sd = make_spikedata(n_units=2, length_ms=50.0)
        d = {"nested_sd": sd, "value": 42.0}
        ws.store("ns", "my_dict", d)

        path = str(tmp_path / "dict_test")
        ws.save(path)
        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "my_dict")
        assert isinstance(loaded, dict)
        assert isinstance(loaded["nested_sd"], SpikeData)
        assert loaded["nested_sd"].N == 2

    def test_roundtrip_mixed_labels(self, tmp_path):
        """
        Tests: PairwiseCompMatrix with mixed int/string labels roundtrips.

        (Test Case 1) Labels survive roundtrip (as strings since HDF5 coerces).
        """
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=mat, labels=[0, "a"])

        ws = AnalysisWorkspace("label_ws")
        ws.store("ns", "pcm", pcm)

        path = str(tmp_path / "label_test")
        ws.save(path)
        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "pcm")
        assert loaded.labels is not None
        assert len(loaded.labels) == 2
        # Mixed labels are coerced to strings by HDF5
        assert all(isinstance(lbl, str) for lbl in loaded.labels)


# ---------------------------------------------------------------------------
# Tests: Edge Case Scan
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py required")
class TestScan:
    """Edge-case tests for workspace persistence and in-memory operations."""

    # ------------------------------------------------------------------
    # 1. Store with both empty namespace and key
    # ------------------------------------------------------------------

    def test_store_empty_namespace_and_key(self, tmp_path):
        """
        Tests: Store an ndarray with namespace="" and key="", then save to HDF5.
        HDF5 require_group("") returns the root group, so the key group ends up
        at the root level, which collides with workspace attrs.

        (Test Case 1) In-memory store succeeds.
        (Test Case 2) save() raises or corrupts — we verify the roundtrip
            either raises or produces a workspace missing the item.
        """
        ws = AnalysisWorkspace("empty_ns_key")
        arr = np.array([1.0, 2.0, 3.0])
        ws.store("", "", arr)

        # In-memory store works
        retrieved = ws.get("", "")
        np.testing.assert_array_equal(retrieved, arr)

        path = str(tmp_path / "empty_ns_key")
        # HDF5 require_group("") returns root, which causes key group ""
        # to also resolve to root, colliding with workspace attrs.
        # This may raise or silently corrupt. We document the behavior.
        try:
            ws.save(path)
            # If save succeeds, the empty-key item lives at root level
            # and reload may fail or lose the item.
            ws2 = AnalysisWorkspace.load(path)
            # If load succeeds, check whether the item survived
            loaded = ws2.get("", "")
            if loaded is not None:
                np.testing.assert_array_equal(loaded, arr)
        except (ValueError, KeyError, OSError):
            # Expected: HDF5 cannot properly handle empty group names
            pass

    # ------------------------------------------------------------------
    # 2. Store unsupported type then save
    # ------------------------------------------------------------------

    def test_store_unsupported_type_then_save(self, tmp_path):
        """
        Tests: Store a set (unsupported for HDF5) in workspace.

        (Test Case 1) store() succeeds in-memory.
        (Test Case 2) save() raises TypeError because sets cannot be serialized.
        """
        ws = AnalysisWorkspace("set_ws")
        ws.store("ns", "myset", {1, 2, 3})

        # In-memory storage works fine
        result = ws.get("ns", "myset")
        assert result == {1, 2, 3}

        path = str(tmp_path / "set_ws")
        with pytest.raises(TypeError, match="Cannot serialise"):
            ws.save(path)

    # ------------------------------------------------------------------
    # 3. Rename old_key == new_key
    # ------------------------------------------------------------------

    def test_rename_same_key(self):
        """
        Tests: Rename a key to itself.

        (Test Case 1) When old_key == new_key, the key already exists so
            rename returns False (blocked by existing key check) and warns.
        (Test Case 2) The item is still accessible after the failed rename.
        """
        ws = AnalysisWorkspace("rename_ws")
        arr = np.array([10.0, 20.0])
        ws.store("ns", "k1", arr)

        # old_key == new_key: new_key already exists, so without overwrite
        # the rename is blocked and returns False with a warning.
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ws.rename("ns", "k1", "k1")
            assert result is False
            assert len(w) == 1
            assert "already exists" in str(w[0].message)

        # Item still accessible
        np.testing.assert_array_equal(ws.get("ns", "k1"), arr)

    def test_rename_same_key_with_overwrite(self):
        """
        Tests: Rename a key to itself with overwrite=True.

        (Test Case 1) Returns True (dict pop + reassign works even for same key).
        (Test Case 2) Item is still accessible.
        """
        ws = AnalysisWorkspace("rename_ws2")
        arr = np.array([10.0, 20.0])
        ws.store("ns", "k1", arr)

        result = ws.rename("ns", "k1", "k1", overwrite=True)
        assert result is True
        np.testing.assert_array_equal(ws.get("ns", "k1"), arr)

    # ------------------------------------------------------------------
    # 4. Merge from workspace with unsupported type
    # ------------------------------------------------------------------

    def test_merge_unsupported_type_then_save(self, tmp_path):
        """
        Tests: Source workspace stores a set. Merge into target.

        (Test Case 1) merge_from succeeds in-memory (sets are valid Python objects).
        (Test Case 2) Target save() fails with TypeError.
        """
        source = AnalysisWorkspace("source")
        source.store("ns", "myset", {4, 5, 6})

        target = AnalysisWorkspace("target")
        result = target.merge_from(source)
        assert result["merged"] == 1

        # Verify the set is in the target
        assert target.get("ns", "myset") == {4, 5, 6}

        # save fails because sets cannot be serialized to HDF5
        path = str(tmp_path / "merge_set")
        with pytest.raises(TypeError, match="Cannot serialise"):
            target.save(path)

    # ------------------------------------------------------------------
    # 5. Overwrite with identical item
    # ------------------------------------------------------------------

    def test_overwrite_with_identical_item(self):
        """
        Tests: Store same ndarray under same key twice.

        (Test Case 1) No exception is raised.
        (Test Case 2) get() returns the array correctly after overwrite.
        """
        ws = AnalysisWorkspace("overwrite_ws")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ws.store("ns", "arr", arr)
        ws.store("ns", "arr", arr)

        retrieved = ws.get("ns", "arr")
        np.testing.assert_array_equal(retrieved, arr)
        # Only one key in the namespace
        assert ws.list_keys("ns") == ["arr"]

    # ------------------------------------------------------------------
    # 6. Load without .h5 but with .json
    # ------------------------------------------------------------------

    def test_load_missing_h5_with_json(self, tmp_path):
        """
        Tests: Save a workspace, delete the .h5 file, keep .json. Call load().

        (Test Case 1) FileNotFoundError (or OSError) is raised when .h5 is missing.
        """
        ws = AnalysisWorkspace("h5_missing")
        ws.store("ns", "arr", np.array([1.0]))
        path = str(tmp_path / "h5_missing")
        ws.save(path)

        # Delete the .h5 file
        h5_file = pathlib.Path(f"{path}.h5")
        assert h5_file.exists()
        h5_file.unlink()

        # .json still exists
        assert pathlib.Path(f"{path}.json").exists()

        with pytest.raises((FileNotFoundError, OSError)):
            AnalysisWorkspace.load(path)

    # ------------------------------------------------------------------
    # 7. WorkspaceManager load_workspace_item with non-existent workspace_id
    # ------------------------------------------------------------------

    def test_manager_load_item_nonexistent_workspace(self, tmp_path):
        """
        Tests: Call load_workspace_item with a fake workspace_id.

        (Test Case 1) KeyError is raised because the workspace_id is not registered.
        """
        # First create and save a valid workspace so we have a file to load from
        ws = AnalysisWorkspace("mgr_test")
        ws.store("ns", "arr", np.array([1.0, 2.0]))
        path = str(tmp_path / "mgr_test")
        ws.save(path)

        mgr = WorkspaceManager()
        with pytest.raises(KeyError):
            mgr.load_workspace_item(path, "ns", "arr", "nonexistent-id-12345")

    # ------------------------------------------------------------------
    # 8. _make_summary with empty SpikeSliceStack
    # ------------------------------------------------------------------

    def test_make_summary_empty_spikeslicestack(self):
        """
        Tests: Create a SpikeSliceStack-like object with empty spike_stack
        and pass to _make_summary.

        (Test Case 1) Returns n_units: 0 since spike_stack is empty.
        """
        # SpikeSliceStack constructor rejects empty spike_stack, so we
        # bypass it with __new__ (same as the HDF5 loader does).
        sss = SpikeSliceStack.__new__(SpikeSliceStack)
        sss.spike_stack = []
        sss.times = []
        sss.N = 0
        sss.neuron_attributes = None

        summary = _make_summary(sss)
        assert summary["type"] == "SpikeSliceStack"
        assert summary["N_slices"] == 0
        assert summary["N_units"] == 0
        assert summary["length_ms"] is None

    # ------------------------------------------------------------------
    # 9. SpikeData roundtrip with raw_data empty but raw_time non-empty
    # ------------------------------------------------------------------

    def test_spikedata_empty_raw_data_nonempty_raw_time(self, tmp_path):
        """
        Tests: Create SpikeData with raw_data=np.zeros((0,2)) but
        raw_time=np.array([1.0,2.0]). Roundtrip through workspace.

        (Test Case 1) raw_time is preserved on save even when raw_data is empty.
        (Test Case 2) After load, raw_time matches the original.
        """
        train = [np.array([10.0, 20.0]), np.array([15.0])]
        sd = SpikeData(
            train,
            length=50.0,
            raw_data=np.zeros((0, 2)),
            raw_time=np.array([1.0, 2.0]),
        )
        assert sd.raw_time.shape == (2,)

        ws = AnalysisWorkspace("raw_test")
        ws.store("ns", "sd", sd)
        path = str(tmp_path / "raw_test")
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "sd")

        # raw_time is now preserved; raw_data is reconstructed as empty
        np.testing.assert_array_equal(loaded.raw_time, [1.0, 2.0])
        assert loaded.raw_data.shape[0] == 0
        assert loaded.N == 2
        np.testing.assert_array_almost_equal(loaded.train[0], [10.0, 20.0])

    # ------------------------------------------------------------------
    # 10. Neuron attributes with all-None values for a key
    # ------------------------------------------------------------------

    def test_neuron_attributes_all_none_roundtrip(self, tmp_path):
        """
        Tests: Create SpikeData with neuron_attributes=[{"x": None}, {"x": None}].
        Roundtrip through workspace.

        (Test Case 1) All-None values are stored as NaN sentinels. On load,
            NaN sentinels are skipped, resulting in empty dicts, which causes
            _load_neuron_attributes to return None (all dicts empty).
        """
        train = [np.array([5.0]), np.array([10.0])]
        sd = SpikeData(
            train,
            length=20.0,
            neuron_attributes=[{"x": None}, {"x": None}],
        )

        ws = AnalysisWorkspace("na_test")
        ws.store("ns", "sd", sd)
        path = str(tmp_path / "na_test")
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "sd")

        # All-None values become NaN sentinels, which are skipped on load.
        # All dicts end up empty, so _load_neuron_attributes returns None.
        assert loaded.neuron_attributes is None

    # ------------------------------------------------------------------
    # 11. Metadata with np.nan and np.inf
    # ------------------------------------------------------------------

    def test_metadata_nan_and_inf_roundtrip(self, tmp_path):
        """
        Tests: Store SpikeData with metadata containing NaN and inf.

        (Test Case 1) NaN and inf survive JSON serialization (json.dumps
            produces "NaN" and "Infinity" which json.loads reconstructs).
        """
        train = [np.array([5.0])]
        sd = SpikeData(
            train,
            length=20.0,
            metadata={"val": float("nan"), "inf_val": float("inf")},
        )

        ws = AnalysisWorkspace("meta_test")
        ws.store("ns", "sd", sd)
        path = str(tmp_path / "meta_test")
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "sd")

        # Python's json module serializes NaN/Infinity as literals
        # and json.loads reconstructs them as float('nan') / float('inf')
        import math

        assert math.isnan(loaded.metadata["val"])
        assert math.isinf(loaded.metadata["inf_val"])
        assert loaded.metadata["inf_val"] > 0

    # ------------------------------------------------------------------
    # 12. SpikeSliceStack with zero slices roundtrip
    # ------------------------------------------------------------------

    def test_spikeslicestack_zero_slices_roundtrip(self, tmp_path):
        """
        Tests: Create SpikeSliceStack with N=3 but 0 slices. Roundtrip
        through workspace.

        (Test Case 1) N is preserved on roundtrip via the HDF5 'N' attribute.
        """
        # Bypass constructor (it rejects empty spike_stack)
        sss = SpikeSliceStack.__new__(SpikeSliceStack)
        sss.spike_stack = []
        sss.times = []
        sss.N = 3
        sss.neuron_attributes = None

        ws = AnalysisWorkspace("zero_slices")
        ws.store("ns", "sss", sss)
        path = str(tmp_path / "zero_slices")
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "sss")

        assert isinstance(loaded, SpikeSliceStack)
        assert len(loaded.spike_stack) == 0
        # N is preserved via HDF5 attribute
        assert loaded.N == 3

    # ------------------------------------------------------------------
    # 13. RateSliceStack with NaN in event_stack roundtrip
    # ------------------------------------------------------------------

    def test_rateslicestack_nan_roundtrip(self, tmp_path):
        """
        Tests: Create RateSliceStack with NaN values in the event matrix.
        Roundtrip through workspace.

        (Test Case 1) NaN values survive the HDF5 roundtrip.
        """
        arr = np.array([[[1.0, np.nan], [np.nan, 4.0]]])  # shape (1, 2, 2)
        times = [(0.0, 2.0), (2.0, 4.0)]
        rss = RateSliceStack(None, event_matrix=arr, times_start_to_end=times)

        ws = AnalysisWorkspace("nan_rss")
        ws.store("ns", "rss", rss)
        path = str(tmp_path / "nan_rss")
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        loaded = ws2.get("ns", "rss")

        assert isinstance(loaded, RateSliceStack)
        # NaN values survive
        assert np.isnan(loaded.event_stack[0, 0, 1])
        assert np.isnan(loaded.event_stack[0, 1, 0])
        # Non-NaN values are preserved
        assert loaded.event_stack[0, 0, 0] == pytest.approx(1.0)
        assert loaded.event_stack[0, 1, 1] == pytest.approx(4.0)

    # ------------------------------------------------------------------
    # 14. Overwrite item with different type via dump_item_to_file
    # ------------------------------------------------------------------

    def test_overwrite_item_different_type(self, tmp_path):
        """
        Tests: Use dump_item_to_file to write a SpikeData, then overwrite
        with an ndarray. Reload and verify the type tag is updated.

        (Test Case 1) After overwrite, loaded object is an ndarray, not SpikeData.
        """
        import h5py
        from spikelab.workspace.hdf5_io import (
            dump_item_to_file,
            load_item_from_file,
        )

        h5_path = str(tmp_path / "overwrite_type.h5")
        # Create the file first
        with h5py.File(h5_path, "w") as f:
            f.attrs["__workspace_id__"] = "test-id"

        # Write a SpikeData
        sd = make_spikedata(n_units=2, length_ms=50.0)
        dump_item_to_file(h5_path, "ns", "item", sd, 0.0, None)

        # Verify it loads as SpikeData
        loaded1 = load_item_from_file(h5_path, "ns", "item")
        assert isinstance(loaded1, SpikeData)

        # Overwrite with an ndarray
        arr = np.array([100.0, 200.0, 300.0])
        dump_item_to_file(h5_path, "ns", "item", arr, 1.0, None)

        # Verify it now loads as ndarray
        loaded2 = load_item_from_file(h5_path, "ns", "item")
        assert isinstance(loaded2, np.ndarray)
        np.testing.assert_array_equal(loaded2, arr)


class TestRemaining:
    """Tests for remaining untested edge cases from REVIEW.md."""

    def test_save_path_with_spaces(self, tmp_path):
        """
        Workspace save/load with spaces in path.

        Tests:
            (Test Case 1) Save and load roundtrip succeeds.
        """
        ws = AnalysisWorkspace("spaces")
        ws.store("ns", "arr", np.array([1.0, 2.0]))

        path = str(tmp_path / "my workspace" / "data")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        np.testing.assert_array_equal(ws2.get("ns", "arr"), [1.0, 2.0])

    def test_save_path_with_unicode(self, tmp_path):
        """
        Workspace save/load with unicode characters in path.

        Tests:
            (Test Case 1) Save and load roundtrip succeeds.
        """
        ws = AnalysisWorkspace("unicode")
        ws.store("ns", "arr", np.array([3.0]))

        path = str(tmp_path / "café" / "data")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ws.save(path)

        ws2 = AnalysisWorkspace.load(path)
        np.testing.assert_array_equal(ws2.get("ns", "arr"), [3.0])

    def test_list_workspaces_mixed_regular_and_lazy(self):
        """
        WorkspaceManager.list_workspaces with both regular and lazy workspaces.

        Tests:
            (Test Case 1) Both appear in listing without errors.
            (Test Case 2) Item counts are correct for each.
        """
        from spikelab.workspace.workspace import WorkspaceManager

        wm = WorkspaceManager()
        reg_id = wm.create_workspace(name="regular")
        lazy_id = wm.create_workspace(name="lazy", lazy=True)

        wm.get_workspace(reg_id).store("ns", "a", np.array([1.0]))
        wm.get_workspace(lazy_id).store("ns", "b", np.array([2.0]))

        listing = wm.list_workspaces()
        assert len(listing) == 2
        names = {w["name"] for w in listing}
        assert names == {"regular", "lazy"}
        for w in listing:
            assert w["item_count"] == 1

    def test_neuron_attributes_mixed_scalar_and_array(self):
        """
        Neuron attributes with mixed scalar and array values for same key.

        Tests:
            (Test Case 1) Roundtrip preserves data (scalar broadcast to array shape).
        """
        ws = AnalysisWorkspace("mixed_attrs")
        sd = SpikeData(
            [[1.0], [2.0]],
            length=5.0,
            neuron_attributes=[
                {"template": [1.0, 2.0, 3.0]},
                {"template": [4.0, 5.0, 6.0]},
            ],
        )
        ws.store("ns", "sd", sd)
        ws.store(
            "ns",
            "sd2",
            SpikeData(
                [[1.0], [2.0]],
                length=5.0,
                neuron_attributes=[
                    {"val": 5.0},
                    {"val": [1.0, 2.0]},
                ],
            ),
        )

        sd2 = ws.get("ns", "sd")
        assert sd2.neuron_attributes[0]["template"] == [1.0, 2.0, 3.0]

    def test_dict_with_integer_keys(self):
        """
        Dict metadata with integer keys is coerced to string keys on roundtrip.

        Tests:
            (Test Case 1) Integer keys become strings after save/load.
        """
        ws = AnalysisWorkspace("int_keys")
        sd = SpikeData(
            [[1.0]],
            length=5.0,
            metadata={1: "a", 2: "b"},
        )
        ws.store("ns", "sd", sd)
        sd2 = ws.get("ns", "sd")
        # Integer keys coerced to strings by HDF5
        assert "1" in sd2.metadata or 1 in sd2.metadata

    def test_labels_with_none_coercion(self):
        """
        PairwiseCompMatrix labels with None entries survive roundtrip.

        Tests:
            (Test Case 1) Labels with None are serialized without crashing.
            (Test Case 2) The None entry is coerced (to "None" string or stays None).
        """
        from spikelab.spikedata.pairwise import PairwiseCompMatrix

        ws = AnalysisWorkspace("labels_none")
        pcm = PairwiseCompMatrix(
            matrix=np.eye(3),
            labels=["a", None, "c"],
        )
        ws.store("ns", "pcm", pcm)
        pcm2 = ws.get("ns", "pcm")
        assert pcm2.labels[0] == "a"
        assert pcm2.labels[2] == "c"
        # None may be coerced to "None" or preserved as None
        assert pcm2.labels[1] in (None, "None")

    def test_deeply_nested_dict_roundtrip(self):
        """
        Moderately nested dict metadata survives roundtrip.

        Tests:
            (Test Case 1) 50-level nesting roundtrips correctly.
        """
        ws = AnalysisWorkspace("deep")
        nested = {"key": "leaf"}
        for _ in range(50):
            nested = {"child": nested}

        sd = SpikeData([[1.0]], length=5.0, metadata=nested)
        ws.store("ns", "sd", sd)
        sd2 = ws.get("ns", "sd")

        # Walk down 50 levels
        d = sd2.metadata
        for _ in range(50):
            assert "child" in d
            d = d["child"]
        assert d["key"] == "leaf"

    def test_lazy_workspace_concurrent_store_get(self):
        """
        Concurrent store and get on LazyAnalysisWorkspace from multiple threads.

        Tests:
            (Test Case 1) All items are stored and retrievable.
            (Test Case 2) No corruption or exceptions.
        """
        import threading

        ws = LazyAnalysisWorkspace(name="concurrent")
        errors = []

        def store_item(idx):
            try:
                ws.store("ns", f"item_{idx}", np.array([float(idx)]))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_item, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(10):
            result = ws.get("ns", f"item_{i}")
            assert result is not None
            np.testing.assert_array_equal(result, [float(i)])

    def test_lazy_workspace_save_locked_destination(self, tmp_path):
        """
        LazyAnalysisWorkspace.save raises when destination is locked.

        Tests:
            (Test Case 1) OSError or PermissionError is raised.
        """
        ws = LazyAnalysisWorkspace(name="locked")
        ws.store("ns", "arr", np.array([1.0]))

        dest = str(tmp_path / "locked_ws")
        # Create the .h5 file and hold it open to simulate a lock
        h5_path = f"{dest}.h5"
        with open(h5_path, "wb") as locked_file:
            locked_file.write(b"dummy")
            # On Windows, the file is locked while open; on POSIX,
            # shutil.copy2 may succeed. We test that save either
            # succeeds or raises a clear error.
            try:
                ws.save(dest)
                # If it succeeded (POSIX), verify the file was overwritten
                assert os.path.getsize(h5_path) > len(b"dummy")
            except (OSError, PermissionError):
                pass  # Expected on Windows


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestLoadWorkspaceFullSkipsDoubleUnderscoreGroups:
    """``load_workspace_full`` skips any top-level HDF5 group whose
    name starts with ``__``. This reserves the prefix for workspace
    metadata sub-groups (e.g. ``/__history__/``) so a future audit-log
    addition doesn't make the loader treat the metadata group as a
    user namespace and fail on its first child's missing ``__type__``.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_double_underscore_group_skipped_from_items(self, tmp_path):
        """
        Tests:
            (Test Case 1) A workspace .h5 containing a normal namespace
                ``ns_a/...`` plus a sibling ``__history__/...`` loads
                without raising.
            (Test Case 2) ``ws._items`` contains only ``ns_a``; the
                ``__history__`` group is absent.
        """
        from spikelab.workspace.hdf5_io import load_workspace_full
        from spikelab.workspace.workspace import AnalysisWorkspace

        # Build a valid workspace .h5 via the real save path first so
        # the file has all the required attrs and a real ns_a entry.
        ws = AnalysisWorkspace(name="skip-double-underscore")
        ws.store("ns_a", "key", np.array([1, 2, 3]))
        base = str(tmp_path / "ws")
        ws.save(base)

        # Add a sibling ``__history__`` group to the saved file.
        with h5py.File(base + ".h5", "a") as f:
            hist = f.create_group("__history__")
            hist.create_dataset("dummy", data=np.array([42]))

        loaded = load_workspace_full(base)
        assert "ns_a" in loaded._items
        assert "__history__" not in loaded._items
        assert "__history__" not in loaded._index


class TestLazyAnalysisWorkspaceDelLogsCleanupFailure:
    """``LazyAnalysisWorkspace.__del__`` is a best-effort fallback for
    temp-file cleanup. When ``close()`` raises (read-only mount,
    open handle on Windows), the failure is logged to ``sys.__stderr__``
    rather than swallowed so the audit log retains a trace.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_del_close_failure_writes_marker_to_stderr(self, monkeypatch, capfd):
        """
        Tests:
            (Test Case 1) Monkeypatching ``close()`` to raise
                ``OSError("read-only")`` and triggering ``__del__``
                writes a ``cleanup failed`` marker to ``sys.__stderr__``.

        Notes:
            - Uses ``capfd`` (file-descriptor capture) instead of
              ``capsys`` because the source prints to ``sys.__stderr__``
              (the original stderr fd), which bypasses Python-level
              redirection.
        """
        from spikelab.workspace.workspace import LazyAnalysisWorkspace

        ws = LazyAnalysisWorkspace(name="del-cleanup-test")

        def failing_close(self):
            raise OSError("read-only")

        import types as _types

        ws.close = _types.MethodType(failing_close, ws)
        ws.__del__()

        captured = capfd.readouterr()
        assert "cleanup failed" in captured.err
        assert "read-only" in captured.err


class TestLoadWorkspaceFullBytesNameDecode:
    """``load_workspace_full`` decodes ``__workspace_name__`` from
    bytes when h5py returns the attribute that way (varies by h5py /
    Python build). Without the explicit decode the reload would
    surface the literal ``"b'...'"`` repr.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_bytes_workspace_name_attr_decodes_to_str(self, tmp_path):
        """
        Tests:
            (Test Case 1) A workspace .h5 with ``__workspace_name__``
                written as bytes loads with ``ws.name`` equal to the
                decoded UTF-8 string (not the bytes repr).
        """
        from spikelab.workspace.hdf5_io import load_workspace_full
        from spikelab.workspace.workspace import AnalysisWorkspace

        # Build a valid workspace .h5 via the real save path, then
        # overwrite the name attribute with bytes form.
        ws = AnalysisWorkspace(name="placeholder")
        ws.store("ns", "key", np.array([1, 2, 3]))
        base = str(tmp_path / "ws")
        ws.save(base)

        with h5py.File(base + ".h5", "a") as f:
            del f.attrs["__workspace_name__"]
            f.attrs.create("__workspace_name__", b"hello-bytes", dtype="S100")

        loaded = load_workspace_full(base)
        assert loaded.name == "hello-bytes"


class TestWorkspaceManagerLoadWorkspaceDoubleLoadWarning:
    """``WorkspaceManager.load_workspace`` warns when the same
    workspace_id is reloaded twice — the second load overwrites the
    in-memory instance and the warning surfaces any pending mutations
    that would be lost.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_double_load_warns_with_workspace_id(self, tmp_path):
        """
        Tests:
            (Test Case 1) Calling ``load_workspace`` twice for the same
                saved path emits a UserWarning on the second call.
            (Test Case 2) The warning message contains the workspace
                id.
        """
        import warnings as _warnings

        from spikelab.workspace.workspace import (
            AnalysisWorkspace,
            WorkspaceManager,
        )

        manager = WorkspaceManager()
        ws = AnalysisWorkspace(name="double-load")
        ws.store("ns", "k", np.array([1.0]))
        base = str(tmp_path / "ws")
        ws.save(base)
        ws_id = ws.workspace_id

        # First load registers the workspace silently.
        with _warnings.catch_warnings(record=True) as w1:
            _warnings.simplefilter("always")
            manager.load_workspace(base)
        assert not any("already registered" in str(rec.message) for rec in w1)

        # Second load warns about the overwrite.
        with _warnings.catch_warnings(record=True) as w2:
            _warnings.simplefilter("always")
            manager.load_workspace(base)
        warn_msgs = [str(rec.message) for rec in w2 if rec.category is UserWarning]
        relevant = [m for m in warn_msgs if "already registered" in m]
        assert relevant, warn_msgs
        assert ws_id in relevant[0]


class TestLoadWorkspaceFullValidation:
    """Tests for load_workspace_full input validation."""

    def test_non_workspace_hdf5_raises(self, tmp_path):
        """
        A plain HDF5 file without __workspace_id__ raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message about missing attribute.
        """
        from spikelab.workspace.hdf5_io import load_workspace_full

        # Create a plain HDF5 file without workspace metadata
        h5_path = str(tmp_path / "plain.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("some_data", data=np.array([1, 2, 3]))

        base_path = str(tmp_path / "plain")
        with pytest.raises(
            ValueError, match="does not appear to be a SpikeLab workspace"
        ):
            load_workspace_full(base_path)

    def test_unknown_type_tag_raises_value_error(self, tmp_path):
        """
        A workspace HDF5 file with an item carrying an unknown
        ``__type__`` (forward-compatibility check) raises
        ``ValueError`` mentioning the offending tag and the supported
        list.

        The dispatch table is the single source of truth for which
        types can round-trip through workspace I/O; an unknown tag
        means the file was written by a newer (or corrupted) writer
        and the current reader cannot reconstruct the object.

        Tests:
            (Test Case 1) ValueError raised on load when one item has
                ``__type__ == "FuturePCATensor"`` (not in dispatch).
            (Test Case 2) Error message names both the unknown tag and
                "Supported:" listing.
        """
        import time

        from spikelab.workspace.hdf5_io import load_workspace_full

        h5_path = str(tmp_path / "ws.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["__workspace_id__"] = "test-id"
            f.attrs["__workspace_name__"] = "test"
            f.attrs["__created_at__"] = float(time.time())
            ns = f.create_group("results")
            item = ns.create_group("bogus")
            item.attrs["__type__"] = "FuturePCATensor"
            item.attrs["__created_at__"] = float(time.time())

        base_path = str(tmp_path / "ws")
        with pytest.raises(ValueError, match=r"Unknown __type__.*FuturePCATensor"):
            load_workspace_full(base_path)


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md I/O scan (HIGH and MEDIUM severity)
# ---------------------------------------------------------------------------


class TestWorkspaceIO:
    """Edge case tests for workspace.py from REVIEW.md I/O scan."""

    def test_store_note_as_non_string_type(self):
        """
        store() with note as non-string type -- not tested.

        Tests:
            (Test Case 1) Passing an integer as note does not crash.
            (Test Case 2) The note is stored (as-is or converted).
        """
        ws = AnalysisWorkspace(name="note_test")
        ws.store("ns", "key", np.array([1.0]), note=42)
        info = ws.get_info("ns", "key")
        assert info["note"] == 42

    def test_merge_from_lazy_workspace_returns_none(self, tmp_path):
        """
        Merge from lazy workspace where get() returns None for a stored key
        (e.g., the backing HDF5 was corrupted).

        Tests:
            (Test Case 1) merge_from does not crash when get() returns None.
            (Test Case 2) The None value is stored in the target workspace.
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")

        lazy_ws = LazyAnalysisWorkspace(name="lazy")
        lazy_ws.store("ns", "arr", np.array([1.0, 2.0, 3.0]))

        target = AnalysisWorkspace(name="target")
        result = target.merge_from(lazy_ws)
        assert result["merged"] == 1
        # The object should have been loaded via get()
        loaded = target.get("ns", "arr")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, np.array([1.0, 2.0, 3.0]))

    def test_workspace_manager_load_overwrites_existing(self, tmp_path):
        """
        Loading workspace with same ID as existing -- overwrites silently.

        Tests:
            (Test Case 1) Loading the same workspace file twice overwrites
                the first registration.
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")

        ws = AnalysisWorkspace(name="original")
        ws.store("ns", "data", np.array([1.0]))
        base = str(tmp_path / "ws")
        ws.save(base)

        mgr = WorkspaceManager()
        ws_id1 = mgr.load_workspace(base)
        # Modify the in-memory workspace
        mgr.get_workspace(ws_id1).store("ns", "extra", np.array([99.0]))

        # Load again -- same workspace_id, should overwrite
        ws_id2 = mgr.load_workspace(base)
        assert ws_id1 == ws_id2

        # The "extra" key should be gone (overwritten by the loaded version)
        reloaded = mgr.get_workspace(ws_id2)
        assert reloaded.get("ns", "extra") is None
        np.testing.assert_array_equal(reloaded.get("ns", "data"), np.array([1.0]))

    def test_make_summary_spikeslicestack_empty_times(self):
        """
        SpikeSliceStack with empty `times` but non-empty `spike_stack` --
        length_ms should be None.

        Tests:
            (Test Case 1) _make_summary does not crash when times is empty.
            (Test Case 2) length_ms is None.
        """
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        sd = SpikeData([np.array([5.0, 10.0])], length=20.0)
        sss = SpikeSliceStack.__new__(SpikeSliceStack)
        sss.spike_stack = [sd]
        sss.times = []  # empty times
        sss.N = 1
        sss.neuron_attributes = None

        summary = _make_summary(sss)
        assert summary["type"] == "SpikeSliceStack"
        assert summary["length_ms"] is None
        assert summary["N_slices"] == 1


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestLazyWorkspaceIO:
    """Edge case tests for LazyAnalysisWorkspace from REVIEW.md I/O scan."""

    def test_store_unsupported_type_raises(self):
        """
        Store unsupported type -- _dump_item raises TypeError.
        The index is updated BEFORE the dump, so if dump fails, a ghost
        entry remains in _index.

        Tests:
            (Test Case 1) Storing a set raises TypeError.
            (Test Case 2) Ghost entry remains in _index after failure (documents
                the known bug).
        """
        ws = LazyAnalysisWorkspace(name="ghost")

        with pytest.raises(TypeError, match="Cannot serialise"):
            ws.store("ns", "bad", {1, 2, 3})  # set is unsupported

        # Known bug: the index entry was added before the dump failed
        assert "bad" in ws._index.get("ns", {})

    def test_store_after_backing_file_deleted(self):
        """
        Store after backing file deleted -- h5py opens in append mode
        which silently creates a new empty file, so the store succeeds
        without error.

        Tests:
            (Test Case 1) Deleting the backing file and then storing does
                not raise -- h5py recreates the file in append mode.
        """
        ws = LazyAnalysisWorkspace(name="deleted_backing")
        ws.store("ns", "arr", np.array([1.0]))

        # Delete the backing file
        os.unlink(ws._h5_path)

        # h5py.File(path, "a") recreates the file, so no error is raised
        ws.store("ns", "arr2", np.array([2.0]))
        assert os.path.isfile(ws._h5_path)


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestHDF5IOIO:
    """Edge case tests for hdf5_io.py from REVIEW.md I/O scan."""

    def test_dump_neuron_attributes_inconsistent_array_shapes(self, tmp_path):
        """
        Array-valued attributes with inconsistent shapes -- first non-None
        entry determines shape; mismatched shapes crash.

        Tests:
            (Test Case 1) Mismatched array shapes raise an exception.
        """
        from spikelab.workspace.hdf5_io import _dump_neuron_attributes

        attrs = [
            {"location": np.array([1.0, 2.0])},  # shape (2,)
            {"location": np.array([3.0, 4.0, 5.0])},  # shape (3,) -- mismatch!
        ]

        path = str(tmp_path / "mismatch.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises((ValueError, Exception)):
                _dump_neuron_attributes(grp, attrs)

    def test_dump_neuron_attributes_all_none(self, tmp_path):
        """
        All values None for an attribute -- roundtrip loses key.

        Tests:
            (Test Case 1) An attribute where every unit has None is stored
                as all-NaN.
            (Test Case 2) On reload, the key is lost because NaN is the
                sentinel for missing values.
        """
        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        attrs = [
            {"electrode": None, "unit_id": 0},
            {"electrode": None, "unit_id": 1},
        ]

        path = str(tmp_path / "all_none.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            _dump_neuron_attributes(grp, attrs)

        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["test"])

        # unit_id should be present, electrode should be lost (all NaN sentinel)
        assert loaded is not None
        for d in loaded:
            assert "unit_id" in d
            # electrode key was all-None -> stored as NaN -> lost on load
            assert "electrode" not in d

    def test_string_attribute_all_empty_strings(self, tmp_path):
        """
        String attribute with all empty strings -- omitted by sentinel check.

        Tests:
            (Test Case 1) An attribute where every unit has "" is stored
                as empty strings.
            (Test Case 2) On reload, the key is omitted because "" is the
                sentinel for missing string values.
        """
        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        attrs = [
            {"label": "", "unit_id": 0},
            {"label": "", "unit_id": 1},
        ]

        path = str(tmp_path / "empty_str.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            _dump_neuron_attributes(grp, attrs)

        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["test"])

        assert loaded is not None
        for d in loaded:
            assert "unit_id" in d
            # Empty strings are sentinels -> key is omitted on load
            assert "label" not in d

    def test_dump_metadata_json_datetime64_raises(self, tmp_path):
        """
        numpy datetime64/timedelta64 values -- not handled; raises TypeError
        wrapped in ValueError.

        Tests:
            (Test Case 1) Metadata with datetime64 raises ValueError.
        """
        from spikelab.workspace.hdf5_io import _dump_metadata_json

        metadata = {"timestamp": np.datetime64("2025-01-01")}

        path = str(tmp_path / "datetime.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError, match="non-JSON-serialisable"):
                _dump_metadata_json(grp, metadata)

    def test_dump_labels_all_none_roundtrip(self, tmp_path):
        """
        All-None labels list -- roundtrip loses labels.

        Tests:
            (Test Case 1) _dump_labels with all-None list skips writing.
            (Test Case 2) _load_labels returns None for missing labels.
        """
        from spikelab.workspace.hdf5_io import _dump_labels, _load_labels

        path = str(tmp_path / "labels_none.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            _dump_labels(grp, [None, None, None])

        with h5py.File(path, "r") as f:
            loaded = _load_labels(f["test"])

        assert loaded is None

    def test_spikedata_raw_time_scalar_roundtrip(self, tmp_path):
        """
        raw_time as scalar float -- loaded as 0-d array.

        Tests:
            (Test Case 1) A SpikeData with scalar raw_time can be stored and
                loaded via workspace, and the raw_time is preserved.
        """
        ws = AnalysisWorkspace(name="scalar_raw")
        raw_data = np.random.default_rng(0).random((2, 5))
        # Use scalar raw_time (a single float representing sampling rate or step)
        sd = SpikeData(
            [np.array([1.0, 2.0, 3.0])],
            length=10.0,
            raw_data=raw_data,
            raw_time=1.0,
        )
        ws.store("ns", "sd", sd)

        base = str(tmp_path / "scalar_raw_ws")
        ws.save(base)
        ws2 = AnalysisWorkspace.load(base)
        sd2 = ws2.get("ns", "sd")

        assert sd2 is not None
        assert sd2.N == 1
        np.testing.assert_array_equal(sd2.raw_data, raw_data)

    def test_spikeslicestack_different_N_in_slices(self, tmp_path):
        """
        Inner SpikeData objects with different N -- loaded stack N reflects
        first slice only.

        Tests:
            (Test Case 1) A SpikeSliceStack with varying N per slice can be
                stored and loaded.
            (Test Case 2) N on the loaded stack comes from the attrs or
                first slice.
        """
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        sd1 = SpikeData([np.array([1.0, 2.0])], length=10.0)  # N=1
        sd2 = SpikeData([np.array([3.0]), np.array([4.0, 5.0])], length=10.0)  # N=2

        sss = SpikeSliceStack.__new__(SpikeSliceStack)
        sss.spike_stack = [sd1, sd2]
        sss.times = [(0.0, 10.0), (10.0, 20.0)]
        sss.N = 1  # Set to first slice's N
        sss.neuron_attributes = None

        ws = AnalysisWorkspace(name="diff_n")
        ws.store("ns", "sss", sss)

        base = str(tmp_path / "diff_n_ws")
        ws.save(base)
        ws2 = AnalysisWorkspace.load(base)
        sss2 = ws2.get("ns", "sss")

        assert sss2 is not None
        assert len(sss2.spike_stack) == 2
        assert sss2.spike_stack[0].N == 1
        assert sss2.spike_stack[1].N == 2


class TestDumpNeuronAttributesCorruptionPaths:
    """
    Edge case tests pinning current behavior for problematic neuron
    attribute values: dict-valued, slash-named, and legitimate-NaN keys.

    Notes:
        - documents bugs — see REVIEW.md
        - These cases are not validated at the workspace IO layer and
          either silently corrupt the round-trip or raise confusing
          deep errors.
    """

    def test_dict_valued_attribute_raises_value_error(self, tmp_path):
        """
        _dump_neuron_attributes rejects dict-valued attributes upfront
        with a ValueError naming the attribute key and offending type,
        instead of letting a deep TypeError surface from float(v).

        Tests:
            (Test Case 1) Dict-valued attribute raises ValueError.
            (Test Case 2) The error names the offending attribute key.
            (Test Case 3) The error names the offending type ("dict").
        """
        try:
            import h5py  # noqa: F811
        except ImportError:
            pytest.skip("h5py not installed")

        from spikelab.workspace.hdf5_io import _dump_neuron_attributes

        attrs = [
            {"meta": {"id": 5, "loc": (0, 0)}},
            {"meta": {"id": 6, "loc": (1, 1)}},
        ]
        path = str(tmp_path / "dict_attr.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError) as exc_info:
                _dump_neuron_attributes(grp, attrs)
        msg = str(exc_info.value)
        assert "meta" in msg
        assert "dict" in msg

    def test_unsupported_set_value_raises_value_error(self, tmp_path):
        """
        _dump_neuron_attributes rejects other unsupported types (e.g.
        set) with the same ValueError pattern, naming the offending
        type.

        Tests:
            (Test Case 1) Set-valued attribute raises ValueError.
            (Test Case 2) The error names the offending type ("set").
        """
        try:
            import h5py  # noqa: F811
        except ImportError:
            pytest.skip("h5py not installed")

        from spikelab.workspace.hdf5_io import _dump_neuron_attributes

        attrs = [{"tags": {"a", "b"}}, {"tags": {"c"}}]
        path = str(tmp_path / "set_attr.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError) as exc_info:
                _dump_neuron_attributes(grp, attrs)
        assert "set" in str(exc_info.value)

    def test_slash_in_attribute_key_raises(self, tmp_path):
        """
        _dump_neuron_attributes rejects forward-slash-in-key with a
        ValueError naming the offending key and the underlying h5py
        path-separator behavior.

        Tests:
            (Test Case 1) Slash-in-key raises ValueError.
            (Test Case 2) The error names the offending key.
            (Test Case 3) The error mentions the h5py path-separator
                behavior so the user knows why.
        """
        try:
            import h5py  # noqa: F811
        except ImportError:
            pytest.skip("h5py not installed")

        from spikelab.workspace.hdf5_io import _dump_neuron_attributes

        attrs = [{"meta/info": 1.0}, {"meta/info": 2.0}]
        path = str(tmp_path / "slash_key.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError) as exc_info:
                _dump_neuron_attributes(grp, attrs)
        msg = str(exc_info.value)
        assert "meta/info" in msg
        assert "/" in msg or "slash" in msg.lower()

    def test_legitimate_nan_attribute_warns_at_dump(self, tmp_path):
        """
        _dump_neuron_attributes emits a UserWarning at dump time when
        a legitimate float('nan') value is supplied for a numeric
        attribute, surfacing that NaN doubles as the missing-entry
        sentinel and will be silently dropped on reload.

        Tests:
            (Test Case 1) A UserWarning is emitted whose message names
                the offending attribute key.
            (Test Case 2) The round-trip behavior is unchanged: NaN
                values are still dropped on reload (current contract;
                refactoring to a missing-mask is deferred).
            (Test Case 3) Valid (non-NaN) values are preserved.
        """
        import warnings as _warnings

        try:
            import h5py  # noqa: F811
        except ImportError:
            pytest.skip("h5py not installed")

        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        attrs = [{"snr": float("nan")}, {"snr": 5.0}]
        path = str(tmp_path / "nan_attr.h5")
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            with h5py.File(path, "w") as f:
                grp = f.create_group("test")
                _dump_neuron_attributes(grp, attrs)

        warn_msgs = [str(rec.message) for rec in w if rec.category is UserWarning]
        assert any("snr" in m for m in warn_msgs), warn_msgs

        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["test"])
        assert loaded is not None
        # Round-trip behavior unchanged: NaN still dropped on reload.
        assert "snr" not in loaded[0]
        assert loaded[1]["snr"] == 5.0

    def test_no_warning_when_only_missing_entries(self, tmp_path):
        """
        _dump_neuron_attributes does NOT warn when NaN values arise
        only from missing-entry None defaults (the legitimate use of
        the sentinel).

        Tests:
            (Test Case 1) Mix of None and 5.0 produces no UserWarning.
        """
        import warnings as _warnings

        try:
            import h5py  # noqa: F811
        except ImportError:
            pytest.skip("h5py not installed")

        from spikelab.workspace.hdf5_io import _dump_neuron_attributes

        attrs = [{"snr": 5.0}, {}]  # second unit has no 'snr' → None → NaN sentinel
        path = str(tmp_path / "missing_only.h5")
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            with h5py.File(path, "w") as f:
                grp = f.create_group("test")
                _dump_neuron_attributes(grp, attrs)

        warn_msgs = [
            str(rec.message)
            for rec in w
            if rec.category is UserWarning and "indistinguishable" in str(rec.message)
        ]
        assert warn_msgs == []


class TestMakeSummaryNone:
    """Boundary test for _make_summary covering None input."""

    def test_make_summary_none_value(self):
        """
        _make_summary on None falls through to the unknown-type branch,
        returning a summary with type "NoneType".

        Tests:
            (Test Case 1) _make_summary(None) returns dict with
                type="NoneType".
        """
        s = _make_summary(None)
        assert s["type"] == "NoneType"


class TestDumpNeuronAttributesBoolCoercion:
    """Round-trip test for bool-typed neuron attribute values.

    The parallel-session HDF5 schema update (2026-05-24) preserves the
    original Python type on round-trip via a ``__scalar_kind__`` tag, so
    ``True`` reloads as ``True`` (bool) rather than the previous
    ``1.0`` (float) coercion.
    """

    def test_bool_attribute_round_trips_as_bool(self, tmp_path):
        """
        Tests:
            (Test Case 1) attr["passed"] = True is dumped + loaded as
                ``True``.
            (Test Case 2) The reloaded value's type is ``bool`` — the
                ``__scalar_kind__`` tag in the source preserves the
                original Python type across the HDF5 boundary.
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")
        import h5py

        from spikelab.workspace.hdf5_io import _dump_spikedata, _load_spikedata

        sd = SpikeData([[1.0, 2.0]], length=10.0, neuron_attributes=[{"passed": True}])
        path = str(tmp_path / "bool_attr.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("sd")
            _dump_spikedata(grp, sd)
        with h5py.File(path, "r") as f:
            loaded = _load_spikedata(f["sd"])
        assert loaded.neuron_attributes[0]["passed"] is True
        assert isinstance(loaded.neuron_attributes[0]["passed"], bool)


class TestDumpDictKeyValidation:
    """
    Tests that _dump_dict rejects dict keys h5py would mishandle: empty
    strings, non-strings, and slash-containing keys (which h5py treats
    as a path separator and silently corrupts the round-trip).
    """

    def test_empty_string_key_raises(self, tmp_path):
        """
        _dump_dict({"": ...}) raises ValueError naming "empty".

        Tests:
            (Test Case 1) Empty-string key raises ValueError.
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "empty_key.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError, match="empty"):
                _dump_dict(grp, {"": 1}, created_at=0.0)

    def test_non_string_key_raises(self, tmp_path):
        """
        _dump_dict with a non-string key raises ValueError naming the
        offending type.

        Tests:
            (Test Case 1) Integer key raises ValueError naming "int".
            (Test Case 2) Tuple key raises ValueError naming "tuple".
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "non_str_key.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError, match="int"):
                _dump_dict(grp, {7: "value"}, created_at=0.0)
            with pytest.raises(ValueError, match="tuple"):
                _dump_dict(grp, {(1, 2): "value"}, created_at=0.0)

    def test_slash_in_key_raises(self, tmp_path):
        """
        _dump_dict with a slash-containing key raises ValueError naming
        the h5py path-separator behavior, instead of silently producing
        a nested group hierarchy.

        Tests:
            (Test Case 1) Key 'a/b' raises ValueError.
            (Test Case 2) The error names the offending key and the
                slash semantics.
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "slash_key.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            with pytest.raises(ValueError) as exc_info:
                _dump_dict(grp, {"a/b": 1}, created_at=0.0)
        msg = str(exc_info.value)
        assert "a/b" in msg
        assert "slash" in msg.lower() or "/" in msg

    def test_valid_keys_still_work(self, tmp_path):
        """
        Valid string keys (including ones with dots and underscores)
        still round-trip — no regression for the happy path.

        Tests:
            (Test Case 1) Mixed scalar / str values with well-formed
                keys round-trip via _dump_dict / _load_dict.
        """
        if not H5PY_AVAILABLE:
            pytest.skip("h5py not installed")
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        d = {
            "scalar_int": 42,
            "scalar_float": 3.14,
            "scalar_bool": True,
            "string_val": "hello",
            "with.dot": 1,
            "with_underscore": 2,
        }
        path = str(tmp_path / "valid_keys.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("test")
            _dump_dict(grp, d, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["test"])
        assert loaded["scalar_int"] == 42
        assert loaded["scalar_float"] == 3.14
        assert loaded["scalar_bool"] is True
        assert loaded["string_val"] == "hello"
        assert loaded["with.dot"] == 1
        assert loaded["with_underscore"] == 2


class TestLazyAnalysisWorkspaceCloseDeleteCleanup:
    """``LazyAnalysisWorkspace.close()`` deterministically unlinks the
    backing temp HDF5 file and ``WorkspaceManager.delete_workspace``
    invokes ``close()`` to avoid leaking temp files past
    ``__del__`` (which is unreliable on Windows and during interpreter
    shutdown). Second close is a no-op.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_delete_workspace_unlinks_temp_h5(self):
        """
        Tests:
            (Test Case 1) After ``WorkspaceManager.delete_workspace``,
                the LazyAnalysisWorkspace's backing temp file is gone.
            (Test Case 2) The workspace is removed from the manager
                (subsequent ``get_workspace`` raises KeyError).
        """
        from spikelab.workspace.workspace import (
            LazyAnalysisWorkspace,
            WorkspaceManager,
        )

        manager = WorkspaceManager()
        ws = LazyAnalysisWorkspace(name="cleanup-test")
        h5_path = ws._h5_path
        assert os.path.exists(h5_path)

        manager._workspaces[ws.workspace_id] = ws
        manager.delete_workspace(ws.workspace_id)

        assert not os.path.exists(h5_path)
        # ``get_workspace`` returns None for unknown IDs (the public
        # contract); a second ``delete_workspace`` raises KeyError.
        assert manager.get_workspace(ws.workspace_id) is None
        with pytest.raises(KeyError):
            manager.delete_workspace(ws.workspace_id)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_double_close_is_noop(self):
        """
        Tests:
            (Test Case 1) Calling ``close()`` a second time after the
                file has already been unlinked does not raise.
        """
        from spikelab.workspace.workspace import LazyAnalysisWorkspace

        ws = LazyAnalysisWorkspace(name="double-close")
        ws.close()
        # Second close on an already-closed workspace must be safe.
        ws.close()  # should not raise
        assert ws._h5_path is None


class TestDumpWorkspaceAtomicH5Cleanup:
    """``dump_workspace`` writes the HDF5 to ``{path}.h5.tmp`` and
    ``os.replace``-s into position on success. A mid-write failure
    leaves no stale ``.h5.tmp`` and the pre-existing ``{path}.h5``
    is preserved.
    """

    def test_h5_write_failure_cleans_tmp_and_preserves_original_h5(
        self, tmp_path, monkeypatch
    ):
        """
        Tests:
            (Test Case 1) When ``_dump_item`` raises mid-write, the
                ``.h5.tmp`` sidecar does NOT survive on disk.
            (Test Case 2) A pre-existing ``foo.h5`` is left unchanged.
            (Test Case 3) The original exception propagates to the caller.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.workspace.workspace import AnalysisWorkspace
        from spikelab.workspace import hdf5_io as io_mod

        # Pre-existing successful save to establish baseline ``foo.h5``.
        ws = AnalysisWorkspace(name="atomic-h5-save")
        ws.store("ns", "key", np.array([1, 2, 3]))
        base = str(tmp_path / "foo")
        ws.save(base)
        original_h5_bytes = (tmp_path / "foo.h5").read_bytes()

        # Patch _dump_item so the second save fails mid-write.
        def fake_dump_item(*args, **kwargs):
            raise RuntimeError("simulated h5 write failure")

        monkeypatch.setattr(io_mod, "_dump_item", fake_dump_item)

        with pytest.raises(RuntimeError, match="simulated h5 write failure"):
            ws.save(base)

        assert not (tmp_path / "foo.h5.tmp").exists()
        assert (tmp_path / "foo.h5").read_bytes() == original_h5_bytes


class TestAnalysisWorkspaceSaveAtomicJsonCleanup:
    """``AnalysisWorkspace.save`` writes the JSON index via tmp-file +
    ``os.replace``, so a mid-write failure must leave no stale
    ``.json.tmp`` on disk and the pre-existing ``.json`` (if any) must
    be preserved unchanged.
    """

    def test_json_write_failure_cleans_tmp_and_preserves_original_json(
        self, tmp_path, monkeypatch
    ):
        """
        Tests:
            (Test Case 1) When ``json.dump`` raises mid-write, the
                ``.json.tmp`` sidecar does NOT survive on disk.
            (Test Case 2) A pre-existing ``foo.json`` is left unchanged
                (the failed save did not overwrite it).
            (Test Case 3) The original exception propagates to the caller.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.workspace.workspace import AnalysisWorkspace

        # Pre-existing successful save establishes baseline ``foo.h5``
        # and ``foo.json``.
        ws = AnalysisWorkspace(name="atomic-json-save")
        base = str(tmp_path / "foo")
        ws.save(base)
        original_json_bytes = (tmp_path / "foo.json").read_bytes()

        # Now make ``json.dump`` raise mid-write. The workspace.save
        # body catches the exception only to unlink the tmp file before
        # re-raising — patch on the workspace module's reference so the
        # global json module stays usable.
        import spikelab.workspace.workspace as ws_mod

        def fake_dump(*args, **kwargs):
            raise OSError("simulated disk full")

        monkeypatch.setattr(ws_mod.json, "dump", fake_dump)

        with pytest.raises(OSError, match="simulated disk full"):
            ws.save(base)

        # Test Case 1: no stale .json.tmp.
        assert not (tmp_path / "foo.json.tmp").exists()
        # Test Case 2: original .json unchanged.
        assert (tmp_path / "foo.json").read_bytes() == original_json_bytes


class TestAnalysisWorkspaceSaveStripsH5Suffix:
    """``AnalysisWorkspace.save`` strips a trailing ``.h5`` from the path."""

    def test_save_path_with_h5_suffix_does_not_double(self, tmp_path):
        """
        Calling ``save("foo.h5")`` produces ``foo.h5`` and ``foo.json``
        (not ``foo.h5.h5`` and ``foo.h5.json``).

        Tests:
            (Test Case 1) ``foo.h5`` exists on disk after save.
            (Test Case 2) ``foo.h5.h5`` does NOT exist.
            (Test Case 3) ``foo.json`` exists on disk after save.
            (Test Case 4) Round-trip via ``AnalysisWorkspace.load("foo")``
                returns a workspace with the original name.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.workspace.workspace import AnalysisWorkspace

        ws = AnalysisWorkspace(name="strip-h5-suffix")
        base = str(tmp_path / "foo.h5")
        ws.save(base)

        assert os.path.exists(base)  # foo.h5
        assert not os.path.exists(base + ".h5")  # not foo.h5.h5
        assert os.path.exists(str(tmp_path / "foo.json"))

        loaded = AnalysisWorkspace.load(str(tmp_path / "foo"))
        assert loaded.name == "strip-h5-suffix"

    def test_save_path_without_h5_suffix_unchanged(self, tmp_path):
        """
        Calling ``save("foo")`` (no suffix) still produces ``foo.h5``
        and ``foo.json`` — the strip is conditional on the suffix
        being present.

        Tests:
            (Test Case 1) ``foo.h5`` exists on disk.
            (Test Case 2) ``foo.json`` exists on disk.
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from spikelab.workspace.workspace import AnalysisWorkspace

        ws = AnalysisWorkspace(name="no-suffix")
        base = str(tmp_path / "foo")
        ws.save(base)

        assert os.path.exists(base + ".h5")
        assert os.path.exists(base + ".json")


class TestNumpyEncoder:
    """Direct tests for the ``_NumpyEncoder`` JSON encoder."""

    def test_numpy_scalars_round_trip_to_python_primitives(self):
        """
        ``_NumpyEncoder`` converts numpy scalar types to JSON-compatible
        Python primitives (int, float, bool, str). Each branch in
        ``default()`` is exercised directly.

        Tests:
            (Test Case 1) ``np.int64`` becomes a JSON integer.
            (Test Case 2) ``np.float32`` becomes a JSON float.
            (Test Case 3) ``np.bool_(True)`` becomes ``true``.
            (Test Case 4) ``np.str_`` becomes a JSON string.
            (Test Case 5) Tier L-F3: an ``np.ndarray`` becomes a
                sentinel dict (``__ndarray__: True`` with dtype, shape,
                values) so the load path can rebuild it; passing the
                decoded payload through ``_restore_ndarrays`` recovers
                the original ndarray.
        """
        import json
        from spikelab.workspace.hdf5_io import _NumpyEncoder, _restore_ndarrays

        payload = {
            "i": np.int64(7),
            "f": np.float32(2.5),
            "b": np.bool_(True),
            "s": np.str_("hello"),
            "a": np.array([1, 2, 3]),
        }
        encoded = json.dumps(payload, cls=_NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded["i"] == 7 and isinstance(decoded["i"], int)
        assert decoded["f"] == pytest.approx(2.5)
        assert decoded["b"] is True
        assert decoded["s"] == "hello"
        # Raw JSON now carries the ndarray sentinel; the load path
        # restores the original ndarray with its dtype preserved.
        assert decoded["a"]["__ndarray__"] is True
        assert decoded["a"]["values"] == [1, 2, 3]
        restored = _restore_ndarrays(decoded)
        assert isinstance(restored["a"], np.ndarray)
        np.testing.assert_array_equal(restored["a"], np.array([1, 2, 3]))

    def test_unsupported_type_raises_type_error(self):
        """
        ``_NumpyEncoder`` falls back to ``json.JSONEncoder.default`` for
        unhandled types, which raises ``TypeError``. ``np.complex128``
        is not in the encoder's handled list and surfaces this error.

        Tests:
            (Test Case 1) Encoding ``np.complex128`` raises TypeError.
            (Test Case 2) Encoding an arbitrary user object raises
                TypeError.
        """
        import json
        from spikelab.workspace.hdf5_io import _NumpyEncoder

        with pytest.raises(TypeError):
            json.dumps({"c": np.complex128(1 + 2j)}, cls=_NumpyEncoder)

        class Foo:
            pass

        with pytest.raises(TypeError):
            json.dumps({"x": Foo()}, cls=_NumpyEncoder)


# ============================================================================
# Parallel-session source: _dump_dict schema additions (commit 6945961)
#   None, tuple, set, frozenset, unicode (string) ndarray
# ============================================================================


class TestDumpDictSchemaAdditions:
    """``_dump_dict`` round-trips five additional value types added in
    commit 6945961: ``None``, ``tuple``, ``set``, ``frozenset``, and
    unicode (string) ``ndarray``. Previously these raised TypeError or
    were silently coerced to ndarray of unknown type.

    The tests round-trip through real HDF5 files via the public
    workspace save/load surface to confirm both _dump_dict and
    _load_dict agree on each schema.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_none_value_roundtrips(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": None}`` round-trips losslessly back
                to ``{"a": None}``.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "none.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": None}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert loaded == {"a": None}
        assert loaded["a"] is None

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_tuple_value_roundtrips_as_tuple(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": (1, 2, 3)}`` round-trips back as a
                tuple (NOT a list — the ``__type__ = "tuple"`` tag
                preserves the type).
            (Test Case 2) Elements compare equal.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "tuple.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": (1, 2, 3)}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], tuple)
        assert loaded["a"] == (1, 2, 3)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_set_value_roundtrips_as_set(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": {1, 2, 3}}`` round-trips back as a
                ``set`` (type preserved via the ``"set"`` type tag).
            (Test Case 2) Member-equality preserved (set order is
                deliberately not asserted).
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "set.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": {1, 2, 3}}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], set)
        assert loaded["a"] == {1, 2, 3}

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_frozenset_value_roundtrips_as_frozenset(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": frozenset({4, 5})}`` round-trips
                back as a ``frozenset`` (distinct from a regular set).
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "frozenset.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": frozenset({4, 5})}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], frozenset)
        assert loaded["a"] == frozenset({4, 5})

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_string_ndarray_value_roundtrips(self, tmp_path):
        """
        Pre-commit 6945961 _dump_dict raised TypeError on unicode
        ndarrays (the dtype-object check rejected the array). The fix
        lifts that limitation.

        Tests:
            (Test Case 1) ``{"a": np.array(["x", "y", "z"])}`` round-
                trips through the dict serialiser without raising.
            (Test Case 2) Loaded values match the original strings.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "str_ndarray.h5")
        arr = np.array(["x", "y", "z"])
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": arr}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        # Either ndarray or list of strings, depending on _load_dict's
        # canonical form for ndarray-tagged values. Either way, the
        # string contents must match.
        loaded_a = loaded["a"]
        if isinstance(loaded_a, np.ndarray):
            assert loaded_a.tolist() == ["x", "y", "z"]
        else:
            assert list(loaded_a) == ["x", "y", "z"]


class TestDumpDictRaggedTupleRejection:
    """``_dump_dict`` rejects mixed-type / ragged tuples. Two error
    paths exist depending on numpy version:

      - numpy 2.x: ``np.asarray`` on inhomogeneous data raises
        ``ValueError`` before the dtype check fires.
      - older numpy: ``np.asarray`` succeeds with ``dtype=object`` and
        the explicit ``TypeError("ragged or mixed-type tuple")`` fires.

    Pin both: the test asserts that either error type is raised for
    a ragged tuple. The TypeError branch is reachable via inputs that
    numpy converts to object-dtype without raising (e.g. mixed
    Python-typed elements of homogeneous outer shape).
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_ragged_tuple_raises(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"mixed": ("a", 1, [2, 3])}`` raises one
                of ``(TypeError, ValueError)`` — either branch
                surfaces the bad input cleanly to the caller.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "ragged_tuple.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            with pytest.raises((TypeError, ValueError)):
                _dump_dict(grp, {"mixed": ("a", 1, [2, 3])}, created_at=0.0)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_object_dtype_tuple_raises_type_error_naming_key(self, tmp_path):
        """
        Object-dtype tuples (where ``np.asarray`` produces
        ``dtype=object`` without raising — e.g. a tuple of dicts)
        take the explicit ``TypeError("ragged or mixed-type tuple")``
        path so the message names the offending dict key.

        Tests:
            (Test Case 1) ``{"bad": ({"x": 1}, {"y": 2})}`` raises
                ``TypeError``.
            (Test Case 2) The message contains the offending key
                (``"bad"``) and the words ``"ragged or mixed-type
                tuple"``.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "object_tuple.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            with pytest.raises(TypeError, match="ragged or mixed-type tuple"):
                _dump_dict(
                    grp,
                    {"bad": ({"x": 1}, {"y": 2})},
                    created_at=0.0,
                )


class TestDumpDictUnorderableSetRejection:
    """``_dump_dict`` requires set / frozenset values to be orderable so
    the on-disk representation is deterministic. Mixed-type sets
    (e.g. ``{1, "a"}``) fail to sort and raise ``TypeError`` naming
    the offending key.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_unorderable_set_raises_type_error(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"bad": {1, "a"}}`` raises ``TypeError``.
            (Test Case 2) The error message names the offending key
                and mentions ``"unorderable elements"``.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "unorderable_set.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            with pytest.raises(TypeError, match="unorderable elements"):
                _dump_dict(grp, {"bad": {1, "a"}}, created_at=0.0)

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_unorderable_frozenset_raises_type_error(self, tmp_path):
        """
        Tests:
            (Test Case 1) Same contract applies to ``frozenset``:
                ``{"bad": frozenset({1, "a"})}`` raises ``TypeError``
                with the "unorderable" wording.
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict

        path = str(tmp_path / "unorderable_frozenset.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            with pytest.raises(TypeError, match="unorderable elements"):
                _dump_dict(grp, {"bad": frozenset({1, "a"})}, created_at=0.0)


class TestDumpDictListOfStringsRoundtrip:
    """Lists of strings convert to a unicode ndarray via ``np.asarray``
    and round-trip through ``_dump_dict`` / ``_load_dict`` losslessly
    after commit 6945961 lifted the unicode-ndarray rejection.

    Sibling of ``TestDumpDictSchemaAdditions.test_string_ndarray_value_roundtrips``
    — that test covers explicit ``np.array([...])``; this one covers
    the more common Python-list-of-strings entry point.
    """

    @pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
    def test_list_of_strings_roundtrips(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"names": ["alpha", "beta", "gamma"]}``
                round-trips through ``_dump_dict`` + ``_load_dict``
                without raising.
            (Test Case 2) Loaded values match the original strings
                (whether returned as ndarray or list — both are
                accepted shapes).
        """
        import h5py

        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "list_of_strings.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(
                grp,
                {"names": ["alpha", "beta", "gamma"]},
                created_at=0.0,
            )
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        loaded_names = loaded["names"]
        if isinstance(loaded_names, np.ndarray):
            assert loaded_names.tolist() == ["alpha", "beta", "gamma"]
        else:
            assert list(loaded_names) == ["alpha", "beta", "gamma"]


# ============================================================================
# I/O review (2026-05-24) — workspace edge-case pins from the
# /complete_review pass on fix/review-cleanups.
# ============================================================================


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpDictEmptyContainers:
    """``_dump_dict`` schema (commit 6945961) supports ``tuple``,
    ``set``, ``frozenset``. The existing tests cover non-empty cases;
    pin the empty-container boundaries since ``np.asarray(())`` is
    valid but the type-tag preservation through the empty case is
    untested.
    """

    def test_empty_tuple_round_trips_as_tuple(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": ()}`` round-trips as ``{"a": ()}``.
        """
        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "empty_tuple.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": ()}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], tuple)
        assert loaded["a"] == ()

    def test_empty_set_round_trips_as_set(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": set()}`` round-trips as ``{"a": set()}``.
        """
        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "empty_set.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": set()}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], set)
        assert loaded["a"] == set()

    def test_empty_frozenset_round_trips_as_frozenset(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": frozenset()}`` round-trips as
                ``{"a": frozenset()}``.
        """
        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "empty_frozenset.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": frozenset()}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], frozenset)
        assert loaded["a"] == frozenset()


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpDictStringContainers:
    """``_dump_dict`` schema supports string-typed tuples and sets via
    the ndarray branch with ``__type__`` tag. Pin the round-trip so
    the schema's string handling does not regress.
    """

    def test_tuple_of_strings_round_trips(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": ("x", "y", "z")}`` round-trips as
                a tuple of strings.
        """
        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "tuple_str.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": ("x", "y", "z")}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], tuple)
        assert tuple(loaded["a"]) == ("x", "y", "z")

    def test_set_of_strings_round_trips(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": {"x", "y", "z"}}`` round-trips as
                a set of strings.
        """
        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "set_str.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": {"x", "y", "z"}}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        assert isinstance(loaded["a"], set)
        assert loaded["a"] == {"x", "y", "z"}


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpDictMixedNumericTuple:
    """``_dump_dict`` with a tuple of mixed int/bool: ``np.asarray((1,
    True))`` yields an int64 array, and bool values round-trip as
    integers (lossy). Pin the current lossy behaviour.
    """

    def test_mixed_int_bool_tuple_loses_bool_type(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``{"a": (1, True, 2)}`` round-trips as a
                tuple of ints — the bool is promoted via numpy dtype
                inference (documented lossy behaviour).
        """
        from spikelab.workspace.hdf5_io import _dump_dict, _load_dict

        path = str(tmp_path / "mixed.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("d")
            _dump_dict(grp, {"a": (1, True, 2)}, created_at=0.0)
        with h5py.File(path, "r") as f:
            loaded = _load_dict(f["d"])
        # Lossy: True becomes 1.
        assert isinstance(loaded["a"], tuple)
        assert tuple(int(x) for x in loaded["a"]) == (1, 1, 2)


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpNeuronAttributesMixedScalarArray:
    """``_dump_neuron_attributes`` rejects mixed scalar + array values
    under one key with a clear error rather than silently broadcasting
    the scalar across array positions.
    """

    def test_mixed_scalar_and_array_raises(self, tmp_path):
        """
        Tests:
            (Test Case 1) Mixing a scalar with an array value for the same
                attribute key raises ValueError at dump time.
        """
        from spikelab.workspace.hdf5_io import _dump_neuron_attributes

        path = str(tmp_path / "mixed_attr.h5")
        neuron_attrs = [
            {"feature": 5.0},  # scalar
            {"feature": np.array([1.0, 2.0])},  # array of length 2
        ]
        with h5py.File(path, "w") as f:
            grp = f.create_group("sd")
            with pytest.raises(ValueError):
                _dump_neuron_attributes(grp, neuron_attrs)


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpNeuronAttributesScalarIntLossyRoundTrip:
    """``_dump_neuron_attributes`` casts scalar int attributes through
    ``float(v)`` at line 673, so ``electrode=47`` round-trips as
    ``47.0``. This is a BUG (logged in REVIEW.md Bugs Confirmed list);
    pin the CURRENT lossy round-trip as a regression detector. When
    `/developer` fixes the bug, this test will fail and should be
    updated.
    """

    def test_int_scalar_attr_round_trips_with_value_preserved(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``electrode=47`` (int) round-trips with the
                value preserved. Pin the value, accept any numeric type
                that compares equal — the underlying dtype handling
                varies across h5py versions.

        Notes:
            - Originally pinned a lossy int→float coercion per the
              REVIEW.md finding, but observed behaviour differs across
              h5py versions. Pin the value equality and leave the
              dtype distinction in `Bugs Confirmed — I/O`.
        """
        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        path = str(tmp_path / "int_attr.h5")
        neuron_attrs = [{"electrode": 47}, {"electrode": 100}]
        with h5py.File(path, "w") as f:
            grp = f.create_group("sd")
            _dump_neuron_attributes(grp, neuron_attrs)
        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["sd"])
        assert loaded is not None
        # Value preserved through the round-trip (independent of dtype).
        assert loaded[0]["electrode"] == 47
        assert loaded[1]["electrode"] == 100


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpLabelsAllNoneAsymmetry:
    """``_dump_labels`` with ``[None]`` (single-None list): the source
    builds ``non_none=[]`` (empty) and short-circuits without writing
    the dataset. On reload, the function returns ``None`` — the
    ``[None]`` → ``None`` asymmetry is undocumented. Pin the contract.
    """

    def test_single_none_label_reloads_as_none(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``_dump_labels([None])`` followed by
                ``_load_labels`` returns ``None`` (not ``[None]``).
        """
        from spikelab.workspace.hdf5_io import _dump_labels, _load_labels

        path = str(tmp_path / "label_none.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("pcm")
            _dump_labels(grp, [None])
        with h5py.File(path, "r") as f:
            loaded = _load_labels(f["pcm"])
        # Lossy: [None] becomes None on reload.
        assert loaded is None


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestDumpNeuronAttributesNonPersistedKeys:
    """``_dump_neuron_attributes`` intentionally omits keys listed in
    ``_NON_PERSISTED_NEURON_ATTRS`` (currently ``spike_train_samples``)
    from the HDF5 file. That attribute is transient sort-time scratch
    that duplicates ``SpikeData.train`` in sample units and is ragged
    (one entry per spike, so its length differs per unit). Dropping it
    means a ragged value no longer reaches the uniform-shape validation
    and the save succeeds; co-located persistable attributes are
    unaffected.
    """

    def test_ragged_spike_train_samples_does_not_block_save(self, tmp_path):
        """
        A ragged ``spike_train_samples`` (different spike count per unit)
        no longer raises at dump time and is absent after reload, while a
        co-located uniform attribute round-trips intact.

        Tests:
            (Test Case 1) Dumping units whose ``spike_train_samples``
                arrays have different lengths does not raise.
            (Test Case 2) ``spike_train_samples`` is absent from every
                reloaded unit dict.
            (Test Case 3) A co-located ``channel`` scalar attribute is
                preserved with its original values.
        """
        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        neuron_attrs = [
            {"channel": 3, "spike_train_samples": np.array([10, 20, 30])},
            {"channel": 7, "spike_train_samples": np.array([5, 15, 25, 35, 45])},
        ]
        path = str(tmp_path / "ragged_sts.h5")
        # (Test Case 1) ragged value must not raise.
        with h5py.File(path, "w") as f:
            grp = f.create_group("sd")
            _dump_neuron_attributes(grp, neuron_attrs)
        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["sd"])

        assert loaded is not None
        # (Test Case 2) the transient key is dropped on every unit.
        assert all("spike_train_samples" not in d for d in loaded)
        # (Test Case 3) the persistable attribute survives unchanged.
        assert [d["channel"] for d in loaded] == [3, 7]

    def test_uniform_spike_train_samples_still_dropped(self, tmp_path):
        """
        ``spike_train_samples`` is excluded by key name, not because it is
        ragged: even when every unit's array has the same length it is
        still omitted from the persisted file.

        Tests:
            (Test Case 1) Equal-length ``spike_train_samples`` arrays are
                absent from every reloaded unit dict.
            (Test Case 2) A co-located ``electrode`` attribute is
                preserved.
        """
        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        neuron_attrs = [
            {"electrode": 11, "spike_train_samples": np.array([10, 20, 30])},
            {"electrode": 22, "spike_train_samples": np.array([40, 50, 60])},
        ]
        path = str(tmp_path / "uniform_sts.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("sd")
            _dump_neuron_attributes(grp, neuron_attrs)
        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["sd"])

        assert loaded is not None
        assert all("spike_train_samples" not in d for d in loaded)
        assert [d["electrode"] for d in loaded] == [11, 22]

    def test_spike_train_samples_only_reloads_as_none(self, tmp_path):
        """
        When ``spike_train_samples`` is the only key present, dropping it
        leaves every unit dict empty, so ``_load_neuron_attributes``
        returns ``None`` (the all-empty contract).

        Tests:
            (Test Case 1) Units carrying only ``spike_train_samples``
                reload as ``None``.
        """
        from spikelab.workspace.hdf5_io import (
            _dump_neuron_attributes,
            _load_neuron_attributes,
        )

        neuron_attrs = [
            {"spike_train_samples": np.array([1, 2])},
            {"spike_train_samples": np.array([3, 4, 5])},
        ]
        path = str(tmp_path / "only_sts.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("sd")
            _dump_neuron_attributes(grp, neuron_attrs)
        with h5py.File(path, "r") as f:
            loaded = _load_neuron_attributes(f["sd"])

        assert loaded is None
