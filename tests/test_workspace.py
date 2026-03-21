"""
Tests for AnalysisWorkspace and WorkspaceManager (workspace/workspace.py).

Covers: store/get round-trips for every supported type, summary generation,
get_info, describe, list_keys, rename, add_note, delete, save/load, note
handling, namespaced isolation, comparison_namespace, WorkspaceManager CRUD,
and the get_workspace_manager singleton.
"""

import json
import pathlib
import sys
import tempfile

import numpy as np
import pytest

try:
    import h5py  # noqa: F401

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.spikedata import SpikeData
from SpikeLab.spikedata.ratedata import RateData
from SpikeLab.spikedata.rateslicestack import RateSliceStack
from SpikeLab.spikedata.spikeslicestack import SpikeSliceStack
from SpikeLab.spikedata.pairwise import (
    PairwiseCompMatrix,
    PairwiseCompMatrixStack,
)
from SpikeLab.workspace.workspace import (
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
            (Test Case 4) Rename on a missing namespace returns False.
            (Test Case 5) Rename on a missing old_key returns False.
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

        assert not self.ws.rename("missing_ns", "old", "new")
        assert not self.ws.rename("rec1", "missing_key", "new")

    # ------------------------------------------------------------------
    # add_note
    # ------------------------------------------------------------------

    def test_add_note(self):
        """
        Tests add_note() attaches and replaces notes on index entries.

        Tests:
            (Test Case 1) Note stored via store() appears in get_info().
            (Test Case 2) add_note() on existing item updates the note.
            (Test Case 3) add_note() on missing item returns False.
        """
        self.ws.store("rec1", "arr", np.zeros(3), note="initial note")
        info = self.ws.get_info("rec1", "arr")
        assert info["note"] == "initial note"

        result = self.ws.add_note("rec1", "arr", "updated note")
        assert result
        assert self.ws.get_info("rec1", "arr")["note"] == "updated note"

        assert not self.ws.add_note("missing_ns", "arr", "note")

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
            (Test Case 4) Deleting a missing key returns False.
        """
        self.ws.store("rec1", "a", np.zeros(2))
        self.ws.store("rec1", "b", np.zeros(2))

        result = self.ws.delete("rec1", "a")
        assert result
        assert self.ws.get("rec1", "a") is None
        assert self.ws.get_info("rec1", "a") is None
        assert self.ws.get("rec1", "b") is not None

        assert not self.ws.delete("rec1", "missing_key")

    def test_delete_namespace(self):
        """
        Tests that delete() without a key removes the entire namespace.

        Tests:
            (Test Case 1) All keys in the namespace are gone after deletion.
            (Test Case 2) Namespace no longer appears in list_keys().
            (Test Case 3) Deleting a missing namespace returns False.
        """
        self.ws.store("rec1", "a", np.zeros(2))
        self.ws.store("rec1", "b", np.zeros(2))

        result = self.ws.delete("rec1")
        assert result
        assert self.ws.get("rec1", "a") is None
        assert "rec1" not in self.ws.list_keys()

        assert not self.ws.delete("missing_ns")

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
        Tests that delete_workspace() removes a workspace and returns False for unknown IDs.

        Tests:
            (Test Case 1) delete_workspace() returns True for an existing workspace.
            (Test Case 2) get_workspace() returns None after deletion.
            (Test Case 3) delete_workspace() returns False for an unknown ID.
        """
        wid = self.mgr.create_workspace()
        result = self.mgr.delete_workspace(wid)
        assert result
        assert self.mgr.get_workspace(wid) is None

        assert not self.mgr.delete_workspace("nonexistent-id")

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
            (Test Case 1) delete_workspace returns True.
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
        assert mgr.delete_workspace(wid) is True
        assert mgr.get_workspace(wid) is None

        # External reference still works
        assert isinstance(external_ref, AnalysisWorkspace)
        np.testing.assert_array_equal(external_ref.get("ns", "arr"), [1.0, 2.0])


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
            assert out.neuron_attributes[i]["waveform"].shape == (10, 5)
            np.testing.assert_array_almost_equal(
                out.neuron_attributes[i]["waveform"], waveforms[i]
            )
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

        Tests:
            (Test Case 1) Save does not raise despite the numpy array value.
            (Test Case 2) The metadata value is recovered as a Python list with equal elements.
        """
        arr = np.array([1.0, 2.0, 3.0])
        sd = SpikeData([[1.0, 2.0], [3.0]], length=20.0, metadata={"positions": arr})
        out = self._roundtrip(sd)
        assert "positions" in out.metadata
        assert out.metadata["positions"] == [1.0, 2.0, 3.0]

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
            (Test Case 5) _items dict is empty (data lives on disk, not in memory).
            (Test Case 6) _index dict is empty for a fresh workspace.
        """
        ws = LazyAnalysisWorkspace(name="test_lazy")

        assert isinstance(ws.workspace_id, str) and len(ws.workspace_id) > 0
        assert ws.name == "test_lazy"
        assert isinstance(ws.created_at, float) and ws.created_at > 0
        assert pathlib.Path(ws._h5_path).exists()
        assert ws._items == {}
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
            (Test Case 4) _items remains empty (data is on disk, not in memory).
        """
        ws = LazyAnalysisWorkspace(name="store_get")
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        ws.store("ns1", "my_array", arr)
        retrieved = ws.get("ns1", "my_array")

        np.testing.assert_array_equal(retrieved, arr)
        assert retrieved.dtype == arr.dtype
        assert retrieved.shape == arr.shape
        assert ws._items == {}

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
            (Test Case 1) delete() returns True for an existing item.
            (Test Case 2) get() returns None after deletion.
            (Test Case 3) The key is removed from list_keys().
            (Test Case 4) Other items in the same namespace are unaffected.
        """
        ws = LazyAnalysisWorkspace(name="delete_item")
        ws.store("ns", "keep", np.array([1.0]))
        ws.store("ns", "remove", np.array([2.0]))

        assert ws.delete("ns", "remove") is True
        assert ws.get("ns", "remove") is None
        assert "remove" not in ws.list_keys("ns")
        np.testing.assert_array_equal(ws.get("ns", "keep"), np.array([1.0]))

    def test_delete_entire_namespace(self):
        """
        Delete an entire namespace and verify all its items are gone.

        Tests:
            (Test Case 1) delete() with key=None returns True.
            (Test Case 2) The namespace is removed from list_namespaces().
            (Test Case 3) get() returns None for any key in the deleted namespace.
        """
        ws = LazyAnalysisWorkspace(name="delete_ns")
        ws.store("remove_ns", "k1", np.array([1.0]))
        ws.store("remove_ns", "k2", np.array([2.0]))
        ws.store("keep_ns", "k1", np.array([3.0]))

        assert ws.delete("remove_ns") is True
        assert "remove_ns" not in ws.list_namespaces()
        assert ws.get("remove_ns", "k1") is None
        assert ws.get("remove_ns", "k2") is None
        np.testing.assert_array_equal(ws.get("keep_ns", "k1"), np.array([3.0]))

    def test_delete_nonexistent_returns_false(self):
        """
        delete() returns False when the target does not exist.

        Tests:
            (Test Case 1) Missing namespace returns False.
            (Test Case 2) Missing key in existing namespace returns False.
        """
        ws = LazyAnalysisWorkspace(name="delete_miss")
        ws.store("ns", "k", np.zeros(1))

        assert ws.delete("nonexistent") is False
        assert ws.delete("ns", "nonexistent") is False

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

    def test_rename_missing_namespace_returns_false(self):
        """
        rename() returns False when the namespace does not exist.

        Tests:
            (Test Case 1) Returns False without error.
        """
        ws = LazyAnalysisWorkspace(name="rename_miss_ns")
        assert ws.rename("nonexistent", "a", "b") is False

    def test_rename_missing_key_returns_false(self):
        """
        rename() returns False when the old_key does not exist in the namespace.

        Tests:
            (Test Case 1) Returns False without error.
            (Test Case 2) Existing keys are unaffected.
        """
        ws = LazyAnalysisWorkspace(name="rename_miss_key")
        ws.store("ns", "exists", np.array([1.0]))
        assert ws.rename("ns", "missing", "new") is False
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
        from SpikeLab.workspace.hdf5_io import (
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
        from SpikeLab.workspace.hdf5_io import (
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
        from SpikeLab.workspace.hdf5_io import (
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
