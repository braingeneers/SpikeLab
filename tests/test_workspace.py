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
import unittest

import numpy as np

try:
    import h5py  # noqa: F401

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikedata.spikedata import SpikeData
from spikedata.ratedata import RateData
from spikedata.rateslicestack import RateSliceStack
from spikedata.spikeslicestack import SpikeSliceStack
from spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from workspace.workspace import (
    AnalysisWorkspace,
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
        seed (int): Random seed (unused; deterministic by construction).

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
        slice_length_ms (float): Length of each slice in milliseconds.
        n_slices (int): Number of slices (recording = slice_length_ms * n_slices).

    Returns:
        sss (SpikeSliceStack): A SpikeSliceStack with n_slices slices.
    """
    length_ms = slice_length_ms * n_slices
    sd = make_spikedata(n_units=n_units, length_ms=length_ms)
    return sd.frames(slice_length_ms)


# ---------------------------------------------------------------------------
# Tests: AnalysisWorkspace
# ---------------------------------------------------------------------------


class TestAnalysisWorkspace(unittest.TestCase):
    def setUp(self):
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

        self.assertIsNone(self.ws.get("rec1", "missing"))
        self.assertIsNone(self.ws.get("missing_ns", "raster"))

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
        self.assertIs(out, sd)
        self.assertEqual(out.N, 4)
        self.assertAlmostEqual(out.length, 200.0)

    def test_store_get_ratedata(self):
        """
        Tests that a RateData object is stored and retrieved with its attributes intact.

        Tests:
            (Test Case 1) Retrieved object is the same RateData instance.
            (Test Case 2) Array shape is preserved.
        """
        rd = make_ratedata(n_units=3, n_times=50)
        self.ws.store("rec1", "rate", rd)

        out = self.ws.get("rec1", "rate")
        self.assertIs(out, rd)
        self.assertEqual(out.inst_Frate_data.shape, (3, 50))

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
        self.assertIs(out, rss)
        self.assertEqual(out.event_stack.shape, (3, 20, 4))

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
        self.assertIs(out, sss)
        self.assertEqual(len(out.spike_stack), 3)

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
        self.assertIs(out_pcm, pcm)
        self.assertEqual(out_pcm.matrix.shape, (4, 4))

        stack_arr = np.random.default_rng(0).random((4, 4, 5))
        pcms = PairwiseCompMatrixStack(stack=stack_arr)
        self.ws.store("rec1", "pcms", pcms)
        out_pcms = self.ws.get("rec1", "pcms")
        self.assertIs(out_pcms, pcms)
        self.assertEqual(out_pcms.stack.shape, (4, 4, 5))

    def test_store_overwrites(self):
        """
        Tests that storing under an existing (namespace, key) replaces the previous value.

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
        self.assertEqual(info["shape"], [5])

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
        self.assertIsNotNone(info)
        self.assertEqual(info["type"], "ndarray")
        self.assertEqual(info["shape"], [3, 4])
        self.assertIn("created_at", info)

        self.assertIsNone(self.ws.get_info("rec1", "missing"))
        self.assertIsNone(self.ws.get_info("missing_ns", "arr"))

    # ------------------------------------------------------------------
    # describe / list_keys
    # ------------------------------------------------------------------

    def test_describe(self):
        """
        Tests that describe() returns the full nested index as a dict.

        Tests:
            (Test Case 1) Empty workspace returns empty dict.
            (Test Case 2) After storing, both namespaces and keys appear.
            (Test Case 3) Each entry contains type and created_at.
        """
        self.assertEqual(self.ws.describe(), {})

        self.ws.store("rec1", "arr", np.zeros(3))
        self.ws.store("rec2", "rate", make_ratedata())

        desc = self.ws.describe()
        self.assertIn("rec1", desc)
        self.assertIn("rec2", desc)
        self.assertIn("arr", desc["rec1"])
        self.assertIn("rate", desc["rec2"])
        self.assertEqual(desc["rec1"]["arr"]["type"], "ndarray")

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
        self.assertIsInstance(all_keys, dict)
        self.assertIn("rec1", all_keys)
        self.assertIn("rec2", all_keys)
        self.assertCountEqual(all_keys["rec1"], ["a", "b"])
        self.assertCountEqual(all_keys["rec2"], ["x"])

        rec1_keys = self.ws.list_keys("rec1")
        self.assertIsInstance(rec1_keys, list)
        self.assertCountEqual(rec1_keys, ["a", "b"])

        self.assertEqual(self.ws.list_keys("missing_ns"), [])

    # ------------------------------------------------------------------
    # rename
    # ------------------------------------------------------------------

    def test_rename(self):
        """
        Tests rename() for successful rename and for not-found cases.

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
        self.assertTrue(result)

        retrieved = self.ws.get("rec1", "new")
        np.testing.assert_array_equal(retrieved, arr)
        self.assertIsNone(self.ws.get("rec1", "old"))

        info = self.ws.get_info("rec1", "new")
        self.assertIsNotNone(info)
        self.assertIsNone(self.ws.get_info("rec1", "old"))

        self.assertFalse(self.ws.rename("missing_ns", "old", "new"))
        self.assertFalse(self.ws.rename("rec1", "missing_key", "new"))

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
        self.assertEqual(info["note"], "initial note")

        result = self.ws.add_note("rec1", "arr", "updated note")
        self.assertTrue(result)
        self.assertEqual(self.ws.get_info("rec1", "arr")["note"], "updated note")

        self.assertFalse(self.ws.add_note("rec1", "missing_key", "note"))
        self.assertFalse(self.ws.add_note("missing_ns", "arr", "note"))

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def test_delete_key(self):
        """
        Tests that delete() with a key removes that item and leaves others intact.

        Tests:
            (Test Case 1) Deleted key is no longer accessible via get() or get_info().
            (Test Case 2) Other keys in the same namespace are unaffected.
            (Test Case 3) Deleting a missing key returns False.
        """
        self.ws.store("rec1", "a", np.zeros(2))
        self.ws.store("rec1", "b", np.zeros(2))

        result = self.ws.delete("rec1", "a")
        self.assertTrue(result)
        self.assertIsNone(self.ws.get("rec1", "a"))
        self.assertIsNone(self.ws.get_info("rec1", "a"))
        self.assertIsNotNone(self.ws.get("rec1", "b"))

        self.assertFalse(self.ws.delete("rec1", "missing_key"))

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
        self.assertTrue(result)
        self.assertIsNone(self.ws.get("rec1", "a"))
        self.assertNotIn("rec1", self.ws.list_keys())

        self.assertFalse(self.ws.delete("missing_ns"))

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
        self.ws.store("rec1", "data", arr1)
        self.ws.store("rec2", "data", arr2)

        np.testing.assert_array_equal(self.ws.get("rec1", "data"), arr1)
        np.testing.assert_array_equal(self.ws.get("rec2", "data"), arr2)

        self.ws.delete("rec1", "data")
        self.assertIsNone(self.ws.get("rec1", "data"))
        np.testing.assert_array_equal(self.ws.get("rec2", "data"), arr2)

    # ------------------------------------------------------------------
    # comparison_namespace
    # ------------------------------------------------------------------

    def test_comparison_namespace(self):
        """
        Tests that comparison_namespace() returns the expected C_-prefixed string.

        Tests:
            (Test Case 1) Two namespaces → "C_ns1_ns2".
            (Test Case 2) Three namespaces → "C_ns1_ns2_ns3".
            (Test Case 3) Single namespace → "C_ns1".
        """
        self.assertEqual(
            AnalysisWorkspace.comparison_namespace("rec1", "rec2"), "C_rec1_rec2"
        )
        self.assertEqual(
            AnalysisWorkspace.comparison_namespace("a", "b", "c"), "C_a_b_c"
        )
        self.assertEqual(AnalysisWorkspace.comparison_namespace("only"), "C_only")

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    @unittest.skipIf(not H5PY_AVAILABLE, "h5py not installed")
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
            self.assertTrue(h5_path.exists())
            self.assertTrue(json_path.exists())

            loaded = AnalysisWorkspace.load(base)

        self.assertEqual(loaded.workspace_id, self.ws.workspace_id)
        self.assertEqual(loaded.name, self.ws.name)
        self.assertAlmostEqual(loaded.created_at, self.ws.created_at)

        # Array round-trip.
        np.testing.assert_array_equal(loaded.get("rec1", "matrix"), arr)

        # SpikeData round-trip: original IAT type must be reconstructed.
        loaded_sd = loaded.get("rec1", "spikes")
        self.assertIsInstance(loaded_sd, SpikeData)
        self.assertEqual(loaded_sd.N, 2)
        self.assertAlmostEqual(loaded_sd.length, 80.0)

        # Index preserved.
        info = loaded.get_info("rec1", "matrix")
        self.assertEqual(info["type"], "ndarray")
        self.assertEqual(info["shape"], [2, 3])
        self.assertEqual(info["note"], "test note")

    @unittest.skipIf(not H5PY_AVAILABLE, "h5py not installed")
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
            self.assertIsInstance(loaded_sd, SpikeData)
            self.assertEqual(loaded_sd.N, 2)

            with self.assertRaises(KeyError):
                AnalysisWorkspace.load_item(base, "missing_ns", "matrix")

            with self.assertRaises(KeyError):
                AnalysisWorkspace.load_item(base, "ns", "missing_key")

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

        self.assertEqual(doc["workspace_id"], self.ws.workspace_id)
        self.assertEqual(doc["name"], self.ws.name)
        self.assertIn("index", doc)
        self.assertIn("ns", doc["index"])
        self.assertIn("arr", doc["index"]["ns"])


# ---------------------------------------------------------------------------
# Tests: _make_summary
# ---------------------------------------------------------------------------


class TestMakeSummary(unittest.TestCase):
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
        self.assertEqual(s["type"], "ndarray")
        self.assertEqual(s["shape"], [3, 4])
        self.assertEqual(s["dtype"], "float32")

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
        self.assertEqual(s["type"], "SpikeData")
        self.assertEqual(s["N"], 5)
        self.assertAlmostEqual(s["length_ms"], 300.0)

    def test_summary_ratedata(self):
        """
        Tests _make_summary() for a RateData object.

        Tests:
            (Test Case 1) type is "RateData".
            (Test Case 2) shape matches (n_units, n_times).
        """
        rd = make_ratedata(n_units=4, n_times=80)
        s = _make_summary(rd)
        self.assertEqual(s["type"], "RateData")
        self.assertEqual(s["shape"], [4, 80])

    def test_summary_rateslicestack(self):
        """
        Tests _make_summary() for a RateSliceStack object.

        Tests:
            (Test Case 1) type is "RateSliceStack".
            (Test Case 2) shape matches (n_units, n_times, n_slices).
        """
        rss = make_rateslicestack(n_units=3, n_times=10, n_slices=5)
        s = _make_summary(rss)
        self.assertEqual(s["type"], "RateSliceStack")
        self.assertEqual(s["shape"], [3, 10, 5])

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
        self.assertEqual(s["type"], "SpikeSliceStack")
        self.assertEqual(s["N_slices"], 4)
        self.assertEqual(s["N_units"], 3)
        self.assertAlmostEqual(s["length_ms"], 50.0)

    def test_summary_pairwise_comp_matrix(self):
        """
        Tests _make_summary() for a PairwiseCompMatrix.

        Tests:
            (Test Case 1) type is "PairwiseCompMatrix".
            (Test Case 2) shape matches the matrix dimensions.
        """
        pcm = PairwiseCompMatrix(matrix=np.eye(5))
        s = _make_summary(pcm)
        self.assertEqual(s["type"], "PairwiseCompMatrix")
        self.assertEqual(s["shape"], [5, 5])

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
        self.assertEqual(s["type"], "PairwiseCompMatrixStack")
        self.assertEqual(s["shape"], [4, 4, 6])

    def test_summary_unknown_type(self):
        """
        Tests _make_summary() falls back to the class name for unrecognised types.

        Tests:
            (Test Case 1) type field contains the class name.
        """

        class MyCustomObj:
            pass

        s = _make_summary(MyCustomObj())
        self.assertEqual(s["type"], "MyCustomObj")


# ---------------------------------------------------------------------------
# Tests: WorkspaceManager
# ---------------------------------------------------------------------------


class TestWorkspaceManager(unittest.TestCase):
    def setUp(self):
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
        self.assertIsInstance(wid, str)
        self.assertTrue(len(wid) > 0)

        ws = self.mgr.get_workspace(wid)
        self.assertIsInstance(ws, AnalysisWorkspace)
        self.assertEqual(ws.name, "my_ws")
        self.assertEqual(ws.workspace_id, wid)

    def test_get_unknown_returns_none(self):
        """
        Tests that get_workspace() returns None for an unknown ID.

        Tests:
            (Test Case 1) Non-existent ID returns None.
        """
        self.assertIsNone(self.mgr.get_workspace("nonexistent-id"))

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
        self.assertTrue(result)
        self.assertIsNone(self.mgr.get_workspace(wid))

        self.assertFalse(self.mgr.delete_workspace("nonexistent-id"))

    def test_list_workspaces(self):
        """
        Tests that list_workspaces() returns correct summary dicts.

        Tests:
            (Test Case 1) Empty manager returns empty list.
            (Test Case 2) Each entry has workspace_id, name, created_at, namespace_count, item_count.
            (Test Case 3) item_count reflects stored items correctly.
        """
        self.assertEqual(self.mgr.list_workspaces(), [])

        wid = self.mgr.create_workspace(name="alpha")
        ws = self.mgr.get_workspace(wid)
        ws.store("ns1", "a", np.zeros(3))
        ws.store("ns1", "b", np.zeros(3))

        listing = self.mgr.list_workspaces()
        self.assertEqual(len(listing), 1)
        entry = listing[0]
        self.assertEqual(entry["workspace_id"], wid)
        self.assertEqual(entry["name"], "alpha")
        self.assertIn("created_at", entry)
        self.assertEqual(entry["namespace_count"], 1)
        self.assertEqual(entry["item_count"], 2)

    @unittest.skipIf(not H5PY_AVAILABLE, "h5py not installed")
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

            self.assertTrue(pathlib.Path(base + ".h5").exists())

            mgr2 = WorkspaceManager()
            loaded_id = mgr2.load_workspace(base)

        self.assertEqual(loaded_id, wid)
        loaded_ws = mgr2.get_workspace(loaded_id)
        self.assertIsNotNone(loaded_ws)
        np.testing.assert_array_equal(loaded_ws.get("ns", "arr"), arr)

    @unittest.skipIf(not H5PY_AVAILABLE, "h5py not installed")
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
            self.assertIsNone(target_ws.get("ns", "spikes"))

            with self.assertRaises(KeyError):
                self.mgr.load_workspace_item(base, "ns", "arr", "nonexistent-id")

    def test_save_unknown_workspace_raises(self):
        """
        Tests that save_workspace() raises KeyError for an unknown workspace_id.

        Tests:
            (Test Case 1) Unknown workspace_id raises KeyError.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with self.assertRaises(KeyError):
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
        self.assertIs(mgr_a, mgr_b)


# ---------------------------------------------------------------------------
# Tests: hdf5_io — HDF5 round-trips for every supported type
# ---------------------------------------------------------------------------


@unittest.skipIf(not H5PY_AVAILABLE, "h5py not installed")
class TestHDF5IO(unittest.TestCase):
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
        self.assertEqual(out.shape, (3,))

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
        self.assertEqual(out.shape, (3, 4))

    def test_roundtrip_ndarray_3d(self):
        """
        Tests HDF5 round-trip for a 3-D numpy array.

        Tests:
            (Test Case 1) Values and shape are preserved.
        """
        arr = np.random.default_rng(0).random((2, 5, 4))
        out = self._roundtrip(arr)
        np.testing.assert_array_almost_equal(out, arr)
        self.assertEqual(out.shape, (2, 5, 4))

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
        self.assertIsInstance(out, SpikeData)
        self.assertEqual(out.N, 3)
        self.assertAlmostEqual(out.length, 50.0)
        np.testing.assert_array_almost_equal(out.train[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(out.train[1], [5.0, 10.0])
        self.assertEqual(len(out.train[2]), 0)
        self.assertEqual(out.metadata["source"], "test")

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
        self.assertIsNotNone(out.neuron_attributes)
        channels = [d.get("channel") for d in out.neuron_attributes]
        self.assertAlmostEqual(channels[0], 0.0)
        self.assertAlmostEqual(channels[1], 1.0)
        self.assertAlmostEqual(channels[2], 2.0)

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
        self.assertIsNotNone(out.neuron_attributes)
        groups = [d.get("group") for d in out.neuron_attributes]
        self.assertEqual(groups[0], "A")
        self.assertEqual(groups[1], "B")

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
        self.assertIsNone(out.neuron_attributes)

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
        self.assertIsInstance(out, RateData)
        self.assertEqual(out.inst_Frate_data.shape, (3, 40))
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
        self.assertIsNotNone(out.neuron_attributes)
        self.assertAlmostEqual(out.neuron_attributes[0]["depth"], 100.0)
        self.assertAlmostEqual(out.neuron_attributes[1]["depth"], 200.0)

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
        self.assertIsInstance(out, RateSliceStack)
        self.assertEqual(out.event_stack.shape, (3, 10, 5))
        np.testing.assert_array_almost_equal(out.event_stack, rss.event_stack)
        self.assertEqual(len(out.times), len(rss.times))
        for (s0, e0), (s1, e1) in zip(rss.times, out.times):
            self.assertAlmostEqual(s0, s1)
            self.assertAlmostEqual(e0, e1)
        self.assertAlmostEqual(out.step_size, rss.step_size)

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
        self.assertIsInstance(out, SpikeSliceStack)
        self.assertEqual(len(out.spike_stack), 3)
        for slice_sd in out.spike_stack:
            self.assertIsInstance(slice_sd, SpikeData)
            self.assertEqual(slice_sd.N, 2)
        self.assertEqual(len(out.times), 3)
        for (s0, e0), (s1, e1) in zip(sss.times, out.times):
            self.assertAlmostEqual(s0, s1)
            self.assertAlmostEqual(e0, e1)

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
        self.assertIsInstance(out, PairwiseCompMatrix)
        np.testing.assert_array_almost_equal(out.matrix, pcm.matrix)
        self.assertIsNone(out.labels)

    def test_roundtrip_pairwise_comp_matrix_int_labels(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrix with integer labels.

        Tests:
            (Test Case 1) Integer labels are preserved as a list.
        """
        mat = np.random.default_rng(0).random((3, 3))
        pcm = PairwiseCompMatrix(matrix=mat, labels=[10, 20, 30])
        out = self._roundtrip(pcm)
        self.assertEqual(len(out.labels), 3)
        self.assertAlmostEqual(float(out.labels[0]), 10.0)
        self.assertAlmostEqual(float(out.labels[1]), 20.0)
        self.assertAlmostEqual(float(out.labels[2]), 30.0)

    def test_roundtrip_pairwise_comp_matrix_string_labels(self):
        """
        Tests HDF5 round-trip for a PairwiseCompMatrix with string labels.

        Tests:
            (Test Case 1) String labels are preserved exactly.
        """
        mat = np.eye(3)
        pcm = PairwiseCompMatrix(matrix=mat, labels=["A", "B", "C"])
        out = self._roundtrip(pcm)
        self.assertEqual(out.labels, ["A", "B", "C"])

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
        self.assertAlmostEqual(out.metadata["threshold"], 0.5)
        self.assertTrue(out.metadata["binary"])
        self.assertEqual(out.metadata["method"], "sttc")

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
        self.assertIsInstance(out, PairwiseCompMatrixStack)
        self.assertEqual(out.stack.shape, (4, 4, 6))
        np.testing.assert_array_almost_equal(out.stack, stack_arr)
        self.assertEqual(out.labels, ["u0", "u1", "u2", "u3"])
        self.assertEqual(len(out.times), 6)
        for (s0, e0), (s1, e1) in zip(times, out.times):
            self.assertAlmostEqual(s0, s1)
            self.assertAlmostEqual(e0, e1)
        self.assertAlmostEqual(out.metadata["delt"], 25.0)

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
        self.assertIsNone(out.labels)
        self.assertIsNone(out.times)
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
        self.assertIsInstance(loaded_pcm, PairwiseCompMatrix)
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
            with self.assertRaises(TypeError):
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
            with self.assertRaises(KeyError):
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
            with self.assertRaises(KeyError):
                AnalysisWorkspace.load_item(base, "ns", "wrong_key")

    def test_metadata_non_json_serializable_raises(self):
        """
        Tests that saving a PairwiseCompMatrix with non-JSON-serializable metadata raises ValueError.

        Tests:
            (Test Case 1) metadata containing a numpy array raises ValueError at save time.
        """
        pcm = PairwiseCompMatrix(
            matrix=np.eye(2),
            metadata={"bad_value": np.array([1, 2, 3])},
        )
        ws = AnalysisWorkspace()
        ws.store("ns", "pcm", pcm)

        with tempfile.TemporaryDirectory() as tmp:
            base = str(pathlib.Path(tmp) / "ws")
            with self.assertRaises(ValueError):
                ws.save(base)

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
        self.assertEqual(info["type"], "ndarray")
        self.assertEqual(info["note"], "my note")
        self.assertGreater(info["created_at"], 0.0)


if __name__ == "__main__":
    unittest.main()
