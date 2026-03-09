"""
Tests for data_loaders -> SpikeData conversion.

These tests use small temporary files and skip format-specific tests
if optional dependencies are not available (e.g., h5py).
"""

from __future__ import annotations

import os
import pickle
import tempfile
import unittest
from typing import Optional
from unittest.mock import patch

import numpy as np

try:  # optional, only needed for HDF5/NWB tests
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

import pathlib
import sys

# Ensure project root is on sys.path, mirroring other tests
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikedata import SpikeData
import data_loaders.data_loaders as loaders


@unittest.skipIf(h5py is None, "h5py not installed; skipping HDF5/NWB tests")
class TestHDF5Loaders(unittest.TestCase):
    def _tmp_h5(self) -> str:
        """Create a temporary HDF5 file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Remove any temporary HDF5 files created during the tests."""
        for attr in ("_last_h5_path",):
            path: Optional[str] = getattr(self, attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_hdf5_raster(self):
        """
        Test loading a 2D raster dataset from HDF5.

        Tests:
        (Method 1)  Creates a small 2D integer array and writes it as 'raster' to HDF5
        (Method 2)  Loads it using load_spikedata_from_hdf5 with raster_bin_size_ms=10.0
        (Test Case 1)  Checks that the resulting SpikeData object has the correct raster and unit count.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        raster = np.array([[0, 2, 0, 1], [1, 0, 0, 0]], dtype=int)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        self.assertIsInstance(sd, SpikeData)
        self.assertTrue(np.all(sd.raster(10.0) == raster))
        self.assertEqual(sd.N, raster.shape[0])

    def test_hdf5_raster_not_2d_raises(self):
        """
        Test that loading a non-2D raster dataset raises ValueError.

        Tests:
        (Method 1)  Writes a 1D array as 'raster'
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to incorrect shape.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=10.0
            )

    def test_hdf5_multiple_styles_raises(self):
        """
        Test that specifying multiple input styles raises ValueError.

        Tests:
        (Method 1)  Writes both a 'raster' dataset and a 'units' group
        (Method 2)  Attempts to load with both raster and group_per_unit arguments
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to multiple styles.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.zeros((1, 2)))
            f.create_group("units")
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
                group_per_unit="units",
            )

    def test_hdf5_idces_times_ms(self):
        """
        Test loading spike indices and times in milliseconds from HDF5.

        Tests:
        (Method 1)  Writes 'idces' and 'times' datasets
        (Method 2)  Loads them using load_spikedata_from_hdf5
        (Test Case 1)  Checks that the idces_times method returns the correct indices and times.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        idces = np.array([0, 1, 0, 1], dtype=int)
        times_ms = np.array([5.0, 10.0, 15.0, 20.0])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_ms)

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        id2, t2 = sd.idces_times()
        self.assertTrue(np.all(id2 == idces))
        self.assertTrue(np.allclose(t2, times_ms))

    def test_hdf5_group_per_unit_seconds(self):
        """
        Test loading spike times from a group-per-unit structure in seconds.

        Tests:
        (Method 1)  Writes 'units' group with two datasets (one per unit) containing spike times in seconds
        (Method 2)  Loads them using load_spikedata_from_hdf5 with group_time_unit="s"
        (Test Case 1)  Checks that the resulting SpikeData object has the correct times in milliseconds.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([0.1, 0.2]))
            g.create_dataset("1", data=np.array([0.05]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="s"
        )
        # Expect ms
        self.assertTrue(np.allclose(sd.train[0], np.array([100.0, 200.0])))
        self.assertTrue(np.allclose(sd.train[1], np.array([50.0])))

    def test_hdf5_group_per_unit_empty_units(self):
        """
        Test loading group-per-unit structure with empty units.

        Tests:
        (Method 1)  Writes 'units' group with two empty datasets
        (Method 2)  Loads them using load_spikedata_from_hdf5 with group_time_unit="ms"
        (Test Case 1)  Checks that the resulting SpikeData object has two units,
        (Test Case 2)  Checks that the length method returns 0.0
        (Test Case 3)  Checks that the train[0] is an empty list
        (Test Case 4)  Checks that the train[1] is an empty list
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([]))
            g.create_dataset("1", data=np.array([]))
        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        self.assertEqual(sd.N, 2)
        self.assertEqual(sd.length, 0.0)
        self.assertEqual(len(sd.train[0]), 0)
        self.assertEqual(len(sd.train[1]), 0)

    def test_hdf5_flat_ragged_spike_times(self):
        """
        Test loading ragged spike times from flat arrays and index.

        Tests:
        (Method 1)  Writes a flat 'spike_times' array and a 'spike_times_index' array
        (Method 2)  Loads them using load_spikedata_from_hdf5 with spike_times_unit="s"
        (Test Case 1)  Checks that the train[0] is [100.0, 200.0]
        (Test Case 2)  Checks that the train[1] is [500.0]
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        # two units: [0.1,0.2], [0.5]
        flat = np.array([0.1, 0.2, 0.5])
        index = np.array([2, 3])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("spike_times", data=flat)
            f.create_dataset("spike_times_index", data=index)

        sd = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="s",
        )
        self.assertTrue(np.allclose(sd.train[0], [100.0, 200.0]))
        self.assertTrue(np.allclose(sd.train[1], [500.0]))

    def test_hdf5_idces_times_samples_with_fs(self):
        """
        Test loading spike indices and times in samples with specified sampling rate.

        Tests:
        (Method 1)  Writes 'idces' and 'times' datasets (times in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with times_unit="samples" and fs_Hz=1000.0
        (Test Cases 1-2)  Checks that the idces_times method returns the correct indices and times.
        train[0] and train[1] are the correct spike times in milliseconds.

        """
        path = self._tmp_h5()
        self._last_h5_path = path
        idces = np.array([0, 1, 0], dtype=int)
        times_samp = np.array([100, 200, 300])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_samp)
        sd = loaders.load_spikedata_from_hdf5(
            path,
            idces_dataset="idces",
            times_dataset="times",
            times_unit="samples",
            fs_Hz=1000.0,
        )
        # samples @1kHz => ms equal to samples
        self.assertTrue(np.allclose(sd.train[0], [100.0, 300.0]))
        self.assertTrue(np.allclose(sd.train[1], [200.0]))

    def test_hdf5_raw_attachment_seconds_and_samples(self):
        """
        Test loading and attaching raw data and raw time from HDF5.

        Tests:
        (Method 1)  Writes 'raster', 'raw', and two raw time datasets (one in seconds, one in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with raw_time_unit="s" and raw_time_unit="samples"
        (Test Case 1)  Checks that the raw_data.shape is (2, 5)
        (Test Case 2)  Checks that the raw_time is [0.0, 0.001, 0.002, 0.003, 0.004] from the seconds dataset
        (Test Case 3)  Checks that the raw_time is [0.0, 1.0, 2.0, 3.0, 4.0] from the samples dataset
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        raster = np.zeros((1, 3))
        raw = np.random.randn(2, 5)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)
            f.create_dataset("raw", data=raw)
            f.create_dataset("raw_time_s", data=np.arange(5) * 0.001)
            f.create_dataset("raw_time_samples", data=np.arange(5))

        # seconds path
        sd_s = loaders.load_spikedata_from_hdf5(
            path,
            raster_dataset="raster",
            raster_bin_size_ms=1.0,
            raw_dataset="raw",
            raw_time_dataset="raw_time_s",
            raw_time_unit="s",
        )
        self.assertEqual(sd_s.raw_data.shape, (2, 5))
        self.assertTrue(np.allclose(sd_s.raw_time, np.arange(5) * 1.0))

        # samples path
        sd_p = loaders.load_spikedata_from_hdf5(
            path,
            raster_dataset="raster",
            raster_bin_size_ms=1.0,
            raw_dataset="raw",
            raw_time_dataset="raw_time_samples",
            raw_time_unit="samples",
            fs_Hz=1000.0,
        )
        self.assertTrue(np.allclose(sd_p.raw_time, np.arange(5) * 1.0))

    def test_hdf5_invalid_style_error(self):
        """
        Test that loading from an HDF5 file with no recognizable style raises ValueError.

        Tests:
        (Method 1)  Writes an empty HDF5 file
        (Method 2)  Loads it using load_spikedata_from_hdf5 without specifying a style
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to missing required datasets/groups.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(path)  # no style specified

    def test_hdf5_samples_without_fs_error(self):
        """
        Test that loading times in samples without specifying fs_Hz raises ValueError.

        Tests:
        (Method 1)  Writes 'idces' and 'times' (in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with times_unit="samples"
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to missing fs_Hz.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        idces = np.array([0, 0, 1])
        times_samples = np.array([10, 20, 30])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_samples)
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, idces_dataset="idces", times_dataset="times", times_unit="samples"
            )

    def test_hdf5_raw_thresholded(self):
        """
        Test thresholding of raw data loaded from HDF5.

        Tests:
        (Method 1)  Writes a 'raw' dataset with two channels, one containing a supra-threshold segment
        (Method 2)  Loads it using load_spikedata_from_hdf5_raw_thresholded
        (Test Case 1)  Checks that the resulting SpikeData object has the correct number of units
        (Test Case 2)  Checks that the train[0] is not empty
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        # two channels, one has a supra-threshold segment
        data = np.zeros((2, 50), dtype=float)
        data[0, 10:13] = 10.0
        data[1, :] = 0.0
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raw", data=data)

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=2.0,
            filter=False,
            hysteresis=True,
            direction="up",
        )
        self.assertIsInstance(sd, SpikeData)
        self.assertEqual(sd.N, 2)
        # should detect at least one event on channel 0
        self.assertTrue(len(sd.train[0]) >= 1)


@unittest.skipIf(h5py is None, "h5py not installed; skipping NWB tests")
class TestNWBLoader(unittest.TestCase):
    def test_nwb_units_via_h5py(self):
        """
        Test loading NWB units group using h5py.

        Tests:
        (Method 1)  Writes a minimal NWB-like file with a 'units' group containing 'spike_times' and 'spike_times_index'
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that the train[0] is [100.0, 200.0]
        (Test Case 2)  Checks that the train[1] is [500.0]
        """
        fd, path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            # minimal NWB-like units group
            with h5py.File(path, "w") as f:  # type: ignore
                g = f.create_group("units")
                g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5]))
                g.create_dataset("spike_times_index", data=np.array([2, 3]))

            sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
            self.assertTrue(np.allclose(sd.train[0], [100.0, 200.0]))
            self.assertTrue(np.allclose(sd.train[1], [500.0]))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_nwb_missing_units_raises(self):
        """
        Test that loading an NWB file missing the 'units' group raises ValueError.

        Tests:
        (Method 1)  Writes an empty NWB file
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that load_spikedata_from_nwb raises a ValueError due to missing 'units'.
        """
        fd, path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            with h5py.File(path, "w") as _:  # type: ignore
                pass
            with self.assertRaises(ValueError):
                loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_nwb_alt_names_with_endswith(self):
        """
        Test loading NWB units group with alternative dataset names.

        Tests:
        (Method 1)  Writes a 'units' group with datasets ending in 'spike_times' and 'spike_times_index' but with prefixes
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that the train[0] is [200.0]
        (Test Case 2)  Checks that the train[1] is [700.0]
        """
        fd, path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            with h5py.File(path, "w") as f:  # type: ignore
                g = f.create_group("units")
                g.create_dataset("xx_spike_times", data=np.array([0.2, 0.7]))
                g.create_dataset("xx_spike_times_index", data=np.array([1, 2]))

            sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
            self.assertTrue(np.allclose(sd.train[0], [200.0]))
            self.assertTrue(np.allclose(sd.train[1], [700.0]))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class TestKiloSortAndSpikeInterface(unittest.TestCase):
    def test_kilosort_basic(self):
        """
        Test loading KiloSort output with two clusters.

        Tests:
        (Method 1)  Writes 'spike_times.npy' and 'spike_clusters.npy' for two clusters
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the cluster_ids metadata matches the trains
        (Test Case 2)  Checks that the spike times are correctly converted to ms and sorted by cluster id
        """
        with tempfile.TemporaryDirectory() as d:
            # two clusters: 2 spikes in 0, 1 spike in 1
            spike_times = np.array([10, 20, 15])  # samples
            spike_clusters = np.array([0, 0, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            # cluster_ids metadata should align with trains
            self.assertEqual(len(sd.train), len(sd.metadata.get("cluster_ids", [])))
            # Expected times in ms
            all_trains_ms = [np.array([10.0, 20.0]), np.array([15.0])]
            # order by cluster id ascending
            for train, truth in zip(sd.train, all_trains_ms):
                self.assertTrue(np.allclose(train, truth))

    def test_spikeinterface_mock(self):
        """
        Test loading from a mock SpikeInterface SortingExtractor.

        Tests:
        (Method 1)  Writes a mock sorting object with two units and known spike trains
        (Method 2)  Loads it using load_spikedata_from_spikeinterface
        (Test Case 1)  Checks that the train[0] is [10.0, 20.0]
        (Test Case 2)  Checks that the train[1] is [2.5]
        """

        class MockSorting:
            def __init__(self):
                self._ids = [10, 20]

            def get_unit_ids(self):
                return self._ids

            def get_sampling_frequency(self):
                return 2000.0

            def get_unit_spike_train(self, unit_id, segment_index=0):
                if unit_id == 10:
                    return np.array([10, 20])
                return np.array([5])

        sorting = MockSorting()
        sd = loaders.load_spikedata_from_spikeinterface(sorting)
        # samples -> ms at 2kHz => 0.5 ms increments
        self.assertTrue(np.allclose(sd.train[0], [5.0, 10.0]))
        self.assertTrue(np.allclose(sd.train[1], [2.5]))

    def test_spikeinterface_base_recording_thresholding(self):
        """
        Test thresholding on a mock SpikeInterface RecordingExtractor.

        Tests:
        (Method 1)  Writes a mock recording object with a supra-threshold burst on one channel
        (Method 2)  Loads it using load_spikedata_from_spikeinterface_recording
        (Test Case 1)  Checks that the resulting SpikeData object has the correct number of units
        (Test Case 2)  Checks that at least one event is detected on the active channel
        (Test Case 3)  Checks that the time x channels input is transposed automatically
        (Test Case 4)  Checks that at least one event is detected on the active channel post transposition

        """

        class MockRecording:
            def __init__(self, data, fs):
                self._data = np.asarray(data)
                self.sampling_frequency = fs

            def get_traces(self, segment_index=0):
                return self._data

            def get_num_channels(self):
                # channels is first dim if 2D
                return self._data.shape[0]

        # channels x time with a clear supra-threshold burst on ch0
        data_ct = np.zeros((2, 100))
        data_ct[0, 50:55] = 10.0
        rec = MockRecording(data_ct, fs=1000.0)
        sd = loaders.load_spikedata_from_spikeinterface_recording(
            rec, threshold_sigma=2.0, filter=False, hysteresis=True, direction="up"
        )
        self.assertEqual(sd.N, 2)
        self.assertTrue(len(sd.train[0]) >= 1)

        # time x channels input gets transposed automatically
        data_tc = data_ct.T
        rec2 = MockRecording(data_tc, fs=1000.0)
        sd2 = loaders.load_spikedata_from_spikeinterface_recording(
            rec2, threshold_sigma=2.0, filter=False, hysteresis=True, direction="up"
        )
        self.assertEqual(sd2.N, 2)
        self.assertTrue(len(sd2.train[0]) >= 1)

    def test_spikeinterface_subset_and_override_fs(self):
        """
        Test loading a subset of units and overriding sampling frequency.

        Tests:
        (Method 1)  Writes a mock sorting object with two units and no sampling frequency
        (Method 2)  Loads it using load_spikedata_from_spikeinterface
        (Test Case 1)  Checks that the resulting SpikeData object has the correct number of units
        (Test Case 2)  Checks that the train[0] is [0.0, 10.0]
        """

        class MockSorting2:
            def get_unit_ids(self):
                return [1, 2]

            def get_sampling_frequency(self):
                return None

            def get_unit_spike_train(self, unit_id, segment_index=0):
                return np.array([0, 10])

        sd = loaders.load_spikedata_from_spikeinterface(
            MockSorting2(), unit_ids=[2], sampling_frequency=1000.0
        )
        # Only unit 2, times in ms equal to samples at 1kHz
        self.assertEqual(sd.N, 1)
        self.assertTrue(np.allclose(sd.train[0], [0.0, 10.0]))

    def test_spikeinterface_invalid_object_raises(self):
        """
        Test that passing an invalid object to load_spikedata_from_spikeinterface raises TypeError.

        Tests:
        (Method 1)  Writes a class with no required methods
        (Method 2)  Loads it using load_spikedata_from_spikeinterface
        (Test Case 1)  Checks that load_spikedata_from_spikeinterface raises TypeError
        """

        class BadSorting:
            pass

        with self.assertRaises(TypeError):
            loaders.load_spikedata_from_spikeinterface(BadSorting())

    def test_kilosort_empty_arrays(self):
        """
        Test loading KiloSort output with empty arrays.

        Tests:
        (Method 1)  Writes empty 'spike_times.npy' and 'spike_clusters.npy'
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the resulting SpikeData object has zero units
        (Test Case 2)  Checks that the length is 0.0
        """
        with tempfile.TemporaryDirectory() as d:
            np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=int))
            np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            self.assertEqual(sd.N, 0)
            self.assertEqual(sd.length, 0.0)

    def test_kilosort_metadata_cluster_ids_alignment(self):
        """
        Test that KiloSort cluster_ids metadata aligns with sorted trains.

        Tests:
        (Method 1)  Writes 'spike_times.npy' and 'spike_clusters.npy' with two cluster ids
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the cluster_ids metadata is sorted and matches the order of spike trains
        """
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20, 15, 30])
            spike_clusters = np.array([5, 5, 3, 5])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            # cluster_ids sorted ascending (np.unique order)
            self.assertEqual(sd.metadata.get("cluster_ids"), [3, 5])

    def test_kilosort_tsv_missing_columns_keeps_all(self):
        """
        Test that KiloSort loader keeps all clusters if cluster_info.tsv is missing expected columns.

        Tests:
        (Method 1)  Writes 'spike_times.npy', 'spike_clusters.npy', and a cluster_info.tsv file without the expected columns
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that all clusters are kept
        """
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20, 15])
            spike_clusters = np.array([0, 0, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            # Create TSV without expected columns to trigger warning path
            with open(os.path.join(d, "cluster_info.tsv"), "w") as f:
                f.write("foo\tbar\n1\tbaz\n")
            sd = loaders.load_spikedata_from_kilosort(
                d, fs_Hz=1000.0, cluster_info_tsv="cluster_info.tsv"
            )
            # Should keep both clusters 0 and 1
            self.assertEqual(len(sd.train), 2)

    def test_kilosort_channel_positions_location(self):
        """
        Test channel_positions → neuron_attributes["location"] behavior.

        Tests:
        (Method 1)  Writes spike_times.npy, spike_clusters.npy with clusters 0 and 1
        (Method 2)  Writes channel_positions.npy with positions for 4 channels
        (Test Case 1)  With matching channel_map.npy: location comes from channel_map lookup
        (Test Case 2)  Without channel_map.npy: fallback uses unit index
        (Test Case 3)  With mismatching channel_map.npy (out-of-bounds): fallback uses unit index
        (Test Case 4)  Non-sequential cluster IDs: fallback uses unit index, not cluster ID
        """
        with tempfile.TemporaryDirectory() as d:
            # Create basic spike data with clusters 0 and 1
            spike_times = np.array([10, 20, 15, 25])
            spike_clusters = np.array([0, 0, 1, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

            # Channel positions: 4 channels with distinct XYZ coordinates
            channel_positions = np.array(
                [
                    [0.0, 0.0, 0.0],  # channel 0
                    [10.0, 20.0, 0.0],  # channel 1
                    [20.0, 40.0, 0.0],  # channel 2
                    [30.0, 60.0, 0.0],  # channel 3
                ]
            )
            np.save(os.path.join(d, "channel_positions.npy"), channel_positions)

            # Test Case 1: With channel_map that maps cluster 0 → channel 2, cluster 1 → channel 3
            channel_map = np.array([2, 3])  # cluster index → channel number
            np.save(os.path.join(d, "channel_map.npy"), channel_map)

            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

            # Cluster 0 maps to channel 2 → position [20.0, 40.0, 0.0]
            # Cluster 1 maps to channel 3 → position [30.0, 60.0, 0.0]
            self.assertEqual(sd.neuron_attributes[0]["location"], [20.0, 40.0, 0.0])
            self.assertEqual(sd.neuron_attributes[1]["location"], [30.0, 60.0, 0.0])
            self.assertEqual(sd.neuron_attributes[0]["electrode"], 2)
            self.assertEqual(sd.neuron_attributes[1]["electrode"], 3)

        # Test Case 2: Without channel_map.npy - fallback to unit index
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20, 15, 25])
            spike_clusters = np.array([0, 0, 1, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            np.save(os.path.join(d, "channel_positions.npy"), channel_positions)
            # No channel_map.npy file

            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

            # Fallback: unit 0 → position[0], unit 1 → position[1]
            self.assertEqual(sd.neuron_attributes[0]["location"], [0.0, 0.0, 0.0])
            self.assertEqual(sd.neuron_attributes[1]["location"], [10.0, 20.0, 0.0])
            # No electrode attribute when channel_map is missing
            self.assertNotIn("electrode", sd.neuron_attributes[0])
            self.assertNotIn("electrode", sd.neuron_attributes[1])

        # Test Case 3: channel_map exists but maps to out-of-bounds channel index
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20, 15, 25])
            spike_clusters = np.array([0, 0, 1, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            np.save(os.path.join(d, "channel_positions.npy"), channel_positions)
            # channel_map maps to channels that exceed channel_positions length
            channel_map_oob = np.array([10, 20])  # both out of bounds (>= 4)
            np.save(os.path.join(d, "channel_map.npy"), channel_map_oob)

            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

            # Fallback: unit index used since channel_map values are out of bounds
            self.assertEqual(sd.neuron_attributes[0]["location"], [0.0, 0.0, 0.0])
            self.assertEqual(sd.neuron_attributes[1]["location"], [10.0, 20.0, 0.0])
            # electrode attribute still set from channel_map (even if out of bounds for positions)
            self.assertEqual(sd.neuron_attributes[0]["electrode"], 10)
            self.assertEqual(sd.neuron_attributes[1]["electrode"], 20)

        # Test Case 4: Non-sequential cluster IDs - fallback uses unit index, not cluster ID
        with tempfile.TemporaryDirectory() as d:
            # Clusters 50 and 100 - IDs that would be out of bounds if used directly
            spike_times = np.array([10, 20, 15, 25])
            spike_clusters = np.array([50, 50, 100, 100])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            np.save(os.path.join(d, "channel_positions.npy"), channel_positions)
            # No channel_map.npy file

            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

            # Fallback uses unit index (0, 1), not cluster ID (50, 100)
            # Unit 0 (cluster 50) → position[0], Unit 1 (cluster 100) → position[1]
            self.assertEqual(sd.neuron_attributes[0]["location"], [0.0, 0.0, 0.0])
            self.assertEqual(sd.neuron_attributes[1]["location"], [10.0, 20.0, 0.0])
            self.assertEqual(sd.neuron_attributes[0]["unit_id"], 50)
            self.assertEqual(sd.neuron_attributes[1]["unit_id"], 100)


class TestPickleLoaders(unittest.TestCase):
    """
    Tests for load_spikedata_from_pickle.

    Tests:
    - Basic pickle loading from local file
    - S3 URL handling via ensure_local_file
    - Validation that non-SpikeData objects raise ValueError
    - Temporary file cleanup when loading from S3
    """

    def _tmp_pkl(self) -> str:
        """Create a temporary pickle file path for testing."""
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Remove any temporary pickle files created during the tests."""
        for attr in ("_last_pkl_path",):
            path: Optional[str] = getattr(self, attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_pickle_basic_load(self):
        """
        Test basic loading of SpikeData from a local pickle file.

        Tests:
        (Method 1) Creates SpikeData, pickles it to a temp file
        (Method 2) Loads using load_spikedata_from_pickle
        (Test Case 1) Loaded object is SpikeData instance
        (Test Case 2) Spike trains match original
        (Test Case 3) Metadata is preserved
        """
        sd = SpikeData(
            [np.array([5.0, 10.0]), np.array([2.5])],
            length=25.0,
            metadata={"label": "test"},
        )
        path = self._tmp_pkl()
        self._last_pkl_path = path
        # Write SpikeData to pickle file
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Load and verify spike trains match
        sd2 = loaders.load_spikedata_from_pickle(path)
        self.assertIsInstance(sd2, SpikeData)
        for a, b in zip(sd.train, sd2.train):
            self.assertTrue(np.allclose(a, b))
        # Verify metadata is preserved
        self.assertEqual(sd.metadata, sd2.metadata)

    @patch("data_loaders.s3_utils.ensure_local_file")
    def test_pickle_s3_url_handling(self, mock_ensure):
        """
        Test that S3 URLs are resolved via ensure_local_file before loading.

        Tests:
        (Method 1) Creates SpikeData pickle in temp file
        (Method 2) Mocks ensure_local_file to return (temp_path, False) for S3 URL
        (Method 3) Calls load_spikedata_from_pickle with s3:// URL
        (Test Case 1) ensure_local_file is called with S3 URL
        (Test Case 2) Loaded SpikeData matches original
        """
        sd = SpikeData(
            [np.array([1.0, 2.0])],
            length=10.0,
            metadata={},
        )
        path = self._tmp_pkl()
        self._last_pkl_path = path
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Mock ensure_local_file to return our temp path (as if S3 was already downloaded)
        mock_ensure.return_value = (path, False)

        # Load via S3 URL; ensure_local_file is mocked so no real S3 call
        sd2 = loaders.load_spikedata_from_pickle("s3://bucket/key.pkl")

        # Verify ensure_local_file was called with S3 URL (and optional cred kwargs)
        mock_ensure.assert_called_once()
        self.assertEqual(mock_ensure.call_args[0][0], "s3://bucket/key.pkl")
        # Verify loaded data matches
        self.assertTrue(np.allclose(sd2.train[0], sd.train[0]))

    def test_pickle_non_spikedata_raises_valueerror(self):
        """
        Test that loading a pickle containing a non-SpikeData object raises ValueError.

        Tests:
        (Method 1) Writes a dict to pickle file (not SpikeData)
        (Method 2) Calls load_spikedata_from_pickle
        (Test Case 1) ValueError is raised with message about wrong type
        """
        path = self._tmp_pkl()
        self._last_pkl_path = path
        # Write non-SpikeData object (dict) to pickle
        with open(path, "wb") as f:
            pickle.dump({"foo": "bar"}, f)

        # Expect ValueError because pickle does not contain SpikeData
        with self.assertRaises(ValueError) as ctx:
            loaders.load_spikedata_from_pickle(path)
        self.assertIn("SpikeData", str(ctx.exception))
        self.assertIn("dict", str(ctx.exception))

    @patch("data_loaders.s3_utils.ensure_local_file")
    def test_pickle_temp_file_cleanup(self, mock_ensure):
        """
        Test that temporary file from S3 download is removed after loading.

        Tests:
        (Method 1) Creates SpikeData pickle in temp file
        (Method 2) Mocks ensure_local_file to return (temp_path, True) so loader treats it as temp
        (Method 3) Loads via S3 URL
        (Test Case 1) Temp file is removed after load completes
        """
        sd = SpikeData(
            [np.array([1.0])],
            length=5.0,
            metadata={},
        )
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Mock ensure_local_file to return our path with is_temp=True
        mock_ensure.return_value = (path, True)

        # Load; loader should remove temp file in finally block
        loaders.load_spikedata_from_pickle("s3://bucket/key.pkl")

        # Verify temp file was removed
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
