"""
Tests for data_loaders -> SpikeData conversion.

These tests use small temporary files and skip format-specific tests
if optional dependencies are not available (e.g., h5py).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Optional

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

        Parameters:
        raster (np.ndarray): a 2D integer array of shape (units, time bins)

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

        Parameters:
        raster (np.ndarray): a 1D array

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

        Parameters:
        raster (np.ndarray): a 2D integer array of shape (units, time bins)

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

        Parameters:
        idces (np.ndarray): a 1D integer array of spike indices
        times_ms (np.ndarray): a 1D float array of spike times in milliseconds

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

        Parameters:
        units (np.ndarray): a 1D float array of spike times in seconds

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

        Parameters:
        units (np.ndarray): an empty 1D float array of spike times in milliseconds

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

        Parameters:
        flat (np.ndarray): a 1D float array of spike times
        index (np.ndarray): a 1D integer array of spike indices

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

        Parameters:
        idces (np.ndarray): a 1D integer array of spike indices
        times_samp (np.ndarray): a 1D float array of spike times in samples

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

        Parameters:
        raster (np.ndarray): a 2D integer array of shape (units, time bins)
        raw (np.ndarray): a 2D float array of shape (channels, time bins)
        raw_time_s (np.ndarray): a 1D float array of raw time in seconds
        raw_time_samples (np.ndarray): a 1D integer array of raw time in samples

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

        Parameters:
        path (str): the path to the HDF5 file

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

        Parameters:
        path (str): the path to the HDF5 file
        idces (np.ndarray): a 1D integer array of spike indices
        times_samples (np.ndarray): a 1D float array of spike times in samples

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

        Parameters:
        path (str): the path to the HDF5 file
        data (np.ndarray): a 2D float array of shape (channels, time bins)

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

        Parameters:
        h5py file object: a file object opened in write mode containing:
            a 'units' group
            a 'spike_times' (np.ndarray): a 1D float array of spike times in seconds
            a 'spike_times_index' (np.ndarray): a 1D integer array of spike indices

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

        Parameters:
        h5py file object: an empty HDF5 file opened in write mode

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

        Parameters:
        h5py file object: a file object opened in write mode containing:
            a 'units' group
            a 'xx_spike_times' (np.ndarray): a 1D float array of spike times in seconds
            a 'xx_spike_times_index' (np.ndarray): a 1D integer array of spike indices

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

        Parameters:
        spike_times (np.ndarray): a 1D float array of spike times in samples
        spike_clusters (np.ndarray): a 1D integer array of spike clusters

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

        Parameters:
        sorting (MockSorting): a mock sorting object with two units and known spike trains

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

        Parameters:
        data (np.ndarray): a 2D float array of shape (channels, time bins)
        fs (float): the sampling frequency in Hz

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

        Parameters:
        sorting (MockSorting2): a mock sorting object with two units and no sampling frequency

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

        Parameters:
        BadSorting: a class with no required methods

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

        Parameters:
        d (str): the path to the temporary kilosort directory containing:
            spike_times (np.ndarray): a 1D integer array of spike times
            spike_clusters (np.ndarray): a 1D integer array of spike clusters

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

        Parameters:
        d (str): the path to the temporary kilosort directory containing:
            spike_times (np.ndarray): a 1D integer array of spike times
            spike_clusters (np.ndarray): a 1D integer array of spike clusters

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

        Parameters:
        d (str): the path to the temporary kilosort directory containing:
            spike_times (np.ndarray): a 1D integer array of spike times
            spike_clusters (np.ndarray): a 1D integer array of spike clusters
            cluster_info.tsv (str): a TSV file with cluster metadata

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


if __name__ == "__main__":
    unittest.main()
