"""
Tests for data_loaders -> SpikeData conversion.

These tests use small temporary files and skip format-specific tests
if optional dependencies are not available (e.g., h5py).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

try:  # optional, only needed for HDF5/NWB tests
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

try:  # optional, only needed for IBL tests
    import pandas as pd  # type: ignore  # noqa: F401

    pandas_available = True
except Exception:  # pragma: no cover
    pandas_available = False

import pathlib
import sys

# Ensure project root is on sys.path, mirroring other tests
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata import SpikeData
import SpikeLab.data_loaders.data_loaders as loaders

skip_no_h5py = pytest.mark.skipif(
    h5py is None, reason="h5py not installed; skipping HDF5/NWB tests"
)

skip_no_pandas = pytest.mark.skipif(
    not pandas_available, reason="pandas not installed; skipping IBL tests"
)


@skip_no_h5py
class TestHDF5Loaders:
    """Tests for loading SpikeData from HDF5 files across all supported styles."""

    def test_hdf5_raster(self, tmp_path):
        """
        Test loading a 2D raster dataset from HDF5.

        Tests:
        (Method 1)  Creates a small 2D integer array and writes it as 'raster' to HDF5
        (Method 2)  Loads it using load_spikedata_from_hdf5 with raster_bin_size_ms=10.0
        (Test Case 1)  Checks that the resulting SpikeData object has the correct raster and unit count.
        """
        path = str(tmp_path / "test.h5")
        raster = np.array([[0, 2, 0, 1], [1, 0, 0, 0]], dtype=int)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        assert isinstance(sd, SpikeData)
        assert np.all(sd.raster(10.0) == raster)
        assert sd.N == raster.shape[0]

    def test_hdf5_raster_not_2d_raises(self, tmp_path):
        """
        Test that loading a non-2D raster dataset raises ValueError.

        Tests:
        (Method 1)  Writes a 1D array as 'raster'
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to incorrect shape.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=1.0
            )

    def test_hdf5_multiple_styles_raises(self, tmp_path):
        """
        Test that specifying multiple input styles raises ValueError.

        Tests:
        (Method 1)  Writes both a 'raster' dataset and a 'units' group
        (Method 2)  Attempts to load with both raster and group_per_unit arguments
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to multiple styles.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.zeros((1, 2)))
            f.create_group("units")
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
                group_per_unit="units",
            )

    def test_hdf5_idces_times_ms(self, tmp_path):
        """
        Test loading spike indices and times in milliseconds from HDF5.

        Tests:
        (Method 1)  Writes 'idces' and 'times' datasets
        (Method 2)  Loads them using load_spikedata_from_hdf5
        (Test Case 1)  Checks that the idces_times method returns the correct indices and times.
        """
        path = str(tmp_path / "test.h5")
        idces = np.array([0, 1, 0, 1], dtype=int)
        times_ms = np.array([5.0, 10.0, 15.0, 20.0])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_ms)

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        loaded_idces, loaded_times = sd.idces_times()
        assert np.allclose(loaded_times, times_ms)

    def test_hdf5_group_per_unit_seconds(self, tmp_path):
        """
        Test loading group-per-unit HDF5 with times in seconds.

        Tests:
        (Method 1)  Writes 'units' group with two datasets (one per unit) containing spike times in seconds
        (Method 2)  Loads them using load_spikedata_from_hdf5 with group_time_unit="s"
        (Test Case 1)  Checks that the resulting SpikeData object has the correct times in milliseconds.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([0.1, 0.2]))
            g.create_dataset("1", data=np.array([0.05]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="s"
        )
        # Expect ms
        assert np.allclose(sd.train[0], np.array([100.0, 200.0]))
        assert np.allclose(sd.train[1], np.array([50.0]))

    def test_hdf5_group_per_unit_empty_units(self, tmp_path):
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
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([]))
            g.create_dataset("1", data=np.array([]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        assert sd.N == 2
        assert sd.length == 0.0
        assert len(sd.train[0]) == 0
        assert len(sd.train[1]) == 0

    def test_hdf5_ragged_spike_times(self, tmp_path):
        """
        Test loading flat (ragged) spike_times with cumulative index in seconds.

        Tests:
        (Method 1)  Writes a flat 'spike_times' array and a 'spike_times_index' array
        (Method 2)  Loads them using load_spikedata_from_hdf5 with spike_times_unit="s"
        (Test Case 1)  Checks that the train[0] is [100.0, 200.0]
        (Test Case 2)  Checks that the train[1] is [500.0]
        """
        path = str(tmp_path / "test.h5")
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
        assert np.allclose(sd.train[0], [100.0, 200.0])
        assert np.allclose(sd.train[1], [500.0])

    def test_hdf5_idces_times_samples_with_fs(self, tmp_path):
        """
        Test loading spike indices and times in samples with specified sampling rate.

        Tests:
        (Method 1)  Writes 'idces' and 'times' datasets (times in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with times_unit="samples" and fs_Hz=1000.0
        (Test Cases 1-2)  Checks that the idces_times method returns the correct indices and times.
        train[0] and train[1] are the correct spike times in milliseconds.

        """
        path = str(tmp_path / "test.h5")
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
        assert np.allclose(sd.train[0], [100.0, 300.0])
        assert np.allclose(sd.train[1], [200.0])

    def test_hdf5_raw_attachment_seconds_and_samples(self, tmp_path):
        """
        Test loading and attaching raw data and raw time from HDF5.

        Tests:
        (Method 1)  Writes 'raster', 'raw', and two raw time datasets (one in seconds, one in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with raw_time_unit="s" and raw_time_unit="samples"
        (Test Case 1)  Checks that the raw_data.shape is (2, 5)
        (Test Case 2)  Checks that the raw_time is [0.0, 0.001, 0.002, 0.003, 0.004] from the seconds dataset
        (Test Case 3)  Checks that the raw_time is [0.0, 1.0, 2.0, 3.0, 4.0] from the samples dataset
        """
        path = str(tmp_path / "test.h5")
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
        assert sd_s.raw_data.shape == (2, 5)
        assert np.allclose(sd_s.raw_time, np.arange(5) * 1.0)

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
        assert np.allclose(sd_p.raw_time, np.arange(5) * 1.0)

    def test_hdf5_no_style_raises(self, tmp_path):
        """
        Test that loading an HDF5 file without specifying a style raises ValueError.

        Tests:
        (Method 1)  Writes an empty HDF5 file
        (Method 2)  Loads it using load_spikedata_from_hdf5 without specifying a style
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to missing required datasets/groups.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(path)  # no style specified

    def test_hdf5_samples_without_fs_error(self, tmp_path):
        """
        Test that loading times in samples without specifying fs_Hz raises ValueError.

        Tests:
        (Method 1)  Writes 'idces' and 'times' (in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with times_unit="samples"
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to missing fs_Hz.
        """
        path = str(tmp_path / "test.h5")
        idces = np.array([0, 0, 1])
        times_samples = np.array([10, 20, 30])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_samples)
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, idces_dataset="idces", times_dataset="times", times_unit="samples"
            )

    def test_hdf5_raw_thresholded(self, tmp_path):
        """
        Test thresholding of raw data loaded from HDF5.

        Tests:
        (Method 1)  Writes a 'raw' dataset with two channels, one containing a supra-threshold segment
        (Method 2)  Loads it using load_spikedata_from_hdf5_raw_thresholded
        (Test Case 1)  Checks that the resulting SpikeData object has 2 units
        (Test Case 2)  Checks that at least one event is detected on channel 0
        """
        path = str(tmp_path / "test.h5")
        data = np.zeros((2, 200))
        data[0, 100:105] = 10.0  # supra-threshold burst on ch0
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
        assert isinstance(sd, SpikeData)
        assert sd.N == 2
        # should detect at least one event on channel 0
        assert len(sd.train[0]) >= 1


@skip_no_h5py
class TestNWBLoader:
    """Tests for loading SpikeData from NWB files."""

    def test_nwb_units_via_h5py(self, tmp_path):
        """
        Test loading NWB units group using h5py.

        Tests:
        (Method 1)  Writes a minimal NWB-like file with a 'units' group containing 'spike_times' and 'spike_times_index'
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that the train[0] is [100.0, 200.0]
        (Test Case 2)  Checks that the train[1] is [500.0]
        """
        path = str(tmp_path / "test.nwb")
        # minimal NWB-like units group
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5]))
            g.create_dataset("spike_times_index", data=np.array([2, 3]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert np.allclose(sd.train[0], [100.0, 200.0])
        assert np.allclose(sd.train[1], [500.0])

    def test_nwb_missing_units_raises(self, tmp_path):
        """
        Test that loading an NWB file missing the 'units' group raises ValueError.

        Tests:
        (Method 1)  Writes an empty NWB file
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that load_spikedata_from_nwb raises a ValueError due to missing 'units'.
        """
        path = str(tmp_path / "test.nwb")
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)

    def test_nwb_alt_names_with_endswith(self, tmp_path):
        """
        Test loading NWB units group with alternative dataset names.

        Tests:
        (Method 1)  Writes a 'units' group with datasets ending in 'spike_times' and 'spike_times_index' but with prefixes
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that the train[0] is [200.0]
        (Test Case 2)  Checks that the train[1] is [700.0]
        """
        path = str(tmp_path / "test.nwb")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("xx_spike_times", data=np.array([0.2, 0.7]))
            g.create_dataset("xx_spike_times_index", data=np.array([1, 2]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert np.allclose(sd.train[0], [200.0])
        assert np.allclose(sd.train[1], [700.0])


class TestKiloSortAndSpikeInterface:
    """Tests for KiloSort and SpikeInterface loaders."""

    def test_kilosort_basic_load(self, tmp_path):
        """
        Test loading KiloSort output with two clusters.

        Tests:
        (Method 1)  Writes 'spike_times.npy' and 'spike_clusters.npy' for two clusters
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the cluster_ids metadata matches the trains
        (Test Case 2)  Checks that the spike times are correctly converted to ms and sorted by cluster id
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        # two clusters: 2 spikes in 0, 1 spike in 1
        spike_times = np.array([10, 20, 15])  # samples
        spike_clusters = np.array([0, 0, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        # cluster_ids metadata should align with trains
        assert len(sd.train) == len(sd.metadata.get("cluster_ids", []))
        # Expected times in ms
        all_trains_ms = [np.array([10.0, 20.0]), np.array([15.0])]
        # order by cluster id ascending
        for train, truth in zip(sd.train, all_trains_ms):
            assert np.allclose(train, truth)

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
            def get_unit_ids(self):
                return [0, 1]

            def get_sampling_frequency(self):
                return 2000.0

            def get_unit_spike_train(self, unit_id, segment_index=0):
                if unit_id == 0:
                    return np.array([20, 40])
                return np.array([5])

        sorting = MockSorting()
        sd = loaders.load_spikedata_from_spikeinterface(sorting)
        # samples -> ms at 2kHz => 0.5 ms increments
        assert np.allclose(sd.train[0], [10.0, 20.0])
        assert np.allclose(sd.train[1], [2.5])

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
        assert sd.N == 2
        assert len(sd.train[0]) >= 1

        # time x channels: should auto-transpose
        data_tc = data_ct.T
        rec2 = MockRecording(data_tc, fs=1000.0)
        sd2 = loaders.load_spikedata_from_spikeinterface_recording(
            rec2, threshold_sigma=2.0, filter=False, hysteresis=True, direction="up"
        )
        assert sd2.N == 2
        assert len(sd2.train[0]) >= 1

    def test_spikeinterface_subset_units(self):
        """
        Test loading a subset of units from a mock SpikeInterface SortingExtractor.

        Tests:
        (Method 1)  Loads with unit_ids=[2] from a sorting with units [1, 2]
        (Test Case 1)  Checks that the resulting SpikeData has 1 unit
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
        assert sd.N == 1
        assert np.allclose(sd.train[0], [0.0, 10.0])

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

        with pytest.raises(TypeError):
            loaders.load_spikedata_from_spikeinterface(BadSorting())

    def test_kilosort_empty_arrays(self, tmp_path):
        """
        Test loading KiloSort output with empty arrays.

        Tests:
        (Method 1)  Writes empty 'spike_times.npy' and 'spike_clusters.npy'
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the resulting SpikeData object has zero units
        (Test Case 2)  Checks that the length is 0.0
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=int))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        assert sd.N == 0
        assert sd.length == 0.0

    def test_kilosort_nonsequential_clusters(self, tmp_path):
        """
        Test that KiloSort loader handles non-sequential cluster IDs correctly.

        Tests:
        (Method 1)  Writes spike data with non-sequential cluster IDs [3, 5]
        (Test Case 1)  Checks that the cluster_ids metadata is sorted and matches the order of spike trains
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15, 30])
        spike_clusters = np.array([5, 5, 3, 5])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        # cluster_ids sorted ascending (np.unique order)
        assert sd.metadata.get("cluster_ids") == [3, 5]

    def test_kilosort_tsv_missing_columns_keeps_all(self, tmp_path):
        """
        Test that KiloSort loader keeps all clusters if cluster_info.tsv is missing expected columns.

        Tests:
        (Method 1)  Writes 'spike_times.npy', 'spike_clusters.npy', and a cluster_info.tsv file without the expected columns
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that all clusters are kept
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
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
        assert len(sd.train) == 2

    def test_kilosort_channel_positions_location(self, tmp_path):
        """
        Test channel_positions -> neuron_attributes["location"] behavior.

        Tests:
        (Method 1)  Writes spike_times.npy, spike_clusters.npy with clusters 0 and 1
        (Method 2)  Writes channel_positions.npy with positions for 4 channels
        (Test Case 1)  With matching channel_map.npy: location comes from channel_map lookup
        (Test Case 2)  Without channel_map.npy: fallback uses unit index
        (Test Case 3)  With mismatching channel_map.npy (out-of-bounds): fallback uses unit index
        (Test Case 4)  Non-sequential cluster IDs: fallback uses unit index, not cluster ID
        """
        # Channel positions: 4 channels with distinct XYZ coordinates
        channel_positions = np.array(
            [
                [0.0, 0.0, 0.0],  # channel 0
                [10.0, 20.0, 0.0],  # channel 1
                [20.0, 40.0, 0.0],  # channel 2
                [30.0, 60.0, 0.0],  # channel 3
            ]
        )

        # Test Case 1: With channel_map that maps cluster 0 -> channel 2, cluster 1 -> channel 3
        d = str(tmp_path / "ks1")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15, 25])
        spike_clusters = np.array([0, 0, 1, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        np.save(os.path.join(d, "channel_positions.npy"), channel_positions)
        channel_map = np.array([2, 3])  # cluster index -> channel number
        np.save(os.path.join(d, "channel_map.npy"), channel_map)

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        # Cluster 0 maps to channel 2 -> position [20.0, 40.0, 0.0]
        # Cluster 1 maps to channel 3 -> position [30.0, 60.0, 0.0]
        assert sd.neuron_attributes[0]["location"] == [20.0, 40.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [30.0, 60.0, 0.0]
        assert sd.neuron_attributes[0]["electrode"] == 2
        assert sd.neuron_attributes[1]["electrode"] == 3

        # Test Case 2: Without channel_map.npy - fallback to unit index
        d2 = str(tmp_path / "ks2")
        os.makedirs(d2)
        np.save(os.path.join(d2, "spike_times.npy"), spike_times)
        np.save(os.path.join(d2, "spike_clusters.npy"), spike_clusters)
        np.save(os.path.join(d2, "channel_positions.npy"), channel_positions)
        # No channel_map.npy file

        sd = loaders.load_spikedata_from_kilosort(d2, fs_Hz=1000.0)

        # Fallback: unit 0 -> position[0], unit 1 -> position[1]
        assert sd.neuron_attributes[0]["location"] == [0.0, 0.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [10.0, 20.0, 0.0]
        # No electrode attribute when channel_map is missing
        assert "electrode" not in sd.neuron_attributes[0]
        assert "electrode" not in sd.neuron_attributes[1]

        # Test Case 3: channel_map exists but maps to out-of-bounds channel index
        d3 = str(tmp_path / "ks3")
        os.makedirs(d3)
        np.save(os.path.join(d3, "spike_times.npy"), spike_times)
        np.save(os.path.join(d3, "spike_clusters.npy"), spike_clusters)
        np.save(os.path.join(d3, "channel_positions.npy"), channel_positions)
        channel_map_oob = np.array([10, 20])  # both out of bounds (>= 4)
        np.save(os.path.join(d3, "channel_map.npy"), channel_map_oob)

        sd = loaders.load_spikedata_from_kilosort(d3, fs_Hz=1000.0)

        # Fallback: unit index used since channel_map values are out of bounds
        assert sd.neuron_attributes[0]["location"] == [0.0, 0.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [10.0, 20.0, 0.0]
        # electrode attribute still set from channel_map (even if out of bounds for positions)
        assert sd.neuron_attributes[0]["electrode"] == 10
        assert sd.neuron_attributes[1]["electrode"] == 20

        # Test Case 4: Non-sequential cluster IDs - fallback uses unit index, not cluster ID
        d4 = str(tmp_path / "ks4")
        os.makedirs(d4)
        # Clusters 50 and 100 - IDs that would be out of bounds if used directly
        spike_times4 = np.array([10, 20, 15, 25])
        spike_clusters4 = np.array([50, 50, 100, 100])
        np.save(os.path.join(d4, "spike_times.npy"), spike_times4)
        np.save(os.path.join(d4, "spike_clusters.npy"), spike_clusters4)
        np.save(os.path.join(d4, "channel_positions.npy"), channel_positions)
        # No channel_map.npy file

        sd = loaders.load_spikedata_from_kilosort(d4, fs_Hz=1000.0)

        # Fallback uses unit index (0, 1), not cluster ID (50, 100)
        assert sd.neuron_attributes[0]["location"] == [0.0, 0.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [10.0, 20.0, 0.0]
        assert sd.neuron_attributes[0]["unit_id"] == 50
        assert sd.neuron_attributes[1]["unit_id"] == 100


class TestPickleLoaders:
    """
    Tests for load_spikedata_from_pickle.

    Tests:
    - Basic pickle loading from local file
    - S3 URL handling via ensure_local_file
    - Validation that non-SpikeData objects raise ValueError
    - Temporary file cleanup when loading from S3
    """

    def test_pickle_basic_load(self, tmp_path):
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
        path = str(tmp_path / "test.pkl")
        # Write SpikeData to pickle file
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Load and verify spike trains match
        sd2 = loaders.load_spikedata_from_pickle(path)
        assert isinstance(sd2, SpikeData)
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)
        # Verify metadata is preserved
        assert sd.metadata == sd2.metadata

    @patch("SpikeLab.data_loaders.s3_utils.ensure_local_file")
    def test_pickle_s3_url_handling(self, mock_ensure, tmp_path):
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
        path = str(tmp_path / "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Mock ensure_local_file to return our temp path (as if S3 was already downloaded)
        mock_ensure.return_value = (path, False)

        # Load via S3 URL; ensure_local_file is mocked so no real S3 call
        sd2 = loaders.load_spikedata_from_pickle("s3://bucket/key.pkl")

        # Verify ensure_local_file was called with S3 URL (and optional cred kwargs)
        mock_ensure.assert_called_once()
        assert mock_ensure.call_args[0][0] == "s3://bucket/key.pkl"
        # Verify loaded data matches
        assert np.allclose(sd2.train[0], sd.train[0])

    def test_pickle_non_spikedata_raises_valueerror(self, tmp_path):
        """
        Test that loading a pickle containing a non-SpikeData object raises ValueError.

        Tests:
        (Method 1) Writes a dict to pickle file (not SpikeData)
        (Method 2) Calls load_spikedata_from_pickle
        (Test Case 1) ValueError is raised with message about wrong type
        """
        path = str(tmp_path / "test.pkl")
        # Write non-SpikeData object (dict) to pickle
        with open(path, "wb") as f:
            pickle.dump({"foo": "bar"}, f)

        # Expect ValueError because pickle does not contain SpikeData
        with pytest.raises(ValueError, match="SpikeData"):
            loaders.load_spikedata_from_pickle(path)

    @patch("SpikeLab.data_loaders.s3_utils.ensure_local_file")
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
        assert not os.path.exists(path)


@skip_no_pandas
class TestIBLLoader:
    """
    Tests for load_spikedata_from_ibl.

    All external dependencies (one-api, brainwidemap) are patched via
    sys.modules so the tests run regardless of whether those packages are
    installed.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_unit_df(pid, n_good=3):
        """Return a mock bwm_units DataFrame with good and bad units."""
        import pandas as pd

        rows = [
            {
                "pid": pid,
                "eid": "test-eid",
                "label": 1,
                "cluster_id": i,
                "Beryl": "VISl",
            }
            for i in range(n_good)
        ]
        # one bad unit for the same probe (label=0)
        rows.append(
            {
                "pid": pid,
                "eid": "test-eid",
                "label": 0,
                "cluster_id": n_good,
                "Beryl": "noise",
            }
        )
        # one good unit on a different probe
        rows.append(
            {
                "pid": "other-pid",
                "eid": "other-eid",
                "label": 1,
                "cluster_id": 99,
                "Beryl": "AUDp",
            }
        )
        return pd.DataFrame(rows)

    @staticmethod
    def _make_trials_df(n_trials=5):
        """Return a mock trials DataFrame with all required columns (times in seconds)."""
        import pandas as pd

        t = np.linspace(1.0, 1.0 + (n_trials - 1) * 2.0, n_trials)
        return pd.DataFrame(
            {
                "intervals_0": t,
                "intervals_1": t + 1.0,
                "stimOn_times": t + 0.10,
                "stimOff_times": t + 0.80,
                "goCue_times": t + 0.05,
                "response_times": t + 0.50,
                "feedback_times": t + 0.55,
                "firstMovement_times": t + 0.45,
                "choice": np.tile([-1.0, 1.0], n_trials)[:n_trials],
                "feedbackType": np.ones(n_trials),
                "contrastLeft": np.full(n_trials, 0.5),
                "contrastRight": np.full(n_trials, 0.5),
                "probabilityLeft": np.full(n_trials, 0.5),
            }
        )

    @staticmethod
    def _make_spikes(cluster_ids, n_spikes=5, duration_s=100.0):
        """Return a mock spikes dict with clusters and times arrays."""
        all_clusters, all_times = [], []
        for cid in cluster_ids:
            times = np.linspace(1.0, duration_s - 1.0, n_spikes)
            all_clusters.extend([cid] * n_spikes)
            all_times.extend(times)
        return {
            "clusters": np.array(all_clusters, dtype=int),
            "times": np.array(all_times, dtype=float),
        }

    def _build_mocks(self, pid, eid, n_good=3, n_spikes=5, fail_collections=None):
        """
        Build mock one_api and brainwidemap modules for a given probe.

        Parameters:
            fail_collections: if not None, a set of collection strings for which
                load_object('spikes', ...) should raise an exception.
        """
        unit_df = self._make_unit_df(pid, n_good=n_good)
        good_ids = unit_df[(unit_df["pid"] == pid) & (unit_df["label"] == 1)][
            "cluster_id"
        ].tolist()
        spikes = self._make_spikes(good_ids, n_spikes=n_spikes)
        trials_df = self._make_trials_df()

        def load_object_side_effect(eid_arg, obj_name, **kwargs):
            if obj_name == "trials":
                mock_trials = MagicMock()
                mock_trials.to_df.return_value = trials_df
                return mock_trials
            if obj_name == "spikes":
                collection = kwargs.get("collection", "")
                if fail_collections and collection in fail_collections:
                    raise Exception(f"collection not found: {collection}")
                return spikes
            raise Exception(f"Unexpected load_object call: {obj_name}")

        mock_one_instance = MagicMock()
        mock_one_instance.load_object.side_effect = load_object_side_effect

        mock_one_class = MagicMock()
        mock_one_class.return_value = mock_one_instance

        mock_one_api = MagicMock()
        mock_one_api.ONE = mock_one_class

        mock_brainwidemap = MagicMock()
        mock_brainwidemap.bwm_units.return_value = unit_df

        return mock_one_api, mock_brainwidemap, trials_df, good_ids, spikes

    def _load(self, eid, pid, mock_one_api, mock_brainwidemap, **kwargs):
        """Call load_spikedata_from_ibl with mocked external modules."""
        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            return loaders.load_spikedata_from_ibl(eid, pid, **kwargs)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_basic_load(self):
        """
        Test that load_spikedata_from_ibl returns a valid SpikeData object.

        Tests:
            (Test Case 1) Returns a SpikeData instance.
            (Test Case 2) Number of units equals the number of good units (label==1) for the probe.
            (Test Case 3) All expected metadata keys are present.
            (Test Case 4) neuron_attributes list has one entry per unit.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, trials_df, good_ids, _ = self._build_mocks(
            pid, eid, n_good=3
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        assert isinstance(sd, SpikeData)
        assert sd.N == 3  # 3 good units
        assert sd.neuron_attributes is not None
        assert len(sd.neuron_attributes) == 3

        expected_keys = {
            "eid",
            "pid",
            "n_trials",
            "trial_start_times",
            "trial_end_times",
            "stim_on_times",
            "stim_off_times",
            "go_cue_times",
            "response_times",
            "feedback_times",
            "first_movement_times",
            "choice",
            "feedback_type",
            "contrast_left",
            "contrast_right",
            "probability_left",
        }
        assert expected_keys.issubset(set(sd.metadata.keys()))

    def test_only_good_units_included(self):
        """
        Test that only units with label==1 for the requested probe are loaded.

        Tests:
            (Test Case 1) Units with label==0 are excluded.
            (Test Case 2) Units from other probes are excluded.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, good_ids, _ = self._build_mocks(
            pid, eid, n_good=2
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        # Only 2 good units for this pid; the bad unit and other-pid unit must be excluded
        assert sd.N == 2

    def test_neuron_attributes_region(self):
        """
        Test that neuron_attributes carries the Beryl atlas region for each unit.

        Tests:
            (Test Case 1) Each unit's neuron_attributes dict contains a 'region' key.
            (Test Case 2) Region value matches the Beryl column of the bwm_units DataFrame.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(pid, eid, n_good=3)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        for attr in sd.neuron_attributes:
            assert "region" in attr
            assert attr["region"] == "VISl"

    def test_spike_times_converted_to_ms(self):
        """
        Test that spike times from the IBL server (seconds) are converted to milliseconds.

        Tests:
            (Test Case 1) Each spike time in the loaded SpikeData is 1000x the source time.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, good_ids, spikes = self._build_mocks(
            pid, eid, n_good=1, n_spikes=4
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        # Source times are in seconds; loaded times must be x 1000
        source_times_s = spikes["times"][spikes["clusters"] == good_ids[0]]
        expected_ms = source_times_s * 1000.0
        assert np.allclose(np.sort(sd.train[0]), np.sort(expected_ms))

    def test_trial_timing_arrays_in_ms(self):
        """
        Test that all trial timing metadata arrays are stored in milliseconds.

        Tests:
            (Test Case 1) stim_on_times values are 1000x the source seconds values.
            (Test Case 2) trial_start_times values are 1000x the source seconds values.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, trials_df, _, _ = self._build_mocks(pid, eid)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        expected_stim_on_ms = trials_df["stimOn_times"].to_numpy() * 1000.0
        assert np.allclose(sd.metadata["stim_on_times"], expected_stim_on_ms)

        expected_start_ms = trials_df["intervals_0"].to_numpy() * 1000.0
        assert np.allclose(sd.metadata["trial_start_times"], expected_start_ms)

    def test_behavioral_arrays_not_converted(self):
        """
        Test that non-timing behavioral arrays (choice, feedback_type, contrasts) are stored as-is.

        Tests:
            (Test Case 1) choice array values match the source DataFrame column exactly.
            (Test Case 2) feedback_type array values match the source DataFrame column exactly.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, trials_df, _, _ = self._build_mocks(pid, eid)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        assert np.allclose(sd.metadata["choice"], trials_df["choice"].to_numpy())
        assert np.allclose(
            sd.metadata["feedback_type"], trials_df["feedbackType"].to_numpy()
        )

    def test_length_inferred_from_max_spike_time(self):
        """
        Test that session length is inferred from the maximum spike time when not provided.

        Tests:
            (Test Case 1) sd.length equals the maximum spike time across all units in ms.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, _, spikes = self._build_mocks(
            pid, eid, n_good=2, n_spikes=5
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        expected_length_ms = float(spikes["times"].max()) * 1000.0
        assert sd.length == pytest.approx(expected_length_ms, abs=1e-3)

    def test_explicit_length_ms_overrides_inference(self):
        """
        Test that an explicitly supplied length_ms takes precedence over inference.

        Tests:
            (Test Case 1) sd.length equals the explicit value, not the max spike time.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(pid, eid)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap, length_ms=999.0)

        assert sd.length == pytest.approx(999.0)

    def test_collection_fallback(self):
        """
        Test that the loader falls back to the next collection when the first fails.

        Tests:
            (Test Case 1) When the first two probe-specific collections raise exceptions,
                spike data is still loaded from the fallback 'alf' collection.
            (Test Case 2) The returned SpikeData has the expected number of units.
        """
        eid, pid = "test-eid", "test-pid"
        # Make the first two collections fail; 'alf' succeeds
        fail_collections = {"alf/probe00/pykilosort", "alf/probe01/pykilosort"}
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(
            pid, eid, fail_collections=fail_collections
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        assert isinstance(sd, SpikeData)
        assert sd.N == 3

    def test_no_spikes_produces_empty_trains(self):
        """
        Test that units get empty spike trains when all spike collections are unavailable.

        Tests:
            (Test Case 1) Each unit's spike train is an empty array.
            (Test Case 2) session length falls back to 10000 ms.
        """
        eid, pid = "test-eid", "test-pid"
        all_collections = {
            "alf/probe00/pykilosort",
            "alf/probe01/pykilosort",
            "alf",
        }
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(
            pid, eid, fail_collections=all_collections
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        for train in sd.train:
            assert len(train) == 0
        assert sd.length == pytest.approx(10_000.0)

    def test_missing_one_api_raises_import_error(self):
        """
        Test that a clear ImportError is raised when one-api is not installed.

        Tests:
            (Test Case 1) ImportError is raised with a message mentioning 'one-api'.
        """
        # Simulate one-api being absent by making the import raise ImportError
        original = sys.modules.pop("one.api", None)
        original_one = sys.modules.pop("one", None)
        try:
            with patch.dict(sys.modules, {"one": None, "one.api": None}):
                with pytest.raises((ImportError, TypeError)):
                    loaders.load_spikedata_from_ibl("eid", "pid")
        finally:
            if original is not None:
                sys.modules["one.api"] = original
            if original_one is not None:
                sys.modules["one"] = original_one


@skip_no_pandas
class TestIBLQuery:
    """
    Tests for query_ibl_probes.

    All external dependencies (one-api, brainwidemap) are patched via
    sys.modules so the tests run regardless of whether those packages are
    installed.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_units_df():
        """
        Return a mock bwm_units DataFrame with three probes across two labs.

        Probe layout:
          pid-A (eid-A): 5 good units, lab=wittenlab, subject=sub-1,
                         regions [VISl, VISl, MOs, MOs, MOs]  -> 3/5 in MOs
          pid-B (eid-B): 3 good units, lab=wittenlab, subject=sub-2,
                         regions [AUDp, AUDp, AUDp]           -> 0/3 in MOs
          pid-C (eid-C): 8 good units, lab=churchland, subject=sub-3,
                         regions [MOs x4, VISl x4]            -> 4/8 in MOs
        One bad unit (label=0) is also included in eid-A.
        """
        import pandas as pd

        rows = []
        # pid-A -- 5 good units
        for i, region in enumerate(["VISl", "VISl", "MOs", "MOs", "MOs"]):
            rows.append(
                {
                    "eid": "eid-A",
                    "pid": "pid-A",
                    "label": 1,
                    "cluster_id": i,
                    "Beryl": region,
                    "subject": "sub-1",
                    "lab": "wittenlab",
                }
            )
        # bad unit in pid-A
        rows.append(
            {
                "eid": "eid-A",
                "pid": "pid-A",
                "label": 0,
                "cluster_id": 99,
                "Beryl": "noise",
                "subject": "sub-1",
                "lab": "wittenlab",
            }
        )
        # pid-B -- 3 good units
        for i, region in enumerate(["AUDp", "AUDp", "AUDp"]):
            rows.append(
                {
                    "eid": "eid-B",
                    "pid": "pid-B",
                    "label": 1,
                    "cluster_id": i,
                    "Beryl": region,
                    "subject": "sub-2",
                    "lab": "wittenlab",
                }
            )
        # pid-C -- 8 good units
        for i, region in enumerate(
            ["MOs", "MOs", "MOs", "MOs", "VISl", "VISl", "VISl", "VISl"]
        ):
            rows.append(
                {
                    "eid": "eid-C",
                    "pid": "pid-C",
                    "label": 1,
                    "cluster_id": i,
                    "Beryl": region,
                    "subject": "sub-3",
                    "lab": "churchland",
                }
            )
        return pd.DataFrame(rows)

    def _query(self, mock_brainwidemap, **kwargs):
        """Call query_ibl_probes with mocked external modules."""
        mock_one_api = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            return loaders.query_ibl_probes(**kwargs)

    def _make_mock_brainwidemap(self):
        """Return a mock brainwidemap module backed by the standard units DataFrame."""
        mock_bwm = MagicMock()
        mock_bwm.bwm_units.return_value = self._make_units_df()
        return mock_bwm

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_return_types(self):
        """
        Test that query_ibl_probes returns a (list, DataFrame) tuple.

        Tests:
            (Test Case 1) First return value is a list.
            (Test Case 2) Second return value is a pandas DataFrame.
            (Test Case 3) Each element of the list is a 2-tuple.
        """
        import pandas as pd

        mock_bwm = self._make_mock_brainwidemap()
        probes, stats = self._query(mock_bwm)

        assert isinstance(probes, list)
        assert isinstance(stats, pd.DataFrame)
        for item in probes:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_no_filters_returns_all_probes(self):
        """
        Test that with no filters all probes are returned.

        Tests:
            (Test Case 1) All three probes appear in the result.
            (Test Case 2) stats DataFrame has one row per probe.
        """
        mock_bwm = self._make_mock_brainwidemap()
        probes, stats = self._query(mock_bwm)

        assert len(probes) == 3
        assert len(stats) == 3

    def test_sorted_by_descending_unit_count(self):
        """
        Test that results are sorted by descending good unit count.

        Tests:
            (Test Case 1) First result has the highest n_good_units.
            (Test Case 2) n_good_units column is monotonically non-increasing.
        """
        mock_bwm = self._make_mock_brainwidemap()
        probes, stats = self._query(mock_bwm)

        counts = stats["n_good_units"].tolist()
        assert counts == sorted(counts, reverse=True)
        # pid-C has 8 units -> should be first
        assert probes[0][1] == "pid-C"

    def test_stats_columns_without_target_regions(self):
        """
        Test that stats DataFrame contains the expected columns when no target_regions given.

        Tests:
            (Test Case 1) eid, pid, subject, lab, n_good_units are present.
            (Test Case 2) n_in_target and fraction_in_target are absent.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm)

        for col in ("eid", "pid", "subject", "lab", "n_good_units"):
            assert col in stats.columns
        assert "n_in_target" not in stats.columns
        assert "fraction_in_target" not in stats.columns

    def test_stats_columns_with_target_regions(self):
        """
        Test that stats DataFrame includes region columns when target_regions is given.

        Tests:
            (Test Case 1) n_in_target column is present.
            (Test Case 2) fraction_in_target column is present.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm, target_regions=["MOs"])

        assert "n_in_target" in stats.columns
        assert "fraction_in_target" in stats.columns

    def test_n_in_target_and_fraction_correct(self):
        """
        Test that n_in_target and fraction_in_target are computed correctly per probe.

        Tests:
            (Test Case 1) pid-A has 3 units in MOs out of 5 -> fraction 0.6.
            (Test Case 2) pid-B has 0 units in MOs out of 3 -> fraction 0.0.
            (Test Case 3) pid-C has 4 units in MOs out of 8 -> fraction 0.5.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm, target_regions=["MOs"])

        row_a = stats[stats["pid"] == "pid-A"].iloc[0]
        assert row_a["n_in_target"] == 3
        assert row_a["fraction_in_target"] == pytest.approx(0.6)

        row_b = stats[stats["pid"] == "pid-B"].iloc[0]
        assert row_b["n_in_target"] == 0
        assert row_b["fraction_in_target"] == pytest.approx(0.0)

        row_c = stats[stats["pid"] == "pid-C"].iloc[0]
        assert row_c["n_in_target"] == 4
        assert row_c["fraction_in_target"] == pytest.approx(0.5)

    def test_min_units_filter(self):
        """
        Test that probes with fewer good units than min_units are excluded.

        Tests:
            (Test Case 1) min_units=4 excludes pid-B (3 units) but keeps pid-A (5) and pid-C (8).
            (Test Case 2) min_units=6 keeps only pid-C (8 units).
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, stats = self._query(mock_bwm, min_units=4)
        returned_pids = {p[1] for p in probes}
        assert "pid-A" in returned_pids
        assert "pid-C" in returned_pids
        assert "pid-B" not in returned_pids

        probes2, _ = self._query(mock_bwm, min_units=6)
        assert len(probes2) == 1
        assert probes2[0][1] == "pid-C"

    def test_min_fraction_in_target_filter(self):
        """
        Test that probes below the minimum fraction in target are excluded.

        Tests:
            (Test Case 1) min_fraction=0.55 keeps pid-A (0.6) and pid-C (0.5 is excluded),
                leaving only pid-A.
            (Test Case 2) min_fraction=0.0 (default) keeps all probes.
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, _ = self._query(
            mock_bwm, target_regions=["MOs"], min_fraction_in_target=0.55
        )
        returned_pids = {p[1] for p in probes}
        assert "pid-A" in returned_pids
        assert "pid-B" not in returned_pids
        assert "pid-C" not in returned_pids

        probes_all, _ = self._query(
            mock_bwm, target_regions=["MOs"], min_fraction_in_target=0.0
        )
        assert len(probes_all) == 3

    def test_min_fraction_ignored_without_target_regions(self):
        """
        Test that min_fraction_in_target has no effect when target_regions is None.

        Tests:
            (Test Case 1) Setting min_fraction_in_target without target_regions returns all probes.
        """
        mock_bwm = self._make_mock_brainwidemap()
        probes, _ = self._query(mock_bwm, min_fraction_in_target=0.9)

        assert len(probes) == 3

    def test_labs_filter(self):
        """
        Test that only probes from the specified labs are returned.

        Tests:
            (Test Case 1) labs=['wittenlab'] returns pid-A and pid-B only.
            (Test Case 2) labs=['churchland'] returns pid-C only.
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, _ = self._query(mock_bwm, labs=["wittenlab"])
        returned_pids = {p[1] for p in probes}
        assert returned_pids == {"pid-A", "pid-B"}

        probes2, _ = self._query(mock_bwm, labs=["churchland"])
        assert len(probes2) == 1
        assert probes2[0][1] == "pid-C"

    def test_subjects_filter(self):
        """
        Test that only probes from the specified subjects are returned.

        Tests:
            (Test Case 1) subjects=['sub-1'] returns only pid-A.
            (Test Case 2) subjects=['sub-1', 'sub-3'] returns pid-A and pid-C.
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, _ = self._query(mock_bwm, subjects=["sub-1"])
        assert len(probes) == 1
        assert probes[0][1] == "pid-A"

        probes2, _ = self._query(mock_bwm, subjects=["sub-1", "sub-3"])
        returned_pids = {p[1] for p in probes2}
        assert returned_pids == {"pid-A", "pid-C"}

    def test_combined_filters(self):
        """
        Test that multiple filters are applied conjunctively.

        Tests:
            (Test Case 1) labs=['wittenlab'] + min_units=4 returns only pid-A (pid-B has 3 units).
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, stats = self._query(mock_bwm, labs=["wittenlab"], min_units=4)
        assert len(probes) == 1
        assert probes[0][1] == "pid-A"

    def test_empty_result(self):
        """
        Test that impossible filter criteria return an empty list and empty DataFrame.

        Tests:
            (Test Case 1) min_units=100 returns an empty probes list.
            (Test Case 2) The stats DataFrame has zero rows.
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, stats = self._query(mock_bwm, min_units=100)
        assert probes == []
        assert len(stats) == 0

    def test_bad_units_excluded_before_aggregation(self):
        """
        Test that units with label != 1 do not contribute to n_good_units.

        Tests:
            (Test Case 1) pid-A has one bad unit (label=0); n_good_units must be 5, not 6.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm)

        row_a = stats[stats["pid"] == "pid-A"].iloc[0]
        assert row_a["n_good_units"] == 5

    def test_missing_one_api_raises_import_error(self):
        """
        Test that a clear ImportError is raised when one-api is not installed.

        Tests:
            (Test Case 1) ImportError or TypeError is raised when one.api is None in sys.modules.
        """
        original = sys.modules.pop("one.api", None)
        original_one = sys.modules.pop("one", None)
        try:
            with patch.dict(sys.modules, {"one": None, "one.api": None}):
                with pytest.raises((ImportError, TypeError)):
                    loaders.query_ibl_probes()
        finally:
            if original is not None:
                sys.modules["one.api"] = original
            if original_one is not None:
                sys.modules["one"] = original_one


class TestDataLoadersEdgeCases:
    def test_kilosort_missing_files(self, tmp_path):
        """
        Verify load_spikedata_from_kilosort raises when required .npy files are missing.

        Tests:
            (Test Case 1) Calling with an empty directory raises FileNotFoundError (or OSError).
        """
        with pytest.raises((FileNotFoundError, OSError)):
            loaders.load_spikedata_from_kilosort(str(tmp_path), fs_Hz=30000.0)

    @skip_no_h5py
    def test_hdf5_paired_empty_idces(self, tmp_path):
        """
        Verify that loading paired-style HDF5 with empty idces/times arrays
        raises an error due to max() on empty sequence in SpikeData.from_idces_times.

        Tests:
            (Test Case 1) Raises ValueError because SpikeData.__init__ calls
                          max() on empty trains when length is not provided.

        Notes:
            - This is a known source bug (BUG-006): from_idces_times does not
              handle the empty-input case gracefully.
        """
        path = str(tmp_path / "empty_paired.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=np.array([], dtype=int))
            f.create_dataset("times", data=np.array([], dtype=float))

        with pytest.raises(ValueError, match="empty sequence"):
            loaders.load_spikedata_from_hdf5(
                path, idces_dataset="idces", times_dataset="times", times_unit="ms"
            )

    @skip_no_pandas
    def test_ibl_all_collections_fail(self):
        """
        Verify that when all ONE API collection lookups fail, the loader
        still returns a SpikeData with empty trains (one per good unit) rather
        than crashing silently.

        Tests:
            (Test Case 1) Returns a SpikeData with the correct number of units.
            (Test Case 2) All spike trains are empty arrays.
        """
        import pandas as pd

        eid, pid = "test-eid", "test-pid"

        # Build a unit_df with 2 good units
        unit_df = pd.DataFrame(
            [
                {"pid": pid, "eid": eid, "label": 1, "cluster_id": 0, "Beryl": "VISl"},
                {"pid": pid, "eid": eid, "label": 1, "cluster_id": 1, "Beryl": "VISl"},
            ]
        )

        # Build a trials DataFrame
        t = np.array([1.0, 3.0])
        trials_df = pd.DataFrame(
            {
                "intervals_0": t,
                "intervals_1": t + 1.0,
                "stimOn_times": t + 0.1,
                "stimOff_times": t + 0.8,
                "goCue_times": t + 0.05,
                "response_times": t + 0.5,
                "feedback_times": t + 0.55,
                "firstMovement_times": t + 0.45,
                "choice": [-1.0, 1.0],
                "feedbackType": [1.0, 1.0],
                "contrastLeft": [0.5, 0.5],
                "contrastRight": [0.5, 0.5],
                "probabilityLeft": [0.5, 0.5],
            }
        )

        def load_object_side_effect(eid_arg, obj_name, **kwargs):
            if obj_name == "trials":
                mock_trials = MagicMock()
                mock_trials.to_df.return_value = trials_df
                return mock_trials
            if obj_name == "spikes":
                raise Exception("collection not found")
            raise Exception(f"Unexpected: {obj_name}")

        mock_one_instance = MagicMock()
        mock_one_instance.load_object.side_effect = load_object_side_effect

        mock_one_class = MagicMock()
        mock_one_class.return_value = mock_one_instance

        mock_one_api = MagicMock()
        mock_one_api.ONE = mock_one_class

        mock_brainwidemap = MagicMock()
        mock_brainwidemap.bwm_units.return_value = unit_df

        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            sd = loaders.load_spikedata_from_ibl(eid, pid)

        assert isinstance(sd, SpikeData)
        assert sd.N == 2
        for train in sd.train:
            assert len(train) == 0

    def test_kilosort_empty_spike_files(self, tmp_path):
        """
        Verify that loading KiloSort files with shape-(0,) arrays
        returns an empty SpikeData with no units.

        Tests:
            (Test Case 1) Returns a valid SpikeData with N == 0.
            (Test Case 2) No spike trains are present.
        """
        d = str(tmp_path / "ks_empty")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=float))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=30000.0)
        assert isinstance(sd, SpikeData)
        assert sd.N == 0
        assert len(sd.train) == 0

    def test_spikeinterface_empty_unit_ids(self):
        """
        Verify that loading from a SpikeInterface sorting with an empty
        unit_ids list returns an empty SpikeData.

        Tests:
            (Test Case 1) Returns a valid SpikeData with N == 0.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = []
        mock_sorting.get_sampling_frequency.return_value = 30000.0
        mock_sorting.get_unit_spike_train.return_value = np.array([], dtype=float)

        sd = loaders.load_spikedata_from_spikeinterface(mock_sorting, unit_ids=[])
        assert isinstance(sd, SpikeData)
        assert sd.N == 0
        assert len(sd.train) == 0

    def test_spikeinterface_negative_sampling_frequency(self):
        """
        Verify that a negative sampling_frequency override raises ValueError.

        Tests:
            (Test Case 1) sampling_frequency=-1000 raises ValueError.
            (Test Case 2) sampling_frequency=0 does not raise (zero is treated as
                          falsy and falls through to the extractor's own frequency).
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0]
        mock_sorting.get_sampling_frequency.return_value = 30000.0
        mock_sorting.get_unit_spike_train.return_value = np.array([100], dtype=float)

        with pytest.raises(ValueError, match="positive"):
            loaders.load_spikedata_from_spikeinterface(
                mock_sorting, sampling_frequency=-1000.0
            )

        # fs=0 is falsy, so the loader falls through to the extractor's frequency
        sd = loaders.load_spikedata_from_spikeinterface(
            mock_sorting, sampling_frequency=0
        )
        assert sd.N == 1

    @skip_no_h5py
    def test_nwb_empty_units_group(self, tmp_path):
        """
        Verify that loading an NWB file whose units group has no
        spike_times datasets raises a clear error.

        Tests:
            (Test Case 1) Raises ValueError mentioning missing spike_times.
        """
        path = str(tmp_path / "empty_units.nwb")
        with h5py.File(path, "w") as f:
            grp = f.create_group("units")
            # Write only an id dataset but no spike_times or spike_times_index
            grp.create_dataset("id", data=np.array([0, 1], dtype=int))

        with pytest.raises(ValueError, match="spike_times"):
            loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)

    def test_trains_from_flat_index_non_monotonic(self):
        """
        Verify that _trains_from_flat_index handles or tolerates
        non-monotonic end_indices by slicing according to the given indices.

        Tests:
            (Test Case 1) Non-monotonic end_indices (e.g. [3, 2, 5]) produce
                segments based on sequential start-stop iteration. The second
                segment will be empty because stop < start after the first
                segment consumed indices 0..2.
        """
        flat_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Non-monotonic: first segment uses [0:3], second tries [3:2] -> empty,
        # third uses [2:5]
        end_indices = np.array([3, 2, 5])

        trains = loaders._trains_from_flat_index(
            flat_times, end_indices, unit="ms", fs_Hz=None
        )
        assert len(trains) == 3
        # First segment: indices 0..2 -> [10, 20, 30]
        assert len(trains[0]) == 3
        # Second segment: start=3, stop=2 -> empty slice
        assert len(trains[1]) == 0
        # Third segment: start=2, stop=5 -> [30, 40, 50]
        assert len(trains[2]) == 3
